"""Compare Hugging Face CPU preprocessing with a fused CUDA tensor-conversion kernel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import image_preprocess_cuda
from inference_lab.vision import load_image_processor, preprocess_path, spatial_uint8_path


def measure_wall(operation, warmup: int, samples: int, synchronize: bool) -> list[float]:
    for _ in range(warmup):
        operation()
    if synchronize:
        torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        if synchronize:
            torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return values


def measure_cuda_events(operation, warmup: int, samples: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            output = operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)
        del output
    return values


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive; warmup cannot be negative")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build and NVIDIA GPU are required")

    processor = load_image_processor()
    mean = [float(value) for value in processor.image_mean]
    std = [float(value) for value in processor.image_std]

    def cpu_path() -> np.ndarray:
        return preprocess_path(args.image, processor)

    def custom_cuda_path() -> np.ndarray:
        rgb = spatial_uint8_path(args.image, processor)
        device_rgb = torch.from_numpy(rgb).to("cuda")
        output = image_preprocess_cuda.rgb_to_normalized(device_rgb, mean, std)
        torch.cuda.synchronize()
        return output.cpu().numpy()

    cpu_reference = cpu_path()
    custom_reference = custom_cuda_path()
    np.testing.assert_allclose(custom_reference, cpu_reference, rtol=1e-6, atol=1e-6)

    spatial_rgb = spatial_uint8_path(args.image, processor)
    device_rgb = torch.from_numpy(spatial_rgb).to("cuda")
    kernel_only = lambda: image_preprocess_cuda.rgb_to_normalized(device_rgb, mean, std)
    cpu_ms = measure_wall(cpu_path, args.warmup, args.samples, synchronize=False)
    end_to_end_cuda_ms = measure_wall(
        custom_cuda_path, args.warmup, args.samples, synchronize=True
    )
    kernel_ms = measure_cuda_events(kernel_only, args.warmup, args.samples, args.repeats)

    report = {
        "image": str(args.image),
        "spatial_shape_hwc": list(spatial_rgb.shape),
        "normalized_shape_nchw": list(custom_reference.shape),
        "processor_mean": mean,
        "processor_std": std,
        "cpu_decode_and_processor_ms": stats(cpu_ms),
        "cpu_spatial_plus_cuda_normalization_ms": stats(end_to_end_cuda_ms),
        "cuda_kernel_only_ms": stats(kernel_ms),
        "max_absolute_error": float(np.max(np.abs(custom_reference - cpu_reference))),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "measurement_note": "CUDA end-to-end includes image decode, Hugging Face spatial resize/crop, host-to-device copy, and fused normalization/layout kernel",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote benchmark report to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
