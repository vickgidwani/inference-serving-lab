"""Compare CPU and CUDA preprocessing with CPU and CUDA ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch  # Load CUDA/cuDNN libraries before creating the ORT CUDA session.
from PIL import Image

import image_preprocess_cuda
from inference_lab.model import INPUT_NAME, OUTPUT_NAME
from inference_lab.vision import decode_rgb, load_image_processor, load_labels, preprocess_path


def make_cuda_session(model: Path, device_id: int, tf32_enabled: bool) -> ort.InferenceSession:
    """Use PyTorch's stream so its custom kernels and ORT execute in order."""
    stream = torch.cuda.current_stream(device_id)
    provider_options: dict[str, str | int] = {
        "device_id": device_id,
        "user_compute_stream": str(stream.cuda_stream),
        "use_tf32": int(tf32_enabled),
    }
    return ort.InferenceSession(
        str(model),
        providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"],
    )


def measure_condition(
    operation: Callable[[], np.ndarray],
    *,
    warmup: int,
    samples: int,
    cuda: bool,
    device_id: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Measure sequential full requests; include device completion in GPU timings."""
    for _ in range(warmup):
        operation()
        if cuda:
            torch.cuda.synchronize(device_id)

    latencies_ms: list[float] = []
    last_logits: np.ndarray | None = None
    started_all = time.perf_counter()
    for _ in range(samples):
        # Drain prior work before starting this request. The end synchronization is
        # timed, so asynchronous CUDA work is included in the request latency.
        if cuda:
            torch.cuda.synchronize(device_id)
        started = time.perf_counter_ns()
        last_logits = operation()
        if cuda:
            torch.cuda.synchronize(device_id)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    elapsed = time.perf_counter() - started_all

    values = np.asarray(latencies_ms, dtype=np.float64)
    return (
        {
            "requests": samples,
            "latency_ms": {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(values.max()),
            },
            "measurement_seconds": elapsed,
            "request_rate_per_s": float(samples / elapsed) if elapsed else 0.0,
        },
        last_logits,
    )


def top5(logits: np.ndarray, labels: dict[int, str]) -> list[dict[str, int | str]]:
    ids = np.argsort(logits[0])[-5:][::-1]
    return [{"class_id": int(index), "label": labels[int(index)]} for index in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/classify_dynamic/1/model.onnx"))
    parser.add_argument("--image", type=Path, default=Path("artifacts/parrots.png"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/exercise-6-ort-four-way.json"),
        help="Write the JSON report here (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("samples must be positive and warmup cannot be negative")
    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}")
    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build and NVIDIA GPU are required")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        parser.error("CUDAExecutionProvider is unavailable in this ONNX Runtime install")

    processor = load_image_processor()
    labels = load_labels()
    device_id = torch.cuda.current_device()
    cuda_tf32_on_session = make_cuda_session(args.model, device_id, tf32_enabled=True)
    cuda_tf32_off_session = make_cuda_session(args.model, device_id, tf32_enabled=False)
    cpu_session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    # This custom CUDA geometry is the known pinned sample: 768x512 -> 384x256,
    # then a 224x224 center crop beginning at x=80, y=16.
    with Image.open(args.image) as image:
        sample_shape = (image.height, image.width, 3)
    if sample_shape != (512, 768, 3):
        parser.error(
            "the custom CUDA path is configured for the recorded parrots sample "
            f"(512, 768, 3); got {sample_shape}"
        )

    mean = [float(value) for value in processor.image_mean]
    std = [float(value) for value in processor.image_std]

    def cpu_preprocess_cpu_ort() -> np.ndarray:
        pixels = preprocess_path(args.image, processor)
        return cpu_session.run([OUTPUT_NAME], {INPUT_NAME: pixels})[0]

    def cpu_preprocess_gpu_ort() -> np.ndarray:
        # session.run accepts the CPU array and ORT performs its H2D input copy
        # and D2H output copy around CUDA inference.
        pixels = preprocess_path(args.image, processor)
        return cuda_tf32_on_session.run([OUTPUT_NAME], {INPUT_NAME: pixels})[0]

    def make_cuda_preprocess_gpu_ort_iobinding(
        session: ort.InferenceSession,
    ) -> Callable[[], np.ndarray]:
        def operation() -> np.ndarray:
            # Decode remains on CPU; upload RGB bytes once, then keep resize,
            # crop, normalization, and model inference on CUDA. Only logits
            # return to CPU for reporting.
            rgb = np.array(decode_rgb(args.image), dtype=np.uint8, copy=True)
            rgb_cuda = torch.from_numpy(rgb).to(device=f"cuda:{device_id}")
            spatial_cuda = image_preprocess_cuda.resize_crop_rgb(rgb_cuda, 256, 384, 16, 80)
            pixels_cuda = image_preprocess_cuda.rgb_to_normalized(spatial_cuda, mean, std)

            binding = session.io_binding()
            binding.bind_input(
                name=INPUT_NAME,
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(pixels_cuda.shape),
                buffer_ptr=pixels_cuda.data_ptr(),
            )
            binding.bind_output(OUTPUT_NAME, device_type="cuda", device_id=device_id)
            session.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()[0]

        return operation

    cuda_tf32_on_operation = make_cuda_preprocess_gpu_ort_iobinding(cuda_tf32_on_session)
    cuda_tf32_off_operation = make_cuda_preprocess_gpu_ort_iobinding(cuda_tf32_off_session)

    # Compare the same CUDA preprocessing path with TF32 toggled, then retain
    # the CPU reference and CPU-preprocess + GPU-ORT comparison.
    conditions: list[tuple[str, str, bool, str | None, Callable[[], np.ndarray]]] = [
        (
            "cuda_preprocess_gpu_ort_iobinding_tf32_on",
            "CPU decode; CUDA resize/crop and normalize",
            True,
            "enabled",
            cuda_tf32_on_operation,
        ),
        (
            "cuda_preprocess_gpu_ort_iobinding_tf32_off",
            "CPU decode; CUDA resize/crop and normalize",
            True,
            "disabled",
            cuda_tf32_off_operation,
        ),
        (
            "cpu_preprocess_cpu_ort",
            "Hugging Face CPU preprocessing",
            False,
            None,
            cpu_preprocess_cpu_ort,
        ),
        (
            "cpu_preprocess_gpu_ort",
            "Hugging Face CPU preprocessing; ORT session.run manages H2D/D2H",
            True,
            "enabled",
            cpu_preprocess_gpu_ort,
        ),
    ]

    condition_results: dict[str, dict[str, object]] = {}
    logits_by_condition: dict[str, np.ndarray] = {}
    for name, preprocessing, uses_cuda, tf32_setting, operation in conditions:
        timing, logits = measure_condition(
            operation,
            warmup=args.warmup,
            samples=args.samples,
            cuda=uses_cuda,
            device_id=device_id,
        )
        if logits is None:
            raise RuntimeError(f"condition {name} did not produce logits")
        logits_by_condition[name] = logits
        condition_results[name] = {
            "preprocessing": preprocessing,
            "inference": "CUDAExecutionProvider" if uses_cuda else "CPUExecutionProvider",
            "cuda_tf32": tf32_setting,
            "latency_scope": "one sequential request: decode + preprocessing + inference + logits available on CPU",
            **timing,
        }

    reference = logits_by_condition["cpu_preprocess_cpu_ort"]
    for name, *_ in conditions:
        logits = logits_by_condition[name]
        condition_results[name]["max_abs_logit_error_vs_cpu_reference"] = float(
            np.max(np.abs(logits - reference))
        )
        condition_results[name]["top5"] = top5(logits, labels)

    tf32_off_logits = logits_by_condition["cuda_preprocess_gpu_ort_iobinding_tf32_off"]
    for name in (
        "cuda_preprocess_gpu_ort_iobinding_tf32_on",
        "cuda_preprocess_gpu_ort_iobinding_tf32_off",
    ):
        condition_results[name]["max_abs_logit_error_vs_cuda_tf32_off"] = float(
            np.max(np.abs(logits_by_condition[name] - tf32_off_logits))
        )

    report = {
        "experiment": "exercise-6-four-way-preprocess-and-ort-comparison",
        "image": str(args.image),
        "model": str(args.model),
        "warmup_requests_per_condition": args.warmup,
        "measured_requests_per_condition": args.samples,
        "concurrency": 1,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device_id),
            "cuda_tf32_on_session_providers": cuda_tf32_on_session.get_providers(),
            "cuda_tf32_off_session_providers": cuda_tf32_off_session.get_providers(),
            "cpu_session_providers": cpu_session.get_providers(),
        },
        "notes": [
            "Conditions run sequentially; throughput is single-client sequential throughput, not a concurrency/load test.",
            "GPU request timing includes device completion; GPU outputs are copied to CPU so logits are available for comparison.",
            "CPU-preprocess + GPU-ORT uses session.run with a CPU NumPy input, so ORT manages H2D/D2H copies.",
            "CUDA-preprocess + GPU-ORT decodes on CPU, then keeps resize/crop, normalization, and inference on GPU; I/O Binding copies only logits back.",
            "The two CUDA-preprocess + GPU-ORT conditions differ only in the explicit ORT use_tf32 provider option.",
            "The CPU-preprocess + GPU-ORT condition uses the TF32-enabled CUDA session.",
        ],
        "conditions": condition_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()
