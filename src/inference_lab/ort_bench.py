"""Measure in-process ONNX Runtime latency and optionally write an ORT trace."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference_lab.benchmark import summarize, write_report
from inference_lab.model import INPUT_NAME, MAX_BATCH_SIZE, OUTPUT_NAME
from inference_lab.vision import load_image_processor, load_labels, preprocess_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to exported model.onnx")
    parser.add_argument("--image", type=Path, required=True, help="JPEG/PNG sample image")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--profile", action="store_true", help="Write an ONNX Runtime operator trace")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > MAX_BATCH_SIZE:
        parser.error(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if args.requests < 1 or args.warmup < 0:
        parser.error("--requests must be positive and --warmup cannot be negative")

    provider_name = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    if provider_name not in ort.get_available_providers():
        parser.error(f"{provider_name} is unavailable in this ONNX Runtime installation")

    image_processor = load_image_processor()
    labels = load_labels()
    one_image = preprocess_path(args.image, image_processor)
    pixel_values = np.repeat(one_image, args.batch_size, axis=0)
    options = ort.SessionOptions()
    options.enable_profiling = args.profile
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=[provider_name])
    for _ in range(args.warmup):
        session.run([OUTPUT_NAME], {INPUT_NAME: pixel_values})

    latencies_ms: list[float] = []
    measurement_started = time.perf_counter()
    for _ in range(args.requests):
        started = time.perf_counter_ns()
        logits = session.run([OUTPUT_NAME], {INPUT_NAME: pixel_values})[0]
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    measurement_seconds = time.perf_counter() - measurement_started
    top_indices = np.argsort(logits[0])[-5:][::-1]

    report = summarize(latencies_ms, args.batch_size, elapsed_seconds=measurement_seconds)
    report.update(
        {
            "backend": "onnxruntime",
            "provider": provider_name,
            "model": str(args.model),
            "image": str(args.image),
            "input_shape": list(pixel_values.shape),
            "output_shape": list(logits.shape),
            "top5": [
                {"class_id": int(index), "label": labels[int(index)]}
                for index in top_indices
            ],
            "warmup_requests": args.warmup,
            "onnxruntime_version": ort.__version__,
            "profiling_enabled": args.profile,
            "preprocessing_included": False,
        }
    )
    if args.profile:
        report["profile_trace"] = session.end_profiling()
    write_report(report, args.output)


if __name__ == "__main__":
    main()
