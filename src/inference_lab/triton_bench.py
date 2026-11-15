"""Closed-loop HTTP load generator for the three Triton serving contracts."""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import tritonclient.http as httpclient

from inference_lab.benchmark import summarize, write_report
from inference_lab.model import INPUT_NAME, MAX_BATCH_SIZE, OUTPUT_NAME
from inference_lab.vision import (
    load_image_processor,
    load_labels,
    preprocess_path,
    spatial_uint8_path,
)


MODEL_BY_CONTRACT = {
    "singleton": "classify_singleton",
    "dynamic": "classify_dynamic",
    "bulk": "classify_bulk",
}
_thread_state = threading.local()


def _get_client(url: str, timeout_s: float) -> httpclient.InferenceServerClient:
    client = getattr(_thread_state, "client", None)
    if client is None:
        client = httpclient.InferenceServerClient(
            url=url,
            concurrency=1,
            connection_timeout=timeout_s,
            network_timeout=timeout_s,
        )
        _thread_state.client = client
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--image", type=Path, required=True, help="JPEG/PNG sample image")
    parser.add_argument("--contract", choices=tuple(MODEL_BY_CONTRACT), default="dynamic")
    parser.add_argument(
        "--preprocess",
        choices=("precomputed", "cpu", "cuda"),
        default="precomputed",
        help="Reuse one prepared tensor or preprocess each image before its request",
    )
    parser.add_argument("--rows", type=int, default=1, help="Images in each request")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--slo-ms", type=float, default=20.0, help="Latency target used to count misses"
    )
    parser.add_argument("--client-timeout-ms", type=float, default=60000.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.rows > MAX_BATCH_SIZE:
        parser.error(f"--rows must be between 1 and {MAX_BATCH_SIZE}")
    if args.contract == "singleton" and args.rows != 1:
        parser.error("the singleton contract accepts exactly one row per request")
    if args.concurrency < 1 or args.requests < 1 or args.warmup < 0:
        parser.error("concurrency and requests must be positive; warmup cannot be negative")
    if args.slo_ms <= 0 or args.client_timeout_ms <= 0:
        parser.error("SLO and client timeout must be positive")

    model_name = MODEL_BY_CONTRACT[args.contract]
    processor = load_image_processor()
    labels = load_labels()
    if args.preprocess == "precomputed":
        prepared_image = preprocess_path(args.image, processor)
    else:
        prepared_image = None
    if args.preprocess == "cuda":
        try:
            import torch
            import image_preprocess_cuda
        except ImportError as exc:
            parser.error(f"install the CUDA extension before --preprocess cuda: {exc}")
        if not torch.cuda.is_available():
            parser.error("--preprocess cuda requires CUDA-enabled PyTorch and an NVIDIA GPU")
        mean = [float(value) for value in processor.image_mean]
        std = [float(value) for value in processor.image_std]

    def make_pixel_values() -> np.ndarray:
        if args.preprocess == "precomputed":
            assert prepared_image is not None
            one_image = prepared_image
        elif args.preprocess == "cpu":
            one_image = preprocess_path(args.image, processor)
        else:
            rgb = spatial_uint8_path(args.image, processor)
            device_rgb = torch.from_numpy(rgb).to("cuda")
            one_image = image_preprocess_cuda.rgb_to_normalized(
                device_rgb, mean, std
            ).cpu().numpy()
        return np.repeat(one_image, args.rows, axis=0)

    def infer_once() -> tuple[float, np.ndarray]:
        client = _get_client(args.url, args.client_timeout_ms / 1000.0)
        started = time.perf_counter_ns()
        pixel_values = make_pixel_values()
        tensor = httpclient.InferInput(INPUT_NAME, pixel_values.shape, "FP32")
        tensor.set_data_from_numpy(pixel_values)
        output = httpclient.InferRequestedOutput(OUTPUT_NAME)
        result = client.infer(
            model_name,
            inputs=[tensor],
            outputs=[output],
        )
        logits = result.as_numpy(OUTPUT_NAME)
        if logits is None or logits.shape != (args.rows, 1000):
            shape = None if logits is None else logits.shape
            raise RuntimeError(f"unexpected logits response shape: {shape}")
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return elapsed_ms, logits

    for _ in range(args.warmup):
        infer_once()

    measurement_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(infer_once) for _ in range(args.requests)]
        outcomes: list[float] = []
        failures: list[str] = []
        final_logits: np.ndarray | None = None
        for future in futures:
            try:
                latency_ms, final_logits = future.result()
                outcomes.append(latency_ms)
            except Exception as exc:  # Keep a failure count while allowing the run to finish.
                failures.append(str(exc))
    measurement_seconds = time.perf_counter() - measurement_started

    if outcomes:
        report: dict[str, Any] = summarize(
            outcomes, args.rows, elapsed_seconds=measurement_seconds
        )
    else:
        report = {
            "requests": 0,
            "rows_per_request": args.rows,
            "latency_ms": {},
            "measurement_seconds": measurement_seconds,
            "request_rate_per_s": 0.0,
            "row_rate_per_s": 0.0,
        }
    report.update(
        {
            "backend": "triton-http-v2",
            "url": args.url,
            "contract": args.contract,
            "model": model_name,
            "image": str(args.image),
            "preprocessing_mode": args.preprocess,
            "preprocessing_included_in_latency": args.preprocess != "precomputed",
            "input_shape": [args.rows, 3, 224, 224],
            "concurrency": args.concurrency,
            "attempted_requests": args.requests,
            "warmup_requests": args.warmup,
            "slo_ms": args.slo_ms,
            "client_timeout_ms": args.client_timeout_ms,
            "deadline_misses": sum(value > args.slo_ms for value in outcomes) + len(failures),
            "deadline_miss_rate": (
                (sum(value > args.slo_ms for value in outcomes) + len(failures))
                / (len(outcomes) + len(failures))
                if outcomes or failures
                else 0.0
            ),
            "failed_requests": len(failures),
            "first_failure": failures[0] if failures else None,
        }
    )
    if final_logits is not None:
        top_indices = np.argsort(final_logits[0])[-5:][::-1]
        report["top5"] = [
            {"class_id": int(index), "label": labels[int(index)]}
            for index in top_indices
        ]
    write_report(report, args.output)


if __name__ == "__main__":
    main()
