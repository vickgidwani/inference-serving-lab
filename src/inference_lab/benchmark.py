"""Shared latency summary helpers."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def summarize(
    latencies_ms: list[float], rows_per_request: int, elapsed_seconds: float | None = None
) -> dict[str, Any]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    measured_seconds = elapsed_seconds or float(values.sum() / 1000.0)
    return {
        "requests": int(values.size),
        "rows_per_request": rows_per_request,
        "latency_ms": {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        },
        "measurement_seconds": measured_seconds,
        "request_rate_per_s": float(values.size / measured_seconds) if measured_seconds else 0.0,
        "row_rate_per_s": float(values.size * rows_per_request / measured_seconds)
        if measured_seconds
        else 0.0,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
    }


def write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {output}")
