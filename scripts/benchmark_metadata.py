"""Stable environment and normalized throughput metadata for benchmark artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np


def benchmark_environment() -> dict:
    stable = {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "timer": "time.perf_counter",
        "timer_resolution_seconds": time.get_clock_info("perf_counter").resolution,
    }
    fingerprint = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        **stable,
        "python_executable": sys.executable,
        "environment_fingerprint_sha256": fingerprint,
        "measurement_scope": (
            "single-process wall time; tracemalloc covers Python-managed allocations, "
            "not total native NumPy process RSS"
        ),
    }


def normalized_performance(performance: dict, recall: dict) -> dict:
    result = dict(performance)
    for method in ("heuristic", "exhaustive"):
        seconds = float(performance[f"{method}_wall_seconds"])
        evaluations = int(recall[f"{method}_physical_evaluations"])
        peak_bytes = int(performance[f"{method}_peak_tracemalloc_bytes"])
        result[f"{method}_physical_evaluations_per_wall_second"] = (
            evaluations / seconds if seconds > 0 else None
        )
        result[f"{method}_peak_tracemalloc_mib"] = peak_bytes / (1024.0 * 1024.0)
    result["heuristic_to_exhaustive_wall_time_ratio"] = (
        float(performance["heuristic_wall_seconds"])
        / max(float(performance["exhaustive_wall_seconds"]), 1e-15)
    )
    result["throughput_interpretation"] = (
        "physical evaluation count divided by end-to-end search wall time; includes ideal "
        "search, ranking, model loading and Python overhead"
    )
    return result
