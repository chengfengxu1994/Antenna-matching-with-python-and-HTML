import hashlib
import json

from scripts.benchmark_metadata import benchmark_environment, normalized_performance


def test_benchmark_environment_has_stable_machine_fingerprint():
    first = benchmark_environment()
    second = benchmark_environment()
    assert first["environment_fingerprint_sha256"] == second["environment_fingerprint_sha256"]
    assert len(first["environment_fingerprint_sha256"]) == 64
    assert first["logical_cpu_count"] > 0
    assert first["python_version"]
    assert first["numpy_version"]
    stable = {
        key: first[key] for key in (
            "operating_system", "machine", "processor", "logical_cpu_count",
            "python_implementation", "python_version", "numpy_version", "timer",
            "timer_resolution_seconds",
        )
    }
    expected = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert first["environment_fingerprint_sha256"] == expected


def test_normalized_performance_keeps_quality_and_throughput_denominators_explicit():
    result = normalized_performance({
        "heuristic_wall_seconds": 2.0,
        "exhaustive_wall_seconds": 5.0,
        "heuristic_peak_tracemalloc_bytes": 2 * 1024 * 1024,
        "exhaustive_peak_tracemalloc_bytes": 5 * 1024 * 1024,
    }, {
        "heuristic_physical_evaluations": 100,
        "exhaustive_physical_evaluations": 500,
    })
    assert result["heuristic_physical_evaluations_per_wall_second"] == 50.0
    assert result["exhaustive_physical_evaluations_per_wall_second"] == 100.0
    assert result["heuristic_peak_tracemalloc_mib"] == 2.0
    assert result["heuristic_to_exhaustive_wall_time_ratio"] == 0.4
    assert "end-to-end" in result["throughput_interpretation"]
