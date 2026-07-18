"""Calibrate hierarchical measured search against a small exhaustive catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

from benchmark_metadata import benchmark_environment, normalized_performance


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import (  # noqa: E402
    __version__ as core_version,
    Band,
    LazyComponentSpec,
    MeasuredComponentOptimizer,
    MeasuredSearchConfig,
    Objective,
    Problem,
    S2PModel,
    exhaustive_measured_search,
    load_touchstone,
    measured_candidate_signature,
    measured_search_recall,
)


DEFAULT_INPUT = ROOT / "benchmarks" / "optenni_exports" / "optimization_settings_original.s1p"
DEFAULT_OUTPUT = ROOT / "artifacts" / "benchmarks" / "optenni-single-port-search-recall.json"
L_VALUES_NH = (3.3, 3.9, 4.7, 5.6, 6.8, 8.2)
C_VALUES_PF = (0.33, 0.39, 0.47, 0.56, 0.68, 0.82)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_spec(
    name: str,
    kind: str,
    value: float,
    frequencies_hz: np.ndarray,
    *,
    z0: float,
) -> LazyComponentSpec:
    def loader() -> S2PModel:
        omega = 2.0 * np.pi * frequencies_hz
        if kind == "L":
            resistance = 2.0 * np.pi * 1e9 * value / 30.0
            impedance = resistance + 1j * omega * value
        else:
            impedance = 0.4 + 1.0 / (1j * omega * value)
        reflection = impedance / (2.0 * z0 + impedance)
        transmission = 2.0 * z0 / (2.0 * z0 + impedance)
        matrices = np.empty((len(frequencies_hz), 2, 2), dtype=complex)
        matrices[:, 0, 0] = reflection
        matrices[:, 1, 1] = reflection
        matrices[:, 0, 1] = transmission
        matrices[:, 1, 0] = transmission
        return S2PModel(
            name, frequencies_hz, matrices, z0, 0.02, kind, value
        )

    return LazyComponentSpec(
        name,
        kind,
        value,
        0.02,
        "Optenni generic Q/ESR calibration grid",
        f"search-recall:{kind}:{value:.16g}",
        loader,
    )


def _timed(callable_):
    tracemalloc.start()
    started = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def _candidate_summary(candidate) -> dict:
    return {
        "score_db": candidate.score_db,
        "topology_code": candidate.topology_code,
        "signature": [list(item) for item in measured_candidate_signature(candidate)],
        "maximum_power_balance_error": candidate.metrics["maximum_power_balance_error"],
    }


def run(input_path: Path, *, top_k: int = 10, frequency_points: int = 17) -> dict:
    touchstone = load_touchstone(input_path)
    active = np.flatnonzero(
        (touchstone.frequencies_hz >= 1.7e9)
        & (touchstone.frequencies_hz <= 2.5e9)
    )
    if len(active) < frequency_points:
        raise ValueError("input does not contain enough active-band frequency points")
    selected = active[np.unique(np.rint(np.linspace(0, len(active) - 1, frequency_points)).astype(int))]
    frequencies = touchstone.frequencies_hz[selected]
    problem = Problem(
        frequencies,
        touchstone.s_parameters[selected],
        {0: (Band(1.7e9, 2.5e9),)},
        touchstone.z0,
    )
    inductors = [
        _component_spec(f"L_{value:g}nH", "L", value * 1e-9, frequencies, z0=touchstone.z0)
        for value in L_VALUES_NH
    ]
    capacitors = [
        _component_spec(f"C_{value:g}pF", "C", value * 1e-12, frequencies, z0=touchstone.z0)
        for value in C_VALUES_PF
    ]
    objective = Objective(
        within_band_average_weight=0.05,
        across_band_average_weight=0.1,
        port_average_weight=0.0,
    )
    config = MeasuredSearchConfig(
        ideal_restarts=4,
        ideal_iterations=12,
        ideal_keep=32,
        nearest_parts=2,
        per_port_keep=50,
        result_keep=50,
        joint_refine_seeds=4,
        joint_refine_passes=1,
        joint_refine_neighbors=6,
        seed=1,
        max_components_per_port=2,
    )
    heuristic, heuristic_seconds, heuristic_peak = _timed(
        lambda: MeasuredComponentOptimizer(
            problem, inductors, capacitors, objective, config
        ).optimize()
    )
    exhaustive, exhaustive_seconds, exhaustive_peak = _timed(
        lambda: exhaustive_measured_search(
            problem,
            inductors,
            capacitors,
            objective,
            max_components_per_port=2,
            max_evaluations=10_000,
        )
    )
    recall = measured_search_recall(heuristic, exhaustive, top_k=top_k)
    raw_performance = {
        "heuristic_wall_seconds": heuristic_seconds,
        "exhaustive_wall_seconds": exhaustive_seconds,
        "heuristic_peak_tracemalloc_bytes": heuristic_peak,
        "exhaustive_peak_tracemalloc_bytes": exhaustive_peak,
    }
    recall_payload = recall.to_dict()
    return {
        "schema_version": 1,
        "case": "Optenni optimization-settings single-port search recall",
        "software": {
            "rfmatch_core_version": core_version,
            "benchmark": "hierarchical measured search vs exhaustive v1",
        },
        "environment": benchmark_environment(),
        "input": {
            "path": str(input_path.relative_to(ROOT)),
            "sha256": _sha256(input_path),
            "band_hz": [1.7e9, 2.5e9],
            "frequency_points": frequencies.tolist(),
            "reference_resistance_ohm": touchstone.z0,
        },
        "component_grid": {
            "inductors_nh": list(L_VALUES_NH),
            "capacitors_pf": list(C_VALUES_PF),
            "inductor_loss": "Q=30 at 1 GHz",
            "capacitor_esr_ohm": 0.4,
        },
        "search_config": {
            field: getattr(config, field)
            for field in config.__dataclass_fields__
        },
        "objective": {
            field: getattr(objective, field)
            for field in objective.__dataclass_fields__
        },
        "recall": recall_payload,
        "performance": normalized_performance(raw_performance, recall_payload),
        "exhaustive_top_k": [
            _candidate_summary(item) for item in exhaustive.candidates[:top_k]
        ],
        "heuristic_top_k": [
            _candidate_summary(item) for item in heuristic.candidates[:top_k]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--frequency-points", type=int, default=17)
    parser.add_argument("--minimum-recall", type=float, default=None)
    args = parser.parse_args()
    result = run(args.input.resolve(), top_k=args.top_k, frequency_points=args.frequency_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        **result["recall"],
        **result["performance"],
    }, indent=2))
    if (
        args.minimum_recall is not None
        and result["recall"]["exact_top_k_recall"] < args.minimum_recall
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
