"""Calibrate six-component progressive measured-S2P search against exhaustive enumeration."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import (  # noqa: E402
    Band, LazyComponentSpec, MeasuredComponentOptimizer, MeasuredPlacement,
    MeasuredSearchConfig, Objective, Problem, S2PModel,
    evaluate_measured_candidate,
)


def _series_model(name: str, kind: str, value: float, frequencies: np.ndarray) -> S2PModel:
    matrices = []
    for frequency in frequencies:
        omega = 2.0 * np.pi * frequency
        impedance = 1j * omega * value if kind == "L" else 1.0 / (1j * omega * value)
        normalized = impedance / 50.0
        reflection = normalized / (2.0 + normalized)
        transmission = 2.0 / (2.0 + normalized)
        matrices.append([[reflection, transmission], [transmission, reflection]])
    return S2PModel(name, frequencies, np.asarray(matrices), 50.0, 0.02, kind, value)


def _catalog(frequencies: np.ndarray):
    specs = []
    for kind, values in (("L", (2.2e-9, 6.8e-9)), ("C", (0.8e-12, 2.2e-12))):
        for index, value in enumerate(values, 1):
            name = f"{kind}{index}"
            model = _series_model(name, kind, value, frequencies)
            specs.append(LazyComponentSpec(
                name, kind, value, 0.02, "synthetic_s2p", f"memory:{name}",
                lambda model=model: model,
            ))
    return tuple(spec for spec in specs if spec.kind == "L"), tuple(spec for spec in specs if spec.kind == "C"), tuple(specs)


def _signature(candidate):
    return tuple((item.connection, item.component.name) for item in candidate.placements)


def _topology(candidate):
    return tuple((item.connection, item.component.kind) for item in candidate.placements)


def run() -> dict:
    frequencies = np.asarray([0.9e9, 1.0e9, 1.1e9])
    load_impedance = 85.0 - 25.0j * frequencies / 1.0e9
    gamma = (load_impedance - 50.0) / (load_impedance + 50.0)
    problem = Problem(frequencies, gamma[:, None, None], {0: (Band(frequencies[0], frequencies[-1]),)})
    inductors, capacitors, all_parts = _catalog(frequencies)
    objective = Objective(within_band_average_weight=0.05, across_band_average_weight=0.1)

    tracemalloc.start()
    started = time.perf_counter()
    progressive = MeasuredComponentOptimizer(
        problem, inductors, capacitors, objective,
        MeasuredSearchConfig(
            ideal_restarts=2, ideal_iterations=8, ideal_keep=160,
            nearest_parts=2, per_port_keep=240, result_keep=300,
            joint_refine_seeds=64, joint_refine_passes=2,
            joint_refine_neighbors=4, seed=11,
            max_components_per_port=6, topology_beam_width=128,
            deep_discrete_topology_seeds=32,
        ),
    ).optimize()
    progressive_seconds = time.perf_counter() - started
    _, progressive_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    started = time.perf_counter()
    exhaustive = []
    model_cache = {}
    for start_connection in ("series", "shunt"):
        connections = tuple(
            start_connection if index % 2 == 0 else ("shunt" if start_connection == "series" else "series")
            for index in range(6)
        )
        for selected in itertools.product(all_parts, repeat=6):
            placements = tuple(
                MeasuredPlacement(connection, 0, component)
                for connection, component in zip(connections, selected)
            )
            exhaustive.append(evaluate_measured_candidate(problem, placements, objective, model_cache))
    exhaustive.sort(key=lambda item: item.score_db, reverse=True)
    exhaustive_seconds = time.perf_counter() - started

    progressive_six = [item for item in progressive.candidates if len(item.placements) == 6]
    exact_signatures = {_signature(item) for item in progressive_six}
    topology_signatures = {_topology(item) for item in progressive_six}
    top_k = 10
    exact_recall = sum(_signature(item) in exact_signatures for item in exhaustive[:top_k]) / top_k
    topology_recall = sum(_topology(item) in topology_signatures for item in exhaustive[:top_k]) / top_k
    best_progressive = max(progressive_six, key=lambda item: item.score_db)
    best_exhaustive = exhaustive[0]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "case": "six-component alternating L/C measured-S2P ladder",
        "scope": {"frequency_points": 3, "catalog_parts": 4, "exhaustive_candidates": len(exhaustive), "top_k": top_k},
        "progressive": {
            "best_score_db": best_progressive.score_db,
            "best_signature": _signature(best_progressive),
            "six_component_candidates_returned": len(progressive_six),
            "ideal_evaluations": progressive.ideal_evaluations,
            "physical_evaluations": progressive.physical_evaluations,
            "elapsed_seconds": progressive_seconds,
            "peak_traced_bytes": progressive_peak,
        },
        "exhaustive": {
            "best_score_db": best_exhaustive.score_db,
            "best_signature": _signature(best_exhaustive),
            "elapsed_seconds": exhaustive_seconds,
            "peak_traced_bytes": None,
        },
        "quality": {
            "best_score_gap_db": best_exhaustive.score_db - best_progressive.score_db,
            "exact_top_k_recall": exact_recall,
            "topology_top_k_recall": topology_recall,
            "maximum_power_balance_error": max(item.metrics["maximum_power_balance_error"] for item in progressive_six),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "six-component-search-baseline.json")
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
