"""Calibrate coupled three-port measured search against exhaustive ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    MeasuredComponentOptimizer,
    MeasuredSearchConfig,
    Objective,
    Problem,
    component_sha256,
    exhaustive_measured_joint_search,
    load_coilcraft_0402hp_catalog,
    load_murata_gqm18_catalog,
    load_touchstone,
    measured_candidate_signature,
    measured_search_recall,
)


RELATIVE_INPUT = Path(
    "4 - Multiantenna system/4.1 Multiantenna system at different bands/3_antennas.s3p"
)
DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/optenni-multiport-search-recall.json"
BANDS = {
    0: ((2.50e9, 2.69e9),),
    1: ((1.92e9, 2.17e9),),
    2: ((1.215e9, 1.30e9),),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_indices(frequencies: np.ndarray, points_per_band: int = 3) -> np.ndarray:
    selected: list[int] = []
    for bands in BANDS.values():
        for start, stop in bands:
            available = np.flatnonzero((frequencies >= start) & (frequencies <= stop))
            if len(available) < points_per_band:
                raise ValueError(f"insufficient points in {start:g}-{stop:g} Hz")
            offsets = np.unique(np.rint(
                np.linspace(0, len(available) - 1, points_per_band)
            ).astype(int))
            selected.extend(available[offsets].tolist())
    return np.asarray(sorted(set(selected)), dtype=int)


def _part_by_value(catalog, value: float):
    return min(catalog, key=lambda item: (abs(np.log(item.value / value)), item.name))


def _timed(callable_):
    tracemalloc.start()
    started = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def _summary(candidate) -> dict:
    return {
        "score_db": candidate.score_db,
        "topology_code": candidate.topology_code,
        "signature": [list(item) for item in measured_candidate_signature(candidate)],
        "port_scores_db": np.asarray(candidate.metrics["port_scores_db"]).tolist(),
        "maximum_power_balance_error": candidate.metrics["maximum_power_balance_error"],
    }


def run(
    tutorial_root: Path,
    component_root: Path,
    *,
    top_k: int = 10,
    per_port_keep: int = 8,
) -> dict:
    input_path = tutorial_root / RELATIVE_INPUT
    touchstone = load_touchstone(input_path)
    selected = _sample_indices(touchstone.frequencies_hz)
    problem = Problem(
        touchstone.frequencies_hz[selected],
        touchstone.s_parameters[selected],
        {
            port: tuple(Band(start, stop) for start, stop in bands)
            for port, bands in BANDS.items()
        },
        touchstone.z0,
    )
    full_inductors = load_coilcraft_0402hp_catalog(
        component_root / "Inductors/Coilcraft Inductors 0402hp"
    )
    full_capacitors = load_murata_gqm18_catalog(
        component_root / "Capacitors/Murata Capacitors gqm18"
    )
    inductors = [_part_by_value(full_inductors, value) for value in (4.7e-9, 39e-9)]
    capacitors = [_part_by_value(full_capacitors, value) for value in (1e-12, 2e-12)]
    objective = Objective(port_average_weight=0.1)
    config = MeasuredSearchConfig(
        ideal_restarts=2,
        ideal_iterations=6,
        ideal_keep=16,
        nearest_parts=2,
        per_port_keep=per_port_keep,
        result_keep=50,
        joint_refine_seeds=4,
        joint_refine_passes=1,
        joint_refine_neighbors=2,
        seed=1,
        max_components_per_port=1,
    )
    heuristic, heuristic_seconds, heuristic_peak = _timed(
        lambda: MeasuredComponentOptimizer(
            problem, inductors, capacitors, objective, config
        ).optimize()
    )
    exhaustive, exhaustive_seconds, exhaustive_peak = _timed(
        lambda: exhaustive_measured_joint_search(
            problem,
            inductors,
            capacitors,
            objective,
            max_components_per_port=1,
            max_evaluations=1_000,
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
        "case": "Optenni three-antenna coupled search recall",
        "software": {
            "rfmatch_core_version": core_version,
            "benchmark": "joint measured search vs exhaustive v1",
        },
        "environment": benchmark_environment(),
        "input": {
            "relative_path": str(RELATIVE_INPUT),
            "sha256": _sha256(input_path),
            "reference_resistance_ohm": touchstone.z0,
            "frequency_points": problem.frequencies_hz.tolist(),
            "bands_by_port_hz": {
                str(port): [list(band) for band in bands]
                for port, bands in BANDS.items()
            },
        },
        "component_grid": [
            {
                "name": item.name,
                "kind": item.kind,
                "value_si": item.value,
                "model_sha256": component_sha256(item),
            }
            for item in [*inductors, *capacitors]
        ],
        "search_config": {
            field: getattr(config, field) for field in config.__dataclass_fields__
        },
        "objective": {
            field: getattr(objective, field) for field in objective.__dataclass_fields__
        },
        "recall": recall_payload,
        "performance": normalized_performance(raw_performance, recall_payload),
        "exhaustive_top_k": [_summary(item) for item in exhaustive.candidates[:top_k]],
        "heuristic_top_k": [_summary(item) for item in heuristic.candidates[:top_k]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-port-keep", type=int, default=8)
    parser.add_argument("--minimum-recall", type=float, default=None)
    args = parser.parse_args()
    result = run(
        args.tutorial_root.resolve(),
        args.component_root.resolve(),
        top_k=args.top_k,
        per_port_keep=args.per_port_keep,
    )
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
