"""Measure whether automatic search rediscovers the saved Optenni winner.

Unlike the numerical golden replay, this benchmark does not inject the saved
placements into the optimizer.  It gives the search only the DUT, bands and the
four measured part models that contain the six-part saved BOM, then reports
whether the exact topology/part assignment is present in the returned ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import (  # noqa: E402
    __version__ as core_version,
    Band,
    ComponentSpec,
    MeasuredComponentOptimizer,
    MeasuredPlacement,
    MeasuredSearchConfig,
    Objective,
    Problem,
    component_sha256,
    evaluate_measured_candidate,
    load_coilcraft_0402hp_catalog,
    load_murata_gqm18_catalog,
    load_touchstone,
    measured_candidate_signature,
    measured_topology_signature,
)


RELATIVE_DIR = Path(
    "4 - Multiantenna system/4.1 Multiantenna system at different bands"
)
DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/optenni-saved-winner-discovery.json"
BANDS = {
    0: (2.50e9, 2.69e9),
    1: (1.92e9, 2.17e9),
    2: (1.215e9, 1.30e9),
}
EXPECTED_TOPOLOGIES = {
    0: frozenset({"SCPL"}),
    1: frozenset({"PCSL"}),
    2: frozenset({"PCSL"}),
}


def _component(
    root: Path, relative_path: str, name: str, kind: str,
    value: float, tolerance: float, family: str,
) -> ComponentSpec:
    return ComponentSpec(
        name, kind, value, tolerance, family, root / relative_path
    )


def _sample_indices(frequencies: np.ndarray, points_per_band: int) -> np.ndarray:
    selected: list[int] = []
    for start, stop in BANDS.values():
        available = np.flatnonzero((frequencies >= start) & (frequencies <= stop))
        if len(available) < points_per_band:
            raise ValueError(f"insufficient points in {start:g}-{stop:g} Hz")
        offsets = np.unique(np.rint(
            np.linspace(0, len(available) - 1, points_per_band)
        ).astype(int))
        selected.extend(available[offsets].tolist())
    return np.asarray(sorted(set(selected)), dtype=int)


def _port_topologies(candidate) -> dict[str, str]:
    return {
        str(port): "".join(
            ("S" if item.connection == "series" else "P") + item.component.kind
            for item in candidate.placements if item.port == port
        ) or "0"
        for port in BANDS
    }


def _summary(candidate) -> dict:
    return {
        "score_db": float(candidate.score_db),
        "topology_by_port": _port_topologies(candidate),
        "signature": [list(item) for item in measured_candidate_signature(candidate)],
    }


def _run_search(
    problem, inductors, capacitors, *, constrained: bool, per_port_keep: int,
    global_refine: bool, joint_ideal: bool,
    port_block_refine: bool,
) -> tuple:
    config = MeasuredSearchConfig(
        ideal_restarts=2,
        ideal_iterations=8,
        ideal_keep=32,
        nearest_parts=2,
        per_port_keep=per_port_keep,
        result_keep=100,
        joint_refine_seeds=1 if port_block_refine else 8,
        joint_refine_passes=3 if (global_refine or port_block_refine) else 1,
        joint_refine_neighbors=None if global_refine else 8,
        joint_refine_variants_per_value=2 if port_block_refine else 1,
        joint_refine_beam_width=8 if port_block_refine else 1,
        joint_refine_port_blocks=port_block_refine,
        joint_ideal_topologies_per_port=(
            (1 if constrained else per_port_keep) if joint_ideal else 0
        ),
        joint_ideal_combination_keep=(1 if constrained else 48),
        joint_ideal_rank_combinations=bool(joint_ideal and not constrained),
        joint_ideal_diverse_combinations=bool(joint_ideal and not constrained),
        joint_ideal_refine_topology_neighbors=bool(
            joint_ideal and not constrained
        ),
        joint_ideal_growth_refine_keep=(8 if joint_ideal and not constrained else 0),
        joint_ideal_growth_refine_restarts=8,
        joint_ideal_growth_refine_iterations=20,
        joint_ideal_growth_refine_nearest_parts=(
            1 if joint_ideal and not constrained else 2
        ),
        joint_ideal_restarts=(8 if constrained else 1) if joint_ideal else 1,
        joint_ideal_iterations=(20 if constrained else 8) if joint_ideal else 6,
        joint_ideal_keep=(4 if constrained else 2) if joint_ideal else 2,
        joint_ideal_nearest_parts=(2 if constrained else 1),
        seed=1,
        max_components_per_port=2,
        allowed_topology_codes_by_port=(EXPECTED_TOPOLOGIES if constrained else None),
    )
    started = time.perf_counter()
    optimizer = MeasuredComponentOptimizer(
        problem, inductors, capacitors, Objective(port_average_weight=0.1), config
    )
    result = optimizer.optimize()
    return result, optimizer, config, time.perf_counter() - started


def run(
    tutorial_root: Path,
    component_root: Path,
    points_per_band: int = 3,
    per_port_keep: int = 8,
    catalog_mode: str = "saved_bom_grid",
    search_mode: str = "both",
    global_refine: bool = False,
    joint_ideal: bool = False,
    port_block_refine: bool = False,
) -> dict:
    if points_per_band < 2:
        raise ValueError("points_per_band must be at least two")
    if per_port_keep < 1:
        raise ValueError("per_port_keep must be positive")
    input_path = tutorial_root / RELATIVE_DIR / "3_antennas.s3p"
    touchstone = load_touchstone(input_path)
    selected = _sample_indices(touchstone.frequencies_hz, points_per_band)
    problem = Problem(
        touchstone.frequencies_hz[selected],
        touchstone.s_parameters[selected],
        {port: (Band(*band),) for port, band in BANDS.items()},
        touchstone.z0,
    )
    if catalog_mode not in {"saved_bom_grid", "full"}:
        raise ValueError("catalog_mode must be saved_bom_grid or full")
    if search_mode not in {"both", "topology_constrained", "automatic"}:
        raise ValueError("search_mode must be both, topology_constrained or automatic")
    if catalog_mode == "full":
        inductors = load_coilcraft_0402hp_catalog(
            component_root / "Inductors/Coilcraft Inductors 0402hp"
        )
        capacitors = load_murata_gqm18_catalog(
            component_root / "Capacitors/Murata Capacitors gqm18",
            unique_values=False,
        )
        c1 = _component(
            component_root,
            "Capacitors/Murata Capacitors gqm18/GQM1885C2A1R0BB01.s2p",
            "GQM1885C2A1R0BB01", "C", 1e-12, 0.1, "Murata GQM18",
        )
        c3 = _component(
            component_root,
            "Capacitors/Murata Capacitors gqm18/GQM1885C2A3R0BB01.s2p",
            "GQM1885C2A3R0BB01", "C", 3e-12, 0.1 / 3.0, "Murata GQM18",
        )
        present = {item.name for item in capacitors}
        capacitors.extend(
            item for item in (c1, c3) if item.name not in present
        )
        capacitors.sort(key=lambda item: (item.value, item.tolerance, item.name))
        by_name = {item.name: item for item in inductors}
        l2, l56 = by_name["04HP2N0"], by_name["04HP5N6"]
    else:
        c1 = _component(
            component_root,
            "Capacitors/Murata Capacitors gqm18/GQM1885C2A1R0BB01.s2p",
            "GQM1885C2A1R0BB01", "C", 1e-12, 0.1, "Murata GQM18",
        )
        c3 = _component(
            component_root,
            "Capacitors/Murata Capacitors gqm18/GQM1885C2A3R0BB01.s2p",
            "GQM1885C2A3R0BB01", "C", 3e-12, 0.1 / 3.0, "Murata GQM18",
        )
        l2 = _component(
            component_root,
            "Inductors/Coilcraft Inductors 0402hp/04HP2N0.s2p",
            "04HP2N0", "L", 2e-9, 0.05, "Coilcraft 0402HP",
        )
        l56 = _component(
            component_root,
            "Inductors/Coilcraft Inductors 0402hp/04HP5N6.s2p",
            "04HP5N6", "L", 5.6e-9, 0.02, "Coilcraft 0402HP",
        )
        inductors, capacitors = [l2, l56], [c1, c3]
    expected = evaluate_measured_candidate(
        problem,
        (
            MeasuredPlacement("series", 0, c1),
            MeasuredPlacement("shunt", 0, l2),
            MeasuredPlacement("shunt", 1, c1),
            MeasuredPlacement("series", 1, l56),
            MeasuredPlacement("shunt", 2, c3),
            MeasuredPlacement("series", 2, l56),
        ),
        Objective(port_average_weight=0.1),
    )
    expected_signature = measured_candidate_signature(expected)
    expected_topology = measured_topology_signature(expected)

    searches = {}
    modes = (
        (("topology_constrained", True), ("automatic", False))
        if search_mode == "both"
        else ((search_mode, search_mode == "topology_constrained"),)
    )
    for name, constrained in modes:
        result, optimizer, config, elapsed = _run_search(
            problem, inductors, capacitors,
            constrained=constrained,
            per_port_keep=per_port_keep,
            global_refine=global_refine,
            joint_ideal=joint_ideal,
            port_block_refine=port_block_refine,
        )
        signatures = [measured_candidate_signature(item) for item in result.candidates]
        topologies = [measured_topology_signature(item) for item in result.candidates]
        exact_rank = (
            signatures.index(expected_signature) + 1
            if expected_signature in signatures else None
        )
        topology_rank = (
            topologies.index(expected_topology) + 1
            if expected_topology in topologies else None
        )
        saved_port_evaluation = {}
        for port in BANDS:
            expected_port_signature = tuple(
                (item.connection, item.port, item.component.name)
                for item in expected.placements if item.port == port
            )
            subproblem = optimizer.subproblems_by_port.get(port)
            evaluated = [
                candidate
                for (problem_id, _signature), candidate in optimizer.evaluation_cache.items()
                if subproblem is not None and problem_id == id(subproblem)
            ]
            ranked = sorted(evaluated, key=lambda item: item.score_db, reverse=True)
            ranked_signatures = [
                tuple(
                    (item.connection, item.port, item.component.name)
                    for item in candidate.placements
                )
                for candidate in ranked
            ]
            saved_port_evaluation[str(port)] = {
                "evaluated": expected_port_signature in ranked_signatures,
                "rank_among_evaluated": (
                    ranked_signatures.index(expected_port_signature) + 1
                    if expected_port_signature in ranked_signatures else None
                ),
                "evaluated_candidate_count": len(ranked),
            }
            if subproblem is not None:
                expected_port_candidate = evaluate_measured_candidate(
                    subproblem,
                    tuple(item for item in expected.placements if item.port == port),
                    Objective(port_average_weight=0.1),
                )
                saved_port_evaluation[str(port)].update({
                    "diagnostic_saved_score_db": expected_port_candidate.score_db,
                    "diagnostic_rank_if_injected": 1 + sum(
                        item.score_db > expected_port_candidate.score_db
                        for item in ranked
                    ),
                    "diagnostic_only_not_injected_into_search": True,
                })
        searches[name] = {
            "wall_seconds": elapsed,
            "candidate_count": len(result.candidates),
            "per_port_candidate_count": {
                str(port): len(candidates)
                for port, candidates in result.per_port_candidates.items()
            },
            "ideal_evaluations": result.ideal_evaluations,
            "physical_evaluations": result.physical_evaluations,
            "loaded_component_models": result.loaded_component_models,
            "exact_saved_winner_rank": exact_rank,
            "saved_topology_rank": topology_rank,
            "saved_port_evaluation": saved_port_evaluation,
            "best_score_db": result.best.score_db,
            "best_minus_saved_score_db": result.best.score_db - expected.score_db,
            "best_candidate": _summary(result.best),
            "config": {
                "ideal_restarts": config.ideal_restarts,
                "ideal_iterations": config.ideal_iterations,
                "ideal_keep": config.ideal_keep,
                "nearest_parts": config.nearest_parts,
                "per_port_keep": config.per_port_keep,
                "result_keep": config.result_keep,
                "joint_refine_seeds": config.joint_refine_seeds,
                "joint_refine_passes": config.joint_refine_passes,
                "joint_refine_neighbors": config.joint_refine_neighbors,
                "joint_refine_port_blocks": config.joint_refine_port_blocks,
                "joint_refine_variants_per_value": config.joint_refine_variants_per_value,
                "joint_refine_beam_width": config.joint_refine_beam_width,
                "joint_ideal_topologies_per_port": config.joint_ideal_topologies_per_port,
                "joint_ideal_combination_keep": config.joint_ideal_combination_keep,
                "joint_ideal_rank_combinations": config.joint_ideal_rank_combinations,
                "joint_ideal_diverse_combinations": config.joint_ideal_diverse_combinations,
                "joint_ideal_refine_topology_neighbors": (
                    config.joint_ideal_refine_topology_neighbors
                ),
                "joint_ideal_growth_refine_keep": config.joint_ideal_growth_refine_keep,
                "joint_ideal_growth_refine_nearest_parts": (
                    config.joint_ideal_growth_refine_nearest_parts
                ),
                "max_components_per_port": config.max_components_per_port,
                "topology_constrained": constrained,
            },
        }

    return {
        "schema_version": 1,
        "case": "Optenni saved three-port winner automatic discovery",
        "software": {"rfmatch_core_version": core_version},
        "input": {
            "relative_path": str(RELATIVE_DIR / input_path.name),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "frequency_points_hz": problem.frequencies_hz.tolist(),
            "points_per_band": points_per_band,
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
        "catalog_mode": catalog_mode,
        "catalog_size": {
            "inductors": len(inductors), "capacitors": len(capacitors)
        },
        "saved_winner": _summary(expected),
        "searches": searches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--points-per-band", type=int, default=3)
    parser.add_argument("--per-port-keep", type=int, default=8)
    parser.add_argument(
        "--catalog", choices=("saved_bom_grid", "full"),
        default="saved_bom_grid",
    )
    parser.add_argument(
        "--search-mode",
        choices=("both", "topology_constrained", "automatic"),
        default="both",
    )
    parser.add_argument("--global-refine", action="store_true")
    parser.add_argument("--joint-ideal", action="store_true")
    parser.add_argument("--port-block-refine", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.tutorial_root.resolve(), args.component_root.resolve(),
        points_per_band=args.points_per_band,
        per_port_keep=args.per_port_keep,
        catalog_mode=args.catalog,
        search_mode=args.search_mode,
        global_refine=args.global_refine,
        joint_ideal=args.joint_ideal,
        port_block_refine=args.port_block_refine,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "saved_score_db": result["saved_winner"]["score_db"],
        "searches": result["searches"],
    }, indent=2))


if __name__ == "__main__":
    main()
