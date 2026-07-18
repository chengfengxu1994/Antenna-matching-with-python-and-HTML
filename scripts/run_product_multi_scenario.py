"""Product replay of Optenni's Multiple impedance configurations tutorial.

The benchmark enters through the same ``MultiScenarioOptimizer`` used by the
API.  It searches complete installed component families and applies one
unchanged physical network to all three official DUT measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from engine.component_lib import ComponentLibrary, scan_s2p_directory  # noqa: E402
from engine.multi_scenario_optimizer import MultiScenarioOptimizer, Scenario  # noqa: E402
from engine.search_quality import build_multi_scenario_search_plan  # noqa: E402
from engine.topology import get_standard_topologies  # noqa: E402
from engine.touchstone import load_touchstone_file  # noqa: E402


RELATIVE_DIR = Path("7 - Multiple impedance configurations")
INPUTS = ("free_space.s1p", "cover.s1p", "cover_w_spacer.s1p")
BANDS_MHZ = ((2400.0, 2483.0), (2500.0, 2690.0))
INDUCTOR_SERIES = Path("Inductors/Coilcraft Inductors 0402cs")
CAPACITOR_SERIES = Path("Capacitors/Murata Capacitors gjm15")
REFERENCE_TOPOLOGY = "L-Network (Series-C, Shunt-L)"
REFERENCE_CAPACITOR_PF = 5.6
REFERENCE_INDUCTOR_NH = 4.3
REFERENCE_CAPACITOR_PART = "GJM1552C1H5R6DB01"
REFERENCE_INDUCTOR_PART = "04CS4N3"
REFERENCE_SCORE_DB = -0.9983384508009628
DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/optenni-product-multi-scenario.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _library(component_root: Path) -> tuple[ComponentLibrary, dict]:
    inductors = scan_s2p_directory(str(component_root / INDUCTOR_SERIES))
    capacitors = scan_s2p_directory(str(component_root / CAPACITOR_SERIES))
    library = ComponentLibrary()
    for component in [*inductors.inductors, *capacitors.capacitors]:
        library.add_component(component)
    if not library.inductors or not library.capacitors:
        raise ValueError(f"component families are unavailable below {component_root}")
    return library, {
        "inductor_series": str(INDUCTOR_SERIES),
        "capacitor_series": str(CAPACITOR_SERIES),
        "inductor_models": len(library.inductors),
        "capacitor_models": len(library.capacitors),
        "inductor_unique_values": len({item.nominal_value for item in library.inductors}),
        "capacitor_unique_values": len({item.nominal_value for item in library.capacitors}),
    }


def _summary(solution: dict) -> dict:
    components = solution.get("components", [])
    return {
        "topology": solution["topology"],
        "score_db": solution["score_db"],
        "search_estimate_score_db": solution.get("search_estimate_score_db"),
        "weighted_average_score_db": solution["weighted_average_score_db"],
        "worst_scenario_score_db": solution["worst_scenario_score_db"],
        "minimum_total_efficiency_db": 10.0 * math.log10(
            max(solution["min_total_efficiency"], 1e-15)
        ),
        "average_total_efficiency_db": 10.0 * math.log10(
            max(solution["avg_total_efficiency"], 1e-15)
        ),
        "maximum_power_balance_error": solution["maximum_power_balance_error"],
        "components": components,
        "scenarios": [{
            "filename": item["filename"],
            "score_db": item["score_db"],
            "minimum_total_efficiency_db": 10.0 * math.log10(
                max(item["min_total_efficiency"], 1e-15)
            ),
            "average_total_efficiency_db": 10.0 * math.log10(
                max(item["avg_total_efficiency"], 1e-15)
            ),
            "minimum_return_loss_db": item["min_return_loss_db"],
        } for item in solution["scenarios"]],
    }


def _is_reference_values(solution: dict) -> bool:
    values = {
        item["component_type"]: float(item["nominal_value"])
        for item in solution.get("components", [])
    }
    return (
        solution.get("topology") == REFERENCE_TOPOLOGY
        and abs(values.get("capacitor", -1.0) - REFERENCE_CAPACITOR_PF) < 1e-9
        and abs(values.get("inductor", -1.0) - REFERENCE_INDUCTOR_NH) < 1e-9
    )


def run(
    tutorial_root: Path,
    component_root: Path,
    *,
    timeout_seconds: float = 45.0,
    beam_width: int = 10,
    num_band_points: int = 5,
    max_candidates_per_position: int = 24,
    verification_band_points: int = 41,
) -> dict:
    library, catalog = _library(component_root)
    input_paths = [tutorial_root / RELATIVE_DIR / name for name in INPUTS]
    scenarios = [
        Scenario(path.name, load_touchstone_file(path)) for path in input_paths
    ]
    topologies = [
        topology for topology in get_standard_topologies()
        if topology.num_components == 2
    ]
    request = {
        "scenarios": [{"snp_filename": path.name} for path in input_paths],
        "topology_names": [item.name for item in topologies],
        "search_quality": "balanced",
        "timeout_seconds": timeout_seconds,
        "beam_width": beam_width,
        "num_band_points": num_band_points,
    }
    optimizer = MultiScenarioOptimizer(
        scenarios=scenarios,
        library=library,
        bands_mhz=BANDS_MHZ,
        input_port=0,
        num_band_points=num_band_points,
        objective="worst_case",
        beam_width=beam_width,
        timeout_seconds=timeout_seconds,
        max_candidates_per_position=max_candidates_per_position,
    )
    started = time.perf_counter()
    solutions = optimizer.optimize(topologies, result_limit=20)
    search_wall_seconds = time.perf_counter() - started
    search_diagnostics = optimizer.diagnostics()
    verification_started = time.perf_counter()
    solutions, verification_diagnostics = optimizer.verify_solutions(
        solutions, verification_band_points
    )
    verification_wall_seconds = time.perf_counter() - verification_started
    wall_seconds = time.perf_counter() - started
    diagnostics = search_diagnostics | {"verification": verification_diagnostics}
    component_by_name = {
        item.part_number: item for item in [*library.inductors, *library.capacitors]
    }
    reference_components = [
        component_by_name[REFERENCE_CAPACITOR_PART],
        component_by_name[REFERENCE_INDUCTOR_PART],
    ]
    reference_candidate = {
        "topology": REFERENCE_TOPOLOGY,
        "score_db": REFERENCE_SCORE_DB,
        "components": [{
            "position": index,
            "connection_type": connection,
            "component_type": component.component_type,
            "part_number": component.part_number,
            "nominal_value": component.nominal_value,
            "nominal_unit": component.nominal_unit,
        } for index, (connection, component) in enumerate(zip(
            ("series", "shunt"), reference_components
        ))],
    }
    reference_verified, _ = optimizer.verify_solutions(
        [reference_candidate], verification_band_points
    )
    reference_topology_rank = next((
        index + 1 for index, item in enumerate(solutions)
        if item["topology"] == REFERENCE_TOPOLOGY
    ), None)
    reference_value_rank = next((
        index + 1 for index, item in enumerate(solutions)
        if _is_reference_values(item)
    ), None)
    best = _summary(solutions[0]) if solutions else None
    return {
        "schema_version": 1,
        "case": "Product Optenni Multiple impedance configurations full-family search",
        "inputs": [{
            "relative_path": str(RELATIVE_DIR / path.name),
            "sha256": _sha256(path),
        } for path in input_paths],
        "bands_mhz": [list(item) for item in BANDS_MHZ],
        "catalog": catalog,
        "request": request | {
            "objective": "worst_case",
            "component_count": 2,
            "max_candidates_per_position": max_candidates_per_position,
            "verification_band_points": verification_band_points,
        },
        "search_plan": build_multi_scenario_search_plan(request),
        "optenni_reference": {
            "topology": REFERENCE_TOPOLOGY,
            "series_capacitor_pf": REFERENCE_CAPACITOR_PF,
            "shunt_inductor_nh": REFERENCE_INDUCTOR_NH,
            "measured_core_score_db": REFERENCE_SCORE_DB,
            "dense_product_replay": _summary(reference_verified[0]),
        },
        "wall_seconds": wall_seconds,
        "search_wall_seconds": search_wall_seconds,
        "verification_wall_seconds": verification_wall_seconds,
        "solutions_count": len(solutions),
        "reference_topology_rank": reference_topology_rank,
        "reference_value_rank": reference_value_rank,
        "best": best,
        "score_delta_vs_measured_core_db": (
            best["score_db"] - REFERENCE_SCORE_DB if best is not None else None
        ),
        "search_diagnostics": diagnostics | {
            "budget_fraction": search_wall_seconds / max(timeout_seconds, 1e-9),
        },
        "top_solutions": [_summary(item) for item in solutions[:20]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--num-band-points", type=int, default=5)
    parser.add_argument("--max-candidates-per-position", type=int, default=24)
    parser.add_argument("--verification-band-points", type=int, default=41)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.tutorial_root.resolve(), args.component_root.resolve(),
        timeout_seconds=args.timeout_seconds,
        beam_width=args.beam_width,
        num_band_points=args.num_band_points,
        max_candidates_per_position=args.max_candidates_per_position,
        verification_band_points=args.verification_band_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
