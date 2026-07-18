"""Cross-software replay of Optenni Lab's official Quick Start example.

The report keeps three questions separate:

1. Can RFMatch reproduce Optenni's exported complex network numerically?
2. Which installed 8.2 nH / 0.2 pF measured BOM is closest to that network?
3. Can the product's unconstrained measured-component search discover a
   competitive circuit from the complete requested component families?
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from engine.component_lib import ComponentLibrary, scan_s2p_directory  # noqa: E402
from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import run_tuning_single  # noqa: E402
from rfmatch_core import (  # noqa: E402
    Band,
    ComponentSpec,
    ModelPlacement,
    Problem,
    build_model_circuit_topology,
    evaluate_physical_problem,
    load_component_model,
    load_optenni_plot_export,
    load_touchstone,
)


RELATIVE_INPUT = Path("1 - START HERE/measured_antenna.s1p")
INDUCTOR_SERIES = Path("Inductors/Coilcraft Inductors 0402cs")
CAPACITOR_SERIES = Path("Capacitors/Murata Capacitors gjm15")
REFERENCE_MANIFEST = (
    ROOT / "benchmarks/optenni_exports"
    / "quick_start_0402cs_gjm15_pcsl_manifest.json"
)
DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/optenni-product-quick-start.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


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


def _spec(component, kind: str) -> ComponentSpec:
    path = Path(component.s2p_filename).resolve()
    return ComponentSpec(
        component.part_number,
        kind,
        component.nominal_value * (1e-9 if kind == "L" else 1e-12),
        0.05,
        path.parent.name,
        path,
    )


def _evaluate_pcsl(dut, capacitor, inductor) -> dict:
    placements = (
        ModelPlacement("shunt", 0, load_component_model(_spec(capacitor, "C"))),
        ModelPlacement("series", 0, load_component_model(_spec(inductor, "L"))),
    )
    problem = Problem(
        dut.frequencies_hz,
        dut.s_parameters,
        {0: (Band(2.5e9, 2.69e9),)},
        dut.z0,
    )
    sweep = evaluate_physical_problem(
        problem, build_model_circuit_topology(1, placements)
    )
    s11_db = 20.0 * np.log10(np.maximum(np.abs(sweep.s_parameters[:, 0, 0]), 1e-15))
    efficiency_db = 10.0 * np.log10(
        np.maximum(sweep.total_efficiency[:, 0], 1e-15)
    )
    return {
        "s11_db": s11_db,
        "efficiency_db": efficiency_db,
        "maximum_power_balance_error": float(
            np.max(np.abs(sweep.power_balance_error[:, 0]))
        ),
    }


def _curve_comparison(
    evaluated: dict, reference_s11_db: np.ndarray,
    reference_efficiency_db: np.ndarray, band_mask: np.ndarray,
) -> dict:
    s11_delta = evaluated["s11_db"] - reference_s11_db
    efficiency_delta = evaluated["efficiency_db"] - reference_efficiency_db
    band_efficiency = evaluated["efficiency_db"][band_mask]
    band_s11 = evaluated["s11_db"][band_mask]
    return {
        "maximum_s11_delta_db_all_points": float(np.max(np.abs(s11_delta))),
        "rms_s11_delta_db_all_points": float(np.sqrt(np.mean(s11_delta ** 2))),
        "maximum_efficiency_delta_db_all_points": float(
            np.max(np.abs(efficiency_delta))
        ),
        "rms_efficiency_delta_db_all_points": float(
            np.sqrt(np.mean(efficiency_delta ** 2))
        ),
        "band": {
            "minimum_total_efficiency_db": float(np.min(band_efficiency)),
            "average_total_efficiency_db": float(np.mean(band_efficiency)),
            "best_s11_db": float(np.min(band_s11)),
            "worst_s11_db": float(np.max(band_s11)),
            "minimum_efficiency_delta_db": float(
                np.min(band_efficiency)
                - np.min(reference_efficiency_db[band_mask])
            ),
            "average_efficiency_delta_db": float(
                np.mean(band_efficiency)
                - np.mean(reference_efficiency_db[band_mask])
            ),
        },
        "maximum_power_balance_error": evaluated["maximum_power_balance_error"],
    }


def _topology(result) -> str:
    return result.search_diagnostics.get("topology_code", "")


def _component_summary(item: dict) -> dict:
    return {
        "part_number": item.get("part_number"),
        "type": item.get("type"),
        "connection": item.get("connection_type"),
        "nominal_value": item.get("nominal_value"),
        "nominal_unit": item.get("nominal_unit"),
    }


def run(
    tutorial_root: Path,
    component_root: Path,
    *,
    timeout_seconds: float,
    beam_width: int,
    num_band_points: int,
) -> dict:
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    input_path = tutorial_root / RELATIVE_INPUT
    export_dir = REFERENCE_MANIFEST.parent
    plot_path = export_dir / manifest["files"]["plotted_response"]["path"]
    plot = load_optenni_plot_export(plot_path)
    dut_core = load_touchstone(input_path)
    if not np.allclose(
        dut_core.frequencies_hz,
        plot.frequencies_hz,
        rtol=0.0,
        atol=np.maximum(1e-6, np.abs(plot.frequencies_hz) * 1e-12),
    ):
        raise ValueError("Quick Start DUT and native plot grids do not match")
    band_mask = (
        (plot.frequencies_hz >= 2.5e9)
        & (plot.frequencies_hz <= 2.69e9)
    )
    library, catalog = _library(component_root)

    reference_inductors = [
        item for item in library.inductors
        if math.isclose(item.nominal_value, 8.2, rel_tol=0.0, abs_tol=1e-12)
    ]
    reference_capacitors = [
        item for item in library.capacitors
        if math.isclose(item.nominal_value, 0.2, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(reference_inductors) != 1 or not reference_capacitors:
        raise ValueError("installed catalog does not contain the displayed reference BOM")

    displayed_bom_variants = []
    for capacitor in reference_capacitors:
        evaluated = _evaluate_pcsl(dut_core, capacitor, reference_inductors[0])
        displayed_bom_variants.append({
            "topology_code": "PCSL",
            "shunt_capacitor": capacitor.part_number,
            "series_inductor": reference_inductors[0].part_number,
            "curve_comparison": _curve_comparison(
                evaluated, plot.s11_db, plot.total_efficiency_db, band_mask
            ),
        })
    displayed_bom_variants.sort(key=lambda item: (
        item["curve_comparison"]["rms_efficiency_delta_db_all_points"],
        item["curve_comparison"]["rms_s11_delta_db_all_points"],
        item["shunt_capacitor"],
    ))

    started = time.perf_counter()
    results = run_tuning_single(
        dut=load_touchstone_file(input_path),
        library=library,
        port_index=0,
        bands_mhz=[[2500.0, 2690.0]],
        max_components=2,
        objective="balanced",
        beam_width=beam_width,
        timeout_seconds=timeout_seconds,
        search_profile_timeout_seconds=timeout_seconds,
        num_band_points=num_band_points,
    )
    wall_seconds = time.perf_counter() - started
    ordered = [results[index] for index in sorted(results)]
    product_candidates = []
    for rank, result in enumerate(ordered, start=1):
        components = result.per_port[0].components
        product_candidates.append({
            "rank": rank,
            "topology_code": _topology(result),
            "score_db": float(result.system_score),
            "minimum_total_efficiency_db": 10.0 * math.log10(
                max(result.min_total_efficiency, 1e-15)
            ),
            "average_total_efficiency_db": 10.0 * math.log10(
                max(result.avg_total_efficiency, 1e-15)
            ),
            "components": [_component_summary(item) for item in components],
        })

    exact_displayed_rank = next((
        item["rank"] for item in product_candidates
        if item["topology_code"] == "PCSL"
        and {part["nominal_value"] for part in item["components"]}
        == {0.2, 8.2}
    ), None)
    best_diagnostics = ordered[0].search_diagnostics if ordered else {}
    return {
        "schema_version": 1,
        "case": "Product Optenni Quick Start full-catalog search",
        "input": {
            "relative_path": str(RELATIVE_INPUT),
            "sha256": _sha256(input_path),
            "points": len(dut_core.frequencies_hz),
            "band_mhz": [2500.0, 2690.0],
        },
        "optenni_reference": {
            "manifest": str(REFERENCE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": _sha256(REFERENCE_MANIFEST),
            "plot_sha256": _sha256(plot_path),
            "topology_code": "PCSL",
            "displayed_shunt_capacitor_pf": 0.2,
            "displayed_series_inductor_nh": 8.2,
            "minimum_total_efficiency_db": float(
                np.min(plot.total_efficiency_db[band_mask])
            ),
            "average_total_efficiency_db": float(
                np.mean(plot.total_efficiency_db[band_mask])
            ),
        },
        "requested_catalog_size": catalog,
        "displayed_bom_measured_model_variants": displayed_bom_variants,
        "closest_displayed_bom_variant": displayed_bom_variants[0],
        "request": {
            "search_quality": "balanced",
            "timeout_seconds": timeout_seconds,
            "beam_width": beam_width,
            "num_band_points": num_band_points,
            "topology_constraints": None,
        },
        "wall_seconds": wall_seconds,
        "solutions_count": len(product_candidates),
        "exact_displayed_value_bom_rank": exact_displayed_rank,
        "best": product_candidates[0] if product_candidates else None,
        "candidates": product_candidates,
        "search_diagnostics": {
            key: best_diagnostics.get(key)
            for key in (
                "numeric_core", "search_profile", "physical_evaluations",
                "ideal_evaluations", "component_models_loaded",
                "stage_physical_evaluations", "component_catalog_size",
                "joint_refine_neighbors", "joint_refine_port_blocks",
                "joint_refine_port_block_max_components",
                "search_truncated", "termination_reason",
                "active_frequency_points",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--num-band-points", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.tutorial_root.resolve(), args.component_root.resolve(),
        timeout_seconds=args.timeout_seconds,
        beam_width=args.beam_width,
        num_band_points=args.num_band_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
