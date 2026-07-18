"""Product single-port replay of Optenni's Optimization Settings tutorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import run_tuning_single  # noqa: E402
from engine.component_lib import (  # noqa: E402
    ComponentLibrary,
    scan_s2p_directory,
)
from run_product_saved_winner_constrained import (  # noqa: E402
    DEFAULT_COMPONENT_ROOT,
    DEFAULT_TUTORIAL_ROOT,
)


RELATIVE_INPUT = Path("3 - Optimization settings/measured_antenna.s1p")
REFERENCE_MANIFEST = (
    ROOT / "benchmarks/optenni_exports/optimization_settings_pcsl_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/benchmarks/optenni-product-optimization-settings.json"
)
INDUCTOR_SERIES = "Inductors/Coilcraft Inductors 0402hp"
CAPACITOR_SERIES = "Capacitors/AVX Capacitors ACCU-P 0402"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topology(result) -> str:
    return str(result.search_diagnostics.get("topology_code") or "0")


def _summary(result) -> dict:
    components = result.per_port[0].components
    band_efficiency = result.per_port[0].band_total_eff
    efficiency_db = [10.0 * math.log10(max(value, 1e-15)) for value in band_efficiency]
    return {
        "solution_index": result.solution_index,
        "score_db": result.system_score,
        "topology_code": _topology(result),
        "components": components,
        "average_total_efficiency": result.avg_total_efficiency,
        "minimum_total_efficiency": result.min_total_efficiency,
        "minimum_total_efficiency_db": min(efficiency_db),
        "average_total_efficiency_db": sum(efficiency_db) / len(efficiency_db),
        "maximum_power_balance_error": result.maximum_power_balance_error,
    }


def _product_library(component_root: Path) -> tuple[ComponentLibrary, dict]:
    inductor_library = scan_s2p_directory(str(component_root / INDUCTOR_SERIES))
    capacitor_library = scan_s2p_directory(str(component_root / CAPACITOR_SERIES))
    library = ComponentLibrary()
    for component in [
        *inductor_library.inductors,
        *capacitor_library.capacitors,
    ]:
        library.add_component(component)
    return library, {
        "inductors": len(library.inductors),
        "capacitors": len(library.capacitors),
        "inductor_series": INDUCTOR_SERIES,
        "capacitor_series": CAPACITOR_SERIES,
        "inductor_range_nh": [
            min(item.nominal_value for item in library.inductors),
            max(item.nominal_value for item in library.inductors),
        ],
        "capacitor_range_pf": [
            min(item.nominal_value for item in library.capacitors),
            max(item.nominal_value for item in library.capacitors),
        ],
    }


def run(
    tutorial_root: Path,
    component_root: Path,
    *,
    timeout_seconds: float = 45.0,
    beam_width: int = 20,
    num_band_points: int = 17,
) -> dict:
    input_path = tutorial_root / RELATIVE_INPUT
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    library, requested_catalog_size = _product_library(component_root)
    started = time.perf_counter()
    results = run_tuning_single(
        dut=load_touchstone_file(input_path),
        library=library,
        port_index=0,
        bands_mhz=[[1700, 2500]],
        max_components=2,
        allowed_topology_codes=None,
        objective="balanced",
        beam_width=beam_width,
        timeout_seconds=timeout_seconds,
        search_profile_timeout_seconds=timeout_seconds,
        num_band_points=num_band_points,
    )
    elapsed = time.perf_counter() - started
    ordered = [results[index] for index in sorted(results)]
    pcsl_rank = next((
        index + 1 for index, result in enumerate(ordered)
        if _topology(result) == "PCSL"
    ), None)
    best = ordered[0] if ordered else None
    best_components = best.per_port[0].components if best is not None else []
    by_type = {item["type"]: item for item in best_components}
    reference_c = float(
        manifest["tolerance_analysis"]["inferred_nominal_capacitance_pf"]
    )
    reference_l = float(
        manifest["tolerance_analysis"]["inferred_nominal_inductance_nh"]
    )
    selected_c = (by_type.get("capacitor") or {}).get("nominal_value")
    selected_l = (by_type.get("inductor") or {}).get("nominal_value")
    diagnostics = best.search_diagnostics if best is not None else {}
    best_summary = _summary(best) if best is not None else None
    reference_minimum_db = float(
        manifest["numeric_band_summary"]["minimum_total_efficiency_db"]
    )
    reference_average_db = float(
        manifest["numeric_band_summary"]["average_total_efficiency_db_from_db_average"]
    )
    return {
        "schema_version": 1,
        "case": "Product Optenni Optimization Settings full-catalog search",
        "input": {
            "relative_path": str(RELATIVE_INPUT),
            "sha256": _sha256(input_path),
            "band_mhz": [1700, 2500],
        },
        "optenni_reference": {
            "manifest": str(REFERENCE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": _sha256(REFERENCE_MANIFEST),
            "topology_code": manifest["selected_candidate"],
            "shunt_capacitor_pf": reference_c,
            "series_inductor_nh": reference_l,
            "minimum_total_efficiency_db": manifest["numeric_band_summary"]["minimum_total_efficiency_db"],
            "average_total_efficiency_db": manifest["numeric_band_summary"]["average_total_efficiency_db_from_db_average"],
        },
        "requested_catalog_size": requested_catalog_size,
        "request": {
            "search_quality": "balanced",
            "timeout_seconds": timeout_seconds,
            "beam_width": beam_width,
            "num_band_points": num_band_points,
            "topology_constraints": None,
        },
        "wall_seconds": elapsed,
        "solutions_count": len(ordered),
        "optenni_topology_rank": pcsl_rank,
        "best": best_summary,
        "cross_software_efficiency_comparison": {
            "scope": "Optenni ideal PCSL versus nearest real 0402 measured-S2P procurement network",
            "minimum_efficiency_delta_db": (
                best_summary["minimum_total_efficiency_db"] - reference_minimum_db
                if best_summary is not None else None
            ),
            "average_efficiency_delta_db": (
                best_summary["average_total_efficiency_db"] - reference_average_db
                if best_summary is not None else None
            ),
        },
        "procurement_value_comparison": {
            "selected_capacitor_pf": selected_c,
            "selected_inductor_nh": selected_l,
            "capacitor_relative_deviation": (
                abs(selected_c / reference_c - 1.0) if selected_c is not None else None
            ),
            "inductor_relative_deviation": (
                abs(selected_l / reference_l - 1.0) if selected_l is not None else None
            ),
        },
        "search_diagnostics": {
            key: diagnostics.get(key)
            for key in (
                "numeric_core", "search_profile", "physical_evaluations",
                "ideal_evaluations", "component_models_loaded",
                "stage_physical_evaluations",
                "component_catalog_size", "topology_code", "search_truncated",
                "termination_reason", "maximum_components_searched",
                "active_frequency_points",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--num-band-points", type=int, default=17)
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
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
