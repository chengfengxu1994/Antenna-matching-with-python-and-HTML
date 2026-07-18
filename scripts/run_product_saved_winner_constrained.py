"""End-to-end product-service replay of the Optenni saved topology.

This benchmark intentionally enters through ``run_tuning_joint`` with the same
``ComponentLibrary`` representation used by the API.  It verifies that catalog
identity preservation, topology constraints, coupled ideal seeding, port-block
beam refinement, result conversion and diagnostics work together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from engine.component_lib import ComponentInfo, ComponentLibrary  # noqa: E402
from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import run_tuning_joint  # noqa: E402
from rfmatch_core import (  # noqa: E402
    __version__ as core_version,
    ComponentSpec,
    load_coilcraft_0402hp_catalog,
    load_murata_gqm18_catalog,
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
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/optenni-product-saved-winner-constrained.json"
# The product service interpolates the requested 1215 MHz band edge exactly;
# the lower-level catalog benchmark selects the nearest measured 1216 MHz row.
SAVED_SCORE_DB = -2.198915799994407
EXPECTED_SIGNATURE = (
    ("series", 0, "GQM1885C2A1R0BB01"),
    ("shunt", 0, "04HP2N0"),
    ("shunt", 1, "GQM1885C2A1R0BB01"),
    ("series", 1, "04HP5N6"),
    ("shunt", 2, "GQM1885C2A3R0BB01"),
    ("series", 2, "04HP5N6"),
)


def _extra_capacitor(
    component_root: Path, name: str, value_pf: float, tolerance: float
) -> ComponentSpec:
    return ComponentSpec(
        name, "C", value_pf * 1e-12, tolerance, "Murata GQM18",
        component_root / "Capacitors/Murata Capacitors gqm18" / f"{name}.s2p",
    )


def _product_library(component_root: Path) -> tuple[ComponentLibrary, dict]:
    inductors = load_coilcraft_0402hp_catalog(
        component_root / "Inductors/Coilcraft Inductors 0402hp"
    )
    capacitors = load_murata_gqm18_catalog(
        component_root / "Capacitors/Murata Capacitors gqm18",
        unique_values=False,
    )
    present = {item.name for item in capacitors}
    for item in (
        _extra_capacitor(component_root, "GQM1885C2A1R0BB01", 1.0, 0.1),
        _extra_capacitor(component_root, "GQM1885C2A3R0BB01", 3.0, 0.1 / 3.0),
    ):
        if item.name not in present:
            capacitors.append(item)
    library = ComponentLibrary()
    for item in [*inductors, *capacitors]:
        scale, unit = (1e9, "nH") if item.kind == "L" else (1e12, "pF")
        library.add_component(ComponentInfo(
            part_number=item.name,
            s2p_filename=str(item.source_path),
            zip_path="__DIR__",
            component_type="inductor" if item.kind == "L" else "capacitor",
            nominal_value=item.value * scale,
            nominal_unit=unit,
            tolerance_pct=item.tolerance * 100.0,
        ))
    return library, {"inductors": len(inductors), "capacitors": len(capacitors)}


def _signature(result) -> tuple:
    return tuple(
        (
            component["connection_type"],
            int(port),
            component["part_number"],
        )
        for port in sorted(result.per_port)
        for component in result.per_port[port].components
    )


def _summary(result) -> dict:
    return {
        "solution_index": result.solution_index,
        "score_db": result.system_score,
        "signature": [list(item) for item in _signature(result)],
        "topology_code": result.search_diagnostics.get("topology_code"),
        "average_total_efficiency": result.avg_total_efficiency,
        "minimum_total_efficiency": result.min_total_efficiency,
        "maximum_power_balance_error": result.maximum_power_balance_error,
    }


def run(
    tutorial_root: Path,
    component_root: Path,
    *,
    timeout_seconds: float = 120.0,
    beam_width: int = 50,
) -> dict:
    dut_path = tutorial_root / RELATIVE_DIR / "3_antennas.s3p"
    library, requested_catalog_size = _product_library(component_root)
    port_specs = [
        {
            "port_index": 0, "bands_mhz": [[2500, 2690]],
            "max_components": 2, "allowed_topology_codes": ["SCPL"],
        },
        {
            "port_index": 1, "bands_mhz": [[1920, 2170]],
            "max_components": 2, "allowed_topology_codes": ["PCSL"],
        },
        {
            "port_index": 2, "bands_mhz": [[1215, 1300]],
            "max_components": 2, "allowed_topology_codes": ["PCSL"],
        },
    ]
    started = time.perf_counter()
    results = run_tuning_joint(
        dut=load_touchstone_file(dut_path),
        library=library,
        port_specs=port_specs,
        objective="balanced",
        beam_width=beam_width,
        timeout_seconds=timeout_seconds,
        search_profile_timeout_seconds=timeout_seconds,
        num_band_points=3,
    )
    elapsed = time.perf_counter() - started
    ordered = [results[index] for index in sorted(results)]
    signatures = [_signature(item) for item in ordered]
    exact_rank = (
        signatures.index(EXPECTED_SIGNATURE) + 1
        if EXPECTED_SIGNATURE in signatures else None
    )
    best = ordered[0] if ordered else None
    diagnostics = best.search_diagnostics if best is not None else {}
    return {
        "schema_version": 1,
        "case": "Product constrained thorough replay of Optenni saved topology",
        "software": {"rfmatch_core_version": core_version},
        "input": {
            "relative_path": str(RELATIVE_DIR / dut_path.name),
            "sha256": hashlib.sha256(dut_path.read_bytes()).hexdigest(),
            "port_specs": port_specs,
        },
        "requested_catalog_size": requested_catalog_size,
        "request": {
            "timeout_seconds": timeout_seconds,
            "beam_width": beam_width,
            "num_band_points": 3,
        },
        "wall_seconds": elapsed,
        "solutions_count": len(ordered),
        "exact_saved_winner_rank": exact_rank,
        "saved_score_db": SAVED_SCORE_DB,
        "best_minus_saved_score_db": (
            best.system_score - SAVED_SCORE_DB if best is not None else None
        ),
        "best": _summary(best) if best is not None else None,
        "saved_winner": (
            _summary(ordered[exact_rank - 1]) if exact_rank is not None else None
        ),
        "search_diagnostics": {
            key: diagnostics.get(key)
            for key in (
                "numeric_core", "search_profile", "per_port_keep",
                "physical_evaluations", "ideal_evaluations",
                "stage_physical_evaluations",
                "component_models_loaded", "component_catalog_size",
                "coupled_ideal_topology_search", "joint_refine_port_blocks",
                "joint_refine_beam_width", "joint_refine_variants_per_value",
                "search_truncated", "termination_reason",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.tutorial_root.resolve(), args.component_root.resolve(),
        timeout_seconds=args.timeout_seconds, beam_width=args.beam_width,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
