"""End-to-end product replay without any topology hints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import run_tuning_joint  # noqa: E402
from rfmatch_core import __version__ as core_version  # noqa: E402
from run_product_saved_winner_constrained import (  # noqa: E402
    DEFAULT_COMPONENT_ROOT,
    DEFAULT_TUTORIAL_ROOT,
    EXPECTED_SIGNATURE,
    RELATIVE_DIR,
    SAVED_SCORE_DB,
    _product_library,
    _signature,
    _summary,
)


DEFAULT_OUTPUT = (
    ROOT / "artifacts/benchmarks/optenni-product-saved-winner-automatic.json"
)


def run(
    tutorial_root: Path,
    component_root: Path,
    *,
    timeout_seconds: float = 150.0,
    beam_width: int = 50,
) -> dict:
    dut_path = tutorial_root / RELATIVE_DIR / "3_antennas.s3p"
    library, requested_catalog_size = _product_library(component_root)
    port_specs = [
        {
            "port_index": 0, "bands_mhz": [[2500, 2690]],
            "max_components": 2, "allowed_topology_codes": None,
        },
        {
            "port_index": 1, "bands_mhz": [[1920, 2170]],
            "max_components": 2, "allowed_topology_codes": None,
        },
        {
            "port_index": 2, "bands_mhz": [[1215, 1300]],
            "max_components": 2, "allowed_topology_codes": None,
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
    saved_topology = "SCPLPCSLPCSL"
    topology_rank = next((
        index + 1 for index, item in enumerate(ordered)
        if item.search_diagnostics.get("topology_code") == saved_topology
    ), None)
    best = ordered[0] if ordered else None
    diagnostics = best.search_diagnostics if best is not None else {}
    return {
        "schema_version": 1,
        "case": "Product automatic deep replay of Optenni saved topology",
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
            "topology_constraints": None,
        },
        "wall_seconds": elapsed,
        "solutions_count": len(ordered),
        "saved_topology_rank": topology_rank,
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
                "coupled_ideal_topology_search",
                "automatic_topology_deep_search", "joint_refine_port_blocks",
                "joint_refine_beam_width", "joint_refine_variants_per_value",
                "search_truncated", "termination_reason",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
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
