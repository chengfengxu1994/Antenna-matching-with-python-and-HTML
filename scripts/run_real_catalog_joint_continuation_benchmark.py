"""Benchmark joint measured-S2P continuation on Optenni's three-antenna DUT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "apps/api"), str(ROOT / "packages/rfmatch-core/src")]

from engine.component_lib import merge_component_libraries, scan_s2p_directory  # noqa: E402
from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import run_tuning_joint  # noqa: E402
from run_product_joint_timeout_benchmark import (  # noqa: E402
    DEFAULT_COMPONENT_ROOT,
    DEFAULT_TUTORIAL_ROOT,
    RELATIVE_INPUT,
    _sha256,
)
from run_real_catalog_six_component_budget import _catalog_fingerprint  # noqa: E402


def _summary(candidates: dict, elapsed_seconds: float) -> dict:
    best = candidates[0]
    diagnostics = best.search_diagnostics
    return {
        "elapsed_seconds": elapsed_seconds,
        "score_db": best.system_score,
        "topology_code": diagnostics["topology_code"],
        "components_by_port": {
            str(port): len(choices) for port, choices in best.component_choices.items()
        },
        "candidate_count": len(candidates),
        "ideal_evaluations": diagnostics["ideal_evaluations"],
        "physical_evaluations": diagnostics["physical_evaluations"],
        "component_models_loaded": diagnostics["component_models_loaded"],
        "search_profile": diagnostics["search_profile"],
        "search_truncated": diagnostics["search_truncated"],
        "termination_reason": diagnostics["termination_reason"],
        "checkpoint_reused": diagnostics.get("checkpoint_reused", False),
        "checkpoint_prior_ideal_evaluations": diagnostics.get("checkpoint_prior_ideal_evaluations", 0),
        "checkpoint_prior_physical_evaluations": diagnostics.get("checkpoint_prior_physical_evaluations", 0),
        "checkpoint_prior_exact_cache_entries": diagnostics.get("checkpoint_prior_exact_cache_entries", 0),
        "minimum_total_efficiency": best.min_total_efficiency,
        "average_total_efficiency": best.avg_total_efficiency,
        "maximum_power_balance_error": best.maximum_power_balance_error,
    }


def run(
    tutorial_root: Path,
    component_root: Path,
    initial_seconds: float = 5.0,
    additional_seconds: float = 15.0,
    max_components_per_port: int = 1,
    frequency_points_per_band: int = 3,
    beam_width: int = 8,
) -> dict:
    catalog_paths = [
        component_root / "Inductors/Coilcraft Inductors 0402hp",
        component_root / "Capacitors/Murata Capacitors gqm18",
    ]
    library = merge_component_libraries(*(
        scan_s2p_directory(str(path)) for path in catalog_paths
    ))
    input_path = tutorial_root / RELATIVE_INPUT
    dut = load_touchstone_file(input_path)
    specs = [
        {"port_index": 0, "bands_mhz": [[2500, 2690]], "max_components": max_components_per_port},
        {"port_index": 1, "bands_mhz": [[1920, 2170]], "max_components": max_components_per_port},
        {"port_index": 2, "bands_mhz": [[1215, 1300]], "max_components": max_components_per_port},
    ]
    total_seconds = float(initial_seconds + additional_seconds)
    common = dict(
        dut=dut,
        library=library,
        port_specs=specs,
        objective="balanced",
        beam_width=beam_width,
        num_band_points=frequency_points_per_band,
    )

    first_checkpoint = {}
    started = time.perf_counter()
    initial = run_tuning_joint(
        **common,
        timeout_seconds=float(initial_seconds),
        search_profile_timeout_seconds=float(initial_seconds),
        checkpoint_store=first_checkpoint,
    )
    initial_summary = _summary(initial, time.perf_counter() - started)

    next_checkpoint = {}
    started = time.perf_counter()
    continued = run_tuning_joint(
        **common,
        timeout_seconds=float(additional_seconds),
        search_profile_timeout_seconds=total_seconds,
        search_checkpoint=first_checkpoint,
        checkpoint_store=next_checkpoint,
    )
    continued_summary = _summary(continued, time.perf_counter() - started)

    started = time.perf_counter()
    cold = run_tuning_joint(
        **common,
        timeout_seconds=total_seconds,
        search_profile_timeout_seconds=total_seconds,
    )
    cold_summary = _summary(cold, time.perf_counter() - started)

    score_delta = continued_summary["score_db"] - cold_summary["score_db"]
    return {
        "schema_version": 1,
        "case": "official three-antenna real-catalog joint continuation checkpoint",
        "scope": (
            "three coupled ports, zero to "
            f"{max_components_per_port} measured components per port"
        ),
        "input": {
            "relative_path": str(RELATIVE_INPUT).replace("\\", "/"),
            "sha256": _sha256(input_path).upper(),
            "bands_mhz_by_port": {
                str(item["port_index"]): item["bands_mhz"] for item in specs
            },
            "frequency_points_per_band": frequency_points_per_band,
            "maximum_components_by_port": {
                "0": max_components_per_port,
                "1": max_components_per_port,
                "2": max_components_per_port,
            },
            "beam_width": beam_width,
            "catalog_size": {
                "inductors": len(library.inductors),
                "capacitors": len(library.capacitors),
            },
            "catalog_fingerprint": _catalog_fingerprint(catalog_paths),
        },
        "budgets": {
            "initial_seconds": float(initial_seconds),
            "additional_seconds": float(additional_seconds),
            "total_seconds": total_seconds,
        },
        "initial": initial_summary,
        "continued": continued_summary,
        "cold_total_budget": cold_summary,
        "comparison": {
            "continued_score_minus_cold_db": score_delta,
            "continued_matches_cold_score": abs(score_delta) <= 1e-12,
            "continued_matches_cold_topology": (
                continued_summary["topology_code"] == cold_summary["topology_code"]
            ),
            "continued_unique_physical_evaluations_equal_cold": (
                continued_summary["physical_evaluations"] == cold_summary["physical_evaluations"]
            ),
            "continued_quality_not_below_initial": (
                continued_summary["score_db"] >= initial_summary["score_db"] - 1e-12
            ),
            "incremental_elapsed_fraction_of_cold": (
                continued_summary["elapsed_seconds"] / max(cold_summary["elapsed_seconds"], 1e-12)
            ),
            "total_checkpoint_elapsed_fraction_of_cold": (
                (initial_summary["elapsed_seconds"] + continued_summary["elapsed_seconds"])
                / max(cold_summary["elapsed_seconds"], 1e-12)
            ),
        },
    }


def validate(payload: dict) -> None:
    comparison = payload["comparison"]
    required = (
        "continued_matches_cold_score",
        "continued_matches_cold_topology",
        "continued_unique_physical_evaluations_equal_cold",
        "continued_quality_not_below_initial",
    )
    failed = [name for name in required if not comparison[name]]
    if failed:
        raise RuntimeError("joint continuation regression: " + ", ".join(failed))
    if payload["initial"]["search_truncated"] is not True:
        raise RuntimeError("initial joint budget no longer exercises partial continuation")
    if payload["continued"]["search_truncated"] or payload["cold_total_budget"]["search_truncated"]:
        raise RuntimeError("joint continuation comparison did not reach a completed frontier")
    if not payload["continued"]["checkpoint_reused"]:
        raise RuntimeError("joint measured-search checkpoint was not reused")
    if payload["continued"]["checkpoint_prior_physical_evaluations"] != payload["initial"]["physical_evaluations"]:
        raise RuntimeError("joint checkpoint lost prior physical-evaluation accounting")
    if max(
        payload[name]["maximum_power_balance_error"]
        for name in ("initial", "continued", "cold_total_budget")
    ) > 1e-9:
        raise RuntimeError("joint continuation power balance regressed")
    if comparison["incremental_elapsed_fraction_of_cold"] >= 0.8:
        raise RuntimeError("joint incremental continuation no longer saves meaningful time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--initial-seconds", type=float, default=5.0)
    parser.add_argument("--additional-seconds", type=float, default=15.0)
    parser.add_argument("--max-components", type=int, default=1)
    parser.add_argument("--frequency-points", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/benchmarks/real-catalog-joint-continuation-checkpoint.json",
    )
    args = parser.parse_args()
    payload = run(
        args.tutorial_root.resolve(), args.component_root.resolve(),
        args.initial_seconds, args.additional_seconds,
        args.max_components, args.frequency_points, args.beam_width,
    )
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
