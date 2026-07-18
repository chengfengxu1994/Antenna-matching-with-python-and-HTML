"""Build product calibration claims directly from reproducible benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CalibrationEvidenceError(ValueError):
    pass


def _artifact(relative_path: str, root: Path = PROJECT_ROOT) -> tuple[dict, dict]:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CalibrationEvidenceError("calibration artifact escapes the project root") from exc
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationEvidenceError(
            f"calibration artifact {relative_path!r} is unavailable or invalid: {exc}"
        ) from exc
    if document.get("schema_version") != 1 or not document.get("case"):
        raise CalibrationEvidenceError(
            f"calibration artifact {relative_path!r} has an unsupported schema"
        )
    evidence = {
        "artifact": relative_path.replace("\\", "/"),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": document["schema_version"],
        "case": document["case"],
        "integrity": "sha256_verified_at_runtime",
    }
    return document, evidence


def single_port_calibration_reference(root: Path = PROJECT_ROOT) -> dict:
    recall_doc, recall_evidence = _artifact(
        "artifacts/benchmarks/optenni-single-port-search-recall.json", root
    )
    product_doc, product_evidence = _artifact(
        "artifacts/benchmarks/optenni-product-optimization-settings.json", root
    )
    recall = recall_doc["recall"]
    config = recall_doc["search_config"]
    grid = recall_doc["component_grid"]
    comparison = product_doc["cross_software_efficiency_comparison"]
    return {
        **recall_evidence,
        "status": "reference_only_not_request_specific",
        "scope": {
            "ports": 1,
            "maximum_components_per_port": config["max_components_per_port"],
            "catalog": (
                f"exhaustive grid: {len(grid['inductors_nh'])} inductors + "
                f"{len(grid['capacitors_pf'])} capacitors"
            ),
            "per_port_keep": config["per_port_keep"],
            "top_k": recall["top_k"],
        },
        "reference_exact_top_k_recall": recall["exact_top_k_recall"],
        "reference_topology_top_k_recall": recall["topology_top_k_recall"],
        "reference_best_score_gap_db": recall["best_score_gap_db"],
        "product_full_catalog_discovery": {
            **product_evidence,
            "status": "topology_and_cross_software_not_exhaustive_recall",
            "catalog": (
                f"{product_doc['requested_catalog_size']['inductors']} inductors + "
                f"{product_doc['requested_catalog_size']['capacitors']} capacitors"
            ),
            "topology": product_doc["optenni_reference"]["topology_code"],
            "topology_rank": product_doc["optenni_topology_rank"],
            "parts": [
                item["part_number"] for item in product_doc["best"]["components"]
            ],
            "maximum_efficiency_delta_db": max(
                abs(comparison["minimum_efficiency_delta_db"]),
                abs(comparison["average_efficiency_delta_db"]),
            ),
            "physical_evaluations": product_doc["search_diagnostics"]["physical_evaluations"],
            "loaded_component_models": product_doc["search_diagnostics"]["component_models_loaded"],
            "wall_seconds": product_doc["wall_seconds"],
        },
    }


def multiport_calibration_reference(root: Path = PROJECT_ROOT) -> dict:
    recall_doc, recall_evidence = _artifact(
        "artifacts/benchmarks/optenni-multiport-search-recall.json", root
    )
    golden_doc, golden_evidence = _artifact(
        "artifacts/benchmarks/optenni-multiport-live-golden.json", root
    )
    saved_doc, saved_evidence = _artifact(
        "artifacts/benchmarks/optenni-saved-winner-discovery.json", root
    )
    full_doc, full_evidence = _artifact(
        "artifacts/benchmarks/optenni-saved-winner-full-catalog-discovery.json", root
    )
    auto_doc, auto_evidence = _artifact(
        "artifacts/benchmarks/optenni-saved-winner-full-catalog-automatic.json", root
    )
    product_doc, product_evidence = _artifact(
        "artifacts/benchmarks/optenni-product-saved-winner-automatic.json", root
    )
    four_doc, four_evidence = _artifact(
        "artifacts/benchmarks/optenni-four-port-search-recall.json", root
    )
    recall = recall_doc["recall"]
    config = recall_doc["search_config"]
    kinds = [item["kind"] for item in recall_doc["component_grid"]]
    efficiency_deltas = [
        abs(value)
        for port in golden_doc["efficiency_comparison"].values()
        for value in port["difference_from_rounded_ui_db"].values()
    ]
    radiated_delta = abs(
        golden_doc["power_balance_comparison"]["difference_from_rounded_ui_linear"]["radiated"]
    )
    saved = saved_doc["searches"]["automatic"]
    full = full_doc["searches"]["topology_constrained"]
    automatic = auto_doc["searches"]["automatic"]
    product_diagnostics = product_doc["search_diagnostics"]
    return {
        **recall_evidence,
        "status": "reference_only_not_request_specific",
        "scope": {
            "ports": len(recall_doc["input"]["bands_by_port_hz"]),
            "maximum_components_per_port": config["max_components_per_port"],
            "catalog": f"{kinds.count('L')} inductors + {kinds.count('C')} capacitors",
            "per_port_keep": config["per_port_keep"],
            "top_k": recall["top_k"],
        },
        "reference_exact_top_k_recall": recall["exact_top_k_recall"],
        "reference_topology_top_k_recall": recall["topology_top_k_recall"],
        "minimum_calibrated_per_port_keep": config["per_port_keep"],
        "four_port_scaling": {
            **four_evidence,
            "status": "reference_only_not_request_specific",
            "scope": {
                "ports": four_doc["input"]["ports"],
                "maximum_components_per_port": four_doc["candidate_space"]["maximum_components_per_port"],
                "catalog": "2 inductors + 2 capacitors; series/shunt placements",
                "exhaustive_candidates": four_doc["candidate_space"]["exhaustive_candidates"],
                "per_port_keep": four_doc["minimum_calibrated_per_port_keep"],
                "top_k": four_doc["recall"]["top_k"],
            },
            "reference_exact_top_k_recall": four_doc["recall"]["exact_top_k_recall"],
            "reference_topology_top_k_recall": four_doc["recall"]["topology_top_k_recall"],
            "reference_best_score_gap_db": four_doc["recall"]["best_score_gap_db"],
            "heuristic_physical_evaluations": four_doc["recall"]["heuristic_physical_evaluations"],
            "exhaustive_physical_evaluations": four_doc["recall"]["exhaustive_physical_evaluations"],
            "heuristic_wall_seconds": four_doc["performance"]["heuristic_wall_seconds"],
            "exhaustive_wall_seconds": four_doc["performance"]["exhaustive_wall_seconds"],
        },
        "numerical_golden": {
            **golden_evidence,
            "maximum_efficiency_delta_from_rounded_ui_db": max(efficiency_deltas),
            "maximum_radiated_power_delta_linear": radiated_delta,
        },
        "saved_winner_discovery": {
            **saved_evidence,
            "status": "reference_only_not_request_specific",
            "ports": 3,
            "maximum_components_per_port": 2,
            "catalog": (
                f"saved BOM grid: {saved_doc['catalog_size']['inductors']} inductors + "
                f"{saved_doc['catalog_size']['capacitors']} capacitors"
            ),
            "per_port_keep": saved["config"]["per_port_keep"],
            "exact_saved_winner_rank": saved["exact_saved_winner_rank"],
            "saved_topology_rank": saved["saved_topology_rank"],
            "physical_evaluations": saved["physical_evaluations"],
        },
        "full_catalog_discovery": {
            **full_evidence,
            "status": "reference_only_not_request_specific",
            "catalog": (
                f"{full_doc['catalog_size']['inductors']} inductors + "
                f"{full_doc['catalog_size']['capacitors']} capacitors"
            ),
            "topology_constraint": "SCPL / PCSL / PCSL",
            "exact_saved_winner_rank": full["exact_saved_winner_rank"],
            "best_score_improvement_db": full["best_minus_saved_score_db"],
            "physical_evaluations": full["physical_evaluations"],
            "loaded_component_models": full["loaded_component_models"],
        },
        "automatic_full_catalog_discovery": {
            **auto_evidence,
            "product_artifact": product_evidence["artifact"],
            "product_artifact_sha256": product_evidence["artifact_sha256"],
            "status": "reference_only_not_request_specific",
            "catalog": (
                f"{auto_doc['catalog_size']['inductors']} inductors + "
                f"{auto_doc['catalog_size']['capacitors']} capacitors"
            ),
            "topology_constraint": None,
            "saved_topology_rank": automatic["saved_topology_rank"],
            "exact_saved_winner_rank": automatic["exact_saved_winner_rank"],
            "best_score_improvement_db": automatic["best_minus_saved_score_db"],
            "wall_seconds": automatic["wall_seconds"],
            "physical_evaluations": automatic["physical_evaluations"],
            "product_wall_seconds": product_doc["wall_seconds"],
            "product_budget_seconds": product_doc["request"]["timeout_seconds"],
            "product_saved_topology_rank": product_doc["saved_topology_rank"],
            "product_exact_saved_winner_rank": product_doc["exact_saved_winner_rank"],
            "product_physical_evaluations": product_diagnostics["physical_evaluations"],
            "product_loaded_component_models": product_diagnostics["component_models_loaded"],
            "product_joint_refine_beam_width": product_diagnostics["joint_refine_beam_width"],
            "product_search_truncated": product_diagnostics["search_truncated"],
        },
    }


def search_performance_reference(root: Path = PROJECT_ROOT) -> dict:
    document, evidence = _artifact(
        "artifacts/benchmarks/search-performance-baseline.json", root
    )
    return {
        **evidence,
        "status": "machine_scoped_reference",
        "environment_fingerprint_sha256": document["environment_fingerprint_sha256"],
        "comparison_policy": document["comparison_policy"],
        "cases": {
            name: {
                "exact_top_k_recall": item["quality"]["exact_top_k_recall"],
                "heuristic_wall_seconds": item["performance"]["heuristic_wall_seconds"],
                "heuristic_physical_evaluations_per_wall_second": item["performance"]["heuristic_physical_evaluations_per_wall_second"],
                "wall_time_ratio_to_exhaustive": item["relative_to_exhaustive_same_run"]["wall_time_ratio"],
                "physical_evaluation_reduction_fraction": item["relative_to_exhaustive_same_run"]["physical_evaluation_reduction_fraction"],
            }
            for name, item in document["cases"].items()
        },
    }
