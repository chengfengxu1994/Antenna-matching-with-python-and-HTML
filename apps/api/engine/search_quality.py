"""Product-level search quality profiles and truthful execution plans."""

from __future__ import annotations

from typing import Any, Mapping

from .calibration_evidence import multiport_calibration_reference


SEARCH_QUALITY_PROFILES = {
    "quick": {
        "label": "Quick exploration",
        "timeout_seconds": 15.0,
        "beam_width": 8,
        "num_band_points": 3,
        "expectation": "Fast candidate exploration; partial search is expected on large catalogs.",
    },
    "balanced": {
        "label": "Balanced engineering",
        "timeout_seconds": 45.0,
        "beam_width": 10,
        "num_band_points": 5,
        "expectation": "Routine engineering search with bounded measured-component refinement.",
    },
    "thorough": {
        "label": "Thorough constrained",
        "timeout_seconds": 120.0,
        "beam_width": 10,
        "num_band_points": 10,
        "expectation": "Deep coupled refinement; calibrated for explicit two-component port topologies.",
    },
    "exhaustive": {
        "label": "Automatic topology deep",
        "timeout_seconds": 150.0,
        "beam_width": 50,
        "num_band_points": 3,
        "expectation": "Offline-quality topology discovery for eligible coupled systems; not a proof of global optimality.",
    },
}


def infer_search_quality(timeout_seconds: float) -> str:
    if timeout_seconds < 30:
        return "quick"
    if timeout_seconds < 60:
        return "balanced"
    if timeout_seconds < 150:
        return "thorough"
    return "exhaustive"


def build_search_plan(request: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the strategy a unified tuning request is eligible to use."""
    timeout = max(0.0, float(request.get("timeout_seconds") or 0.0))
    requested = str(request.get("search_quality") or "auto")
    effective = (
        requested if requested in SEARCH_QUALITY_PROFILES
        else infer_search_quality(timeout)
    )
    enabled = [item for item in request.get("ports", ()) if item.get("enabled", True)]
    max_components = max(
        (int(item.get("max_components", 2)) for item in enabled), default=0
    )
    constraints = [
        item.get("allowed_topology_codes") for item in enabled
    ]
    all_constrained = bool(enabled) and all(bool(value) for value in constraints)
    no_constraints = bool(enabled) and all(value is None for value in constraints)
    automatic_eligible = bool(
        request.get("mode") == "joint"
        and 2 <= len(enabled) <= 3
        and max_components <= 2
        and no_constraints
        and timeout >= 150
    )
    constrained_coupled = bool(
        request.get("mode") == "joint"
        and len(enabled) >= 2
        and max_components <= 2
        and all_constrained
        and timeout >= 60
    )
    calibration = multiport_calibration_reference()
    if automatic_eligible:
        strategy = "automatic_topology_deep"
        automatic = calibration["automatic_full_catalog_discovery"]
        evidence = (
            f"Hashed official product 3-port reference: topology rank "
            f"{automatic['product_saved_topology_rank']}, exact BOM rank "
            f"{automatic['product_exact_saved_winner_rank']}."
        )
    elif constrained_coupled:
        strategy = "constrained_coupled_thorough"
        constrained = calibration["full_catalog_discovery"]
        evidence = (
            f"Hashed official constrained 3-port reference: exact BOM rank "
            f"{constrained['exact_saved_winner_rank']}."
        )
    else:
        strategy = "hierarchical_measured"
        evidence = "Hierarchical measured search; quality remains request- and catalog-dependent."
    profile = SEARCH_QUALITY_PROFILES[effective]
    return {
        "requested_quality": requested,
        "effective_quality": effective,
        "label": profile["label"],
        "strategy": strategy,
        "budget_seconds": timeout,
        "output_candidates": int(request.get("beam_width") or 0),
        "band_points": int(request.get("num_band_points") or 0),
        "automatic_topology_eligible": automatic_eligible,
        "constrained_coupled_eligible": constrained_coupled,
        "expectation": profile["expectation"],
        "calibration_evidence": evidence,
        "calibration_artifact": calibration["artifact"],
        "calibration_artifact_sha256": calibration["artifact_sha256"],
    }


def build_multi_scenario_search_plan(request: Mapping[str, Any]) -> dict[str, Any]:
    """Describe a shared-network search without implying exhaustive coverage."""
    timeout = max(0.0, float(request.get("timeout_seconds") or 0.0))
    requested = str(request.get("search_quality") or "auto")
    effective = (
        requested if requested in SEARCH_QUALITY_PROFILES
        else infer_search_quality(timeout)
    )
    profile = SEARCH_QUALITY_PROFILES[effective]
    return {
        "requested_quality": requested,
        "effective_quality": effective,
        "label": profile["label"],
        "strategy": "shared_network_measured_beam",
        "budget_seconds": timeout,
        "beam_width": int(request.get("beam_width") or 0),
        "band_points": int(request.get("num_band_points") or 0),
        "verification_band_points": int(
            request.get("verification_band_points") or request.get("num_band_points") or 0
        ),
        "scenario_count": len(request.get("scenarios") or ()),
        "topology_count": len(request.get("topology_names") or ()),
        "expectation": (
            "One identical physical network is scored across every scenario. "
            "The measured-component beam is bounded and is not a proof of global optimality."
        ),
        "calibration_evidence": (
            "Official three-state full-family product gate: all 16 two-component topologies screened, "
            "Optenni topology rank 1 and reference-value network rank 2 after dense verification."
        ),
    }
