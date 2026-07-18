import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/benchmarks/real-catalog-joint-two-component-continuation.json"


def test_real_catalog_joint_two_component_checkpoint_matches_completed_cold_search():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["input"]["catalog_size"] == {"inductors": 49, "capacitors": 322}
    assert payload["input"]["maximum_components_by_port"] == {"0": 2, "1": 2, "2": 2}
    assert payload["input"]["frequency_points_per_band"] == 2
    assert payload["budgets"] == {
        "initial_seconds": 8.0,
        "additional_seconds": 20.0,
        "total_seconds": 28.0,
    }
    assert payload["initial"]["search_truncated"] is True
    assert payload["continued"]["search_truncated"] is False
    assert payload["cold_total_budget"]["search_truncated"] is False
    assert payload["continued"]["checkpoint_reused"] is True
    assert payload["continued"]["checkpoint_prior_physical_evaluations"] == payload["initial"]["physical_evaluations"]
    assert payload["continued"]["checkpoint_prior_exact_cache_entries"] == payload["initial"]["physical_evaluations"]

    comparison = payload["comparison"]
    assert comparison["continued_matches_cold_score"] is True
    assert comparison["continued_matches_cold_topology"] is True
    assert comparison["continued_unique_physical_evaluations_equal_cold"] is True
    assert comparison["continued_quality_not_below_initial"] is True
    assert comparison["incremental_elapsed_fraction_of_cold"] < 0.8
    assert payload["continued"]["physical_evaluations"] == 722
    assert payload["continued"]["maximum_power_balance_error"] <= 1e-9
