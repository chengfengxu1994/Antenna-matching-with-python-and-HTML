import json
from pathlib import Path


def test_product_joint_time_budget_artifact_meets_interactive_gate():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-product-joint-time-budget.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == 1
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert artifact["input"]["sha256"] == (
        "4af62325474067d752dac32138c447b1da5e714a4140bafed484f0649131f6bc"
    )
    assert artifact["input"]["timeout_seconds"] == 10.0
    result = artifact["result"]
    assert result["elapsed_seconds"] <= 11.5
    assert result["candidate_count"] == 10
    assert result["score_db"] > -5.0
    assert all(count > 0 for count in result["components_by_port"].values())
    assert result["search_profile"] == "time_budgeted"
    assert result["search_truncated"] is True
    assert result["termination_reason"] == "cancelled during joint ranking"
    assert result["maximum_power_balance_error"] < 1e-12
