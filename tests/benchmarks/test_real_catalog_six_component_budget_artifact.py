import json
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "benchmarks" / "real-catalog-six-component-budget.json"


def test_real_catalog_six_component_budget_curve_is_anytime_and_traceable():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["input"]["catalog_size"]["inductors"] >= 40
    assert payload["input"]["catalog_size"]["capacitors"] >= 50
    assert len(payload["input"]["catalog_fingerprint"]) == 64
    rows = payload["budgets"]
    assert [row["budget_seconds"] for row in rows] == sorted(row["budget_seconds"] for row in rows)
    assert all(later["score_db"] >= earlier["score_db"] - 1e-12 for earlier, later in zip(rows, rows[1:]))
    assert rows[0]["search_truncated"] is True
    assert rows[-1]["component_count"] == 6
    assert rows[-1]["search_truncated"] is False
    assert rows[-1]["component_models_loaded"] < sum(payload["input"]["catalog_size"].values())
    assert all(row["maximum_power_balance_error"] < 1e-10 for row in rows)
    assert all(row["elapsed_seconds"] <= row["budget_seconds"] + 0.25 for row in rows)
