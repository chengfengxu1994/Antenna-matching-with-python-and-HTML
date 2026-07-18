import json
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "benchmarks" / "six-component-search-baseline.json"


def test_six_component_search_baseline_proves_exact_small_catalog_recall():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["scope"]["exhaustive_candidates"] == 8192
    assert payload["quality"]["best_score_gap_db"] <= 1e-12
    assert payload["quality"]["exact_top_k_recall"] >= 0.95
    assert payload["quality"]["topology_top_k_recall"] >= 0.95
    assert payload["quality"]["maximum_power_balance_error"] < 1e-10
    assert payload["progressive"]["best_signature"] == payload["exhaustive"]["best_signature"]
    assert payload["progressive"]["peak_traced_bytes"] < 64 * 1024 * 1024
    assert payload["progressive"]["physical_evaluations"] <= 2048
    assert payload["progressive"]["ideal_evaluations"] <= 7000
    assert payload["progressive"]["elapsed_seconds"] <= payload["exhaustive"]["elapsed_seconds"] * 0.8
