import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "benchmarks" / "optenni-four-port-search-recall.json"


def test_four_port_search_recall_artifact_meets_quality_gate():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["input"]["ports"] == 4
    assert payload["input"]["source_frequency_points"] == 1001
    assert len(payload["input"]["sha256"]) == 64
    assert payload["candidate_space"] == {
        "choices_per_port": 9,
        "ports": 4,
        "exhaustive_candidates": 6561,
        "maximum_components_per_port": 1,
    }
    assert payload["recall"]["exact_top_k_recall"] >= 0.95
    assert payload["recall"]["topology_top_k_recall"] >= 0.95
    assert abs(payload["recall"]["best_score_gap_db"]) <= 1e-12
    assert payload["recall"]["exhaustive_physical_evaluations"] == 6561
    assert max(
        item["maximum_power_balance_error"]
        for item in payload["heuristic_top_k"]
    ) <= 1e-9
    assert len(payload["environment"]["environment_fingerprint_sha256"]) == 64
    assert payload["performance"]["heuristic_physical_evaluations_per_wall_second"] > 0
