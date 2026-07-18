import json
from pathlib import Path


def test_optenni_three_port_search_recall_artifact_meets_p1_gate():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts" / "benchmarks" / "optenni-multiport-search-recall.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == 1
    assert len(artifact["environment"]["environment_fingerprint_sha256"]) == 64
    assert artifact["environment"]["logical_cpu_count"] > 0
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert artifact["input"]["sha256"] == (
        "4af62325474067d752dac32138c447b1da5e714a4140bafed484f0649131f6bc"
    )
    assert artifact["search_config"]["per_port_keep"] == 8
    assert artifact["recall"]["exhaustive_candidate_count"] == 729
    assert artifact["recall"]["top_k"] == 10
    assert artifact["recall"]["exact_top_k_recall"] >= 0.95
    assert artifact["recall"]["topology_top_k_recall"] == 1.0
    assert artifact["recall"]["best_score_gap_db"] < 1e-12
    assert (
        artifact["recall"]["heuristic_physical_evaluations"]
        < artifact["recall"]["exhaustive_physical_evaluations"]
    )
    assert artifact["performance"]["heuristic_physical_evaluations_per_wall_second"] > 0
    assert artifact["performance"]["exhaustive_physical_evaluations_per_wall_second"] > 0
    assert artifact["performance"]["heuristic_peak_tracemalloc_mib"] > 0
    assert all(
        candidate["maximum_power_balance_error"] < 1e-9
        for candidate in artifact["exhaustive_top_k"]
    )
