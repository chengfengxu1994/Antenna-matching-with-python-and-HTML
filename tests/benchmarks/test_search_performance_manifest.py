import hashlib
import json
from pathlib import Path


def test_search_performance_manifest_is_hashed_machine_scoped_and_quality_gated():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "artifacts/benchmarks/search-performance-baseline.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert len(manifest["environment_fingerprint_sha256"]) == 64
    assert "environment fingerprint matches" in manifest["comparison_policy"]["cross_machine"]
    for case in manifest["cases"].values():
        source = root / case["source_artifact"]
        assert case["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert case["environment"]["environment_fingerprint_sha256"] == manifest["environment_fingerprint_sha256"]
        assert case["quality"]["exact_top_k_recall"] >= 0.95
        assert case["quality"]["topology_top_k_recall"] == 1.0
        assert case["quality"]["best_score_gap_db"] < 1e-12
        assert case["performance"]["heuristic_physical_evaluations_per_wall_second"] > 0
        assert 0 < case["relative_to_exhaustive_same_run"]["wall_time_ratio"] < 1
        assert case["relative_to_exhaustive_same_run"]["physical_evaluation_reduction_fraction"] > 0
