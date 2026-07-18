import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "benchmarks" / "priority-weight-scoring-baseline.json"


def test_priority_weight_artifact_locks_candidate_decision():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "deterministic_scoring_contract"
    assert payload["default_priorities"]["winner"] == "A"
    assert payload["port_two_priority"]["winner"] == "B"
    assert payload["default_priorities"]["effective_band_weights_by_port"] == {
        "0": [1.0], "1": [1.0],
    }
    assert payload["port_two_priority"]["effective_band_weights_by_port"] == {
        "0": [1.0], "1": [3.0],
    }
    assert len(payload["input_contract_sha256"]) == 64
    assert len(payload["environment"]["environment_fingerprint_sha256"]) == 64
