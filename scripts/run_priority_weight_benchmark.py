"""Build a deterministic golden for port/band priority scoring semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from benchmark_metadata import benchmark_environment
from rfmatch_core.evaluator import score_sweep
from rfmatch_core.models import Band, Candidate, Objective, Problem


OUTPUT = ROOT / "artifacts" / "benchmarks" / "priority-weight-scoring-baseline.json"


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _matched_from_efficiency(total_efficiency: np.ndarray) -> np.ndarray:
    frequencies, ports = total_efficiency.shape
    matched = np.zeros((frequencies, ports, ports), dtype=complex)
    for index in range(ports):
        matched[:, index, index] = np.sqrt(np.maximum(0.0, 1.0 - total_efficiency[:, index]))
    return matched


def _score_case(port_two_weight: float) -> dict:
    frequencies = np.asarray([1.0e9, 2.0e9])
    bands = {
        0: (Band(1.0e9, 1.0e9, weight=1.0),),
        1: (Band(2.0e9, 2.0e9, weight=port_two_weight),),
    }
    problem = Problem(frequencies, np.zeros((2, 2, 2), dtype=complex), bands)
    objective = Objective(
        within_band_average_weight=1.0,
        across_band_average_weight=1.0,
        port_average_weight=1.0,
    )
    traces = {
        "A": np.asarray([[0.95, 0.60], [0.95, 0.60]]),
        "B": np.asarray([[0.75, 0.75], [0.75, 0.75]]),
    }
    scores = {}
    for name, efficiency in traces.items():
        evaluated = score_sweep(
            problem,
            Candidate([]),
            objective,
            _matched_from_efficiency(efficiency),
            efficiency,
        )
        scores[name] = {
            "score_db": float(evaluated.score_db),
            "port_scores_db": [float(value) for value in evaluated.metrics["port_scores_db"]],
        }
    winner = max(scores, key=lambda name: scores[name]["score_db"])
    return {
        "effective_band_weights_by_port": {"0": [1.0], "1": [float(port_two_weight)]},
        "scores": scores,
        "winner": winner,
    }


def build_document() -> dict:
    input_contract = {
        "frequencies_hz": [1.0e9, 2.0e9],
        "candidate_total_efficiency": {
            "A": [[0.95, 0.60], [0.95, 0.60]],
            "B": [[0.75, 0.75], [0.75, 0.75]],
        },
        "objective": {
            "within_band_average_weight": 1.0,
            "across_band_average_weight": 1.0,
            "port_average_weight": 1.0,
        },
    }
    default = _score_case(1.0)
    port_two_priority = _score_case(3.0)
    if default["winner"] != "A" or port_two_priority["winner"] != "B":
        raise RuntimeError("priority-weight scoring no longer changes the expected engineering decision")
    return {
        "schema_version": 1,
        "case": "Port and band priority changes candidate decision",
        "status": "deterministic_scoring_contract",
        "input_contract": input_contract,
        "input_contract_sha256": _canonical_sha(input_contract),
        "weight_semantics": (
            "effective band weight = port_weight * band_weight; the authoritative evaluator "
            "multiplies each dB target margin before across-band and across-port aggregation"
        ),
        "default_priorities": default,
        "port_two_priority": port_two_priority,
        "acceptance": {
            "default_winner": "A",
            "prioritized_winner": "B",
            "default_weights_preserve_existing_behavior": True,
        },
        "environment": benchmark_environment(),
    }


def main() -> None:
    document = build_document()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "default_winner": document["default_priorities"]["winner"],
        "prioritized_winner": document["port_two_priority"]["winner"],
        "environment": document["environment"]["environment_fingerprint_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
