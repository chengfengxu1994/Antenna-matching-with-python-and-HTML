"""Assemble recall and normalized performance into one machine-scoped manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/search-performance-baseline.json"
CASES = {
    "single_port": ROOT / "artifacts/benchmarks/optenni-single-port-search-recall.json",
    "multiport": ROOT / "artifacts/benchmarks/optenni-multiport-search-recall.json",
    "four_port": ROOT / "artifacts/benchmarks/optenni-four-port-search-recall.json",
}


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def build() -> dict:
    result = {}
    fingerprints = set()
    for name, path in CASES.items():
        document, digest = _load(path)
        environment = document["environment"]
        fingerprints.add(environment["environment_fingerprint_sha256"])
        recall = document["recall"]
        performance = document["performance"]
        result[name] = {
            "source_artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": digest,
            "case": document["case"],
            "rfmatch_core_version": document["software"]["rfmatch_core_version"],
            "environment": environment,
            "quality": {
                "top_k": recall["top_k"],
                "exact_top_k_recall": recall["exact_top_k_recall"],
                "topology_top_k_recall": recall["topology_top_k_recall"],
                "best_score_gap_db": recall["best_score_gap_db"],
            },
            "workload": {
                "heuristic_physical_evaluations": recall["heuristic_physical_evaluations"],
                "exhaustive_physical_evaluations": recall["exhaustive_physical_evaluations"],
            },
            "performance": performance,
            "relative_to_exhaustive_same_run": {
                "wall_time_ratio": performance["heuristic_to_exhaustive_wall_time_ratio"],
                "physical_evaluation_reduction_fraction": 1.0 - (
                    recall["heuristic_physical_evaluations"]
                    / recall["exhaustive_physical_evaluations"]
                ),
                "python_peak_allocation_reduction_fraction": 1.0 - (
                    performance["heuristic_peak_tracemalloc_bytes"]
                    / performance["exhaustive_peak_tracemalloc_bytes"]
                ),
            },
        }
    if len(fingerprints) != 1:
        raise ValueError(
            "search performance artifacts were captured on different environments"
        )
    return {
        "schema_version": 1,
        "case": "RF Match measured-search recall and machine-scoped performance baseline",
        "comparison_policy": {
            "quality_gate": "exact top-10 recall >= 0.95 and zero best-score gap",
            "cross_machine": "compare quality/workload freely; compare wall time only when environment fingerprint matches",
            "memory_scope": "tracemalloc Python allocations only, not total process RSS",
        },
        "environment_fingerprint_sha256": fingerprints.pop(),
        "cases": result,
    }


def main() -> None:
    document = build()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(DEFAULT_OUTPUT),
        "environment_fingerprint_sha256": document["environment_fingerprint_sha256"],
        "cases": {
            name: {
                "exact_top_k_recall": case["quality"]["exact_top_k_recall"],
                "wall_time_ratio": case["relative_to_exhaustive_same_run"]["wall_time_ratio"],
                "evaluations_per_second": case["performance"]["heuristic_physical_evaluations_per_wall_second"],
            }
            for name, case in document["cases"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
