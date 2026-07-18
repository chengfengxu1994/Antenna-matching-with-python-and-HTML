"""Validate and summarize a user-exported Optenni golden CSV."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core.golden import load_golden_csv, validate_golden_points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_golden_points(load_golden_csv(args.csv))
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    payload = asdict(summary)
    payload["ports"] = [port + 1 for port in summary.ports]
    payload["s_parameter_pairs"] = [
        {
            "source_port": source + 1,
            "destination_port": destination + 1,
            "parameter": f"S{destination + 1}{source + 1}",
        }
        for source, destination in summary.s_parameter_pairs
    ]
    print(json.dumps({"status": "valid", **payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
