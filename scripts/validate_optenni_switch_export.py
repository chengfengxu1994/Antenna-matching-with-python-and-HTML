"""Compare Optenni switch curve CSV exports with the rfmatch full-wave baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core.switch_golden import compare_switch_export, load_switch_export_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="one combined CSV or one CSV per Set")
    parser.add_argument(
        "--configuration",
        action="append",
        help="configuration name for a CSV without a configuration column; repeat in CSV order",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "optenni-switch-full-network-measured-baseline.json",
    )
    parser.add_argument("--s11-tolerance-db", type=float, default=0.05)
    parser.add_argument("--efficiency-tolerance", type=float, default=0.005)
    parser.add_argument("--report", type=Path, help="optional JSON report output")
    args = parser.parse_args()
    configurations = args.configuration or []
    if configurations and len(configurations) != len(args.csv):
        parser.error("--configuration must be repeated once for every CSV")
    try:
        document = json.loads(args.baseline.read_text(encoding="utf-8"))
        result = next(
            item for item in document["results"]
            if item.get("mode") == "switch-mdif-full-network-measured-synthesis"
        )
        curves = result["configuration_curves"]
        points = []
        for index, path in enumerate(args.csv):
            points.extend(load_switch_export_csv(
                path,
                default_configuration=configurations[index] if configurations else None,
            ))
        report = compare_switch_export(
            points,
            curves,
            s11_tolerance_db=args.s11_tolerance_db,
            efficiency_tolerance=args.efficiency_tolerance,
        )
        report["baseline"] = str(args.baseline)
        report["exports"] = [str(path) for path in args.csv]
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        report = {"passed": False, "error": str(exc)}
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
