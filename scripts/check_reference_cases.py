"""Inventory the licensed local Optenni tutorial inputs used by parity benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core.touchstone import load_touchstone


DEFAULT_TUTORIAL_ROOT = Path(r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials")
MANIFEST = ROOT / "benchmarks" / "optenni" / "cases.json"


def _paths(case: dict) -> list[tuple[str, str]]:
    result = []
    if case.get("input"):
        result.append(("input", case["input"]))
    result.extend(("input", value) for value in case.get("inputs", []))
    for key in ("efficiency", "component_model"):
        if case.get(key):
            result.append((key, case[key]))
    result.extend(("component_model", value) for value in case.get("component_models", []))
    return result


def inspect_cases(tutorial_root: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case_results = []
    ready = True
    for case in manifest["cases"]:
        files = []
        port_counts = []
        for role, relative in _paths(case):
            path = tutorial_root / Path(relative)
            item = {"role": role, "relative_path": relative, "exists": path.is_file()}
            if item["exists"] and role == "input" and path.suffix.lower().startswith(".s") and path.suffix.lower().endswith("p"):
                try:
                    data = load_touchstone(path)
                    item["ports"] = int(data.s_parameters.shape[1])
                    item["frequency_points"] = int(len(data.frequencies_hz))
                    port_counts.append(item["ports"])
                except Exception as exc:
                    item["parse_error"] = str(exc)
            files.append(item)
        case_ready = all(item["exists"] and not item.get("parse_error") for item in files)
        expected_ports = case.get("ports")
        if port_counts and expected_ports is not None:
            case_ready = case_ready and all(value == expected_ports for value in port_counts)
        ready = ready and case_ready
        case_results.append({"id": case["id"], "status": case["status"], "ready": case_ready, "files": files})
    return {"tutorial_root": str(tutorial_root), "ready": ready, "cases": case_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tutorial-root",
        type=Path,
        default=Path(os.environ.get("OPTENNI_TUTORIAL_ROOT", DEFAULT_TUTORIAL_ROOT)),
    )
    args = parser.parse_args()
    result = inspect_cases(args.tutorial_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
