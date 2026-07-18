"""Extract a reproducible, read-only manifest from an Optenni OPR project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import parse_optenni_opr  # noqa: E402


RELATIVE_PROJECT = Path(
    "4 - Multiantenna system/4.1 Multiantenna system at different bands/"
    "multiantenna_project.opr"
)
DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/optenni_exports/multiantenna_project_opr_manifest.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project", nargs="?", type=Path,
        default=DEFAULT_TUTORIAL_ROOT / RELATIVE_PROJECT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = parse_optenni_opr(args.project.resolve())
    result["project_relative_path"] = str(RELATIVE_PROJECT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "project_sha256": result["project_sha256"],
        "candidate_count": result["candidate_count"],
        "saved_winner": result["saved_winner"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
