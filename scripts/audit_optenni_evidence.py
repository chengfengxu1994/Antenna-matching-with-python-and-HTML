"""Audit the tracked Optenni evidence matrix without licensed source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "optenni" / "cases.json"
EVIDENCE_LEVELS = {
    "native_curve_export",
    "saved_project_plus_ui_summary",
    "published_rounded_reference",
    "tutorial_reference_circuit",
    "rfmatch_recompute_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    for case in manifest["cases"]:
        evidence = case.get("evidence") or {}
        level = evidence.get("level")
        files = []
        for kind in ("manifests", "artifacts"):
            for relative in evidence.get(kind, []):
                path = (ROOT / relative).resolve()
                inside_root = path == ROOT or ROOT in path.parents
                exists = inside_root and path.is_file()
                files.append({
                    "kind": kind[:-1],
                    "path": relative,
                    "exists": exists,
                    "sha256": sha256_file(path) if exists else None,
                })
        valid = bool(
            case.get("status") == "implemented"
            and level in EVIDENCE_LEVELS
            and files
            and all(item["exists"] for item in files)
            and isinstance(evidence.get("remaining_gap"), str)
            and evidence["remaining_gap"].strip()
            and (
                level != "rfmatch_recompute_only"
                or evidence.get("cross_software_numeric") is False
            )
        )
        results.append({
            "id": case["id"],
            "valid": valid,
            "level": level,
            "cross_software_numeric": evidence.get("cross_software_numeric"),
            "files": files,
            "remaining_gap": evidence.get("remaining_gap"),
        })
    return {
        "schema_version": manifest.get("schema_version"),
        "valid": all(item["valid"] for item in results),
        "case_count": len(results),
        "native_curve_export_count": sum(
            item["level"] == "native_curve_export" for item in results
        ),
        "cases": results,
    }


def main() -> int:
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
