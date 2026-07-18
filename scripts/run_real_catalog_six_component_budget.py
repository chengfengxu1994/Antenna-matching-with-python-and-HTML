"""Benchmark six-component product search under fixed budgets on real vendor S2P catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "apps/api"), str(ROOT / "packages/rfmatch-core/src")]

from engine.component_lib import merge_component_libraries, scan_s2p_directory  # noqa: E402
from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import run_tuning_single  # noqa: E402


DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
RELATIVE_INPUT = Path("3 - Optimization settings/measured_antenna.s1p")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _catalog_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in paths:
        for path in sorted(root.rglob("*.s2p"), key=lambda item: str(item).lower()):
            stat = path.stat()
            digest.update(str(path.relative_to(root)).replace("\\", "/").lower().encode())
            digest.update(str(stat.st_size).encode())
    return digest.hexdigest().upper()


def run(tutorial_root: Path, component_root: Path, budgets: list[float]) -> dict:
    catalog_paths = [
        component_root / "Inductors/Coilcraft Inductors 0402hp",
        component_root / "Capacitors/Murata Capacitors gqm18",
    ]
    library = merge_component_libraries(*(
        scan_s2p_directory(str(path)) for path in catalog_paths
    ))
    input_path = tutorial_root / RELATIVE_INPUT
    dut = load_touchstone_file(input_path)
    rows = []
    for budget in budgets:
        started = time.perf_counter()
        candidates = run_tuning_single(
            dut, library, 0, [[1700.0, 2500.0]],
            max_components=6, objective="balanced", beam_width=10,
            timeout_seconds=float(budget), num_band_points=5,
        )
        elapsed = time.perf_counter() - started
        best = candidates[0]
        diagnostics = best.search_diagnostics
        rows.append({
            "budget_seconds": float(budget),
            "elapsed_seconds": elapsed,
            "score_db": best.system_score,
            "minimum_total_efficiency": best.min_total_efficiency,
            "average_total_efficiency": best.avg_total_efficiency,
            "topology_code": diagnostics.get("topology_code"),
            "component_count": best.total_component_count,
            "ideal_evaluations": diagnostics.get("ideal_evaluations"),
            "physical_evaluations": diagnostics.get("physical_evaluations"),
            "component_models_loaded": diagnostics.get("component_models_loaded"),
            "search_truncated": diagnostics.get("search_truncated"),
            "termination_reason": diagnostics.get("termination_reason"),
            "maximum_power_balance_error": best.maximum_power_balance_error,
        })
    reference_score = max(row["score_db"] for row in rows)
    for row in rows:
        row["gap_to_best_budget_db"] = reference_score - row["score_db"]
    return {
        "schema_version": 1,
        "case": "real 0402HP/GQM18 six-component product budget curve",
        "input": {
            "relative_path": str(RELATIVE_INPUT).replace("\\", "/"),
            "sha256": _sha256(input_path),
            "bands_mhz": [[1700.0, 2500.0]],
            "frequency_points": 5,
            "maximum_components": 6,
            "beam_width": 10,
            "catalog_size": {"inductors": len(library.inductors), "capacitors": len(library.capacitors)},
            "catalog_fingerprint": _catalog_fingerprint(catalog_paths),
        },
        "budgets": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--budget", type=float, action="append", dest="budgets")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/real-catalog-six-component-budget.json")
    args = parser.parse_args()
    payload = run(
        args.tutorial_root.resolve(), args.component_root.resolve(),
        args.budgets or [2.0, 8.0, 20.0],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
