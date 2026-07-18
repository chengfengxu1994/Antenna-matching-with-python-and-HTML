"""Benchmark the product joint-search deadline on the official three-port case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
CORE_ROOT = ROOT / "packages" / "rfmatch-core" / "src"
sys.path[:0] = [str(API_ROOT), str(CORE_ROOT)]

from engine.component_lib import (  # noqa: E402
    merge_component_libraries,
    scan_s2p_directory,
)
from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import run_tuning_joint  # noqa: E402
from rfmatch_core import __version__ as core_version  # noqa: E402


DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
RELATIVE_INPUT = Path(
    "4 - Multiantenna system/4.1 Multiantenna system at different bands/3_antennas.s3p"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/benchmarks/optenni-product-joint-time-budget.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(tutorial_root: Path, component_root: Path, timeout_seconds: float) -> dict:
    library = merge_component_libraries(
        scan_s2p_directory(str(
            component_root / "Inductors/Coilcraft Inductors 0402hp"
        )),
        scan_s2p_directory(str(
            component_root / "Capacitors/Murata Capacitors gqm18"
        )),
    )
    input_path = tutorial_root / RELATIVE_INPUT
    specs = [
        {"port_index": 0, "bands_mhz": [[2500, 2690]], "max_components": 2},
        {"port_index": 1, "bands_mhz": [[1920, 2170]], "max_components": 2},
        {"port_index": 2, "bands_mhz": [[1215, 1300]], "max_components": 2},
    ]
    started = time.perf_counter()
    candidates = run_tuning_joint(
        load_touchstone_file(input_path),
        library,
        specs,
        objective="balanced",
        beam_width=10,
        timeout_seconds=timeout_seconds,
        num_band_points=3,
    )
    elapsed = time.perf_counter() - started
    best = candidates[0]
    diagnostics = best.search_diagnostics
    return {
        "schema_version": 1,
        "case": "product joint measured-S2P soft deadline",
        "software": {"rfmatch_core_version": core_version},
        "input": {
            "relative_path": str(RELATIVE_INPUT),
            "sha256": _sha256(input_path),
            "raw_catalog_size": {
                "inductors": len(library.inductors),
                "capacitors": len(library.capacitors),
            },
            "maximum_components_by_port": {"0": 2, "1": 2, "2": 2},
            "frequency_points_per_band": 3,
            "beam_width": 10,
            "timeout_seconds": timeout_seconds,
        },
        "result": {
            "elapsed_seconds": elapsed,
            "candidate_count": len(candidates),
            "score_db": best.system_score,
            "topology_code": diagnostics["topology_code"],
            "components_by_port": {
                str(port): len(choices)
                for port, choices in best.component_choices.items()
            },
            "physical_evaluations": diagnostics["physical_evaluations"],
            "component_models_loaded": diagnostics["component_models_loaded"],
            "search_profile": diagnostics["search_profile"],
            "search_truncated": diagnostics["search_truncated"],
            "termination_reason": diagnostics["termination_reason"],
            "maximum_power_balance_error": best.maximum_power_balance_error,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.tutorial_root.resolve(), args.component_root.resolve(),
        args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **result["result"]}, indent=2))


if __name__ == "__main__":
    main()
