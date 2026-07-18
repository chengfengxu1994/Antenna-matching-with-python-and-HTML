"""Product-service replay of Optenni's radiation-efficiency tutorial.

The benchmark searches the complete installed Coilcraft 0402HP and Murata
GQM18 families twice: once with the official radiation-efficiency curve in the
objective and once using mismatch/component loss only.  Both returned physical
networks are then densely re-evaluated with the official efficiency curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from engine.efficiency_data import load_efficiency_file  # noqa: E402
from engine.touchstone import load_touchstone_file  # noqa: E402
from engine.tuning_service import (  # noqa: E402
    _core_s2p_from_library_component,
    _reference_impedance,
    run_tuning_single,
)
from run_product_saved_winner_constrained import _product_library  # noqa: E402
from rfmatch_core import (  # noqa: E402
    Band, Candidate, ModelPlacement, Objective, Problem,
    build_model_circuit_topology, evaluate_physical_problem,
)
from rfmatch_core.evaluator import score_sweep  # noqa: E402


RELATIVE_DIR = Path("9 - Radiation efficiency")
DUT_FILENAME = "Radiation_Efficiency_Tutorial.s1p"
EFFICIENCY_FILENAME = "radiation_efficiency.txt"
BANDS_MHZ = ((880.0, 960.0), (1710.0, 2155.0), (2300.0, 2400.0))
TOPOLOGY_CODE = "PCSLPC"
DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/optenni-product-radiation-efficiency.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _network_summary(result) -> dict:
    return {
        "score_db": result.system_score,
        "topology_code": result.search_diagnostics.get("topology_code"),
        "objective_weights": result.search_diagnostics.get("objective_weights"),
        "generic_synthesis_loss": result.search_diagnostics.get("generic_synthesis_loss"),
        "components": result.per_port[0].components,
        "search_average_total_efficiency_db": 10.0 * math.log10(
            max(result.avg_total_efficiency, 1e-15)
        ),
        "search_minimum_total_efficiency_db": 10.0 * math.log10(
            max(result.min_total_efficiency, 1e-15)
        ),
        "maximum_power_balance_error": result.maximum_power_balance_error,
        "search_diagnostics": {
            key: result.search_diagnostics.get(key)
            for key in (
                "search_profile", "active_frequency_points", "ideal_evaluations",
                "physical_evaluations", "component_models_loaded", "search_truncated",
                "termination_reason",
            )
        },
    }


def _dense_summary(dut, library, result, efficiency) -> dict:
    frequencies = np.asarray([
        frequency for frequency in dut.frequencies
        if any(start * 1e6 <= frequency <= stop * 1e6 for start, stop in BANDS_MHZ)
    ], dtype=float)
    problem = Problem(
        frequencies,
        np.asarray([dut.get_s_matrix_interpolated(float(frequency)) for frequency in frequencies]),
        {0: tuple(Band(start * 1e6, stop * 1e6) for start, stop in BANDS_MHZ)},
        _reference_impedance(dut),
        {0: efficiency.get_efficiency_array(frequencies)},
    )
    placements = []
    for choice in result.component_choices[0]:
        model = _core_s2p_from_library_component(choice.component, library, frequencies)
        placements.append(ModelPlacement(choice.connection_type, 0, model))
    physical = evaluate_physical_problem(
        problem, build_model_circuit_topology(1, placements)
    )
    scored = score_sweep(
        problem,
        Candidate([]),
        Objective(0.5, 0.1, 0.0),
        physical.s_parameters,
        physical.total_efficiency,
    )
    total = physical.total_efficiency[:, 0]
    return_loss = scored.metrics["return_loss_db"][:, 0]
    band_summaries = []
    active_total = []
    for start_mhz, stop_mhz in BANDS_MHZ:
        mask = (frequencies >= start_mhz * 1e6) & (frequencies <= stop_mhz * 1e6)
        values = total[mask]
        active_total.extend(values.tolist())
        band_summaries.append({
            "band_mhz": [start_mhz, stop_mhz],
            "frequency_points": int(np.count_nonzero(mask)),
            "minimum_total_efficiency_db": 10.0 * math.log10(max(float(np.min(values)), 1e-15)),
            "average_total_efficiency_db": float(np.mean(10.0 * np.log10(np.maximum(values, 1e-15)))),
            "minimum_return_loss_db": float(np.min(return_loss[mask])),
        })
    active = np.asarray(active_total)
    return {
        "basis": "rfmatch_core_physical_measured_s2p_with_radiation",
        "active_frequency_points": len(active),
        "objective_score_db": float(scored.score_db),
        "minimum_total_efficiency_db": 10.0 * math.log10(max(float(np.min(active)), 1e-15)),
        "average_total_efficiency_db": float(np.mean(10.0 * np.log10(np.maximum(active, 1e-15)))),
        "maximum_power_balance_error": float(np.max(np.abs(physical.power_balance_error))),
        "bands": band_summaries,
    }


def run(
    tutorial_root: Path,
    component_root: Path,
    *,
    timeout_seconds: float = 60.0,
    beam_width: int = 20,
    num_band_points: int = 7,
) -> dict:
    case_dir = tutorial_root / RELATIVE_DIR
    dut_path = case_dir / DUT_FILENAME
    efficiency_path = case_dir / EFFICIENCY_FILENAME
    dut = load_touchstone_file(dut_path)
    efficiency = load_efficiency_file(str(efficiency_path))
    library, catalog = _product_library(component_root)

    runs = {}
    for name, search_efficiency in (
        ("efficiency_aware", efficiency),
        ("mismatch_only", None),
    ):
        started = time.perf_counter()
        candidates = run_tuning_single(
            dut=dut,
            library=library,
            port_index=0,
            bands_mhz=[list(item) for item in BANDS_MHZ],
            max_components=3,
            allowed_topology_codes=[TOPOLOGY_CODE],
            objective="balanced",
            within_band_average_weight=0.5,
            across_band_average_weight=0.1,
            generic_synthesis_loss={
                "inductor_q": 50.0,
                "inductor_q_reference_hz": 1e9,
                "inductor_esr_ohm": 0.0,
                "capacitor_esr_ohm": 0.3,
            },
            beam_width=beam_width,
            timeout_seconds=timeout_seconds,
            search_profile_timeout_seconds=timeout_seconds,
            num_band_points=num_band_points,
            global_efficiency=search_efficiency,
        )
        if not candidates:
            raise RuntimeError(f"{name} search returned no physical candidate")
        best = candidates[min(candidates)]
        runs[name] = {
            "wall_seconds": time.perf_counter() - started,
            "network": _network_summary(best),
            "dense_official_efficiency_evaluation": _dense_summary(dut, library, best, efficiency),
        }

    aware_dense = runs["efficiency_aware"]["dense_official_efficiency_evaluation"]
    mismatch_dense = runs["mismatch_only"]["dense_official_efficiency_evaluation"]
    return {
        "schema_version": 1,
        "case": "Product full-catalog radiation-efficiency optimization",
        "reference_manifest": "benchmarks/optenni_exports/radiation_efficiency_tutorial_manifest.json",
        "input": {
            "dut_relative_path": str(RELATIVE_DIR / DUT_FILENAME),
            "dut_sha256": _sha256(dut_path),
            "efficiency_relative_path": str(RELATIVE_DIR / EFFICIENCY_FILENAME),
            "efficiency_sha256": _sha256(efficiency_path),
            "bands_mhz": [list(item) for item in BANDS_MHZ],
        },
        "catalog": catalog,
        "request": {
            "topology_code": TOPOLOGY_CODE,
            "maximum_components": 3,
            "within_band_average_weight": 0.5,
            "across_band_average_weight": 0.1,
            "generic_synthesis_loss": {
                "inductor_q": 50.0,
                "inductor_q_reference_hz": 1e9,
                "inductor_esr_ohm": 0.0,
                "capacitor_esr_ohm": 0.3,
                "scope": "continuous_topology_prior_only",
            },
            "timeout_seconds_per_run": timeout_seconds,
            "beam_width": beam_width,
            "num_band_points": num_band_points,
        },
        "runs": runs,
        "comparison": {
            "efficiency_aware_minus_mismatch_only_objective_db": (
                aware_dense["objective_score_db"] - mismatch_dense["objective_score_db"]
            ),
            "efficiency_aware_minus_mismatch_only_worst_case_db": (
                aware_dense["minimum_total_efficiency_db"]
                - mismatch_dense["minimum_total_efficiency_db"]
            ),
            "networks_differ": (
                runs["efficiency_aware"]["network"]["components"]
                != runs["mismatch_only"]["network"]["components"]
            ),
            "tutorial_published_improvement_db": 0.3,
            "comparison_note": "The tutorial uses generic lossy components; this product replay uses complete installed measured S2P families, so exact values are not expected to match.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--num-band-points", type=int, default=7)
    args = parser.parse_args()
    report = run(
        args.tutorial_root,
        args.component_root,
        timeout_seconds=args.timeout_seconds,
        beam_width=args.beam_width,
        num_band_points=args.num_band_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
