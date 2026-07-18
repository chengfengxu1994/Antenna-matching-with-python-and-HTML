"""Recompute the saved Optenni 4.3 three-antenna tutorial winner exactly.

The reference values and BOM were read from the running Optenni Lab project
``multiantenna_project.opr``.  This script evaluates the same DUT and measured
component files with rfmatch_core and records the rounded UI deltas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import (  # noqa: E402
    __version__ as core_version,
    Band,
    ComponentSpec,
    MeasuredPlacement,
    Objective,
    Problem,
    component_sha256,
    evaluate_measured_candidate,
    load_touchstone,
    parse_optenni_opr,
)


RELATIVE_DIR = Path(
    "4 - Multiantenna system/4.1 Multiantenna system at different bands"
)
DEFAULT_TUTORIAL_ROOT = Path(os.environ.get(
    "OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
))
DEFAULT_COMPONENT_ROOT = Path(os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    str(Path.home() / "AppData/Roaming/Optenni/ComponentLibrary"),
))
DEFAULT_OUTPUT = ROOT / "artifacts/benchmarks/optenni-multiport-live-golden.json"
BANDS = {
    0: (2.50e9, 2.69e9),
    1: (1.92e9, 2.17e9),
    2: (1.215e9, 1.30e9),
}
OPTENNI_UI_EFFICIENCY_DB = {
    0: {"minimum": -1.7, "average": -1.1},
    1: {"minimum": -2.2, "average": -1.4},
    2: {"minimum": -2.1, "average": -1.6},
}
OPTENNI_UI_POWER_AT_1P24_GHZ_PORT3 = {
    "reflected": 0.154,
    "component_loss": 0.013,
    "coupling": 0.140,
    "radiated": 0.693,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(tutorial_root: Path, component_root: Path) -> dict:
    case_dir = tutorial_root / RELATIVE_DIR
    dut_path = case_dir / "3_antennas.s3p"
    project_path = case_dir / "multiantenna_project.opr"
    touchstone = load_touchstone(dut_path)
    project = parse_optenni_opr(project_path)
    winner = project["saved_winner"]
    if winner is None:
        raise ValueError("Optenni project contains no saved matched candidate")
    placements_list = []
    for item in winner["components"]:
        relative_tolerance = item["relative_tolerance_pct"]
        if relative_tolerance is not None:
            tolerance = relative_tolerance / 100.0
        else:
            tolerance = (item["absolute_tolerance"] or 0.0) / max(item["value"], 1e-15)
        component = ComponentSpec(
            item["part_number"],
            "C" if item["component_type"] == "capacitor" else "L",
            item["value_si"],
            tolerance,
            f"{item['manufacturer']} {item['series']}".strip(),
            component_root / item["library_subdirectory"] / item["model_filename"],
        )
        placements_list.append(MeasuredPlacement(
            item["connection"], item["port"], component
        ))
    placements = tuple(placements_list)
    bands = {
        item["port"]: (item["start_hz"], item["stop_hz"])
        for item in project["bands"]
    }
    problem = Problem(
        touchstone.frequencies_hz,
        touchstone.s_parameters,
        {port: (Band(*band),) for port, band in bands.items()},
        touchstone.z0,
    )
    candidate = evaluate_measured_candidate(
        problem, placements,
        Objective(port_average_weight=project["objective"]["alpha_total"]),
    )
    total_efficiency = np.asarray(candidate.metrics["total_efficiency"])
    s_parameters = np.asarray(candidate.metrics["s_parameters"])
    component_loss = np.asarray(candidate.metrics["component_loss"])
    dut_absorbed = np.asarray(candidate.metrics["dut_absorbed_power"])

    efficiency = {}
    for port, (start, stop) in bands.items():
        mask = (touchstone.frequencies_hz >= start) & (touchstone.frequencies_hz <= stop)
        values_db = 10.0 * np.log10(np.maximum(total_efficiency[mask, port], 1e-15))
        calculated = {
            "minimum": float(np.min(values_db)),
            "average": float(np.mean(values_db)),
        }
        reference = OPTENNI_UI_EFFICIENCY_DB[port]
        efficiency[str(port)] = {
            "band_hz": [start, stop],
            "optenni_ui_db": reference,
            "rfmatch_core_db": calculated,
            "difference_from_rounded_ui_db": {
                key: calculated[key] - reference[key] for key in calculated
            },
            "frequency_points": int(np.count_nonzero(mask)),
        }

    sample_index = int(np.argmin(np.abs(touchstone.frequencies_hz - 1.24e9)))
    driven_port = 2
    reflected = float(abs(s_parameters[sample_index, driven_port, driven_port]) ** 2)
    coupling = float(sum(
        abs(s_parameters[sample_index, destination, driven_port]) ** 2
        for destination in range(touchstone.s_parameters.shape[1])
        if destination != driven_port
    ))
    calculated_power = {
        "reflected": reflected,
        "component_loss": float(component_loss[sample_index, driven_port]),
        "coupling": coupling,
        "radiated": float(dut_absorbed[sample_index, driven_port]),
    }
    return {
        "schema_version": 1,
        "case": "Optenni Lab 4.3 multiantenna tutorial saved winner",
        "source": {
            "capture_date": "2026-07-15",
            "application": "Optenni Lab 4.3",
            "project_relative_path": str(RELATIVE_DIR / project_path.name),
            "project_sha256": _sha256(project_path),
            "dut_relative_path": str(RELATIVE_DIR / dut_path.name),
            "dut_sha256": _sha256(dut_path),
            "reference_note": (
                "BOM, topology and optimization bands were parsed directly from the saved OPR XML; "
                "rounded min/average efficiency and 1.24 GHz power balance were read from the running project."
            ),
            "opr_evidence": {
                "optenni_version": project["optenni_version"],
                "has_results": project["has_results"],
                "candidate_count": project["candidate_count"],
                "matched_candidate_count": project["matched_candidate_count"],
                "saved_winner_index": winner["index"],
                "embedded_input_rows": project["impedance_configuration"]["embedded_data_rows"],
                "embedded_input_sha256": project["impedance_configuration"]["embedded_data_sha256"],
            },
        },
        "software": {"rfmatch_core_version": core_version},
        "objective": {
            "alpha_in_band": 0.05,
            "alpha_total": 0.1,
            "radiation_efficiency": 1.0,
        },
        "bom": [
            {
                "port": item.port,
                "connection": item.connection,
                "kind": item.component.kind,
                "part_number": item.component.name,
                "value_si": item.component.value,
                "model_sha256": component_sha256(item.component),
            }
            for item in placements
        ],
        "topology_by_port": winner["topology_by_port"],
        "efficiency_comparison": efficiency,
        "power_balance_comparison": {
            "frequency_hz": float(touchstone.frequencies_hz[sample_index]),
            "driven_port": driven_port,
            "optenni_ui_linear": OPTENNI_UI_POWER_AT_1P24_GHZ_PORT3,
            "rfmatch_core_linear": calculated_power,
            "difference_from_rounded_ui_linear": {
                key: calculated_power[key] - OPTENNI_UI_POWER_AT_1P24_GHZ_PORT3[key]
                for key in calculated_power
            },
            "rfmatch_core_sum": float(sum(calculated_power.values())),
        },
        "maximum_power_balance_error": float(
            candidate.metrics["maximum_power_balance_error"]
        ),
        "score_db": float(candidate.score_db),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.tutorial_root.resolve(), args.component_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "efficiency_comparison": result["efficiency_comparison"],
        "power_balance_comparison": result["power_balance_comparison"],
        "maximum_power_balance_error": result["maximum_power_balance_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
