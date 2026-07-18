"""Generate a deterministic RI/MA/DB and non-50-ohm Touchstone baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import (  # noqa: E402
    Band, Candidate, Element, LumpedLossModel, Objective, Problem,
    parse_touchstone_text, renormalize_s_parameters,
)
from rfmatch_core.evaluator import evaluate_lumped_physical  # noqa: E402


def _serialize(matrices: np.ndarray, fmt: str) -> str:
    rows = [f"# GHZ S {fmt} R 75"]
    for frequency, matrix in zip((1.0, 2.0, 3.0), matrices):
        fields = [f"{frequency:g}"]
        for source in range(matrix.shape[1]):
            for destination in range(matrix.shape[0]):
                value = matrix[destination, source]
                if fmt == "RI":
                    pair = value.real, value.imag
                elif fmt == "MA":
                    pair = abs(value), np.rad2deg(np.angle(value))
                else:
                    pair = 20.0 * np.log10(abs(value)), np.rad2deg(np.angle(value))
                fields.extend(f"{item:.15g}" for item in pair)
        rows.append(" ".join(fields) + " ! numeric comment 12345")
    return "\n".join(rows)


def run() -> dict:
    matrices = np.asarray([
        [[0.30 + 0.08j, 0.12 - 0.02j], [0.09 + 0.01j, 0.24 - 0.05j]],
        [[0.28 + 0.07j, 0.11 - 0.03j], [0.08 + 0.02j, 0.22 - 0.04j]],
        [[0.26 + 0.06j, 0.10 - 0.02j], [0.07 + 0.01j, 0.20 - 0.03j]],
    ], dtype=complex)
    parsed = {
        fmt: parse_touchstone_text(_serialize(matrices, fmt), f"format-{fmt}.s2p")
        for fmt in ("RI", "MA", "DB")
    }
    evaluated = {}
    for fmt, data in parsed.items():
        problem = Problem(
            data.frequencies_hz,
            data.s_parameters,
            {0: (Band(1e9, 3e9),), 1: (Band(1e9, 3e9),)},
            data.z0,
        )
        result = evaluate_lumped_physical(
            problem,
            Candidate([Element("series", "L", 0, 2.2e-9)]),
            Objective(),
            LumpedLossModel(inductor_q=40.0, capacitor_esr=0.2),
        )
        evaluated[fmt] = {
            "score_db": float(result.score_db),
            "s_parameters": np.asarray(result.metrics["s_parameters"]),
            "total_efficiency": np.asarray(result.metrics["total_efficiency"]),
            "maximum_power_balance_error": float(result.metrics["maximum_power_balance_error"]),
        }

    reference = evaluated["RI"]
    comparisons = {}
    for fmt in ("MA", "DB"):
        comparisons[fmt] = {
            "input_s_maximum_complex_error": float(np.max(np.abs(
                parsed[fmt].s_parameters - parsed["RI"].s_parameters
            ))),
            "matched_s_maximum_complex_error": float(np.max(np.abs(
                evaluated[fmt]["s_parameters"] - reference["s_parameters"]
            ))),
            "total_efficiency_maximum_absolute_error": float(np.max(np.abs(
                evaluated[fmt]["total_efficiency"] - reference["total_efficiency"]
            ))),
            "score_db_absolute_error": float(abs(
                evaluated[fmt]["score_db"] - reference["score_db"]
            )),
        }
    per_port_z0 = np.asarray([50.0, 90.0])
    per_port_s = renormalize_s_parameters(
        parsed["RI"].s_parameters, 75.0, per_port_z0
    )
    round_trip = renormalize_s_parameters(per_port_s, per_port_z0, 75.0)
    per_port_result = evaluate_lumped_physical(
        Problem(
            parsed["RI"].frequencies_hz, per_port_s,
            {0: (Band(1e9, 3e9),), 1: (Band(1e9, 3e9),)}, per_port_z0,
        ),
        Candidate([Element("series", "L", 1, 2.2e-9)]),
        Objective(), LumpedLossModel(inductor_q=40.0, capacitor_esr=0.2),
    )
    incorrectly_collapsed = evaluate_lumped_physical(
        Problem(
            parsed["RI"].frequencies_hz, per_port_s,
            {0: (Band(1e9, 3e9),), 1: (Band(1e9, 3e9),)}, 75.0,
        ),
        Candidate([Element("series", "L", 1, 2.2e-9)]),
        Objective(), LumpedLossModel(inductor_q=40.0, capacitor_esr=0.2),
    )
    nonreciprocal_three_port = np.asarray([
        [0.11 + 0.01j, 0.12 + 0.02j, 0.13 + 0.03j],
        [0.21 - 0.01j, 0.22 - 0.02j, 0.23 - 0.03j],
        [0.31 + 0.04j, 0.32 + 0.05j, 0.33 + 0.06j],
    ])
    row_fields = []
    for value in nonreciprocal_three_port.flat:
        row_fields.extend((f"{value.real:g}", f"{value.imag:g}"))
    parsed_three_port = parse_touchstone_text(
        "[Version] 2.0\n# GHZ S RI R 50\n[Number of Ports] 3\n"
        "[Number of Frequencies] 1\n[Reference] 50\n75 90\n"
        "[Matrix Format] Full\n[Network Data]\n1 " + " ".join(row_fields) + "\n[End]",
        "row-major.ts",
    )
    two_port_12_21 = parse_touchstone_text(
        "[Version] 2.0\n# GHZ S RI R 50\n[Number of Ports] 2\n"
        "[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n"
        "[Network Data]\n1 0.1 0 0.12 0.01 0.21 -0.02 0.4 0\n[End]",
        "two-port-order.ts",
    )
    dc_data = parse_touchstone_text(
        "# HZ S RI R 50\n0 0.1 0\n10000000 0.2 0",
        "dc.s1p",
    )
    return {
        "schema_version": 2,
        "case": "two-port RI/MA/DB equivalence at 75 ohms",
        "reference_impedance_ohm": 75.0,
        "frequency_points": 3,
        "formats": ["RI", "MA", "DB"],
        "network": {"topology": "series-L on port 1", "value_h": 2.2e-9},
        "comparisons_against_ri": comparisons,
        "maximum_power_balance_error": max(
            item["maximum_power_balance_error"] for item in evaluated.values()
        ),
        "per_port_reference": {
            "impedances_ohm": per_port_z0.tolist(),
            "renormalization_round_trip_maximum_complex_error": float(
                np.max(np.abs(round_trip - parsed["RI"].s_parameters))
            ),
            "physical_power_balance_error": float(
                per_port_result.metrics["maximum_power_balance_error"]
            ),
            "score_db": float(per_port_result.score_db),
            "incorrect_scalar_score_db": float(incorrectly_collapsed.score_db),
            "score_delta_from_incorrect_scalar_db": float(
                per_port_result.score_db - incorrectly_collapsed.score_db
            ),
        },
        "touchstone_2_semantics": {
            "three_port_row_major_maximum_complex_error": float(np.max(np.abs(
                parsed_three_port.s_parameters[0] - nonreciprocal_three_port
            ))),
            "multiline_reference_impedances_ohm": np.asarray(parsed_three_port.z0).tolist(),
            "two_port_12_21_s12": [
                float(two_port_12_21.s_parameters[0, 0, 1].real),
                float(two_port_12_21.s_parameters[0, 0, 1].imag),
            ],
            "two_port_12_21_s21": [
                float(two_port_12_21.s_parameters[0, 1, 0].real),
                float(two_port_12_21.s_parameters[0, 1, 0].imag),
            ],
            "dc_frequency_hz": float(dc_data.frequencies_hz[0]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "touchstone-format-baseline.json",
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
