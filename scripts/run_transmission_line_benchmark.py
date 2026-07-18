"""Generate an analytic regression baseline for physical lines and stubs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import (  # noqa: E402
    Branch,
    CircuitTopology,
    Band,
    LineSearchConfig,
    Problem,
    TransmissionLineOptimizer,
    TransmissionLineModel,
    TransmissionLineStubModel,
    evaluate_circuit,
)


def run() -> dict:
    reference_hz = 1.0e9

    lossless = TransmissionLineModel("lossless", 50.0, 73.0, reference_hz)
    lossless_s = lossless.s_parameters(reference_hz)
    expected_phase = np.exp(-1j * np.deg2rad(73.0))

    lossy = TransmissionLineModel(
        "lossy", 50.0, 45.0, reference_hz,
        attenuation_db=1.0, loss_frequency_exponent=0.5,
    )
    lossy_topology = CircuitTopology(
        external_nodes=("input",), dut_nodes=("dut",),
        branches=(Branch("line", "input", "dut", lossy),),
    )
    lossy_points = []
    maximum_balance_error = 0.0
    for frequency in (0.8e9, 1.0e9, 1.2e9):
        result = evaluate_circuit(np.asarray([[0j]]), lossy_topology, frequency)
        expected_delivered = 10 ** (
            -(1.0 * (frequency / reference_hz) ** 0.5) / 10.0
        )
        balance_error = float(abs(result.power_balance_error[0]))
        maximum_balance_error = max(maximum_balance_error, balance_error)
        lossy_points.append({
            "frequency_hz": frequency,
            "delivered_power": float(result.dut_absorbed_power[0]),
            "expected_delivered_power": expected_delivered,
            "delivered_power_error": float(abs(result.dut_absorbed_power[0] - expected_delivered)),
            "component_loss": float(result.component_loss[0, 0]),
            "power_balance_error": balance_error,
        })

    quarter = TransmissionLineModel("quarter", 75.0, 90.0, reference_hz)
    load_ohm = 100.0
    load_gamma = (load_ohm - 50.0) / (load_ohm + 50.0)
    quarter_result = evaluate_circuit(
        np.asarray([[load_gamma + 0j]]),
        CircuitTopology(
            external_nodes=("input",), dut_nodes=("dut",),
            branches=(Branch("line", "input", "dut", quarter),),
        ),
        reference_hz,
    )
    transformed_ohm = 75.0**2 / load_ohm
    expected_gamma = (transformed_ohm - 50.0) / (transformed_ohm + 50.0)

    stub_line = TransmissionLineModel("stub", 50.0, 45.0, reference_hz)
    open_y = TransmissionLineStubModel(stub_line, "open").input_admittance(reference_hz)
    short_y = TransmissionLineStubModel(stub_line, "short").input_admittance(reference_hz)

    load_gamma = (load_ohm - 50.0) / (load_ohm + 50.0)
    synthesis = TransmissionLineOptimizer(
        Problem(
            np.asarray([reference_hz]),
            np.asarray([[[load_gamma + 0j]]]),
            {0: (Band(reference_hz, reference_hz),)},
            50.0,
        ),
        0,
        reference_hz,
        config=LineSearchConfig(
            characteristic_impedance_min_ohm=60.0,
            characteristic_impedance_max_ohm=80.0,
            electrical_length_min_deg=70.0,
            electrical_length_max_deg=110.0,
            topologies=("through_line",),
            restarts=8,
            iterations=40,
            seed=11,
        ),
    ).optimize()
    synthesized = synthesis.best
    synthesized_reflection = abs(synthesized.metrics["s_parameters"][0, 0, 0])

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "case": "rfmatch-core physical transmission-line analytic baseline",
        "reference_frequency_hz": reference_hz,
        "lossless_matched_line": {
            "characteristic_impedance_ohm": 50.0,
            "electrical_length_deg": 73.0,
            "maximum_reflection_magnitude": float(max(abs(lossless_s[0, 0]), abs(lossless_s[1, 1]))),
            "forward_complex_error": float(abs(lossless_s[1, 0] - expected_phase)),
        },
        "lossy_matched_line": {
            "attenuation_db_at_reference": 1.0,
            "loss_frequency_exponent": 0.5,
            "points": lossy_points,
            "maximum_power_balance_error": maximum_balance_error,
        },
        "quarter_wave_transform": {
            "characteristic_impedance_ohm": 75.0,
            "load_impedance_ohm": load_ohm,
            "expected_input_impedance_ohm": transformed_ohm,
            "reflection_complex_error": float(abs(quarter_result.s_parameters[0, 0] - expected_gamma)),
        },
        "shunt_stubs": {
            "characteristic_impedance_ohm": 50.0,
            "electrical_length_deg": 45.0,
            "open_admittance_siemens": [float(open_y.real), float(open_y.imag)],
            "short_admittance_siemens": [float(short_y.real), float(short_y.imag)],
            "maximum_analytic_error": float(max(abs(open_y - 0.02j), abs(short_y + 0.02j))),
        },
        "automatic_synthesis": {
            "case": "recover 100 ohm load with a quarter-wave transformer",
            "topology": synthesized.topology,
            "recovered_characteristic_impedance_ohm": synthesized.characteristic_impedance_ohm,
            "expected_characteristic_impedance_ohm": float(np.sqrt(50.0 * load_ohm)),
            "recovered_electrical_length_deg": synthesized.line_length_deg,
            "expected_electrical_length_deg": 90.0,
            "reflection_magnitude": float(synthesized_reflection),
            "evaluations": synthesis.evaluations,
            "stopped_reason": synthesis.stopped_reason,
            "maximum_power_balance_error": float(synthesized.metrics["maximum_power_balance_error"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "transmission-line-physical-baseline.json",
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
