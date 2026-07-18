"""Generate the manufacturable microstrip geometry and loss baseline."""

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
    Band,
    Branch,
    CircuitTopology,
    LineSearchConfig,
    MicrostripDesignRules,
    MicrostripLineModel,
    PCBSubstrate,
    Problem,
    TransmissionLineOptimizer,
    evaluate_circuit,
    microstrip_properties,
    solve_microstrip_width,
)


REFERENCE = {
    # Independent scikit-rf MLine 0.24.1, tand=0 for real-valued H/J + K/J
    # electrical parameters. Geometry: er=4.5, h=1.6 mm, t=35 um, w=3 mm.
    1.0e9: (49.642641716079446, 3.3842599274103184),
    2.45e9: (49.69080507005367, 3.4232860265568927),
    10.0e9: (52.411119324261804, 3.6885057112497623),
}


def run() -> dict:
    substrate = PCBSubstrate(
        "FR-4 engineering model", 4.5, 1.6e-3,
        loss_tangent=0.02, copper_thickness_m=35e-6,
        copper_resistivity_ohm_m=1.68e-8,
        copper_roughness_rms_m=0.15e-6,
    )
    cross_validation = []
    for frequency, (reference_z0, reference_ee) in REFERENCE.items():
        properties = microstrip_properties(substrate, 3.0e-3, frequency)
        cross_validation.append({
            "frequency_hz": frequency,
            "characteristic_impedance_ohm": properties.characteristic_impedance_ohm,
            "reference_characteristic_impedance_ohm": reference_z0,
            "impedance_error_ohm": abs(properties.characteristic_impedance_ohm - reference_z0),
            "effective_permittivity": properties.effective_permittivity,
            "reference_effective_permittivity": reference_ee,
            "effective_permittivity_error": abs(properties.effective_permittivity - reference_ee),
        })

    reference_hz = 2.45e9
    width = solve_microstrip_width(substrate, 50.0, reference_hz, 0.1e-3, 10e-3)
    line = MicrostripLineModel.from_electrical_design(
        "quarter_wave", substrate, 50.0, 90.0, reference_hz,
        0.1e-3, 10e-3,
    )
    properties = line.properties_at(reference_hz)
    evaluation = evaluate_circuit(
        np.asarray([[0j]]),
        CircuitTopology(
            external_nodes=("input",), dut_nodes=("dut",),
            branches=(Branch("line", "input", "dut", line),),
        ),
        reference_hz,
    )

    load_ohm = 100.0
    load_gamma = (load_ohm - 50.0) / (load_ohm + 50.0)
    synthesis = TransmissionLineOptimizer(
        Problem(
            np.asarray([1.0e9]), np.asarray([[[load_gamma + 0j]]]),
            {0: (Band(1.0e9, 1.0e9),)}, 50.0,
        ),
        0, 1.0e9,
        config=LineSearchConfig(
            characteristic_impedance_min_ohm=60.0,
            characteristic_impedance_max_ohm=80.0,
            electrical_length_min_deg=70.0,
            electrical_length_max_deg=110.0,
            topologies=("through_line",), restarts=8, iterations=40, seed=11,
            microstrip_rules=MicrostripDesignRules(substrate, 0.1e-3, 10e-3),
        ),
    ).optimize()
    synthesized = synthesis.best
    synthesized_component = synthesized.components()[0]
    synthesized_reflection = abs(synthesized.metrics["s_parameters"][0, 0, 0])
    synthesized_efficiency = float(synthesized.metrics["total_efficiency"][0, 0])

    narrow = microstrip_properties(substrate, width * 0.9, reference_hz)
    wide = microstrip_properties(substrate, width * 1.1, reference_hz)
    low_er = microstrip_properties(
        PCBSubstrate("low-er", 4.3, 1.6e-3, 0.02, 35e-6), width, reference_hz
    )
    high_er = microstrip_properties(
        PCBSubstrate("high-er", 4.7, 1.6e-3, 0.02, 35e-6), width, reference_hz
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "case": "manufacturable microstrip geometry, dispersion, and loss baseline",
        "models": {
            "quasi_static": "Hammerstad-Jensen 1980",
            "dispersion": "Kirschning-Jansen 1982",
            "conductor_loss": "Wheeler incremental inductance with roughness correction",
            "independent_reference": "scikit-rf MLine 0.24.1",
            "reference_source": "https://scikit-rf.readthedocs.io/en/latest/api/media/generated/skrf.media.mline.MLine.html",
        },
        "validity_scope": {
            "included": [
                "homogeneous isotropic substrate", "finite copper thickness",
                "microstrip dispersion", "skin-effect conductor loss",
                "RMS roughness correction", "dielectric loss tangent",
            ],
            "excluded": [
                "solder mask", "side grounds", "anisotropic glass weave",
                "bends/tees/launches/vias", "radiation loss", "open-end fringing extension",
            ],
        },
        "substrate": {
            "relative_permittivity": substrate.relative_permittivity,
            "height_m": substrate.height_m,
            "loss_tangent": substrate.loss_tangent,
            "copper_thickness_m": substrate.copper_thickness_m,
            "copper_roughness_rms_m": substrate.copper_roughness_rms_m,
        },
        "cross_validation": cross_validation,
        "fifty_ohm_quarter_wave": {
            "frequency_hz": reference_hz,
            "width_m": width,
            "length_m": line.length_m,
            "characteristic_impedance_ohm": properties.characteristic_impedance_ohm,
            "effective_permittivity": properties.effective_permittivity,
            "conductor_loss_db": 20.0 / np.log(10.0) * properties.conductor_attenuation_np_per_m * line.length_m,
            "dielectric_loss_db": 20.0 / np.log(10.0) * properties.dielectric_attenuation_np_per_m * line.length_m,
            "total_loss_db": properties.attenuation_db_per_m * line.length_m,
            "delivered_power": float(evaluation.dut_absorbed_power[0]),
            "component_loss": float(evaluation.component_loss[0, 0]),
            "power_balance_error": float(abs(evaluation.power_balance_error[0])),
        },
        "automatic_synthesis": {
            "topology": synthesized.topology,
            "target_load_ohm": load_ohm,
            "target_transformer_impedance_ohm": float(np.sqrt(50.0 * load_ohm)),
            "recovered_impedance_ohm": synthesized.characteristic_impedance_ohm,
            "recovered_electrical_length_deg": synthesized.line_length_deg,
            "width_m": synthesized_component["width_m"],
            "length_m": synthesized_component["length_m"],
            "total_loss_db": synthesized_component["attenuation_db"],
            "reflection_magnitude": float(synthesized_reflection),
            "total_efficiency": synthesized_efficiency,
            "evaluations": synthesis.evaluations,
            "maximum_power_balance_error": synthesized.metrics["maximum_power_balance_error"],
        },
        "sensitivity": {
            "minus_10pct_width_impedance_ohm": narrow.characteristic_impedance_ohm,
            "nominal_width_impedance_ohm": properties.characteristic_impedance_ohm,
            "plus_10pct_width_impedance_ohm": wide.characteristic_impedance_ohm,
            "er_4p3_impedance_ohm": low_er.characteristic_impedance_ohm,
            "er_4p7_impedance_ohm": high_er.characteristic_impedance_ohm,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "microstrip-geometry-baseline.json",
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
