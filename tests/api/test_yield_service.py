from types import SimpleNamespace

import numpy as np

from rfmatch_core import (
    Band,
    FrequencyConfiguration,
    MDIFState,
    S2PModel,
    SwitchTunableProblem,
    preload_switch_state,
)
from engine.touchstone import parse_touchstone
from engine.tuning_service import TuningResult, run_tuning_yield_analysis


class MeasuredInductor:
    part_number = "L5N6_TEST"
    component_type = "inductor"
    nominal_value = 5.6
    nominal_unit = "nH"
    tolerance_pct = 10.0

    def get_s_matrix_at_freq(self, frequency_hz):
        impedance = 0.4 + 1j * 2 * np.pi * frequency_hz * self.nominal_value * 1e-9
        denominator = 100.0 + impedance
        reflection = impedance / denominator
        transmission = 100.0 / denominator
        return np.array([[reflection, transmission], [transmission, reflection]])


def test_product_yield_service_ranks_and_persists_deterministic_analysis():
    dut = parse_touchstone(
        "# GHZ S RI R 50\n"
        "0.9 0.3 0.1\n1.0 0.35 0.05\n1.1 0.4 0.0\n",
        "antenna.s1p",
    )
    measured_component = MeasuredInductor()
    measured_component.tempco_ppm_per_c = 250.0
    measured_component.systematic_bias_pct = -0.25
    measured_component.environment_metadata = {
        "evidence_level": "laboratory_measurement",
        "source_document": "LAB-RF-2026-014",
    }
    choice = SimpleNamespace(
        component=measured_component, connection_type="series", port=0,
    )
    supported = TuningResult(
        port_indices=[0], mode="single", system_score=0.8,
        component_choices={0: [choice]}, search_diagnostics={},
    )
    unsupported = TuningResult(
        port_indices=[0], mode="single", system_score=0.7,
        component_choices={}, search_diagnostics={},
    )
    request = {
        "ports": [{
            "port_index": 0, "enabled": True,
            "bands_mhz": [[900, 1100]],
        }]
    }
    first = run_tuning_yield_analysis(
        dut, object(), [supported, unsupported], request,
        samples=40, seed=17, minimum_total_efficiency=0.2,
        minimum_average_total_efficiency=0.25,
        minimum_return_loss_db=1.0,
        batch_correlation=0.6,
        temperature_min_c=-40.0,
        temperature_max_c=85.0,
        inductor_tempco_ppm_per_c=120.0,
        inductor_bias_pct=1.5,
    )
    second = run_tuning_yield_analysis(
        dut, object(), [supported, unsupported], request,
        samples=40, seed=17, minimum_total_efficiency=0.2,
        minimum_average_total_efficiency=0.25,
        minimum_return_loss_db=1.0,
        batch_correlation=0.6,
        temperature_min_c=-40.0,
        temperature_max_c=85.0,
        inductor_tempco_ppm_per_c=120.0,
        inductor_bias_pct=1.5,
    )
    assert first == second
    assert first["frequency_points"] == 3
    assert first["criteria"]["minimum_average_total_efficiency"] == 0.25
    assert first["variation_model"]["batch_correlation"] == 0.6
    assert first["variation_model"]["temperature_min_c"] == -40.0
    assert first["variation_model"]["inductor_bias_pct"] == 1.5
    assert first["ranked_candidates"][0]["solution_index"] == 0
    assert first["ranked_candidates"][0]["component_tolerances"] == [{
        "part_number": "L5N6_TEST",
        "tolerance_pct": 10.0,
        "source": "component_metadata",
    }]
    assert first["ranked_candidates"][0]["component_environment"] == [{
        "position": "component_1",
        "part_number": "L5N6_TEST",
        "kind": "L",
        "tempco_ppm_per_c": 250.0,
        "tempco_source": "laboratory_measurement",
        "systematic_bias_pct": -0.25,
        "bias_source": "laboratory_measurement",
    }]
    assert first["unsupported_candidates"][0]["solution_index"] == 1
    assert supported.search_diagnostics["yield_analysis"]["seed"] == 17
    assert "temperature_c" in supported.search_diagnostics["yield_analysis"]["worst_sample"]


def test_product_yield_service_dispatches_joint_switch_context():
    dut = parse_touchstone(
        "# GHZ S RI R 50\n"
        "0.9 0.4 0.1\n1.0 0.35 0.05\n1.1 0.3 0.0\n",
        "switch_antenna.s1p",
    )
    frequencies = np.asarray(dut.frequencies)
    dut_s11 = np.asarray([dut.get_s_matrix(index)[0, 0] for index in range(3)])
    switch_s = np.zeros((3, 2, 2), dtype=complex)
    switch_s[:, 0, 1] = 1.0
    switch_s[:, 1, 0] = 1.0
    state = MDIFState("state", "through", None, frequencies, switch_s)
    problem = SwitchTunableProblem(
        frequencies,
        dut_s11,
        (
            FrequencyConfiguration("low", {0: (Band(0.9e9, 1.0e9),)}),
            FrequencyConfiguration("high", {0: (Band(1.0e9, 1.1e9),)}),
        ),
    )
    omega = 2 * np.pi * frequencies
    impedance = 0.2 + 1.0 / (1j * omega * 1.2e-12)
    denominator = 100.0 + impedance
    reflection = impedance / denominator
    transmission = 100.0 / denominator
    matrices = np.empty((3, 2, 2), dtype=complex)
    matrices[:, 0, 0] = matrices[:, 1, 1] = reflection
    matrices[:, 0, 1] = matrices[:, 1, 0] = transmission
    branch = S2PModel(
        "C1", frequencies, matrices, 50.0, 0.1, "C", 1.2e-12
    )
    candidate = TuningResult(
        port_indices=[0],
        mode="switch",
        system_score=0.5,
        search_diagnostics={},
    )
    candidate.yield_context = {
        "problem": problem,
        "loaded_states": {
            "through": preload_switch_state(frequencies, dut_s11, state)
        },
        "branch_models": (branch,),
        "input_elements": (),
        "state_by_configuration": {"low": "through", "high": "through"},
        "component_tolerances": [{
            "position": "branch_1", "part_number": "C1",
            "tolerance_pct": 10.0, "source": "component_metadata",
        }],
    }
    result = run_tuning_yield_analysis(
        dut,
        object(),
        [candidate],
        {"ports": [{"port_index": 0, "enabled": True, "bands_mhz": [[900, 1100]]}]},
        samples=30,
        seed=29,
        minimum_total_efficiency=0.2,
        minimum_return_loss_db=1.0,
    )
    analysis = result["ranked_candidates"][0]
    assert analysis["analysis_scope"] == "joint_switch_configurations"
    assert set(analysis["configuration_yield_fraction"]) == {"low", "high"}
    assert candidate.to_dict().get("yield_context") is None
