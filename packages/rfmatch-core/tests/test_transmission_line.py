import numpy as np
import pytest

from rfmatch_core import (
    Branch,
    CircuitTopology,
    TransmissionLineModel,
    TransmissionLineStubModel,
    evaluate_circuit,
)


def test_matched_lossless_line_has_analytic_phase_and_zero_reflection():
    line = TransmissionLineModel("TL", 50.0, 73.0, 2.0e9)
    scattering = line.s_parameters(2.0e9)
    assert scattering[0, 0] == pytest.approx(0j, abs=1e-14)
    assert scattering[1, 1] == pytest.approx(0j, abs=1e-14)
    assert scattering[1, 0] == pytest.approx(np.exp(-1j * np.deg2rad(73.0)), abs=1e-14)
    assert scattering[0, 1] == pytest.approx(scattering[1, 0], abs=1e-14)


def test_matched_lossy_line_accounts_for_dissipated_and_delivered_power():
    frequency = 1.0e9
    line = TransmissionLineModel("lossy", 50.0, 45.0, frequency, attenuation_db=1.2)
    topology = CircuitTopology(
        external_nodes=("input",),
        dut_nodes=("dut",),
        branches=(Branch("line", "input", "dut", line),),
    )
    result = evaluate_circuit(np.asarray([[0j]]), topology, frequency)
    delivered = 10 ** (-1.2 / 10.0)
    assert abs(result.s_parameters[0, 0]) < 1e-12
    assert result.dut_absorbed_power[0] == pytest.approx(delivered, abs=1e-12)
    assert result.component_loss[0, 0] == pytest.approx(1.0 - delivered, abs=1e-12)
    assert result.power_balance_error[0] == pytest.approx(0.0, abs=1e-12)


def test_quarter_wave_transform_matches_closed_form_input_impedance():
    frequency = 1.0e9
    line = TransmissionLineModel("quarter", 75.0, 90.0, frequency)
    load_ohm = 100.0
    dut_gamma = (load_ohm - 50.0) / (load_ohm + 50.0)
    topology = CircuitTopology(
        external_nodes=("input",), dut_nodes=("dut",),
        branches=(Branch("line", "input", "dut", line),),
    )
    result = evaluate_circuit(np.asarray([[dut_gamma + 0j]]), topology, frequency)
    transformed = 75.0**2 / load_ohm
    expected_gamma = (transformed - 50.0) / (transformed + 50.0)
    assert result.s_parameters[0, 0] == pytest.approx(expected_gamma + 0j, abs=1e-12)
    assert result.power_balance_error[0] == pytest.approx(0.0, abs=1e-12)


def test_open_and_short_stubs_follow_analytic_45_degree_admittance():
    frequency = 1.0e9
    line = TransmissionLineModel("stub", 50.0, 45.0, frequency)
    opened = TransmissionLineStubModel(line, "open")
    shorted = TransmissionLineStubModel(line, "short")
    assert opened.input_admittance(frequency) == pytest.approx(1j / 50.0, abs=1e-14)
    assert shorted.input_admittance(frequency) == pytest.approx(-1j / 50.0, abs=1e-14)


def test_line_can_be_exported_as_a_general_s2p_block():
    frequencies = np.asarray([0.8e9, 1.0e9, 1.2e9])
    line = TransmissionLineModel("layout", 63.0, 38.0, 1.0e9, attenuation_db=0.4)
    model = line.as_s2p_model(frequencies)
    for index, frequency in enumerate(frequencies):
        assert model.s_parameters[index] == pytest.approx(line.s_parameters(frequency), abs=1e-14)
