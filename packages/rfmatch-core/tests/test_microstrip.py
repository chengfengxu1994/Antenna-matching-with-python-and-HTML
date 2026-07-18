import numpy as np
import pytest

from rfmatch_core import (
    Branch,
    CircuitTopology,
    MicrostripLineModel,
    MicrostripStubModel,
    PCBSubstrate,
    evaluate_circuit,
    microstrip_properties,
    microstrip_quasi_static,
    solve_microstrip_width,
)


@pytest.fixture
def fr4():
    return PCBSubstrate(
        "FR-4 engineering model", 4.5, 1.6e-3,
        loss_tangent=0.02,
        copper_thickness_m=35e-6,
        copper_resistivity_ohm_m=1.68e-8,
        copper_roughness_rms_m=0.15e-6,
    )


@pytest.mark.parametrize(
    ("frequency_hz", "expected_z0", "expected_effective_permittivity"),
    [
        (1.0e9, 49.642641716079446, 3.3842599274103184),
        (2.45e9, 49.69080507005367, 3.4232860265568927),
        (10.0e9, 52.411119324261804, 3.6885057112497623),
    ],
)
def test_hammerstad_kirschning_matches_independent_skrf_qucs_reference(
    fr4, frequency_hz, expected_z0, expected_effective_permittivity,
):
    # Golden values were generated with scikit-rf MLine using a real-valued
    # er, Hammerstad/Jensen quasi-static and Kirschning/Jansen dispersion.
    properties = microstrip_properties(fr4, 3.0e-3, frequency_hz)
    assert properties.characteristic_impedance_ohm == pytest.approx(expected_z0, abs=1e-12)
    assert properties.effective_permittivity == pytest.approx(expected_effective_permittivity, abs=1e-12)


def test_finite_copper_thickness_increases_effective_width(fr4):
    _, _, effective_width = microstrip_quasi_static(fr4, 3.0e-3)
    zero_copper = PCBSubstrate("zero", 4.5, 1.6e-3, copper_thickness_m=0.0)
    _, _, zero_effective_width = microstrip_quasi_static(zero_copper, 3.0e-3)
    assert effective_width > 3.0e-3
    assert zero_effective_width == pytest.approx(3.0e-3)


def test_width_inverse_and_electrical_length_are_manufacturable(fr4):
    width = solve_microstrip_width(fr4, 50.0, 2.45e9, 0.1e-3, 10.0e-3)
    line = MicrostripLineModel.from_electrical_design(
        "quarter_wave", fr4, 50.0, 90.0, 2.45e9, 0.1e-3, 10.0e-3
    )
    assert width == pytest.approx(2.9686013930477198e-3, abs=1e-14)
    assert line.width_m == pytest.approx(width, abs=1e-15)
    assert line.length_m == pytest.approx(16.542323530424584e-3, abs=1e-14)
    assert np.rad2deg(line.propagation_length(2.45e9).imag) == pytest.approx(90.0, abs=1e-10)


def test_lossy_matched_microstrip_closes_physical_power_balance(fr4):
    frequency = 2.45e9
    line = MicrostripLineModel.from_electrical_design(
        "matched", fr4, 50.0, 90.0, frequency, 0.1e-3, 10.0e-3
    )
    topology = CircuitTopology(
        external_nodes=("input",), dut_nodes=("dut",),
        branches=(Branch("microstrip", "input", "dut", line),),
    )
    result = evaluate_circuit(np.asarray([[0j]]), topology, frequency)
    insertion_loss_db = line.properties_at(frequency).attenuation_db_per_m * line.length_m
    expected_delivered = 10 ** (-insertion_loss_db / 10.0)
    assert abs(result.s_parameters[0, 0]) < 1e-9
    assert result.dut_absorbed_power[0] == pytest.approx(expected_delivered, abs=1e-11)
    assert result.component_loss[0, 0] == pytest.approx(1.0 - expected_delivered, abs=1e-11)
    assert result.power_balance_error[0] == pytest.approx(0.0, abs=1e-12)


def test_microstrip_stub_uses_frequency_dependent_line_properties(fr4):
    line = MicrostripLineModel.from_electrical_design(
        "stub", fr4, 50.0, 45.0, 2.45e9, 0.1e-3, 10.0e-3
    )
    opened = MicrostripStubModel(line, "open")
    shorted = MicrostripStubModel(line, "short")
    assert opened.input_admittance(2.45e9).imag > 0
    assert shorted.input_admittance(2.45e9).imag < 0
    assert opened.input_admittance(10.0e9) != opened.input_admittance(2.45e9)


def test_width_solver_rejects_unmanufacturable_target(fr4):
    with pytest.raises(ValueError, match="outside the manufacturable range"):
        solve_microstrip_width(fr4, 250.0, 2.45e9, 0.2e-3, 5.0e-3)
