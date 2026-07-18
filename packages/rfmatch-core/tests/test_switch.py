from pathlib import Path

import numpy as np
import pytest

from rfmatch_core import InputModelPlacement, InputReactance, SeriesReactance, evaluate_loaded_switch_physical_power, evaluate_loaded_switch_power, evaluate_switched_matching, load_coilcraft_0402cs_catalog, load_component_model, load_mdif, load_murata_gjm15_catalog, load_touchstone, preload_switch_state
from rfmatch_core.network import z_to_s
from rfmatch_core.switch import reduce_switch_with_series_branches


def test_single_throw_reduction_matches_direct_series_impedance():
    z0 = 50.0
    # Ideal through two-port: terminating port 0 in gamma exposes that same
    # impedance at port 1.
    switch_s = np.array([[0, 1], [1, 0]], dtype=complex)
    load_z = 30 + 20j
    load_gamma = (load_z - z0) / (load_z + z0)
    branch_z = -15j
    fixed_z = 10j
    actual = reduce_switch_with_series_branches(
        switch_s,
        common_port=0,
        dut_gamma=load_gamma,
        branch_impedances=[branch_z],
        input_series_impedance=fixed_z,
        z0=z0,
    )
    expected_z = load_z + branch_z + fixed_z
    expected = (expected_z - z0) / (expected_z + z0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_tied_throws_use_full_coupled_admittance_matrix():
    # Start at an arbitrary positive two-port impedance matrix, convert it to
    # S, and embed it behind a perfectly matched/disconnected common port.
    throws_z = np.array([[60 + 10j, 5j], [5j, 70 - 5j]])
    throws_s = z_to_s(throws_z)
    switch_s = np.zeros((3, 3), dtype=complex)
    switch_s[1:, 1:] = throws_s
    branches = np.array([2j, -3j])
    expected_z = 1.0 / np.sum(np.linalg.inv(throws_z + np.diag(branches)))
    expected = (expected_z - 50) / (expected_z + 50)
    actual = reduce_switch_with_series_branches(
        switch_s,
        common_port=0,
        dut_gamma=0,
        branch_impedances=branches,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_official_sp2t_tutorial_topology_is_finite_and_state_dependent():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    dut = load_touchstone(root / "Switch_Tuner_Tutorial.s1p")
    switch = load_mdif(root / "tutorial_SP2T.mdif")
    frequencies = np.array([725e6, 826e6, 920e6, 2045e6])
    dut_s11 = np.interp(frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].real) + 1j * np.interp(
        frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].imag
    )
    branches = [SeriesReactance("C", 1.2e-12), SeriesReactance("C", 0.8e-12)]
    fixed = [SeriesReactance("L", 13e-9)]
    traces = {
        label: evaluate_switched_matching(frequencies, dut_s11, switch.state(label), branches, input_series_reactances=fixed)
        for label in ("all on", "RFC-RF1", "RFC-RF2")
    }
    assert all(np.all(np.isfinite(trace)) for trace in traces.values())
    assert not np.allclose(traces["all on"], traces["RFC-RF1"])
    assert not np.allclose(traces["RFC-RF1"], traces["RFC-RF2"])


def test_official_switch_tutorial_center_frequency_golden_values():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    dut = load_touchstone(root / "Switch_Tuner_Tutorial.s1p")
    switch = load_mdif(root / "tutorial_SP2T.mdif")
    frequencies = np.array([725e6, 2045e6, 826.5e6, 2045e6, 920e6, 2045e6])
    dut_s11 = np.interp(frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].real) + 1j * np.interp(
        frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].imag
    )
    branches = [SeriesReactance("C", 1.2e-12), SeriesReactance("C", 0.8e-12)]
    fixed = [SeriesReactance("L", 13e-9)]
    states = ("all on", "RFC-RF1", "RFC-RF2")
    actual = []
    for index, state in enumerate(states):
        pair = slice(index * 2, index * 2 + 2)
        gamma = evaluate_switched_matching(
            frequencies[pair], dut_s11[pair], switch.state(state), branches,
            input_series_reactances=fixed,
        )
        actual.extend(-20 * np.log10(np.abs(gamma)))
    np.testing.assert_allclose(
        actual,
        [7.218402147262476, 15.175565143402709, 8.087665962012657, 9.65415097166401, 10.113360876845991, 5.651871077727789],
        atol=1e-9,
    )


def test_official_switch_wave_power_separates_dut_and_closes_balance():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    dut = load_touchstone(root / "Switch_Tuner_Tutorial.s1p")
    switch = load_mdif(root / "tutorial_SP2T.mdif").state("all on")
    frequencies = np.array([725e6, 2045e6])
    dut_s11 = np.interp(frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].real) + 1j * np.interp(
        frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].imag
    )
    loaded = preload_switch_state(frequencies, dut_s11, switch)
    power = evaluate_loaded_switch_power(
        loaded,
        [SeriesReactance("C", 1.2e-12), SeriesReactance("C", 0.8e-12)],
        input_reactances=[InputReactance("series", "L", 13e-9)],
    )
    assert np.all(power.dut_absorbed_power > 0.8)
    assert np.max(np.abs(power.power_balance_error)) < 5e-6
    np.testing.assert_allclose(
        power.input_accepted_power,
        power.dut_absorbed_power + power.switch_loss + power.power_balance_error,
        atol=1e-12,
    )


def test_general_nodal_switch_solver_matches_fast_ideal_solver():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    dut = load_touchstone(root / "Switch_Tuner_Tutorial.s1p")
    state = load_mdif(root / "tutorial_SP2T.mdif").state("all on")
    frequencies = np.array([725e6, 2045e6])
    dut_s11 = np.interp(frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].real) + 1j * np.interp(
        frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].imag
    )
    loaded = preload_switch_state(frequencies, dut_s11, state)
    branches = [SeriesReactance("C", 1.2e-12), SeriesReactance("C", 0.8e-12)]
    fixed = [InputReactance("series", "L", 13e-9)]
    fast = evaluate_loaded_switch_power(loaded, branches, input_reactances=fixed)
    nodal = evaluate_loaded_switch_physical_power(loaded, branches, input_elements=fixed)
    np.testing.assert_allclose(nodal.input_gamma, fast.input_gamma, atol=2e-12)
    np.testing.assert_allclose(nodal.dut_absorbed_power, fast.dut_absorbed_power, atol=2e-12)
    assert np.max(np.abs(nodal.power_balance_error)) < 1e-10


def test_measured_switch_branches_and_input_parts_close_power_balance():
    tutorial = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    components = Path(r"C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary")
    if not tutorial.exists() or not components.exists():
        pytest.skip("licensed Optenni switch/component inputs are not installed")
    capacitors = load_murata_gjm15_catalog(
        components / "Capacitors" / "Murata Capacitors gjm15"
    )
    inductors = load_coilcraft_0402cs_catalog(
        components / "Inductors" / "Coilcraft Inductors 0402cs"
    )
    c12 = min(capacitors, key=lambda item: abs(item.value - 1.2e-12))
    c08 = min(capacitors, key=lambda item: abs(item.value - 0.8e-12))
    l13 = min(inductors, key=lambda item: abs(item.value - 13e-9))
    dut = load_touchstone(tutorial / "Switch_Tuner_Tutorial.s1p")
    state = load_mdif(tutorial / "tutorial_SP2T.mdif").state("all on")
    frequencies = np.array([725e6, 2045e6])
    dut_s11 = np.interp(frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].real) + 1j * np.interp(
        frequencies, dut.frequencies_hz, dut.s_parameters[:, 0, 0].imag
    )
    power = evaluate_loaded_switch_physical_power(
        preload_switch_state(frequencies, dut_s11, state),
        [load_component_model(c12), load_component_model(c08)],
        input_elements=[InputModelPlacement("series", load_component_model(l13))],
    )
    assert np.all(power.matching_network_loss > 0)
    assert np.max(np.abs(power.power_balance_error)) < 1e-10
