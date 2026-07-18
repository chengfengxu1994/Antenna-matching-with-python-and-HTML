from dataclasses import replace

import numpy as np
import pytest

from rfmatch_core import (
    Band,
    FrequencyConfiguration,
    InputModelPlacement,
    MDIFModel,
    MDIFState,
    ModelPlacement,
    MicrostripLineModel,
    PCBSubstrate,
    Problem,
    S2PModel,
    SwitchTunableProblem,
    TunableProblem,
    ToleranceModel,
    YieldCriteria,
    evaluate_loaded_switch_physical_power,
    monte_carlo_switch_yield,
    monte_carlo_tunable_yield,
    preload_switch_state,
)


def _series_model(name, kind, value, tolerance, frequencies):
    omega = 2 * np.pi * frequencies
    impedance = 0.2 + (
        1j * omega * value if kind == "L" else 1.0 / (1j * omega * value)
    )
    denominator = 100.0 + impedance
    reflection = impedance / denominator
    transmission = 100.0 / denominator
    matrices = np.empty((len(frequencies), 2, 2), dtype=complex)
    matrices[:, 0, 0] = reflection
    matrices[:, 1, 1] = reflection
    matrices[:, 0, 1] = transmission
    matrices[:, 1, 0] = transmission
    return S2PModel(
        name, frequencies, matrices, 50.0, tolerance, kind, value
    )


def _fixture():
    frequencies = np.array([0.9e9, 1.0e9, 1.1e9, 1.9e9, 2.0e9, 2.1e9])
    switch_s = np.zeros((len(frequencies), 2, 2), dtype=complex)
    switch_s[:, 0, 1] = 1.0
    switch_s[:, 1, 0] = 1.0
    state = MDIFState("state", "through", None, frequencies, switch_s)
    dut_gamma = np.array([0.4 + 0.1j, 0.35 + 0.1j, 0.3 + 0.1j,
                          0.2 - 0.2j, 0.25 - 0.2j, 0.3 - 0.2j])
    problem = SwitchTunableProblem(
        frequencies,
        dut_gamma,
        (
            FrequencyConfiguration("low", {0: (Band(0.9e9, 1.1e9),)}),
            FrequencyConfiguration("high", {0: (Band(1.9e9, 2.1e9),)}),
        ),
    )
    loaded = preload_switch_state(frequencies, dut_gamma, state)
    branch = _series_model("C1", "C", 1.2e-12, 0.1, frequencies)
    input_model = _series_model("L1", "L", 8e-9, 0.1, frequencies)
    return problem, loaded, branch, input_model


def test_switch_physical_solver_scales_measured_reactance_not_complete_s_matrix():
    _, loaded, branch, input_model = _fixture()
    nominal = evaluate_loaded_switch_physical_power(
        loaded, [branch], input_elements=[InputModelPlacement("series", input_model)]
    )
    varied = evaluate_loaded_switch_physical_power(
        loaded,
        [branch],
        input_elements=[InputModelPlacement("series", input_model)],
        branch_scales=[1.1],
        input_scales=[0.9],
    )
    assert not np.allclose(nominal.input_gamma, varied.input_gamma)
    assert np.max(np.abs(varied.power_balance_error)) < 1e-10


def test_switch_yield_uses_one_shared_draw_across_all_configurations():
    problem, loaded, branch, input_model = _fixture()
    kwargs = dict(
        problem=problem,
        loaded_states={"through": loaded},
        branch_models=[branch],
        input_elements=[InputModelPlacement("series", input_model)],
        state_by_configuration={"low": "through", "high": "through"},
        criteria=YieldCriteria(
            minimum_total_efficiency=0.45,
            minimum_average_total_efficiency=0.5,
            minimum_return_loss_db=2.0,
        ),
        samples=60,
        seed=19,
    )
    first = monte_carlo_switch_yield(**kwargs)
    second = monte_carlo_switch_yield(**kwargs)
    np.testing.assert_array_equal(first.sample_scores_db, second.sample_scores_db)
    assert first.distribution == "uniform"
    assert set(first.configuration_yield_fraction) == {"low", "high"}
    assert first.yield_fraction <= min(first.configuration_yield_fraction.values())
    assert set(first.worst_sample) == {"branch_1", "input_1"}
    assert all(0.9 <= value <= 1.1 for value in first.worst_sample.values())


def test_switch_yield_models_perfect_batch_correlation_and_shared_temperature():
    problem, loaded, branch, input_model = _fixture()
    result = monte_carlo_switch_yield(
        problem,
        {"through": loaded},
        [branch],
        [InputModelPlacement("series", input_model)],
        {"low": "through", "high": "through"},
        YieldCriteria(minimum_total_efficiency=0.1),
        samples=30,
        seed=29,
        tolerance_model=ToleranceModel(
            batch_correlation=1.0,
            temperature_min_c=85.0,
            temperature_max_c=85.0,
            capacitor_tempco_ppm_per_c=-500.0,
            inductor_tempco_ppm_per_c=1000.0,
        ),
    )
    worst = result.worst_sample
    assert worst["temperature_c"] == 85.0
    assert worst["temperature_delta_c"] == 60.0
    # Both parts have the same 10% tolerance and a perfectly correlated batch
    # draw, so their scale ratio contains only their opposite temperature drift.
    assert worst["branch_1"] / worst["input_1"] == pytest.approx(0.97 / 1.06)
    assert result.variation_model["batch_correlation"] == 1.0
    assert result.variation_model["correlation_method"] == "gaussian_copula"


def test_switch_yield_prefers_exact_part_environment_over_global_fallback():
    problem, loaded, branch, input_model = _fixture()
    branch = replace(
        branch, tempco_ppm_per_c=0.0, systematic_bias_pct=10.0,
        environment_provenance="laboratory_measurement",
    )
    input_model = replace(
        input_model, tempco_ppm_per_c=0.0, systematic_bias_pct=-5.0,
        environment_provenance="manufacturer_datasheet",
    )
    result = monte_carlo_switch_yield(
        problem,
        {"through": loaded},
        [branch],
        [InputModelPlacement("series", input_model)],
        {"low": "through", "high": "through"},
        YieldCriteria(),
        samples=5,
        seed=31,
        tolerance_model=ToleranceModel(
            batch_correlation=1.0,
            temperature_min_c=85.0,
            temperature_max_c=85.0,
            capacitor_tempco_ppm_per_c=-9000.0,
            inductor_tempco_ppm_per_c=9000.0,
            capacitor_bias_pct=-20.0,
            inductor_bias_pct=20.0,
        ),
    )
    # Perfect batch correlation gives both 10%-tolerance parts the same
    # manufacturing draw. Their ratio therefore isolates the exact-part bias;
    # zero exact-part tempco also proves the global temperature fallback lost.
    assert result.worst_sample["branch_1"] / result.worst_sample["input_1"] == pytest.approx(
        1.10 / 0.95
    )


def test_tunable_yield_reuses_fixed_component_draw_across_states():
    switch_problem, _, branch, _ = _fixture()
    frequencies = switch_problem.frequencies_hz
    dut = switch_problem.dut_s11[:, None, None]
    base = Problem(
        frequencies,
        dut,
        {0: (Band(0.9e9, 1.1e9),)},
    )
    problem = TunableProblem(base, switch_problem.configurations)
    def resistive_two_port(resistance):
        matrix = np.empty((len(frequencies), 2, 2), dtype=complex)
        reflection = resistance / (100.0 + resistance)
        transmission = 100.0 / (100.0 + resistance)
        matrix[:, 0, 0] = reflection
        matrix[:, 1, 1] = reflection
        matrix[:, 0, 1] = transmission
        matrix[:, 1, 0] = transmission
        return matrix
    through = resistive_two_port(0.05)
    lossy = resistive_two_port(3.0)
    tuner = MDIFModel(
        "tuner",
        (
            MDIFState("state", "A", None, frequencies, through),
            MDIFState("state", "B", None, frequencies, lossy),
        ),
    )
    result = monte_carlo_tunable_yield(
        problem,
        [ModelPlacement("series", 0, branch)],
        tuner,
        {"low": "A", "high": "B"},
        YieldCriteria(minimum_total_efficiency=0.2, minimum_return_loss_db=1.0),
        samples=40,
        seed=23,
    )
    assert set(result.configuration_yield_fraction) == {"low", "high"}
    assert result.yield_fraction <= min(result.configuration_yield_fraction.values())
    assert set(result.worst_sample) == {"fixed_1"}
    assert 0.9 <= result.worst_sample["fixed_1"] <= 1.1

    microstrip = MicrostripLineModel.from_electrical_design(
        "shared_board_line",
        PCBSubstrate("FR-4", 4.5, 1.6e-3, loss_tangent=0.02),
        50.0, 15.0, 1.0e9, 0.1e-3, 10e-3,
        width_tolerance=0.1,
        length_tolerance=0.02,
        substrate_height_tolerance=0.05,
        relative_permittivity_tolerance=0.04,
    )
    microstrip_kwargs = dict(
        problem=problem,
        fixed_placements=[ModelPlacement("series", 0, microstrip)],
        tuner=tuner,
        state_by_configuration={"low": "A", "high": "B"},
        criteria=YieldCriteria(minimum_total_efficiency=0.2, minimum_return_loss_db=1.0),
        samples=20,
        seed=29,
    )
    first_board = monte_carlo_tunable_yield(**microstrip_kwargs)
    second_board = monte_carlo_tunable_yield(**microstrip_kwargs)
    np.testing.assert_array_equal(first_board.sample_scores_db, second_board.sample_scores_db)
    assert set(first_board.worst_sample) == {
        "fixed_1.trace_width",
        "fixed_1.physical_length",
        "fixed_1.substrate_height",
        "fixed_1.relative_permittivity",
    }
    assert set(first_board.configuration_yield_fraction) == {"low", "high"}
