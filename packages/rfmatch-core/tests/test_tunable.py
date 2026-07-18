from pathlib import Path

import numpy as np
import pytest

from rfmatch_core import (
    Band,
    FrequencyConfiguration,
    MDIFModel,
    MDIFState,
    Objective,
    Problem,
    LumpedModel,
    ModelPlacement,
    OptimizationCancelled,
    TunableProblem,
    TunableMeasuredComponentOptimizer,
    evaluate_tunable_physical,
    rank_tunable_fixed_networks,
)


def _series_fixture(frequencies: np.ndarray, capacitance: float) -> MDIFState:
    z0 = 50.0
    matrices = []
    for frequency in frequencies:
        impedance = 1.0 / (1j * 2 * np.pi * frequency * capacitance)
        denominator = 2 * z0 + impedance
        matrices.append([
            [impedance / denominator, 2 * z0 / denominator],
            [2 * z0 / denominator, impedance / denominator],
        ])
    return MDIFState("state", capacitance * 1e12, "pF", frequencies, np.asarray(matrices), z0)


def test_tunable_physical_selects_state_independently_for_each_configuration():
    frequencies = np.asarray([700e6, 900e6, 2e9])
    # A deliberately simple matched load verifies state/configuration plumbing
    # and physical power conservation without relying on an external library.
    dut = np.zeros((len(frequencies), 1, 1), dtype=complex)
    base = Problem(frequencies, dut, {0: [Band(700e6, 2e9)]})
    problem = TunableProblem(
        base,
        (
            FrequencyConfiguration("low", {0: [Band(700e6, 700e6)]}),
            FrequencyConfiguration("high", {0: [Band(2e9, 2e9)]}),
        ),
    )
    tuner = MDIFModel(
        "synthetic",
        (_series_fixture(frequencies, 1e-12), _series_fixture(frequencies, 8e-12)),
    )
    candidate = evaluate_tunable_physical(problem, (), tuner, Objective())
    assert set(candidate.state_by_configuration) == {"low", "high"}
    assert np.isfinite(candidate.score_db)
    assert candidate.metrics["maximum_power_balance_error"] < 1e-12
    assert candidate.metrics["physical_sweep_evaluations"] == 2


def test_tunable_physical_honors_explicit_assignments_and_rejects_unknown_names():
    frequencies = np.asarray([1e9])
    base = Problem(frequencies, np.zeros((1, 1, 1), complex), {0: [Band(1e9, 1e9)]})
    problem = TunableProblem(base, (FrequencyConfiguration("set", {0: [Band(1e9, 1e9)]}),))
    tuner = MDIFModel("synthetic", (_series_fixture(frequencies, 1e-12), _series_fixture(frequencies, 2e-12)))
    candidate = evaluate_tunable_physical(problem, (), tuner, state_by_configuration={"set": 2.0})
    assert candidate.state_by_configuration == {"set": "2 pF"}
    assert candidate.metrics["physical_sweep_evaluations"] == 1
    try:
        evaluate_tunable_physical(problem, (), tuner, state_by_configuration={"missing": 1.0})
    except ValueError as exc:
        assert "unknown configurations" in str(exc)
    else:
        raise AssertionError("unknown configuration assignment was accepted")


def test_tunable_fixed_network_ranking_is_deterministic_and_deduplicated():
    frequencies = np.asarray([1e9, 1.5e9])
    base = Problem(frequencies, np.zeros((2, 1, 1), complex), {0: [Band(1e9, 1.5e9)]})
    problem = TunableProblem(
        base,
        (FrequencyConfiguration("set", {0: [Band(1e9, 1.5e9)]}),),
    )
    tuner = MDIFModel("synthetic", (_series_fixture(frequencies, 1e-12),))
    extra = (ModelPlacement("series", 0, LumpedModel("extra", "L", 10e-9)),)

    ranked = rank_tunable_fixed_networks(problem, [extra, (), extra], tuner)
    reversed_ranked = rank_tunable_fixed_networks(problem, [(), extra], tuner)

    assert ranked.physical_evaluations == 2
    assert len(ranked.candidates) == 2
    assert [candidate.score_db for candidate in ranked.candidates] == [
        candidate.score_db for candidate in reversed_ranked.candidates
    ]
    assert [candidate.fixed_placements for candidate in ranked.candidates] == [
        candidate.fixed_placements for candidate in reversed_ranked.candidates
    ]


def test_tunable_optimizer_cancellation_and_progress_contract():
    frequencies = np.asarray([1e9])
    problem = TunableProblem(
        Problem(frequencies, np.zeros((1, 1, 1), complex), {0: [Band(1e9, 1e9)]}),
        (FrequencyConfiguration("set", {0: [Band(1e9, 1e9)]}),),
    )
    tuner = MDIFModel("synthetic", (_series_fixture(frequencies, 1e-12),))
    events = []
    optimizer = TunableMeasuredComponentOptimizer(
        problem,
        tuner,
        (),
        (),
        progress_callback=events.append,
        cancel_check=lambda: True,
    )

    optimizer._emit_progress("test", 1, 2, "working")
    with pytest.raises(OptimizationCancelled):
        optimizer.optimize()

    assert events == [{
        "stage": "test", "current": 1, "total": 2, "message": "working",
    }]
