import numpy as np
import pytest

from rfmatch_core import (
    Band,
    LazyComponentSpec,
    MeasuredComponentOptimizer,
    MeasuredOptimizationResult,
    MeasuredSearchConfig,
    Objective,
    Problem,
    S2PModel,
    exhaustive_measured_joint_search,
    exhaustive_measured_search,
    measured_search_recall,
)


def _ideal_spec(name, kind, value, frequencies, z0=50.0):
    def loader():
        omega = 2.0 * np.pi * frequencies
        impedance = (
            1j * omega * value
            if kind == "L"
            else 1.0 / (1j * omega * value)
        )
        reflection = impedance / (2.0 * z0 + impedance)
        transmission = 2.0 * z0 / (2.0 * z0 + impedance)
        matrices = np.empty((len(frequencies), 2, 2), dtype=complex)
        matrices[:, 0, 0] = reflection
        matrices[:, 1, 1] = reflection
        matrices[:, 0, 1] = transmission
        matrices[:, 1, 0] = transmission
        return S2PModel(name, frequencies, matrices, z0, 0.0, kind, value)

    return LazyComponentSpec(name, kind, value, 0.05, "calibration", name, loader)


def _fixture(catalog_size=2):
    frequencies = np.linspace(1.0e9, 2.0e9, 5)
    phase = np.linspace(-0.4, 0.5, len(frequencies))
    gamma = 0.58 * np.exp(1j * phase)
    problem = Problem(
        frequencies,
        gamma[:, None, None],
        {0: [Band(1.0e9, 2.0e9)]},
    )
    l_values = (1.0, 2.2, 4.7, 10.0)[:catalog_size]
    c_values = (0.5, 1.0, 2.2, 4.7)[:catalog_size]
    inductors = [
        _ideal_spec(f"L{value}", "L", value * 1e-9, frequencies)
        for value in l_values
    ]
    capacitors = [
        _ideal_spec(f"C{value}", "C", value * 1e-12, frequencies)
        for value in c_values
    ]
    return problem, inductors, capacitors


def test_exhaustive_two_part_catalog_has_complete_expected_candidate_count():
    problem, inductors, capacitors = _fixture(2)
    result = exhaustive_measured_search(
        problem, inductors, capacitors, max_components_per_port=2
    )
    # empty + four one-element families x 2 parts + eight two-element
    # topology families x 2 x 2 assignments
    assert result.physical_evaluations == 1 + 4 * 2 + 8 * 2 * 2
    assert len(result.candidates) == result.physical_evaluations
    assert result.loaded_component_models == 4
    assert result.best.metrics["maximum_power_balance_error"] < 1e-10


def test_exhaustive_calibration_refuses_unbounded_combinatorial_request():
    problem, inductors, capacitors = _fixture(2)
    with pytest.raises(ValueError, match="requires 41 evaluations"):
        exhaustive_measured_search(
            problem,
            inductors,
            capacitors,
            max_components_per_port=2,
            max_evaluations=40,
        )


def test_recall_report_uses_exact_parts_and_topology_metrics():
    problem, inductors, capacitors = _fixture(2)
    exhaustive = exhaustive_measured_search(problem, inductors, capacitors)
    heuristic = MeasuredOptimizationResult(
        exhaustive.candidates[:3],
        {0: exhaustive.candidates[:3]},
        ideal_evaluations=7,
        physical_evaluations=3,
        loaded_component_models=4,
    )
    report = measured_search_recall(heuristic, exhaustive, top_k=5)
    assert report.exact_matches == 3
    assert report.exact_top_k_recall == pytest.approx(0.6)
    assert report.topology_top_k_recall >= report.exact_top_k_recall
    assert report.best_score_gap_db == pytest.approx(0.0)
    assert report.best_exhaustive_rank_found == 1


def test_full_neighborhood_hierarchical_search_matches_small_exhaustive_top_ten():
    problem, inductors, capacitors = _fixture(3)
    objective = Objective()
    heuristic = MeasuredComponentOptimizer(
        problem,
        inductors,
        capacitors,
        objective,
        MeasuredSearchConfig(
            ideal_restarts=2,
            ideal_iterations=5,
            ideal_keep=32,
            nearest_parts=3,
            per_port_keep=100,
            result_keep=100,
            joint_refine_seeds=0,
            joint_refine_passes=0,
            seed=3,
            max_components_per_port=2,
        ),
    ).optimize()
    exhaustive = exhaustive_measured_search(problem, inductors, capacitors, objective)
    report = measured_search_recall(heuristic, exhaustive, top_k=10)
    assert report.exact_top_k_recall == pytest.approx(1.0)
    assert report.best_score_gap_db < 1e-12


def test_joint_exhaustive_calibration_preserves_coupling_and_top_ten_ranking():
    frequencies = np.array([1.0e9, 1.5e9, 2.0e9])
    matrix = np.array([[0.45 + 0.1j, 0.12], [0.12, 0.32 - 0.08j]])
    problem = Problem(
        frequencies,
        np.stack([matrix, matrix * 0.95, matrix * 0.9]),
        {0: [Band(1.0e9, 2.0e9)], 1: [Band(1.0e9, 2.0e9)]},
    )
    inductors = [
        _ideal_spec(f"L{value}", "L", value * 1e-9, frequencies)
        for value in (1.0, 4.7)
    ]
    capacitors = [
        _ideal_spec(f"C{value}", "C", value * 1e-12, frequencies)
        for value in (0.5, 2.2)
    ]
    config = MeasuredSearchConfig(
        ideal_restarts=1,
        ideal_iterations=2,
        ideal_keep=16,
        nearest_parts=2,
        per_port_keep=9,
        result_keep=100,
        joint_refine_seeds=0,
        joint_refine_passes=0,
        seed=5,
        max_components_per_port=1,
    )
    heuristic = MeasuredComponentOptimizer(
        problem, inductors, capacitors, Objective(), config
    ).optimize()
    exhaustive = exhaustive_measured_joint_search(
        problem,
        inductors,
        capacitors,
        max_components_per_port=1,
        max_evaluations=100,
    )
    report = measured_search_recall(heuristic, exhaustive, top_k=10)
    assert exhaustive.physical_evaluations == 9 ** 2
    assert report.exact_top_k_recall == pytest.approx(1.0)
    assert report.best_score_gap_db < 1e-12
    assert exhaustive.best.metrics["maximum_power_balance_error"] < 1e-9
