import unittest

import numpy as np

from rfmatch_core.evaluator import evaluate, evaluate_lumped_physical, score_sweep
from rfmatch_core.models import Band, Candidate, Component, Element, IsolationTarget, LumpedLossModel, LumpedModel, Objective, Problem
from rfmatch_core.optimizer import MatchingOptimizer, SearchConfig
from rfmatch_core.benchmarks import _directed_isolation_summary
from rfmatch_core.physical import evaluate_physical_problem
from rfmatch_core.physical_optimizer import ModelPlacement, build_model_circuit_topology


class EvaluatorOptimizerTests(unittest.TestCase):
    def test_vectorized_one_port_lossy_evaluation_matches_nodal_solver(self):
        frequencies = np.array([0.8e9, 1.0e9, 1.2e9])
        problem = Problem(
            frequencies,
            np.array([[[0.3 + 0.1j]], [[0.2 - 0.1j]], [[0.4 + 0.05j]]]),
            {0: [Band(0.8e9, 1.2e9)]},
        )
        candidate = Candidate([
            Element("shunt", "C", 0, 0.8e-12),
            Element("series", "L", 0, 6e-9),
        ])
        loss = LumpedLossModel(30.0, 1e9, 0.0, 0.4)
        fast = evaluate_lumped_physical(problem, candidate, Objective(), loss)
        topology = build_model_circuit_topology(1, [
            ModelPlacement("shunt", 0, LumpedModel("C", "C", 0.8e-12, esr=0.4)),
            ModelPlacement("series", 0, LumpedModel("L", "L", 6e-9, q=30.0, q_reference_hz=1e9)),
        ])
        exact = evaluate_physical_problem(problem, topology)
        np.testing.assert_allclose(
            fast.metrics["total_efficiency"], exact.total_efficiency, atol=1e-12
        )
        np.testing.assert_allclose(
            fast.metrics["component_loss"], exact.component_loss, atol=1e-12
        )

    def test_radiation_efficiency_changes_total_efficiency(self):
        f = np.array([1e9, 1.1e9])
        s = np.array([[[0.5 + 0j]], [[0.5 + 0j]]])
        problem = Problem(f, s, {0: [Band(1e9, 1.1e9)]}, radiation_efficiency={0: np.array([0.5, 0.5])})
        result = evaluate(problem, Candidate([]), Objective())
        self.assertTrue(np.allclose(result.metrics["total_efficiency"][:, 0], 0.375))

    def test_continuous_then_discrete_search_improves_match(self):
        f = np.linspace(0.95e9, 1.05e9, 5)
        gamma = (25 + 20j - 50) / (25 + 20j + 50)
        problem = Problem(f, np.full((5, 1, 1), gamma, dtype=complex), {0: [Band(f[0], f[-1])]})
        optimizer = MatchingOptimizer(problem, config=SearchConfig(restarts=4, iterations=12, seed=4))
        baseline = evaluate(problem, Candidate([]), Objective()).score_db
        library = [Component("L0p8", "L", 0.82e-9), Component("L1", "L", 1e-9), Component("C3", "C", 3.3e-12), Component("C5", "C", 4.7e-12)]
        result = optimizer.optimize([[('series', 'L', 0), ('shunt', 'C', 0)]], library)
        self.assertGreater(result.best.score_db, baseline)
        self.assertTrue(all(element.name for element in result.best.elements))

    def test_progressive_ladder_search_reaches_six_components_with_bounded_work(self):
        frequencies = np.array([0.9e9, 1.0e9, 1.1e9])
        problem = Problem(
            frequencies,
            np.full((3, 1, 1), 0.45 + 0.2j, dtype=complex),
            {0: [Band(frequencies[0], frequencies[-1])]},
        )
        optimizer = MatchingOptimizer(
            problem,
            config=SearchConfig(restarts=1, iterations=1, keep=100, seed=3),
        )
        result = optimizer.discover_ladder_topologies(
            port=0, maximum_components=6, topology_beam_width=3,
        )
        six_element = [item for item in result.candidates if len(item.elements) == 6]
        self.assertTrue(six_element)
        for candidate in six_element:
            connections = [item.connection for item in candidate.elements]
            self.assertTrue(all(a != b for a, b in zip(connections, connections[1:])))
        self.assertLessEqual(result.evaluations, 400)

    def test_objective_emphasizes_worst_frequency(self):
        f = np.array([1e9, 2e9])
        s = np.array([[[0.1 + 0j]], [[0.9 + 0j]]])
        problem = Problem(f, s, {0: [Band(1e9, 2e9)]})
        worst = evaluate(problem, Candidate([]), Objective(within_band_average_weight=0.0))
        average = evaluate(problem, Candidate([]), Objective(within_band_average_weight=1.0))
        self.assertLess(worst.score_db, average.score_db)

    def test_band_priority_weight_scales_target_margin(self):
        frequencies = np.array([1e9, 2e9])
        s = np.array([[[0.8 + 0j]], [[0.2 + 0j]]])
        default_problem = Problem(frequencies, s, {
            0: [Band(1e9, 1e9), Band(2e9, 2e9)],
        })
        priority_problem = Problem(frequencies, s, {
            0: [Band(1e9, 1e9, weight=3.0), Band(2e9, 2e9)],
        })
        objective = Objective(
            within_band_average_weight=1.0,
            across_band_average_weight=1.0,
            port_average_weight=1.0,
        )
        default = evaluate(default_problem, Candidate([]), objective)
        priority = evaluate(priority_problem, Candidate([]), objective)
        self.assertLess(priority.score_db, default.score_db)
        self.assertEqual(priority.metrics["bands"][(0, 0)]["weight"], 3.0)

    def test_invalid_band_priority_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "band weight"):
            Band(1e9, 2e9, weight=-1.0)

    def test_precomputed_sweep_scoring_matches_standard_evaluation(self):
        f = np.array([1e9, 1.1e9])
        s = np.array([[[0.2 + 0.1j]], [[0.1 - 0.1j]]])
        problem = Problem(f, s, {0: [Band(1e9, 1.1e9)]})
        candidate = Candidate([])
        objective = Objective()
        standard = evaluate(problem, candidate, objective)
        rescored = score_sweep(problem, candidate, objective, s, 1.0 - np.abs(s[:, 0, 0])[:, None] ** 2)
        self.assertAlmostEqual(standard.score_db, rescored.score_db, places=12)
        self.assertTrue(np.allclose(standard.metrics["port_scores_db"], rescored.metrics["port_scores_db"]))

    def test_directed_isolation_target_uses_s_destination_source(self):
        f = np.array([1e9])
        # S21 is poor (-6.02 dB), while S12 is good (-40 dB).
        s = np.array([[[0.1, 0.01], [0.5, 0.1]]], dtype=complex)
        bands = {0: [Band(1e9, 1e9)]}
        poor_direction = Problem(f, s, bands, isolation_targets=(IsolationTarget(0, 1, 1e9, 1e9, -20.0),))
        good_direction = Problem(f, s, bands, isolation_targets=(IsolationTarget(1, 0, 1e9, 1e9, -20.0),))
        poor = evaluate(poor_direction, Candidate([]), Objective())
        good = evaluate(good_direction, Candidate([]), Objective())
        expected_penalty = 20.0 * np.log10(0.5) - (-20.0)
        self.assertAlmostEqual(poor.metrics["isolation_penalty_db"], expected_penalty, places=12)
        self.assertEqual(good.metrics["isolation_penalty_db"], 0.0)
        self.assertAlmostEqual(good.score_db - poor.score_db, expected_penalty, places=12)
        self.assertFalse(poor.metrics["isolation_targets"][0]["passed"])
        self.assertTrue(good.metrics["isolation_targets"][0]["passed"])

    def test_invalid_isolation_target_is_rejected(self):
        f = np.array([1e9])
        s = np.zeros((1, 2, 2), dtype=complex)
        with self.assertRaisesRegex(ValueError, "distinct ports"):
            Problem(f, s, {0: [Band(1e9, 1e9)]}, isolation_targets=(IsolationTarget(0, 0, 1e9, 1e9),))

    def test_isolation_summary_uses_standard_sij_labels(self):
        transmission = np.full((1, 2, 2), -100.0)
        transmission[0, 1, 0] = -12.0
        transmission[0, 0, 1] = -34.0
        summary = _directed_isolation_summary(transmission)
        self.assertEqual(summary["S21"]["worst_db"], -12.0)
        self.assertEqual(summary["S12"]["worst_db"], -34.0)


if __name__ == "__main__":
    unittest.main()
