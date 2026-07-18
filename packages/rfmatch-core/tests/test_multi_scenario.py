import unittest

import numpy as np

from rfmatch_core.evaluator import evaluate
from rfmatch_core.models import Band, Candidate, Element, Objective, Problem
from rfmatch_core.multi_scenario import (
    MultiScenarioMatchingOptimizer,
    MultiScenarioProblem,
    ScenarioProblem,
    evaluate_multi_scenario,
)
from rfmatch_core.optimizer import SearchConfig


def _problem(gamma: complex) -> Problem:
    frequencies = np.array([0.9e9, 1.0e9, 1.1e9])
    matrices = np.full((len(frequencies), 1, 1), gamma, dtype=complex)
    return Problem(
        frequencies,
        matrices,
        {0: [Band(0.9e9, 1.1e9)]},
    )


class MultiScenarioTests(unittest.TestCase):
    def test_balanced_score_blends_weighted_average_and_true_worst_scenario(self):
        objective = Objective()
        first = ScenarioProblem("free space", _problem(0.0), 1.0)
        second = ScenarioProblem("cover", _problem(0.8), 3.0)
        multi = MultiScenarioProblem((first, second), scenario_average_weight=0.5)

        result = evaluate_multi_scenario(multi, Candidate([]), objective)
        first_score = evaluate(first.problem, Candidate([]), objective).score_db
        second_score = evaluate(second.problem, Candidate([]), objective).score_db
        weighted = (first_score + 3.0 * second_score) / 4.0
        expected = 0.5 * min(first_score, second_score) + 0.5 * weighted

        self.assertAlmostEqual(result.score_db, expected, places=12)
        self.assertAlmostEqual(result.metrics["weighted_average_score_db"], weighted, places=12)
        self.assertEqual([item["name"] for item in result.metrics["scenarios"]], ["free space", "cover"])

    def test_shared_optimizer_keeps_one_identical_network_for_all_scenarios(self):
        multi = MultiScenarioProblem.from_mode((
            ScenarioProblem("a", _problem(0.6 + 0.2j)),
            ScenarioProblem("b", _problem(0.4 - 0.3j)),
        ))
        optimizer = MultiScenarioMatchingOptimizer(
            multi,
            Objective(),
            SearchConfig(restarts=2, iterations=5, keep=4, seed=3),
        )
        result = optimizer.optimize([[('series', 'L', 0)]])

        self.assertTrue(result.candidates)
        best = result.best
        self.assertEqual(len(best.elements), 1)
        self.assertEqual(len(best.metrics["scenarios"]), 2)
        self.assertTrue(all(
            item["metrics"]["total_efficiency"].shape == (3, 1)
            for item in best.metrics["scenarios"]
        ))

    def test_invalid_scenario_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            MultiScenarioProblem((ScenarioProblem("only", _problem(0.2)),))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ScenarioProblem("bad", _problem(0.2), -1.0)
        with self.assertRaisesRegex(ValueError, "unknown"):
            MultiScenarioProblem.from_mode((
                ScenarioProblem("a", _problem(0.2)),
                ScenarioProblem("b", _problem(0.3)),
            ), "median")


if __name__ == "__main__":
    unittest.main()
