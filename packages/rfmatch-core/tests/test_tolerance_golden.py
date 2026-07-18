import tempfile
from pathlib import Path
import unittest

import numpy as np

from rfmatch_core import (
    Band,
    Branch,
    CircuitTopology,
    LumpedModel,
    Problem,
    S2PModel,
    ToleranceModel,
    YieldCriteria,
    load_optenni_tolerance_export,
    monte_carlo_yield,
    summarize_optenni_tolerance,
)
from rfmatch_core.golden import compare_golden, load_golden_csv


class ToleranceGoldenTests(unittest.TestCase):
    def test_environmental_tolerance_model_validates_physical_bounds(self):
        with self.assertRaisesRegex(ValueError, "batch_correlation"):
            ToleranceModel(batch_correlation=1.1)
        with self.assertRaisesRegex(ValueError, "provided together"):
            ToleranceModel(temperature_min_c=-40.0)
        with self.assertRaisesRegex(ValueError, "at least"):
            ToleranceModel(temperature_min_c=85.0, temperature_max_c=-40.0)
        with self.assertRaisesRegex(ValueError, "inductor_bias_pct"):
            ToleranceModel(inductor_bias_pct=-100.0)
        with self.assertRaisesRegex(ValueError, "capacitor_bias_pct"):
            ToleranceModel(capacitor_bias_pct=-101.0)

    def test_systematic_lc_bias_is_applied_and_zero_bias_is_legacy_compatible(self):
        frequency = 1e9
        problem = Problem(
            np.array([frequency]), np.array([[[0j]]]),
            {0: [Band(frequency, frequency)]},
        )
        topology = CircuitTopology(
            external_nodes=("in",), dut_nodes=("dut",),
            branches=(
                Branch("L1", "in", "dut", LumpedModel("L1", "L", 5e-9, tolerance=0.1)),
                Branch("C1", "dut", None, LumpedModel("C1", "C", 1e-12, tolerance=0.1)),
            ),
        )
        default = monte_carlo_yield(
            problem, topology, YieldCriteria(), samples=20, seed=23,
        )
        explicit_zero = monte_carlo_yield(
            problem, topology, YieldCriteria(), samples=20, seed=23,
            tolerance_model=ToleranceModel(inductor_bias_pct=0, capacitor_bias_pct=0),
        )
        biased = monte_carlo_yield(
            problem, topology, YieldCriteria(), samples=20, seed=23,
            tolerance_model=ToleranceModel(inductor_bias_pct=5, capacitor_bias_pct=-4),
        )

        self.assertTrue(np.array_equal(default.sample_scores_db, explicit_zero.sample_scores_db))
        self.assertAlmostEqual(
            biased.worst_sample["L1"] / default.worst_sample["L1"], 1.05,
        )
        self.assertAlmostEqual(
            biased.worst_sample["C1"] / default.worst_sample["C1"], 0.96,
        )
        self.assertEqual(biased.variation_model["bias_scope"], "systematic_by_component_kind")

    def test_native_optenni_duplicate_tolerance_columns_are_preserved(self):
        text = """% PlotTitle: \"Matching circuit, port 1\"
% X axis: \"Frequency [GHz]\"
% Y axis: \"S parameters/loss [dB]\"
\"Frequency [GHz]\"\t\"S11\"\t\"S11 tolerance data\"\t\"S11 tolerance data\"\t\"Total efficiency\"\t\"Total efficiency tolerance data\"\t\"Total efficiency tolerance data\"
1.7\t-10\t-9\t-8\t-1.0\t-0.9\t-1.1
2.5\t-11\t-10\t-9\t-0.8\t-0.7\t-1.2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tolerance.txt"
            path.write_text(text, encoding="utf-8")
            export = load_optenni_tolerance_export(path)
        self.assertEqual(export.samples, 2)
        self.assertEqual(export.s11_variants_db.shape, (2, 2))
        self.assertAlmostEqual(export.frequencies_hz[-1], 2.5e9)
        self.assertAlmostEqual(export.total_efficiency_variants[0, 0], 10 ** -0.09)

    def test_optenni_joint_minimum_and_average_efficiency_yield(self):
        text = """% Y axis: \"S parameters/loss [dB]\"
\"Frequency [GHz]\"\t\"S11\"\t\"S11 tolerance data\"\t\"S11 tolerance data\"\t\"Total efficiency\"\t\"Total efficiency tolerance data\"\t\"Total efficiency tolerance data\"
1.7\t-10\t-9\t-8\t-1.0\t-0.9\t-1.1
2.5\t-11\t-10\t-9\t-0.8\t-0.7\t-1.2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tolerance.txt"
            path.write_text(text, encoding="utf-8")
            export = load_optenni_tolerance_export(path)
        minimum_only = summarize_optenni_tolerance(
            export, 1.7e9, 2.5e9,
            minimum_total_efficiency=10 ** (-1.2 / 10),
        )
        joint = summarize_optenni_tolerance(
            export, 1.7e9, 2.5e9,
            minimum_total_efficiency=10 ** (-1.2 / 10),
            minimum_average_total_efficiency=10 ** (-0.85 / 10),
        )
        self.assertEqual(minimum_only["passed_samples"], 2)
        self.assertEqual(joint["passed_samples"], 1)

    def test_real_optenni_tolerance_baseline_reproduces_displayed_zero_yield(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "benchmarks"
            / "optenni_exports"
            / "optimization_settings_pcsl_tolerance_100.txt"
        )
        if not path.exists():
            self.skipTest("Optenni tolerance baseline is not available")
        export = load_optenni_tolerance_export(path)
        summary = summarize_optenni_tolerance(
            export,
            1.7e9,
            2.5e9,
            minimum_total_efficiency=10 ** (-1.0 / 10),
            minimum_average_total_efficiency=10 ** (-0.7 / 10),
        )
        self.assertEqual(export.samples, 100)
        self.assertEqual(summary["band_points"], 81)
        self.assertEqual(summary["passed_samples"], 0)
        self.assertAlmostEqual(
            summary["nominal_minimum_total_efficiency_db"], -0.989126, places=5
        )
        self.assertAlmostEqual(
            summary["nominal_average_total_efficiency_db"],
            -0.7165826049382715,
            places=9,
        )

    def test_tolerance_is_deterministic_and_reports_yield(self):
        frequencies = np.array([1e9])
        problem = Problem(frequencies, np.array([[[0j]]]), {0: [Band(1e9, 1e9)]})
        topology = CircuitTopology(
            external_nodes=("in",), dut_nodes=("dut",),
            branches=(Branch("r", "in", "dut", LumpedModel("R5", "R", 5.0, tolerance=0.1)),),
        )
        first = monte_carlo_yield(problem, topology, YieldCriteria(0.8, 15.0), samples=40, seed=7)
        second = monte_carlo_yield(problem, topology, YieldCriteria(0.8, 15.0), samples=40, seed=7)
        self.assertTrue(np.array_equal(first.sample_scores_db, second.sample_scores_db))
        self.assertGreaterEqual(first.yield_fraction, 0.0)
        self.assertLessEqual(first.yield_fraction, 1.0)
        self.assertEqual(first.passed_samples, round(first.yield_fraction * first.samples))
        self.assertLessEqual(first.yield_confidence_interval[0], first.yield_fraction)
        self.assertGreaterEqual(first.yield_confidence_interval[1], first.yield_fraction)
        self.assertEqual(first.seed, 7)
        self.assertEqual(first.distribution, "uniform")
        self.assertEqual(set(first.score_percentiles_db), {1, 5, 50, 95, 99})

    def test_average_efficiency_criterion_is_enforced(self):
        frequencies = np.array([1e9, 2e9])
        problem = Problem(
            frequencies,
            np.array([[[0j]], [[0.8 + 0j]]]),
            {0: [Band(1e9, 2e9)]},
        )
        topology = CircuitTopology(
            external_nodes=("in",),
            dut_nodes=("dut",),
            branches=(Branch("wire", "in", "dut", LumpedModel("wire", "R", 1e-6)),),
        )
        without_average = monte_carlo_yield(
            problem,
            topology,
            YieldCriteria(minimum_total_efficiency=0.3),
            samples=5,
        )
        with_average = monte_carlo_yield(
            problem,
            topology,
            YieldCriteria(
                minimum_total_efficiency=0.3,
                minimum_average_total_efficiency=0.7,
            ),
            samples=5,
        )
        self.assertEqual(without_average.yield_fraction, 1.0)
        self.assertEqual(with_average.yield_fraction, 0.0)

    def test_measured_s2p_tolerance_uses_nominal_reactance_metadata(self):
        frequency = 1e9
        value = 5.6e-9
        impedance = 0.5 + 1j * 2 * np.pi * frequency * value
        denominator = 100.0 + impedance
        fixture = np.array([[[(impedance / denominator), 100.0 / denominator],
                             [100.0 / denominator, (impedance / denominator)]]])
        model = S2PModel(
            "measured L", np.array([frequency]), fixture,
            50.0, 0.1, "L", value,
        )
        problem = Problem(
            np.array([frequency]), np.array([[[0j]]]),
            {0: [Band(frequency, frequency)]},
        )
        topology = CircuitTopology(
            external_nodes=("in",), dut_nodes=("dut",),
            branches=(Branch("measured", "in", "dut", model),),
        )
        result = monte_carlo_yield(
            problem, topology, YieldCriteria(), samples=20, seed=11
        )
        self.assertEqual(result.samples, 20)
        self.assertTrue(all(
            0.9 <= scale <= 1.1
            for scale in result.worst_sample.values()
        ))

    def test_golden_comparison_is_per_frequency(self):
        text = "frequency_hz,port,s11_real,s11_imag,total_efficiency,component_loss\n1000000000,1,0.1,0.2,0.8,0.05\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.csv"
            path.write_text(text, encoding="utf-8")
            points = load_golden_csv(path)
        comparison = compare_golden(points, np.array([1e9]), np.array([[[0.1001 + 0.1999j]]]), np.array([[0.8002]]), np.array([[0.0501]]), s_tolerance=3e-4, efficiency_tolerance=3e-4)
        self.assertTrue(comparison.passed)
        self.assertEqual(len(comparison.rows), 1)

    def test_golden_failure_is_visible(self):
        text = "frequency_hz,port,s11_real,s11_imag,total_efficiency\n1000000000,1,0,0,0.9\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.csv"
            path.write_text(text, encoding="utf-8")
            points = load_golden_csv(path)
        comparison = compare_golden(points, np.array([1e9]), np.array([[[0.1j]]]), np.array([[0.7]]))
        self.assertFalse(comparison.passed)

    def test_complete_matrix_comparison_uses_standard_sij_orientation(self):
        text = """frequency_hz,source_port,destination_port,s_real,s_imag,total_efficiency,component_loss
1000000000,1,1,0.1,0.01,0.8,0.02
1000000000,1,2,0.2,0.02,,
1000000000,2,1,0.3,0.03,,
1000000000,2,2,0.4,0.04,0.7,0.03
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            path.write_text(text, encoding="utf-8")
            points = load_golden_csv(path)
        # Matrix indexing is [destination/output, source/driven].
        s_parameters = np.array([[[0.1 + 0.01j, 0.3 + 0.03j],
                                  [0.2 + 0.02j, 0.4 + 0.04j]]])
        comparison = compare_golden(
            points,
            np.array([1e9]),
            s_parameters,
            np.array([[0.8, 0.7]]),
            np.array([[0.02, 0.03]]),
        )
        self.assertTrue(comparison.passed)
        self.assertEqual([row["s_parameter"] for row in comparison.rows], ["S11", "S21", "S12", "S22"])

        reversed_matrix = s_parameters.copy()
        reversed_matrix[0, 0, 1], reversed_matrix[0, 1, 0] = (
            reversed_matrix[0, 1, 0],
            reversed_matrix[0, 0, 1],
        )
        failed = compare_golden(
            points,
            np.array([1e9]),
            reversed_matrix,
            np.array([[0.8, 0.7]]),
            np.array([[0.02, 0.03]]),
        )
        self.assertFalse(failed.passed)
        self.assertGreater(failed.maximum_s_error, 0.1)


if __name__ == "__main__":
    unittest.main()
