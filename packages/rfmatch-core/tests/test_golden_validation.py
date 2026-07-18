import csv
from pathlib import Path
import tempfile
import unittest

from rfmatch_core.golden import load_golden_csv, validate_golden_points


FIELDS = ["frequency_hz", "port", "s11_real", "s11_imag", "total_efficiency", "component_loss"]


class GoldenValidationTests(unittest.TestCase):
    def _write(self, rows):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "golden.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(directory.cleanup)
        return path

    def test_valid_export_has_a_machine_readable_summary(self):
        points = load_golden_csv(self._write([
            {"frequency_hz": 1e9, "port": 1, "s11_real": 0.1, "s11_imag": 0.2, "total_efficiency": 0.8, "component_loss": 0.03},
            {"frequency_hz": 1.1e9, "port": 1, "s11_real": 0.2, "s11_imag": 0.1, "total_efficiency": 0.75, "component_loss": 0.04},
        ]))
        summary = validate_golden_points(points)
        self.assertEqual(summary.rows, 2)
        self.assertEqual(summary.ports, (0,))
        self.assertEqual(summary.s_parameter_pairs, ((0, 0),))
        self.assertTrue(summary.includes_complete_s_matrix)
        self.assertTrue(summary.includes_total_efficiency)
        self.assertTrue(summary.includes_component_loss)

    def test_duplicate_frequency_and_port_is_rejected(self):
        path = self._write([
            {"frequency_hz": 1e9, "port": 1, "s11_real": 0, "s11_imag": 0, "total_efficiency": 0.8},
            {"frequency_hz": 1e9, "port": 1, "s11_real": 0, "s11_imag": 0, "total_efficiency": 0.8},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate golden row"):
            load_golden_csv(path)

    def test_invalid_efficiency_is_rejected(self):
        path = self._write([
            {"frequency_hz": 1e9, "port": 1, "s11_real": 0, "s11_imag": 0, "total_efficiency": 80},
        ])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            load_golden_csv(path)

    def test_complete_matrix_schema_reports_full_coverage(self):
        text = """frequency_hz,source_port,destination_port,s_real,s_imag,total_efficiency,component_loss
1000000000,1,1,0.1,0.01,0.8,0.02
1100000000,1,1,0.2,0.02,0.7,0.03
1000000000,1,2,0.3,0.03,,
1100000000,1,2,0.4,0.04,,
1000000000,2,1,0.5,0.05,,
1100000000,2,1,0.6,0.06,,
1000000000,2,2,0.7,0.07,0.6,0.04
1100000000,2,2,0.8,0.08,0.5,0.05
"""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "matrix.csv"
        path.write_text(text, encoding="utf-8")
        summary = validate_golden_points(load_golden_csv(path))
        self.assertEqual(summary.ports, (0, 1))
        self.assertEqual(summary.s_parameter_pairs, ((0, 0), (0, 1), (1, 0), (1, 1)))
        self.assertTrue(summary.includes_complete_s_matrix)
        self.assertTrue(summary.includes_total_efficiency)
        self.assertTrue(summary.includes_component_loss)

    def test_inconsistent_repeated_driven_port_metric_is_rejected(self):
        text = """frequency_hz,source_port,destination_port,s_real,s_imag,total_efficiency
1000000000,1,1,0,0,0.8
1000000000,1,2,0,0,0.7
"""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "inconsistent.csv"
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inconsistent total_efficiency"):
            load_golden_csv(path)


if __name__ == "__main__":
    unittest.main()
