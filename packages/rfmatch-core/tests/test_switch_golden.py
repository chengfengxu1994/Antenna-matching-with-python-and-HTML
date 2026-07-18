import tempfile
import unittest
from pathlib import Path

from rfmatch_core.switch_golden import compare_switch_export, load_switch_export_csv


class SwitchGoldenTests(unittest.TestCase):
    def test_loads_common_optenni_labels_and_db_efficiency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "set1.csv"
            path.write_text(
                "Frequency [GHz],S11 [dB],Total efficiency [dB]\n"
                "1.0,-10,-1\n1.1,-12,-2\n",
                encoding="utf-8",
            )
            points = load_switch_export_csv(path, default_configuration="Set 1")
        self.assertEqual([point.frequency_hz for point in points], [1e9, 1.1e9])
        self.assertAlmostEqual(points[0].total_efficiency, 10 ** -0.1)

    def test_loads_native_optenni_43_tab_delimited_plot_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.txt"
            path.write_text(
                '% PlotTitle: "Matching circuit, port 1"\n'
                '% X axis: "Frequency [GHz]"\n'
                '% Y axis: "S parameters/loss [dB]"\n'
                '"Frequency [GHz]"\t"S11"\t"Total efficiency"\n'
                '1.0\t-10\t-1\n1.1\t-12\t-2\n',
                encoding="utf-8",
            )
            points = load_switch_export_csv(path, default_configuration="PCSL")
        self.assertEqual([point.frequency_hz for point in points], [1e9, 1.1e9])
        self.assertEqual([point.s11_db for point in points], [-10.0, -12.0])
        self.assertAlmostEqual(points[0].total_efficiency, 10 ** -0.1)

    def test_comparison_reports_interpolated_db_and_efficiency_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.csv"
            path.write_text(
                "configuration,frequency_mhz,s11_db,total_efficiency_pct\n"
                "Set 1,1050,-11.02,79\n",
                encoding="utf-8",
            )
            points = load_switch_export_csv(path)
        report = compare_switch_export(points, [{
            "configuration": "Set 1",
            "frequency_hz": [1e9, 1.1e9],
            "s11_db": [-10.0, -12.0],
            "total_efficiency": [0.8, 0.78],
        }], s11_tolerance_db=0.05, efficiency_tolerance=0.005)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["maximum_s11_error_db"], 0.02)
        self.assertAlmostEqual(report["maximum_efficiency_error"], 0.0)

    def test_rejects_duplicate_configuration_frequency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text(
                "configuration,frequency_hz,s11_db\n"
                "Set 1,1000000000,-10\nSet 1,1000000000,-11\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_switch_export_csv(path)


if __name__ == "__main__":
    unittest.main()
