from pathlib import Path
import unittest

import numpy as np

from rfmatch_core import Branch, CircuitTopology, evaluate_circuit, load_s2p_model, load_touchstone
from rfmatch_core.switch_golden import load_switch_export_csv


class OptenniNativeExportTests(unittest.TestCase):
    def test_exported_pcsl_network_reproduces_native_plot(self):
        root = Path(__file__).resolve().parents[3] / "benchmarks" / "optenni_exports"
        dut = load_touchstone(root / "optimization_settings_original.s1p")
        model = load_s2p_model(root / "optimization_settings_pcsl_circuit.s2p")
        golden = load_switch_export_csv(
            root / "optimization_settings_pcsl_plot.txt",
            default_configuration="PCSL",
        )
        self.assertTrue(np.allclose(
            dut.frequencies_hz,
            np.asarray([point.frequency_hz for point in golden]),
            rtol=0.0,
            atol=1e-6,
        ))
        topology = CircuitTopology(
            external_nodes=("input",),
            dut_nodes=("dut",),
            branches=(Branch("optenni_pcsl", "input", "dut", model),),
        )
        predicted_db = np.asarray([
            20.0 * np.log10(max(abs(evaluate_circuit(
                dut.s_parameters[index], topology, point.frequency_hz,
            ).s_parameters[0, 0]), 1e-15))
            for index, point in enumerate(golden)
        ])
        exported_db = np.asarray([point.s11_db for point in golden])
        error_db = np.abs(predicted_db - exported_db)
        self.assertLess(float(np.max(error_db)), 1e-3)
        self.assertLess(float(np.sqrt(np.mean(error_db ** 2))), 1e-4)


if __name__ == "__main__":
    unittest.main()
