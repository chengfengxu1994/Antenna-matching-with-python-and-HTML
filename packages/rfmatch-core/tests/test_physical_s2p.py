import unittest
from pathlib import Path

import numpy as np

from rfmatch_core import Branch, CircuitTopology, LumpedModel, S2PModel, evaluate_circuit, load_s2p_model


class PhysicalS2PTests(unittest.TestCase):
    @staticmethod
    def _series_fixture(impedance, z0=50.0):
        denominator = 2.0 * z0 + impedance
        reflection = impedance / denominator
        transmission = 2.0 * z0 / denominator
        return np.array([[[reflection, transmission], [transmission, reflection]]])

    def test_matched_attenuator_power_is_conserved_and_loss_is_exact(self):
        frequency = 1e9
        transmission = 10 ** (-3.0 / 20.0)
        attenuator_s = np.array([[[0, transmission], [transmission, 0]]], dtype=complex)
        model = S2PModel("3dB", np.array([frequency]), attenuator_s)
        topology = CircuitTopology(
            external_nodes=("in",),
            dut_nodes=("dut",),
            branches=(Branch("attenuator", "in", "dut", model),),
        )
        solved = evaluate_circuit(np.array([[0j]]), topology, frequency)
        expected_transmitted = transmission ** 2
        self.assertAlmostEqual(abs(solved.s_parameters[0, 0]), 0.0, places=10)
        self.assertAlmostEqual(solved.component_loss[0, 0], 1.0 - expected_transmitted, places=10)
        self.assertAlmostEqual(solved.dut_absorbed_power[0], expected_transmitted, places=10)
        self.assertLess(abs(solved.power_balance_error[0]), 1e-12)

    def test_physical_internal_node_series_element(self):
        topology = CircuitTopology(
            external_nodes=("in",),
            dut_nodes=("dut",),
            branches=(Branch("r", "in", "dut", LumpedModel("R10", "R", 10.0)),),
        )
        solved = evaluate_circuit(np.array([[0j]]), topology, 1e9)
        expected_gamma = 10.0 / 110.0
        self.assertAlmostEqual(solved.s_parameters[0, 0].real, expected_gamma, places=10)
        self.assertAlmostEqual(np.sum(solved.component_loss) + solved.dut_absorbed_power[0] + abs(solved.s_parameters[0, 0]) ** 2, 1.0, places=10)

    def test_real_tutorial_component_s2p_is_usable(self):
        path = Path(r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\6 - Custom component libraries\Custom component library\C_01_5.s2p")
        model = load_s2p_model(path)
        topology = CircuitTopology(external_nodes=("in",), dut_nodes=("dut",), branches=(Branch("c", "in", "dut", model),))
        solved = evaluate_circuit(np.array([[0j]]), topology, 1e9)
        accounted = abs(solved.s_parameters[0, 0]) ** 2 + solved.component_loss[0, 0] + solved.dut_absorbed_power[0]
        self.assertAlmostEqual(accounted, 1.0, places=8)

    def test_s2p_component_can_be_grounded_as_a_shunt(self):
        resistance = 25.0
        z0 = 50.0
        normalized = resistance / z0
        reflection = normalized / (2.0 + normalized)
        transmission = 2.0 / (2.0 + normalized)
        model = S2PModel(
            "R25 fixture",
            np.array([1e9]),
            np.array([[[reflection, transmission], [transmission, reflection]]]),
            z0,
        )
        s_dut = np.array([[0.0j]])
        measured = CircuitTopology(
            external_nodes=("p1",),
            dut_nodes=("p1",),
            branches=(Branch("measured", "p1", None, model),),
        )
        lumped = CircuitTopology(
            external_nodes=("p1",),
            dut_nodes=("p1",),
            branches=(Branch("lumped", "p1", None, LumpedModel("R25", "R", resistance)),),
        )
        measured_result = evaluate_circuit(s_dut, measured, 1e9, z0)
        lumped_result = evaluate_circuit(s_dut, lumped, 1e9, z0)
        self.assertTrue(np.allclose(measured_result.s_parameters, lumped_result.s_parameters, atol=1e-12))
        self.assertTrue(np.allclose(measured_result.component_loss, lumped_result.component_loss, atol=1e-12))

    def test_dut_node_can_alias_matching_external_port(self):
        s_dut = np.array([[0.2 + 0.1j]])
        topology = CircuitTopology(external_nodes=("p1",), dut_nodes=("p1",), branches=())
        result = evaluate_circuit(s_dut, topology, 1e9)
        self.assertTrue(np.allclose(result.s_parameters, s_dut, atol=1e-12))

    def test_measured_inductor_variation_matches_analytic_series_value(self):
        frequency = 1e9
        value = 5.6e-9
        resistance = 0.7
        impedance = resistance + 1j * 2 * np.pi * frequency * value
        measured_model = S2PModel(
            "measured L", np.array([frequency]), self._series_fixture(impedance),
            50.0, 0.1, "L", value,
        )
        measured = CircuitTopology(
            external_nodes=("in",), dut_nodes=("dut",),
            branches=(Branch("l", "in", "dut", measured_model),),
        )
        analytic = CircuitTopology(
            external_nodes=("in",), dut_nodes=("dut",),
            branches=(Branch(
                "l", "in", "dut",
                LumpedModel("analytic L", "L", value, esr=resistance),
            ),),
        )
        varied_measured = evaluate_circuit(
            np.array([[0j]]), measured, frequency, variation={"l": 1.1}
        )
        varied_analytic = evaluate_circuit(
            np.array([[0j]]), analytic, frequency, variation={"l": 1.1}
        )
        self.assertTrue(np.allclose(
            varied_measured.s_parameters, varied_analytic.s_parameters, atol=1e-12
        ))
        self.assertTrue(np.allclose(
            varied_measured.component_loss, varied_analytic.component_loss, atol=1e-12
        ))


if __name__ == "__main__":
    unittest.main()
