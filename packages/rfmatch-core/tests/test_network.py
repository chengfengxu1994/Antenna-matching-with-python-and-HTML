import unittest

import numpy as np

from rfmatch_core.models import Element
from rfmatch_core.network import (
    apply_elements, apply_elements_sweep, cascade_s2p, deembed_s2p,
    flip_s2p_ports, renormalize_s_parameters, s_to_z, terminate, z_to_s,
)


class NetworkTests(unittest.TestCase):
    def test_batched_sweep_matches_point_by_point_application(self):
        frequencies = np.array([0.9e9, 1.0e9, 1.1e9])
        s = np.array([
            [[0.2 + 0.1j, 0.05j], [0.05j, -0.1 + 0.2j]],
            [[0.25 + 0.05j, 0.04j], [0.04j, -0.05 + 0.15j]],
            [[0.3 + 0.02j, 0.03j], [0.03j, -0.02 + 0.1j]],
        ])
        elements = [Element("shunt", "C", 0, 1.2e-12), Element("series", "L", 1, 3.3e-9)]
        expected = np.asarray([apply_elements(matrix, elements, frequency) for matrix, frequency in zip(s, frequencies)])
        actual = apply_elements_sweep(s, elements, frequencies)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_round_trip_non_50_ohm(self):
        s = np.array([[0.2 + 0.1j, 0.05], [0.05, -0.1j]])
        self.assertTrue(np.allclose(z_to_s(s_to_z(s, 75.0), 75.0), s, atol=1e-12))

    def test_termination_returns_original_port_map(self):
        s = np.diag([0.1, 0.2, 0.3]).astype(complex)
        reduced, ports = terminate(s, {0: 0.0, 2: 0.0})
        self.assertEqual(ports, [1])
        self.assertAlmostEqual(reduced[0, 0], 0.2)

    def test_element_on_nonzero_port(self):
        s = np.zeros((3, 3), dtype=complex)
        out = apply_elements(s, [Element("series", "L", 2, 10e-9)], 1e9, 50.0)
        self.assertAlmostEqual(abs(out[0, 0]), 0.0)
        self.assertGreater(abs(out[2, 2]), 0.0)

    def test_shunt_respects_reference_impedance(self):
        s = np.array([[0.25 + 0.1j]])
        element = [Element("shunt", "C", 0, 2e-12)]
        self.assertFalse(np.allclose(apply_elements(s, element, 2e9, 50.0), apply_elements(s, element, 2e9, 75.0)))

    def test_s2p_port_flip_swaps_reflections_and_transmissions(self):
        s = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex)
        np.testing.assert_array_equal(
            flip_s2p_ports(s), np.array([[0.4, 0.3], [0.2, 0.1]])
        )

    def test_power_wave_renormalization_round_trips_a_sweep(self):
        s = np.array([
            [[0.2 + 0.1j, 0.5 - 0.1j], [0.45 + 0.05j, -0.1j]],
            [[0.1 - 0.05j, 0.6], [0.55, -0.2 + 0.03j]],
        ])
        at_75 = renormalize_s_parameters(s, 50.0, 75.0)
        np.testing.assert_allclose(
            renormalize_s_parameters(at_75, 75.0, 50.0), s, atol=2e-15
        )

    def test_per_port_power_wave_renormalization_round_trips(self):
        s = np.array([
            [[0.2 + 0.1j, 0.5 - 0.1j], [0.45 + 0.05j, -0.1j]],
            [[0.1 - 0.05j, 0.6], [0.55, -0.2 + 0.03j]],
        ])
        source = np.array([50.0, 75.0])
        target = np.array([60.0, 90.0])
        converted = renormalize_s_parameters(s, source, target)
        np.testing.assert_allclose(
            renormalize_s_parameters(converted, target, source), s, atol=3e-15
        )

    def test_per_port_renormalization_matches_s_to_z_reference(self):
        s = np.array([[0.2 + 0.1j, 0.4], [0.35, -0.1 + 0.05j]])
        source = np.array([50.0, 75.0])
        target = np.array([60.0, 90.0])
        expected = z_to_s(s_to_z(s, source), target)
        np.testing.assert_allclose(
            renormalize_s_parameters(s, source, target), expected, atol=4e-16
        )

    def test_renormalization_handles_ideal_through_without_z_singularity(self):
        through = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        np.testing.assert_allclose(
            renormalize_s_parameters(through, 50.0, 75.0), through, atol=1e-15
        )

    def test_two_sided_fixture_deembedding_recovers_dut_sweep(self):
        left = np.array([[[0.05, 0.88], [0.9, -0.03]]], dtype=complex)
        dut = np.array([[[0.25 + 0.1j, 0.65], [0.7, -0.15j]]], dtype=complex)
        right = np.array([[[-0.04, 0.91], [0.89, 0.02]]], dtype=complex)
        measured = cascade_s2p(left, dut, right)
        recovered, diagnostics = deembed_s2p(
            measured, left_fixture=left, right_fixture=right,
        )
        np.testing.assert_allclose(recovered, dut, atol=2e-15)
        self.assertLess(diagnostics["maximum_recascade_residual"], 2e-15)

    def test_deembedding_rejects_fixture_with_zero_forward_transmission(self):
        measured = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        blocked = np.zeros((2, 2), dtype=complex)
        with self.assertRaisesRegex(ValueError, "S21"):
            deembed_s2p(measured, left_fixture=blocked)


if __name__ == "__main__":
    unittest.main()
