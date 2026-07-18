import os
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from engine.core_adapter import to_core_problem, to_core_touchstone
from engine.efficiency_data import parse_efficiency_data
from engine.touchstone import parse_touchstone
from rfmatch_core.touchstone import load_touchstone
from rfmatch_core.network import s_to_z


TUTORIAL_ROOT = Path(os.environ.get("OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"))


class CoreAdapterTests(unittest.TestCase):
    def test_product_parser_uses_authoritative_column_major_order(self):
        product = parse_touchstone(
            "# GHz S RI R 50\n1.0 0.11 0 0.21 0 0.12 0 0.22 0",
            filename="nonreciprocal.s2p",
        )
        matrix = product.get_s_matrix(0)
        np.testing.assert_allclose(matrix, [[0.11, 0.12], [0.21, 0.22]])

    def test_product_parser_propagates_strict_core_validation(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_touchstone(
                "# GHZ S RI R 75\n2.0 0.1 0\n1.0 0.2 0",
                filename="invalid.s1p",
            )

    def test_product_adapter_preserves_real_per_port_reference_impedances(self):
        product = parse_touchstone(
            "[Version] 2.0\n[Number of Ports] 2\n[Reference] 50 75\n"
            "# GHZ S RI R 50\n[Network Data]\n"
            "1 0.1 0 0.2 0 0.3 0 0.4 0\n[End]",
            filename="per-port.ts",
        )
        converted = to_core_touchstone(product)
        np.testing.assert_array_equal(converted.z0, [50.0, 75.0])
        np.testing.assert_allclose(
            product.get_z_matrix(0),
            s_to_z(product.get_s_matrix(0), np.array([50.0, 75.0])),
            atol=1e-14,
        )

    def _legacy(self, relative: str):
        path = TUTORIAL_ROOT / relative
        return parse_touchstone(path.read_text(encoding="utf-8", errors="replace"), filename=path.name)

    def test_single_port_conversion_matches_core_parser(self):
        relative = "1 - START HERE/measured_antenna.s1p"
        converted = to_core_touchstone(self._legacy(relative))
        direct = load_touchstone(TUTORIAL_ROOT / relative)
        np.testing.assert_allclose(converted.frequencies_hz, direct.frequencies_hz)
        np.testing.assert_allclose(converted.s_parameters, direct.s_parameters, atol=1e-14)
        self.assertEqual(converted.z0, direct.z0)

    def test_three_port_conversion_preserves_full_coupling_matrix(self):
        relative = "4 - Multiantenna system/4.1 Multiantenna system at different bands/3_antennas.s3p"
        converted = to_core_touchstone(self._legacy(relative))
        direct = load_touchstone(TUTORIAL_ROOT / relative)
        np.testing.assert_allclose(converted.s_parameters, direct.s_parameters, atol=1e-14)

    def test_efficiency_is_interpolated_onto_problem_grid(self):
        legacy = self._legacy("9 - Radiation efficiency/Radiation_Efficiency_Tutorial.s1p")
        efficiency = parse_efficiency_data("880 0.5\n2400 0.8", freq_unit="mhz", eff_format="linear")
        problem = to_core_problem(legacy, {0: [(880e6, 960e6)]}, {0: efficiency})
        self.assertEqual(problem.radiation_efficiency[0].shape, problem.frequencies_hz.shape)
        self.assertTrue(np.all((problem.radiation_efficiency[0] >= 0.5) & (problem.radiation_efficiency[0] <= 0.8)))


if __name__ == "__main__":
    unittest.main()
