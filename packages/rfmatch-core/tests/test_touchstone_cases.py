import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rfmatch_core.benchmarks import CASES, MULTIPORT_CASES
from rfmatch_core.touchstone import load_touchstone, parse_touchstone_text


TUTORIAL_ROOT = Path(os.environ.get("OPTENNI_TUTORIAL_ROOT", r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"))


def _touchstone_text(matrices: np.ndarray, fmt: str, *, z0: float = 75.0) -> str:
    rows = [f"# GHZ S {fmt} R {z0:g}"]
    for frequency_ghz, matrix in zip((1.0, 2.0), matrices):
        values = []
        for destination in range(matrix.shape[0]):
            for source in range(matrix.shape[1]):
                value = matrix[destination, source]
                if fmt == "RI":
                    pair = (value.real, value.imag)
                elif fmt == "MA":
                    pair = (abs(value), np.rad2deg(np.angle(value)))
                else:
                    pair = (20.0 * np.log10(abs(value)), np.rad2deg(np.angle(value)))
                values.extend(f"{item:.15g}" for item in pair)
        midpoint = 10
        rows.append(f"{frequency_ghz:g} " + " ".join(values[:midpoint]) + " ! inline 12345")
        rows.append(" ".join(values[midpoint:]))
    return "\n".join(rows)


class TutorialCaseTests(unittest.TestCase):
    def test_reference_inputs_are_readable(self):
        for _, data_path, _ in CASES.values():
            data = load_touchstone(TUTORIAL_ROOT / data_path)
            self.assertGreater(len(data.frequencies_hz), 2)
            self.assertEqual(data.s_parameters.shape[1:], (1, 1))

    def test_multiport_reference_inputs_are_readable(self):
        for data_path, bands_by_port in MULTIPORT_CASES.values():
            data = load_touchstone(TUTORIAL_ROOT / data_path)
            self.assertGreater(len(data.frequencies_hz), 2)
            self.assertEqual(data.s_parameters.shape[1:], (3, 3))
            self.assertEqual(set(bands_by_port), {0, 1, 2})

    def test_three_port_ri_ma_db_and_non_50_ohm_are_numerically_equivalent(self):
        base = np.asarray([
            [
                [0.20 + 0.05j, 0.08 - 0.02j, 0.04 + 0.01j],
                [0.06 + 0.03j, 0.18 - 0.04j, 0.07 + 0.02j],
                [0.03 - 0.01j, 0.05 + 0.04j, 0.15 + 0.06j],
            ],
            [
                [0.22 + 0.04j, 0.07 - 0.03j, 0.05 + 0.02j],
                [0.05 + 0.02j, 0.17 - 0.03j, 0.06 + 0.01j],
                [0.02 - 0.02j, 0.04 + 0.03j, 0.14 + 0.05j],
            ],
        ], dtype=complex)
        parsed = {
            fmt: parse_touchstone_text(_touchstone_text(base, fmt), f"equivalent-{fmt}.s3p")
            for fmt in ("RI", "MA", "DB")
        }
        for fmt, data in parsed.items():
            self.assertEqual(data.z0, 75.0)
            self.assertEqual(data.data_format, fmt)
            np.testing.assert_allclose(data.frequencies_hz, [1e9, 2e9])
            np.testing.assert_allclose(data.s_parameters, base, atol=2e-15)
            self.assertIn("inline 12345", data.comments)

    def test_touchstone_2_network_section_and_uniform_reference(self):
        text = """[Version] 2.0
[Number of Ports] 2
[Number of Frequencies] 1
[Reference] 75 75
# GHZ S RI R 50
[Network Data]
1.0 0.1 0 0.2 0 0.3 0 0.4 0
[End]
"""
        data = parse_touchstone_text(text, "network.ts")
        self.assertEqual(data.z0, 75.0)
        np.testing.assert_allclose(data.s_parameters[0], [[0.1, 0.3], [0.2, 0.4]])

    def test_touchstone_2_two_port_12_21_order_is_not_transposed(self):
        text = """[Version] 2.0
# GHZ S RI R 50
[Number of Ports] 2
[Two-Port Data Order] 12_21
[Number of Frequencies] 1
[Network Data]
1.0 0.1 0 0.12 0.01 0.21 -0.02 0.4 0
[End]
"""
        data = parse_touchstone_text(text, "ordered.ts")
        np.testing.assert_allclose(
            data.s_parameters[0],
            [[0.1, 0.12 + 0.01j], [0.21 - 0.02j, 0.4]],
        )

    def test_touchstone_2_multiline_references_and_triangular_matrices(self):
        common = """[Version] 2.0
# GHZ S RI R 50
[Number of Ports] 3
[Number of Frequencies] 1
[Reference] 50
75 90
[Matrix Format] {matrix_format}
[Network Data]
1.0 {values}
[End]
"""
        lower = parse_touchstone_text(common.format(
            matrix_format="Lower",
            values="1 0 0.21 0.01 2 0 0.31 0.02 0.32 0.03 3 0",
        ), "lower.ts")
        upper = parse_touchstone_text(common.format(
            matrix_format="Upper",
            values="1 0 0.21 0.01 0.31 0.02 2 0 0.32 0.03 3 0",
        ), "upper.ts")
        expected = np.asarray([
            [1, 0.21 + 0.01j, 0.31 + 0.02j],
            [0.21 + 0.01j, 2, 0.32 + 0.03j],
            [0.31 + 0.02j, 0.32 + 0.03j, 3],
        ])
        np.testing.assert_array_equal(lower.z0, [50.0, 75.0, 90.0])
        np.testing.assert_allclose(lower.s_parameters[0], expected)
        np.testing.assert_allclose(upper.s_parameters[0], expected)

    def test_dc_touchstone_point_is_accepted_for_file_compatibility(self):
        data = parse_touchstone_text(
            "# HZ S RI R 50\n0 0.1 0\n10000000 0.2 0",
            "dc.s1p",
        )
        np.testing.assert_array_equal(data.frequencies_hz, [0.0, 1e7])

    def test_invalid_touchstone_fails_instead_of_silently_changing_meaning(self):
        invalid = (
            ("# GHZ Z RI R 50\n1 1 0", "only S-parameter"),
            ("# GHZ S RI R 50\n2 0.1 0\n1 0.2 0", "strictly increasing"),
            ("# GHZ S RI R 50\n-1 0.1 0\n1 0.2 0", "non-negative"),
            ("# GHZ S RI R 50\n1 0.1", "incomplete"),
            ("# GHZ S RI R 50\n1 nan 0", "finite"),
            (
                "[Version] 2.0\n# GHZ S RI R 50\n[Number of Ports] 2\n"
                "[Reference] 50\n[Network Data]\n1 0 0 0 0 0 0 0 0",
                "exactly one real value per port",
            ),
            (
                "[Version] 2.0\n# GHZ S RI R 50\n[Number of Ports] 1\n"
                "[Number of Frequencies] 2\n[Network Data]\n1 0 0",
                "declares 2, found 1",
            ),
            (
                "[Version] 2.0\n# GHZ S RI R 50\n[Number of Ports] 1\n"
                "[Reference] 50+j10\n[Network Data]\n1 0 0",
                "only real-valued",
            ),
            (
                "[Version] 2.0\n# GHZ S RI R 50\n[Number of Ports] 2\n"
                "[Mixed-Mode Order] D1,2 C1,2\n[Network Data]\n1 0 0 0 0 0 0 0 0",
                "mixed-mode",
            ),
        )
        for text, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_touchstone_text(
                        text,
                        "invalid.ts" if "[Number of Ports]" in text else "invalid.s1p",
                    )
        per_port = parse_touchstone_text(
            "[Version] 2.0\n[Number of Ports] 2\n[Reference] 50 75\n"
            "# GHZ S RI R 50\n[Network Data]\n"
            "1 0.1 0 0.2 0 0.3 0 0.4 0\n[End]",
            "per-port.ts",
        )
        np.testing.assert_array_equal(per_port.z0, [50.0, 75.0])


if __name__ == "__main__":
    unittest.main()
