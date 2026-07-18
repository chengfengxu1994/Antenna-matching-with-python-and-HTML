from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rfmatch_core.benchmarks import run_switch_tutorial_case


TUTORIAL_ROOT = Path(r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials")
CASE_ROOT = TUTORIAL_ROOT / "10 - Tunable antennas/10.6 Impedance tuning using a switch"
ARTIFACT = Path(__file__).resolve().parents[3] / "artifacts/benchmarks/optenni-switch-tuning-baseline.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_switch_tutorial_reference_topologies_and_curves():
    required = (
        CASE_ROOT / "Switch_Tuner_Tutorial.s1p",
        CASE_ROOT / "SP3T_ideal.mdif",
        CASE_ROOT / "tutorial_SP2T.mdif",
        CASE_ROOT / "Impedance tuning using a switch Tutorial.pdf",
    )
    if not all(path.exists() for path in required):
        pytest.skip("licensed Optenni switch tutorial inputs are not installed")

    stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))["results"][0]
    actual = run_switch_tutorial_case(TUTORIAL_ROOT, "switch-tuning")

    assert stored["reference_source"]["sha256"] == _sha256(required[-1])
    assert actual["reference_source"]["verified_pages"] == [12, 13, 16]
    assert [item["design_role"] for item in actual["variants"]] == [
        "best_performance",
        "simplified_near_best",
        "reduced_bom_near_equivalent",
    ]
    assert actual["variants"][0]["input_series"] == [
        {"kind": "L", "value_si": 15e-9},
        {"kind": "C", "value_si": 2e-12},
    ]
    assert actual["variants"][0]["branches"] == [
        {"throw": 1, "kind": "L", "value_si": 1e-9},
        {"throw": 2, "kind": "C", "value_si": 2.6e-12},
        {"throw": 3, "kind": "C", "value_si": 1.2e-12},
    ]

    for actual_variant, stored_variant in zip(actual["variants"], stored["variants"], strict=True):
        assert actual_variant["name"] == stored_variant["name"]
        assert actual_variant["input_series"] == stored_variant["input_series"]
        assert actual_variant["branches"] == stored_variant["branches"]
        for actual_configuration, stored_configuration in zip(
            actual_variant["configurations"], stored_variant["configurations"], strict=True
        ):
            assert actual_configuration["name"] == stored_configuration["name"]
            assert actual_configuration["state"] == stored_configuration["state"]
            assert actual_configuration["minimum_return_loss_db"] == pytest.approx(
                stored_configuration["minimum_return_loss_db"], abs=1e-9
            )
            assert actual_configuration["mean_return_loss_db"] == pytest.approx(
                stored_configuration["mean_return_loss_db"], abs=1e-9
            )
            assert actual_configuration["center_return_loss_db"] == pytest.approx(
                stored_configuration["center_return_loss_db"], abs=1e-9
            )
