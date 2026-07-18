from pathlib import Path

import pytest

from rfmatch_core.benchmarks import run_tunable_variable_capacitor_case


TUTORIAL_ROOT = Path(r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials")
COMPONENT_ROOT = Path(r"C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary")


def test_official_variable_capacitor_recall_when_installed():
    if not TUTORIAL_ROOT.exists() or not COMPONENT_ROOT.exists():
        pytest.skip("licensed Optenni reference inputs are not installed")
    result = run_tunable_variable_capacitor_case(
        TUTORIAL_ROOT, "variable-capacitor", COMPONENT_ROOT
    )
    assert result["state_by_configuration"] == {
        "Set 1": "8 pF",
        "Set 2": "2 pF",
        "Set 3": "1 pF",
    }
    assert result["fixed_elements"][0]["name"] == "GJM1555C1H2R8WB01"
    assert result["fixed_elements"][1]["name"] == "04CS15N"
    assert result["score_db"] == pytest.approx(-1.640599673090044, abs=1e-9)
    assert result["maximum_power_balance_error"] < 1e-12
