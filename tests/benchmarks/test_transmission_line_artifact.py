import json
from pathlib import Path

from scripts.run_transmission_line_benchmark import run


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "benchmarks" / "transmission-line-physical-baseline.json"


def _assert_physical_errors(payload):
    assert payload["lossless_matched_line"]["maximum_reflection_magnitude"] < 1e-12
    assert payload["lossless_matched_line"]["forward_complex_error"] < 1e-12
    assert payload["lossy_matched_line"]["maximum_power_balance_error"] < 1e-12
    assert max(
        point["delivered_power_error"]
        for point in payload["lossy_matched_line"]["points"]
    ) < 1e-12
    assert payload["quarter_wave_transform"]["reflection_complex_error"] < 1e-12
    assert payload["shunt_stubs"]["maximum_analytic_error"] < 1e-12
    automatic = payload["automatic_synthesis"]
    assert automatic["topology"] == "through_line"
    assert automatic["reflection_magnitude"] < 2e-4
    assert abs(
        automatic["recovered_characteristic_impedance_ohm"]
        - automatic["expected_characteristic_impedance_ohm"]
    ) < 0.02
    assert abs(
        automatic["recovered_electrical_length_deg"]
        - automatic["expected_electrical_length_deg"]
    ) < 0.02
    assert automatic["maximum_power_balance_error"] < 1e-12
    assert automatic["evaluations"] <= 5000


def test_committed_transmission_line_baseline_is_present_and_physical():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    _assert_physical_errors(payload)


def test_transmission_line_baseline_recomputes_without_drift():
    current = run()
    _assert_physical_errors(current)
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for section in (
        "lossless_matched_line", "lossy_matched_line",
        "quarter_wave_transform", "shunt_stubs", "automatic_synthesis",
    ):
        assert current[section] == committed[section]
