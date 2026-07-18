import json
from pathlib import Path

from scripts.run_touchstone_format_benchmark import run


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "benchmarks" / "touchstone-format-baseline.json"


def _assert_format_equivalence(payload):
    assert payload["schema_version"] == 2
    assert payload["reference_impedance_ohm"] == 75.0
    assert payload["formats"] == ["RI", "MA", "DB"]
    assert payload["maximum_power_balance_error"] < 1e-12
    for comparison in payload["comparisons_against_ri"].values():
        assert comparison["input_s_maximum_complex_error"] < 2e-15
        assert comparison["matched_s_maximum_complex_error"] < 2e-14
        assert comparison["total_efficiency_maximum_absolute_error"] < 2e-14
        assert comparison["score_db_absolute_error"] < 2e-12
    per_port = payload["per_port_reference"]
    assert per_port["impedances_ohm"] == [50.0, 90.0]
    assert per_port["renormalization_round_trip_maximum_complex_error"] < 3e-15
    assert per_port["physical_power_balance_error"] < 1e-12
    assert abs(per_port["score_delta_from_incorrect_scalar_db"]) > 1e-4
    semantics = payload["touchstone_2_semantics"]
    assert semantics["three_port_row_major_maximum_complex_error"] == 0.0
    assert semantics["multiline_reference_impedances_ohm"] == [50.0, 75.0, 90.0]
    assert semantics["two_port_12_21_s12"] == [0.12, 0.01]
    assert semantics["two_port_12_21_s21"] == [0.21, -0.02]
    assert semantics["dc_frequency_hz"] == 0.0


def test_committed_touchstone_format_baseline_is_present_and_physical():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    _assert_format_equivalence(payload)


def test_touchstone_format_baseline_recomputes_without_drift():
    current = run()
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    _assert_format_equivalence(current)
    assert current == committed
