import json
from pathlib import Path

from scripts.run_microstrip_geometry_benchmark import run


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "benchmarks" / "microstrip-geometry-baseline.json"


def _assert_baseline(payload):
    assert payload["schema_version"] == 1
    assert max(item["impedance_error_ohm"] for item in payload["cross_validation"]) < 1e-12
    assert max(item["effective_permittivity_error"] for item in payload["cross_validation"]) < 1e-12
    quarter = payload["fifty_ohm_quarter_wave"]
    assert abs(quarter["characteristic_impedance_ohm"] - 50.0) < 1e-6
    assert quarter["conductor_loss_db"] > 0
    assert quarter["dielectric_loss_db"] > 0
    assert abs(quarter["delivered_power"] + quarter["component_loss"] - 1.0) < 1e-10
    assert quarter["power_balance_error"] < 1e-12
    automatic = payload["automatic_synthesis"]
    assert abs(automatic["recovered_impedance_ohm"] - automatic["target_transformer_impedance_ohm"]) < 0.1
    # Loss-aware total-efficiency optimization is expected to shorten the
    # theoretical lossless 90-degree transformer slightly.
    assert 80.0 < automatic["recovered_electrical_length_deg"] < 90.0
    assert automatic["width_m"] > 0 and automatic["length_m"] > 0
    assert automatic["total_loss_db"] > 0
    assert automatic["reflection_magnitude"] < 0.1
    assert automatic["total_efficiency"] > 0.95
    assert automatic["maximum_power_balance_error"] < 1e-12
    sensitivity = payload["sensitivity"]
    assert sensitivity["minus_10pct_width_impedance_ohm"] > sensitivity["nominal_width_impedance_ohm"]
    assert sensitivity["plus_10pct_width_impedance_ohm"] < sensitivity["nominal_width_impedance_ohm"]
    assert sensitivity["er_4p3_impedance_ohm"] > sensitivity["er_4p7_impedance_ohm"]


def test_committed_microstrip_baseline_is_physical_and_traceable():
    _assert_baseline(json.loads(ARTIFACT.read_text(encoding="utf-8")))


def test_microstrip_baseline_recomputes_without_drift():
    current = run()
    _assert_baseline(current)
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for section in (
        "models", "validity_scope", "substrate", "cross_validation",
        "fifty_ohm_quarter_wave", "automatic_synthesis", "sensitivity",
    ):
        assert current[section] == committed[section]
