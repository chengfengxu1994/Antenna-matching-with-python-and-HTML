import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "benchmarks" / "optenni-product-radiation-efficiency.json"
MANIFEST = ROOT / "benchmarks" / "optenni_exports" / "radiation_efficiency_tutorial_manifest.json"


def test_radiation_efficiency_reference_manifest_is_explicit_about_evidence_scope():
    reference = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert reference["evidence_kind"] == "published_tutorial_rounded_reference"
    assert reference["optimization"]["within_band_average_weight"] == 0.5
    assert reference["sources"]["radiation_efficiency"]["rows"] == 51
    assert reference["published_circuits"]["with_radiation_efficiency"]["three_components"]["topology_code_input_to_dut"] == "PCSLPC"


def test_product_radiation_efficiency_artifact_preserves_the_expected_benefit():
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert report["input"]["dut_sha256"] == "ba6078737518af9694e3bec4e6371c99a43c961e02bcba175d9b633691a77853"
    assert report["input"]["efficiency_sha256"] == "783815d5bee10d09d0236dd50c00dd9b69451dc2dc468a11972402d87555c355"
    assert report["catalog"] == {"inductors": 41, "capacitors": 104}
    assert report["request"]["generic_synthesis_loss"] == {
        "inductor_q": 50.0,
        "inductor_q_reference_hz": 1e9,
        "inductor_esr_ohm": 0.0,
        "capacitor_esr_ohm": 0.3,
        "scope": "continuous_topology_prior_only",
    }
    assert report["comparison"]["networks_differ"] is True
    assert report["comparison"]["efficiency_aware_minus_mismatch_only_objective_db"] > 0.0
    assert report["comparison"]["efficiency_aware_minus_mismatch_only_worst_case_db"] >= 0.3

    aware = report["runs"]["efficiency_aware"]
    assert aware["network"]["topology_code"] == "PCSLPC"
    assert aware["network"]["objective_weights"] == {
        "within_band_average": 0.5,
        "across_band_average": 0.1,
        "port_average": 0.0,
    }
    assert aware["network"]["generic_synthesis_loss"] == report["request"]["generic_synthesis_loss"]
    dense = aware["dense_official_efficiency_evaluation"]
    assert dense["active_frequency_points"] == 65
    assert dense["maximum_power_balance_error"] < 1e-12
