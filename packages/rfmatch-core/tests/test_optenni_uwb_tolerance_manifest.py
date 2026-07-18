import json
from pathlib import Path


def test_uwb_optenni_manifest_pins_microstrip_width_tolerance_semantics():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads(
        (root / "benchmarks/optenni_exports/uwb_microstrip_tolerance_settings.json")
        .read_text(encoding="utf-8")
    )
    settings = manifest["manufacturing_tolerance_settings"]
    assert manifest["project_sha256"] == (
        "2b0ed92f8634acd79fd210c90e9a5ffea245fd474a069be05c4a6db5aa3b2692"
    )
    assert settings["microstrip_width_tolerance_pct"] == 20.0
    assert settings["uses_transmission_line_synthesis"] is True
    assert settings["includes_substrate"] is True
    assert manifest["rfmatch_mapping"]["microstrip_width_tolerance_pct"] == (
        "transmission_line.microstrip.width_tolerance_pct"
    )
