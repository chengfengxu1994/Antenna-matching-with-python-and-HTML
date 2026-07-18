import json
from pathlib import Path


def test_optenni_three_port_live_golden_matches_saved_winner_ui():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-multiport-live-golden.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == 1
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert artifact["source"]["application"] == "Optenni Lab 4.3"
    assert artifact["source"]["project_sha256"] == (
        "d2c29220fbbf152dc52897ef6b1900fed179ba8b791c55cb225343eb5e3ad4b1"
    )
    assert artifact["source"]["dut_sha256"] == (
        "4af62325474067d752dac32138c447b1da5e714a4140bafed484f0649131f6bc"
    )
    opr = artifact["source"]["opr_evidence"]
    assert opr["optenni_version"] == "4.3"
    assert opr["has_results"] is True
    assert opr["candidate_count"] == 101
    assert opr["matched_candidate_count"] == 100
    assert opr["saved_winner_index"] == 1
    assert opr["embedded_input_rows"] == 1001
    assert artifact["topology_by_port"] == {
        "0": "SCPL", "1": "PCSL", "2": "PCSL"
    }
    assert [item["part_number"] for item in artifact["bom"]] == [
        "GQM1885C2A1R0BB01", "04HP2N0",
        "GQM1885C2A1R0BB01", "04HP5N6",
        "GQM1885C2A3R0BB01", "04HP5N6",
    ]
    efficiency_differences = [
        abs(value)
        for port in artifact["efficiency_comparison"].values()
        for value in port["difference_from_rounded_ui_db"].values()
    ]
    assert max(efficiency_differences) <= 0.051

    power = artifact["power_balance_comparison"]
    differences = power["difference_from_rounded_ui_linear"]
    assert abs(differences["reflected"]) <= 0.002
    assert abs(differences["radiated"]) <= 0.002
    # Optenni and the nodal core draw the component/coupling accounting
    # boundary differently, while their combined non-radiated accepted power
    # agrees to UI rounding precision.
    combined_ui = sum(
        power["optenni_ui_linear"][key]
        for key in ("component_loss", "coupling")
    )
    combined_core = sum(
        power["rfmatch_core_linear"][key]
        for key in ("component_loss", "coupling")
    )
    assert abs(combined_core - combined_ui) <= 0.002
    assert abs(power["rfmatch_core_sum"] - 1.0) < 1e-12
    assert artifact["maximum_power_balance_error"] < 1e-12


def test_saved_optenni_opr_manifest_is_machine_extracted_and_stable():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads(
        (root / "benchmarks/optenni_exports/multiantenna_project_opr_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["project_sha256"] == (
        "d2c29220fbbf152dc52897ef6b1900fed179ba8b791c55cb225343eb5e3ad4b1"
    )
    assert manifest["has_results"] is True
    assert manifest["impedance_configuration"]["ports"] == 3
    assert manifest["impedance_configuration"]["frequency_points"] == 1001
    assert manifest["impedance_configuration"]["embedded_data_rows"] == 1001
    assert manifest["candidate_count"] == 101
    assert manifest["matched_candidate_count"] == 100
    winner = manifest["saved_winner"]
    assert winner["index"] == 1
    assert winner["topology_by_port"] == {"0": "SCPL", "1": "PCSL", "2": "PCSL"}
    assert [item["part_number"] for item in winner["components"]] == [
        "GQM1885C2A1R0BB01", "04HP2N0",
        "GQM1885C2A1R0BB01", "04HP5N6",
        "GQM1885C2A3R0BB01", "04HP5N6",
    ]
    assert [(item["port"], item["start_hz"], item["stop_hz"]) for item in manifest["bands"]] == [
        (0, 2.5e9, 2.69e9),
        (1, 1.92e9, 2.17e9),
        (2, 1.215e9, 1.3e9),
    ]
