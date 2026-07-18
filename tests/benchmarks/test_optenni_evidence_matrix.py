"""The parity matrix must distinguish real Optenni evidence from RFMatch replay."""

from scripts.audit_optenni_evidence import audit


def test_optenni_evidence_matrix_is_complete_and_honest():
    result = audit()
    assert result["schema_version"] == 2
    assert result["valid"] is True
    assert result["case_count"] == 7
    assert result["native_curve_export_count"] >= 2
    for case in result["cases"]:
        assert case["files"]
        assert all(item["exists"] and len(item["sha256"]) == 64 for item in case["files"])
        assert case["remaining_gap"]
        if case["level"] == "rfmatch_recompute_only":
            assert case["cross_software_numeric"] is False
