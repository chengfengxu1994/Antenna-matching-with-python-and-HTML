import hashlib
import json
from pathlib import Path


def test_optenni_single_port_search_recall_artifact_meets_p0_gate():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts" / "benchmarks" / "optenni-single-port-search-recall.json")
        .read_text(encoding="utf-8")
    )
    input_path = root / artifact["input"]["path"]
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == artifact["input"]["sha256"]
    assert artifact["schema_version"] == 1
    assert len(artifact["environment"]["environment_fingerprint_sha256"]) == 64
    assert artifact["environment"]["logical_cpu_count"] > 0
    assert artifact["recall"]["top_k"] == 10
    assert artifact["recall"]["exact_top_k_recall"] >= 0.95
    assert artifact["recall"]["topology_top_k_recall"] == 1.0
    assert artifact["recall"]["best_score_gap_db"] < 1e-12
    assert (
        artifact["recall"]["heuristic_physical_evaluations"]
        < artifact["recall"]["exhaustive_physical_evaluations"]
    )
    assert artifact["performance"]["heuristic_physical_evaluations_per_wall_second"] > 0
    assert artifact["performance"]["exhaustive_physical_evaluations_per_wall_second"] > 0
    assert artifact["performance"]["heuristic_peak_tracemalloc_mib"] > 0
    assert all(
        candidate["maximum_power_balance_error"] < 1e-9
        for candidate in artifact["exhaustive_top_k"]
    )


def test_product_single_port_full_catalog_recovers_optenni_topology():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-product-optimization-settings.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["input"]["sha256"] == (
        "fad7716e0f2aae52319082dd0b2c94b4eef5818c8698d34ed54aa6b0bf63c949"
    )
    manifest_path = root / artifact["optenni_reference"]["manifest"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        artifact["optenni_reference"]["manifest_sha256"]
    )
    catalog = artifact["requested_catalog_size"]
    assert catalog["inductors"] == 49
    assert catalog["capacitors"] == 64
    assert catalog["inductor_range_nh"][0] <= 5.9151 <= catalog["inductor_range_nh"][1]
    assert catalog["capacitor_range_pf"][0] <= 0.48399 <= catalog["capacitor_range_pf"][1]
    assert artifact["optenni_topology_rank"] == 1
    assert artifact["best"]["topology_code"] == "PCSL"
    assert [
        (item["connection_type"], item["type"], item["part_number"])
        for item in artifact["best"]["components"]
    ] == [
        ("shunt", "capacitor", "C0402SEr45"),
        ("series", "inductor", "04HP5N6"),
    ]
    values = artifact["procurement_value_comparison"]
    assert values["capacitor_relative_deviation"] < 0.08
    assert values["inductor_relative_deviation"] < 0.06
    efficiency = artifact["cross_software_efficiency_comparison"]
    assert abs(efficiency["minimum_efficiency_delta_db"]) < 0.05
    assert abs(efficiency["average_efficiency_delta_db"]) < 0.05
    diagnostic = artifact["search_diagnostics"]
    assert diagnostic["numeric_core"] == "rfmatch_core"
    assert diagnostic["search_profile"] == "thorough"
    assert diagnostic["search_truncated"] is False
    assert diagnostic["physical_evaluations"] <= 120
    assert diagnostic["component_models_loaded"] <= 50
    assert sum(diagnostic["stage_physical_evaluations"].values()) == (
        diagnostic["physical_evaluations"]
    )
    assert artifact["wall_seconds"] < 10.0
    assert artifact["best"]["maximum_power_balance_error"] < 1e-12
