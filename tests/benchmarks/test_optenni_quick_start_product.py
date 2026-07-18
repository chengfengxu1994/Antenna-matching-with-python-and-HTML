import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/benchmarks/optenni-product-quick-start.json"


def test_product_search_rediscovers_native_optenni_quick_start_winner():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == 1
    assert artifact["input"]["sha256"] == (
        "FAD7716E0F2AAE52319082DD0B2C94B4EEF5818C8698D34ED54AA6B0BF63C949"
    )
    assert artifact["requested_catalog_size"]["inductor_models"] == 43
    assert artifact["requested_catalog_size"]["capacitor_models"] == 882
    assert artifact["exact_displayed_value_bom_rank"] == 1

    best = artifact["best"]
    assert best["topology_code"] == "PCSL"
    parts = {item["type"]: item for item in best["components"]}
    assert parts["capacitor"]["part_number"] == "GJM1554C1HR20BB01"
    assert parts["inductor"]["part_number"] == "04CS8N2"
    assert abs(
        best["minimum_total_efficiency_db"]
        - artifact["optenni_reference"]["minimum_total_efficiency_db"]
    ) < 1e-5

    comparison = artifact["closest_displayed_bom_variant"]["curve_comparison"]
    assert comparison["maximum_s11_delta_db_all_points"] < 5e-5
    assert comparison["maximum_efficiency_delta_db_all_points"] < 5e-5
    assert comparison["maximum_power_balance_error"] < 1e-12

    diagnostics = artifact["search_diagnostics"]
    assert diagnostics["numeric_core"] == "rfmatch_core"
    assert diagnostics["joint_refine_neighbors"] == 12
    assert diagnostics["joint_refine_port_blocks"] is True
    assert diagnostics["search_truncated"] is False
    assert diagnostics["physical_evaluations"] <= 400
