import json
from pathlib import Path


def test_product_multi_scenario_full_family_search_recovers_optenni_reference():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-product-multi-scenario.json")
        .read_text(encoding="utf-8")
    )
    expected_hashes = {
        "free_space.s1p": "af6c537fdaf9ba643feaa6a1e158270dd82d345ff9c6f0cacdce8f646ed31eec",
        "cover.s1p": "f33e197cb3184d834d60f9749457fd1eacc88761de2efc388e423d7654f2ed6b",
        "cover_w_spacer.s1p": "f25325d76505db7550d6fa48eece1d7aa354e1bd5b5894802af9150d767ff1b2",
    }
    assert artifact["schema_version"] == 1
    for item in artifact["inputs"]:
        assert item["sha256"] == expected_hashes[Path(item["relative_path"]).name]

    catalog = artifact["catalog"]
    assert catalog["inductor_models"] == 43
    assert catalog["capacitor_models"] == 882
    assert catalog["inductor_unique_values"] == 43
    assert catalog["capacitor_unique_values"] == 116
    assert artifact["request"]["max_candidates_per_position"] == 24
    assert artifact["request"]["verification_band_points"] == 41

    reference = artifact["optenni_reference"]
    assert artifact["reference_topology_rank"] == 1
    assert artifact["reference_value_rank"] <= 3
    assert artifact["best"]["topology"] == reference["topology"]
    assert abs(reference["dense_product_replay"]["score_db"] - reference["measured_core_score_db"]) < 1e-12
    assert abs(artifact["score_delta_vs_measured_core_db"]) < 0.01
    assert artifact["best"]["maximum_power_balance_error"] < 1e-12
    assert all(
        item["maximum_power_balance_error"] < 1e-12
        for item in artifact["top_solutions"]
    )

    diagnostics = artifact["search_diagnostics"]
    assert diagnostics["topologies_requested"] == 16
    assert diagnostics["topologies_screened"] == 16
    assert diagnostics["topologies_started"] >= 3
    assert diagnostics["topologies_completed"] >= 3
    assert diagnostics["ideal_ranked_topologies"] >= 10
    assert diagnostics["local_refinement_evaluations"] >= 100
    assert diagnostics["physical_evaluations"] <= 2100
    assert diagnostics["component_models_built"] <= 70
    assert diagnostics["verification"]["physical_evaluations"] == artifact["solutions_count"]
    assert diagnostics["verification"]["frequency_points"] == 82
    assert artifact["search_wall_seconds"] <= 46.0
    assert artifact["verification_wall_seconds"] <= 6.0
    assert artifact["wall_seconds"] <= 52.0
