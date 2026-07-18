import json
from pathlib import Path


def test_saved_optenni_winner_is_automatically_rediscovered_exactly():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-saved-winner-discovery.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["schema_version"] == 1
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert artifact["input"]["sha256"] == (
        "4af62325474067d752dac32138c447b1da5e714a4140bafed484f0649131f6bc"
    )
    assert artifact["saved_winner"]["topology_by_port"] == {
        "0": "SCPL", "1": "PCSL", "2": "PCSL"
    }
    assert len(artifact["component_grid"]) == 4

    constrained = artifact["searches"]["topology_constrained"]
    automatic = artifact["searches"]["automatic"]
    for search in (constrained, automatic):
        assert search["exact_saved_winner_rank"] == 1
        assert search["saved_topology_rank"] == 1
        assert abs(search["best_minus_saved_score_db"]) < 1e-12
        assert search["loaded_component_models"] == 4
    assert constrained["physical_evaluations"] < automatic["physical_evaluations"]
    assert automatic["config"]["per_port_keep"] == 8
    assert automatic["physical_evaluations"] <= 700
    assert automatic["best_candidate"]["signature"] == artifact["saved_winner"]["signature"]


def test_saved_optenni_winner_survives_full_reference_catalog_search():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-saved-winner-full-catalog-discovery.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["catalog_mode"] == "full"
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert artifact["catalog_size"] == {"inductors": 41, "capacitors": 104}
    search = artifact["searches"]["topology_constrained"]
    assert search["saved_topology_rank"] == 1
    assert 1 <= search["exact_saved_winner_rank"] <= 20
    assert search["best_minus_saved_score_db"] > 0.0
    assert search["physical_evaluations"] <= 3_500
    assert search["loaded_component_models"] <= 60
    assert search["config"]["joint_refine_port_blocks"] is True
    assert search["config"]["joint_refine_beam_width"] == 8
    assert search["config"]["joint_refine_variants_per_value"] == 2


def test_product_service_rediscovers_saved_winner_through_full_catalog():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-product-saved-winner-constrained.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["input"]["sha256"] == (
        "4af62325474067d752dac32138c447b1da5e714a4140bafed484f0649131f6bc"
    )
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert artifact["requested_catalog_size"] == {
        "inductors": 41, "capacitors": 104
    }
    assert 1 <= artifact["exact_saved_winner_rank"] <= 20
    assert artifact["best_minus_saved_score_db"] > 0.0
    assert abs(
        artifact["saved_winner"]["score_db"] - artifact["saved_score_db"]
    ) < 1e-12
    diagnostic = artifact["search_diagnostics"]
    assert diagnostic["numeric_core"] == "rfmatch_core"
    assert diagnostic["component_catalog_size"] == artifact["requested_catalog_size"]
    assert diagnostic["per_port_keep"] == 8
    assert diagnostic["coupled_ideal_topology_search"] is True
    assert diagnostic["joint_refine_port_blocks"] is True
    assert diagnostic["search_truncated"] is False
    assert diagnostic["physical_evaluations"] <= 4_300
    assert sum(diagnostic["stage_physical_evaluations"].values()) == (
        diagnostic["physical_evaluations"]
    )
    assert artifact["best"]["maximum_power_balance_error"] < 1e-12


def test_full_catalog_automatic_search_recovers_saved_topology_without_hint():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-saved-winner-full-catalog-automatic.json")
        .read_text(encoding="utf-8")
    )
    search = artifact["searches"]["automatic"]
    assert artifact["catalog_size"] == {"inductors": 41, "capacitors": 104}
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert search["saved_topology_rank"] == 1
    assert 1 <= search["exact_saved_winner_rank"] <= 20
    assert search["best_minus_saved_score_db"] > 0.0
    assert search["best_candidate"]["topology_by_port"] == {
        "0": "SCPL", "1": "PCSL", "2": "PCSL"
    }
    assert search["physical_evaluations"] <= 6_500
    assert search["wall_seconds"] < 180.0
    config = search["config"]
    assert config["topology_constrained"] is False
    assert config["per_port_keep"] == 13
    assert config["joint_ideal_combination_keep"] == 48
    assert config["joint_ideal_diverse_combinations"] is True
    assert config["joint_ideal_refine_topology_neighbors"] is True
    assert config["joint_ideal_growth_refine_keep"] == 8


def test_product_service_recovers_full_catalog_topology_without_hint():
    root = Path(__file__).resolve().parents[3]
    artifact = json.loads(
        (root / "artifacts/benchmarks/optenni-product-saved-winner-automatic.json")
        .read_text(encoding="utf-8")
    )
    assert artifact["requested_catalog_size"] == {
        "inductors": 41, "capacitors": 104
    }
    assert artifact["software"]["rfmatch_core_version"] == "0.3.0"
    assert artifact["request"]["topology_constraints"] is None
    assert artifact["saved_topology_rank"] == 1
    assert 1 <= artifact["exact_saved_winner_rank"] <= 20
    assert artifact["best_minus_saved_score_db"] > 0.0
    assert artifact["wall_seconds"] < artifact["request"]["timeout_seconds"]
    assert artifact["best"]["maximum_power_balance_error"] < 1e-12
    diagnostics = artifact["search_diagnostics"]
    assert diagnostics["numeric_core"] == "rfmatch_core"
    assert diagnostics["automatic_topology_deep_search"] is True
    assert diagnostics["search_truncated"] is False
    assert diagnostics["physical_evaluations"] <= 4_700
    assert diagnostics["component_models_loaded"] <= 75
    assert diagnostics["joint_refine_beam_width"] == 4
    assert sum(diagnostics["stage_physical_evaluations"].values()) == (
        diagnostics["physical_evaluations"]
    )
