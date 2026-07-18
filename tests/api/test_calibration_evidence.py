import hashlib
import json
import asyncio
from pathlib import Path
import shutil

import pytest

from engine.calibration_evidence import (
    CalibrationEvidenceError,
    PROJECT_ROOT,
    multiport_calibration_reference,
    search_performance_reference,
    single_port_calibration_reference,
)
from api import server


def test_single_port_calibration_claims_are_derived_from_hashed_artifacts():
    reference = single_port_calibration_reference()
    artifact = PROJECT_ROOT / reference["artifact"]
    assert reference["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert reference["integrity"] == "sha256_verified_at_runtime"
    assert reference["reference_exact_top_k_recall"] == 1.0
    product = reference["product_full_catalog_discovery"]
    assert product["topology"] == "PCSL"
    assert product["topology_rank"] == 1
    assert product["parts"] == ["C0402SEr45", "04HP5N6"]
    assert product["artifact_sha256"] == hashlib.sha256(
        (PROJECT_ROOT / product["artifact"]).read_bytes()
    ).hexdigest()


def test_multiport_calibration_claims_and_every_nested_artifact_are_hashed():
    reference = multiport_calibration_reference()
    assert reference["reference_exact_top_k_recall"] == 1.0
    assert reference["saved_winner_discovery"]["exact_saved_winner_rank"] == 1
    assert reference["full_catalog_discovery"]["exact_saved_winner_rank"] == 17
    assert reference["automatic_full_catalog_discovery"]["product_exact_saved_winner_rank"] == 15
    assert reference["four_port_scaling"]["reference_exact_top_k_recall"] == 1.0
    assert reference["four_port_scaling"]["scope"]["exhaustive_candidates"] == 6561
    for name in (
        "numerical_golden", "saved_winner_discovery",
        "full_catalog_discovery", "automatic_full_catalog_discovery",
        "four_port_scaling",
    ):
        item = reference[name]
        assert item["artifact_sha256"] == hashlib.sha256(
            (PROJECT_ROOT / item["artifact"]).read_bytes()
        ).hexdigest()


def test_calibration_builder_tracks_regenerated_claims_instead_of_hardcoding(tmp_path: Path):
    relative_files = (
        "artifacts/benchmarks/optenni-single-port-search-recall.json",
        "artifacts/benchmarks/optenni-product-optimization-settings.json",
    )
    for relative in relative_files:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, destination)
    recall_path = tmp_path / relative_files[0]
    document = json.loads(recall_path.read_text(encoding="utf-8"))
    document["recall"]["exact_top_k_recall"] = 0.7
    recall_path.write_text(json.dumps(document), encoding="utf-8")

    reference = single_port_calibration_reference(tmp_path)
    assert reference["reference_exact_top_k_recall"] == 0.7
    assert reference["artifact_sha256"] == hashlib.sha256(recall_path.read_bytes()).hexdigest()


def test_invalid_calibration_artifact_fails_closed(tmp_path: Path):
    path = tmp_path / "artifacts/benchmarks/optenni-single-port-search-recall.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": 999, "case": "bad"}', encoding="utf-8")
    with pytest.raises(CalibrationEvidenceError, match="unsupported schema"):
        single_port_calibration_reference(tmp_path)


def test_calibration_status_api_exposes_verified_reference_hashes():
    status = asyncio.run(server.calibration_status())
    assert status["status"] == "verified"
    assert status["integrity"] == "sha256_verified_at_runtime"
    assert len(status["single_port"]["artifact_sha256"]) == 64
    assert len(status["multiport"]["artifact_sha256"]) == 64
    assert len(status["performance"]["artifact_sha256"]) == 64
    assert status["performance"]["cases"]["single_port"]["heuristic_physical_evaluations_per_wall_second"] > 0
    assert status["performance"]["cases"]["four_port"]["exact_top_k_recall"] == 1.0


def test_performance_reference_is_explicitly_machine_scoped():
    reference = search_performance_reference()
    assert reference["status"] == "machine_scoped_reference"
    assert len(reference["environment_fingerprint_sha256"]) == 64
    assert "environment fingerprint matches" in reference["comparison_policy"]["cross_machine"]
    assert reference["cases"]["multiport"]["exact_top_k_recall"] == 1.0
    assert reference["cases"]["four_port"]["exact_top_k_recall"] == 1.0
