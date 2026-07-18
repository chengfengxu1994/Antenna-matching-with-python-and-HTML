import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_layout_block_benchmark import run


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "benchmarks" / "layout-block-physical-baseline.json"


def _assert_baseline(payload):
    assert payload["schema_version"] == 1
    assert payload["layout_diagnostics"]["passive"] is True
    assert payload["layout_diagnostics"]["maximum_reciprocity_error"] < 1e-14
    for location, result in payload["locations"].items():
        assert result["maximum_total_efficiency_error"] < 1e-12, location
        assert result["maximum_component_loss_error"] < 1e-12, location
        assert result["maximum_power_balance_error"] < 1e-12, location
        assert "layout_s2p" in result["component_order"]
    assert payload["locations"]["dut_side"]["component_order"][0] == "layout_s2p"
    assert payload["locations"]["connector_side"]["component_order"][-1] == "layout_s2p"


def test_committed_layout_block_baseline_is_physical():
    _assert_baseline(json.loads(ARTIFACT.read_text(encoding="utf-8")))


def test_layout_block_baseline_recomputes_without_drift():
    current = run()
    _assert_baseline(current)
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for section in ("input", "layout_diagnostics", "expected", "locations"):
        assert current[section] == committed[section]
    current_transforms = deepcopy(current["reference_plane_transforms"])
    committed_transforms = deepcopy(committed["reference_plane_transforms"])
    for field in (
        "maximum_renormalization_round_trip_error",
        "maximum_deembedding_recovery_error",
    ):
        assert current_transforms.pop(field) == pytest.approx(
            committed_transforms.pop(field), abs=1e-14
        )
    assert current_transforms == committed_transforms
    transforms = current["reference_plane_transforms"]
    assert transforms["directional_flipped"][0][0] == transforms["directional_original"][1][1]
    assert transforms["directional_flipped"][0][1] == transforms["directional_original"][1][0]
    assert transforms["maximum_renormalization_round_trip_error"] < 1e-14
    assert transforms["maximum_deembedding_recovery_error"] < 1e-14
    assert transforms["deembedding_diagnostics"]["maximum_recascade_residual"] < 1e-14
