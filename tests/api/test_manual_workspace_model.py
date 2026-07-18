import pytest
from pydantic import ValidationError

from api.models import ManualWorkspaceSnapshot


def _variant(variant_id="manual-a"):
    return {
        "variant_id": variant_id,
        "name": "P1 baseline",
        "input_port": 0,
        "target_frequency_hz": 2.45e9,
        "components": [{
            "comp_type": "capacitor", "connection_type": "shunt",
            "value": 2.2, "use_ideal": False, "part_number": "C_EXACT", "port": 0,
        }],
        "port_states": [{"port_index": 1, "state": "load"}],
        "metrics": {
            "return_loss_db": 17.2,
            "return_loss_improvement_db": 8.1,
            "vswr": 1.32,
            "input_impedance_real": 47.0,
            "input_impedance_imag": -3.0,
            "maximum_power_balance_error": 1e-12,
            "numeric_core": "rfmatch_core",
        },
        "created_at": "2026-07-18T00:00:00+00:00",
    }


def test_manual_workspace_accepts_bounded_reproducible_snapshot():
    workspace = ManualWorkspaceSnapshot(
        target_frequency_hz=2.45e9,
        working_networks={"0": _variant()["components"]},
        variants=[_variant()],
        selected_variant_id="manual-a",
        overlay_variant_ids=["manual-a"],
    )
    assert workspace.variants[0].components[0]["part_number"] == "C_EXACT"
    assert workspace.selected_variant_id == "manual-a"
    assert workspace.overlay_variant_ids == ["manual-a"]


@pytest.mark.parametrize(
    "changes",
    [
        {"working_networks": {"not-a-port": []}},
        {"variants": [_variant("manual-a"), _variant("manual-a")]},
        {"selected_variant_id": "manual-missing"},
        {"working_networks": {"0": [{}] * 13}},
        {"overlay_variant_ids": ["manual-missing"]},
        {"overlay_variant_ids": ["manual-a", "manual-a"]},
    ],
)
def test_manual_workspace_rejects_ambiguous_or_unbounded_state(changes):
    payload = {
        "target_frequency_hz": 2.45e9,
        "working_networks": {"0": []},
        "variants": [_variant()],
        "selected_variant_id": "manual-a",
        **changes,
    }
    with pytest.raises(ValidationError):
        ManualWorkspaceSnapshot(**payload)
