import math

import pytest
import numpy as np
from pydantic import ValidationError

from api.models import TuneSingleRequest, TuningOptimizeRequest
from engine.efficiency_data import EfficiencyData
from engine.tuning_service import (
    _resolve_generic_synthesis_loss,
    _resolve_single_objective_weights,
    compute_sweep,
)


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("worst_case", (0.0, 0.1)),
        ("average_efficiency", (1.0, 0.1)),
        ("balanced", (0.05, 0.1)),
    ],
)
def test_single_objective_weight_presets_remain_backward_compatible(objective, expected):
    assert _resolve_single_objective_weights(objective) == expected


def test_explicit_optenni_style_weights_override_the_named_preset():
    assert _resolve_single_objective_weights("balanced", 0.5, 0.5) == (0.5, 0.5)


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf])
def test_runtime_weight_resolution_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _resolve_single_objective_weights("balanced", value, 0.1)


def test_unified_and_legacy_requests_accept_custom_objective_weights():
    unified = TuningOptimizeRequest(
        within_band_average_weight=0.5,
        across_band_average_weight=0.25,
    )
    legacy = TuneSingleRequest(
        within_band_average_weight=0.5,
        across_band_average_weight=0.25,
    )
    assert unified.within_band_average_weight == legacy.within_band_average_weight == 0.5
    assert unified.across_band_average_weight == legacy.across_band_average_weight == 0.25


def test_request_schema_rejects_out_of_range_objective_weights():
    with pytest.raises(ValidationError):
        TuningOptimizeRequest(within_band_average_weight=1.1)
    with pytest.raises(ValidationError):
        TuneSingleRequest(across_band_average_weight=-0.1)


def test_unified_request_normalizes_port_and_band_priorities():
    request = TuningOptimizeRequest(ports=[{
        "port_index": 1,
        "enabled": True,
        "bands_mhz": [[700, 900], [1800, 2200]],
        "band_weights": [2, 0.5],
        "port_weight": 3,
        "max_components": 2,
    }])
    assert request.ports[0]["band_weights"] == [2.0, 0.5]
    assert request.ports[0]["port_weight"] == 3.0


def test_priority_request_defaults_and_invalid_shapes_are_explicit():
    default = TuningOptimizeRequest(ports=[{
        "port_index": 0, "enabled": True,
        "bands_mhz": [[700, 900], [1800, 2200]],
    }])
    assert default.ports[0]["band_weights"] == [1.0, 1.0]
    assert default.ports[0]["port_weight"] == 1.0
    with pytest.raises(ValidationError, match="band_weights must match"):
        TuningOptimizeRequest(ports=[{
            "port_index": 0, "enabled": True,
            "bands_mhz": [[700, 900], [1800, 2200]],
            "band_weights": [1],
        }])
    with pytest.raises(ValidationError, match="positive effective weight"):
        TuningOptimizeRequest(ports=[{
            "port_index": 0, "enabled": True,
            "bands_mhz": [[700, 900]], "port_weight": 0,
        }])


def test_legacy_single_request_persists_priorities():
    request = TuneSingleRequest(
        bands_mhz=[[700, 900], [1800, 2200]],
        band_weights=[2, 0.5],
        port_weight=3,
    )
    assert request.band_weights == [2.0, 0.5]
    assert request.port_weight == 3.0


def test_generic_synthesis_loss_defaults_and_radiation_tutorial_profile():
    default_model, default_diagnostics = _resolve_generic_synthesis_loss()
    assert default_model.inductor_q == 30.0
    assert default_model.capacitor_esr == 0.4
    assert default_diagnostics["scope"] == "continuous_topology_prior_only"

    model, diagnostics = _resolve_generic_synthesis_loss({
        "inductor_q": 50.0,
        "inductor_q_reference_hz": 1e9,
        "inductor_esr_ohm": 0.0,
        "capacitor_esr_ohm": 0.3,
    })
    assert model.inductor_q == 50.0
    assert model.capacitor_esr == 0.3
    assert diagnostics["capacitor_esr_ohm"] == 0.3


def test_request_schema_persists_generic_synthesis_loss_configuration():
    request = TuningOptimizeRequest(generic_synthesis_loss={
        "inductor_q": 50.0,
        "inductor_q_reference_hz": 1e9,
        "capacitor_esr_ohm": 0.3,
    })
    assert request.generic_synthesis_loss.inductor_q == 50.0
    assert request.generic_synthesis_loss.capacitor_esr_ohm == 0.3
    with pytest.raises(ValidationError):
        TuneSingleRequest(generic_synthesis_loss={"inductor_q": 0.0})


def test_live_sweep_applies_loaded_radiation_efficiency_to_total_efficiency():
    class Dut:
        frequencies = [1.0e9, 2.0e9]
        num_ports = 1

        @staticmethod
        def get_s_matrix_interpolated(_frequency):
            return np.asarray([[[0.5 + 0.0j]]]).reshape(1, 1)

    efficiency = EfficiencyData(
        np.asarray([1.0e9, 2.0e9]), np.asarray([0.8, 0.6]), "test"
    )
    sweep = compute_sweep(
        Dut(), {0: []}, 0, 1.0e9, 2.0e9,
        global_efficiency=efficiency,
    )
    assert sweep["efficiency"]["basis"] == "radiation_x_delivered_power"
    assert sweep["efficiency"]["radiation_pct"] == pytest.approx([80.0, 60.0])
    assert sweep["efficiency"]["total_pct"] == pytest.approx([60.0, 45.0])
    assert sweep["s11_real"] == pytest.approx([0.5, 0.5])
    assert sweep["s11_imag"] == pytest.approx([0.0, 0.0])
    assert sweep["raw_real"] == pytest.approx([0.5, 0.5])
    assert sweep["raw_imag"] == pytest.approx([0.0, 0.0])
