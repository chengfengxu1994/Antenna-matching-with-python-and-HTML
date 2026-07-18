import numpy as np
import pytest

from engine.touchstone import parse_touchstone
from engine.tuning_service import (
    optimize_manual_network_physical, run_manual_tuning_physical,
    run_manual_yield_analysis_physical,
)
from rfmatch_core import OptimizationCancelled


class _MeasuredComponent:
    def __init__(self, part_number, nominal_value):
        self.part_number = part_number
        self.nominal_value = nominal_value
        self.nominal_unit = "nH"
        self.component_type = "inductor"
        self.tolerance_pct = 5.0
        self._data = None

    def get_s_matrix_at_freq(self, frequency_hz):
        impedance = 0.2 + 1j * 2 * np.pi * frequency_hz * self.nominal_value * 1e-9
        denominator = 100.0 + impedance
        reflection = impedance / denominator
        transmission = 100.0 / denominator
        return np.asarray([[reflection, transmission], [transmission, reflection]])


class _MeasuredCapacitor(_MeasuredComponent):
    def __init__(self, part_number, nominal_value):
        super().__init__(part_number, nominal_value)
        self.nominal_unit = "pF"
        self.component_type = "capacitor"

    def get_s_matrix_at_freq(self, frequency_hz):
        impedance = 0.15 + 1.0 / (1j * 2 * np.pi * frequency_hz * self.nominal_value * 1e-12)
        denominator = 100.0 + impedance
        reflection = impedance / denominator
        transmission = 100.0 / denominator
        return np.asarray([[reflection, transmission], [transmission, reflection]])


class _MeasuredLibrary:
    def __init__(self):
        self.inductors = [_MeasuredComponent("L_EXACT", 5.6), _MeasuredComponent("L_OTHER", 5.6)]
        self.capacitors = [_MeasuredCapacitor("C_1P0", 1.0), _MeasuredCapacitor("C_10P", 10.0)]

    def find_nearest_inductor(self, value):
        return self.inductors[1]

    def find_nearest_capacitor(self, value):
        return min(self.capacitors, key=lambda item: abs(item.nominal_value - value))


def _matched_dut():
    return parse_touchstone(
        "# GHZ S RI R 50\n"
        "0.8 0 0\n"
        "1.0 0 0\n"
        "1.2 0 0\n",
        "matched.s1p",
    )


def _mismatched_dut():
    return parse_touchstone(
        "# GHZ S RI R 50\n"
        "0.8 0.5 -0.25\n"
        "1.0 0.5 -0.25\n"
        "1.2 0.5 -0.25\n",
        "mismatched.s1p",
    )


def test_manual_service_uses_component_order_as_physical_cascade_order():
    series_l = {
        "comp_type": "inductor", "connection_type": "series", "port": 0,
        "value": 8.2, "use_ideal": True,
    }
    shunt_c = {
        "comp_type": "capacitor", "connection_type": "shunt", "port": 0,
        "value": 2.2, "use_ideal": True,
    }
    source_first = run_manual_tuning_physical(
        _mismatched_dut(), None, target_frequency_hz=1.0e9,
        input_port=0, port_states=[], components=[series_l, shunt_c],
    )
    dut_first = run_manual_tuning_physical(
        _mismatched_dut(), None, target_frequency_hz=1.0e9,
        input_port=0, port_states=[], components=[shunt_c, series_l],
    )

    assert [item["comp_type"] for item in source_first["components"]] == [
        "inductor", "capacitor",
    ]
    assert [item["comp_type"] for item in dut_first["components"]] == [
        "capacitor", "inductor",
    ]
    assert abs(source_first["s11_db"] - dut_first["s11_db"]) > 1.0


def test_manual_service_accounts_for_matched_line_loss_and_power_balance():
    result = run_manual_tuning_physical(
        _matched_dut(), None,
        target_frequency_hz=1.0e9,
        input_port=0,
        port_states=[],
        components=[{
            "comp_type": "transmission_line",
            "connection_type": "series",
            "port": 0,
            "characteristic_impedance_ohm": 50.0,
            "electrical_length_deg": 45.0,
            "reference_frequency_hz": 1.0e9,
            "attenuation_db": 1.0,
        }],
        sweep_start_hz=0.8e9,
        sweep_stop_hz=1.2e9,
        sweep_points=3,
        use_snp_points=False,
    )
    delivered = 10 ** (-1.0 / 10.0)
    efficiency = result["sweep"]["efficiency"]
    assert result["numeric_core"] == "rfmatch_core"
    assert result["s11_magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert efficiency["total_pct"][1] == pytest.approx(100.0 * delivered, abs=1e-10)
    assert efficiency["component_loss_pct"][1] == pytest.approx(100.0 * (1.0 - delivered), abs=1e-10)
    assert result["maximum_power_balance_error"] < 1e-12


@pytest.mark.parametrize("kind", ["open_stub", "short_stub"])
def test_manual_service_supports_physical_shunt_stubs(kind):
    result = run_manual_tuning_physical(
        _matched_dut(), None,
        target_frequency_hz=1.0e9,
        input_port=0,
        port_states=[],
        components=[{
            "comp_type": kind,
            "connection_type": "shunt",
            "port": 0,
            "characteristic_impedance_ohm": 50.0,
            "electrical_length_deg": 45.0,
            "reference_frequency_hz": 1.0e9,
            "attenuation_db": 0.0,
        }],
    )
    assert result["s11_magnitude"] == pytest.approx(1 / np.sqrt(5), abs=1e-12)
    assert result["maximum_power_balance_error"] < 1e-12


@pytest.mark.parametrize(
    ("kind", "connection"),
    [("transmission_line", "shunt"), ("open_stub", "series")],
)
def test_manual_service_rejects_invalid_line_placement(kind, connection):
    with pytest.raises(ValueError, match="must use a"):
        run_manual_tuning_physical(
            _matched_dut(), None,
            target_frequency_hz=1.0e9,
            input_port=0,
            port_states=[],
            components=[{
                "comp_type": kind,
                "connection_type": connection,
                "characteristic_impedance_ohm": 50.0,
                "electrical_length_deg": 45.0,
                "reference_frequency_hz": 1.0e9,
            }],
        )


def test_manual_service_preserves_requested_measured_part_instead_of_first_nominal_match():
    result = run_manual_tuning_physical(
        _matched_dut(), _MeasuredLibrary(),
        target_frequency_hz=1.0e9,
        input_port=0,
        port_states=[],
        components=[{
            "comp_type": "inductor", "connection_type": "series", "port": 0,
            "value": 5.6, "use_ideal": False, "part_number": "L_EXACT",
        }],
    )
    assert result["components"][0]["part_number"] == "L_EXACT"


def test_manual_service_rejects_missing_requested_measured_part():
    with pytest.raises(ValueError, match="is not available"):
        run_manual_tuning_physical(
            _matched_dut(), _MeasuredLibrary(),
            target_frequency_hz=1.0e9,
            input_port=0,
            port_states=[],
            components=[{
                "comp_type": "inductor", "connection_type": "series", "port": 0,
                "value": 5.6, "use_ideal": False, "part_number": "L_MISSING",
            }],
        )


def test_manual_fixed_topology_refinement_improves_or_preserves_physical_band_score():
    progress = []
    result = optimize_manual_network_physical(
        _mismatched_dut(), _MeasuredLibrary(),
        target_frequency_hz=1.0e9,
        input_port=0,
        port_states=[],
        components=[
            {"comp_type": "inductor", "connection_type": "series", "port": 0,
             "value": 1.0, "use_ideal": True},
            {"comp_type": "capacitor", "connection_type": "shunt", "port": 0,
             "value": 1.0, "use_ideal": True},
        ],
        bands_mhz=[[900, 1100]],
        target_return_loss_db=10,
        objective="balanced",
        max_passes=4,
        progress_callback=progress.append,
    )
    assert result["mode"] == "manual_fixed_topology_refinement"
    assert result["variable_count"] == 2
    assert result["optimized"]["score_db"] >= result["baseline"]["score_db"]
    assert result["score_improvement_db"] >= 0
    assert result["evaluations"] <= 2 + 2 * 2 * (4 + 1) + 4 * 9 * 2 + 4 * 5 + 4 * 2
    assert len(result["sensitivity"]) == 2
    assert result["sensitivity"][0]["score_impact_db"] >= result["sensitivity"][1]["score_impact_db"]
    assert {item["parameter"] for item in result["sensitivity"]} == {"value"}
    assert all(item["perturbation_pct"] == 10 for item in result["sensitivity"])
    assert all(item["preferred_direction"] in {"hold", "increase", "decrease"}
               for item in result["sensitivity"])
    bottleneck = result["optimized"]["bands"][0]
    assert 900e6 <= bottleneck["worst_frequency_hz"] <= 1100e6
    assert bottleneck["worst_impedance_real_ohm"] is not None
    assert result["topology_probe_evaluations"] == 4 * 9 * 2
    assert result["topology_probes"]
    assert all(item["score_improvement_db"] > 0 for item in result["topology_probes"])
    assert result["topology_probes"] == sorted(
        result["topology_probes"], key=lambda item: item["score_improvement_db"], reverse=True,
    )
    measured = [alternative for probe in result["topology_probes"]
                for alternative in probe.get("measured_alternatives", [])]
    assert result["measured_probe_evaluations"] > 0
    assert measured
    assert all(item["component"]["use_ideal"] is False for item in measured)
    assert all(item["component"]["part_number"] == item["part_number"] for item in measured)
    assert result["measured_full_verification_evaluations"] > 0
    assert all(item["verification"] == "full_dut_plus_band_grid" for item in measured)
    assert all(item["verification_points"] == result["verification_points"] for item in measured)
    assert result["optimized_full"]["bands"][0]["worst_frequency_hz"] in result["result"]["sweep"]["frequencies"]
    assert result["result"]["maximum_power_balance_error"] < 1e-9
    assert progress and progress[-1]["stage"] == "manual_full_verification"


def test_manual_fixed_topology_refinement_honors_cooperative_cancellation():
    with pytest.raises(OptimizationCancelled):
        optimize_manual_network_physical(
            _mismatched_dut(), None,
            target_frequency_hz=1.0e9,
            input_port=0,
            port_states=[],
            components=[{"comp_type": "inductor", "connection_type": "series",
                         "port": 0, "value": 1.0, "use_ideal": True}],
            bands_mhz=[[900, 1100]],
            cancel_check=lambda: True,
        )


def test_manual_refinement_skips_topology_probes_when_goal_already_passes():
    result = optimize_manual_network_physical(
        _matched_dut(), None,
        target_frequency_hz=1.0e9,
        input_port=0,
        port_states=[],
        components=[{"comp_type": "inductor", "connection_type": "series",
                     "port": 0, "value": 0.05, "use_ideal": True}],
        bands_mhz=[[900, 1100]],
        target_return_loss_db=0.01,
        max_passes=1,
    )
    assert result["optimized"]["passes_all_bands"] is True
    assert result["optimized_full"]["passes_all_bands"] is True
    assert result["topology_probe_evaluations"] == 0
    assert result["topology_probes"] == []
    assert result["measured_full_verification_evaluations"] == 0


def test_manual_yield_analysis_is_deterministic_and_reports_component_risk():
    progress = []
    kwargs = dict(
        input_port=0,
        port_states=[],
        components=[{"comp_type": "inductor", "connection_type": "series", "port": 0,
                     "value": 0.2, "use_ideal": True, "tolerance_pct": 10.0}],
        bands_mhz=[[900, 1100]],
        target_return_loss_db=6.0,
        samples=60,
        seed=23,
        default_tolerance_pct=5.0,
    )
    first = run_manual_yield_analysis_physical(
        _matched_dut(), _MeasuredLibrary(), progress_callback=progress.append, **kwargs,
    )
    second = run_manual_yield_analysis_physical(
        _matched_dut(), _MeasuredLibrary(), **kwargs,
    )
    assert first["yield_fraction"] == second["yield_fraction"]
    assert first["return_loss_percentiles_db"] == second["return_loss_percentiles_db"]
    assert first["analysis_scope"] == "manual_exact_network_monte_carlo"
    assert first["frequency_points"] >= 17
    assert first["component_tolerances"][0]["tolerance_pct"] == 10.0
    assert first["risk_components"][0]["position"] == 1
    assert first["risk_components"][0]["variation_key"] in first["worst_sample"]
    assert abs(first["risk_components"][0]["worst_deviation_pct"]) > 0
    assert first["maximum_nominal_power_balance_error"] < 1e-9
    assert progress[-1]["stage"] == "manual_yield"
    assert progress[-1]["current"] == 60


def test_manual_yield_analysis_honors_cooperative_cancellation():
    with pytest.raises(OptimizationCancelled):
        run_manual_yield_analysis_physical(
            _matched_dut(), None,
            input_port=0, port_states=[],
            components=[{"comp_type": "capacitor", "connection_type": "shunt",
                         "port": 0, "value": 1.0, "use_ideal": True}],
            bands_mhz=[[900, 1100]], samples=20,
            cancel_check=lambda: True,
        )
