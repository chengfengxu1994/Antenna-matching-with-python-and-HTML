from pathlib import Path

import pytest

from engine.component_lib import ComponentInfo, ComponentLibrary
from engine.touchstone import parse_touchstone
from engine.tuning_service import (
    run_tuning_switch_mdif_auto,
    run_tuning_tunable_mdif,
    run_tuning_yield_analysis,
)


TUTORIAL = Path(
    r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.5 Impedance tuning using a variable capacitor"
)
COMPONENTS = Path(r"C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary")


def test_product_service_replays_official_mdif_state_mapping_when_installed():
    antenna = TUTORIAL / "Variable_capacitor_Tutorial_antenna.s1p"
    tuner = TUTORIAL / "Variable_capacitor_tutorial.mdif"
    capacitor = COMPONENTS / "Capacitors/Murata Capacitors gjm15/GJM1555C1H2R8WB01.s2p"
    inductor = COMPONENTS / "Inductors/Coilcraft Inductors 0402cs/04CS15N.S2P"
    if not all(path.exists() for path in (antenna, tuner, capacitor, inductor)):
        pytest.skip("licensed Optenni product reference inputs are not installed")
    dut = parse_touchstone(antenna.read_text(encoding="utf-8", errors="replace"), antenna.name)
    library = ComponentLibrary()
    library.add_component(ComponentInfo("GJM1555C1H2R8WB01", str(capacitor), "__DIR__", "capacitor", 2.8, "pF"))
    library.add_component(ComponentInfo("04CS15N", str(inductor), "__DIR__", "inductor", 15.0, "nH"))
    results = run_tuning_tunable_mdif(
        dut,
        library,
        0,
        str(tuner),
        [
            {"name": "Set 1", "bands_mhz": [[704, 746], [1920, 2170]], "weight": 1},
            {"name": "Set 2", "bands_mhz": [[791, 862], [1920, 2170]], "weight": 1},
            {"name": "Set 3", "bands_mhz": [[880, 960], [1920, 2170]], "weight": 1},
        ],
        [
            {"connection": "series", "kind": "C", "value": 2.8},
            {"connection": "series", "kind": "L", "value": 15.0},
        ],
        "balanced",
    )
    result = results[0]
    assert result.tunable_states == {"Set 1": "8 pF", "Set 2": "2 pF", "Set 3": "1 pF"}
    assert result.system_score == pytest.approx(-1.640599673090044, abs=1e-9)
    assert result.maximum_power_balance_error < 1e-12
    assert result.efficiency_basis == "rfmatch_core_physical_mdif"
    assert result.system_power_balance["system_efficiency"] == pytest.approx(
        result.avg_total_efficiency
    )
    assert result.per_port[0].components[1]["part_number"] == "GJM1555C1H2R8WB01"
    assert result.yield_context is not None
    yield_result = run_tuning_yield_analysis(
        dut,
        library,
        [result],
        {"ports": [{
            "port_index": 0,
            "enabled": True,
            "bands_mhz": [[704, 960], [1920, 2170]],
        }]},
        samples=20,
        seed=31,
        minimum_total_efficiency=0.2,
        minimum_return_loss_db=0.0,
    )
    analysis = yield_result["ranked_candidates"][0]
    assert analysis["analysis_scope"] == "joint_tunable_configurations"
    assert set(analysis["configuration_yield_fraction"]) == {"Set 1", "Set 2", "Set 3"}


def test_product_service_synthesizes_official_sp2t_switch_mapping_when_installed():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    antenna = root / "Switch_Tuner_Tutorial.s1p"
    switch = root / "tutorial_SP2T.mdif"
    c12 = COMPONENTS / "Capacitors/Murata Capacitors gjm15/GJM1552C1H1R2WB01.s2p"
    c08 = COMPONENTS / "Capacitors/Murata Capacitors gjm15/GJM1554C1HR80WB01.s2p"
    l13 = COMPONENTS / "Inductors/Coilcraft Inductors 0402cs/04CS13N.S2P"
    if not all(path.exists() for path in (antenna, switch, c12, c08, l13)):
        pytest.skip("licensed Optenni switch reference inputs are not installed")
    dut = parse_touchstone(antenna.read_text(encoding="utf-8", errors="replace"), antenna.name)
    library = ComponentLibrary()
    library.add_component(ComponentInfo("GJM1552C1H1R2WB01", str(c12), "__DIR__", "capacitor", 1.2, "pF"))
    library.add_component(ComponentInfo("GJM1554C1HR80WB01", str(c08), "__DIR__", "capacitor", 0.8, "pF"))
    library.add_component(ComponentInfo("04CS13N", str(l13), "__DIR__", "inductor", 13.0, "nH"))
    results = run_tuning_switch_mdif_auto(
        dut,
        0,
        str(switch),
        [
            {"name": "Set 1", "bands_mhz": [[704, 746], [1920, 2170]], "weight": 1},
            {"name": "Set 2", "bands_mhz": [[791, 862], [1920, 2170]], "weight": 1},
            {"name": "Set 3", "bands_mhz": [[880, 960], [1920, 2170]], "weight": 1},
        ],
        "balanced",
        library=library,
        measured_refine=True,
    )
    result = results[0]
    assert len(results) == 3
    assert [item.solution_index for item in results.values()] == [0, 1, 2]
    assert all(item.num_solutions_found == 3 for item in results.values())
    assert {
        item.search_diagnostics["input_component_count"] for item in results.values()
    } == {0, 1, 2}
    assert [item.system_score for item in results.values()] == sorted(
        (item.system_score for item in results.values()), reverse=True
    )
    one_part = next(
        item for item in results.values()
        if item.search_diagnostics["input_component_count"] == 1
    )
    assert len(one_part.search_diagnostics["input_network"]) == 1
    assert one_part.total_component_count == 4  # switch + two throw branches + shared input
    assert result.tunable_states == {
        "Set 1": "all on", "Set 2": "RFC-RF1", "Set 3": "RFC-RF2"
    }
    assert result.efficiency_basis == "rfmatch_core_switch_mdif_wave_power"
    assert result.search_diagnostics["mode"] == "switch_mdif_measured_synthesis"
    assert result.search_diagnostics["switch_state_precomputations"] == 4
    assert result.search_diagnostics["active_frequency_points"] == 46
    assert [item["kind"] for item in result.search_diagnostics["branch_network"]] == ["C", "C"]
    assert result.search_diagnostics["input_network"][0]["kind"] == "L"
    assert [item["part_number"] for item in result.search_diagnostics["branch_network"]] == [
        "GJM1552C1H1R2WB01", "GJM1554C1HR80WB01"
    ]
    assert result.search_diagnostics["input_network"][0]["part_number"] == "04CS13N"
    assert result.search_diagnostics["physical_evaluations"] > 0
    assert {
        item["input_component_count"]
        for item in result.search_diagnostics["complexity_alternatives"]
    } == {0, 1, 2}
    alternatives = result.search_diagnostics["complexity_alternatives"]
    assert alternatives[0]["recommendation_role"] == "best_performance"
    assert alternatives[0]["score_delta_from_best_db"] == pytest.approx(0.0)
    assert next(
        item for item in alternatives if item["input_component_count"] == 0
    )["recommendation_role"] == "simplest_bom"
    assert all(item["score_delta_from_best_db"] <= 0 for item in alternatives)
    assert result.search_diagnostics["calibration_reference"]["verified_pages"] == [12, 13, 16]
    assert result.search_diagnostics["calibration_reference"]["status"] == (
        "reference_only_not_request_specific"
    )
    assert result.maximum_power_balance_error < 5e-6
    assert result.num_solutions_found > 0
