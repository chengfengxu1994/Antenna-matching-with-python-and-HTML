import asyncio
import numpy as np
import pytest
import tempfile
from pathlib import Path

from engine.touchstone import parse_touchstone
from engine.tuning_service import (
    compute_transmission_line_sweep,
    core_s2p_layout_from_touchstone,
    run_tuning_transmission_line,
    run_tuning_yield_analysis,
)


def _hundred_ohm_dut():
    gamma = (100.0 - 50.0) / (100.0 + 50.0)
    return parse_touchstone(
        "# GHZ S RI R 50\n"
        f"0.999 {gamma} 0\n"
        f"1.000 {gamma} 0\n"
        f"1.001 {gamma} 0\n",
        "load_100ohm.s1p",
    )


def test_product_service_recovers_quarter_wave_line_and_canonical_result():
    results = run_tuning_transmission_line(
        _hundred_ohm_dut(),
        port_index=0,
        bands_mhz=[[999.0, 1001.0]],
        objective="balanced",
        num_band_points=3,
        timeout_seconds=10.0,
        search_config={
            "characteristic_impedance_min_ohm": 60.0,
            "characteristic_impedance_max_ohm": 80.0,
            "electrical_length_min_deg": 70.0,
            "electrical_length_max_deg": 110.0,
            "topologies": ["through_line"],
            "restarts": 8,
            "iterations": 40,
            "maximum_evaluations": 5000,
        },
    )
    best = results[0]
    diagnostic = best.search_diagnostics
    component = best.per_port[0].components[0]
    assert best.mode == "transmission_line"
    assert best.efficiency_basis == "rfmatch_core_physical_transmission_line"
    assert component["comp_type"] == "transmission_line"
    assert component["characteristic_impedance_ohm"] == pytest.approx(np.sqrt(5000.0), abs=0.05)
    assert component["electrical_length_deg"] == pytest.approx(90.0, abs=0.05)
    assert diagnostic["mode"] == "transmission_line_auto_synthesis"
    assert best.maximum_power_balance_error < 1e-12
    assert best.system_power_balance["per_port"]["0"]["sum_check"] == pytest.approx(1.0, abs=1e-12)
    assert best.per_port[0].s11_db > 60.0

    sweep = compute_transmission_line_sweep(
        _hundred_ohm_dut(), best,
        start_hz=0.999e9, stop_hz=1.001e9,
        num_points=7, use_snp_points=False,
    )
    assert len(sweep["frequencies"]) == 7
    assert sweep["maximum_power_balance_error"] < 1e-12
    assert sweep["s11_db"][3] > 60.0
    assert sweep["efficiency"]["total_pct"][3] == pytest.approx(100.0, abs=1e-6)


def test_product_service_cooperatively_cancels():
    from rfmatch_core import OptimizationCancelled

    with pytest.raises(OptimizationCancelled):
        run_tuning_transmission_line(
            _hundred_ohm_dut(),
            port_index=0,
            bands_mhz=[[999.0, 1001.0]],
            search_config={"topologies": ["through_line"]},
            cancel_check=lambda: True,
        )


def test_product_service_returns_manufacturable_microstrip_geometry():
    results = run_tuning_transmission_line(
        _hundred_ohm_dut(),
        port_index=0,
        bands_mhz=[[999.0, 1001.0]],
        num_band_points=3,
        timeout_seconds=10.0,
        search_config={
            "characteristic_impedance_min_ohm": 60.0,
            "characteristic_impedance_max_ohm": 80.0,
            "electrical_length_min_deg": 70.0,
            "electrical_length_max_deg": 110.0,
            "topologies": ["through_line"],
            "restarts": 4,
            "iterations": 24,
            "maximum_evaluations": 3000,
            "microstrip": {
                "enabled": True,
                "substrate_name": "FR-4 engineering model",
                "relative_permittivity": 4.5,
                "substrate_height_mm": 1.6,
                "loss_tangent": 0.02,
                "copper_thickness_um": 35.0,
                "copper_resistivity_ohm_m": 1.68e-8,
                "copper_roughness_rms_um": 0.15,
                "minimum_width_mm": 0.1,
                "maximum_width_mm": 10.0,
                "width_tolerance_pct": 20.0,
                "length_tolerance_pct": 2.0,
                "substrate_height_tolerance_pct": 5.0,
                "relative_permittivity_tolerance_pct": 4.0,
            },
        },
    )
    best = results[0]
    component = best.per_port[0].components[0]
    assert best.search_diagnostics["physical_microstrip"] is True
    assert component["physical_model"] == "microstrip_hammerstad_kirschning_wheeler"
    assert 0.1 < component["width_m"] * 1e3 < 10.0
    assert component["length_m"] > 0
    assert component["conductor_loss_db"] > 0
    assert component["dielectric_loss_db"] > 0
    assert component["manufacturing_tolerances_pct"] == {
        "trace_width": 20.0,
        "physical_length": 2.0,
        "substrate_height": 5.0,
        "relative_permittivity": 4.0,
    }
    assert "mm x" in component["value"] and "Z0" in component["value"]
    assert best.total_component_loss > 0
    assert best.maximum_power_balance_error < 1e-12

    yield_result = run_tuning_yield_analysis(
        _hundred_ohm_dut(), None, [best],
        {"ports": [{"port_index": 0, "bands_mhz": [[999.0, 1001.0]], "enabled": True}]},
        samples=20, seed=5, minimum_total_efficiency=0.0,
        minimum_average_total_efficiency=0.0, minimum_return_loss_db=0.0,
    )
    analysis = yield_result["ranked_candidates"][0]
    assert analysis["analysis_scope"] == "physical_transmission_line_manufacturing"
    assert analysis["component_tolerances"][0]["variable"] == "trace_width"
    assert "b1_p1_MS1.trace_width" in analysis["worst_sample"] or any(
        name.endswith(".trace_width") for name in analysis["worst_sample"]
    )


def test_product_service_cascades_traceable_measured_layout_block():
    from rfmatch_core import TransmissionLineModel

    frequencies = np.asarray([0.999e9, 1.0e9, 1.001e9])
    layout = TransmissionLineModel(
        "launch.s2p", 50.0, 10.0, 1.0e9, attenuation_db=1.0
    ).as_s2p_model(frequencies)
    results = run_tuning_transmission_line(
        _hundred_ohm_dut(),
        port_index=0,
        bands_mhz=[[999.0, 1001.0]],
        num_band_points=3,
        timeout_seconds=10.0,
        search_config={
            "characteristic_impedance_min_ohm": 60.0,
            "characteristic_impedance_max_ohm": 80.0,
            "electrical_length_min_deg": 70.0,
            "electrical_length_max_deg": 110.0,
            "topologies": ["through_line"],
            "restarts": 4,
            "iterations": 24,
        },
        layout_blocks=[{
            "model": layout,
            "filename": "layouts/launch.s2p",
            "sha256": "A" * 64,
            "location": "connector_side",
            "passivity": {"passive": True, "maximum_singular_value": 10 ** (-1 / 20)},
        }],
    )
    best = results[0]
    layouts = [
        item for item in best.per_port[0].components
        if item["comp_type"] == "layout_s2p"
    ]
    assert len(layouts) == 1
    assert layouts[0]["filename"] == "layouts/launch.s2p"
    assert layouts[0]["sha256"] == "A" * 64
    assert layouts[0]["location"] == "connector_side"
    assert best.total_component_loss > 0.1
    assert best.search_diagnostics["layout_blocks"][0]["sha256"] == "A" * 64
    assert best.maximum_power_balance_error < 1e-12


@pytest.mark.parametrize(("transmission", "passive"), [(0.8, True), (1.1, False)])
def test_layout_touchstone_conversion_reports_passivity(transmission, passive):
    layout = parse_touchstone(
        "# GHZ S RI R 50\n"
        f"1.0 0 0 {transmission} 0 {transmission} 0 0 0\n",
        "layout.s2p",
    )
    model, diagnostics = core_s2p_layout_from_touchstone(layout)
    assert model.s_parameters.shape == (1, 2, 2)
    assert diagnostics["passive"] is passive
    assert diagnostics["maximum_singular_value"] == pytest.approx(transmission)


def test_layout_touchstone_conversion_can_flip_and_renormalize_ports():
    from rfmatch_core import flip_s2p_ports, renormalize_s_parameters

    layout = parse_touchstone(
        "# GHZ S RI R 75\n"
        "1.0 0.1 0 0.3 0 0.2 0 0.4 0\n",
        "directional.s2p",
    )
    native = np.asarray([layout.get_s_matrix(0)])
    model, diagnostics = core_s2p_layout_from_touchstone(
        layout, reverse_ports=True, target_reference_impedance_ohm=50.0,
    )
    expected = renormalize_s_parameters(flip_s2p_ports(native), 75.0, 50.0)
    np.testing.assert_allclose(model.s_parameters, expected, atol=1e-14)
    assert model.z0 == 50.0
    assert diagnostics["ports_reversed"] is True
    assert diagnostics["renormalized"] is True
    assert diagnostics["native_reference_impedance_ohm"] == 75.0
    assert diagnostics["reference_impedance_ohm"] == 50.0


def test_unified_tuning_api_mode_does_not_require_component_library():
    from api import server
    from api.models import TuningOptimizeRequest
    from engine.tuning_service import reset_session

    previous = (
        server.state.loaded_snp,
        server.state.component_library,
        server.state.db_library,
        server.state.use_db,
    )
    try:
        server.state.loaded_snp = _hundred_ohm_dut()
        server.state.component_library = None
        server.state.db_library = None
        server.state.use_db = False
        reset_session()
        response = asyncio.run(server._tuning_optimize_impl(TuningOptimizeRequest(
            mode="transmission_line",
            ports=[{
                "port_index": 0, "bands_mhz": [[999.0, 1001.0]],
                "max_components": 2, "enabled": True,
            }],
            timeout_seconds=10.0,
            num_band_points=3,
            transmission_line={
                "characteristic_impedance_min_ohm": 60.0,
                "characteristic_impedance_max_ohm": 80.0,
                "electrical_length_min_deg": 70.0,
                "electrical_length_max_deg": 110.0,
                "topologies": ["through_line"],
                "restarts": 4,
                "iterations": 30,
                "maximum_evaluations": 2000,
            },
        )))
        assert response["status"] == "ok"
        assert response["mode"] == "transmission_line"
        assert response["solutions_count"] > 0
        assert response["best_solution"]["search_diagnostics"]["numeric_core"] == "rfmatch_core"
    finally:
        (
            server.state.loaded_snp,
            server.state.component_library,
            server.state.db_library,
            server.state.use_db,
        ) = previous
        reset_session()


def test_unified_api_loads_layout_block_from_safe_snp_directory():
    from api import server
    from api.models import TuningOptimizeRequest
    from engine.tuning_service import reset_session

    previous = (
        server.state.loaded_snp, server.state.loaded_snp_filename,
        server.state.snp_dir, server.state.component_library,
        server.state.db_library, server.state.use_db,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        layout_path = root / "launch.s2p"
        layout_path.write_text(
            "# GHZ S RI R 50\n"
            "0.999 0 0 0.9 0 0.9 0 0 0\n"
            "1.000 0 0 0.9 0 0.9 0 0 0\n"
            "1.001 0 0 0.9 0 0.9 0 0 0\n",
            encoding="utf-8",
        )
        fixture_path = root / "fixture.s2p"
        fixture_path.write_text(
            "# GHZ S RI R 50\n"
            "0.999 0 0 1 0 1 0 0 0\n"
            "1.000 0 0 1 0 1 0 0 0\n"
            "1.001 0 0 1 0 1 0 0 0\n",
            encoding="utf-8",
        )
        try:
            server.state.loaded_snp = _hundred_ohm_dut()
            server.state.loaded_snp_filename = "load_100ohm.s1p"
            server.state.snp_dir = directory
            server.state.component_library = None
            server.state.db_library = None
            server.state.use_db = False
            reset_session()
            response = asyncio.run(server._tuning_optimize_impl(TuningOptimizeRequest(
                mode="transmission_line",
                ports=[{"port_index": 0, "bands_mhz": [[999.0, 1001.0]], "enabled": True}],
                timeout_seconds=10.0,
                num_band_points=3,
                transmission_line={
                    "characteristic_impedance_min_ohm": 60.0,
                    "characteristic_impedance_max_ohm": 80.0,
                    "electrical_length_min_deg": 70.0,
                    "electrical_length_max_deg": 110.0,
                    "topologies": ["through_line"],
                    "restarts": 2,
                    "iterations": 12,
                    "maximum_evaluations": 1000,
                    "layout_blocks": [{
                        "filename": "launch.s2p",
                        "location": "connector_side",
                        "passivity_policy": "reject",
                        "reverse_ports": True,
                        "reference_impedance_mode": "system",
                        "left_fixture_filename": "fixture.s2p",
                        "left_fixture_reverse_ports": True,
                    }],
                },
            )))
            best = response["best_solution"]
            layouts = [
                item for item in best["per_port"]["0"]["components"]
                if item["comp_type"] == "layout_s2p"
            ]
            assert len(layouts) == 1
            assert layouts[0]["filename"] == "launch.s2p"
            assert len(layouts[0]["sha256"]) == 64
            assert layouts[0]["passivity"]["passive"] is True
            assert layouts[0]["reverse_ports"] is True
            assert layouts[0]["reference_impedance_mode"] == "system"
            assert layouts[0]["passivity"]["ports_reversed"] is True
            assert layouts[0]["passivity"]["renormalized"] is True
            assert layouts[0]["passivity"]["deembedded"] is True
            assert layouts[0]["fixtures"]["left"]["filename"] == "fixture.s2p"
            assert layouts[0]["passivity"]["deembedding"]["maximum_recascade_residual"] < 1e-12
        finally:
            (
                server.state.loaded_snp, server.state.loaded_snp_filename,
                server.state.snp_dir, server.state.component_library,
                server.state.db_library, server.state.use_db,
            ) = previous
            reset_session()
