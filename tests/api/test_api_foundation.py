"""Regression tests for API state, request contracts, and safe file access."""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

from fastapi import HTTPException

from api.models import (
    ComponentAlternativesRequest,
    ComponentLibraryPreviewRequest,
    ComponentParameterFilter,
    DataDirConfig,
    EfficiencyClearRequest,
    EfficiencyInlineRequest,
    ManualRefineRequest,
    ManualTuneRequest,
    ManualYieldRequest,
    OptimizeRequest,
    SNPImportRequest,
    TuningOptimizeRequest,
    TuningContinueRequest,
    TuningYieldRequest,
    TuneSingleRequest,
)
import api.server as server
from engine.tuning_service import PerPortTuningMetrics, TuningResult
from engine.component_lib import ComponentInfo, ComponentLibrary
from engine.touchstone import TouchstoneData, parse_touchstone, touchstone_network_sha256
from api.server import (
    _safe_data_path,
    clear_efficiency,
    efficiency_status,
    load_efficiency_inline,
    state,
)


class ApiFoundationTests(unittest.TestCase):
    def tearDown(self):
        state.clear_efficiency()

    def test_multi_scenario_job_uses_shared_network_plan_and_generic_status_store(self):
        def fake_runner(job_id, payload):
            server._execute_tuning_job(job_id, lambda progress, cancelled: {
                "status": "ok", "solutions_count": 0, "solutions": [],
            })

        payload = {
            "scenarios": [{"snp_filename": "a.s1p"}, {"snp_filename": "b.s1p"}],
            "timeout_seconds": 15, "search_quality": "quick", "beam_width": 8,
            "num_band_points": 3, "verification_band_points": 21,
            "topology_names": ["1-Element (Series-L)"],
        }
        started = server._start_tuning_job_thread(
            fake_runner, payload, "test-multi",
            plan_builder=server.build_multi_scenario_search_plan,
            job_type="multi_scenario",
        )
        deadline = time.time() + 2
        while time.time() < deadline:
            with server._tuning_jobs_lock:
                job = server._public_tuning_job(server._tuning_jobs[started["job_id"]])
            if job["status"] == "completed":
                break
            time.sleep(0.01)

        self.assertEqual(job["job_type"], "multi_scenario")
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["search_plan"]["strategy"], "shared_network_measured_beam")
        self.assertEqual(job["search_plan"]["scenario_count"], 2)

    def test_ideal_manual_tuning_does_not_require_component_library(self):
        original = (
            server.state.loaded_snp,
            server.state.loaded_snp_filename,
            server.state.component_library,
            server.state.db_library,
            server.state.use_db,
            server.run_manual_tuning_physical,
            server.touchstone_network_sha256,
        )
        captured = {}

        def fake_manual(dut, library, **kwargs):
            captured["dut"] = dut
            captured["library"] = library
            captured.update(kwargs)
            return {"status": "ok"}

        try:
            server.state.loaded_snp = object()
            server.state.loaded_snp_filename = "dut.s1p"
            server.state.component_library = None
            server.state.db_library = None
            server.state.use_db = False
            server.run_manual_tuning_physical = fake_manual
            server.touchstone_network_sha256 = lambda dut: "a" * 64

            response = asyncio.run(server.manual_tune(ManualTuneRequest(
                snp_filename="dut.s1p",
                components=[{
                    "comp_type": "inductor", "connection_type": "series",
                    "value": 3.3, "use_ideal": True,
                }],
            )))

            self.assertEqual(response, {
                "status": "ok",
                "dut_identity": {"filename": "dut.s1p", "network_sha256": "a" * 64},
            })
            self.assertIsNone(captured["library"])
        finally:
            (
                server.state.loaded_snp,
                server.state.loaded_snp_filename,
                server.state.component_library,
                server.state.db_library,
                server.state.use_db,
                server.run_manual_tuning_physical,
                server.touchstone_network_sha256,
            ) = original

    def test_measured_manual_tuning_still_requires_component_library(self):
        original = (
            server.state.loaded_snp,
            server.state.loaded_snp_filename,
            server.state.component_library,
            server.state.db_library,
            server.state.use_db,
        )
        try:
            server.state.loaded_snp = object()
            server.state.loaded_snp_filename = "dut.s1p"
            server.state.component_library = None
            server.state.db_library = None
            server.state.use_db = False

            with self.assertRaisesRegex(HTTPException, "Component library not loaded"):
                asyncio.run(server.manual_tune(ManualTuneRequest(
                    snp_filename="dut.s1p",
                    components=[{
                        "comp_type": "capacitor", "connection_type": "shunt",
                        "value": 1.2, "use_ideal": False,
                    }],
                )))
        finally:
            (
                server.state.loaded_snp,
                server.state.loaded_snp_filename,
                server.state.component_library,
                server.state.db_library,
                server.state.use_db,
            ) = original

    def test_manual_tuning_rejects_stale_dut_filename_and_revision(self):
        original = (
            server.state.loaded_snp, server.state.loaded_snp_filename,
            server.touchstone_network_sha256,
        )
        try:
            server.state.loaded_snp = object()
            server.state.loaded_snp_filename = "current.s1p"
            server.touchstone_network_sha256 = lambda dut: "b" * 64
            with self.assertRaisesRegex(HTTPException, "DUT changed") as filename_error:
                asyncio.run(server.manual_tune(ManualTuneRequest(snp_filename="old.s1p")))
            self.assertEqual(filename_error.exception.status_code, 409)
            with self.assertRaisesRegex(HTTPException, "revision changed") as revision_error:
                asyncio.run(server.manual_tune(ManualTuneRequest(
                    snp_filename="current.s1p", expected_network_sha256="a" * 64,
                )))
            self.assertEqual(revision_error.exception.status_code, 409)
        finally:
            (
                server.state.loaded_snp, server.state.loaded_snp_filename,
                server.touchstone_network_sha256,
            ) = original

    def test_manual_refinement_job_binds_dut_identity_and_fixed_topology_plan(self):
        original = (
            server.state.loaded_snp, server.state.loaded_snp_filename,
            server.touchstone_network_sha256, server._start_tuning_job_thread,
        )
        captured = {}
        try:
            server.state.loaded_snp = object()
            server.state.loaded_snp_filename = "current.s1p"
            server.touchstone_network_sha256 = lambda dut: "c" * 64

            def fake_start(target, payload, prefix, **kwargs):
                captured.update(payload=payload, prefix=prefix, kwargs=kwargs)
                return {"job_id": "manual-job", "job_type": kwargs["job_type"]}

            server._start_tuning_job_thread = fake_start
            response = asyncio.run(server.start_manual_refine_job(ManualRefineRequest(
                snp_filename="current.s1p",
                expected_network_sha256="c" * 64,
                components=[{"comp_type": "inductor", "connection_type": "series",
                             "value": 10, "use_ideal": True}],
                bands_mhz=[[900, 1100]],
            )))
            self.assertEqual(response["job_type"], "manual_refine")
            self.assertEqual(captured["payload"]["components"][0]["value"], 10)
            plan = captured["kwargs"]["plan_builder"](captured["payload"])
            self.assertEqual(plan["variable_count"], 1)
            self.assertEqual(plan["estimated_evaluations"], 112)
        finally:
            (
                server.state.loaded_snp, server.state.loaded_snp_filename,
                server.touchstone_network_sha256, server._start_tuning_job_thread,
            ) = original

    def test_manual_yield_job_uses_sample_budget_and_dut_identity(self):
        original = (
            server.state.loaded_snp, server.state.loaded_snp_filename,
            server.touchstone_network_sha256, server._start_tuning_job_thread,
        )
        captured = {}
        try:
            server.state.loaded_snp = object()
            server.state.loaded_snp_filename = "current.s1p"
            server.touchstone_network_sha256 = lambda dut: "d" * 64

            def fake_start(target, payload, prefix, **kwargs):
                captured.update(payload=payload, kwargs=kwargs)
                return {"job_id": "yield-job", "job_type": kwargs["job_type"]}

            server._start_tuning_job_thread = fake_start
            response = asyncio.run(server.start_manual_yield_job(ManualYieldRequest(
                snp_filename="current.s1p",
                expected_network_sha256="d" * 64,
                components=[{"comp_type": "inductor", "connection_type": "series",
                             "value": 1.0, "use_ideal": True}],
                bands_mhz=[[900, 1100]],
                samples=320,
            )))
            self.assertEqual(response["job_type"], "manual_yield")
            plan = captured["kwargs"]["plan_builder"](captured["payload"])
            self.assertEqual(plan["estimated_evaluations"], 320)
            self.assertEqual(plan["samples"], 320)
        finally:
            (
                server.state.loaded_snp, server.state.loaded_snp_filename,
                server.touchstone_network_sha256, server._start_tuning_job_thread,
            ) = original

    def test_inline_global_efficiency_is_visible_in_status(self):
        request = EfficiencyInlineRequest(
            content="% Freq(MHz) eff(lin)\n2400 0.75\n2500 0.80",
        )
        asyncio.run(load_efficiency_inline(request))
        status = asyncio.run(efficiency_status())

        self.assertTrue(status["loaded"])
        self.assertEqual(status["global"]["num_points"], 2)
        self.assertAlmostEqual(status["global"]["efficiency_min"], 0.75)

        asyncio.run(clear_efficiency(EfficiencyClearRequest()))
        self.assertFalse(asyncio.run(efficiency_status())["loaded"])

    def test_data_dirs_load_audited_component_environment_sidecar(self):
        attribute_names = (
            "snp_dir", "murata_dir", "optenni_component_dir",
            "environment_metadata_path", "component_environment_catalog",
            "component_library", "full_component_library", "db_library",
            "use_db", "tunable_component_library",
        )
        original = {name: getattr(server.state, name) for name in attribute_names}
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snp_dir = root / "snp"
                component_dir = root / "components"
                snp_dir.mkdir()
                component_dir.mkdir()
                sidecar = component_dir / "component_environment.json"
                sidecar.write_text(json.dumps({
                    "schema_version": 1,
                    "source": {
                        "name": "test fixture",
                        "evidence_level": "engineering_assumption",
                    },
                    "components": [{
                        "part_number": "TEST-L1",
                        "tempco_ppm_per_c": 100.0,
                    }],
                }), encoding="utf-8")

                response = asyncio.run(server.set_data_dirs(DataDirConfig(
                    snp_dir=str(snp_dir), murata_dir=str(component_dir),
                )))

                self.assertEqual(response["mode"], "s2p_scan")
                self.assertEqual(response["environment_metadata"]["component_count"], 1)
                self.assertEqual(response["environment_metadata"]["matched_components"], 0)
                self.assertEqual(
                    server.state.component_environment_catalog.lookup("test-l1").tempco_ppm_per_c,
                    100.0,
                )
        finally:
            for name, value in original.items():
                setattr(server.state, name, value)
            server.reset_session()

    def test_safe_data_path_rejects_parent_traversal(self):
        original = state.snp_dir
        try:
            with tempfile.TemporaryDirectory() as directory:
                state.snp_dir = directory
                with self.assertRaises(HTTPException) as context:
                    _safe_data_path(os.path.join("..", "outside.s1p"))
                self.assertEqual(context.exception.status_code, 400)
        finally:
            state.snp_dir = original

    def test_cst_touchstone_import_validates_and_preserves_existing_files(self):
        original = state.snp_dir
        content = "# GHZ S RI R 50\n1.0 0.1 0\n1.1 0.2 0\n"
        try:
            with tempfile.TemporaryDirectory() as directory:
                state.snp_dir = directory
                first = asyncio.run(server.import_snp(SNPImportRequest(
                    filename="cst_antenna.s1p", content=content, source="CST",
                )))
                second = asyncio.run(server.import_snp(SNPImportRequest(
                    filename="cst_antenna.s1p", content=content, source="CST",
                )))
                self.assertEqual(first["filename"], "cst_antenna.s1p")
                self.assertEqual(second["filename"], "cst_antenna-2.s1p")
                self.assertEqual(first["num_ports"], 1)
                self.assertEqual(first["freq_count"], 2)
                self.assertEqual(first["source"], "CST")
                self.assertTrue(Path(directory, first["filename"]).is_file())
                self.assertTrue(Path(directory, second["filename"]).is_file())
        finally:
            state.snp_dir = original

    def test_cst_touchstone_import_rejects_invalid_content_without_writing(self):
        original = state.snp_dir
        try:
            with tempfile.TemporaryDirectory() as directory:
                state.snp_dir = directory
                with self.assertRaisesRegex(HTTPException, "Invalid CST Touchstone"):
                    asyncio.run(server.import_snp(SNPImportRequest(
                        filename="broken.s2p", content="# GHZ S RI R 50\n1.0 0.1 0\n", source="CST",
                    )))
                self.assertEqual(list(Path(directory).iterdir()), [])
        finally:
            state.snp_dir = original

    def test_touchstone_network_fingerprint_ignores_text_but_detects_numeric_changes(self):
        first = parse_touchstone(
            "! export at 10:00\n# GHZ S RI R 50\n1.0 0.1 0\n1.1 0.2 0\n",
            "first.s1p",
        )
        reformatted = parse_touchstone(
            "! export at 10:01\n# GHZ S RI R 50\n1.000 0.1000 0.000\n1.100 0.200 0\n",
            "second.s1p",
        )
        changed = parse_touchstone(
            "# GHZ S RI R 50\n1.0 0.11 0\n1.1 0.2 0\n",
            "changed.s1p",
        )
        self.assertEqual(
            touchstone_network_sha256(first), touchstone_network_sha256(reformatted)
        )
        self.assertNotEqual(
            touchstone_network_sha256(first), touchstone_network_sha256(changed)
        )

    def test_snp_load_reports_strict_format_errors_without_replacing_active_dut(self):
        original = (state.snp_dir, state.loaded_snp, state.loaded_snp_filename)
        active = SimpleNamespace(filename="active.s1p")
        try:
            with tempfile.TemporaryDirectory() as directory:
                state.snp_dir = directory
                state.loaded_snp = active
                state.loaded_snp_filename = "active.s1p"
                Path(directory, "invalid.s1p").write_text(
                    "# GHZ S RI R 75\n2.0 0.1 0\n1.0 0.2 0\n",
                    encoding="ascii",
                )
                with self.assertRaisesRegex(HTTPException, "strictly increasing") as context:
                    asyncio.run(server.load_snp("invalid.s1p"))
                self.assertEqual(context.exception.status_code, 400)
                self.assertIs(state.loaded_snp, active)
                self.assertEqual(state.loaded_snp_filename, "active.s1p")
        finally:
            state.snp_dir, state.loaded_snp, state.loaded_snp_filename = original

    def test_snp_load_exposes_format_and_reference_impedance(self):
        original = (state.snp_dir, state.loaded_snp, state.loaded_snp_filename)
        try:
            with tempfile.TemporaryDirectory() as directory:
                state.snp_dir = directory
                Path(directory, "valid.s2p").write_text(
                    "# GHZ S DB R 75\n"
                    "1.0 -20 0 -30 10 -25 -15 -18 5\n",
                    encoding="ascii",
                )
                response = asyncio.run(server.load_snp("valid.s2p"))
                self.assertEqual(response["num_ports"], 2)
                self.assertEqual(response["parameter_format"], "DB")
                self.assertEqual(response["reference_impedance_ohm"], 75.0)
                self.assertEqual(response["reference_impedances_ohm"], [75.0, 75.0])
                self.assertRegex(response["network_sha256"], r"^[0-9a-f]{64}$")
        finally:
            state.snp_dir, state.loaded_snp, state.loaded_snp_filename = original
            server.reset_session()

    def test_snp_load_preserves_session_only_for_same_electrical_network(self):
        original = (state.snp_dir, state.loaded_snp, state.loaded_snp_filename, state.last_solutions)
        try:
            with tempfile.TemporaryDirectory() as directory:
                state.snp_dir = directory
                path = Path(directory, "stable.s1p")
                path.write_text(
                    "! CST run 1\n# GHZ S RI R 50\n1.0 0.1 0\n1.1 0.2 0\n",
                    encoding="ascii",
                )
                first = asyncio.run(server.load_snp("stable.s1p"))
                self.assertTrue(first["electrical_dut_changed"])
                candidate = TuningResult(port_indices=[0])
                session = server.get_session()
                session.candidate_solutions = [candidate]
                state.last_solutions = [candidate]

                path.write_text(
                    "! CST run 2\n# GHZ S RI R 50\n1.000 0.1000 0\n1.100 0.200 0\n",
                    encoding="ascii",
                )
                same = asyncio.run(server.load_snp("stable.s1p"))
                self.assertFalse(same["electrical_dut_changed"])
                self.assertIs(server.get_session(), session)
                self.assertIs(server.get_session().candidate_solutions[0], candidate)
                self.assertIs(state.last_solutions[0], candidate)

                path.write_text(
                    "# GHZ S RI R 50\n1.0 0.11 0\n1.1 0.2 0\n",
                    encoding="ascii",
                )
                changed = asyncio.run(server.load_snp("stable.s1p"))
                self.assertTrue(changed["electrical_dut_changed"])
                self.assertEqual(server.get_session().candidate_solutions, [])
                self.assertEqual(state.last_solutions, [])
        finally:
            state.snp_dir, state.loaded_snp, state.loaded_snp_filename, state.last_solutions = original
            server.reset_session()

    def test_snp_load_exposes_per_port_reference_impedances(self):
        original = (state.snp_dir, state.loaded_snp, state.loaded_snp_filename)
        try:
            with tempfile.TemporaryDirectory() as directory:
                state.snp_dir = directory
                Path(directory, "per-port.s2p").write_text(
                    "[Version] 2.0\n[Number of Ports] 2\n[Reference] 50 75\n"
                    "# GHZ S RI R 50\n[Network Data]\n"
                    "1.0 0.1 0 0.2 0 0.3 0 0.4 0\n[End]\n",
                    encoding="ascii",
                )
                response = asyncio.run(server.load_snp("per-port.s2p"))
                self.assertIsNone(response["reference_impedance_ohm"])
                self.assertEqual(response["reference_impedances_ohm"], [50.0, 75.0])
        finally:
            state.snp_dir, state.loaded_snp, state.loaded_snp_filename = original
            server.reset_session()

    def test_request_collection_defaults_are_not_shared(self):
        first = OptimizeRequest(snp_filename="a.s1p")
        second = OptimizeRequest(snp_filename="b.s1p")
        first.port_states.append({"port_index": 0, "state": "load"})
        self.assertEqual(second.port_states, [])

    def test_component_series_selection_filters_the_actual_tuning_library(self):
        original = (
            server.state.loaded_snp,
            server.state.component_library,
            server.state.full_component_library,
            server.state.db_library,
            server.state.use_db,
            server.run_tuning_single,
        )
        full = ComponentLibrary()
        for component_type, series, part, value in (
            ("inductor", "L_A", "LA1", 1.0),
            ("inductor", "L_B", "LB1", 2.0),
            ("capacitor", "C_A", "CA1", 1.0),
            ("capacitor", "C_B", "CB1", 2.0),
        ):
            component = ComponentInfo(
                part, f"/{series}/{part}.s2p", "__DIR__", component_type,
                value, "nH" if component_type == "inductor" else "pF",
            )
            component.series = series
            component.manufacturer = "Murata" if series.endswith("B") else "TDK"
            component.size_code = "0402" if series.endswith("B") else "0603"
            component.tolerance_pct = 2.0 if series.endswith("B") else 10.0
            component.voltage_code = "1H" if series.endswith("B") else "2A"
            component.dielectric = "C0G" if series.endswith("B") else "X7R"
            component.metadata_provenance = {
                "manufacturer": "database", "package_code": "database",
                "tolerance_pct": "database",
                "voltage_code": "part_number_inferred", "dielectric": "part_number_inferred",
            }
            delta = 0.01 if part.endswith("1") and series.endswith("A") else 0.02
            component._data = TouchstoneData(
                filename=f"{part}.s2p", frequency_unit="HZ", parameter_type="S",
                data_format="RI", reference_resistance=50.0, num_ports=2,
                frequencies=[2.4e9, 2.5e9],
                sparameters={
                    (1, 1): [delta + 0j, delta + 0j],
                    (2, 1): [1 - delta + 0j, 1 - delta + 0j],
                    (1, 2): [1 - delta + 0j, 1 - delta + 0j],
                    (2, 2): [delta + 0j, delta + 0j],
                },
            )
            full.add_component(component)
        default = server.filter_component_library_by_series(full, ["L::L_A", "C::C_A"])
        captured = {}
        candidate = TuningResult(port_indices=[0], mode="single", system_score=-1.0)

        def fake_single(**kwargs):
            captured["inductors"] = [item.part_number for item in kwargs["library"].inductors]
            captured["capacitors"] = [item.part_number for item in kwargs["library"].capacitors]
            return {0: candidate}

        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=1)
            server.state.component_library = default
            server.state.full_component_library = full
            server.state.db_library = None
            server.state.use_db = False
            server.run_tuning_single = fake_single
            inventory = asyncio.run(server.list_component_series())
            self.assertEqual(
                {item["id"] for item in inventory["series"]},
                {"L::L_A", "L::L_B", "C::C_A", "C::C_B"},
            )
            self.assertEqual(inventory["default_selected"], ["C::C_A", "L::L_A"])
            self.assertEqual(inventory["facets"]["manufacturers"], ["Murata", "TDK"])
            self.assertEqual(inventory["facets"]["dielectrics"], ["C0G", "X7R"])
            self.assertEqual(inventory["facets"]["voltage_codes"], ["1H", "2A"])
            preview = asyncio.run(server.preview_component_library(
                ComponentLibraryPreviewRequest(
                    component_series=["L::L_B", "C::C_B"],
                    component_filter=ComponentParameterFilter(
                        manufacturers=["Murata"], package_codes=["0402"],
                        maximum_tolerance_pct=5.0,
                        voltage_codes=["1H"], dielectrics=["C0G"],
                        unknown_metadata_policy="exclude",
                    ),
                )
            ))
            self.assertTrue(preview["valid_for_lc_search"])
            self.assertTrue(preview["valid_for_measured_search"])
            self.assertEqual((preview["inductors"], preview["capacitors"]), (1, 1))
            self.assertEqual(preview["filter_statistics"]["matched_components"], 2)
            inductor_only, inductor_metadata = server._library_for_series(
                ["L::L_B"], required_component_types={"inductors"}
            )
            self.assertEqual([item.part_number for item in inductor_only.inductors], ["LB1"])
            self.assertEqual(inductor_only.capacitors, [])
            self.assertEqual(inductor_metadata["capacitors"], 0)
            detail = asyncio.run(server.component_detail("LA1"))
            self.assertEqual(detail["matches"][0]["manufacturer"], "TDK")
            alternatives = asyncio.run(server.component_alternatives(
                ComponentAlternativesRequest(
                    part_number="LA1",
                    component_series=["L::L_A", "L::L_B", "C::C_A", "C::C_B"],
                    bands_mhz=[[2400, 2500]], maximum_nominal_deviation_pct=200,
                )
            ))
            self.assertEqual(alternatives["alternatives"][0]["part_number"], "LB1")
            self.assertEqual(alternatives["physically_evaluated"], 1)
            self.assertGreater(alternatives["alternatives"][0]["sparameter_rms_difference"], 0)
            response = asyncio.run(server._tuning_optimize_impl(TuningOptimizeRequest(
                ports=[{"port_index": 0, "enabled": True, "bands_mhz": [[2400, 2500]], "max_components": 1}],
                component_series=["L::L_B", "C::C_B"],
            )))

            self.assertEqual(captured, {"inductors": ["LB1"], "capacitors": ["CB1"]})
            filter_metadata = response["component_library_filter"]
            self.assertEqual({
                key: filter_metadata[key]
                for key in ("mode", "selected_series", "inductors", "capacitors")
            }, {
                "mode": "selected_series",
                "selected_series": ["C::C_B", "L::L_B"],
                "inductors": 1,
                "capacitors": 1,
            })
            self.assertEqual(len(filter_metadata["catalog_fingerprint"]), 64)
            self.assertEqual(
                response["best_solution"]["search_diagnostics"]["component_library_filter"],
                response["component_library_filter"],
            )
            inductor_only_response = asyncio.run(server._tuning_optimize_impl(
                TuningOptimizeRequest(
                    ports=[{
                        "port_index": 0, "enabled": True,
                        "bands_mhz": [[2400, 2500]], "max_components": 1,
                    }],
                    component_series=["L::L_B"],
                )
            ))
            self.assertEqual(captured, {"inductors": ["LB1"], "capacitors": []})
            self.assertEqual(
                inductor_only_response["component_library_filter"]["selected_series"],
                ["L::L_B"],
            )
            with self.assertRaises(HTTPException) as context:
                asyncio.run(server._tuning_optimize_impl(TuningOptimizeRequest(
                    component_series=[],
                )))
            self.assertEqual(context.exception.status_code, 400)
        finally:
            (
                server.state.loaded_snp,
                server.state.component_library,
                server.state.full_component_library,
                server.state.db_library,
                server.state.use_db,
                server.run_tuning_single,
            ) = original
            server.reset_session()

    def test_component_alternatives_rank_measured_smatrix_similarity(self):
        original_state = (
            server.state.component_library, server.state.full_component_library,
            server.state.db_library, server.state.use_db,
        )
        library = ComponentLibrary()

        def component(part, delta):
            item = SimpleNamespace(
                part_number=part, s2p_filename=f"/{part}.s2p", zip_path="__DIR__",
                component_type="inductor", nominal_value=5.6, nominal_unit="nH",
                series="Measured family", manufacturer="Vendor", size_code="0402",
                tolerance_pct=5.0, voltage_code="", dielectric="",
                metadata_provenance={"manufacturer": "database", "package_code": "database"},
            )
            item.get_s_matrix_at_freq = lambda _frequency: np.array([
                [delta + 0j, 1 - delta + 0j],
                [1 - delta + 0j, delta + 0j],
            ])
            return item

        try:
            library.add_component(component("REFERENCE", 0.05))
            library.add_component(component("CLOSE", 0.051))
            library.add_component(component("FAR", 0.20))
            server.state.component_library = library
            server.state.full_component_library = library
            server.state.db_library = None
            server.state.use_db = False
            result = asyncio.run(server.component_alternatives(ComponentAlternativesRequest(
                part_number="REFERENCE", bands_mhz=[[2400, 2500]], limit=2,
            )))
            self.assertEqual([item["part_number"] for item in result["alternatives"]], ["CLOSE", "FAR"])
            self.assertLess(
                result["alternatives"][0]["sparameter_rms_difference"],
                result["alternatives"][1]["sparameter_rms_difference"],
            )
        finally:
            (
                server.state.component_library, server.state.full_component_library,
                server.state.db_library, server.state.use_db,
            ) = original_state

    def test_deprecated_single_endpoint_is_only_an_authoritative_service_adapter(self):
        original = (
            server.state.loaded_snp, server.state.loaded_snp_filename,
            server.state.component_library, server.state.full_component_library,
            server.state.db_library, server.state.use_db, server.run_tuning_single,
        )
        library = ComponentLibrary()
        library.add_component(SimpleNamespace(
            part_number="L1", component_type="inductor", nominal_value=1.0,
            nominal_unit="nH", s2p_filename="/L1.s2p", zip_path="__DIR__",
        ))
        library.add_component(SimpleNamespace(
            part_number="C1", component_type="capacitor", nominal_value=1.0,
            nominal_unit="pF", s2p_filename="/C1.s2p", zip_path="__DIR__",
        ))
        candidate = TuningResult(
            port_indices=[0], mode="single", system_score=-0.25,
            avg_total_efficiency=0.9, min_total_efficiency=0.8,
            per_port={0: PerPortTuningMetrics(
                port_index=0, band_total_eff=[0.8, 0.9],
            )},
        )
        captured = {}

        def fake_authoritative(**kwargs):
            captured.update(kwargs)
            return {0: candidate}

        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=1)
            server.state.loaded_snp_filename = "dut.s1p"
            server.state.component_library = library
            server.state.full_component_library = library
            server.state.db_library = None
            server.state.use_db = False
            server.run_tuning_single = fake_authoritative
            response = asyncio.run(server.tune_single(TuneSingleRequest(
                bands_mhz=[[2400, 2500]], topology_filter=["1-Element (Series-L)"],
            )))
            self.assertIs(captured["dut"], server.state.loaded_snp)
            self.assertIs(captured["library"], library)
            self.assertEqual(captured["topology_filter"], ["1-Element (Series-L)"])
            self.assertTrue(response["deprecated_endpoint"])
            self.assertEqual(response["authoritative_solver"], "run_tuning_single")
            self.assertEqual(response["best_score"], -0.25)
            self.assertIs(server.get_session().candidate_solutions[0], candidate)
        finally:
            (
                server.state.loaded_snp, server.state.loaded_snp_filename,
                server.state.component_library, server.state.full_component_library,
                server.state.db_library, server.state.use_db, server.run_tuning_single,
            ) = original
            server.reset_session()

    def test_legacy_solver_routes_are_marked_deprecated_in_openapi(self):
        paths = server.app.openapi()["paths"]
        for path in ("/api/optimize", "/api/multipass", "/api/joint-optimize", "/api/tune/single", "/api/tune/joint"):
            self.assertTrue(paths[path]["post"]["deprecated"], path)
        self.assertFalse(paths["/api/tuning/optimize"]["post"].get("deprecated", False))

    def test_zero_component_unified_api_does_not_require_a_component_library(self):
        original = (
            server.state.loaded_snp, server.state.loaded_snp_filename,
            server.state.component_library, server.state.full_component_library,
            server.state.db_library, server.state.use_db,
        )
        try:
            server.state.loaded_snp = parse_touchstone(
                "# GHZ S RI R 50\n2.4 0.2 0.0\n2.5 0.1 0.0\n",
                filename="bare.s1p",
            )
            server.state.loaded_snp_filename = "bare.s1p"
            server.state.component_library = None
            server.state.full_component_library = None
            server.state.db_library = None
            server.state.use_db = False
            response = asyncio.run(server._tuning_optimize_impl(TuningOptimizeRequest(
                ports=[{
                    "port_index": 0, "enabled": True,
                    "bands_mhz": [[2400, 2500]], "max_components": 0,
                }],
                objective="balanced", mode="single", num_band_points=2,
            )))
            self.assertEqual(response["component_library_filter"]["mode"], "not_required")
            self.assertEqual(response["best_solution"]["total_component_count"], 0)
            self.assertTrue(
                response["best_solution"]["search_diagnostics"]["bare_dut_core_baseline"]
            )
        finally:
            (
                server.state.loaded_snp, server.state.loaded_snp_filename,
                server.state.component_library, server.state.full_component_library,
                server.state.db_library, server.state.use_db,
            ) = original
            server.reset_session()

    def test_zero_component_joint_api_does_not_require_a_component_library(self):
        original = (
            server.state.loaded_snp, server.state.loaded_snp_filename,
            server.state.component_library, server.state.full_component_library,
            server.state.db_library, server.state.use_db,
        )
        try:
            server.state.loaded_snp = parse_touchstone(
                "# GHZ S RI R 50\n"
                "2.4 0.2 0 0.1 0 0.1 0 0.15 0\n"
                "2.5 0.1 0 0.1 0 0.1 0 0.12 0\n",
                filename="bare-joint.s2p",
            )
            server.state.loaded_snp_filename = "bare-joint.s2p"
            server.state.component_library = None
            server.state.full_component_library = None
            server.state.db_library = None
            server.state.use_db = False
            response = asyncio.run(server._tuning_optimize_impl(TuningOptimizeRequest(
                ports=[
                    {"port_index": 0, "enabled": True, "bands_mhz": [[2400, 2500]], "max_components": 0},
                    {"port_index": 1, "enabled": True, "bands_mhz": [[2400, 2500]], "max_components": 0},
                ],
                objective="balanced", mode="joint", num_band_points=2,
            )))
            self.assertEqual(response["component_library_filter"]["mode"], "not_required")
            self.assertEqual(response["best_solution"]["total_component_count"], 0)
            self.assertEqual(
                response["best_solution"]["efficiency_basis"],
                "rfmatch_core_physical_bare_dut_joint",
            )
            self.assertTrue(
                response["best_solution"]["search_diagnostics"]["bare_dut_core_baseline"]
            )
        finally:
            (
                server.state.loaded_snp, server.state.loaded_snp_filename,
                server.state.component_library, server.state.full_component_library,
                server.state.db_library, server.state.use_db,
            ) = original
            server.reset_session()

    def test_continue_rerun_merges_without_regressing_previous_best(self):
        original_impl = server._tuning_optimize_impl
        old = TuningResult(port_indices=[0], mode="single", system_score=-1.0)
        new = TuningResult(port_indices=[0], mode="single", system_score=-2.0)

        async def fake_impl(request, **_kwargs):
            session = server.get_session()
            session.last_tuning_request = (
                request.model_dump(mode="json") if hasattr(request, "model_dump") else request.dict()
            )
            session.candidate_solutions = [new]
            return {"status": "ok", "solutions": [new.to_dict()], "best_solution": new.to_dict()}

        try:
            server.reset_session()
            session = server.get_session()
            session.candidate_solutions = [old]
            session.last_tuning_request = {
                "ports": [{"port_index": 0, "enabled": True, "bands_mhz": [[2400, 2500]], "max_components": 2}],
                "mode": "single", "timeout_seconds": 5.0,
            }
            session.restoration_mode = "live"
            server._tuning_optimize_impl = fake_impl
            response = asyncio.run(server.tuning_continue(TuningContinueRequest(additional_seconds=7.0)))
            self.assertEqual(response["best_score"], -1.0)
            self.assertEqual(response["continuation"]["total_timeout_seconds"], 12.0)
            self.assertEqual(response["continuation"]["strategy"], "deterministic_rerun_merge")
            self.assertFalse(response["continuation"]["checkpoint_reused"])
            self.assertEqual(server.get_session().last_tuning_request["timeout_seconds"], 12.0)
        finally:
            server._tuning_optimize_impl = original_impl
            server.reset_session()

    def test_continue_passes_live_checkpoint_and_additional_budget_slice(self):
        original_impl = server._tuning_optimize_impl
        old = TuningResult(port_indices=[0], mode="single", system_score=-2.0)
        new = TuningResult(port_indices=[0], mode="single", system_score=-1.0)
        checkpoint = {"kind": "single_measured_s2p", "optimizer": object()}
        captured = {}

        async def fake_impl(request, **kwargs):
            captured.update(kwargs)
            session = server.get_session()
            session.last_tuning_request = (
                request.model_dump(mode="json") if hasattr(request, "model_dump") else request.dict()
            )
            session.candidate_solutions = [new]
            session.search_checkpoint = {
                "kind": "single_measured_s2p", "optimizer": checkpoint["optimizer"],
                "resumed": True,
            }
            return {"status": "ok", "solutions": [new.to_dict()], "best_solution": new.to_dict()}

        try:
            server.reset_session()
            session = server.get_session()
            session.candidate_solutions = [old]
            session.search_checkpoint = checkpoint
            session.last_tuning_request = {
                "ports": [{"port_index": 0, "enabled": True, "bands_mhz": [[2400, 2500]], "max_components": 2}],
                "mode": "single", "timeout_seconds": 5.0,
            }
            session.restoration_mode = "live"
            server._tuning_optimize_impl = fake_impl
            response = asyncio.run(server.tuning_continue(TuningContinueRequest(additional_seconds=7.0)))

            self.assertIs(captured["resume_checkpoint"], checkpoint)
            self.assertEqual(captured["continuation_budget_seconds"], 7.0)
            self.assertEqual(response["best_score"], -1.0)
            self.assertEqual(response["continuation"]["strategy"], "in_memory_measured_checkpoint")
            self.assertTrue(response["continuation"]["checkpoint_reused"])
        finally:
            server._tuning_optimize_impl = original_impl
            server.reset_session()

    def test_tuning_yield_endpoint_passes_contract_to_service(self):
        original = (
            server.state.loaded_snp,
            server.state.component_library,
            server.state.db_library,
            server.state.use_db,
            server.run_tuning_yield_analysis,
        )
        captured = {}

        def fake_analysis(dut, library, candidates, tuning_request, **kwargs):
            captured.update(kwargs)
            captured["candidate_count"] = len(candidates)
            captured["tuning_request"] = tuning_request
            return {"status": "ok", "ranked_candidates": [{"solution_index": 0}]}

        try:
            server.state.loaded_snp = object()
            server.state.component_library = object()
            server.state.db_library = None
            server.state.use_db = False
            server.run_tuning_yield_analysis = fake_analysis
            server.reset_session()
            session = server.get_session()
            session.candidate_solutions = [TuningResult(port_indices=[0])]
            session.last_tuning_request = {"ports": [{"port_index": 0, "enabled": True}]}

            response = asyncio.run(server.tuning_yield_analysis(TuningYieldRequest(
                solution_indices=[0], samples=80, seed=19,
                distribution="uniform", confidence_level=0.9,
                minimum_total_efficiency=0.6,
                minimum_return_loss_db=8.0,
                default_tolerance_pct=7.5,
                batch_correlation=0.4,
                temperature_min_c=-40.0,
                temperature_max_c=85.0,
                inductor_tempco_ppm_per_c=100.0,
                capacitor_tempco_ppm_per_c=-30.0,
                inductor_bias_pct=1.5,
                capacitor_bias_pct=-0.75,
            )))

            self.assertEqual(response["ranked_candidates"][0]["solution_index"], 0)
            self.assertEqual(captured["candidate_count"], 1)
            self.assertEqual(captured["samples"], 80)
            self.assertEqual(captured["seed"], 19)
            self.assertEqual(captured["distribution"], "uniform")
            self.assertEqual(captured["confidence_level"], 0.9)
            self.assertEqual(captured["default_tolerance_pct"], 7.5)
            self.assertEqual(captured["batch_correlation"], 0.4)
            self.assertEqual(captured["temperature_min_c"], -40.0)
            self.assertEqual(captured["temperature_max_c"], 85.0)
            self.assertEqual(captured["inductor_tempco_ppm_per_c"], 100.0)
            self.assertEqual(captured["capacitor_tempco_ppm_per_c"], -30.0)
            self.assertEqual(captured["inductor_bias_pct"], 1.5)
            self.assertEqual(captured["capacitor_bias_pct"], -0.75)
        finally:
            (
                server.state.loaded_snp,
                server.state.component_library,
                server.state.db_library,
                server.state.use_db,
                server.run_tuning_yield_analysis,
            ) = original
            server.reset_session()

    def test_tunable_library_scan_targets_only_required_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inductor_dir = root / "Inductors" / "Coilcraft Inductors 0402cs"
            capacitor_dir = root / "Capacitors" / "Murata Capacitors gjm15"
            unrelated_dir = root / "Inductors" / "Unrelated"
            inductor_dir.mkdir(parents=True)
            capacitor_dir.mkdir(parents=True)
            unrelated_dir.mkdir(parents=True)
            (inductor_dir / "04CS15N.S2P").write_text("", encoding="ascii")
            (capacitor_dir / "GJM1553C1H2R8WB01.s2p").write_text("", encoding="ascii")
            (unrelated_dir / "04HP10N.s2p").write_text("", encoding="ascii")

            library = server._scan_tunable_component_families(str(root))

            self.assertEqual([item.part_number for item in library.inductors], ["04CS15N"])
            self.assertEqual(
                [item.part_number for item in library.capacitors],
                ["GJM1553C1H2R8WB01"],
            )

    def test_unified_tuning_passes_isolation_targets_and_persists_request(self):
        original = (
            server.state.loaded_snp,
            server.state.component_library,
            server.state.db_library,
            server.state.use_db,
            server.run_tuning_joint,
        )
        captured = {}

        def fake_joint(**kwargs):
            captured.update(kwargs)
            return {0: TuningResult(
                port_indices=[0, 1],
                mode="joint",
                isolation_targets=[{
                    "source_port": 0, "destination_port": 1,
                    "maximum_allowed_db": -20.0,
                    "worst_transmission_db": -21.0,
                    "average_transmission_db": -23.0,
                    "penalty_db": 0.0, "passed": True,
                }],
            )}

        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=2)
            server.state.component_library = object()
            server.state.db_library = None
            server.state.use_db = False
            server.run_tuning_joint = fake_joint
            request = TuningOptimizeRequest(
                ports=[
                    {"port_index": 0, "enabled": True, "bands_mhz": [[2400, 2500]], "max_components": 2},
                    {"port_index": 1, "enabled": True, "bands_mhz": [[2400, 2500]], "max_components": 2},
                ],
                isolation_targets=[{
                    "source_port": 0, "destination_port": 1,
                    "band_mhz": [2400, 2500], "maximum_db": -20,
                }],
            )
            result = asyncio.run(server.tuning_optimize(request))
            self.assertEqual(captured["isolation_targets"][0]["destination_port"], 1)
            self.assertEqual(server.get_session().last_tuning_request["isolation_targets"][0]["maximum_db"], -20.0)
            self.assertTrue(result["best_solution"]["isolation_constraints_passed"])
        finally:
            (
                server.state.loaded_snp,
                server.state.component_library,
                server.state.db_library,
                server.state.use_db,
                server.run_tuning_joint,
            ) = original
            server.reset_session()

    def test_isolation_target_must_reference_distinct_enabled_ports(self):
        original = (server.state.loaded_snp, server.state.component_library, server.state.use_db)
        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=2)
            server.state.component_library = object()
            server.state.use_db = False
            request = TuningOptimizeRequest(
                ports=[
                    {"port_index": 0, "enabled": True},
                    {"port_index": 1, "enabled": True},
                ],
                isolation_targets=[{
                    "source_port": 0, "destination_port": 0,
                    "band_mhz": [2400, 2500],
                }],
            )
            with self.assertRaisesRegex(HTTPException, "must be different"):
                asyncio.run(server.tuning_optimize(request))
        finally:
            server.state.loaded_snp, server.state.component_library, server.state.use_db = original
            server.reset_session()

    def test_unified_tuning_rejects_invalid_topology_code_with_http_400(self):
        original = server.state.loaded_snp
        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=1)
            request = TuningOptimizeRequest(
                mode="single",
                ports=[{
                    "port_index": 0, "enabled": True, "max_components": 0,
                    "allowed_topology_codes": ["BAD"],
                }],
            )
            with self.assertRaisesRegex(HTTPException, "invalid topology code") as context:
                asyncio.run(server._tuning_optimize_impl(request))
            self.assertEqual(context.exception.status_code, 400)
        finally:
            server.state.loaded_snp = original
            server.reset_session()

    def test_unified_zero_component_port_passes_canonical_bare_topology(self):
        original = (server.state.loaded_snp, server.run_tuning_single)
        captured = {}

        def fake_single(**kwargs):
            captured.update(kwargs)
            return {0: TuningResult(port_indices=[0], mode="single")}

        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=1)
            server.run_tuning_single = fake_single
            asyncio.run(server._tuning_optimize_impl(TuningOptimizeRequest(
                mode="single",
                ports=[{"port_index": 0, "enabled": True, "max_components": 0}],
            )))
            self.assertEqual(captured["allowed_topology_codes"], ["0"])
        finally:
            server.state.loaded_snp, server.run_tuning_single = original
            server.reset_session()

    def test_unified_tuning_routes_mdif_auto_synthesis(self):
        original = (
            server.state.loaded_snp,
            server.state.component_library,
            server.state.db_library,
            server.state.use_db,
            server.state.optenni_component_dir,
            server.run_tuning_tunable_mdif_auto,
        )
        captured = {}

        def fake_auto(**kwargs):
            captured.update(kwargs)
            return {0: TuningResult(
                port_indices=[0], mode="tunable",
                search_diagnostics={"mode": "tunable_mdif_auto_synthesis"},
            )}

        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=1)
            server.state.component_library = object()
            server.state.db_library = None
            server.state.use_db = False
            server.state.optenni_component_dir = None
            server.run_tuning_tunable_mdif_auto = fake_auto
            request = TuningOptimizeRequest(
                mode="tunable",
                ports=[{"port_index": 0, "enabled": True}],
                tuner_mdif_path="variable.mdif",
                frequency_configurations=[{
                    "name": "Set 1", "bands_mhz": [[700, 900]], "weight": 1,
                }],
                tunable_auto_synthesize=True,
            )

            result = asyncio.run(server.tuning_optimize(request))

            self.assertEqual(captured["tuner_mdif_path"], "variable.mdif")
            self.assertNotIn("fixed_components", captured)
            self.assertEqual(
                result["best_solution"]["search_diagnostics"]["mode"],
                "tunable_mdif_auto_synthesis",
            )
        finally:
            (
                server.state.loaded_snp,
                server.state.component_library,
                server.state.db_library,
                server.state.use_db,
                server.state.optenni_component_dir,
                server.run_tuning_tunable_mdif_auto,
            ) = original
            server.reset_session()

    def test_unified_tuning_routes_switch_mdif_synthesis_and_state_constraints(self):
        original = (
            server.state.loaded_snp,
            server.state.component_library,
            server.state.db_library,
            server.state.use_db,
            server.run_tuning_switch_mdif_auto,
        )
        captured = {}

        def fake_switch(**kwargs):
            captured.update(kwargs)
            return {
                index: TuningResult(
                    port_indices=[0], mode="switch", solution_index=index,
                    num_solutions_found=3, system_score=float(3 - index),
                    tunable_states={"Set 1": f"state-{index}"},
                    search_diagnostics={
                        "mode": "switch_mdif_auto_synthesis",
                        "input_component_count": 2 - index,
                    },
                )
                for index in range(3)
            }

        try:
            server.state.loaded_snp = SimpleNamespace(num_ports=1)
            server.state.component_library = object()
            server.state.db_library = None
            server.state.use_db = False
            server.run_tuning_switch_mdif_auto = fake_switch
            request = TuningOptimizeRequest(
                mode="switch",
                ports=[{"port_index": 0, "enabled": True}],
                tuner_mdif_path="sp3t.mdif",
                frequency_configurations=[{
                    "name": "Set 1", "bands_mhz": [[700, 900]], "weight": 1,
                }],
                switch_state_options={"Set 1": ["100"]},
            )

            result = asyncio.run(server.tuning_optimize(request))

            self.assertEqual(captured["switch_mdif_path"], "sp3t.mdif")
            self.assertEqual(captured["state_options_by_configuration"], {"Set 1": ["100"]})
            self.assertEqual(result["mode"], "switch")
            self.assertEqual(result["solutions_count"], 3)
            self.assertEqual(
                result["best_solution"]["search_diagnostics"]["mode"],
                "switch_mdif_auto_synthesis",
            )
            selected = asyncio.run(server.tuning_select_solution(2))
            self.assertEqual(selected["selected_index"], 2)
            self.assertEqual(selected["solution"]["tunable_states"], {"Set 1": "state-2"})
            status = asyncio.run(server.tuning_status())
            self.assertEqual(status["num_solutions"], 3)
            self.assertEqual(status["selected_index"], 2)
        finally:
            (
                server.state.loaded_snp,
                server.state.component_library,
                server.state.db_library,
                server.state.use_db,
                server.run_tuning_switch_mdif_auto,
            ) = original
            server.reset_session()

    def test_background_tuning_job_reports_progress_and_cancels(self):
        original = server._tuning_optimize_impl

        async def fake_impl(request, progress_callback=None, cancel_check=None):
            progress_callback({
                "stage": "ideal_search", "current": 100, "total": 0,
                "message": "working",
            })
            for _ in range(200):
                if cancel_check():
                    raise server.OptimizationCancelled("cancelled")
                await asyncio.sleep(0.005)
            return {"status": "ok"}

        try:
            server._tuning_optimize_impl = fake_impl
            with server._tuning_jobs_lock:
                server._tuning_jobs.clear()
            started = asyncio.run(server.start_tuning_job(TuningOptimizeRequest()))
            job_id = started["job_id"]
            deadline = time.time() + 2
            status = started
            while time.time() < deadline:
                status = asyncio.run(server.tuning_job_status(job_id))
                if status["progress"]["stage"] == "ideal_search":
                    break
                time.sleep(0.01)
            self.assertEqual(status["progress"]["current"], 100)

            asyncio.run(server.cancel_tuning_job(job_id))
            while time.time() < deadline:
                status = asyncio.run(server.tuning_job_status(job_id))
                if status["status"] == "cancelled":
                    break
                time.sleep(0.01)
            self.assertEqual(status["status"], "cancelled")
            self.assertIsNone(status["result"])
        finally:
            server._tuning_optimize_impl = original
            with server._tuning_jobs_lock:
                for job in server._tuning_jobs.values():
                    job["cancel_event"].set()
                server._tuning_jobs.clear()

    def test_completed_background_job_exposes_terminal_progress(self):
        original = server._tuning_optimize_impl

        async def fake_impl(request, progress_callback=None, cancel_check=None):
            progress_callback({
                "stage": "physical_refine", "current": 3, "total": 8,
                "message": "refining",
            })
            return {"status": "ok"}

        try:
            server._tuning_optimize_impl = fake_impl
            with server._tuning_jobs_lock:
                server._tuning_jobs.clear()
            started = asyncio.run(server.start_tuning_job(TuningOptimizeRequest()))
            deadline = time.time() + 2
            status = started
            while time.time() < deadline:
                status = asyncio.run(server.tuning_job_status(started["job_id"]))
                if status["status"] == "completed":
                    break
                time.sleep(0.01)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["progress"]["stage"], "complete")
            self.assertEqual(status["progress"]["current"], 8)
            self.assertEqual(status["progress"]["total"], 8)
            self.assertEqual(
                status["progress"]["message"], "Optimization complete"
            )
            self.assertEqual(status["progress"]["budget_seconds"], 120.0)
            self.assertEqual(
                status["progress"]["search_plan"]["effective_quality"],
                "thorough",
            )
            self.assertGreaterEqual(status["progress"]["elapsed_seconds"], 0.0)
        finally:
            server._tuning_optimize_impl = original
            with server._tuning_jobs_lock:
                server._tuning_jobs.clear()

    def test_cancelled_background_job_preserves_completed_partial_results(self):
        original = server._tuning_optimize_impl
        partial = TuningResult(port_indices=[0], mode="single", system_score=-3.0)

        async def fake_impl(request, progress_callback=None, cancel_check=None):
            progress_callback({
                "stage": "measured_expansion", "current": 2, "total": 10,
                "message": "Port 1: Measured S2P candidate expansion",
            })
            while not cancel_check():
                await asyncio.sleep(0.005)
            return {
                "status": "ok", "mode": "single", "solutions_count": 1,
                "solutions": [partial.to_dict()], "best_solution": partial.to_dict(),
            }

        try:
            server._tuning_optimize_impl = fake_impl
            with server._tuning_jobs_lock:
                server._tuning_jobs.clear()
            started = asyncio.run(server.start_tuning_job(TuningOptimizeRequest()))
            deadline = time.time() + 2
            status = started
            while time.time() < deadline:
                status = asyncio.run(server.tuning_job_status(started["job_id"]))
                if status["progress"]["stage"] == "measured_expansion":
                    break
                time.sleep(0.01)
            asyncio.run(server.cancel_tuning_job(started["job_id"]))
            while time.time() < deadline:
                status = asyncio.run(server.tuning_job_status(started["job_id"]))
                if status["status"] == "cancelled":
                    break
                time.sleep(0.01)

            self.assertEqual(status["status"], "cancelled")
            self.assertEqual(status["result"]["solutions_count"], 1)
            self.assertEqual(status["progress"]["stage"], "cancelled")
            self.assertIn("partial candidates", status["progress"]["message"])
        finally:
            server._tuning_optimize_impl = original
            with server._tuning_jobs_lock:
                for job in server._tuning_jobs.values():
                    job["cancel_event"].set()
                server._tuning_jobs.clear()

    def test_continue_background_job_is_non_blocking_and_uses_shared_progress_contract(self):
        original = server._tuning_continue_impl
        candidate = TuningResult(port_indices=[0], mode="single", system_score=-1.0)

        async def fake_continue(request, progress_callback=None, cancel_check=None):
            progress_callback({
                "stage": "measured_expansion", "current": 1, "total": 3,
                "message": "Port 1: Measured S2P candidate expansion",
            })
            await asyncio.sleep(0.15)
            return {
                "status": "ok", "mode": "single", "solutions_count": 1,
                "solutions": [candidate.to_dict()], "best_solution": candidate.to_dict(),
            }

        try:
            server._tuning_continue_impl = fake_continue
            server.reset_session()
            session = server.get_session()
            session.last_tuning_request = {"mode": "single", "ports": [{"port_index": 0}]}
            session.candidate_solutions = [candidate]
            session.restoration_mode = "live"
            with server._tuning_jobs_lock:
                server._tuning_jobs.clear()

            started_at = time.perf_counter()
            started = asyncio.run(server.start_tuning_continue_job(
                TuningContinueRequest(additional_seconds=5.0)
            ))
            self.assertLess(time.perf_counter() - started_at, 0.08)
            deadline = time.time() + 2
            status = started
            while time.time() < deadline:
                status = asyncio.run(server.tuning_job_status(started["job_id"]))
                if status["status"] == "completed":
                    break
                time.sleep(0.01)

            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["result"]["solutions_count"], 1)
            self.assertEqual(status["progress"]["stage"], "complete")
        finally:
            server._tuning_continue_impl = original
            with server._tuning_jobs_lock:
                for job in server._tuning_jobs.values():
                    job["cancel_event"].set()
                server._tuning_jobs.clear()
            server.reset_session()


if __name__ == "__main__":
    unittest.main()
