"""Project persistence, integrity, and safe restore regression tests."""

import asyncio
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import pdfplumber


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

from fastapi import HTTPException

import api.server as server
from api.models import (
    ProjectImportRequest, ProjectLoadRequest, ProjectRelinkRequest, ProjectSaveRequest,
)
from engine.touchstone import parse_touchstone
from engine.component_lib import ComponentInfo, ComponentLibrary
from engine.tuning_service import PerPortTuningMetrics, TuningResult
from project_store import (
    ProjectStore,
    ProjectValidationError,
    document_digest,
    sign_document,
    validate_document,
)
from pdf_reporting import render_project_pdf
from reporting import (
    _generic_synthesis_loss_status,
    _priority_weight_status,
    render_project_report,
)


class ProjectStoreTests(unittest.TestCase):
    def _payload(self):
        return {
            "name": "WiFi antenna",
            "input_snapshot": {"filename": "antenna.s1p", "sha256": "abc"},
            "configuration": {"tuning_request": {"objective": "balanced"}},
            "results": {"selected_index": 0, "candidates": []},
            "software": {"application": "RF Matching", "api_version": "2"},
        }

    def test_report_formats_generic_synthesis_loss_as_topology_prior(self):
        solution = {"search_diagnostics": {"generic_synthesis_loss": {
            "inductor_q": 50.0,
            "inductor_q_reference_hz": 1e9,
            "inductor_esr_ohm": 0.0,
            "capacitor_esr_ohm": 0.3,
        }}}
        self.assertEqual(
            _generic_synthesis_loss_status(solution),
            "L Q=50 @ 1 GHz; L ESR=0 Ω; C ESR=0.3 Ω; continuous topology prior only",
        )

    def test_round_trip_update_and_listing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            first = store.save(**self._payload())
            loaded = store.load(first["project_id"])
            self.assertEqual(loaded["schema_version"], 2)
            self.assertEqual(loaded["project_format"], "rfmatch.project")
            self.assertEqual(loaded["migration_history"], [])
            self.assertEqual(loaded["input"]["filename"], "antenna.s1p")
            listed = store.list()[0]
            self.assertEqual(listed["status"], "valid")
            self.assertEqual(listed["schema_version"], 2)
            self.assertIsNone(listed["migrated_from_version"])
            self.assertEqual(listed["manual_variants_count"], 0)

            updated = store.save(**self._payload(), project_id=first["project_id"])
            self.assertEqual(updated["created_at"], first["created_at"])
            self.assertEqual(updated["project_id"], first["project_id"])

    def test_import_preserves_verified_document_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as target_directory:
            source = ProjectStore(source_directory)
            target = ProjectStore(target_directory)
            exported = source.save(**self._payload())

            imported = target.import_document(exported)
            self.assertEqual(imported["status"], "imported")
            self.assertEqual(imported["document"], exported)
            duplicate = target.import_document(exported)
            self.assertEqual(duplicate["status"], "unchanged")
            self.assertEqual(duplicate["document"]["integrity"], exported["integrity"])
            self.assertEqual(len(list(Path(target_directory).glob("*.rfmatch.json"))), 1)

    def test_signed_but_invalid_known_manual_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            document = store.save(**self._payload())
            document.pop("integrity", None)
            document["extensions"]["manual_workspace"] = {
                "schema_version": 1,
                "active_input_port": 0,
                "target_frequency_hz": 2.45e9,
                "working_networks": {"terminated-port": []},
                "variants": [],
                "selected_variant_id": None,
            }
            with self.assertRaisesRegex(ProjectValidationError, "working network port"):
                store.import_document(sign_document(document))

    def test_import_conflict_creates_traced_copy_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            existing = store.save(**self._payload(), project_id="shared-project")
            incoming = dict(existing)
            incoming["name"] = "Different remote project"
            incoming = sign_document(incoming)

            result = store.import_document(incoming)
            copied = result["document"]
            self.assertEqual(result["status"], "copied")
            self.assertNotEqual(copied["project_id"], existing["project_id"])
            self.assertEqual(store.load(existing["project_id"])["name"], existing["name"])
            provenance = copied["extensions"]["import"]
            self.assertEqual(provenance["source_project_id"], "shared-project")
            self.assertEqual(
                provenance["source_integrity_sha256"], incoming["integrity"]["digest"]
            )
            self.assertEqual(validate_document(copied), copied)
            listed_copy = next(
                item for item in store.list() if item["project_id"] == copied["project_id"]
            )
            self.assertEqual(listed_copy["imported_from_project_id"], "shared-project")

    def test_import_rejects_tampering_before_writing(self):
        with tempfile.TemporaryDirectory() as source_directory, tempfile.TemporaryDirectory() as target_directory:
            exported = ProjectStore(source_directory).save(**self._payload())
            exported["name"] = "Tampered after export"
            target = ProjectStore(target_directory)
            with self.assertRaisesRegex(ProjectValidationError, "integrity check failed"):
                target.import_document(exported)
            self.assertEqual(target.list(), [])

    def test_import_migrates_verified_v1_document(self):
        legacy = sign_document({
            "schema_version": 1,
            "project_id": "legacy-import",
            "name": "Legacy import",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "software": {}, "input": {}, "configuration": {}, "results": {},
        })
        with tempfile.TemporaryDirectory() as directory:
            result = ProjectStore(directory).import_document(legacy)
            self.assertEqual(result["document"]["schema_version"], 2)
            self.assertEqual(result["document"]["migration_history"][0]["from_version"], 1)

    def test_replace_document_requires_matching_existing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            saved = store.save(**self._payload(), project_id="project-a")
            with self.assertRaisesRegex(ProjectValidationError, "does not match"):
                store.replace_document(saved, expected_project_id="project-b")

    def test_v1_snapshot_is_integrity_checked_then_migrated_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            legacy = sign_document({
                "schema_version": 1,
                "project_id": "legacy-project",
                "name": "Legacy",
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-01T00:00:00+00:00",
                "software": self._payload()["software"],
                "input": self._payload()["input_snapshot"],
                "configuration": self._payload()["configuration"],
                "results": self._payload()["results"],
            })
            path = Path(directory) / "legacy-project.rfmatch.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            migrated = store.load("legacy-project")
            self.assertEqual(migrated["schema_version"], 2)
            self.assertEqual(migrated["project_format"], "rfmatch.project")
            self.assertEqual(migrated["migration_history"], [{
                "from_version": 1,
                "to_version": 2,
                "source_sha256": legacy["integrity"]["digest"],
            }])
            self.assertEqual(migrated["integrity"]["digest"], document_digest(migrated))
            self.assertEqual(validate_document(migrated), migrated)
            self.assertEqual(store.load("legacy-project"), migrated)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 1)
            listed = store.list()[0]
            self.assertEqual(listed["schema_version"], 2)
            self.assertEqual(listed["migrated_from_version"], 1)

            upgraded = store.save(**self._payload(), project_id="legacy-project")
            self.assertEqual(upgraded["schema_version"], 2)
            self.assertEqual(upgraded["created_at"], legacy["created_at"])
            self.assertEqual(upgraded["migration_history"], migrated["migration_history"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 2)

    def test_v2_update_preserves_extension_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            saved = store.save(**self._payload())
            path = Path(directory) / f"{saved['project_id']}.rfmatch.json"
            saved["extensions"] = {"vendor.example": {"fixture_id": "A-17"}}
            path.write_text(json.dumps(sign_document(saved)), encoding="utf-8")

            updated = store.save(**self._payload(), project_id=saved["project_id"])
            self.assertEqual(
                updated["extensions"],
                {"vendor.example": {"fixture_id": "A-17"}},
            )

    def test_tampered_v1_snapshot_is_rejected_before_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            legacy = sign_document({
                "schema_version": 1,
                "project_id": "legacy-project",
                "name": "Legacy",
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-01T00:00:00+00:00",
                "software": {}, "input": {}, "configuration": {}, "results": {},
            })
            legacy["name"] = "Changed after signing"
            path = Path(directory) / "legacy-project.rfmatch.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ProjectValidationError, "integrity check failed"):
                store.load("legacy-project")

    def test_future_schema_is_rejected_without_guessing(self):
        future = sign_document({
            "schema_version": 99,
            "project_id": "future-project",
            "name": "Future",
            "created_at": "2030-01-01T00:00:00+00:00",
            "updated_at": "2030-01-01T00:00:00+00:00",
            "software": {}, "input": {}, "configuration": {}, "results": {},
        })
        with self.assertRaisesRegex(ProjectValidationError, "supported range is 1–2"):
            validate_document(future)

    def test_signed_v2_with_forged_migration_lineage_is_rejected(self):
        forged = sign_document({
            "schema_version": 2,
            "project_format": "rfmatch.project",
            "project_id": "forged-project",
            "name": "Forged",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "migration_history": [{
                "from_version": 2,
                "to_version": 1,
                "source_sha256": "not-a-digest",
            }],
            "extensions": {},
            "software": {}, "input": {}, "configuration": {}, "results": {},
        })
        with self.assertRaisesRegex(ProjectValidationError, "migration_history entry is invalid"):
            validate_document(forged)

    def test_tampering_is_detected_and_visible_in_listing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            saved = store.save(**self._payload())
            path = Path(directory) / f"{saved['project_id']}.rfmatch.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["name"] = "tampered"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ProjectValidationError, "integrity check failed"):
                store.load(saved["project_id"])
            self.assertEqual(store.list()[0]["status"], "invalid")

    def test_project_id_cannot_escape_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            with self.assertRaisesRegex(ProjectValidationError, "project_id is invalid"):
                store.load("../outside")

    def test_report_escapes_project_content(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            payload = self._payload()
            payload["name"] = "<script>alert(1)</script>"
            document = store.save(**payload)
            report = render_project_report(document)
            self.assertNotIn("<script>alert(1)</script>", report)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)

    def test_report_formats_effective_port_and_band_priorities(self):
        solution = {
            "port_indices": [0, 1],
            "search_diagnostics": {
                "priority_weights_by_port": {
                    "0": {"effective_band_weights": [2.0, 0.5]},
                    "1": {"effective_band_weights": [3.0]},
                },
            },
        }
        self.assertEqual(_priority_weight_status(solution), "P1: 2.0 / 0.5; P2: 3.0")

    def test_native_pdf_embeds_cjk_project_name(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            payload = self._payload()
            payload["name"] = "天线匹配项目"
            document = store.save(**payload)
            with pdfplumber.open(BytesIO(render_project_pdf(document))) as rendered:
                text = "\n".join(page.extract_text() or "" for page in rendered.pages)
            self.assertIn("天线匹配项目", text)


class ProjectApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.original_store = server.project_store
        self.original_snp_dir = server.state.snp_dir
        self.original_component_library = server.state.component_library
        self.original_full_component_library = server.state.full_component_library
        self.original_db_library = server.state.db_library
        self.original_use_db = server.state.use_db
        server.project_store = ProjectStore(Path(self.directory.name) / "projects")
        server.state.snp_dir = self.directory.name
        self.input_path = Path(self.directory.name) / "antenna.s1p"
        self.input_path.write_text(
            "# GHZ S RI R 50\n1.0 0.1 0.2\n1.1 0.2 0.1\n",
            encoding="utf-8",
        )
        server.state.loaded_snp = parse_touchstone(
            self.input_path.read_text(encoding="utf-8"), filename="antenna.s1p"
        )
        server.state.loaded_snp_filename = "antenna.s1p"
        server.reset_session()
        session = server.get_session()
        session.dut = server.state.loaded_snp
        session.dut_filename = "antenna.s1p"
        session.last_tuning_request = {
            "objective": "balanced", "ports": [],
            "component_series": ["L::Coilcraft Inductors 0402hp", "C::Murata Capacitors gqm18"],
            "component_filter": {
                "manufacturers": ["Murata", "Coilcraft"],
                "package_codes": ["0402"],
                "voltage_codes": ["1H"],
                "dielectrics": ["C0G"],
                "maximum_tolerance_pct": 5.0,
                "unknown_metadata_policy": "include",
            },
        }
        session.candidate_solutions = [
            TuningResult(
                port_indices=[0],
                per_port={0: PerPortTuningMetrics(
                    port_index=0,
                    total_efficiency=0.8,
                    components=[{
                        "connection_type": "series", "type": "inductor",
                        "part_number": "L_TEST", "value": "5.6nH",
                        "manufacturer": "Murata", "series": "LQP family",
                        "package_code": "0402", "tolerance_pct": 5.0,
                        "voltage_code": "1H", "dielectric": "C0G",
                        "tempco_ppm_per_c": 125.0,
                        "systematic_bias_pct": -0.4,
                        "environment_metadata": {
                            "evidence_level": "laboratory_measurement",
                            "source_document": "LAB-RF-2026-014",
                        },
                        "metadata_provenance": {
                            "manufacturer": "database", "package_code": "database",
                            "tolerance_pct": "database",
                            "voltage_code": "part_number_inferred",
                            "dielectric": "part_number_inferred",
                        },
                    }],
                    band_freqs_hz=[1e9, 1.1e9, 2e9, 2.1e9],
                    band_s11_db=[10.0, 12.0, 8.0, 15.0],
                    band_total_eff=[0.7, 0.8, 0.65, 0.82],
                )},
                avg_total_efficiency=0.8,
                min_total_efficiency=0.8,
                isolation_targets=[{
                    "source_port": 0, "destination_port": 1,
                    "maximum_allowed_db": -20.0,
                    "worst_transmission_db": -22.0,
                    "average_transmission_db": -24.0,
                    "penalty_db": 0.0, "passed": True,
                }],
                isolation_constraints_passed=True,
                search_diagnostics={
                    "measured_physical_search": True,
                    "component_model_backends": ["adapter"],
                    "physical_evaluations": 151,
                    "component_models_loaded": 65,
                    "component_library_filter": {
                        "mode": "selected_series",
                        "selected_series": [
                            "L::Coilcraft Inductors 0402hp",
                            "C::Murata Capacitors gqm18",
                        ],
                        "inductors": 49,
                        "capacitors": 322,
                        "parameter_filter": {
                            "manufacturers": ["Murata", "Coilcraft"],
                            "package_codes": ["0402"],
                            "voltage_codes": ["1H"],
                            "dielectrics": ["C0G"],
                            "maximum_tolerance_pct": 5.0,
                            "unknown_metadata_policy": "include",
                        },
                        "catalog_fingerprint": "A" * 64,
                    },
                    "active_frequency_points": 6,
                    "loss_aware_ideal_seed": {
                        "topology_signature": [["shunt", "C"], ["series", "L"]],
                        "elements": [
                            {"connection": "shunt", "kind": "C", "value_si": 0.48e-12},
                            {"connection": "series", "kind": "L", "value_si": 5.9e-9},
                        ],
                        "score_db": -0.95,
                        "evaluations": 4444,
                    },
                    "yield_analysis": {
                        "yield_fraction": 0.9,
                        "yield_confidence_interval": [0.8, 0.96],
                        "score_percentiles_db": {"5": 1.25},
                        "distribution": "uniform",
                        "variation_model": {
                            "batch_correlation": 0.5,
                            "reference_temperature_c": 25.0,
                            "temperature_min_c": -40.0,
                            "temperature_max_c": 85.0,
                            "inductor_tempco_ppm_per_c": 100.0,
                            "capacitor_tempco_ppm_per_c": -30.0,
                            "inductor_bias_pct": 1.25,
                            "capacitor_bias_pct": -0.5,
                        },
                    },
                },
            )
        ]

    def tearDown(self):
        server.project_store = self.original_store
        server.state.snp_dir = self.original_snp_dir
        server.state.component_library = self.original_component_library
        server.state.full_component_library = self.original_full_component_library
        server.state.db_library = self.original_db_library
        server.state.use_db = self.original_use_db
        server.state.loaded_snp = None
        server.state.loaded_snp_filename = ""
        server.reset_session()
        self.directory.cleanup()

    def test_api_save_and_snapshot_restore_round_trip(self):
        manual_workspace = {
            "schema_version": 1,
            "active_input_port": 0,
            "target_frequency_hz": 1.0e9,
            "working_networks": {"0": [{
                "comp_type": "inductor", "connection_type": "series",
                "value": 5.6, "use_ideal": True, "port": 0,
            }, {
                "comp_type": "capacitor", "connection_type": "shunt",
                "value": 2.2, "use_ideal": True, "port": 0,
            }]},
            "variants": [{
                "variant_id": "manual-baseline",
                "name": "P1 low VSWR",
                "input_port": 0,
                "target_frequency_hz": 1.0e9,
                "components": [{
                    "comp_type": "inductor", "connection_type": "series",
                    "value": 5.6, "use_ideal": True, "port": 0,
                }],
                "port_states": [],
                "metrics": {
                    "return_loss_db": 18.5,
                    "return_loss_improvement_db": 7.2,
                    "vswr": 1.27,
                    "input_impedance_real": 48.0,
                    "input_impedance_imag": -2.5,
                    "maximum_power_balance_error": 0.0,
                    "numeric_core": "rfmatch_core",
                },
                "created_at": "2026-07-18T00:00:00+00:00",
            }],
            "selected_variant_id": "manual-baseline",
            "overlay_variant_ids": ["manual-baseline"],
        }
        saved = asyncio.run(server.save_project(ProjectSaveRequest(
            name="Demo", manual_workspace=manual_workspace,
        )))
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(saved["manual_variants_count"], 1)
        report = asyncio.run(server.project_html_report(saved["project_id"]))
        self.assertEqual(report.media_type, "text/html; charset=utf-8")
        self.assertIn("attachment", report.headers["content-disposition"])
        report_text = report.body.decode("utf-8")
        self.assertIn("RF Matching traceability report", report_text)
        self.assertIn("Project schema", report_text)
        self.assertIn("v2", report_text)
        self.assertIn("Touchstone SHA-256", report_text)
        self.assertIn("Candidate comparison", report_text)
        self.assertIn("Frozen manual variants", report_text)
        self.assertIn("P1 low VSWR", report_text)
        self.assertIn("series L 5.6", report_text)
        self.assertIn("18.500 dB", report_text)
        self.assertIn("80.0%", report_text)
        self.assertIn("L_TEST", report_text)
        self.assertIn("90.0%", report_text)
        self.assertIn("80.0% – 96.0%", report_text)
        self.assertIn("1.250 dB", report_text)
        self.assertIn("batch correlation 50.0%", report_text)
        self.assertIn("L/C systematic bias 1.25/-0.5%", report_text)
        self.assertIn("temperature -40–85 °C", report_text)
        self.assertIn("shunt C → series L", report_text)
        self.assertIn("0.48 pF", report_text)
        self.assertIn("5.9 nH", report_text)
        self.assertIn("-0.950 dB", report_text)
        self.assertIn("4444", report_text)
        self.assertIn("full-band physical (adapter)", report_text)
        self.assertIn("151 / 65", report_text)
        self.assertIn("Coilcraft Inductors 0402hp", report_text)
        self.assertIn("49 L / 322 C", report_text)
        self.assertIn("packages=0402", report_text)
        self.assertIn("tolerance≤5.0%", report_text)
        self.assertIn("voltage code 1H", report_text)
        self.assertIn("C0G (inferred)", report_text)
        self.assertIn("tempco 125.0 ppm/°C [laboratory_measurement]", report_text)
        self.assertIn("systematic bias -0.4% [laboratory_measurement]", report_text)
        self.assertIn("A" * 64, report_text)
        self.assertEqual(report_text.count("<polyline"), 4)

        pdf = asyncio.run(server.project_pdf_report(saved["project_id"]))
        self.assertEqual(pdf.media_type, "application/pdf")
        self.assertIn("attachment", pdf.headers["content-disposition"])
        self.assertTrue(pdf.body.startswith(b"%PDF-"))
        with pdfplumber.open(BytesIO(pdf.body)) as rendered:
            self.assertGreaterEqual(len(rendered.pages), 3)
            pdf_text = "\n".join(page.extract_text() or "" for page in rendered.pages)
        self.assertIn("RF MATCHING", pdf_text)
        self.assertIn("Candidate comparison", pdf_text)
        self.assertIn("Frozen manual variants", pdf_text)
        self.assertIn("P1 low VSWR", pdf_text)
        self.assertIn("Selected solution", pdf_text)
        self.assertIn("batch correlation 50.0%", pdf_text)
        self.assertIn("L_TEST", pdf_text)
        self.assertIn("Optimization configuration", pdf_text)
        self.assertIn("Component catalog", pdf_text)
        self.assertIn("49 L / 322 C", pdf_text)
        self.assertIn("Page 1", pdf_text)
        inline_pdf = asyncio.run(server.project_pdf_report(saved["project_id"], download=False))
        self.assertIn("inline", inline_pdf.headers["content-disposition"])
        bom = asyncio.run(server.project_bom_csv(saved["project_id"]))
        self.assertEqual(bom.media_type, "text/csv; charset=utf-8")
        self.assertIn("attachment", bom.headers["content-disposition"])
        bom_text = bom.body.decode("utf-8-sig")
        self.assertIn("Part Number", bom_text)
        self.assertIn("L_TEST", bom_text)
        self.assertIn("P1:1", bom_text)
        self.assertIn("part_number_inferred", bom_text)
        snapshot = asyncio.run(server.project_snapshot_json(saved["project_id"]))
        self.assertEqual(snapshot.media_type, "application/json; charset=utf-8")
        self.assertIn("attachment", snapshot.headers["content-disposition"])
        exported = json.loads(snapshot.body.decode("utf-8"))
        self.assertEqual(exported["project_id"], saved["project_id"])
        self.assertEqual(exported["schema_version"], 2)
        self.assertEqual(exported["integrity"]["algorithm"], "sha256")
        self.assertEqual(
            exported["extensions"]["manual_workspace"]["variants"][0]["name"],
            "P1 low VSWR",
        )
        server.state.loaded_snp = None
        server.reset_session()

        restored = asyncio.run(
            server.load_project(ProjectLoadRequest(project_id=saved["project_id"]))
        )
        self.assertTrue(restored["input_verified"])
        self.assertEqual(restored["input_sha256"], saved["input_sha256"])
        self.assertEqual(restored["schema_version"], 2)
        self.assertIsNone(restored["migrated_from_version"])
        self.assertEqual(restored["restoration_mode"], "snapshot")
        self.assertEqual(restored["manual_workspace"]["selected_variant_id"], "manual-baseline")
        self.assertEqual(restored["manual_workspace"]["overlay_variant_ids"], ["manual-baseline"])
        self.assertEqual(restored["manual_workspace"]["working_networks"]["0"][0]["value"], 5.6)
        self.assertEqual(
            [component["comp_type"] for component in restored["manual_workspace"]["working_networks"]["0"]],
            ["inductor", "capacitor"],
        )
        self.assertEqual(restored["solutions_count"], 1)
        self.assertEqual(restored["tuning_request"]["component_series"], [
            "L::Coilcraft Inductors 0402hp", "C::Murata Capacitors gqm18",
        ])
        self.assertEqual(
            restored["tuning_request"]["component_filter"]["package_codes"], ["0402"]
        )
        self.assertEqual(
            restored["tuning_request"]["component_filter"]["dielectrics"], ["C0G"]
        )
        self.assertAlmostEqual(
            server.get_session().get_selected_result().avg_total_efficiency, 0.8
        )
        self.assertTrue(server.get_session().get_selected_result().isolation_constraints_passed)
        self.assertEqual(server.get_session().get_selected_result().isolation_targets[0]["worst_transmission_db"], -22.0)
        status = asyncio.run(server.tuning_status())
        self.assertFalse(status["can_recompute_exact_sweep"])

    def test_api_import_is_verified_and_duplicate_safe(self):
        saved = asyncio.run(server.save_project(ProjectSaveRequest(name="Import source")))
        document = server.project_store.load(saved["project_id"])
        duplicate = asyncio.run(server.import_project(ProjectImportRequest(document=document)))
        self.assertEqual(duplicate["status"], "unchanged")
        self.assertEqual(duplicate["project_id"], saved["project_id"])
        self.assertEqual(duplicate["integrity_sha256"], document["integrity"]["digest"])

        conflicting = dict(document)
        conflicting["name"] = "Remote variant"
        conflicting = sign_document(conflicting)
        copied = asyncio.run(server.import_project(ProjectImportRequest(document=conflicting)))
        self.assertEqual(copied["status"], "copied")
        self.assertNotEqual(copied["project_id"], saved["project_id"])
        self.assertEqual(copied["source_project_id"], saved["project_id"])
        self.assertEqual(len(server.project_store.list()), 2)

    def test_api_rejects_changed_input(self):
        saved = asyncio.run(server.save_project(ProjectSaveRequest(name="Demo")))
        self.input_path.write_text(
            "# GHZ S RI R 50\n1.0 0.9 0.0\n", encoding="utf-8"
        )
        with self.assertRaises(HTTPException) as context:
            asyncio.run(
                server.load_project(ProjectLoadRequest(project_id=saved["project_id"]))
            )
        self.assertEqual(context.exception.status_code, 409)

    def test_api_allows_read_only_snapshot_when_input_is_unavailable(self):
        saved = asyncio.run(server.save_project(ProjectSaveRequest(name="Portable review")))
        self.input_path.unlink()

        restored = asyncio.run(server.load_project(ProjectLoadRequest(
            project_id=saved["project_id"], verify_input=False,
        )))
        self.assertFalse(restored["input_verified"])
        self.assertFalse(restored["exact_recompute_available"])
        self.assertEqual(restored["restoration_mode"], "snapshot")
        self.assertEqual(restored["solutions_count"], 1)
        self.assertEqual(restored["num_ports"], 1)
        self.assertEqual(restored["freq_count"], 2)
        self.assertIsNone(server.state.loaded_snp)
        self.assertEqual(
            server.get_session().get_selected_result().per_port[0].components[0]["part_number"],
            "L_TEST",
        )

    def test_api_relinks_byte_identical_input_and_restores_recompute(self):
        saved = asyncio.run(server.save_project(ProjectSaveRequest(name="Relinkable")))
        relocated = Path(self.directory.name) / "relocated"
        relocated.mkdir()
        replacement = relocated / "renamed-antenna.s1p"
        self.input_path.replace(replacement)

        result = asyncio.run(server.relink_project_files(ProjectRelinkRequest(
            project_id=saved["project_id"], apply_matches=True,
        )))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["matched_count"], 1)
        self.assertTrue(result["changed"])
        self.assertEqual(result["targets"][0]["linked_filename"], os.path.join("relocated", "renamed-antenna.s1p"))

        document = server.project_store.load(saved["project_id"])
        link = document["extensions"]["file_relinks"]["input"]
        self.assertEqual(link["original_filename"], "antenna.s1p")
        self.assertEqual(link["expected_sha256"], document["input"]["sha256"])
        restored = asyncio.run(server.load_project(ProjectLoadRequest(
            project_id=saved["project_id"], verify_input=True,
        )))
        self.assertTrue(restored["exact_recompute_available"])
        self.assertEqual(restored["snapshot_input_filename"], "antenna.s1p")
        self.assertEqual(restored["input_filename"], os.path.join("relocated", "renamed-antenna.s1p"))

    def test_api_does_not_enable_exact_recompute_with_unverified_component_database(self):
        saved = asyncio.run(server.save_project(ProjectSaveRequest(name="Catalog gate")))
        document = server.project_store.load(saved["project_id"])
        document.pop("integrity", None)
        document["configuration"]["tuning_request"]["ports"] = [{
            "port_index": 0, "enabled": True, "max_components": 2,
            "bands_mhz": [[1000, 1100]],
        }]
        document["configuration"]["component_library"] = {
            "mode": "database",
            "filename": "missing-components.db",
            "sha256": "0" * 64,
        }
        server.project_store.replace_document(
            sign_document(document), expected_project_id=saved["project_id"]
        )

        restored = asyncio.run(server.load_project(ProjectLoadRequest(
            project_id=saved["project_id"], verify_input=True,
        )))
        self.assertTrue(restored["input_verified"])
        self.assertFalse(restored["exact_recompute_available"])
        self.assertTrue(restored["component_library_status"]["required"])
        self.assertEqual(
            restored["component_library_status"]["reason"], "database_hash_mismatch"
        )

    def test_s2p_catalog_manifest_is_portable_across_directory_roots(self):
        def measured_library(path):
            result = ComponentLibrary()
            result.add_component(ComponentInfo(
                part_number="04HP5N6",
                s2p_filename=str(path),
                zip_path="__DIR__",
                component_type="inductor",
                nominal_value=5.6,
                nominal_unit="nH",
                manufacturer="Coilcraft",
                series="Coilcraft Inductors 0402hp",
                size_code="0402",
            ))
            return result

        first_root = Path(self.directory.name) / "catalog-a"
        second_root = Path(self.directory.name) / "catalog-b"
        first_root.mkdir(); second_root.mkdir()
        source = first_root / "04HP5N6.s2p"
        replacement = second_root / "04HP5N6.s2p"
        component_bytes = b"# GHZ S RI R 50\n1 0 0 1 0 1 0 0 0\n"
        source.write_bytes(component_bytes)
        replacement.write_bytes(component_bytes)
        server.state.use_db = False
        server.state.db_library = None
        server.state.component_library = measured_library(source)
        server.state.full_component_library = server.state.component_library
        session = server.get_session()
        session.library = server.state.component_library
        session.last_tuning_request["ports"] = [{
            "port_index": 0, "enabled": True, "max_components": 2,
            "bands_mhz": [[1000, 1100]],
        }]

        saved = asyncio.run(server.save_project(ProjectSaveRequest(name="Portable catalog")))
        document = server.project_store.load(saved["project_id"])
        manifest = document["configuration"]["component_library"]["catalog_manifest"]
        self.assertTrue(manifest["content_verified"])

        server.state.component_library = measured_library(replacement)
        server.state.full_component_library = server.state.component_library
        restored = asyncio.run(server.load_project(ProjectLoadRequest(
            project_id=saved["project_id"], verify_input=True,
        )))
        self.assertTrue(restored["component_library_status"]["matches"])
        self.assertTrue(restored["exact_recompute_available"])

        replacement.write_bytes(component_bytes + b"! changed\n")
        changed = asyncio.run(server.load_project(ProjectLoadRequest(
            project_id=saved["project_id"], verify_input=True,
        )))
        self.assertFalse(changed["component_library_status"]["matches"])
        self.assertFalse(changed["exact_recompute_available"])

    def test_api_hashes_and_verifies_layout_s2p_dependencies(self):
        layout_path = Path(self.directory.name) / "launch.s2p"
        layout_path.write_text(
            "# GHZ S RI R 50\n"
            "1.0 0 0 0.8 0 0.8 0 0 0\n"
            "1.1 0 0 0.8 0 0.8 0 0 0\n",
            encoding="utf-8",
        )
        fixture_path = Path(self.directory.name) / "left-fixture.s2p"
        fixture_path.write_text(
            "# GHZ S RI R 50\n1.0 0 0 1 0 1 0 0 0\n1.1 0 0 1 0 1 0 0 0\n",
            encoding="utf-8",
        )
        server.get_session().last_tuning_request = {
            "mode": "transmission_line",
            "ports": [{"port_index": 0, "bands_mhz": [[1000, 1100]], "enabled": True}],
            "transmission_line": {
                "layout_blocks": [{
                    "filename": "launch.s2p",
                    "location": "connector_side",
                    "passivity_policy": "warn",
                    "reverse_ports": True,
                    "reference_impedance_mode": "system",
                    "left_fixture_filename": "left-fixture.s2p",
                    "left_fixture_reverse_ports": True,
                }],
            },
        }
        saved = asyncio.run(server.save_project(ProjectSaveRequest(name="Layout project")))
        document = server.project_store.load(saved["project_id"])
        dependencies = document["configuration"]["input_dependencies"]
        self.assertEqual(len(dependencies), 2)
        self.assertEqual(dependencies[0]["role"], "layout_s2p")
        self.assertEqual(dependencies[0]["filename"], "launch.s2p")
        self.assertTrue(dependencies[0]["reverse_ports"])
        self.assertEqual(dependencies[0]["reference_impedance_mode"], "system")
        self.assertEqual(len(dependencies[0]["sha256"]), 64)
        html_report = render_project_report(document)
        self.assertIn("Input dependencies", html_report)
        self.assertIn("launch.s2p", html_report)
        self.assertIn("left-fixture.s2p", html_report)
        self.assertIn(dependencies[0]["sha256"], html_report)
        self.assertIn("flip 1↔2; system Z0", html_report)
        with pdfplumber.open(BytesIO(render_project_pdf(document))) as rendered:
            pdf_text = "\n".join(page.extract_text() or "" for page in rendered.pages)
        self.assertIn("Input dependencies", pdf_text)
        self.assertIn("launch.s2p", pdf_text)
        self.assertIn("flip; system Z0", pdf_text)

        restored = asyncio.run(
            server.load_project(ProjectLoadRequest(project_id=saved["project_id"]))
        )
        self.assertTrue(restored["dependency_status"][0]["matches"])

        fixture_path.write_text(
            "# GHZ S RI R 50\n1.0 0 0 0.5 0 0.5 0 0 0\n",
            encoding="utf-8",
        )
        with self.assertRaises(HTTPException) as context:
            asyncio.run(
                server.load_project(ProjectLoadRequest(project_id=saved["project_id"]))
            )
        self.assertEqual(context.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
