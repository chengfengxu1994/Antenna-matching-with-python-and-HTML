"""
FastAPI backend server for RF Matching application.
Integrated with SQLite DB adapter for fast component queries.
"""

import os
import sys
import time
import re
import platform
import asyncio
import threading
import uuid
import hashlib
import zipfile
import json
from copy import deepcopy
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from project_paths import PROJECT_ROOT, PROJECTS_DIR, WEB_DIST_DIR, resolve_project_path
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "rfmatch-core" / "src"))

from engine.touchstone import parse_touchstone, touchstone_network_sha256
from engine.snp_monitor import TouchstoneDirectoryMonitor
from engine.cst_bridge import CSTBridge, CSTBridgeError
from engine.calibration_evidence import (
    CalibrationEvidenceError,
    multiport_calibration_reference,
    search_performance_reference,
    single_port_calibration_reference,
)
from engine.snp_provenance import SnpProvenanceStore
from engine.search_quality import build_search_plan
from engine.component_lib import (
    scan_murata_directory, scan_s2p_directory, filter_component_library,
    merge_component_libraries, component_series_id, component_series_name,
    component_metadata, filter_component_library_by_parameters,
    filter_component_library_by_series,
    component_library_content_fingerprint,
    COMPONENT_CATALOG_FINGERPRINT_SCHEMA,
)
from engine.murata_db_adapter import load_murata_db
from engine.component_environment import (
    apply_component_environment_catalog,
    load_component_environment_catalog,
)
from engine.topology import get_standard_topologies
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from engine.multiport_optimizer import JointMultiPortOptimizer, PortMatchConfig, evaluate_joint_solution
from engine.cost_function import (
    get_optimization_mode, OPTIMIZATION_MODES,
)
from engine.efficiency_data import load_efficiency_file, parse_efficiency_data
from engine.multi_scenario_optimizer import (
    MultiScenarioOptimizer, Scenario, find_component,
)
from engine.search_quality import build_multi_scenario_search_plan
from api.models import (
    DataDirConfig, EfficiencyClearRequest, EfficiencyInlineRequest,
    EfficiencyLoadRequest, JointOptimizeRequest, ManualRefineRequest, ManualTuneRequest,
    ManualYieldRequest,
    MultiPortOptimizeRequest, MultiScenarioManualRequest,
    MultiScenarioOptimizeRequest, OptimizeRequest, PortStateConfig,
    ComponentAlternativesRequest, ComponentLibraryPreviewRequest,
    ProjectImportRequest, ProjectLoadRequest, ProjectRelinkRequest, ProjectSaveRequest,
    TuningContinueRequest,
    ScenarioConfig, SinglePortConfig, TuneJointRequest,
    TunePowerBalanceRequest, TuneSingleRequest, TuningOptimizeRequest,
    TuningYieldRequest,
    SNPImportRequest,
)
from api.state import AppState
from project_store import (
    ProjectStore, ProjectValidationError, sha256_file, sign_document, utc_now,
)
from reporting import render_project_report
from pdf_reporting import render_project_pdf
from bom_export import render_project_bom_csv

from rfmatch_core import OptimizationCancelled, __version__ as core_version

app = FastAPI(title="RF Matching Engine", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── State ───

state = AppState()
project_store = ProjectStore(PROJECTS_DIR)
snp_monitor = TouchstoneDirectoryMonitor(parse_touchstone)
cst_bridge = CSTBridge(PROJECT_ROOT / "scripts" / "cst_bridge_worker.py")
snp_provenance_store = SnpProvenanceStore()

# ─── Helpers ───

def _get_library():
    """Return the active component library."""
    if state.use_db and state.db_library:
        return state.db_library
    return state.component_library

def _ensure_snp_loaded():
    if not state.loaded_snp:
        raise HTTPException(400, "No SNP loaded")

def _ensure_library():
    if not _get_library():
        raise HTTPException(400, "Component library not loaded")


def _touchstone_reference_metadata(data):
    values = [
        float(complex(value).real)
        for value in (
            data.port_impedances
            or [data.reference_resistance] * data.num_ports
        )
    ]
    uniform = all(abs(value - values[0]) <= 1e-12 for value in values)
    return {
        "parameter_format": data.data_format,
        "reference_impedance_ohm": values[0] if uniform else None,
        "reference_impedances_ohm": values,
    }


def _library_for_series(
    selected_series, component_filter=None, validate=True,
    required_component_types=("inductors", "capacitors"),
    require_any_component=False,
):
    """Resolve one request-scoped catalog without mutating the global default."""
    base = state.full_component_library or _get_library()
    def fingerprint(library) -> str:
        digest = hashlib.sha256()
        components = [
            *(getattr(library, "inductors", []) or []),
            *(getattr(library, "capacitors", []) or []),
        ]
        for component in sorted(components, key=lambda item: (
            getattr(item, "component_type", ""),
            getattr(item, "part_number", ""),
            getattr(item, "s2p_filename", ""),
        )):
            digest.update("|".join((
                str(getattr(component, "component_type", "")),
                str(getattr(component, "part_number", "")),
                str(getattr(component, "nominal_value", "")),
                str(getattr(component, "s2p_filename", "")),
                str(getattr(component, "zip_path", "")),
            )).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest().upper()
    filter_config = _model_dump(component_filter) if component_filter is not None else {
        "manufacturers": [], "package_codes": [], "voltage_codes": [],
        "dielectrics": [], "maximum_tolerance_pct": None,
        "unknown_metadata_policy": "include",
    }
    if selected_series is None:
        series_library = _get_library()
        mode = "default"
    else:
        series_library = filter_component_library_by_series(base, selected_series)
        mode = "selected_series"
    has_parameter_constraints = bool(
        filter_config.get("manufacturers")
        or filter_config.get("package_codes")
        or filter_config.get("voltage_codes")
        or filter_config.get("dielectrics")
        or filter_config.get("maximum_tolerance_pct") is not None
    )
    if has_parameter_constraints:
        library, parameter_stats = filter_component_library_by_parameters(
            series_library, filter_config
        )
    else:
        library = series_library
        total = len(getattr(library, "inductors", []) or []) + len(
            getattr(library, "capacitors", []) or []
        )
        parameter_stats = {
            "input_components": total, "matched_components": total,
            "excluded_by_value": 0, "excluded_unknown": 0,
            "included_with_unknown": 0, "metadata_sources": {},
        }
    counts = {
        "inductors": len(getattr(library, "inductors", []) or []),
        "capacitors": len(getattr(library, "capacitors", []) or []),
    }
    if validate and selected_series is not None and not selected_series:
        raise HTTPException(400, "Select at least one inductor series and one capacitor series")
    required_component_types = set(required_component_types or ())
    missing = [
        kind for kind, count in counts.items()
        if kind in required_component_types and count == 0
    ]
    if validate and missing and (selected_series is not None or has_parameter_constraints):
        raise HTTPException(
            400,
            "Component library filters produced no " + " or ".join(missing)
            + "; relax the series or parameter constraints",
        )
    if (
        validate
        and require_any_component
        and not any(counts.values())
        and (selected_series is not None or has_parameter_constraints)
    ):
        raise HTTPException(
            400,
            "Component library filters produced no measured inductors or capacitors; "
            "relax the series or parameter constraints",
        )
    return library, {
        "mode": mode,
        "selected_series": None if selected_series is None else sorted(set(selected_series)),
        "parameter_filter": filter_config,
        "filter_statistics": parameter_stats,
        "catalog_fingerprint": fingerprint(library),
        **counts,
    }


def _required_component_types_for_topologies(topology_names, max_components):
    if int(max_components) <= 0:
        return set()
    if not topology_names:
        return {"inductors", "capacitors"}
    allowed = {str(name) for name in topology_names}
    required = set()
    for topology in get_standard_topologies():
        if topology.name not in allowed:
            continue
        for element in topology.elements:
            required.add(
                "inductors" if element.component_type == "inductor" else "capacitors"
            )
    return required or {"inductors", "capacitors"}


def _scan_tunable_component_families(root_dir: str):
    """Load only the two measured families used by variable-C synthesis."""
    root = Path(root_dir)
    inductor_dirs = sorted((root / "Inductors").glob("*0402cs*"))
    capacitor_dirs = sorted((root / "Capacitors").glob("*gjm15*"))
    if inductor_dirs and capacitor_dirs:
        return merge_component_libraries(
            *(scan_s2p_directory(str(path)) for path in [*inductor_dirs, *capacitor_dirs])
        )
    return filter_component_library(
        scan_s2p_directory(root_dir),
        inductor_tokens=["COILCRAFT INDUCTORS 0402CS", "0402CS", "04CS"],
        capacitor_tokens=["MURATA CAPACITORS GJM15", "GJM15"],
    )


def _model_dump(value) -> dict:
    """Support both Pydantic v1 and v2 without leaking model objects."""
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value.dict()


def _component_library_snapshot() -> dict:
    library = _get_library()
    if library is None:
        return {"mode": "none"}
    try:
        session = get_session()
        active_library = session.library or library
        selected_series = (session.last_tuning_request or {}).get("component_series")
        component_filter = (session.last_tuning_request or {}).get("component_filter")
    except NameError:
        active_library = library
        selected_series = None
        component_filter = None
    active_counts = {
        "inductors": len(getattr(active_library, "inductors", []) or []),
        "capacitors": len(getattr(active_library, "capacitors", []) or []),
    }
    if state.use_db and state.db_library:
        db_path = Path(state.db_library.db_path).resolve()
        environment_catalog = state.component_environment_catalog
        return {
            "mode": "database",
            "filename": db_path.name,
            "sha256": sha256_file(db_path) if db_path.is_file() else None,
            "stats": getattr(state.db_library, "stats", {}),
            "selected_series": selected_series,
            "component_filter": component_filter,
            "active_counts": active_counts,
            "environment_metadata": environment_catalog.summary() if environment_catalog else None,
        }
    snapshot = {
        "mode": "s2p_directory",
        **active_counts,
        "selected_series": selected_series,
        "component_filter": component_filter,
    }
    manifest = component_library_content_fingerprint(active_library)
    snapshot["catalog_manifest"] = manifest
    snapshot["catalog_fingerprint"] = manifest["digest"]
    snapshot["environment_metadata"] = (
        state.component_environment_catalog.summary()
        if state.component_environment_catalog else None
    )
    return snapshot


def _component_library_verification(document):
    saved = document["configuration"].get("component_library") or {"mode": "none"}
    request = document["configuration"].get("tuning_request") or {}
    ports = request.get("ports") or []
    required = any(
        bool(port.get("enabled", True)) and int(port.get("max_components", 0) or 0) > 0
        for port in ports if isinstance(port, dict)
    )
    mode = saved.get("mode", "none")
    status = {
        "required": required,
        "mode": mode,
        "matches": not required,
        "reason": "not_required" if not required else "unverified",
    }
    if not required:
        return status
    if mode == "database":
        expected = saved.get("sha256")
        active_path = (
            Path(state.db_library.db_path).resolve()
            if state.use_db and state.db_library else None
        )
        actual = sha256_file(active_path) if active_path and active_path.is_file() else None
        saved_environment = saved.get("environment_metadata") or {}
        active_environment = state.component_environment_catalog
        expected_environment = saved_environment.get("sha256")
        actual_environment = active_environment.sha256 if active_environment else None
        environment_matches = (
            actual_environment == expected_environment
            if expected_environment else active_environment is None
        )
        database_matches = bool(expected and actual == expected)
        status.update({
            "expected_sha256": expected,
            "actual_sha256": actual,
            "expected_environment_sha256": expected_environment,
            "actual_environment_sha256": actual_environment,
            "matches": database_matches and environment_matches,
            "reason": (
                "verified" if database_matches and environment_matches
                else "database_hash_mismatch" if not database_matches
                else "environment_metadata_hash_mismatch"
            ),
        })
        return status
    if mode == "s2p_directory":
        expected_manifest = saved.get("catalog_manifest") or {}
        expected = expected_manifest.get("digest")
        try:
            active_library, _ = _library_for_series(
                request.get("component_series"),
                request.get("component_filter"),
                validate=False,
            )
            actual_manifest = component_library_content_fingerprint(active_library)
            actual = actual_manifest.get("digest")
        except (OSError, TypeError, ValueError, HTTPException):
            actual_manifest = None
            actual = None
        manifest_compatible = bool(
            expected_manifest.get("schema") == COMPONENT_CATALOG_FINGERPRINT_SCHEMA
            and expected_manifest.get("algorithm") == "sha256"
            and expected_manifest.get("content_verified") is True
            and actual_manifest
            and actual_manifest.get("content_verified") is True
        )
        status.update({
            "manifest_schema": expected_manifest.get("schema"),
            "expected_fingerprint": expected,
            "actual_fingerprint": actual,
            "matches": bool(manifest_compatible and expected and actual == expected),
            "reason": "verified" if manifest_compatible and expected and actual == expected else (
                "snapshot_has_no_portable_manifest" if not expected else (
                    "catalog_sources_unverified" if not manifest_compatible
                    else "catalog_fingerprint_mismatch"
                )
            ),
        })
        return status
    status["reason"] = "component_library_unavailable"
    return status


def _safe_data_path(relative_path: str) -> str:
    """Resolve a user-selected file below the configured SNP data directory."""
    if not state.snp_dir:
        raise HTTPException(400, "SNP directory not configured")
    root = os.path.realpath(os.path.abspath(state.snp_dir))
    candidate = os.path.realpath(os.path.abspath(os.path.join(root, relative_path)))
    try:
        inside = os.path.commonpath([root, candidate]) == root
    except ValueError:
        inside = False
    if not inside:
        raise HTTPException(400, "File must be inside the configured SNP directory")
    if not os.path.isfile(candidate):
        raise HTTPException(404, "File not found: " + relative_path)
    return candidate


def _load_scenario_configs(configs: List[ScenarioConfig]) -> List[Scenario]:
    if len(configs) < 2:
        raise HTTPException(400, "Select at least two SNP scenario files")
    scenarios = []
    port_counts = set()
    for config in configs:
        path = _safe_data_path(config.snp_filename)
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            dut = parse_touchstone(handle.read(), filename=config.snp_filename)
        port_counts.add(dut.num_ports)
        efficiency = None
        if config.efficiency_filename:
            efficiency = load_efficiency_file(_safe_data_path(config.efficiency_filename))
        kind = config.efficiency_kind.lower()
        if kind not in ('radiation', 'total'):
            raise HTTPException(400, "efficiency_kind must be radiation or total")
        scenarios.append(Scenario(
            filename=config.snp_filename,
            dut=dut,
            weight=max(config.weight, 0.0),
            efficiency=efficiency,
            efficiency_kind=kind,
        ))
    if len(port_counts) != 1:
        raise HTTPException(400, "All scenario SNP files must have the same port count")
    return scenarios

BAND_PRESETS = {
    "GPS L1": [1574, 1576], "GPS L5": [1176, 1177], "GPS L1+L5": [1176, 1576],
    "WiFi 2.4GHz": [2400, 2500], "WiFi 5GHz": [5150, 5850], "WiFi 6GHz": [5925, 7125],
    "UWB": [3100, 10600], "LTE B1": [1920, 2170], "LTE B3": [1710, 1880],
    "LTE B7": [2500, 2690], "5G n77": [3300, 4200], "5G n78": [3300, 3800],
    "5G n79": [4400, 5000], "Bluetooth": [2400, 2480], "NB-IoT": [700, 960],
    "LoRa 868": [863, 870], "ISM 915": [902, 928],
}

# ─── API Routes ───

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "db_active": state.use_db and state.db_library is not None}


@app.get("/api/calibration/status")
async def calibration_status():
    """Expose the exact hashed artifacts behind product search-quality claims."""
    try:
        single = await asyncio.to_thread(single_port_calibration_reference)
        multi = await asyncio.to_thread(multiport_calibration_reference)
        performance = await asyncio.to_thread(search_performance_reference)
        return {
            "status": "verified",
            "integrity": "sha256_verified_at_runtime",
            "single_port": single,
            "multiport": multi,
            "performance": performance,
        }
    except CalibrationEvidenceError as exc:
        return {"status": "invalid", "integrity": "failed_closed", "error": str(exc)}

@app.post("/api/config/dirs")
async def set_data_dirs(config: DataDirConfig):
    state.snp_dir = str(resolve_project_path(config.snp_dir))
    state.murata_dir = str(resolve_project_path(config.murata_dir))
    state.optenni_component_dir = config.optenni_component_dir
    state.environment_metadata_path = config.environment_metadata_path
    state.component_environment_catalog = None
    state.tunable_component_library = None
    state.full_component_library = None
    # A catalog change invalidates runtime model handles and measured-search
    # checkpoints. Keep the loaded SNP in AppState, but never continue an old
    # optimizer against a newly selected library.
    reset_session()
    state.last_solutions = []
    state.last_joint_results = None
    state.last_multi_scenario_results = None

    explicit_environment_path = str(config.environment_metadata_path or "").strip()
    conventional_environment_path = Path(state.murata_dir) / "component_environment.json"
    environment_path = (
        resolve_project_path(explicit_environment_path)
        if explicit_environment_path else conventional_environment_path.resolve()
    )
    if explicit_environment_path or environment_path.is_file():
        if not environment_path.is_file():
            raise HTTPException(400, f"Component environment metadata not found: {environment_path}")
        try:
            state.component_environment_catalog = load_component_environment_catalog(environment_path)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"Invalid component environment metadata: {exc}") from exc

    # For Optenni case1 parity, use only the real component families requested:
    # Murata GQM18 capacitors and Coilcraft 0402HP inductors.
    if config.optenni_component_dir and os.path.isdir(config.optenni_component_dir):
        optenni_library = scan_s2p_directory(config.optenni_component_dir)
        environment_matches = apply_component_environment_catalog(
            optenni_library.all_components, state.component_environment_catalog
        )
        state.full_component_library = optenni_library
        state.component_library = filter_component_library(
            optenni_library,
            inductor_tokens=["COILCRAFT INDUCTORS 0402HP", "0402HP", "04HP"],
            capacitor_tokens=["MURATA CAPACITORS GQM18", "GQM18"],
        )
        if state.component_library.inductors and state.component_library.capacitors:
            state.db_library = None
            state.use_db = False
            return {
                "status": "ok", "mode": "optenni_s2p_filtered",
                "component_dir": config.optenni_component_dir,
                "inductor_filter": "Coilcraft 0402HP",
                "capacitor_filter": "Murata GQM18",
                "inductors": len(state.component_library.inductors),
                "capacitors": len(state.component_library.capacitors),
                "unique_inductor_values": len(state.component_library.get_unique_inductor_values()),
                "unique_capacitor_values": len(state.component_library.get_unique_capacitor_values()),
                "environment_metadata": {
                    **state.component_environment_catalog.summary(),
                    "matched_components": environment_matches,
                } if state.component_environment_catalog else None,
            }

    # Try DB first — use the configured Murata directory, not a machine-specific path.
    for candidate, label in state.database_candidates():
        if os.path.isfile(candidate):
            try:
                state.db_library = load_murata_db(
                    candidate, environment_catalog=state.component_environment_catalog
                )
                state.full_component_library = None
                state.use_db = True
                stats = state.db_library.stats
                return {
                    "status": "ok", "mode": "database",
                    "db_label": label,
                    "db_path": candidate,
                    "total_components": stats['total_components'],
                    "primary_components": stats['primary_components'],
                    "inductors": stats.get('primary_inductors', 0),
                    "capacitors": stats.get('primary_capacitors', 0),
                    "environment_metadata": {
                        **state.component_environment_catalog.summary(),
                        **state.db_library.environment_coverage(),
                    } if state.component_environment_catalog else None,
                }
            except Exception as e:
                pass

    # Fallback to S2P scan
    if os.path.isdir(config.murata_dir):
        state.component_library = scan_murata_directory(config.murata_dir)
        environment_matches = apply_component_environment_catalog(
            state.component_library.all_components, state.component_environment_catalog
        )
        state.full_component_library = state.component_library
        state.use_db = False
        return {
            "status": "ok", "mode": "s2p_scan",
            "inductors": len(state.component_library.inductors),
            "capacitors": len(state.component_library.capacitors),
            "environment_metadata": {
                **state.component_environment_catalog.summary(),
                "matched_components": environment_matches,
            } if state.component_environment_catalog else None,
        }

    return {"status": "warning", "message": "Murata directory not found"}

@app.get("/api/snp/list")
async def list_snp_files():
    if not state.snp_dir or not os.path.isdir(state.snp_dir):
        return {"files": []}
    files = []
    invalid_files = []
    for root, dirs, filenames in os.walk(state.snp_dir):
        for f in sorted(filenames):
            if re.search(r'\.s\d+p$', f, flags=re.IGNORECASE):
                # Relative path from snp_dir
                rel_path = os.path.relpath(os.path.join(root, f), state.snp_dir)
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                        data = parse_touchstone(fh.read(), filename=rel_path)
                    files.append({
                        "filename": rel_path,
                        "num_ports": data.num_ports,
                        "freq_count": len(data.frequencies),
                        "freq_min_hz": min(data.frequencies) if data.frequencies else 0,
                        "freq_max_hz": max(data.frequencies) if data.frequencies else 0,
                        **_touchstone_reference_metadata(data),
                    })
                except (OSError, ValueError) as exc:
                    invalid_files.append({"filename": rel_path, "error": str(exc)})
    return {"files": files, "invalid_files": invalid_files}


@app.post("/api/snp/watch/start")
async def start_snp_watch(stable_ms: int = 1000, source: str = "CST"):
    """Start a client-owned watch for new or overwritten solver exports."""
    allowed_sources = {"CST", "HFSS", "VNA", "Touchstone", "other"}
    if source not in allowed_sources:
        raise HTTPException(400, f"Unsupported Touchstone source: {source}")
    if not state.snp_dir:
        raise HTTPException(400, "SNP directory not configured")
    try:
        return snp_monitor.start(
            state.snp_dir, stable_seconds=stable_ms / 1000.0, source=source
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/snp/watch/status")
async def snp_watch_status(watch_id: str):
    """Return solver exports that became stable since a watch was started."""
    try:
        result = snp_monitor.status(watch_id)
        for item in result["ready"]:
            path = Path(_safe_data_path(item["filename"]))
            digest = sha256_file(path)
            try:
                item["provenance"] = snp_provenance_store.record(
                    state.snp_dir, item["filename"], sha256=digest,
                    source=item["source"], ingestion_method="directory_watch",
                    details={"size_bytes": item["size"], "mtime_ns": item["mtime_ns"]},
                )
            except (OSError, TypeError, ValueError) as provenance_error:
                item["provenance_error"] = str(provenance_error)
        return result
    except KeyError as exc:
        raise HTTPException(404, "Touchstone watch session not found or expired") from exc


@app.post("/api/snp/watch/stop")
async def stop_snp_watch(watch_id: str):
    return {"stopped": snp_monitor.stop(watch_id), "watch_id": watch_id}


@app.get("/api/cst/status")
async def cst_status(force: bool = False):
    """Detect the official CST runtime and enumerate open Design Environments."""
    return await asyncio.to_thread(cst_bridge.status, force=force)


@app.get("/api/cst/project-tree")
async def cst_project_tree(pid: int, project_path: str):
    """Read S-parameter result nodes from an explicitly selected open project."""
    try:
        return await asyncio.to_thread(cst_bridge.project_tree, pid, project_path)
    except CSTBridgeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/cst/export-touchstone")
async def cst_export_touchstone(pid: int, project_path: str):
    """Export and validate the latest S-parameters from an open CST project."""
    if not state.snp_dir:
        raise HTTPException(400, "SNP directory not configured")
    root = Path(state.snp_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    export_base = root / f".rfmatch-cst-{uuid.uuid4().hex}"
    exported_path = None
    try:
        exported = await asyncio.to_thread(
            cst_bridge.export_touchstone, pid, project_path, export_base,
            allowed_root=root,
        )
        exported_path = Path(exported["exported_path"]).resolve()
        content_bytes = exported_path.read_bytes()
        content = content_bytes.decode("utf-8", errors="replace")
        parsed = parse_touchstone(content, filename=exported_path.name)
        network_digest = touchstone_network_sha256(parsed)

        project_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(project_path).stem).strip("-.") or "cst-result"
        suffix = exported_path.suffix.lower()
        resolved_project_path = str(Path(project_path).resolve())
        destination = root / f"{project_stem}{suffix}"
        duplicate_index = 2
        previous_sha256 = None
        previous_network_sha256 = None
        while destination.exists():
            existing_digest = sha256_file(destination)
            try:
                existing_provenance = snp_provenance_store.lookup(
                    root, os.path.relpath(destination, root), sha256=existing_digest,
                )
            except (OSError, TypeError, ValueError):
                existing_provenance = None
            details = existing_provenance.get("details", {}) if existing_provenance else {}
            same_cst_project = (
                existing_provenance
                and existing_provenance.get("ingestion_method") == "cst_python_bridge"
                and str(details.get("project_path", "")).casefold() == resolved_project_path.casefold()
            )
            if same_cst_project:
                previous_sha256 = existing_digest
                previous_network_sha256 = details.get("network_sha256")
                break
            destination = root / f"{project_stem}-{duplicate_index}{suffix}"
            duplicate_index += 1
        os.replace(exported_path, destination)
        exported_path = None

        digest = hashlib.sha256(content_bytes).hexdigest()
        relative_name = os.path.relpath(destination, root)
        provenance = snp_provenance_store.record(
            root, relative_name, sha256=digest, source="CST",
            ingestion_method="cst_python_bridge",
            details={
                "project_path": resolved_project_path,
                "pid": int(pid),
                "revision_of_sha256": previous_sha256,
                "network_sha256": network_digest,
            },
        )
        return {
            "status": "ok", "filename": relative_name, "source": "CST",
            "sha256": digest, "provenance": provenance,
            "replaced_existing": previous_sha256 is not None,
            "previous_sha256": previous_sha256,
            "network_sha256": network_digest,
            "content_changed": (
                previous_network_sha256 != network_digest
                if previous_network_sha256 else previous_sha256 != digest
            ),
            "num_ports": parsed.num_ports, "freq_count": len(parsed.frequencies),
            "freq_min_hz": min(parsed.frequencies), "freq_max_hz": max(parsed.frequencies),
            **_touchstone_reference_metadata(parsed),
        }
    except (CSTBridgeError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Unable to export CST S-parameters: {exc}") from exc
    finally:
        if exported_path is not None:
            try:
                exported_path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/api/snp/import")
async def import_snp(request: SNPImportRequest):
    """Import and validate a CST/HFSS/VNA Touchstone export into the SNP workspace."""
    if not state.snp_dir:
        raise HTTPException(400, "SNP directory not configured")
    filename = os.path.basename(request.filename.strip())
    if filename != request.filename.strip() or not re.fullmatch(r"[^\\/:*?\"<>|]+\.s\d+p", filename, re.IGNORECASE):
        raise HTTPException(400, "Select a Touchstone file named *.sNp (for example antenna.s2p)")
    try:
        parsed = parse_touchstone(request.content, filename=filename)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid {request.source} Touchstone export: {exc}") from exc

    root = os.path.realpath(os.path.abspath(state.snp_dir))
    os.makedirs(root, exist_ok=True)
    stem, suffix = os.path.splitext(filename)
    destination = os.path.join(root, filename)
    duplicate_index = 2
    while os.path.exists(destination):
        destination = os.path.join(root, f"{stem}-{duplicate_index}{suffix}")
        duplicate_index += 1
    temporary = destination + f".{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            handle.write(request.content)
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
        raise HTTPException(400, f"Unable to store imported Touchstone file: {exc}") from exc

    relative_name = os.path.relpath(destination, root)
    digest = hashlib.sha256(request.content.encode("utf-8")).hexdigest()
    provenance = None
    provenance_error = None
    try:
        provenance = snp_provenance_store.record(
            root, relative_name, sha256=digest, source=request.source,
            ingestion_method="file_import",
            details={"original_filename": request.filename},
        )
    except (OSError, TypeError, ValueError) as exc:
        provenance_error = str(exc)
    return {
        "status": "ok",
        "filename": relative_name,
        "source": request.source,
        "sha256": digest,
        "provenance": provenance,
        "provenance_error": provenance_error,
        "num_ports": parsed.num_ports,
        "freq_count": len(parsed.frequencies),
        "freq_min_hz": min(parsed.frequencies),
        "freq_max_hz": max(parsed.frequencies),
        **_touchstone_reference_metadata(parsed),
    }


@app.get("/api/efficiency-files/list")
async def list_efficiency_files():
    if not state.snp_dir or not os.path.isdir(state.snp_dir):
        return {"files": []}
    result = []
    for root, _, filenames in os.walk(state.snp_dir):
        for filename in sorted(filenames):
            if filename.lower().endswith(('.txt', '.csv', '.tsv', '.eff')):
                result.append(os.path.relpath(os.path.join(root, filename), state.snp_dir))
    return {"files": result}

@app.post("/api/snp/load")
async def load_snp(filename: str):
    filepath = _safe_data_path(filename)
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        parsed_snp = parse_touchstone(content, filename=filename)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Invalid Touchstone input: {exc}") from exc
    network_digest = touchstone_network_sha256(parsed_snp)
    previous_network_digest = None
    if state.loaded_snp is not None and state.loaded_snp_filename == filename:
        try:
            previous_network_digest = touchstone_network_sha256(state.loaded_snp)
        except (AttributeError, KeyError, TypeError, ValueError):
            previous_network_digest = None
    electrical_dut_changed = previous_network_digest != network_digest
    state.loaded_snp = parsed_snp
    state.loaded_snp_filename = filename
    input_digest = sha256_file(Path(filepath))
    provenance_error = None
    try:
        state.loaded_snp_provenance = snp_provenance_store.lookup(
            state.snp_dir, filename, sha256=input_digest
        )
    except (OSError, TypeError, ValueError) as exc:
        state.loaded_snp_provenance = None
        provenance_error = str(exc)
    # Preserve expensive candidates when CST only rewrites comments or formatting.
    # Any frequency/S/Z0 change still invalidates every DUT-dependent result.
    from engine.tuning_service import reset_session, get_session
    if electrical_dut_changed:
        state.last_solutions = []
        state.last_joint_results = None
        reset_session()
    session = get_session()
    session.dut = state.loaded_snp
    session.dut_filename = filename
    return {
        "status": "ok", "filename": filename,
        "num_ports": state.loaded_snp.num_ports,
        "freq_count": len(state.loaded_snp.frequencies),
        "freq_min_hz": min(state.loaded_snp.frequencies) if state.loaded_snp.frequencies else 0,
        "freq_max_hz": max(state.loaded_snp.frequencies) if state.loaded_snp.frequencies else 0,
        "frequencies": state.loaded_snp.frequencies[:100],
        "sha256": input_digest,
        "network_sha256": network_digest,
        "electrical_dut_changed": electrical_dut_changed,
        "provenance": state.loaded_snp_provenance,
        "provenance_error": provenance_error,
        **_touchstone_reference_metadata(state.loaded_snp),
    }

@app.get("/api/band-presets")
async def get_band_presets():
    return {"presets": BAND_PRESETS}

@app.get("/api/component-series")
async def list_component_series():
    _ensure_library()
    catalog = state.full_component_library or _get_library()
    active = _get_library()
    grouped = {}
    for component in [
        *(getattr(catalog, "inductors", []) or []),
        *(getattr(catalog, "capacitors", []) or []),
    ]:
        series_id = component_series_id(component)
        series_name = component_series_name(component)
        metadata = component_metadata(component)
        manufacturer = metadata["manufacturer"]
        package_code = metadata["package_code"]
        tolerance = metadata["tolerance_pct"]
        item = grouped.setdefault(series_id, {
            "id": series_id,
            "name": series_name,
            "component_type": getattr(component, "component_type", ""),
            "manufacturer": manufacturer,
            "package_code": package_code,
            "count": 0,
            "tolerance_min_pct": None,
            "tolerance_max_pct": None,
            "metadata_sources": {},
            "voltage_codes": set(),
            "dielectrics": set(),
        })
        item["count"] += 1
        for source in metadata["provenance"].values():
            item["metadata_sources"][source] = item["metadata_sources"].get(source, 0) + 1
        if tolerance is not None:
            tolerance = float(tolerance)
            item["tolerance_min_pct"] = (
                tolerance if item["tolerance_min_pct"] is None
                else min(item["tolerance_min_pct"], tolerance)
            )
            item["tolerance_max_pct"] = (
                tolerance if item["tolerance_max_pct"] is None
                else max(item["tolerance_max_pct"], tolerance)
            )
        if metadata["voltage_code"]:
            item["voltage_codes"].add(metadata["voltage_code"])
        if metadata["dielectric"]:
            item["dielectrics"].add(metadata["dielectric"])
    active_ids = sorted({
        component_series_id(component)
        for component in [
            *(getattr(active, "inductors", []) or []),
            *(getattr(active, "capacitors", []) or []),
        ]
    })
    series = sorted(grouped.values(), key=lambda item: (
        item["component_type"], item["manufacturer"], item["name"], item["id"]
    ))
    for item in series:
        item["voltage_codes"] = sorted(item["voltage_codes"])
        item["dielectrics"] = sorted(item["dielectrics"])
    return {
        "series": series,
        "default_selected": active_ids,
        "facets": {
            "manufacturers": sorted({item["manufacturer"] for item in series if item["manufacturer"]}),
            "package_codes": sorted({item["package_code"] for item in series if item["package_code"]}),
            "tolerance_available": any(item["tolerance_min_pct"] is not None for item in series),
            "voltage_codes": sorted({value for item in series for value in item["voltage_codes"]}),
            "dielectrics": sorted({value for item in series for value in item["dielectrics"]}),
        },
        "inductor_series": {
            item["id"]: item["count"] for item in series
            if item["component_type"] == "inductor"
        },
        "capacitor_series": {
            item["id"]: item["count"] for item in series
            if item["component_type"] == "capacitor"
        },
    }


@app.post("/api/component-library/preview")
async def preview_component_library(request: ComponentLibraryPreviewRequest):
    _ensure_library()
    _, metadata = _library_for_series(
        request.component_series, request.component_filter, validate=False
    )
    metadata["valid_for_lc_search"] = bool(
        metadata["inductors"] and metadata["capacitors"]
    )
    metadata["valid_for_measured_search"] = bool(
        metadata["inductors"] or metadata["capacitors"]
    )
    return metadata

@app.get("/api/components/list")
async def list_components(comp_type: Optional[str] = None, limit: int = 100):
    _ensure_library()
    if state.use_db and state.db_library:
        if comp_type == 'inductor':
            comps = state.db_library.db.get_primary_inductors()[:limit]
        elif comp_type == 'capacitor':
            comps = state.db_library.db.get_primary_capacitors()[:limit]
        else:
            comps = state.db_library.db.get_primary_inductors()[:limit//2] + \
                    state.db_library.db.get_primary_capacitors()[:limit//2]
        return {"components": [{"part_number": c.part_number, "component_type": c.component_type,
                "nominal_value": c.nominal_value, "nominal_unit": c.nominal_unit} for c in comps]}
    lib = state.component_library
    result = []
    if comp_type != 'capacitor':
        for c in lib.inductors[:limit]:
            result.append({"part_number": c.part_number, "component_type": "inductor",
                          "nominal_value": c.nominal_value, "nominal_unit": "nH"})
    if comp_type != 'inductor':
        for c in lib.capacitors[:limit]:
            result.append({"part_number": c.part_number, "component_type": "capacitor",
                          "nominal_value": c.nominal_value, "nominal_unit": "pF"})
    return {"components": result}


@app.get("/api/components/search")
async def search_components(q: str = "", comp_type: Optional[str] = None, limit: int = 80):
    _ensure_library()
    query = q.strip().lower()
    library = _get_library()
    if comp_type == 'inductor':
        source = list(library.inductors)
    elif comp_type == 'capacitor':
        source = list(library.capacitors)
    else:
        source = list(library.inductors) + list(library.capacitors)
    matches = []
    for component in source:
        label = f"{component.part_number} {component.nominal_value} {component.nominal_unit}".lower()
        if query and query not in label:
            continue
        matches.append({
            "part_number": component.part_number,
            "component_type": component.component_type,
            "nominal_value": component.nominal_value,
            "nominal_unit": component.nominal_unit,
            "series": getattr(component, 'series', ''),
        })
        if len(matches) >= min(max(limit, 1), 500):
            break
    return {"components": matches}


def _catalog_components(library=None):
    library = library or state.full_component_library or _get_library()
    return [
        *(getattr(library, "inductors", []) or []),
        *(getattr(library, "capacitors", []) or []),
    ]


def _component_model_identity(component) -> dict:
    filename = str(getattr(component, "s2p_filename", "") or "")
    archive = str(getattr(component, "zip_path", "") or "")
    digest = None
    source_kind = "database" if hasattr(component, "_record") else "directory"
    try:
        if archive == "__DIR__" and Path(filename).is_file():
            digest = sha256_file(Path(filename))
        elif archive and Path(archive).is_file() and filename:
            source_kind = "zip"
            with zipfile.ZipFile(archive, "r") as handle:
                digest = hashlib.sha256(handle.read(filename)).hexdigest().upper()
    except (OSError, KeyError, zipfile.BadZipFile):
        digest = None
    return {
        "source_kind": source_kind,
        "model_filename": Path(filename).name if filename else None,
        "archive_filename": Path(archive).name if archive and archive != "__DIR__" else None,
        "model_sha256": digest,
    }


def _component_detail(component, include_identity=True) -> dict:
    metadata = component_metadata(component)
    detail = {
        "part_number": getattr(component, "part_number", ""),
        "component_type": getattr(component, "component_type", ""),
        "nominal_value": getattr(component, "nominal_value", None),
        "nominal_unit": getattr(component, "nominal_unit", ""),
        "series": component_series_name(component),
        **{key: metadata[key] for key in (
            "manufacturer", "package_code", "tolerance_pct",
            "voltage_code", "dielectric", "tempco_ppm_per_c",
            "systematic_bias_pct", "environment_metadata",
        )},
        "metadata_provenance": metadata["provenance"],
    }
    if include_identity:
        detail.update(_component_model_identity(component))
    return detail


@app.get("/api/components/detail")
async def component_detail(part_number: str):
    _ensure_library()
    needle = part_number.strip().casefold()
    if not needle:
        raise HTTPException(400, "part_number is required")
    matches = [
        component for component in _catalog_components()
        if str(getattr(component, "part_number", "")).casefold() == needle
    ]
    if not matches:
        raise HTTPException(404, f"Component not found: {part_number}")
    return {"part_number": part_number, "matches": [_component_detail(item) for item in matches]}


def _request_band_frequencies(bands_mhz, num_points):
    bands = list(bands_mhz or [])
    if not bands:
        try:
            for port in (get_session().last_tuning_request or {}).get("ports", []):
                if port.get("enabled", True):
                    bands.extend(port.get("bands_mhz") or [])
        except NameError:
            pass
    if not bands:
        bands = [[2400.0, 2500.0]]
    frequencies = []
    for band in bands:
        if len(band) != 2 or float(band[0]) <= 0 or float(band[1]) < float(band[0]):
            raise HTTPException(400, "Each alternative-analysis band must be [start_mhz, stop_mhz]")
        frequencies.extend(np.linspace(float(band[0]), float(band[1]), num_points) * 1e6)
    return sorted({float(value) for value in frequencies})


@app.post("/api/components/alternatives")
async def component_alternatives(request: ComponentAlternativesRequest):
    """Rank replacement candidates by measured/modelled two-port behavior."""
    _ensure_library()
    needle = request.part_number.strip().casefold()
    original = next((
        component for component in _catalog_components()
        if str(getattr(component, "part_number", "")).casefold() == needle
    ), None)
    if original is None:
        raise HTTPException(404, f"Component not found: {request.part_number}")
    candidate_library, filter_metadata = _library_for_series(
        request.component_series, request.component_filter, validate=False
    )
    source = (
        getattr(candidate_library, "inductors", [])
        if original.component_type == "inductor"
        else getattr(candidate_library, "capacitors", [])
    ) or []
    original_value = float(original.nominal_value)
    maximum_fraction = request.maximum_nominal_deviation_pct / 100.0
    candidates = [
        component for component in source
        if str(getattr(component, "part_number", "")).casefold() != needle
        and abs(float(component.nominal_value) - original_value) / max(abs(original_value), 1e-30)
        <= maximum_fraction
    ]
    candidates.sort(key=lambda item: (
        abs(float(item.nominal_value) - original_value),
        str(getattr(item, "part_number", "")),
    ))
    preselection_limit = 500
    preselected = candidates[:preselection_limit]
    frequencies = _request_band_frequencies(request.bands_mhz, request.num_band_points)
    try:
        reference = np.asarray([
            original.get_s_matrix_at_freq(frequency) for frequency in frequencies
        ], dtype=complex)
    except Exception as exc:
        raise HTTPException(400, f"Unable to load reference component model: {exc}") from exc
    ranked = []
    failures = 0
    for component in preselected:
        try:
            matrices = np.asarray([
                component.get_s_matrix_at_freq(frequency) for frequency in frequencies
            ], dtype=complex)
            difference = np.abs(matrices - reference)
            if not np.all(np.isfinite(difference)):
                raise ValueError("non-finite model data")
            rms = float(np.sqrt(np.mean(difference ** 2)))
            maximum = float(np.max(difference))
            nominal_deviation = abs(float(component.nominal_value) - original_value) / max(
                abs(original_value), 1e-30
            )
            rank_score = rms + 0.05 * abs(np.log(max(float(component.nominal_value), 1e-30) / max(original_value, 1e-30)))
            ranked.append(({
                **_component_detail(component, include_identity=False),
                "nominal_deviation_pct": nominal_deviation * 100.0,
                "sparameter_rms_difference": rms,
                "sparameter_maximum_difference": maximum,
                "rank_score": float(rank_score),
            }, component))
        except Exception:
            failures += 1
    ranked.sort(key=lambda item: (
        item[0]["rank_score"], item[0]["sparameter_rms_difference"], item[0]["part_number"]
    ))
    alternatives = []
    for detail, component in ranked[:request.limit]:
        detail.update(_component_model_identity(component))
        alternatives.append(detail)
    return {
        "reference": _component_detail(original),
        "analysis_frequencies_hz": frequencies,
        "catalog_filter": filter_metadata,
        "nominal_candidates": len(candidates),
        "physically_evaluated": len(preselected) - failures,
        "model_failures": failures,
        "preselection_truncated": len(candidates) > preselection_limit,
        "ranking_basis": "measured/modelled S-matrix RMS plus a small nominal-value tie-break penalty",
        "alternatives": alternatives,
    }

@app.get("/api/topologies/list")
async def list_topologies(max_components: int = 4):
    topos = [t for t in get_standard_topologies() if t.num_components <= max_components]
    return {"topologies": [
        {"name": t.name, "num_components": t.num_components, "description": t.description,
         "elements": [{"position": e.position, "connection_type": e.connection_type.value,
                       "port": e.port, "component_type": e.component_type} for e in t.elements]}
        for t in topos
    ]}


def _multi_scenario_optimize_impl(
    request: MultiScenarioOptimizeRequest, progress_callback=None, cancel_check=None,
):
    _ensure_library()
    scenarios = _load_scenario_configs(request.scenarios)
    topologies = [
        topology for topology in get_standard_topologies()
        if topology.num_components == request.component_count
        and (not request.topology_names or topology.name in request.topology_names)
    ]
    if not topologies:
        raise HTTPException(400, "No topology matches the selected names and component count")
    try:
        optimizer = MultiScenarioOptimizer(
            scenarios=scenarios, library=_get_library(), bands_mhz=request.bands_mhz,
            input_port=request.input_port, num_band_points=request.num_band_points,
            objective=request.objective, beam_width=request.beam_width,
            timeout_seconds=request.timeout_seconds,
            max_candidates_per_position=request.max_candidates_per_position,
        )
        started = time.monotonic()
        solutions = optimizer.optimize(
            topologies, progress_callback=progress_callback, cancel_check=cancel_check,
        )
        search_elapsed = time.monotonic() - started
        search_diagnostics = optimizer.diagnostics()
        verification_started = time.monotonic()
        solutions, verification_diagnostics = optimizer.verify_solutions(
            solutions, request.verification_band_points,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
        verification_elapsed = time.monotonic() - verification_started
        elapsed = time.monotonic() - started
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    search_plan = build_multi_scenario_search_plan(request.model_dump())
    diagnostics = search_diagnostics
    diagnostics["verification"] = verification_diagnostics
    diagnostics["search_elapsed_seconds"] = search_elapsed
    diagnostics["verification_elapsed_seconds"] = verification_elapsed
    diagnostics["total_elapsed_seconds"] = elapsed
    diagnostics["budget_fraction"] = search_elapsed / max(request.timeout_seconds, 1e-9)
    for solution in solutions:
        solution["search_plan"] = search_plan
        solution["search_diagnostics"] = diagnostics
    state.last_multi_scenario_results = solutions
    return {
        "status": "ok", "solutions_count": len(solutions),
        "elapsed_seconds": elapsed, "search_plan": search_plan,
        "search_diagnostics": diagnostics, "solutions": solutions,
        "best_solution": solutions[0] if solutions else None,
    }


@app.post("/api/multi-scenario/optimize")
async def multi_scenario_optimize(request: MultiScenarioOptimizeRequest):
    """Compatibility endpoint; CPU work runs outside the API event loop."""
    return await asyncio.to_thread(_multi_scenario_optimize_impl, request)


@app.post("/api/multi-scenario/manual")
async def multi_scenario_manual(request: MultiScenarioManualRequest):
    _ensure_library()
    scenarios = _load_scenario_configs(request.scenarios)
    topology = next((t for t in get_standard_topologies() if t.name == request.topology_name), None)
    if topology is None:
        raise HTTPException(400, "Unknown topology: " + request.topology_name)
    if len(request.components) != topology.num_components:
        raise HTTPException(400, f"Topology requires {topology.num_components} components")
    specs = []
    for index, (component_config, element) in enumerate(zip(request.components, topology.elements)):
        use_ideal = bool(component_config.get('use_ideal', False))
        component = None
        if not use_ideal:
            part_number = str(component_config.get('part_number', '')).strip()
            component = find_component(_get_library(), part_number)
            if component is None:
                raise HTTPException(400, f"Component #{index + 1} not found: {part_number}")
            if component.component_type != element.component_type:
                raise HTTPException(400, f"Component #{index + 1} must be {element.component_type}")
        value = float(component_config.get('value', 1.0))
        specs.append({
            "position": element.position,
            "connection_type": element.connection_type.value,
            "component_type": element.component_type,
            "component": component,
            "value": value,
        })
    try:
        optimizer = MultiScenarioOptimizer(
            scenarios=scenarios, library=_get_library(), bands_mhz=request.bands_mhz,
            input_port=request.input_port, num_band_points=request.num_band_points,
            objective=request.objective, beam_width=1, timeout_seconds=10,
        )
        result = optimizer.evaluate(specs, topology.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result["components"] = [{
        "position": spec["position"], "connection_type": spec["connection_type"],
        "component_type": spec["component_type"],
        "part_number": spec["component"].part_number if spec["component"] else None,
        "nominal_value": spec["component"].nominal_value if spec["component"] else spec["value"],
        "nominal_unit": 'nH' if spec["component_type"] == 'inductor' else 'pF',
        "use_ideal": spec["component"] is None,
    } for spec in specs]
    return {"status": "ok", "result": result}

@app.post("/api/optimize", deprecated=True)
async def run_optimization(request: OptimizeRequest):
    _ensure_snp_loaded()
    _ensure_library()
    config = OptimizerConfig(
        target_frequency_hz=request.target_frequency_hz,
        max_components=request.max_components,
        beam_width=request.beam_width,
        timeout_seconds=request.timeout_seconds,
        bands_mhz=request.bands_mhz,
        num_band_points=request.num_band_points,
    )
    lib = _get_library()
    state.optimizer = MatchingOptimizer(state.loaded_snp, lib, config)
    port_states = {}
    for ps in request.port_states:
        smap = {'open': PortState.OPEN, 'short': PortState.SHORT, 'load': PortState.LOAD}
        port_states[ps.port_index] = smap.get(ps.state, PortState.LOAD)
    topos = get_standard_topologies()
    if request.topologies_filter:
        topos = [t for t in topos if t.name in request.topologies_filter]
    topos = [t for t in topos if t.num_components <= request.max_components]
    if port_states:
        solutions = state.optimizer.optimize_full(port_states=port_states, topologies=topos, input_port=request.input_port)
    else:
        solutions = state.optimizer.optimize_all_port_configs(topologies=topos, input_port=request.input_port)
    state.last_solutions = solutions
    return {"status": "ok", "solutions_count": len(solutions),
            "solutions": [s.to_dict() for s in solutions[:20]],
            "best_s11_db": solutions[0].s11_db if solutions else None,
            "best_vswr": solutions[0].vswr if solutions else None}

@app.post("/api/multipass", deprecated=True)
async def multi_port_optimize(request: MultiPortOptimizeRequest):
    _ensure_snp_loaded()
    _ensure_library()
    matching_ports = [p for p in request.ports if p.use_matching]
    if not matching_ports:
        raise HTTPException(400, "At least one port must have use_matching=True")
    lib = _get_library()
    results_per_port = {}
    all_results = []
    for mp in matching_ports:
        pi = mp.port_index
        band = mp.band_mhz
        center_hz = (band[0] + band[1]) / 2.0 * 1e6
        port_states = {}
        for p in request.ports:
            if p.port_index == pi:
                port_states[p.port_index] = PortState.COMPONENT
            else:
                smap = {'load': PortState.LOAD, 'short': PortState.SHORT, 'open': PortState.OPEN}
                port_states[p.port_index] = smap.get(p.state, PortState.LOAD)
        config = OptimizerConfig(
            target_frequency_hz=center_hz, max_components=mp.max_components,
            beam_width=request.beam_width, timeout_seconds=request.timeout_seconds,
            bands_mhz=[band], num_band_points=mp.num_band_points,
        )
        opt = MatchingOptimizer(state.loaded_snp, lib, config)
        topos = [t for t in get_standard_topologies() if t.num_components <= mp.max_components]
        solutions = opt.optimize_full(port_states=port_states, topologies=topos, input_port=pi)
        # Isolation
        isolation = {}
        if state.loaded_snp.num_ports > 1:
            n = state.loaded_snp.num_ports
            for other in range(n):
                if other == pi:
                    continue
                iso_key = "S%d,%d" % (pi+1, other+1)
                freqs_iso = np.linspace(band[0]*1e6, band[1]*1e6, mp.num_band_points)
                vals = [float(20*np.log10(max(abs(state.loaded_snp.get_s_matrix_interpolated(f)[pi, other]), 1e-15)))
                        for f in freqs_iso]
                isolation[iso_key] = {"avg_db": float(np.mean(vals)), "min_db": float(np.min(vals))}
        results_per_port[pi] = {
            "port_index": pi, "band_mhz": band, "max_components": mp.max_components,
            "solutions_count": len(solutions),
            "solutions": [s.to_dict() for s in solutions[:10]],
            "best_s11_db": solutions[0].s11_db if solutions else None,
            "best_vswr": solutions[0].vswr if solutions else None,
            "isolation": isolation,
        }
        all_results.extend(solutions)
    state.last_solutions = all_results
    return {"status": "ok", "ports_processed": len(matching_ports),
            "results_per_port": results_per_port, "total_solutions": len(all_results)}


@app.get("/api/multipass/sweep-all")
async def sweep_all_ports(start_hz: float = 2.0e9, stop_hz: float = 3.0e9, num_points: int = 200):
    """
    Sweep all matched ports simultaneously.
    Takes the best solution per port from last multipass run,
    applies ALL matching networks to the full S-matrix, returns Sii for each port.
    """
    if not state.loaded_snp:
        raise HTTPException(400, "No SNP loaded")

    from engine.network import (
        terminate_ports, _embed_series_on_port, _embed_shunt_to_ground
    )

    # Get last multipass results
    # We need to reconstruct the best matching per port from last_solutions
    # The last_solutions list has solutions from all ports mixed together
    # We need to group by port and pick the best per port

    # For now, use the port_states from the solutions to identify which port each belongs to
    # The best approach: store per-port best solutions separately
    # Re-run the sweep using the actual component choices from solutions

    freqs = np.linspace(start_hz, stop_hz, num_points)
    N = state.loaded_snp.num_ports

    # Find best solution per port (by return_loss_db)
    port_best = {}
    for sol in state.last_solutions:
        # Identify which port is the COMPONENT port
        comp_port = None
        for p, st in sol.port_states.items():
            if st == PortState.COMPONENT:
                comp_port = p
                break
        if comp_port is None:
            continue
        if comp_port not in port_best or sol.s11_magnitude < port_best[comp_port].s11_magnitude:
            port_best[comp_port] = sol

    if not port_best:
        raise HTTPException(400, "No solutions found. Run multipass optimization first.")

    # Compute raw Sii (without matching)
    raw_data = {}
    for pi in range(N):
        raw_data[pi] = {"s11_db": [], "s11_mag": []}
    for freq in freqs:
        S_full = state.loaded_snp.get_s_matrix_interpolated(freq)
        for pi in range(N):
            mag = abs(S_full[pi, pi])
            raw_data[pi]["s11_mag"].append(float(mag))
            raw_data[pi]["s11_db"].append(float(-20 * np.log10(max(mag, 1e-15))))

    # Compute matched Sii: apply ALL matching networks simultaneously
    matched_data = {}
    for pi in range(N):
        matched_data[pi] = {"s11_db": [], "s11_mag": [], "efficiency": []}

    for freq in freqs:
        S = state.loaded_snp.get_s_matrix_interpolated(freq).copy()

        # Apply each port's best matching network
        for comp_port, sol in port_best.items():
            for choice in sol.component_choices:
                try:
                    comp_s = choice.component.get_s_matrix_at_freq(freq)
                    if choice.connection_type == 'series':
                        S = _embed_series_on_port(S, comp_s, comp_port)
                    elif choice.connection_type == 'shunt':
                        S = _embed_shunt_to_ground(S, comp_s, comp_port)
                except Exception:
                    pass  # Skip failed embeddings

        # Extract Sii for each port
        for pi in range(N):
            if pi < S.shape[0]:
                mag = abs(S[pi, pi])
            else:
                mag = 1.0
            eff = (1 - mag**2) * 100
            matched_data[pi]["s11_mag"].append(float(mag))
            matched_data[pi]["s11_db"].append(float(-20 * np.log10(max(mag, 1e-15))))
            matched_data[pi]["efficiency"].append(float(eff))

    return {
        "frequencies": freqs.tolist(),
        "num_ports": N,
        "ports_matched": list(port_best.keys()),
        "per_port": {
            str(pi): {
                "raw_s11_db": raw_data[pi]["s11_db"],
                "matched_s11_db": matched_data[pi]["s11_db"],
                "matched_efficiency": matched_data[pi]["efficiency"],
                "best_solution": port_best[pi].to_dict() if pi in port_best else None,
            }
            for pi in range(N)
        },
    }


# ─── Per-port Radiation Efficiency ───

@app.post("/api/efficiency/load")
async def load_efficiency(request: EfficiencyLoadRequest):
    """
    Load radiation efficiency data for a specific port.
    port_index=-1 means apply to all ports (global).
    """
    filepath = request.filepath
    port_index = request.port_index
    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(400, "File not found: " + filepath)
    try:
        eff_data = load_efficiency_file(filepath)
        if port_index < 0:
            state.global_efficiency_data = eff_data
            port_label = "all ports (global)"
        else:
            state.per_port_efficiency_data[port_index] = eff_data
            port_label = f"port {port_index}"
        return {
            "status": "ok",
            "port_index": port_index,
            "port_label": port_label,
            "efficiency": eff_data.to_dict(),
        }
    except Exception as e:
        raise HTTPException(400, f"Failed to load efficiency file: {e}")


@app.post("/api/efficiency/inline")
async def load_efficiency_inline(request: EfficiencyInlineRequest):
    """Load efficiency data from pasted text content."""
    port_index = request.port_index
    content = request.content
    filename = request.filename
    if not content.strip():
        raise HTTPException(400, "Empty content")
    try:
        eff_data = parse_efficiency_data(content, filename=filename)
        if port_index < 0:
            state.global_efficiency_data = eff_data
        else:
            state.per_port_efficiency_data[port_index] = eff_data
        return {"status": "ok", "port_index": port_index, "efficiency": eff_data.to_dict()}
    except Exception as e:
        raise HTTPException(400, f"Failed to parse efficiency data: {e}")


@app.get("/api/efficiency/status")
async def efficiency_status():
    """Check which ports have efficiency data loaded."""
    result = {
        "loaded": state.global_efficiency_data is not None or bool(state.per_port_efficiency_data),
        "global": state.global_efficiency_data.to_dict() if state.global_efficiency_data else None,
        "per_port": {
            str(k): v.to_dict() for k, v in state.per_port_efficiency_data.items()
        },
    }
    return result


@app.post("/api/efficiency/clear")
async def clear_efficiency(request: EfficiencyClearRequest):
    """Clear efficiency data. port_index=-1 clears all."""
    port_index = request.port_index
    state.clear_efficiency(port_index)
    return {"status": "ok"}


@app.get("/api/optimization-modes")
async def list_optimization_modes():
    """Return available optimization modes with their descriptions."""
    return {
        "modes": [
            {
                "name": mode.name,
                "label": mode.label,
                "description": mode.description,
            }
            for mode in OPTIMIZATION_MODES.values()
        ]
    }


@app.post("/api/joint-optimize", deprecated=True)
async def joint_optimize(request: JointOptimizeRequest):
    _ensure_snp_loaded()
    _ensure_library()
    matching_ports = [p for p in request.ports if p.use_matching]
    if len(matching_ports) < 2:
        raise HTTPException(400, "Joint optimization requires at least 2 matching ports")
    lib = _get_library()

    # Build per-port configs with center frequency from each port's band
    port_configs = []
    band_list_mhz = []
    for mp in matching_ports:
        band = mp.band_mhz or [2400, 2500]
        center_hz = (band[0] + band[1]) / 2.0 * 1e6
        port_configs.append(PortMatchConfig(
            port_index=mp.port_index,
            max_components=mp.max_components,
            target_frequency_hz=center_hz,
        ))
        band_list_mhz.append(band)

    # Determine which efficiency data to use
    eff_data = None
    per_port_eff = None
    if state.per_port_efficiency_data:
        per_port_eff = dict(state.per_port_efficiency_data)
    if state.global_efficiency_data:
        eff_data = state.global_efficiency_data

    opt = JointMultiPortOptimizer(
        dut=state.loaded_snp,
        component_library=lib if not state.use_db else state.component_library,
        port_configs=port_configs,
        top_candidates_per_port=request.beam_width or 8,
        timeout_seconds=request.timeout_seconds or 120.0,
        min_avg_balance=0.5,
        optimization_mode=request.optimization_goal or 'efficiency',
        radiation_efficiency=eff_data,
        per_port_efficiency=per_port_eff,
    )

    # Run joint optimization with multi-point band evaluation
    t_start = time.time()
    joint_solutions = opt.optimize(
        bands_mhz=band_list_mhz if band_list_mhz else None,
        num_band_points=5,
    )
    total_time_s = time.time() - t_start
    state.last_joint_results = joint_solutions

    if not joint_solutions:
        return {"status": "ok", "mode": "joint",
                "ports_optimized": [mp.port_index for mp in matching_ports],
                "total_time_s": total_time_s, "solutions_count": 0,
                "results_per_port": {},
                "warning": "No valid joint solutions found"}

    best = joint_solutions[0]

    # Build per-port results
    results_per_port = {}
    for pi in [mp.port_index for mp in matching_ports]:
        if pi in best.port_metrics:
            m = best.port_metrics[pi]
            power_bal = best.power_balance.get(pi, {})
            results_per_port[str(pi)] = {
                "s11_magnitude": m.get('s11_magnitude', 0),
                "s11_db": m.get('s11_db', 0),
                "efficiency_pct": m.get('mismatch_efficiency', 0) * 100,
                "coupling_loss": m.get('coupling_loss', 0),
                "total_efficiency": m.get('total_efficiency', 0),
                "radiated_efficiency": m.get('radiated_efficiency', 0),
                "component_loss": power_bal.get('component_loss', 0),
                "power_balance": power_bal,
                "components": [
                    {
                        "part": c.get('part', ''),
                        "type": c.get('type', ''),
                        "value": c.get('value', ''),
                    }
                    for c in best.to_dict().get('components_summary', {}).get(str(pi), [])
                ],
            }
        else:
            results_per_port[str(pi)] = {"s11_magnitude": 1, "s11_db": 0, "efficiency_pct": 0}

    return {
        "status": "ok", "mode": "joint",
        "ports_optimized": [mp.port_index for mp in matching_ports],
        "total_time_s": total_time_s,
        "results_per_port": results_per_port,
        "system_metrics": {
            "balanced_score": best.balanced_score,
            "min_system_efficiency": best.min_system_efficiency,
            "avg_system_efficiency": best.avg_system_efficiency,
            "min_total_efficiency": best.min_total_efficiency,
            "avg_total_efficiency": best.avg_total_efficiency,
            "max_coupling_loss": best.max_coupling_loss,
            "component_loss_total": best.component_loss_total,
        },
        "solutions_count": len(joint_solutions),
    }

@app.get("/api/joint-optimize/sweep")
async def joint_optimize_sweep(
    port_index: int = 0,
    start_hz: float = 1e9,
    stop_hz: float = 3e9,
    num_points: int = 200,
):
    """Frequency sweep for a specific port using the best joint-optimize solution."""
    _ensure_snp_loaded()
    if not state.last_joint_results:
        raise HTTPException(400, "No joint optimization results. Run /api/joint-optimize first.")

    from engine.network import _embed_series_on_port, _embed_shunt_to_ground

    best = state.last_joint_results[0]
    freqs = np.linspace(start_hz, stop_hz, num_points)
    N = state.loaded_snp.num_ports

    # Build per-port matching networks from the joint solution
    # port_solutions: dict {port_index: MatchingSolution}
    port_matching = {}
    for pi, sol in best.port_solutions.items():
        port_matching[pi] = sol.component_choices  # list of ComponentChoice

    raw_db, raw_mag, raw_real, raw_imag = [], [], [], []
    match_db, match_mag, match_real, match_imag = [], [], [], []

    for freq in freqs:
        S_base = state.loaded_snp.get_s_matrix_interpolated(freq)

        # Raw S11 for this port
        s11_raw = complex(S_base[port_index, port_index])
        rm = abs(s11_raw)
        raw_db.append(float(-20 * np.log10(max(rm, 1e-15))))
        raw_mag.append(float(rm))
        raw_real.append(float(s11_raw.real))
        raw_imag.append(float(s11_raw.imag))

        # Apply ALL matching networks simultaneously
        S = S_base.copy()
        for pi, choices in port_matching.items():
            for choice in choices:
                try:
                    comp_s = choice.component.get_s_matrix_at_freq(freq)
                    if choice.connection_type == 'series':
                        S = _embed_series_on_port(S, comp_s, pi)
                    elif choice.connection_type == 'shunt':
                        S = _embed_shunt_to_ground(S, comp_s, pi)
                except Exception:
                    pass

        s11_m = complex(S[port_index, port_index]) if port_index < S.shape[0] else 1.0+0j
        mm = abs(s11_m)
        match_db.append(float(-20 * np.log10(max(mm, 1e-15))))
        match_mag.append(float(mm))
        match_real.append(float(s11_m.real))
        match_imag.append(float(s11_m.imag))

    return {
        "frequencies": freqs.tolist(),
        "s11_db": match_db,
        "s11_magnitude": match_mag,
        "s11_real": match_real,
        "s11_imag": match_imag,
        "raw_db": raw_db,
        "raw_magnitude": raw_mag,
        "raw_real": raw_real,
        "raw_imag": raw_imag,
        "port_index": port_index,
    }

@app.get("/api/optimize/results")
async def get_results(limit: int = 50):
    if not state.last_solutions:
        return {"solutions": [], "count": 0}
    return {"solutions": [s.to_dict() for s in state.last_solutions[:limit]],
            "count": len(state.last_solutions)}

@app.get("/api/snp/frequency-sweep")
async def frequency_sweep(start_hz: float = 40e6, stop_hz: float = 100e6,
                          num_points: int = 200, solution_index: int = 0):
    if not state.last_solutions:
        raise HTTPException(400, "No optimization results")
    if solution_index >= len(state.last_solutions):
        raise HTTPException(404, "Solution index out of range")
    sol = state.last_solutions[solution_index]
    freqs = np.linspace(start_hz, stop_hz, num_points)
    s11_mags, s11_dbs, s11_real, s11_imag = [], [], [], []
    for freq in freqs:
        S = state.loaded_snp.get_s_matrix_interpolated(freq)
        term = {p: 0.0 for p, st in sol.port_states.items() if st in [PortState.OPEN, PortState.SHORT, PortState.LOAD]}
        if term:
            from engine.network import terminate_ports
            S = terminate_ports(S, term)
        try:
            for ch in sol.component_choices:
                cs = ch.component.get_s_matrix_at_freq(freq)
                if ch.connection_type == 'series':
                    S = _embed_series_on_port(S, cs, ch.port)
                elif ch.connection_type == 'shunt':
                    S = _embed_shunt_to_ground(S, cs, ch.port)
                elif ch.connection_type == 'parallel':
                    S = connect_2port_to_multiport(S, cs, ch.port, ch.port2)
        except:
            s11_mags.append(1.0); s11_dbs.append(0.0)
            s11_real.append(1.0); s11_imag.append(0.0)
            continue
        if S.shape[0] > 0:
            s11 = S[0, 0]
            s11_mags.append(float(abs(s11)))
            s11_dbs.append(float(-20*np.log10(max(abs(s11), 1e-15))))
            s11_real.append(float(s11.real))
            s11_imag.append(float(s11.imag))
        else:
            s11_mags.append(1.0); s11_dbs.append(0.0)
            s11_real.append(1.0); s11_imag.append(0.0)
    # Raw S11
    raw_mags, raw_dbs, raw_real, raw_imag = [], [], [], []
    for freq in freqs:
        S_raw = state.loaded_snp.get_s_matrix_interpolated(freq)
        term = {p: 0.0 for p, st in sol.port_states.items() if st in [PortState.OPEN, PortState.SHORT, PortState.LOAD]}
        if term:
            from engine.network import terminate_ports
            S_raw = terminate_ports(S_raw, term)
        s11r = S_raw[0, 0] if S_raw.shape[0] > 0 else 1.0
        raw_mags.append(float(abs(s11r)))
        raw_dbs.append(float(-20*np.log10(max(abs(s11r), 1e-15))))
        raw_real.append(float(s11r.real))
        raw_imag.append(float(s11r.imag))
    return {"frequencies": freqs.tolist(),
            "s11_magnitude": s11_mags, "s11_db": s11_dbs,
            "s11_real": s11_real, "s11_imag": s11_imag,
            "raw_magnitude": raw_mags, "raw_db": raw_dbs,
            "raw_real": raw_real, "raw_imag": raw_imag,
            "solution": sol.to_dict()}

@app.post("/api/manual-tune")
async def manual_tune(request: ManualTuneRequest):
    _ensure_snp_loaded()
    active_filename = state.loaded_snp_filename
    if active_filename and request.snp_filename != active_filename:
        raise HTTPException(
            409,
            f"Manual-tuning DUT changed: requested {request.snp_filename!r}, "
            f"but the active DUT is {active_filename!r}",
        )
    needs_measured_library = any(
        isinstance(component, dict)
        and not bool(component.get("use_ideal", True))
        and str(component.get("comp_type", "")).lower() in {"inductor", "capacitor"}
        for component in request.components
    )
    if needs_measured_library:
        _ensure_library()
    dut = state.loaded_snp
    network_digest = touchstone_network_sha256(dut)
    if (
        request.expected_network_sha256
        and request.expected_network_sha256 != network_digest
    ):
        raise HTTPException(
            409,
            "Manual-tuning DUT revision changed; discard the stale result and recompute",
        )
    try:
        result = await asyncio.to_thread(
            run_manual_tuning_physical,
            dut,
            _get_library(),
            target_frequency_hz=request.target_frequency_hz,
            input_port=request.input_port,
            port_states=[_model_dump(item) for item in request.port_states],
            components=request.components,
            sweep_start_hz=request.sweep_start_hz,
            sweep_stop_hz=request.sweep_stop_hz,
            sweep_points=request.sweep_points,
            use_snp_points=request.use_snp_points,
        )
        result["dut_identity"] = {
            "filename": active_filename or request.snp_filename,
            "network_sha256": network_digest,
        }
        return result
    except (OSError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, f"Unable to evaluate manual network: {exc}") from exc

# ═══════════════════════════════════════════════════════════════════════════
# TUNING API — unified Optenni-style entry point
# ═══════════════════════════════════════════════════════════════════════════

from engine.tuning_service import (
    TuningSession, TuningResult, PerPortTuningMetrics,
    get_session, reset_session,
    run_tuning_single, run_tuning_joint, run_tuning_grid_s2p, run_tuning_tunable_c, run_tuning_tunable_mdif, run_tuning_tunable_mdif_auto, run_tuning_switch_mdif_auto, run_tuning_yield_analysis, run_manual_tuning_physical, run_manual_yield_analysis_physical, optimize_manual_network_physical, run_tuning_transmission_line, core_s2p_layout_from_touchstone,
    compute_sweep as _service_sweep,
    compute_transmission_line_sweep as _service_line_sweep,
    compute_power_balance as _service_pb,
    normalize_allowed_topology_codes,
)


@app.post("/api/tuning/optimize")
async def tuning_optimize(request: TuningOptimizeRequest):
    return await _tuning_optimize_impl(request)


def _tuning_result_network_signature(result) -> tuple:
    payload = result.to_dict()
    per_port = payload.get("per_port") or {}
    components = []
    for port, metrics in sorted(per_port.items(), key=lambda item: int(item[0])):
        items = []
        for component in metrics.get("components") or []:
            items.append((
                component.get("connection_type") or component.get("connection"),
                component.get("type") or component.get("comp_type"),
                component.get("part_number") or component.get("part"),
                str(component.get("value")),
            ))
        components.append((int(port), tuple(items)))
    return (
        payload.get("mode"), tuple(components),
        tuple(sorted((payload.get("tunable_states") or {}).items())),
    )


@app.post("/api/tuning/continue")
async def tuning_continue(request: TuningContinueRequest):
    return await _tuning_continue_impl(request)


async def _tuning_continue_impl(
    request: TuningContinueRequest,
    progress_callback=None,
    cancel_check=None,
):
    """Continue a live measured search, with deterministic rerun as fallback."""
    session = get_session()
    if not session.last_tuning_request or not session.candidate_solutions:
        raise HTTPException(409, "No live tuning result is available to continue")
    if session.restoration_mode != "live":
        raise HTTPException(409, "Snapshot results must be recomputed before they can be continued")
    previous_candidates = list(session.candidate_solutions)
    previous_request = dict(session.last_tuning_request)
    if previous_request.get("mode", "single") not in {"single", "joint"}:
        raise HTTPException(409, "Continue with larger budget currently supports lumped single/joint tuning")
    previous_timeout = float(previous_request.get("timeout_seconds", 0.0))
    next_timeout = previous_timeout + float(request.additional_seconds)
    continued_from_quality = previous_request.get("search_quality", "auto")
    # A continuation is an intentional custom cumulative budget. Reapplying
    # the original named preset during model validation would reset the total.
    previous_request["search_quality"] = "custom"
    previous_request["timeout_seconds"] = next_timeout
    checkpoint = session.search_checkpoint
    if checkpoint is not None:
        response = await _tuning_optimize_impl(
            TuningOptimizeRequest(**previous_request),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            resume_checkpoint=checkpoint,
            continuation_budget_seconds=float(request.additional_seconds),
        )
    else:
        response = await _tuning_optimize_impl(
            TuningOptimizeRequest(**previous_request),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    new_candidates = list(get_session().candidate_solutions)
    merged = {}
    for candidate in [*previous_candidates, *new_candidates]:
        signature = _tuning_result_network_signature(candidate)
        current = merged.get(signature)
        if current is None or candidate.system_score > current.system_score:
            merged[signature] = candidate
    ranked = sorted(
        merged.values(),
        key=lambda item: (bool(item.isolation_constraints_passed), item.system_score),
        reverse=True,
    )[:100]
    active_checkpoint = get_session().search_checkpoint or {}
    checkpoint_reused = bool(active_checkpoint.get("resumed"))
    continuation = {
        "strategy": (
            "in_memory_measured_checkpoint"
            if checkpoint_reused else "deterministic_rerun_merge"
        ),
        "checkpoint_reused": checkpoint_reused,
        "previous_timeout_seconds": previous_timeout,
        "additional_seconds": float(request.additional_seconds),
        "total_timeout_seconds": next_timeout,
        "continued_from_quality": continued_from_quality,
        "previous_candidates": len(previous_candidates),
        "new_candidates": len(new_candidates),
        "merged_candidates": len(ranked),
    }
    for index, candidate in enumerate(ranked):
        candidate.solution_index = index
    if ranked:
        if ranked[0].search_diagnostics is None:
            ranked[0].search_diagnostics = {}
        ranked[0].search_diagnostics["continuation"] = continuation
    session = get_session()
    session.candidate_solutions = ranked
    session.selected_index = 0
    solution_dicts = [candidate.to_dict() for candidate in ranked]
    best = ranked[0] if ranked else None
    response.update({
        "solutions_count": len(ranked),
        "solutions": solution_dicts,
        "best_solution": solution_dicts[0] if solution_dicts else None,
        "best_score": best.system_score if best else 0.0,
        "best_avg_efficiency": best.avg_total_efficiency if best else 0.0,
        "best_min_efficiency": best.min_total_efficiency if best else 0.0,
        "enabled_port_avg_efficiency": best.avg_total_efficiency if best else 0.0,
        "system_efficiency_pct": (
            (best.system_power_balance or {}).get("system_efficiency", best.avg_total_efficiency) * 100.0
            if best else 0.0
        ),
        "efficiency_basis": best.efficiency_basis if best else "",
        "system_power_balance": solution_dicts[0].get("system_power_balance") if solution_dicts else None,
        "power_balance_chart": solution_dicts[0].get("power_balance_chart") if solution_dicts else [],
        "continuation": continuation,
    })
    return response


async def _tuning_optimize_impl(
    request: TuningOptimizeRequest,
    progress_callback=None,
    cancel_check=None,
    resume_checkpoint=None,
    continuation_budget_seconds=None,
):
    """
    Unified tuning entry point — the ONE endpoint for all tuning operations.
    
    Determines single vs joint based on how many ports are enabled.
    Returns a list of candidate TuningResult objects, stored in TuningSession.
    """
    _ensure_snp_loaded()
    enabled_ports = [p for p in request.ports if p.get('enabled', True)]
    if not enabled_ports:
        raise HTTPException(400, "At least one port must be enabled")
    component_limits = []
    for port in enabled_ports:
        raw_limit = port.get("max_components", 2)
        if isinstance(raw_limit, bool):
            raise HTTPException(400, "max_components must be an integer between zero and six")
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "max_components must be an integer between zero and six") from exc
        if limit != raw_limit or not 0 <= limit <= 6:
            raise HTTPException(400, "max_components must be an integer between zero and six")
        component_limits.append(limit)
        if request.mode in {"single", "joint"}:
            try:
                allowed_codes = normalize_allowed_topology_codes(
                    port.get("allowed_topology_codes"),
                    limit,
                    port_index=int(port.get("port_index", 0)),
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
            if allowed_codes is not None:
                port["allowed_topology_codes"] = sorted(allowed_codes)
    requested_component_count = any(limit > 0 for limit in component_limits)
    requires_component_library = request.mode != "transmission_line" and not (
        request.mode == "switch" and not request.switch_measured_refine
    )
    if request.mode in {"single", "joint"} and not requested_component_count:
        requires_component_library = False
    if requires_component_library:
        _ensure_library()

    if not requires_component_library:
        lib = _get_library()
        library_filter = {
            "mode": "not_required", "selected_series": request.component_series,
            "reason": (
                "zero-component bare-DUT analysis"
                if request.mode in {"single", "joint"}
                else "component library is not used by this mode"
            ),
            "inductors": 0, "capacitors": 0,
        }
    else:
        lib, library_filter = _library_for_series(
            request.component_series,
            request.component_filter,
            required_component_types=(
                () if request.mode in {"single", "joint"} else ("inductors", "capacitors")
            ),
            require_any_component=request.mode in {"single", "joint"},
        )
    dut = state.loaded_snp
    enabled_indices = {int(port.get("port_index", 0)) for port in enabled_ports}
    for target in request.isolation_targets:
        if target.source_port == target.destination_port:
            raise HTTPException(400, "Isolation source and destination ports must be different")
        if target.source_port not in enabled_indices or target.destination_port not in enabled_indices:
            raise HTTPException(400, "Isolation targets must reference enabled ports")
        if len(target.band_mhz) != 2 or min(target.band_mhz) <= 0 or target.band_mhz[0] == target.band_mhz[1]:
            raise HTTPException(400, "Isolation target band_mhz must contain two distinct positive values")
        if target.weight < 0 or not 0 <= target.average_weight <= 1:
            raise HTTPException(400, "Isolation target weight or average_weight is invalid")

    session = get_session()
    if resume_checkpoint is None:
        session.search_checkpoint = None
    session.last_tuning_request = _model_dump(request)
    session.restoration_mode = "live"
    
    t_start = time.time()
    force_joint = len(enabled_ports) > 1
    
    actual_mode = "joint" if force_joint else request.mode
    next_search_checkpoint = None

    if request.mode == "transmission_line":
        if len(enabled_ports) > 1:
            raise HTTPException(400, "Transmission-line synthesis currently supports one enabled port")
        pc = enabled_ports[0]
        try:
            layout_blocks = []
            for block_request in request.transmission_line.layout_blocks:
                block = _model_dump(block_request)
                filename = block["filename"]
                if not filename.lower().endswith(".s2p"):
                    raise ValueError(f"layout block {filename!r} must use the .s2p extension")
                layout_path = Path(_safe_data_path(filename))
                layout = parse_touchstone(
                    layout_path.read_text(encoding="utf-8", errors="replace"),
                    filename=filename,
                )
                fixture_metadata = {}

                def load_fixture(side):
                    fixture_filename = block.get(f"{side}_fixture_filename")
                    if not fixture_filename:
                        return None
                    if not str(fixture_filename).lower().endswith(".s2p"):
                        raise ValueError(f"{side} fixture {fixture_filename!r} must use the .s2p extension")
                    fixture_path = Path(_safe_data_path(fixture_filename))
                    fixture_data = parse_touchstone(
                        fixture_path.read_text(encoding="utf-8", errors="replace"),
                        filename=fixture_filename,
                    )
                    reversed_ports = bool(block.get(f"{side}_fixture_reverse_ports", False))
                    fixture_metadata[side] = {
                        "filename": fixture_filename,
                        "sha256": sha256_file(fixture_path),
                        "reverse_ports": reversed_ports,
                    }
                    return fixture_data, reversed_ports

                left_fixture = load_fixture("left")
                right_fixture = load_fixture("right")
                target_z0 = (
                    float(dut.reference_resistance)
                    if block.get("reference_impedance_mode", "native") == "system"
                    else None
                )
                model, passivity = core_s2p_layout_from_touchstone(
                    layout,
                    reverse_ports=bool(block.get("reverse_ports", False)),
                    target_reference_impedance_ohm=target_z0,
                    left_fixture=left_fixture,
                    right_fixture=right_fixture,
                    maximum_deembedding_condition_number=float(
                        block.get("maximum_deembedding_condition_number", 1e10)
                    ),
                )
                if block["passivity_policy"] == "reject" and not passivity["passive"]:
                    raise ValueError(
                        f"layout block {filename!r} is non-passive; maximum singular value "
                        f"is {passivity['maximum_singular_value']:.6g}"
                    )
                layout_blocks.append({
                    "model": model,
                    "filename": filename,
                    "sha256": sha256_file(layout_path),
                    "location": block["location"],
                    "passivity_policy": block["passivity_policy"],
                    "reverse_ports": bool(block.get("reverse_ports", False)),
                    "reference_impedance_mode": block.get("reference_impedance_mode", "native"),
                    "fixtures": fixture_metadata,
                    "passivity": passivity,
                })
            candidates = run_tuning_transmission_line(
                dut,
                port_index=int(pc.get("port_index", 0)),
                bands_mhz=pc.get("bands_mhz", [[2400, 2500]]),
                objective=request.objective,
                num_band_points=request.num_band_points,
                search_config=_model_dump(request.transmission_line),
                timeout_seconds=request.timeout_seconds,
                global_efficiency=state.global_efficiency_data,
                per_port_efficiency=state.per_port_efficiency_data or None,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                layout_blocks=layout_blocks,
            )
            actual_mode = "transmission_line"
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise HTTPException(400, str(exc)) from exc
    elif request.mode == "grid_s2p":
        candidates = run_tuning_grid_s2p(
            dut=dut,
            port_specs=enabled_ports,
            objective=request.objective,
            num_band_points=request.num_band_points,
        )
        actual_mode = "grid_s2p"
    elif request.mode == "switch" and request.tuner_mdif_path:
        if len(enabled_ports) > 1:
            raise HTTPException(400, "Switch mode currently supports only a single-port DUT")
        try:
            switch_library = None
            if request.switch_measured_refine:
                switch_library = _get_library()
                if state.optenni_component_dir and os.path.isdir(state.optenni_component_dir):
                    if state.tunable_component_library is None:
                        state.tunable_component_library = _scan_tunable_component_families(
                            state.optenni_component_dir
                        )
                    switch_library = state.tunable_component_library
            candidates = run_tuning_switch_mdif_auto(
                dut=dut,
                port_index=int(enabled_ports[0].get("port_index", 0)),
                switch_mdif_path=request.tuner_mdif_path,
                frequency_configurations=[_model_dump(item) for item in request.frequency_configurations],
                objective=request.objective,
                state_options_by_configuration=request.switch_state_options,
                library=switch_library,
                measured_refine=request.switch_measured_refine,
                max_input_components=request.switch_max_input_components,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            actual_mode = "switch"
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
    elif request.mode == "tunable" and request.tuner_mdif_path:
        if len(enabled_ports) > 1:
            raise HTTPException(400, "Tunable mode currently supports only a single-port DUT")
        try:
            tunable_library = lib
            if state.optenni_component_dir and os.path.isdir(state.optenni_component_dir):
                if state.tunable_component_library is None:
                    state.tunable_component_library = _scan_tunable_component_families(
                        state.optenni_component_dir
                    )
                if state.tunable_component_library.inductors and state.tunable_component_library.capacitors:
                    tunable_library = state.tunable_component_library
            tunable_runner = (
                run_tuning_tunable_mdif_auto
                if request.tunable_auto_synthesize
                else run_tuning_tunable_mdif
            )
            candidates = tunable_runner(
                dut=dut,
                library=tunable_library,
                port_index=int(enabled_ports[0].get("port_index", 0)),
                tuner_mdif_path=request.tuner_mdif_path,
                frequency_configurations=[_model_dump(item) for item in request.frequency_configurations],
                objective=request.objective,
                **({} if request.tunable_auto_synthesize else {
                    "fixed_components": [_model_dump(item) for item in request.tunable_fixed_components]
                }),
                **({
                    "progress_callback": progress_callback,
                    "cancel_check": cancel_check,
                } if request.tunable_auto_synthesize else {}),
            )
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
    elif request.mode == "tunable" and request.band_state_map:
        # Tunable C multi-state mode — currently single-port only
        if len(enabled_ports) > 1:
            raise HTTPException(400, "Tunable mode currently supports only single-port optimization")
        pc = enabled_ports[0]
        candidates = run_tuning_tunable_c(
            dut=dut, library=lib,
            port_index=pc.get('port_index', 0),
            band_state_map=request.band_state_map,
            beam_width=request.beam_width,
            num_band_points=request.num_band_points,
            global_efficiency=state.global_efficiency_data,
            per_port_efficiency=state.per_port_efficiency_data or None,
        )
    elif len(enabled_ports) == 1 and not force_joint:
        pc = enabled_ports[0]
        next_search_checkpoint = {}
        candidates = run_tuning_single(
            dut=dut, library=lib,
            port_index=pc.get('port_index', 0),
            bands_mhz=pc.get('bands_mhz', [[2400, 2500]]),
            band_weights=pc.get('band_weights'),
            port_weight=pc.get('port_weight', 1.0),
            max_components=pc.get('max_components', 2),
            allowed_topology_codes=pc.get('allowed_topology_codes'),
            objective=request.objective,
            within_band_average_weight=request.within_band_average_weight,
            across_band_average_weight=request.across_band_average_weight,
            generic_synthesis_loss=_model_dump(request.generic_synthesis_loss),
            beam_width=request.beam_width,
            timeout_seconds=(
                float(continuation_budget_seconds)
                if resume_checkpoint is not None and continuation_budget_seconds is not None
                else request.timeout_seconds
            ),
            num_band_points=request.num_band_points,
            global_efficiency=state.global_efficiency_data,
            per_port_efficiency=state.per_port_efficiency_data or None,
            search_checkpoint=resume_checkpoint,
            checkpoint_store=next_search_checkpoint,
            search_profile_timeout_seconds=request.timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    else:
        next_search_checkpoint = {}
        candidates = run_tuning_joint(
            dut=dut, library=lib,
            port_specs=enabled_ports,
            objective=request.objective,
            beam_width=request.beam_width,
            timeout_seconds=(
                float(continuation_budget_seconds)
                if resume_checkpoint is not None and continuation_budget_seconds is not None
                else request.timeout_seconds
            ),
            num_band_points=request.num_band_points,
            global_efficiency=state.global_efficiency_data,
            per_port_efficiency=state.per_port_efficiency_data or None,
            debug=request.debug,
            debug_top_n=request.debug_top_n,
            isolation_targets=[_model_dump(target) for target in request.isolation_targets],
            search_checkpoint=resume_checkpoint,
            checkpoint_store=next_search_checkpoint,
            search_profile_timeout_seconds=request.timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    
    elapsed = time.time() - t_start
    
    if not candidates:
        return {"status": "ok", "solutions_count": 0, "solutions": [], "warning": "No valid solutions found"}

    for candidate in candidates.values():
        if candidate.search_diagnostics is None:
            candidate.search_diagnostics = {}
        candidate.search_diagnostics["component_library_filter"] = library_filter
        candidate.search_diagnostics["search_quality_requested"] = request.search_quality
        candidate.search_diagnostics["search_plan"] = build_search_plan(
            _model_dump(request)
        )
    
    # Store in session
    session.dut = dut
    session.library = lib
    session.search_checkpoint = next_search_checkpoint or None
    session.candidate_solutions = list(candidates.values())
    session.selected_index = 0
    
    solution_dicts = [s.to_dict() for s in session.candidate_solutions]
    best = session.candidate_solutions[0]
    
    # Extract system-level efficiency from power balance
    pb_sys_eff = 0.0
    if solution_dicts:
        spb = solution_dicts[0].get('system_power_balance')
        if spb:
            pb_sys_eff = spb.get('system_efficiency', 0.0)

    result = {
        "status": "ok", "mode": actual_mode, "objective": request.objective,
        "ports_optimized": [p.get('port_index') for p in enabled_ports],
        "total_time_s": round(elapsed, 2),
        "solutions_count": len(session.candidate_solutions),
        "solutions": solution_dicts,
        "best_solution": solution_dicts[0] if solution_dicts else None,
        "system_power_balance": solution_dicts[0].get('system_power_balance') if solution_dicts else None,
        "power_balance_chart": solution_dicts[0].get('power_balance_chart') if solution_dicts else [],
        # Clear naming: avg of enabled ports only
        "enabled_port_avg_efficiency": best.avg_total_efficiency if best else 0,
        # Power balance captures all ports + all losses (reflected, coupled, component, antenna)
        "system_efficiency_pct": pb_sys_eff * 100,
        "best_avg_efficiency": best.avg_total_efficiency if best else 0,
        "best_min_efficiency": best.min_total_efficiency if best else 0,
        "best_score": best.system_score if best else 0,
        "efficiency_basis": best.efficiency_basis if best else "",
        "component_library_filter": library_filter,
        "search_plan": build_search_plan(_model_dump(request)),
    }
    
    # Attach debug info when requested
    if request.debug:
        debug_info = getattr(session, 'last_debug_info', {})
        if debug_info:
            result["debug"] = debug_info
    
    return result


_tuning_jobs = {}
_tuning_jobs_lock = threading.RLock()


def _public_tuning_job(job: dict) -> dict:
    return {
        key: value
        for key, value in job.items()
        if key not in {"cancel_event", "thread"}
    }


def _execute_tuning_job(job_id: str, operation) -> None:
    def update_progress(progress: dict) -> None:
        with _tuning_jobs_lock:
            job = _tuning_jobs[job_id]
            elapsed = max(0.0, time.time() - job.get("started_at", time.time()))
            plan = job.get("search_plan") or {}
            budget = float(plan.get("budget_seconds") or 0.0)
            job["progress"] = {
                **dict(progress),
                "elapsed_seconds": round(elapsed, 2),
                "budget_seconds": budget,
                "budget_fraction": (
                    min(1.0, elapsed / budget) if budget > 0 else None
                ),
                "search_plan": plan,
            }
            job["updated_at"] = time.time()

    with _tuning_jobs_lock:
        job = _tuning_jobs[job_id]
        job["status"] = "running"
        job["started_at"] = time.time()
        job["updated_at"] = time.time()
        cancel_event = job["cancel_event"]
    try:
        result = operation(update_progress, cancel_event.is_set)
        with _tuning_jobs_lock:
            job = _tuning_jobs[job_id]
            if cancel_event.is_set():
                job["status"] = "cancelled"
                # Measured optimizers deliberately return a power-balanced
                # baseline plus every exact candidate completed before the
                # cooperative cancellation point. Preserve that useful work.
                job["result"] = (
                    result if int(result.get("solutions_count") or 0) > 0 else None
                )
                previous = job.get("progress") or {}
                job["progress"] = {
                    **previous,
                    "stage": "cancelled",
                    "message": (
                        "Cancelled; completed partial candidates are available"
                        if job["result"] is not None else "Cancelled"
                    ),
                }
            else:
                job["status"] = "completed"
                job["result"] = result
                previous = job.get("progress") or {}
                total = int(previous.get("total") or previous.get("current") or 1)
                job["progress"] = {
                    **previous,
                    "stage": "complete",
                    "current": total,
                    "total": total,
                    "message": "Optimization complete",
                }
            job["updated_at"] = time.time()
    except OptimizationCancelled:
        with _tuning_jobs_lock:
            job = _tuning_jobs[job_id]
            job["status"] = "cancelled"
            job["result"] = None
            job["progress"] = {
                **(job.get("progress") or {}),
                "stage": "cancelled",
                "message": "Cancelled before a partial candidate was available",
            }
            job["updated_at"] = time.time()
    except Exception as exc:
        with _tuning_jobs_lock:
            job = _tuning_jobs[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            job["updated_at"] = time.time()


def _run_tuning_job(job_id: str, request_payload: dict) -> None:
    def operation(progress_callback, cancel_check):
        request = TuningOptimizeRequest(**request_payload)
        return asyncio.run(_tuning_optimize_impl(
            request,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        ))

    _execute_tuning_job(job_id, operation)


def _run_tuning_continue_job(job_id: str, request_payload: dict) -> None:
    def operation(progress_callback, cancel_check):
        request = TuningContinueRequest(**request_payload)
        return asyncio.run(_tuning_continue_impl(
            request,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        ))

    _execute_tuning_job(job_id, operation)


def _run_multi_scenario_job(job_id: str, request_payload: dict) -> None:
    def operation(progress_callback, cancel_check):
        request = MultiScenarioOptimizeRequest(**request_payload)
        return _multi_scenario_optimize_impl(
            request, progress_callback=progress_callback, cancel_check=cancel_check,
        )

    _execute_tuning_job(job_id, operation)


def _manual_refine_search_plan(request_payload: dict) -> dict:
    components = request_payload.get("components") or []
    variable_count = sum(
        1 if str(component.get("comp_type", "")).lower() in {"inductor", "capacitor", "resistor"}
        and bool(component.get("use_ideal", True))
        else 2 if str(component.get("comp_type", "")).lower() in {"transmission_line", "open_stub", "short_stub"}
        else 0
        for component in components if isinstance(component, dict)
    )
    passes = int(request_payload.get("max_passes") or 4)
    probe_positions = 1 if not components else 2
    evaluations = 2 + 2 * variable_count * (passes + 1) + 4 * 9 * probe_positions + 4 * 5 + 4 * 2
    return {
        "profile": "manual_fixed_topology",
        "label": "Fixed-topology local refinement",
        "budget_seconds": max(2.0, min(60.0, evaluations * 0.12)),
        "estimated_evaluations": evaluations,
        "variable_count": variable_count,
        "passes": passes,
    }


def _run_manual_refine_job(job_id: str, request_payload: dict) -> None:
    def operation(progress_callback, cancel_check):
        request = ManualRefineRequest(**request_payload)
        if state.loaded_snp is None:
            raise ValueError("No SNP loaded")
        if state.loaded_snp_filename and request.snp_filename != state.loaded_snp_filename:
            raise ValueError("Manual-refinement DUT changed before the job started")
        dut = state.loaded_snp
        network_digest = touchstone_network_sha256(dut)
        if request.expected_network_sha256 and request.expected_network_sha256 != network_digest:
            raise ValueError("Manual-refinement DUT revision changed before the job started")
        needs_library = any(
            isinstance(component, dict)
            and not bool(component.get("use_ideal", True))
            and str(component.get("comp_type", "")).lower() in {"inductor", "capacitor"}
            for component in request.components
        )
        if needs_library and _get_library() is None:
            raise ValueError("Component library not loaded")
        result = optimize_manual_network_physical(
            dut, _get_library(),
            target_frequency_hz=request.target_frequency_hz,
            input_port=request.input_port,
            port_states=[_model_dump(item) for item in request.port_states],
            components=request.components,
            bands_mhz=request.bands_mhz,
            target_return_loss_db=request.target_return_loss_db,
            objective=request.objective,
            max_passes=request.max_passes,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        result["dut_identity"] = {
            "filename": state.loaded_snp_filename or request.snp_filename,
            "network_sha256": network_digest,
        }
        result["result"]["dut_identity"] = dict(result["dut_identity"])
        return result

    _execute_tuning_job(job_id, operation)


def _manual_yield_plan(request_payload: dict) -> dict:
    samples = int(request_payload.get("samples") or 200)
    return {
        "profile": "manual_monte_carlo_yield",
        "label": "Manual-network tolerance yield",
        "budget_seconds": max(2.0, min(120.0, samples * 0.02)),
        "estimated_evaluations": samples,
        "samples": samples,
    }


def _run_manual_yield_job(job_id: str, request_payload: dict) -> None:
    def operation(progress_callback, cancel_check):
        request = ManualYieldRequest(**request_payload)
        if state.loaded_snp is None:
            raise ValueError("No SNP loaded")
        if state.loaded_snp_filename and request.snp_filename != state.loaded_snp_filename:
            raise ValueError("Manual-yield DUT changed before the job started")
        network_digest = touchstone_network_sha256(state.loaded_snp)
        if request.expected_network_sha256 and request.expected_network_sha256 != network_digest:
            raise ValueError("Manual-yield DUT revision changed before the job started")
        result = run_manual_yield_analysis_physical(
            state.loaded_snp, _get_library(),
            input_port=request.input_port,
            port_states=[_model_dump(item) for item in request.port_states],
            components=request.components,
            bands_mhz=request.bands_mhz,
            target_return_loss_db=request.target_return_loss_db,
            samples=request.samples,
            seed=request.seed,
            distribution=request.distribution,
            confidence_level=request.confidence_level,
            default_tolerance_pct=request.default_tolerance_pct,
            batch_correlation=request.batch_correlation,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        result["dut_identity"] = {
            "filename": state.loaded_snp_filename or request.snp_filename,
            "network_sha256": network_digest,
        }
        return result

    _execute_tuning_job(job_id, operation)


def _start_tuning_job_thread(
    target, request_payload: dict, name_prefix: str,
    plan_builder=build_search_plan, job_type: str = "tuning",
) -> dict:
    """Create one bounded-history background job for optimization or continuation."""
    with _tuning_jobs_lock:
        active = [
            job for job in _tuning_jobs.values()
            if job["status"] in {"queued", "running", "cancelling"}
        ]
        if active:
            raise HTTPException(409, "A tuning job is already running")
        job_id = uuid.uuid4().hex
        plan_request = request_payload
        if "additional_seconds" in request_payload:
            plan_request = dict(get_session().last_tuning_request or {})
            plan_request["search_quality"] = "custom"
            plan_request["timeout_seconds"] = (
                float(plan_request.get("timeout_seconds") or 0.0)
                + float(request_payload.get("additional_seconds") or 0.0)
            )
        search_plan = plan_builder(plan_request)
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "progress": {
                "stage": "queued", "current": 0, "total": 0,
                "message": f"Queued · {search_plan['label']}",
                "elapsed_seconds": 0.0,
                "budget_seconds": search_plan["budget_seconds"],
                "budget_fraction": 0.0,
                "search_plan": search_plan,
            },
            "search_plan": search_plan,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "cancel_event": threading.Event(),
            "thread": None,
        }
        _tuning_jobs[job_id] = job
        finished = sorted(
            (item for item in _tuning_jobs.values() if item["status"] in {"completed", "cancelled", "failed"}),
            key=lambda item: item["updated_at"],
        )
        for old in finished[:-20]:
            _tuning_jobs.pop(old["job_id"], None)
        thread = threading.Thread(
            target=target,
            args=(job_id, request_payload),
            name=f"{name_prefix}-{job_id[:8]}",
            daemon=True,
        )
        job["thread"] = thread
        thread.start()
        return _public_tuning_job(job)


@app.post("/api/tuning/jobs", status_code=202)
async def start_tuning_job(request: TuningOptimizeRequest):
    """Start one cancellable tuning task without blocking the API event loop."""
    return _start_tuning_job_thread(
        _run_tuning_job, _model_dump(request), "rfmatch-tuning",
    )


@app.post("/api/tuning/continue/jobs", status_code=202)
async def start_tuning_continue_job(request: TuningContinueRequest):
    """Continue a live search in the same cancellable background-job system."""
    session = get_session()
    if not session.last_tuning_request or not session.candidate_solutions:
        raise HTTPException(409, "No live tuning result is available to continue")
    if session.restoration_mode != "live":
        raise HTTPException(409, "Snapshot results must be recomputed before they can be continued")
    return _start_tuning_job_thread(
        _run_tuning_continue_job, _model_dump(request), "rfmatch-continue",
    )


@app.post("/api/multi-scenario/jobs", status_code=202)
async def start_multi_scenario_job(request: MultiScenarioOptimizeRequest):
    """Start one cancellable shared-network optimization task."""
    return _start_tuning_job_thread(
        _run_multi_scenario_job, _model_dump(request), "rfmatch-multi-scenario",
        plan_builder=build_multi_scenario_search_plan, job_type="multi_scenario",
    )


@app.post("/api/manual-refine/jobs", status_code=202)
async def start_manual_refine_job(request: ManualRefineRequest):
    """Refine continuous values without changing the user's manual topology."""
    _ensure_snp_loaded()
    if state.loaded_snp_filename and request.snp_filename != state.loaded_snp_filename:
        raise HTTPException(409, "Manual-refinement DUT changed; reload the active DUT")
    network_digest = touchstone_network_sha256(state.loaded_snp)
    if request.expected_network_sha256 and request.expected_network_sha256 != network_digest:
        raise HTTPException(409, "Manual-refinement DUT revision changed; recompute first")
    needs_library = any(
        isinstance(component, dict)
        and not bool(component.get("use_ideal", True))
        and str(component.get("comp_type", "")).lower() in {"inductor", "capacitor"}
        for component in request.components
    )
    if needs_library:
        _ensure_library()
    return _start_tuning_job_thread(
        _run_manual_refine_job, _model_dump(request), "rfmatch-manual-refine",
        plan_builder=_manual_refine_search_plan, job_type="manual_refine",
    )


@app.post("/api/manual-yield/jobs", status_code=202)
async def start_manual_yield_job(request: ManualYieldRequest):
    """Run cancellable Monte Carlo yield on the exact manual network."""
    _ensure_snp_loaded()
    if state.loaded_snp_filename and request.snp_filename != state.loaded_snp_filename:
        raise HTTPException(409, "Manual-yield DUT changed; reload the active DUT")
    network_digest = touchstone_network_sha256(state.loaded_snp)
    if request.expected_network_sha256 and request.expected_network_sha256 != network_digest:
        raise HTTPException(409, "Manual-yield DUT revision changed; recompute first")
    needs_library = any(
        isinstance(component, dict)
        and not bool(component.get("use_ideal", True))
        and str(component.get("comp_type", "")).lower() in {"inductor", "capacitor"}
        for component in request.components
    )
    if needs_library:
        _ensure_library()
    return _start_tuning_job_thread(
        _run_manual_yield_job, _model_dump(request), "rfmatch-manual-yield",
        plan_builder=_manual_yield_plan, job_type="manual_yield",
    )


@app.get("/api/tuning/jobs/{job_id}")
async def tuning_job_status(job_id: str):
    with _tuning_jobs_lock:
        job = _tuning_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Tuning job not found")
        return _public_tuning_job(job)


@app.post("/api/tuning/jobs/{job_id}/cancel")
async def cancel_tuning_job(job_id: str):
    with _tuning_jobs_lock:
        job = _tuning_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Tuning job not found")
        if job["status"] in {"completed", "cancelled", "failed"}:
            return _public_tuning_job(job)
        job["cancel_event"].set()
        job["status"] = "cancelling"
        job["updated_at"] = time.time()
        return _public_tuning_job(job)


@app.get("/api/tuning/sweep")
async def tuning_sweep(
    port_index: int = 0,
    start_hz: Optional[float] = None,
    stop_hz: Optional[float] = None,
    num_points: int = 200,
    solution_index: int = 0,
    use_snp_points: bool = True,
):
    """Frequency sweep for a selected tuning solution (uses TuningSession)."""
    _ensure_snp_loaded()
    session = get_session()
    
    if not session.candidate_solutions:
        raise HTTPException(400, "No tuning results. Run /api/tuning/optimize first.")
    
    if not session.select_solution(solution_index):
        raise HTTPException(404, f"Solution index {solution_index} out of range")
    
    result = session.get_selected_result()
    if not result or not result.per_port:
        raise HTTPException(400, "Selected solution has no per-port data")
    
    pp = result.per_port.get(port_index)
    if not pp:
        raise HTTPException(404, f"Port {port_index} not in solution")

    freqs = state.loaded_snp.frequencies or []
    sweep_start = start_hz if start_hz is not None else (min(freqs) if freqs else 1.0e6)
    sweep_stop = stop_hz if stop_hz is not None else (max(freqs) if freqs else sweep_start * 2)
    choices = result.component_choices or {}

    if result.mode == "transmission_line" and result.yield_context:
        try:
            sweep = _service_line_sweep(
                state.loaded_snp,
                result,
                start_hz=sweep_start,
                stop_hz=sweep_stop,
                num_points=num_points,
                use_snp_points=use_snp_points,
                global_efficiency=state.global_efficiency_data,
                per_port_efficiency=state.per_port_efficiency_data or None,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise HTTPException(400, str(exc)) from exc
        sweep["source"] = "full_snp_recompute" if use_snp_points else "dense_recompute"
        sweep["solution_index"] = solution_index
        return sweep

    if choices:
        sweep = _service_sweep(
            state.loaded_snp,
            choices,
            port_index=port_index,
            start_hz=sweep_start,
            stop_hz=sweep_stop,
            num_points=num_points,
            include_efficiency=True,
            use_snp_points=use_snp_points,
            global_efficiency=state.global_efficiency_data,
            per_port_efficiency=state.per_port_efficiency_data or None,
        )
        sweep["source"] = "full_snp_recompute" if use_snp_points else "dense_recompute"
        sweep["solution_index"] = solution_index
        return sweep

    return {
        "frequencies": pp.band_freqs_hz if pp.band_freqs_hz else [],
        "s11_db": pp.band_s11_db if pp.band_s11_db else [],
        "s11_magnitude": [10 ** (d / -20) if d else 0 for d in pp.band_s11_db] if pp.band_s11_db else [],
        "port_index": port_index,
        "source": "band_points_fallback",
        "solution_index": solution_index,
        "efficiency": {
            "total_pct": [e * 100 for e in pp.band_total_eff] if pp.band_total_eff else [],
        } if pp.band_total_eff else None,
    }


@app.get("/api/tuning/power-balance")
async def tuning_power_balance(frequency_hz: float = 2.45e9):
    """Power balance for the currently selected tuning solution."""
    _ensure_snp_loaded()
    session = get_session()
    result = session.get_selected_result()
    
    if not result or not result.system_power_balance:
        from engine.power_balance import compute_power_balance as pb_fn
        from engine.power_balance import power_balance_to_chart_data as pb_chart
        S = state.loaded_snp.get_s_matrix_interpolated(frequency_hz)
        pb = pb_fn(S, component_loss_total=0.0, matched_ports=[])
        return {"frequency_hz": frequency_hz, "mode": "raw_dut",
                "power_balance": pb.to_dict(), "chart_data": pb_chart(pb)}
    
    return {
        "frequency_hz": frequency_hz, "mode": "matched",
        "power_balance": result.system_power_balance,
        "chart_data": result.power_balance_chart,
        "system_efficiency_pct": result.avg_total_efficiency * 100,
    }


@app.post("/api/tuning/select")
async def tuning_select_solution(solution_index: int = 0):
    """Select a specific candidate solution by index."""
    session = get_session()
    if not session.candidate_solutions:
        raise HTTPException(400, "No tuning results available")
    if not session.select_solution(solution_index):
        raise HTTPException(404, f"Solution index {solution_index} out of range (0-{len(session.candidate_solutions)-1})")
    result = session.get_selected_result()
    return {"status": "ok", "selected_index": solution_index,
            "solution": result.to_dict() if result else None}


@app.get("/api/tuning/status")
async def tuning_status():
    """Get current tuning session status."""
    session = get_session()
    result = session.get_selected_result()
    return {
        "has_results": len(session.candidate_solutions) > 0,
        "num_solutions": len(session.candidate_solutions),
        "selected_index": session.selected_index,
        "dut_loaded": session.dut is not None,
        "dut_filename": session.dut_filename or "",
        "selected_solution": result.to_dict() if result else None,
        "restoration_mode": session.restoration_mode,
        "can_recompute_exact_sweep": session.restoration_mode == "live",
    }


@app.post("/api/tuning/yield")
async def tuning_yield_analysis(request: TuningYieldRequest):
    """Run deterministic measured-component Monte Carlo analysis and rank candidates."""
    _ensure_snp_loaded()
    _ensure_library()
    session = get_session()
    if not session.candidate_solutions:
        raise HTTPException(400, "No tuning results available")
    try:
        return await asyncio.to_thread(
            run_tuning_yield_analysis,
            state.loaded_snp,
            _get_library(),
            session.candidate_solutions,
            session.last_tuning_request,
            solution_indices=request.solution_indices,
            samples=request.samples,
            seed=request.seed,
            distribution=request.distribution,
            confidence_level=request.confidence_level,
            minimum_total_efficiency=request.minimum_total_efficiency,
            minimum_average_total_efficiency=request.minimum_average_total_efficiency,
            minimum_return_loss_db=request.minimum_return_loss_db,
            default_tolerance_pct=request.default_tolerance_pct,
            batch_correlation=request.batch_correlation,
            reference_temperature_c=request.reference_temperature_c,
            temperature_min_c=request.temperature_min_c,
            temperature_max_c=request.temperature_max_c,
            inductor_tempco_ppm_per_c=request.inductor_tempco_ppm_per_c,
            capacitor_tempco_ppm_per_c=request.capacitor_tempco_ppm_per_c,
            inductor_bias_pct=request.inductor_bias_pct,
            capacitor_bias_pct=request.capacitor_bias_pct,
            global_efficiency=state.global_efficiency_data,
            per_port_efficiency=state.per_port_efficiency_data,
        )
    except (OSError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, f"Unable to analyze candidate yield: {exc}") from exc


@app.get("/api/projects")
async def list_projects():
    """List valid and corrupt local project snapshots without hiding errors."""
    return {"projects": project_store.list()}


def _project_relinks(document):
    relinks = (document.get("extensions", {}).get("file_relinks") or {})
    return relinks if isinstance(relinks, dict) else {}


def _project_linked_filename(document, key, original_filename):
    entry = _project_relinks(document).get(key) or {}
    linked = entry.get("linked_filename") if isinstance(entry, dict) else None
    return linked if isinstance(linked, str) and linked else original_filename


def _locate_project_file(document, key, original_filename, expected_hash):
    """Prefer a recorded relink, but safely fall back to the original path."""
    linked = _project_linked_filename(document, key, original_filename)
    filenames = list(dict.fromkeys([linked, original_filename]))
    first_existing = None
    for filename in filenames:
        try:
            path = Path(_safe_data_path(filename))
            actual_hash = sha256_file(path)
        except (HTTPException, OSError):
            continue
        candidate = (filename, path, actual_hash, actual_hash == expected_hash)
        if candidate[3]:
            return candidate
        if first_existing is None:
            first_existing = candidate
    return first_existing or (linked, None, None, False)


def _replace_exact_filenames(value, replacements):
    """Apply verified runtime filename aliases without mutating the signed snapshot."""
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_exact_filenames(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_exact_filenames(item, replacements)
            for key, item in value.items()
        }
    return value


@app.post("/api/projects/import")
async def import_project(request: ProjectImportRequest):
    """Verify and import a signed project snapshot using an explicit conflict policy."""
    try:
        result = project_store.import_document(
            request.document,
            conflict_policy=request.conflict_policy,
        )
    except (OSError, ProjectValidationError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Unable to import project: {exc}") from exc
    document = result["document"]
    return {
        "status": result["status"],
        "project_id": document["project_id"],
        "name": document["name"],
        "schema_version": document["schema_version"],
        "updated_at": document["updated_at"],
        "integrity_sha256": document["integrity"]["digest"],
        "solutions_count": len(document["results"].get("candidates", [])),
        "source_project_id": (
            (document.get("extensions", {}).get("import") or {}).get(
                "source_project_id"
            )
        ),
    }


@app.post("/api/projects/relink")
async def relink_project_files(request: ProjectRelinkRequest):
    """Find byte-identical project inputs in the active data directory."""
    try:
        document = project_store.load(request.project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ProjectValidationError) as exc:
        raise HTTPException(400, f"Unable to inspect project: {exc}") from exc

    dependencies = document["configuration"].get("input_dependencies", [])
    targets = [{
        "key": "input",
        "role": "dut_touchstone",
        "original_filename": str(document["input"].get("filename", "")),
        "expected_sha256": document["input"].get("sha256"),
        "size_bytes": document["input"].get("size_bytes"),
    }]
    targets.extend({
        "key": f"dependency:{index}",
        "role": dependency.get("role", "dependency"),
        "original_filename": str(dependency.get("filename", "")),
        "expected_sha256": dependency.get("sha256"),
        "size_bytes": dependency.get("size_bytes"),
    } for index, dependency in enumerate(dependencies))

    expected_hashes = {
        item["expected_sha256"] for item in targets
        if isinstance(item["expected_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", item["expected_sha256"])
    }
    candidate_sizes = {
        int(item["size_bytes"]) for item in targets
        if isinstance(item.get("size_bytes"), int)
    }
    use_size_filter = bool(targets) and all(
        isinstance(item.get("size_bytes"), int) for item in targets
    )
    matches_by_hash = {digest: [] for digest in expected_hashes}
    if state.snp_dir and os.path.isdir(state.snp_dir):
        for root, _, filenames in os.walk(state.snp_dir):
            for filename in filenames:
                if not re.search(r"\.s\d+p$", filename, flags=re.IGNORECASE):
                    continue
                path = Path(root) / filename
                try:
                    if use_size_filter and path.stat().st_size not in candidate_sizes:
                        continue
                    relative = os.path.relpath(path, state.snp_dir)
                    safe_path = _safe_data_path(relative)
                    digest = sha256_file(safe_path)
                except (OSError, HTTPException):
                    continue
                if digest in matches_by_hash:
                    matches_by_hash[digest].append(relative)

    existing_relinks = deepcopy(_project_relinks(document))
    updated_relinks = deepcopy(existing_relinks)
    resolved_targets = []
    for target in targets:
        digest = target["expected_sha256"]
        matches = sorted(
            set(matches_by_hash.get(digest, [])),
            key=lambda value: (len(value), value.lower()),
        )
        current = _project_linked_filename(
            document, target["key"], target["original_filename"]
        )
        selected = current if current in matches else (matches[0] if matches else None)
        if request.apply_matches and selected:
            if selected == target["original_filename"]:
                updated_relinks.pop(target["key"], None)
            else:
                updated_relinks[target["key"]] = {
                    "role": target["role"],
                    "original_filename": target["original_filename"],
                    "linked_filename": selected,
                    "expected_sha256": digest,
                    "linked_at": utc_now(),
                }
        resolved_targets.append({
            **target,
            "matched": selected is not None,
            "linked_filename": selected,
            "candidate_filenames": matches,
        })

    changed = request.apply_matches and updated_relinks != existing_relinks
    if changed:
        updated = deepcopy(document)
        updated.pop("integrity", None)
        updated["updated_at"] = utc_now()
        updated.setdefault("extensions", {})["file_relinks"] = updated_relinks
        try:
            document = project_store.replace_document(
                sign_document(updated), expected_project_id=request.project_id
            )
        except (OSError, ProjectValidationError) as exc:
            raise HTTPException(400, f"Unable to save project relinks: {exc}") from exc

    return {
        "status": "ready" if all(item["matched"] for item in resolved_targets) else "incomplete",
        "project_id": request.project_id,
        "data_directory": state.snp_dir,
        "changed": changed,
        "matched_count": sum(item["matched"] for item in resolved_targets),
        "total_count": len(resolved_targets),
        "targets": resolved_targets,
    }


@app.get("/api/projects/{project_id}/snapshot.json")
async def project_snapshot_json(project_id: str):
    """Download the signed, versioned project document for reproducible handoff."""
    try:
        document = project_store.load(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ProjectValidationError) as exc:
        raise HTTPException(400, f"Unable to export project snapshot: {exc}") from exc
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", document["name"]).strip("-.")
    filename = f"{safe_name or project_id}.rfmatch.json"
    content = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/projects/{project_id}/report")
async def project_html_report(project_id: str, download: bool = True):
    """Download a self-contained traceability report from a validated snapshot."""
    try:
        document = project_store.load(project_id)
        content = render_project_report(document)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ProjectValidationError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Unable to generate report: {exc}") from exc
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", document["name"]).strip("-.")
    filename = f"{safe_name or project_id}-rfmatch-report.html"
    headers = (
        {"Content-Disposition": f'attachment; filename="{filename}"'}
        if download else {}
    )
    return Response(content=content, media_type="text/html; charset=utf-8", headers=headers)


@app.get("/api/projects/{project_id}/report.pdf")
async def project_pdf_report(project_id: str, download: bool = True):
    """Download a native, paginated PDF from a validated project snapshot."""
    try:
        document = project_store.load(project_id)
        content = render_project_pdf(document)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ProjectValidationError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Unable to generate PDF report: {exc}") from exc
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", document["name"]).strip("-.")
    filename = f"{safe_name or project_id}-rfmatch-report.pdf"
    disposition = "attachment" if download else "inline"
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


@app.get("/api/projects/{project_id}/bom.csv")
async def project_bom_csv(project_id: str):
    """Download the selected solution as an aggregated procurement BOM."""
    try:
        document = project_store.load(project_id)
        content = render_project_bom_csv(document)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ProjectValidationError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Unable to generate BOM: {exc}") from exc
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", document["name"]).strip("-.")
    filename = f"{safe_name or project_id}-bom.csv"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/projects/save")
async def save_project(request: ProjectSaveRequest):
    """Atomically save the loaded DUT, reproducible settings, and result snapshot."""
    _ensure_snp_loaded()
    if not state.loaded_snp_filename:
        raise HTTPException(400, "Loaded SNP has no source filename")
    input_path = Path(_safe_data_path(state.loaded_snp_filename))
    session = get_session()
    efficiency = {
        "global": getattr(state.global_efficiency_data, "source", None),
        "per_port": {
            str(port): getattr(data, "source", None)
            for port, data in state.per_port_efficiency_data.items()
        },
    }
    tuning_request = session.last_tuning_request or {}
    input_dependencies = []
    for layout in (
        (tuning_request.get("transmission_line") or {}).get("layout_blocks") or []
    ):
        dependency_filename = str(layout.get("filename", ""))
        if not dependency_filename:
            continue
        dependency_path = Path(_safe_data_path(dependency_filename))
        input_dependencies.append({
            "role": "layout_s2p",
            "filename": dependency_filename,
            "sha256": sha256_file(dependency_path),
            "size_bytes": dependency_path.stat().st_size,
            "location": layout.get("location", "connector_side"),
            "reverse_ports": bool(layout.get("reverse_ports", False)),
            "reference_impedance_mode": layout.get("reference_impedance_mode", "native"),
        })
        for side in ("left", "right"):
            fixture_filename = layout.get(f"{side}_fixture_filename")
            if not fixture_filename:
                continue
            fixture_path = Path(_safe_data_path(str(fixture_filename)))
            input_dependencies.append({
                "role": f"{side}_deembedding_fixture_s2p",
                "filename": str(fixture_filename),
                "sha256": sha256_file(fixture_path),
                "size_bytes": fixture_path.stat().st_size,
                "location": layout.get("location", "connector_side"),
                "reverse_ports": bool(layout.get(f"{side}_fixture_reverse_ports", False)),
                "reference_impedance_mode": "layout_effective",
            })
    try:
        document = project_store.save(
            name=request.name,
            project_id=request.project_id,
            extensions=(
                {"manual_workspace": _model_dump(request.manual_workspace)}
                if request.manual_workspace is not None else {}
            ),
            software={
                "application": app.title,
                "api_version": app.version,
                "rfmatch_core_version": core_version,
                "python_version": platform.python_version(),
            },
            input_snapshot={
                "filename": state.loaded_snp_filename,
                "sha256": sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
                "num_ports": state.loaded_snp.num_ports,
                "frequency_count": len(state.loaded_snp.frequencies),
                "frequency_min_hz": min(state.loaded_snp.frequencies),
                "frequency_max_hz": max(state.loaded_snp.frequencies),
                "reference_resistance_ohm": state.loaded_snp.reference_resistance,
                "provenance": deepcopy(state.loaded_snp_provenance),
            },
            configuration={
                "tuning_request": tuning_request,
                "component_library": _component_library_snapshot(),
                "efficiency_sources": efficiency,
                "input_dependencies": input_dependencies,
            },
            results={
                "selected_index": session.selected_index,
                "candidates": [item.to_dict() for item in session.candidate_solutions],
                "snapshot_only": True,
            },
        )
    except (OSError, ProjectValidationError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Unable to save project: {exc}") from exc
    return {
        "status": "ok",
        "project_id": document["project_id"],
        "name": document["name"],
        "schema_version": document["schema_version"],
        "updated_at": document["updated_at"],
        "input_sha256": document["input"]["sha256"],
        "solutions_count": len(document["results"]["candidates"]),
        "manual_variants_count": len(
            ((document.get("extensions") or {}).get("manual_workspace") or {}).get("variants", [])
        ),
    }


@app.post("/api/projects/load")
async def load_project(request: ProjectLoadRequest):
    """Restore a safe display snapshot and verify its referenced DUT before use."""
    try:
        document = project_store.load(request.project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ProjectValidationError) as exc:
        raise HTTPException(400, f"Unable to load project: {exc}") from exc

    try:
        restored = [
            TuningResult.from_dict(item)
            for item in document["results"].get("candidates", [])
        ]
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(400, f"Project snapshot data is invalid: {exc}") from exc

    snapshot_filename = document["input"].get("filename", "")
    expected_hash = document["input"].get("sha256")
    filename, input_path, actual_hash, input_matches = _locate_project_file(
        document, "input", snapshot_filename, expected_hash
    )
    if request.verify_input and input_path is None:
        raise HTTPException(
            409,
            "Project input is unavailable under the configured SNP directory: "
            + snapshot_filename,
        )
    if request.verify_input and not input_matches:
        raise HTTPException(409, "Project input SHA-256 does not match the saved snapshot")

    dependency_status = []
    for index, dependency in enumerate(
        document["configuration"].get("input_dependencies", [])
    ):
        original_dependency_filename = str(dependency.get("filename", ""))
        dependency_filename, _, dependency_hash, dependency_matches = _locate_project_file(
            document,
            f"dependency:{index}",
            original_dependency_filename,
            dependency.get("sha256"),
        )
        dependency_status.append({
            "role": dependency.get("role"),
            "filename": dependency_filename,
            "original_filename": original_dependency_filename,
            "expected_sha256": dependency.get("sha256"),
            "actual_sha256": dependency_hash,
            "matches": dependency_matches,
        })
        if request.verify_input and not dependency_matches:
            raise HTTPException(
                409, "Project dependency is unavailable or changed: " + dependency_filename,
            )

    dut = None
    if input_matches and input_path is not None:
        try:
            content = input_path.read_text(encoding="utf-8", errors="replace")
            dut = parse_touchstone(content, filename=filename)
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"Project input data is invalid: {exc}") from exc

    state.loaded_snp = dut
    state.loaded_snp_filename = filename if dut is not None else ""
    state.loaded_snp_provenance = deepcopy(document["input"].get("provenance")) if dut is not None else None
    state.last_solutions = []
    state.last_joint_results = None
    state.last_multi_scenario_results = None
    state.clear_efficiency()
    reset_session()
    session = get_session()
    session.dut = dut
    session.dut_filename = filename
    session.library = _get_library()
    session.candidate_solutions = restored
    selected_index = int(document["results"].get("selected_index", 0))
    session.selected_index = selected_index if 0 <= selected_index < len(restored) else 0
    filename_replacements = {
        item["original_filename"]: item["filename"]
        for item in dependency_status if item["matches"]
    }
    session.last_tuning_request = _replace_exact_filenames(
        document["configuration"].get("tuning_request", {}),
        filename_replacements,
    )
    session.restoration_mode = "snapshot"
    component_library_status = _component_library_verification(document)

    return {
        "status": "ok",
        "project_id": document["project_id"],
        "name": document["name"],
        "schema_version": document["schema_version"],
        "migrated_from_version": (
            document["migration_history"][0].get("from_version")
            if document.get("migration_history") else None
        ),
        "input_filename": filename,
        "snapshot_input_filename": snapshot_filename,
        "input_sha256": expected_hash,
        "num_ports": dut.num_ports if dut is not None else document["input"].get("num_ports", 0),
        "freq_count": len(dut.frequencies) if dut is not None else document["input"].get("frequency_count", 0),
        "freq_min_hz": min(dut.frequencies) if dut is not None else document["input"].get("frequency_min_hz"),
        "freq_max_hz": max(dut.frequencies) if dut is not None else document["input"].get("frequency_max_hz"),
        "input_verified": input_matches,
        "exact_recompute_available": (
            input_matches
            and all(item["matches"] for item in dependency_status)
            and component_library_status["matches"]
        ),
        "component_library_status": component_library_status,
        "dependency_status": dependency_status,
        "solutions_count": len(restored),
        "solutions": [item.to_dict() for item in restored],
        "selected_index": session.selected_index,
        "restoration_mode": "snapshot",
        "warning": (
            "Saved curves and metrics are restored. Exact sweep recomputation requires "
            "rerunning the saved tuning_request with the verified component library."
        ),
        "tuning_request": session.last_tuning_request,
        "manual_workspace": deepcopy(
            (document.get("extensions") or {}).get("manual_workspace")
        ),
        "selected_solution": (
            session.get_selected_result().to_dict()
            if session.get_selected_result() is not None
            else None
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TUNE API — Optenni-style antenna tuning endpoints (legacy, delegate to service)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/tune/single", deprecated=True)
async def tune_single(request: TuneSingleRequest):
    """Deprecated protocol adapter over the authoritative tuning service."""
    _ensure_snp_loaded()
    required_component_types = _required_component_types_for_topologies(
        request.topology_filter, request.max_components
    )
    if required_component_types:
        _ensure_library()
        library, library_filter = _library_for_series(
            request.component_series,
            required_component_types=required_component_types,
        )
    else:
        library = _get_library()
        library_filter = {
            "mode": "not_required", "selected_series": request.component_series,
            "reason": "zero-component bare-DUT analysis",
            "inductors": 0, "capacitors": 0,
        }
    try:
        candidates = await asyncio.to_thread(
            run_tuning_single,
            dut=state.loaded_snp,
            library=library,
            port_index=request.port_index,
            bands_mhz=request.bands_mhz,
            band_weights=request.band_weights,
            port_weight=request.port_weight,
            max_components=request.max_components,
            objective=request.objective,
            within_band_average_weight=request.within_band_average_weight,
            across_band_average_weight=request.across_band_average_weight,
            generic_synthesis_loss=_model_dump(request.generic_synthesis_loss),
            beam_width=request.beam_width,
            timeout_seconds=request.timeout_seconds,
            num_band_points=request.num_band_points,
            global_efficiency=state.global_efficiency_data,
            per_port_efficiency=state.per_port_efficiency_data or None,
            component_series=request.component_series,
            topology_filter=request.topology_filter,
        )
    except (OSError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, f"Unable to tune port: {exc}") from exc

    ordered = [candidates[index] for index in sorted(candidates)]
    for candidate in ordered:
        candidate.search_diagnostics = candidate.search_diagnostics or {}
        candidate.search_diagnostics.setdefault(
            "component_library_filter", library_filter
        )
        candidate.search_diagnostics["api_entrypoint"] = "/api/tune/single"
        candidate.search_diagnostics["authoritative_solver"] = "run_tuning_single"

    session = get_session()
    session.dut = state.loaded_snp
    session.dut_filename = state.loaded_snp_filename
    session.library = library
    session.candidate_solutions = ordered
    session.selected_index = 0
    session.last_tuning_request = {
        "mode": "single",
        "ports": [{
            "port_index": request.port_index,
            "bands_mhz": request.bands_mhz,
            "band_weights": request.band_weights,
            "port_weight": request.port_weight,
            "max_components": request.max_components,
            "enabled": True,
        }],
        "objective": request.objective,
        "beam_width": request.beam_width,
        "timeout_seconds": request.timeout_seconds,
        "num_band_points": request.num_band_points,
        "component_series": request.component_series,
        "topology_filter": request.topology_filter,
    }
    state.last_solutions = []
    serialized = []
    for candidate in ordered[:20]:
        item = candidate.to_dict()
        item["efficiency_score"] = candidate.system_score
        item["avg_efficiency"] = candidate.avg_total_efficiency
        item["min_efficiency"] = candidate.min_total_efficiency
        first_port = candidate.per_port.get(request.port_index)
        item["band_efficiency_points"] = (
            list(first_port.band_total_eff) if first_port is not None else []
        )
        serialized.append(item)
    best = ordered[0] if ordered else None
    return {
        "status": "ok",
        "mode": "single_port_tune",
        "port_index": request.port_index,
        "objective": request.objective,
        "bands_mhz": request.bands_mhz,
        "solutions_count": len(serialized),
        "solutions": serialized,
        "best_avg_efficiency": best.avg_total_efficiency if best else 0,
        "best_min_efficiency": best.min_total_efficiency if best else 0,
        "best_score": best.system_score if best else 0,
        "deprecated_endpoint": True,
        "authoritative_endpoint": "/api/tuning/optimize",
        "authoritative_solver": "run_tuning_single",
        "component_library_filter": library_filter,
    }


@app.post("/api/tune/joint", deprecated=True)
async def tune_joint(request: TuneJointRequest):
    """
    Multi-port joint tuning with system-level efficiency scoring.

    Unlike the legacy /api/joint-optimize endpoint, this endpoint:
    1. Takes TuningPlan-style configuration
    2. Returns richer results with full efficiency chain
    3. Includes power balance breakdown
    4. Scores by system average total efficiency
    """
    _ensure_snp_loaded()
    _ensure_library()

    from engine.scoring import get_objective_preset, score_multi_port, estimate_total_component_loss
    from engine.power_balance import compute_power_balance, power_balance_to_chart_data
    from engine.topology import get_standard_topologies

    lib = _get_library()
    n_ports = state.loaded_snp.num_ports

    # Parse port configs from request
    matching_ports = [p for p in request.ports if p.get('enabled', True)]
    if len(matching_ports) < 1:
        raise HTTPException(400, "At least one port must be enabled")

    # Build PortMatchConfig for joint optimizer
    port_configs = []
    band_list_mhz = []
    for mp in matching_ports:
        bands = mp.get('bands_mhz', [[2400, 2500]])
        center_hz = (bands[0][0] + bands[0][1]) / 2.0 * 1e6
        port_configs.append(PortMatchConfig(
            port_index=mp.get('port_index', 0),
            max_components=mp.get('max_components', 2),
            target_frequency_hz=center_hz,
        ))
        band_list_mhz.extend(bands)

    joint_opt = JointMultiPortOptimizer(
        dut=state.loaded_snp,
        component_library=lib if not state.use_db else state.component_library,
        port_configs=port_configs,
        top_candidates_per_port=request.beam_width or 8,
        timeout_seconds=request.timeout_seconds or 120.0,
        min_avg_balance=0.5,
        optimization_mode=request.objective or 'balanced',
        radiation_efficiency=state.global_efficiency_data,
        per_port_efficiency=state.per_port_efficiency_data or None,
    )

    t_start = time.time()
    joint_solutions = joint_opt.optimize(
        bands_mhz=band_list_mhz if band_list_mhz else None,
        num_band_points=request.num_band_points or 5,
    )
    total_time_s = time.time() - t_start

    if not joint_solutions:
        return {
            "status": "ok",
            "mode": "joint_tune",
            "ports_optimized": [p.get('port_index') for p in matching_ports],
            "total_time_s": total_time_s,
            "solutions_count": 0,
            "warning": "No valid joint solutions found",
        }

    # Build rich results for top solutions
    result_solutions = []
    preset = get_objective_preset(request.objective)

    for js in joint_solutions[:20]:
        # Per-port detailed results
        per_port = {}
        for pi, pm in js.port_metrics.items():
            pb = js.power_balance.get(pi, {})
            per_port[str(pi)] = {
                "s11_magnitude": pm.get('s11_magnitude', 0),
                "s11_db": pm.get('s11_db', 0),
                "mismatch_efficiency": pm.get('mismatch_efficiency', 0),
                "coupling_loss": pm.get('coupling_loss', 0),
                "radiated_efficiency": pm.get('radiated_efficiency', 0),
                "total_efficiency": pm.get('total_efficiency', 0),
                "component_loss": pb.get('component_loss', 0),
                "power_balance": pb,
                "components": [
                    {
                        "part": c.get('part', ''),
                        "type": c.get('type', ''),
                        "value": c.get('value', ''),
                    }
                    for c in js.to_dict().get('components_summary', {}).get(str(pi), [])
                ],
            }

        # System-level metrics
        all_total_effs = [
            pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
            for pm in js.port_metrics.values()
        ]
        all_coupling = [
            pm.get('coupling_loss', 0) for pm in js.port_metrics.values()
        ]

        result_solutions.append({
            "balanced_score": js.balanced_score,
            "avg_system_efficiency": float(np.mean(all_total_effs)),
            "min_system_efficiency": float(np.min(all_total_effs)),
            "avg_total_efficiency": js.avg_total_efficiency,
            "min_total_efficiency": js.min_total_efficiency,
            "max_coupling_loss": js.max_coupling_loss,
            "avg_coupling_loss": float(np.mean(all_coupling)),
            "component_loss_total": js.component_loss_total,
            "component_count": sum(
                len(sol.component_choices) for sol in js.port_solutions.values()
            ),
            "per_port": per_port,
        })

    # Compute overall system power balance for best solution
    best = joint_solutions[0]
    pb_system = None
    if best.system_s_matrix is not None:
        matched_ports_list = list(best.port_solutions.keys())
        pb_system = compute_power_balance(
            best.system_s_matrix,
            component_loss_total=best.component_loss_total,
            matched_ports=matched_ports_list,
            n_matched_ports=len(matched_ports_list),
        )

    return {
        "status": "ok",
        "mode": "joint_tune",
        "objective": request.objective,
        "ports_optimized": [p.get('port_index') for p in matching_ports],
        "total_time_s": total_time_s,
        "solutions_count": len(joint_solutions),
        "solutions": result_solutions,
        "best_solution": result_solutions[0] if result_solutions else None,
        "system_power_balance": pb_system.to_dict() if pb_system else None,
        "power_balance_chart": power_balance_to_chart_data(pb_system) if pb_system else [],
    }


@app.get("/api/tune/sweep")
async def tune_sweep(
    port_index: int = 0,
    start_hz: float = 1e9,
    stop_hz: float = 3e9,
    num_points: int = 200,
    solution_index: int = 0,
    include_efficiency: bool = True,
    include_power_balance: bool = False,
):
    """
    Frequency sweep for a tuned solution.

    Works with both /api/tune/single and /api/tune/joint results.
    Returns S11 and efficiency curves across the sweep range.
    """
    _ensure_snp_loaded()

    from engine.scoring import estimate_total_component_loss
    from engine.power_balance import compute_power_balance, power_balance_to_chart_data

    # Check which result source to use
    port_matching = {}
    source = "unknown"

    # Try single-port tune results first (solutions in last_solutions)
    if state.last_solutions and solution_index < len(state.last_solutions):
        sol = state.last_solutions[solution_index]
        # Identify which port this solution is for
        comp_port = None
        for p, st in sol.port_states.items():
            if st == PortState.COMPONENT:
                comp_port = p
                break
        if comp_port is not None:
            port_matching[comp_port] = sol.component_choices
            source = "single"
    elif state.last_joint_results:
        best = state.last_joint_results[0]
        for pi, sol in best.port_solutions.items():
            port_matching[pi] = sol.component_choices
            source = "joint"

    if not port_matching:
        raise HTTPException(400, "No tuning results. Run /api/tune/single or /api/tune/joint first.")

    freqs = np.linspace(start_hz, stop_hz, num_points)
    N = state.loaded_snp.num_ports

    raw_db, raw_mag = [], []
    match_db, match_mag = [], []
    eff_accepted, eff_coupling, eff_comp_loss, eff_total = [], [], [], []

    for freq in freqs:
        S_base = state.loaded_snp.get_s_matrix_interpolated(freq)

        # Raw S11
        s11_raw = abs(S_base[port_index, port_index]) if port_index < N else 1.0
        raw_db.append(float(-20 * np.log10(max(s11_raw, 1e-15))))
        raw_mag.append(float(s11_raw))

        # Apply matching
        S = S_base.copy()
        for pi, choices in port_matching.items():
            for ch in choices:
                try:
                    cs = ch.component.get_s_matrix_at_freq(freq)
                    if ch.connection_type == 'series':
                        S = _embed_series_on_port(S, cs, pi)
                    elif ch.connection_type == 'shunt':
                        S = _embed_shunt_to_ground(S, cs, pi)
                except Exception:
                    pass

        s11_m = abs(S[port_index, port_index]) if port_index < S.shape[0] else 1.0
        match_db.append(float(-20 * np.log10(max(s11_m, 1e-15))))
        match_mag.append(float(s11_m))

        if include_efficiency:
            accepted = 1.0 - s11_m ** 2
            coupling = sum(abs(S[j, port_index]) ** 2 for j in range(S.shape[0]) if j != port_index)
            comp_params = []
            for pi, choices in port_matching.items():
                for ch in choices:
                    try:
                        cs = ch.component.get_s_matrix_at_freq(freq)
                        comp_params.append((cs, ch.connection_type))
                    except Exception:
                        pass
            comp_loss = estimate_total_component_loss(comp_params)
            total_eff = max(0.0, accepted - coupling - comp_loss)

            eff_accepted.append(float(accepted * 100))
            eff_coupling.append(float(coupling * 100))
            eff_comp_loss.append(float(comp_loss * 100))
            eff_total.append(float(total_eff * 100))

    result = {
        "frequencies": freqs.tolist(),
        "s11_db": match_db,
        "s11_magnitude": match_mag,
        "raw_db": raw_db,
        "raw_magnitude": raw_mag,
        "port_index": port_index,
        "source": source,
    }

    if include_efficiency:
        result["efficiency"] = {
            "accepted_pct": eff_accepted,
            "coupling_pct": eff_coupling,
            "component_loss_pct": eff_comp_loss,
            "total_pct": eff_total,
        }

    if include_power_balance and port_index < S.shape[0]:
        matched_ports = list(port_matching.keys())
        pb = compute_power_balance(
            S,
            component_loss_total=float(np.mean(eff_comp_loss[-1])) if eff_comp_loss else 0,
            matched_ports=matched_ports,
        )
        result["power_balance"] = pb.to_dict()
        result["power_balance_chart"] = power_balance_to_chart_data(pb)

    return result


@app.get("/api/tune/power-balance")
async def tune_power_balance(
    frequency_hz: float = 2.45e9,
):
    """
    Power balance analysis for the current tuning state.

    Returns the per-port breakdown of:
    - Reflected power
    - Coupled power (to other ports)
    - Component loss (in matching components)
    - Antenna loss (ohmic, if radiation efficiency data is loaded)
    - Radiated power

    Works with the best solution from the last /api/tune/joint call.
    """
    _ensure_snp_loaded()

    from engine.power_balance import compute_power_balance, power_balance_to_chart_data, SystemPowerBalance

    if not state.last_joint_results:
        # Compute power balance for unmatched DUT
        S = state.loaded_snp.get_s_matrix_interpolated(frequency_hz)
        pb = compute_power_balance(S, component_loss_total=0.0, matched_ports=[])
        return {
            "frequency_hz": frequency_hz,
            "mode": "raw_dut",
            "power_balance": pb.to_dict(),
            "chart_data": power_balance_to_chart_data(pb),
        }

    best = state.last_joint_results[0]
    if best.system_s_matrix is None:
        raise HTTPException(400, "No system S-matrix available. Run joint optimization first.")

    matched_ports = list(best.port_solutions.keys())
    pb = compute_power_balance(
        best.system_s_matrix,
        component_loss_total=best.component_loss_total,
        matched_ports=matched_ports,
        n_matched_ports=len(matched_ports),
        radiation_efficiency={
            pi: best.port_metrics.get(pi, {}).get('total_efficiency', 1.0)
            for pi in matched_ports
        } if False else None,
    )

    return {
        "frequency_hz": frequency_hz,
        "mode": "matched",
        "power_balance": pb.to_dict(),
        "chart_data": power_balance_to_chart_data(pb),
        "system_efficiency_pct": pb.system_efficiency * 100,
        "per_port": {str(k): v.to_dict() for k, v in pb.per_port.items()},
    }


# ─── Static files (frontend) ───
from engine.network import _embed_series_on_port, _embed_shunt_to_ground, connect_2port_to_multiport

if WEB_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("RFMATCH_HOST", "127.0.0.1"), port=8000)
