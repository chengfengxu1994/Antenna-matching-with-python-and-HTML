"""
FastAPI backend server for RF Matching application.
Integrated with SQLite DB adapter for fast component queries.
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.touchstone import parse_touchstone, TouchstoneData
from engine.component_lib import scan_murata_directory, ComponentLibrary
from engine.murata_db_adapter import load_murata_db, MurataDBLibrary
from engine.topology import get_standard_topologies
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from engine.multiport_optimizer import JointMultiPortOptimizer, PortMatchConfig, evaluate_joint_solution
from engine.cost_function import (
    get_optimization_mode, OPTIMIZATION_MODES,
)
from engine.efficiency_data import load_efficiency_file, parse_efficiency_data, EfficiencyData

app = FastAPI(title="RF Matching Engine", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── State ───

class AppState:
    def __init__(self):
        self.snp_dir = ""
        self.murata_dir = ""
        self.db_path = ""
        self.loaded_snp = None
        self.loaded_snp_filename = ""
        self.component_library = None     # ComponentLibrary (S2P fallback)
        self.db_library = None            # MurataDBLibrary (primary)
        self.use_db = True
        self.optimizer = None
        self.last_solutions = []
        self.last_joint_results = None

state = AppState()

DB_CANDIDATE = r"E:\RF matching\Murata\murata_components.db"

# ─── Models ───

class DataDirConfig(BaseModel):
    snp_dir: str = r"E:\RF matching\snp"
    murata_dir: str = r"E:\RF matching\Murata"

class PortStateConfig(BaseModel):
    port_index: int
    state: str = 'load'

class SinglePortConfig(BaseModel):
    port_index: int
    state: str = 'load'
    use_matching: bool = False
    max_components: int = 2
    l_min_nh: float = 0.1
    l_max_nh: float = 20.0
    c_min_pf: float = 0.1
    c_max_pf: float = 20.0
    band_mhz: List[float] = [2400, 2500]
    num_band_points: int = 5

class MultiPortOptimizeRequest(BaseModel):
    snp_filename: str
    ports: List[SinglePortConfig]
    beam_width: int = 10
    timeout_seconds: float = 60.0
    optimization_goal: str = 'efficiency'
    component_series: Optional[List[str]] = None

class JointOptimizeRequest(BaseModel):
    snp_filename: str
    ports: List[SinglePortConfig]
    beam_width: int = 10
    timeout_seconds: float = 120.0
    optimization_goal: str = 'efficiency'

class OptimizeRequest(BaseModel):
    snp_filename: str
    target_frequency_hz: float = 64e6
    max_components: int = 4
    port_states: List[PortStateConfig] = []
    input_port: int = 0
    topologies_filter: Optional[List[str]] = None
    beam_width: int = 10
    timeout_seconds: float = 60.0
    bands_mhz: Optional[List[List[float]]] = None
    num_band_points: int = 5

class ManualTuneRequest(BaseModel):
    snp_filename: str
    target_frequency_hz: float = 64e6
    input_port: int = 0
    port_states: List[PortStateConfig] = []
    components: list = []
    sweep_start_hz: Optional[float] = None
    sweep_stop_hz: Optional[float] = None
    sweep_points: int = 100

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

@app.post("/api/config/dirs")
async def set_data_dirs(config: DataDirConfig):
    state.snp_dir = config.snp_dir
    state.murata_dir = config.murata_dir

    # Try DB first
    db_path = os.path.join(config.murata_dir, "murata_components.db")
    if os.path.isfile(db_path):
        try:
            state.db_library = load_murata_db(db_path)
            state.use_db = True
            stats = state.db_library.stats
            return {
                "status": "ok", "mode": "database",
                "db_path": db_path,
                "total_components": stats['total_components'],
                "primary_components": stats['primary_components'],
                "inductors": stats.get('primary_inductors', 0),
                "capacitors": stats.get('primary_capacitors', 0),
            }
        except Exception as e:
            pass

    # Fallback to S2P scan
    if os.path.isdir(config.murata_dir):
        state.component_library = scan_murata_directory(config.murata_dir)
        state.use_db = False
        return {
            "status": "ok", "mode": "s2p_scan",
            "inductors": len(state.component_library.inductors),
            "capacitors": len(state.component_library.capacitors),
        }

    return {"status": "warning", "message": "Murata directory not found"}

@app.get("/api/snp/list")
async def list_snp_files():
    if not state.snp_dir or not os.path.isdir(state.snp_dir):
        return {"files": []}
    files = []
    for f in sorted(os.listdir(state.snp_dir)):
        if f.lower().endswith(('.s1p', '.s2p', '.s3p', '.s4p', '.s5p', '.s6p')):
            filepath = os.path.join(state.snp_dir, f)
            try:
                with open(filepath, 'r') as fh:
                    data = parse_touchstone(fh.read(), filename=f)
                files.append({
                    "filename": f,
                    "num_ports": data.num_ports,
                    "freq_count": len(data.frequencies),
                    "freq_min_hz": min(data.frequencies) if data.frequencies else 0,
                    "freq_max_hz": max(data.frequencies) if data.frequencies else 0,
                })
            except:
                pass
    return {"files": files}

@app.post("/api/snp/load")
async def load_snp(filename: str):
    if not state.snp_dir:
        raise HTTPException(400, "SNP directory not configured")
    filepath = os.path.join(state.snp_dir, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(404, "File not found: " + filename)
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    state.loaded_snp = parse_touchstone(content, filename=filename)
    state.loaded_snp_filename = filename
    state.last_solutions = []
    state.last_joint_results = None
    return {
        "status": "ok", "filename": filename,
        "num_ports": state.loaded_snp.num_ports,
        "freq_count": len(state.loaded_snp.frequencies),
        "freq_min_hz": min(state.loaded_snp.frequencies) if state.loaded_snp.frequencies else 0,
        "freq_max_hz": max(state.loaded_snp.frequencies) if state.loaded_snp.frequencies else 0,
        "frequencies": state.loaded_snp.frequencies[:100],
    }

@app.get("/api/band-presets")
async def get_band_presets():
    return {"presets": BAND_PRESETS}

@app.get("/api/component-series")
async def list_component_series():
    _ensure_library()
    if state.use_db and state.db_library:
        series_list = state.db_library.get_series_summary()
        inductor_series = {s['name']: s['count'] for s in series_list if s['type'] == 'inductor'}
        capacitor_series = {s['name']: s['count'] for s in series_list if s['type'] == 'capacitor'}
    else:
        import re
        inductor_series, capacitor_series = {}, {}
        for c in state.component_library.inductors:
            m = re.match(r'(LQ[A-Z]\d{2}[A-Z]{2})', c.part_number.upper())
            s = m.group(1) if m else c.part_number[:7]
            inductor_series[s] = inductor_series.get(s, 0) + 1
        for c in state.component_library.capacitors:
            m = re.match(r'(G[A-Z]{2}\d{2})', c.part_number.upper())
            s = m.group(1) if m else c.part_number[:5]
            capacitor_series[s] = capacitor_series.get(s, 0) + 1
    return {"inductor_series": dict(sorted(inductor_series.items())),
            "capacitor_series": dict(sorted(capacitor_series.items()))}

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

@app.get("/api/topologies/list")
async def list_topologies(max_components: int = 4):
    topos = [t for t in get_standard_topologies() if t.num_components <= max_components]
    return {"topologies": [
        {"name": t.name, "num_components": t.num_components, "description": t.description,
         "elements": [{"position": e.position, "connection_type": e.connection_type.value,
                       "port": e.port, "component_type": e.component_type} for e in t.elements]}
        for t in topos
    ]}

@app.post("/api/optimize")
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

@app.post("/api/multipass")
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

# In-memory storage: port_index → EfficiencyData
_per_port_efficiency_data: Dict[int, EfficiencyData] = {}
_global_efficiency_data: Optional[EfficiencyData] = None


@app.post("/api/efficiency/load")
async def load_efficiency(port_index: int = -1, filepath: str = ""):
    """
    Load radiation efficiency data for a specific port.
    port_index=-1 means apply to all ports (global).
    """
    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(400, "File not found: " + filepath)
    try:
        eff_data = load_efficiency_file(filepath)
        if port_index < 0:
            _global_efficiency_data = eff_data
            port_label = "all ports (global)"
        else:
            _per_port_efficiency_data[port_index] = eff_data
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
async def load_efficiency_inline(port_index: int = -1, content: str = "", filename: str = "pasted_data"):
    """Load efficiency data from pasted text content."""
    if not content.strip():
        raise HTTPException(400, "Empty content")
    try:
        eff_data = parse_efficiency_data(content, filename=filename)
        if port_index < 0:
            _global_efficiency_data = eff_data
        else:
            _per_port_efficiency_data[port_index] = eff_data
        return {"status": "ok", "port_index": port_index, "efficiency": eff_data.to_dict()}
    except Exception as e:
        raise HTTPException(400, f"Failed to parse efficiency data: {e}")


@app.get("/api/efficiency/status")
async def efficiency_status():
    """Check which ports have efficiency data loaded."""
    result = {
        "loaded": _global_efficiency_data is not None or len(_per_port_efficiency_data) > 0,
        "global": _global_efficiency_data.to_dict() if _global_efficiency_data else None,
        "per_port": {
            str(k): v.to_dict() for k, v in _per_port_efficiency_data.items()
        },
    }
    return result


@app.post("/api/efficiency/clear")
async def clear_efficiency(port_index: int = -1):
    """Clear efficiency data. port_index=-1 clears all."""
    if port_index < 0:
        _per_port_efficiency_data.clear()
        _global_efficiency_data = None
    elif port_index in _per_port_efficiency_data:
        del _per_port_efficiency_data[port_index]
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


@app.post("/api/joint-optimize")
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
    if _per_port_efficiency_data:
        per_port_eff = dict(_per_port_efficiency_data)
    if _global_efficiency_data:
        eff_data = _global_efficiency_data

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

@app.post("/api/joint-optimize/sweep")
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
    _ensure_library()
    freq = request.target_frequency_hz
    port_states = {}
    for ps in request.port_states:
        smap = {'open': PortState.OPEN, 'short': PortState.SHORT, 'load': PortState.LOAD}
        port_states[ps.port_index] = smap.get(ps.state, PortState.LOAD)
    S = state.loaded_snp.get_s_matrix_interpolated(freq)
    term = {p: 0.0 for p, st in port_states.items() if st in [PortState.OPEN, PortState.SHORT, PortState.LOAD]}
    if term:
        from engine.network import terminate_ports
        S = terminate_ports(S, term)
    for comp_req in request.components:
        omega = 2 * np.pi * freq
        if isinstance(comp_req, dict):
            ct = comp_req.get('comp_type', 'inductor')
            val = comp_req.get('value', 10.0)
            ideal = comp_req.get('use_ideal', True)
            conn = comp_req.get('connection_type', 'series')
            port = comp_req.get('port', 0)
        else:
            ct, val, ideal, conn, port = 'inductor', 10.0, True, 'series', 0
        if ideal:
            if ct == 'inductor':
                Z = 1j * omega * (val * 1e-9)
            else:
                Z = 1.0 / (1j * omega * (val * 1e-12))
            Z0 = 50.0
            gamma = Z / (Z + 2*Z0)
            thru = 2*Z0 / (Z + 2*Z0)
            comp_s = np.array([[gamma, thru], [thru, gamma]], dtype=complex)
        else:
            lib = _get_library()
            if ct == 'inductor':
                nearest = lib.find_nearest_inductor(val) if hasattr(lib, 'find_nearest_inductor') else None
            else:
                nearest = lib.find_nearest_capacitor(val) if hasattr(lib, 'find_nearest_capacitor') else None
            if nearest is None:
                raise HTTPException(404, "No component found near %s %s" % (val, ct))
            comp_s = nearest.get_s_matrix_at_freq(freq) if hasattr(nearest, 'get_s_matrix_at_freq') else None
            if comp_s is None:
                raise HTTPException(500, "Cannot get S-parameters")
        try:
            if conn == 'series':
                S = _embed_series_on_port(S, comp_s, port)
            elif conn == 'shunt':
                S = _embed_shunt_to_ground(S, comp_s, port)
        except Exception as e:
            raise HTTPException(400, "Embedding failed: %s" % str(e))
    s11 = S[0, 0] if S.shape[0] > 0 else 1.0+0j
    mag = abs(s11)
    result = {
        "s11_magnitude": float(mag),
        "s11_db": float(-20*np.log10(max(mag, 1e-15))),
        "s11_real": float(s11.real), "s11_imag": float(s11.imag),
        "vswr": float((1+mag)/(1-mag)) if mag < 1 else float('inf'),
        "frequency_hz": freq,
    }
    if request.sweep_start_hz and request.sweep_stop_hz:
        sweep_freqs = np.linspace(request.sweep_start_hz, request.sweep_stop_hz, request.sweep_points)
        sweep_mags, sweep_dbs, sweep_real, sweep_imag = [], [], [], []
        for sf in sweep_freqs:
            S_sf = state.loaded_snp.get_s_matrix_interpolated(sf)
            if term:
                from engine.network import terminate_ports
                S_sf = terminate_ports(S_sf, term)
            for comp_req in request.components:
                omega_s = 2 * np.pi * sf
                if isinstance(comp_req, dict):
                    ct = comp_req.get('comp_type', 'inductor')
                    val = comp_req.get('value', 10.0)
                    ideal = comp_req.get('use_ideal', True)
                    conn = comp_req.get('connection_type', 'series')
                    port = comp_req.get('port', 0)
                else:
                    continue
                if ideal:
                    if ct == 'inductor':
                        Z = 1j * omega_s * (val * 1e-9)
                    else:
                        Z = 1.0 / (1j * omega_s * (val * 1e-12))
                    Z0 = 50.0
                    g = Z / (Z + 2*Z0)
                    t = 2*Z0 / (Z + 2*Z0)
                    comp_sf = np.array([[g, t], [t, g]], dtype=complex)
                else:
                    continue
                try:
                    if conn == 'series':
                        S_sf = _embed_series_on_port(S_sf, comp_sf, port)
                    elif conn == 'shunt':
                        S_sf = _embed_shunt_to_ground(S_sf, comp_sf, port)
                except:
                    pass
            s11_sf = S_sf[0, 0] if S_sf.shape[0] > 0 else 1.0+0j
            m = abs(s11_sf)
            sweep_mags.append(float(m))
            sweep_dbs.append(float(-20*np.log10(max(m, 1e-15))))
            sweep_real.append(float(s11_sf.real))
            sweep_imag.append(float(s11_sf.imag))
        result["sweep"] = {"frequencies": sweep_freqs.tolist(),
                          "s11_magnitude": sweep_mags, "s11_db": sweep_dbs,
                          "s11_real": sweep_real, "s11_imag": sweep_imag}
    return result

# ─── Static files (frontend) ───
from engine.network import _embed_series_on_port, _embed_shunt_to_ground, connect_2port_to_multiport

frontend_dist = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'dist')
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
