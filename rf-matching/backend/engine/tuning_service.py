"""
Unified tuning service layer.

This is the single entry point for all tuning operations in the API.
It replaces the scattered pipeline where server.py directly orchestrates
optimizers, sweep, and power balance logic.

Key design:
  - TuningSession holds the current project state (loaded SNP, results, selections)
  - TuningResult is the canonical output for any tuning operation
  - server.py delegates all tuning logic to this service
  - The service manages its own state, separate from server.py's AppState
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import math
import numpy as np
import time
import logging

from .touchstone import TouchstoneData
from .scoring import (
    get_objective_preset, ObjectivePreset,
    score_single_port, score_multi_port,
    estimate_total_component_loss,
    efficiency_chain,
)
from .power_balance import (
    compute_power_balance as _compute_power_balance,
    power_balance_to_chart_data,
    SystemPowerBalance,
)
from .tuning import TuningPlan, PortTuningSpec, TuningState, ObjectiveConfig, TuningMode
from .network import _embed_series_on_port, _embed_shunt_to_ground, terminate_ports, s_to_y, y_to_s
from .optimizer import MatchingOptimizer, OptimizerConfig, PortState, MatchingSolution
from .multiport_optimizer import JointMultiPortOptimizer, PortMatchConfig, evaluate_joint_solution
from .topology import get_standard_topologies

logger = logging.getLogger("rf_matching.optimizer")


def _format_components(components: List[dict]) -> str:
    if not components:
        return "none"
    return ", ".join(
        f"{c.get('type', '?')}:{c.get('value', '?')}:{c.get('part', '')}"
        for c in components
    )


# ── Canonical result model ──────────────────────────────────────────────

@dataclass
class PerPortTuningMetrics:
    """Canonical per-port tuning metrics."""
    port_index: int
    s11_magnitude: float = 0.0
    s11_db: float = 0.0
    accepted_efficiency: float = 0.0    # 1 - |S11|^2
    coupling_loss: float = 0.0          # Σ|Sji|^2 for j≠i
    radiated_efficiency: float = 0.0    # accepted - coupling
    component_loss: float = 0.0         # estimated loss in matching components
    total_efficiency: float = 0.0       # radiated - component_loss
    # Component info
    components: List[dict] = field(default_factory=list)

    # Band-swept values for charts
    band_freqs_hz: List[float] = field(default_factory=list)
    band_s11_db: List[float] = field(default_factory=list)
    band_total_eff: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'port_index': self.port_index,
            's11_magnitude': float(self.s11_magnitude),
            's11_db': float(self.s11_db),
            'accepted_efficiency': float(self.accepted_efficiency),
            'coupling_loss': float(self.coupling_loss),
            'radiated_efficiency': float(self.radiated_efficiency),
            'component_loss': float(self.component_loss),
            'total_efficiency': float(self.total_efficiency),
            'components': self.components,
            'band_freqs_hz': self.band_freqs_hz,
            'band_s11_db': self.band_s11_db,
            'band_total_eff': self.band_total_eff,
        }


@dataclass
class TuningResult:
    """
    Canonical result from any tuning operation.

    All numbers use 0-1 ratio internally (not %).
    All S11 values use magnitude (0-1) not dB.
    """
    # Identification
    port_indices: List[int] = field(default_factory=list)
    mode: str = "single"            # single | joint | tunable
    objective: str = "average_efficiency"

    # Per-port metrics at center frequency
    per_port: Dict[int, PerPortTuningMetrics] = field(default_factory=dict)

    # System-level metrics (averaged/aggregated)
    system_score: float = 0.0
    avg_total_efficiency: float = 0.0      # mean of all per-port total_efficiency
    min_total_efficiency: float = 0.0      # worst per-port total_efficiency
    avg_coupling_loss: float = 0.0
    max_coupling_loss: float = 0.0
    total_component_loss: float = 0.0
    total_component_count: int = 0

    # Power balance
    system_power_balance: Optional[dict] = None
    power_balance_chart: List[dict] = field(default_factory=list)

    # Runtime-only component choices used for accurate sweep recomputation.
    component_choices: Dict[int, list] = field(default_factory=dict, repr=False)

    # Sweep data (at selected solution)
    sweep_freqs_hz: List[float] = field(default_factory=list)

    # Metadata
    total_time_s: float = 0.0
    num_solutions_found: int = 0
    solution_index: int = 0          # which candidate this is (0 = best)

    def to_dict(self) -> dict:
        return {
            'port_indices': self.port_indices,
            'mode': self.mode,
            'objective': self.objective,
            'per_port': {str(k): v.to_dict() for k, v in self.per_port.items()},
            'system_score': float(self.system_score),
            'avg_total_efficiency': float(self.avg_total_efficiency),
            'min_total_efficiency': float(self.min_total_efficiency),
            'avg_coupling_loss': float(self.avg_coupling_loss),
            'max_coupling_loss': float(self.max_coupling_loss),
            'total_component_loss': float(self.total_component_loss),
            'total_component_count': self.total_component_count,
            'system_power_balance': self.system_power_balance,
            'power_balance_chart': self.power_balance_chart,
            'sweep_freqs_hz': self.sweep_freqs_hz,
            'total_time_s': float(self.total_time_s),
            'num_solutions_found': self.num_solutions_found,
            'solution_index': self.solution_index,
        }


def _component_type_from_choice(choice) -> str:
    comp = getattr(choice, 'component', None)
    explicit = getattr(comp, 'component_type', '') if comp is not None else ''
    if explicit:
        return str(explicit).lower()
    unit = str(getattr(comp, 'nominal_unit', '')).lower()
    if unit == 'nh':
        return 'inductor'
    if unit == 'pf':
        return 'capacitor'
    part = str(getattr(comp, 'part_number', '')).upper()
    return 'inductor' if part.startswith('L') else 'capacitor'


def _component_choice_to_dict(choice) -> dict:
    comp = getattr(choice, 'component', None)
    value = getattr(comp, 'nominal_value', None)
    unit = getattr(comp, 'nominal_unit', '')
    comp_type = _component_type_from_choice(choice)
    return {
        'position': getattr(choice, 'position', 0),
        'part': getattr(comp, 'part_number', ''),
        'part_number': getattr(comp, 'part_number', ''),
        'type': comp_type,
        'comp_type': comp_type,
        'connection_type': getattr(choice, 'connection_type', 'series'),
        'port': getattr(choice, 'port', 0),
        'nominal_value': float(value) if value is not None else None,
        'nominal_unit': unit,
        'value': f"{value}{unit}" if value is not None else '',
    }


@dataclass
class TuningSession:
    """
    Current tuning session state.
    Replaces the scattered state.last_solutions / state.last_joint_results pattern.
    """
    # Loaded SNP
    dut: Optional[TouchstoneData] = None
    dut_filename: str = ""

    # Component library
    library: Optional[object] = None

    # Current tuning plan
    plan: Optional[TuningPlan] = None

    # All candidate solutions (canonical TuningResult objects)
    candidate_solutions: List[TuningResult] = field(default_factory=list)

    # Which candidate is currently selected (0 = best)
    selected_index: int = 0

    # Current sweep data for the selected solution
    current_sweep: Optional[dict] = None
    current_power_balance: Optional[dict] = None

    # Efficiency data
    global_efficiency: Optional[object] = None
    per_port_efficiency: Dict[int, object] = field(default_factory=dict)

    # Debug info from last optimization (phase1 candidates, joint combos, final ranking)
    last_debug_info: dict = field(default_factory=dict)

    def get_selected_result(self) -> Optional[TuningResult]:
        if self.candidate_solutions and self.selected_index < len(self.candidate_solutions):
            return self.candidate_solutions[self.selected_index]
        return None

    def select_solution(self, index: int) -> bool:
        if 0 <= index < len(self.candidate_solutions):
            self.selected_index = index
            return True
        return False


# ── Global session (singleton, replaces AppState scattered fields) ──────

_session = TuningSession()


def get_session() -> TuningSession:
    return _session


def reset_session():
    """Reset the tuning session (e.g., when loading a new SNP)."""
    global _session
    _session = TuningSession()


# ── Service functions ───────────────────────────────────────────────────

def run_tuning_single(
    dut: TouchstoneData,
    library: object,
    port_index: int,
    bands_mhz: List[List[float]],
    max_components: int = 2,
    objective: str = "average_efficiency",
    beam_width: int = 20,
    timeout_seconds: float = 60.0,
    num_band_points: int = 10,
    global_efficiency: Optional[object] = None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
    component_series: Optional[List[str]] = None,
) -> Dict[int, TuningResult]:
    """
    Single-port tuning — the ONE entry point.

    Returns dict: {solution_index: TuningResult} for all candidates.
    The best solution is at index 0.
    """
    t_start = time.time()
    preset = get_objective_preset(objective)
    n_ports = dut.num_ports

    # Build port states
    port_states = {}
    for pi in range(n_ports):
        port_states[pi] = PortState.COMPONENT if pi == port_index else PortState.LOAD

    # Gather band frequencies
    band_freqs = []
    for band in bands_mhz:
        band_freqs.extend(
            np.linspace(band[0] * 1e6, band[1] * 1e6, num_band_points).tolist()
        )
    band_freqs = sorted(set(band_freqs))
    center_freq = (bands_mhz[0][0] + bands_mhz[0][1]) / 2.0 * 1e6

    # Run optimizer at center frequency
    config = OptimizerConfig(
        target_frequency_hz=center_freq,
        max_components=max_components,
        beam_width=beam_width,
        timeout_seconds=timeout_seconds,
        bands_mhz=bands_mhz,
        num_band_points=num_band_points,
    )
    opt = MatchingOptimizer(dut, library, config)

    topos = get_standard_topologies()
    topos = [t for t in topos if t.num_components <= max_components]

    solutions = opt.optimize_full(
        port_states=port_states,
        topologies=topos,
        input_port=port_index,
    )

    if not solutions:
        return {}

    # Convert to canonical TuningResult list
    candidates = {}
    for idx, sol in enumerate(solutions[:beam_width]):
        # Evaluate across all band frequencies
        band_s11_db = []
        band_total_eff = []
        for freq_hz in band_freqs:
            S = dut.get_s_matrix_interpolated(freq_hz)
            # Apply terminations
            term = {p: 0.0 for p, st in port_states.items()
                    if st in [PortState.OPEN, PortState.SHORT, PortState.LOAD]}
            if term:
                S = terminate_ports(S, term)

            # Apply matching
            try:
                for ch in sol.component_choices:
                    cs = ch.component.get_s_matrix_at_freq(freq_hz)
                    if ch.connection_type == 'series':
                        S = _embed_series_on_port(S, cs, ch.port)
                    elif ch.connection_type == 'shunt':
                        S = _embed_shunt_to_ground(S, cs, ch.port)
            except Exception:
                continue

            if S.shape[0] > 0:
                s11_mag = abs(S[0, 0])
                accepted = 1.0 - s11_mag ** 2
                coupling = sum(abs(S[j, 0]) ** 2 for j in range(1, S.shape[0]))

                # Component loss at this freq
                comp_params = []
                for ch in sol.component_choices:
                    try:
                        cs = ch.component.get_s_matrix_at_freq(freq_hz)
                        comp_params.append((cs, ch.connection_type))
                    except Exception:
                        pass
                comp_loss = estimate_total_component_loss(comp_params)

                total_eff = max(0.0, accepted - coupling - comp_loss)
                band_s11_db.append(float(-20 * np.log10(max(s11_mag, 1e-15))))
                band_total_eff.append(float(total_eff))

        if not band_total_eff:
            continue

        # Center-frequency metrics
        S_center = dut.get_s_matrix_interpolated(center_freq)
        term = {p: 0.0 for p, st in port_states.items()
                if st in [PortState.OPEN, PortState.SHORT, PortState.LOAD]}
        if term:
            S_center = terminate_ports(S_center, term)
        for ch in sol.component_choices:
            try:
                cs = ch.component.get_s_matrix_at_freq(center_freq)
                if ch.connection_type == 'series':
                    S_center = _embed_series_on_port(S_center, cs, ch.port)
                elif ch.connection_type == 'shunt':
                    S_center = _embed_shunt_to_ground(S_center, cs, ch.port)
            except Exception:
                pass

        s11_mag_c = abs(S_center[0, 0]) if S_center.shape[0] > 0 else 1.0
        accepted_c = 1.0 - s11_mag_c ** 2
        coupling_c = sum(abs(S_center[j, 0]) ** 2 for j in range(1, S_center.shape[0])) if S_center.shape[0] > 1 else 0.0
        comp_params_c = []
        for ch in sol.component_choices:
            try:
                cs = ch.component.get_s_matrix_at_freq(center_freq)
                comp_params_c.append((cs, ch.connection_type))
            except Exception:
                pass
        comp_loss_c = estimate_total_component_loss(comp_params_c)
        total_c = max(0.0, accepted_c - coupling_c - comp_loss_c)

        eff_array = np.array(band_total_eff)
        eff_score = score_single_port(eff_array, len(sol.component_choices), preset)

        pm = PerPortTuningMetrics(
            port_index=port_index,
            s11_magnitude=float(s11_mag_c),
            s11_db=float(-20 * np.log10(max(s11_mag_c, 1e-15))),
            accepted_efficiency=float(accepted_c),
            coupling_loss=float(coupling_c),
            component_loss=float(comp_loss_c),
            total_efficiency=float(total_c),
            radiated_efficiency=max(0.0, float(accepted_c - coupling_c)),
            components=[_component_choice_to_dict(c) for c in sol.component_choices],
            band_freqs_hz=band_freqs,
            band_s11_db=band_s11_db,
            band_total_eff=band_total_eff,
        )

        result = TuningResult(
            port_indices=[port_index],
            mode="single",
            objective=objective,
            per_port={port_index: pm},
            system_score=float(eff_score),
            avg_total_efficiency=float(np.mean(eff_array)),
            min_total_efficiency=float(np.min(eff_array)),
            total_component_count=len(sol.component_choices),
            component_choices={port_index: sol.component_choices},
            total_time_s=time.time() - t_start,
            num_solutions_found=len(solutions),
            solution_index=idx,
        )
        candidates[idx] = result

    return candidates


def run_tuning_joint(
    dut: TouchstoneData,
    library: object,
    port_specs: List[dict],
    objective: str = "balanced",
    beam_width: int = 10,
    timeout_seconds: float = 120.0,
    num_band_points: int = 5,
    global_efficiency: Optional[object] = None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
    debug: bool = False,
    debug_top_n: int = 10,
) -> Dict[int, TuningResult]:
    """
    Multi-port joint tuning — the ONE entry point.

    Args:
        port_specs: list of {port_index, bands_mhz, max_components, enabled}
        debug: When True, collect phase1/joint/final debug info
        debug_top_n: Number of entries to keep in each debug section

    Returns:
        dict: {solution_index: TuningResult}
    """
    t_start = time.time()
    preset = get_objective_preset(objective)

    # Build port configs
    port_configs = []
    all_bands_mhz = []
    for ps in port_specs:
        bands = ps.get('bands_mhz', [[2400, 2500]])
        center_hz = (bands[0][0] + bands[0][1]) / 2.0 * 1e6
        port_configs.append(PortMatchConfig(
            port_index=ps.get('port_index', 0),
            max_components=ps.get('max_components', 2),
            target_frequency_hz=center_hz,
        ))
        all_bands_mhz.extend(bands)

    n_ports = dut.num_ports

    # Run joint optimizer
    opt = JointMultiPortOptimizer(
        dut=dut,
        component_library=library,
        port_configs=port_configs,
        top_candidates_per_port=beam_width,
        timeout_seconds=timeout_seconds,
        min_avg_balance=0.5,
        optimization_mode=objective,
        radiation_efficiency=global_efficiency,
        per_port_efficiency=per_port_efficiency,
        debug=debug,
        debug_top_n=debug_top_n,
    )

    joint_solutions = opt.optimize(
        bands_mhz=all_bands_mhz if all_bands_mhz else None,
        num_band_points=num_band_points,
    )

    elapsed = time.time() - t_start

    if not joint_solutions:
        return {}

    # Get enabled port indices — only these contribute to system metrics
    enabled_port_indices = set(ps.get('port_index') for ps in port_specs)

    # Convert to canonical TuningResult list
    candidates = {}
    for idx, js in enumerate(joint_solutions[:beam_width]):
        per_port = {}
        all_total_effs = []
        all_coupling = []

        # Pre-compute band frequency points for sweep
        band_freqs = []
        if all_bands_mhz:
            for b in all_bands_mhz:
                band_freqs.extend(np.linspace(b[0] * 1e6, b[1] * 1e6, num_band_points).tolist())
            band_freqs = sorted(set(band_freqs))

        # Evaluate at each band frequency to get sweep data
        port_band_s11 = {pi: [] for pi in js.port_metrics}
        port_band_eff = {pi: [] for pi in js.port_metrics}

        for freq_hz in band_freqs:
            port_comp_choices = {
                pi: sol.component_choices
                for pi, sol in js.port_solutions.items()
            }
            sub_result = evaluate_joint_solution(
                dut, port_comp_choices, freq_hz,
                Z0=50.0,
                radiation_efficiency=global_efficiency,
                per_port_efficiency=per_port_efficiency,
            )
            if sub_result.get('valid'):
                for pi, pm in sub_result['port_metrics'].items():
                    s11_db = pm.get('s11_db', 0)
                    total_eff = pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
                    port_band_s11[pi].append(s11_db)
                    port_band_eff[pi].append(total_eff)
            else:
                for pi in port_band_s11:
                    port_band_s11[pi].append(0)
                    port_band_eff[pi].append(0)

        # Generate per-port metrics for ALL ports (including non-enabled for monitoring)
        for pi, pm in js.port_metrics.items():
            pb = js.power_balance.get(pi, {})
            total_eff = pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
            coupling = pm.get('coupling_loss', 0)
            accepted = pm.get('mismatch_efficiency', 0)

            # Only enabled ports contribute to system-level aggregates
            if pi in enabled_port_indices:
                all_total_effs.append(total_eff)
                all_coupling.append(coupling)
            else:
                # Mark non-enabled ports for monitoring
                pass

            per_port[pi] = PerPortTuningMetrics(
                port_index=pi,
                s11_magnitude=pm.get('s11_magnitude', 0),
                s11_db=pm.get('s11_db', 0),
                accepted_efficiency=float(accepted),
                coupling_loss=float(coupling),
                component_loss=pb.get('component_loss', 0),
                total_efficiency=float(total_eff),
                radiated_efficiency=pm.get('radiated_efficiency', 0),
                components=[
                    _component_choice_to_dict(c)
                    for c in js.port_solutions.get(pi).component_choices
                ] if pi in js.port_solutions else [],
                # Band sweep data (pre-computed for charts)
                band_freqs_hz=band_freqs,
                band_s11_db=port_band_s11.get(pi, []),
                band_total_eff=port_band_eff.get(pi, []),
            )

        avg_eff = float(np.mean(all_total_effs)) if all_total_effs else 0.0
        min_eff = float(np.min(all_total_effs)) if all_total_effs else 0.0
        avg_coupling = float(np.mean(all_coupling)) if all_coupling else 0.0
        max_coupling = float(np.max(all_coupling)) if all_coupling else 0.0

        # Compute power balance for this solution
        pb_system = None
        pb_chart = []
        if js.system_s_matrix is not None:
            matched_ports_list = list(js.port_solutions.keys())
            pb_system = _compute_power_balance(
                js.system_s_matrix,
                component_loss_total=js.component_loss_total,
                matched_ports=matched_ports_list,
                n_matched_ports=len(matched_ports_list),
            )
            pb_chart = power_balance_to_chart_data(pb_system) if pb_system else []

        # Recompute system score using only enabled-port efficiency
        system_score = 0.0
        if enabled_port_indices:
            enabled_effs = {}
            enabled_coupling = {}
            for pi in enabled_port_indices:
                if pi in per_port:
                    enabled_effs[pi] = np.array([per_port[pi].total_efficiency])
                    enabled_coupling[pi] = per_port[pi].coupling_loss
            comp_count = sum(len(sol.component_choices) for sol in js.port_solutions.values())
            system_score = score_multi_port(enabled_effs, enabled_coupling, comp_count, preset)

        result = TuningResult(
            port_indices=list(per_port.keys()),
            mode="joint",
            objective=objective,
            per_port=per_port,
            system_score=system_score,
            avg_total_efficiency=avg_eff,
            min_total_efficiency=min_eff,
            avg_coupling_loss=avg_coupling,
            max_coupling_loss=max_coupling,
            total_component_loss=js.component_loss_total,
            total_component_count=sum(
                len(sol.component_choices) for sol in js.port_solutions.values()
            ),
            system_power_balance=pb_system.to_dict() if pb_system else None,
            power_balance_chart=pb_chart,
            component_choices={
                pi: sol.component_choices
                for pi, sol in js.port_solutions.items()
            },
            total_time_s=elapsed,
            num_solutions_found=len(joint_solutions),
            solution_index=idx,
        )
        candidates[idx] = result

    # Re-sort candidates by recomputed system_score (descending)
    sorted_items = sorted(candidates.items(), key=lambda x: x[1].system_score, reverse=True)
    candidates = {new_idx: item for new_idx, (_, item) in enumerate(sorted_items)}

    logger.info(
        "Tuning joint canonical ranking: objective=%s enabled_ports=%s candidates=%d",
        objective,
        sorted(enabled_port_indices),
        len(candidates),
    )
    for rank, result in list(candidates.items())[: min(10, len(candidates))]:
        port_bits = []
        for pi in sorted(enabled_port_indices):
            pm = result.per_port.get(pi)
            if not pm:
                continue
            port_bits.append(
                "P%d RL=%.2fdB total=%.5f coupling=%.5f comps=[%s]" % (
                    pi + 1,
                    pm.s11_db,
                    pm.total_efficiency,
                    pm.coupling_loss,
                    _format_components(pm.components),
                )
            )
        logger.info(
            "  canonical#%02d score=%.5f avg=%.5f min=%.5f comp_loss=%.5f %s",
            rank + 1,
            result.system_score,
            result.avg_total_efficiency,
            result.min_total_efficiency,
            result.total_component_loss,
            " | ".join(port_bits),
        )

    # Store debug info in session for the API endpoint to read
    if debug:
        session = get_session()
        session.last_debug_info = getattr(opt, '_debug_info', {})

    return candidates


def compute_sweep(
    dut: TouchstoneData,
    component_choices: Dict[int, list],
    port_index: int,
    start_hz: float,
    stop_hz: float,
    num_points: int = 200,
    include_efficiency: bool = True,
    use_snp_points: bool = True,
) -> dict:
    """
    Compute frequency sweep for a selected solution.

    Returns dict with:
      frequencies, s11_db, s11_magnitude, raw_db, raw_magnitude
      efficiency: {accepted_pct, coupling_pct, component_loss_pct, total_pct}
    """
    if use_snp_points and dut.frequencies:
        freqs = np.array([
            f for f in dut.frequencies
            if (start_hz is None or f >= start_hz) and (stop_hz is None or f <= stop_hz)
        ], dtype=float)
        if len(freqs) == 0:
            freqs = np.array(dut.frequencies, dtype=float)
    else:
        freqs = np.linspace(start_hz, stop_hz, max(2, num_points))
    N = dut.num_ports

    raw_db, raw_mag = [], []
    match_db, match_mag = [], []
    eff_total, eff_accepted, eff_coupling, eff_comp_loss = [], [], [], []

    for freq in freqs:
        S_base = dut.get_s_matrix_interpolated(freq)

        # Raw S11
        s11_raw = abs(S_base[port_index, port_index]) if port_index < N else 1.0
        raw_db.append(float(-20 * np.log10(max(s11_raw, 1e-15))))
        raw_mag.append(float(s11_raw))

        # Apply matching
        S = S_base.copy()
        for pi, choices in component_choices.items():
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
            for pi, choices in component_choices.items():
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
    }

    if include_efficiency:
        result["efficiency"] = {
            "accepted_pct": eff_accepted,
            "coupling_pct": eff_coupling,
            "component_loss_pct": eff_comp_loss,
            "total_pct": eff_total,
        }

    return result


def compute_power_balance(
    dut: TouchstoneData,
    component_choices: Dict[int, list],
    component_loss_total: float = 0.0,
    frequency_hz: float = 2.45e9,
) -> dict:
    """Compute power balance for a given solution at a frequency."""
    S = dut.get_s_matrix_interpolated(frequency_hz)

    # Apply matching
    for pi, choices in component_choices.items():
        for ch in choices:
            try:
                cs = ch.component.get_s_matrix_at_freq(frequency_hz)
                if ch.connection_type == 'series':
                    S = _embed_series_on_port(S, cs, pi)
                elif ch.connection_type == 'shunt':
                    S = _embed_shunt_to_ground(S, cs, pi)
            except Exception:
                pass

    matched_ports = list(component_choices.keys())
    pb = _compute_power_balance(
        S,
        component_loss_total=component_loss_total,
        matched_ports=matched_ports,
        n_matched_ports=len(matched_ports),
    )
    return {
        "power_balance": pb.to_dict(),
        "chart_data": power_balance_to_chart_data(pb),
        "system_efficiency_pct": pb.system_efficiency * 100,
    }


# ── Tunable C multi-state search (MVP) ─────────────────────────────────

def run_tuning_tunable_c(
    dut: TouchstoneData,
    library: object,
    port_index: int,
    band_state_map: Dict[str, List[float]],
    max_states: int = 3,
    beam_width: int = 20,
    num_band_points: int = 5,
    global_efficiency: Optional[object] = None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
) -> Dict[int, TuningResult]:
    """
    Tunable capacitor search MVP.

    Topology 1: series L (shared) + shunt tunable C (per-state).
    Topology 2: shunt L (shared) + series tunable C (per-state).

    Each band state uses a different C value; L is shared across all states.
    Goal: maximize average total efficiency across all states.

    Args:
        dut: Touchstone data
        library: Component library
        port_index: Port to optimize
        band_state_map: {"state_name": [start_mhz, stop_mhz], ...}
            e.g. {"LB": [700, 960], "HB": [1710, 2170]}
        max_states: Max number of states
        beam_width: Number of top candidates

    Returns:
        Dict of {solution_index: TuningResult}
    """
    t_start = time.time()
    n_ports = dut.num_ports

    inductors = _get_available_inductors(library)
    capacitors = _get_available_capacitors(library)
    if not inductors or not capacitors:
        # Fallback: use ideal components with broad range
        inductors = [(v, None) for v in [1.0, 1.5, 2.2, 3.3, 4.7, 6.8, 10, 15, 22, 33, 47, 68, 100]]
        capacitors = [(v, None) for v in [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.2, 3.3, 4.7, 6.8, 10, 15, 22]]

    # Sample broadly across the value range (not just the first N)
    inductors = _sample_broadly(inductors, 50)
    capacitors = _sample_broadly(capacitors, 50)

    state_names = list(band_state_map.keys())[:max_states]
    state_freqs = {}
    state_centers = {}
    for name in state_names:
        band = band_state_map[name]
        freqs = np.linspace(band[0] * 1e6, band[1] * 1e6, num_band_points).tolist()
        state_freqs[name] = sorted(set(freqs))
        state_centers[name] = (band[0] + band[1]) / 2.0 * 1e6

    candidates = {}
    sol_idx = 0
    preset_score = get_objective_preset('average_efficiency')

    def _get_s_matrix(comp, val, freq_hz, comp_type):
        """Get S-matrix from real component or compute ideal."""
        if comp is not None:
            try:
                return comp.get_s_matrix_at_freq(freq_hz)
            except (AttributeError, TypeError, Exception):
                # ComponentRecord or other type without get_s_matrix_at_freq
                # Try to extract nominal value
                try:
                    if hasattr(comp, 'nominal_value') and comp.nominal_value is not None:
                        val = comp.nominal_value
                except Exception:
                    pass
                # Fall through to ideal calculation
        # Ideal component
        omega = 2 * math.pi * freq_hz
        if comp_type == 'L':
            Z = 1j * omega * val * 1e-9
        else:
            Z = 1.0 / (1j * omega * val * 1e-12)
        Z0 = 50.0
        gamma = Z / (Z + 2 * Z0)
        thru = 2 * Z0 / (Z + 2 * Z0)
        return np.array([[gamma, thru], [thru, gamma]], dtype=complex)

    # ── Topology 1: series L (shared) + shunt tunable C (per-state) ──
    for L_val, L_comp in inductors[:beam_width]:
        state_best_C = {}
        total_effs = []

        for s_name in state_names:
            best_c_val = None
            best_c_eff = -1.0
            best_c_raw_eff = 0.0
            best_c_comp = None

            for C_val, C_comp in capacitors:
                freq_hz = state_centers[s_name]
                S = dut.get_s_matrix_interpolated(freq_hz)
                # Terminate all other ports with matched load
                term = {}
                for pi in range(n_ports):
                    if pi != port_index:
                        term[pi] = 0.0  # matched load
                if term:
                    try:
                        S = terminate_ports(S, term)
                    except Exception:
                        continue

                # After termination, S is 1x1 — components go on port 0
                local_port = 0
                try:
                    L_s = _get_s_matrix(L_comp, L_val, freq_hz, 'L')
                    S = _embed_series_on_port(S, L_s, local_port)
                except Exception:
                    continue

                try:
                    C_s = _get_s_matrix(C_comp, C_val, freq_hz, 'C')
                    gamma_shunt = C_s[0, 0] + C_s[0, 1] * (-1) * C_s[1, 0] / (1 - (-1) * C_s[1, 1])
                    Y0 = 1.0 / 50.0
                    Y_shunt_val = Y0 * (1.0 - gamma_shunt) / (1.0 + gamma_shunt)
                    Y = s_to_y(S)
                    Y[local_port, local_port] += Y_shunt_val
                    S = y_to_s(Y)
                except Exception:
                    continue

                if S.shape[0] > 0:
                    s11_mag = abs(S[0, 0])
                    accepted = 1.0 - s11_mag ** 2
                    coupling = sum(abs(S[j, 0]) ** 2 for j in range(1, S.shape[0])) if S.shape[0] > 1 else 0.0
                    try:
                        comp_params = [(L_s, 'series'), (C_s, 'shunt')]
                        comp_loss = estimate_total_component_loss(comp_params)
                    except Exception:
                        comp_loss = 0.0
                    total_eff = max(0.0, accepted - coupling - comp_loss)
                    # Mildly discourage hard endpoint values; keep raw efficiency
                    # for reported metrics so a truly better edge value can still win.
                    adjusted_eff = total_eff - _boundary_penalty(C_val, capacitors)
                    if adjusted_eff > best_c_eff:
                        best_c_eff = adjusted_eff
                        best_c_raw_eff = total_eff
                        best_c_val = C_val
                        best_c_comp = C_comp

            state_best_C[s_name] = (best_c_val, best_c_comp)
            if best_c_val is None:
                # No valid C found — skip this L candidate
                total_effs = []
                break
            total_effs.append(best_c_raw_eff)

        if not total_effs:
            continue

        avg_eff = float(np.mean(total_effs))
        eff_array = np.array(total_effs)
        score = score_single_port(eff_array, 2, preset_score)
        score = max(0.0, score - _boundary_penalty(L_val, inductors) * 0.5)

        c_value_str = ', '.join([f'{s}={v:.1f}pF' for s, (v, _c) in state_best_C.items() if v is not None])
        pm = PerPortTuningMetrics(
            port_index=port_index,
            s11_magnitude=float(math.sqrt(max(0.0, 1.0 - avg_eff))),
            s11_db=float(-20 * math.log10(max(math.sqrt(max(0.0, 1.0 - avg_eff)), 1e-15))),
            accepted_efficiency=avg_eff,
            coupling_loss=0.0,
            component_loss=0.0,
            total_efficiency=avg_eff,
            radiated_efficiency=avg_eff,
            components=[
                {'part': f'L={L_val}nH (shared)', 'type': 'inductor', 'value': f'{L_val}nH'},
                {'part': 'Tunable C', 'type': 'capacitor', 'value': c_value_str},
            ],
        )
        result = TuningResult(
            port_indices=[port_index], mode="tunable", objective="average_efficiency",
            per_port={port_index: pm}, system_score=score,
            avg_total_efficiency=avg_eff, min_total_efficiency=float(np.min(total_effs)),
            total_component_count=2, total_time_s=time.time() - t_start,
            num_solutions_found=sol_idx + 1, solution_index=sol_idx,
        )
        candidates[sol_idx] = result
        sol_idx += 1

    # ── Topology 2: shunt L (shared) + series tunable C (per-state) ──
    for L_val, L_comp in inductors[:beam_width]:
        state_best_C = {}
        total_effs = []

        for s_name in state_names:
            best_c_val = None
            best_c_eff = -1.0
            best_c_raw_eff = 0.0
            best_c_comp = None

            for C_val, C_comp in capacitors:
                freq_hz = state_centers[s_name]
                S = dut.get_s_matrix_interpolated(freq_hz)
                # Terminate all other ports
                term = {}
                for pi in range(n_ports):
                    if pi != port_index:
                        term[pi] = 0.0
                if term:
                    try:
                        S = terminate_ports(S, term)
                    except Exception:
                        continue
                local_port = 0  # port 0 after reduction

                try:
                    C_s = _get_s_matrix(C_comp, C_val, freq_hz, 'C')
                    S = _embed_series_on_port(S, C_s, local_port)
                except Exception:
                    continue

                try:
                    L_s = _get_s_matrix(L_comp, L_val, freq_hz, 'L')
                    gamma_shunt = L_s[0, 0] + L_s[0, 1] * (-1) * L_s[1, 0] / (1 - (-1) * L_s[1, 1])
                    Y0 = 1.0 / 50.0
                    Y_shunt_val = Y0 * (1.0 - gamma_shunt) / (1.0 + gamma_shunt)
                    Y = s_to_y(S)
                    Y[local_port, local_port] += Y_shunt_val
                    S = y_to_s(Y)
                except Exception:
                    continue

                if S.shape[0] > 0:
                    s11_mag = abs(S[0, 0])
                    accepted = 1.0 - s11_mag ** 2
                    coupling = sum(abs(S[j, 0]) ** 2 for j in range(1, S.shape[0])) if S.shape[0] > 1 else 0.0
                    try:
                        comp_params = [(C_s, 'series'), (L_s, 'shunt')]
                        comp_loss = estimate_total_component_loss(comp_params)
                    except Exception:
                        comp_loss = 0.0
                    total_eff = max(0.0, accepted - coupling - comp_loss)
                    # Mildly discourage hard endpoint values; keep raw efficiency
                    # for reported metrics so a truly better edge value can still win.
                    adjusted_eff = total_eff - _boundary_penalty(C_val, capacitors)
                    if adjusted_eff > best_c_eff:
                        best_c_eff = adjusted_eff
                        best_c_raw_eff = total_eff
                        best_c_val = C_val
                        best_c_comp = C_comp

            state_best_C[s_name] = (best_c_val, best_c_comp)
            if best_c_val is None:
                # No valid C found — skip this L candidate
                total_effs = []
                break
            total_effs.append(best_c_raw_eff)

        if not total_effs:
            continue

        avg_eff = float(np.mean(total_effs))
        eff_array = np.array(total_effs)
        score = score_single_port(eff_array, 2, preset_score)
        score = max(0.0, score - _boundary_penalty(L_val, inductors) * 0.5)

        c_value_str = ', '.join([f'{s}={v:.1f}pF' for s, (v, _c) in state_best_C.items() if v is not None])
        pm = PerPortTuningMetrics(
            port_index=port_index,
            s11_magnitude=float(math.sqrt(max(0.0, 1.0 - avg_eff))),
            s11_db=float(-20 * math.log10(max(math.sqrt(max(0.0, 1.0 - avg_eff)), 1e-15))),
            accepted_efficiency=avg_eff,
            coupling_loss=0.0,
            component_loss=0.0,
            total_efficiency=avg_eff,
            radiated_efficiency=avg_eff,
            components=[
                {'part': f'L={L_val}nH (shared)', 'type': 'inductor', 'value': f'{L_val}nH'},
                {'part': 'Tunable C', 'type': 'capacitor', 'value': c_value_str},
            ],
        )
        result = TuningResult(
            port_indices=[port_index], mode="tunable", objective="average_efficiency",
            per_port={port_index: pm}, system_score=score,
            avg_total_efficiency=avg_eff, min_total_efficiency=float(np.min(total_effs)),
            total_component_count=2, total_time_s=time.time() - t_start,
            num_solutions_found=sol_idx + 1, solution_index=sol_idx,
        )
        candidates[sol_idx] = result
        sol_idx += 1

    return candidates


# ── Library helper functions ──

def _get_available_inductors(library) -> List[Tuple[float, object]]:
    """Get (value_nH, component) pairs sorted by value."""
    result = []
    try:
        if hasattr(library, 'db') and hasattr(library.db, 'get_primary_inductors'):
            for c in library.db.get_primary_inductors():
                if hasattr(c, 'nominal_value') and c.nominal_value is not None:
                    result.append((c.nominal_value, c))
        elif hasattr(library, 'inductors'):
            for c in library.inductors:
                if hasattr(c, 'nominal_value'):
                    result.append((c.nominal_value, c))
    except Exception:
        pass
    return sorted(result, key=lambda x: x[0])


def _get_available_capacitors(library) -> List[Tuple[float, object]]:
    """Get (value_pF, component) pairs sorted by value."""
    result = []
    try:
        if hasattr(library, 'db') and hasattr(library.db, 'get_primary_capacitors'):
            for c in library.db.get_primary_capacitors():
                if hasattr(c, 'nominal_value') and c.nominal_value is not None:
                    result.append((c.nominal_value, c))
        elif hasattr(library, 'capacitors'):
            for c in library.capacitors:
                if hasattr(c, 'nominal_value'):
                    result.append((c.nominal_value, c))
    except Exception:
        pass
    return sorted(result, key=lambda x: x[0])


def _boundary_penalty(value: float, items: List) -> float:
    """
    Mild penalty for component values near the edge of the available range.

    Returns 0.0 for interior values, rising quadratically to ~0.01 at the
    extremes.  Strong enough to break ties and gently discourage endpoints,
    not enough to override a genuinely better boundary result.
    """
    if len(items) < 4:
        return 0.0
    # Sorted unique values
    vals = sorted({v for v, _ in items if v is not None and v > 0})
    if len(vals) < 4:
        return 0.0
    v_min, v_max = vals[0], vals[-1]
    if v_max <= v_min:
        return 0.0
    pos = (value - v_min) / (v_max - v_min)  # 0..1
    edge_dist = min(pos, 1.0 - pos)          # 0 at ends, 0.5 at centre
    if edge_dist > 0.15:
        return 0.0
    # Quadratic: 0.01 at edge, 0 at 15 % from edge
    return 0.01 * (1.0 - edge_dist / 0.15) ** 2


def _sample_broadly(items: List, n: int) -> List:
    """Sample ~n items spread evenly across the *value* range (not index).

    Deduplicates by nominal value, drops invalid (<= 0) values, samples by
    value percentiles, and avoids hard min/max endpoints when enough values
    exist.  Falls back to index sampling for short lists.
    """
    if not items:
        return []
    # Deduplicate by nominal value and drop invalid entries
    seen = set()
    cleaned = []
    for val, comp in items:
        if val is not None and val > 0 and val not in seen:
            seen.add(val)
            cleaned.append((val, comp))
    if len(cleaned) <= n:
        return cleaned
    cleaned.sort(key=lambda x: x[0])
    values = np.array([v for v, _ in cleaned])
    m = len(cleaned)
    # Use interior percentiles to avoid hard 0 % and 100 % endpoints
    if m > n + 2 and n > 3:
        percentiles = np.linspace(100.0 / (n + 1), 100.0 - 100.0 / (n + 1), n)
    else:
        percentiles = np.linspace(0, 100, n)
    # Pick the closest actual item for each target percentile
    result = []
    used = set()
    for p in percentiles:
        target = float(np.percentile(values, p))
        best_i = min((i for i in range(m) if i not in used),
                     key=lambda i: abs(values[i] - target))
        used.add(best_i)
        result.append(cleaned[best_i])
    result.sort(key=lambda x: x[0])
    return result
