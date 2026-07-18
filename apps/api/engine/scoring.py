"""
Optenni-style unified scoring for RF matching/tuning.

Core philosophy (from the user's guidance):
  The primary objective is NOT lowest S11 — it is
  **maximum system average total efficiency** while controlling
  worst-case frequency points, coupling loss, component loss,
  and component complexity.

Score formulas (single-port):
    score =
      0.75 * avg(total_efficiency over band)
    + 0.25 * min(total_efficiency over band)
    - 0.03 * component_count
    - tolerance_penalty

Score formulas (multi-port):
    score =
      0.55 * avg(total_efficiency over all ports/bands)
    + 0.25 * min(total_efficiency over all ports/bands)
    - 0.15 * avg(coupling_loss)
    - 0.05 * complexity_penalty

Where:
    accepted_efficiency  = 1 - |Sii|^2
    radiated_efficiency  = accepted_efficiency - coupling_loss
    coupling_loss        = sum_{j!=i} |Sji|^2
    total_efficiency     = radiation_efficiency * radiated_efficiency
                           - component_loss
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# ── Preset objective modes ──────────────────────────────────────────────

@dataclass
class ObjectivePreset:
    """Named tuning objective with weights and description."""
    name: str
    label: str
    description: str
    # Weights for single-port scoring
    w_avg_eff: float = 0.75
    w_min_eff: float = 0.25
    w_complexity: float = 0.03
    # Additional weights for multi-port scoring
    w_avg_port_eff: float = 0.55
    w_min_port_eff: float = 0.25
    w_coupling: float = 0.15
    w_complexity_multi: float = 0.05


OBJECTIVE_PRESETS: Dict[str, ObjectivePreset] = {
    'average_efficiency': ObjectivePreset(
        name='average_efficiency',
        label='Best Average Efficiency',
        description='Maximize average total efficiency across all ports and bands',
        w_avg_eff=0.85, w_min_eff=0.15, w_complexity=0.03,
        w_avg_port_eff=0.65, w_min_port_eff=0.20,
        w_coupling=0.10, w_complexity_multi=0.05,
    ),
    'worst_case': ObjectivePreset(
        name='worst_case',
        label='Best Worst-Case',
        description='Maximize the minimum (worst) efficiency across all ports and bands',
        w_avg_eff=0.20, w_min_eff=0.80, w_complexity=0.05,
        w_avg_port_eff=0.15, w_min_port_eff=0.65,
        w_coupling=0.12, w_complexity_multi=0.08,
    ),
    'balanced': ObjectivePreset(
        name='balanced',
        label='Balanced',
        description='Equal emphasis on average and worst-case performance',
        w_avg_eff=0.55, w_min_eff=0.45, w_complexity=0.04,
        w_avg_port_eff=0.40, w_min_port_eff=0.35,
        w_coupling=0.15, w_complexity_multi=0.10,
    ),
    'low_coupling': ObjectivePreset(
        name='low_coupling',
        label='Low Coupling / MIMO Safe',
        description='Heavily penalize inter-port coupling for MIMO applications',
        w_avg_eff=0.35, w_min_eff=0.20, w_complexity=0.03,
        w_avg_port_eff=0.25, w_min_port_eff=0.15,
        w_coupling=0.45, w_complexity_multi=0.05,
    ),
    'low_cost': ObjectivePreset(
        name='low_cost',
        label='Low Component Count',
        description='Penalize complex multi-component networks',
        w_avg_eff=0.50, w_min_eff=0.25, w_complexity=0.25,
        w_avg_port_eff=0.35, w_min_port_eff=0.15,
        w_coupling=0.10, w_complexity_multi=0.30,
    ),
}


def get_objective_preset(name: str) -> ObjectivePreset:
    """Get a predefined objective by name; fallback to balanced."""
    if name in OBJECTIVE_PRESETS:
        return OBJECTIVE_PRESETS[name]
    return OBJECTIVE_PRESETS['balanced']


# ── Efficiency chain ────────────────────────────────────────────────────

def compute_accepted_efficiency(s11_mag: float) -> float:
    """η_accepted = 1 - |Sii|² — power accepted by the antenna port."""
    return max(0.0, 1.0 - s11_mag ** 2)


def compute_coupling_loss(s_matrix: np.ndarray, port: int) -> float:
    """Power coupled from port i into all other ports: Σ_{j≠i} |Sji|²."""
    n = s_matrix.shape[0]
    loss = sum(abs(s_matrix[j, port]) ** 2 for j in range(n) if j != port)
    return float(loss)


def compute_radiated_efficiency(accepted_eff: float, coupling_loss: float) -> float:
    """η_radiated = η_accepted - coupling_loss."""
    return max(0.0, accepted_eff - coupling_loss)


def compute_total_efficiency(
    radiated_eff: float,
    radiation_efficiency: float = 1.0,
    component_loss: float = 0.0,
) -> float:
    """
    η_total = η_radiation * η_radiated - component_loss.
    radiation_efficiency is the antenna's own efficiency (from pattern measurement).
    component_loss is power dissipated in matching components.
    """
    return max(0.0, radiation_efficiency * radiated_eff - component_loss)


def efficiency_chain(
    s11_mag: float,
    coupling_loss: float,
    radiation_efficiency: float = 1.0,
    component_loss: float = 0.0,
) -> Dict[str, float]:
    """
    Compute the full efficiency chain for one port at one frequency.

    Returns:
        dict with keys: accepted_efficiency, coupling_loss, radiated_efficiency,
                        component_loss, total_efficiency
    """
    accepted = compute_accepted_efficiency(s11_mag)
    radiated = compute_radiated_efficiency(accepted, coupling_loss)
    total = compute_total_efficiency(radiated, radiation_efficiency, component_loss)
    return {
        'accepted_efficiency': float(accepted),
        'coupling_loss': float(coupling_loss),
        'radiated_efficiency': float(radiated),
        'component_loss': float(component_loss),
        'total_efficiency': float(total),
    }


# ── Single-port scoring ─────────────────────────────────────────────────

def score_single_port(
    efficiencies: np.ndarray,      # total_efficiency across all band frequencies
    component_count: int,
    preset: ObjectivePreset,
    tolerance_penalty: float = 0.0,
) -> float:
    """
    Score a single-port solution.

    Args:
        efficiencies: Array of total_efficiency values at each band frequency point
        component_count: Number of matching components
        preset: Objective preset
        tolerance_penalty: Optional penalty from tolerance analysis

    Returns:
        Score in [0, 1], higher is better
    """
    if len(efficiencies) == 0:
        return 0.0

    avg_eff = float(np.mean(efficiencies))
    min_eff = float(np.min(efficiencies))

    raw = (
        + preset.w_avg_eff * avg_eff
        + preset.w_min_eff * min_eff
        - preset.w_complexity * min(component_count / 8.0, 1.0)
        - tolerance_penalty
    )

    # Normalize to [0, 1]
    max_raw = preset.w_avg_eff + preset.w_min_eff
    if max_raw > 0:
        return float(np.clip((raw + preset.w_complexity) / (max_raw + preset.w_complexity), 0.0, 1.0))
    return 0.0


# ── Multi-port scoring ──────────────────────────────────────────────────

def score_multi_port(
    per_port_efficiencies: Dict[int, np.ndarray],  # port -> array of total_efficiency
    per_port_coupling: Dict[int, float],            # port -> coupling_loss (per-port avg)
    component_count: int,
    preset: ObjectivePreset,
) -> float:
    """
    Score a multi-port joint solution.

    Args:
        per_port_efficiencies: {port_index: np.array of total_efficiency at each freq point}
        per_port_coupling: {port_index: average coupling loss across band}
        component_count: Total number of matching components
        preset: Objective preset

    Returns:
        Score in [0, 1], higher is better
    """
    if not per_port_efficiencies:
        return 0.0

    # Flatten all efficiencies
    all_effs = np.concatenate(list(per_port_efficiencies.values()))
    if len(all_effs) == 0:
        return 0.0

    avg_eff = float(np.mean(all_effs))
    min_eff = float(np.min(all_effs))

    # Per-port average efficiencies
    port_avgs = [float(np.mean(e)) for e in per_port_efficiencies.values()]
    port_mins = [float(np.min(e)) for e in per_port_efficiencies.values()]
    avg_port_eff = float(np.mean(port_avgs))
    min_port_eff = float(np.min(port_mins))

    # Coupling: average across ports
    coupling_vals = list(per_port_coupling.values())
    avg_coupling = float(np.mean(coupling_vals)) if coupling_vals else 0.0

    # Complexity penalty
    complexity_penalty = min(component_count / 12.0, 1.0)

    raw = (
        + preset.w_avg_port_eff * avg_port_eff
        + preset.w_min_port_eff * min_port_eff
        + preset.w_avg_eff * avg_eff * 0.3   # global average bonus
        + preset.w_min_eff * min_eff * 0.2    # global min bonus
        - preset.w_coupling * avg_coupling
        - preset.w_complexity_multi * complexity_penalty
    )

    # Normalize
    positive_weights = (
        preset.w_avg_port_eff + preset.w_min_port_eff
        + preset.w_avg_eff * 0.3 + preset.w_min_eff * 0.2
    )
    negative_weights = preset.w_coupling + preset.w_complexity_multi

    if positive_weights + negative_weights > 0:
        return float(np.clip(
            (raw + negative_weights) / (positive_weights + negative_weights),
            0.0, 1.0
        ))
    return 0.0


# ─── Component loss estimation (from S-parameters) ─────────────────────

def estimate_component_loss_power(
    comp_s: np.ndarray,
    incident_wave_ratio: float = 1.0,
) -> float:
    """
    Estimate the fraction of incident power dissipated in a 2-port component
    from its S-parameters.

    For a 2-port network:
        P_diss / P_inc = 1 - |S11|² - |S21|²

    Args:
        comp_s: 2x2 S-parameter matrix
        incident_wave_ratio: Fraction of incident power reaching this component

    Returns:
        Fraction of incident power dissipated [0, 1]
    """
    s11, s21 = comp_s[0, 0], comp_s[1, 0]
    reflected = abs(s11) ** 2
    transmitted = abs(s21) ** 2
    dissipated = max(0.0, 1.0 - reflected - transmitted)
    return float(dissipated * incident_wave_ratio)


def estimate_total_component_loss(
    component_s_params: List[Tuple[np.ndarray, str]],
) -> float:
    """
    Sum power dissipation across all matching components.

    For series: nearly all power flows through (ratio=1.0)
    For shunt: heuristic ratio=0.3
    For parallel: heuristic ratio=0.5

    Args:
        component_s_params: List of (2x2 S-matrix, connection_type) tuples

    Returns:
        Total fractional power loss [0, 1]
    """
    total_loss = 0.0
    for comp_s, conn_type in component_s_params:
        if conn_type == 'shunt':
            incident_ratio = 0.3
        elif conn_type == 'parallel':
            incident_ratio = 0.5
        else:  # series
            incident_ratio = 1.0
        total_loss += estimate_component_loss_power(comp_s, incident_ratio)
    return min(total_loss, 1.0)


# ── Convenience: score from evaluated metrics ───────────────────────────

def score_from_evaluation(
    port_metrics: Dict[int, dict],
    component_s_params: List[Tuple[np.ndarray, str]],
    component_count: int,
    preset: ObjectivePreset,
    n_ports: int,
) -> float:
    """
    Compute score directly from the output of evaluate_joint_solution().

    Args:
        port_metrics: Dict from evaluate_joint_solution()['port_metrics']
        component_s_params: List of (S-matrix, connection_type) for each component
        component_count: Number of matching components
        preset: Objective preset
        n_ports: Total number of DUT ports

    Returns:
        Unified score in [0, 1]
    """
    per_port_effs = {}
    per_port_coupling = {}

    for i in range(n_ports):
        pm = port_metrics.get(i, {})
        # Use total_efficiency where available, fall back to mismatch_efficiency
        eff = pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
        per_port_effs[i] = np.array([eff])
        per_port_coupling[i] = pm.get('coupling_loss', 0)

    return score_multi_port(
        per_port_effs, per_port_coupling,
        component_count, preset,
    )
