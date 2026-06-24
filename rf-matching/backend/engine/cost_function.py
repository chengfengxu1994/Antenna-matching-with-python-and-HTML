"""
Unified cost function for RF matching optimization.
Optenni-style multi-objective scoring that balances efficiency,
coupling, component loss, and complexity.

The unified score is:

    score =
      + w_avg_band_eff × avg_band_efficiency
      + w_min_band_eff × min_band_efficiency
      + w_avg_port_eff  × avg_port_efficiency
      + w_min_port_eff  × min_port_efficiency
      - w_coupling      × coupling_penalty
      - w_comp_loss     × component_loss_penalty
      - w_complexity    × component_count_penalty

Each term is in [0, 1] so the final score is in [0, 1].
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ── Predefined optimization modes ──────────────────────────────────────

@dataclass
class OptimizationMode:
    """Named optimization preset with weight vector."""
    name: str
    label: str
    description: str
    w_avg_band_eff: float = 0.0
    w_min_band_eff: float = 0.0
    w_avg_port_eff: float = 0.0
    w_min_port_eff: float = 0.0
    w_coupling: float = 0.0
    w_comp_loss: float = 0.0
    w_complexity: float = 0.0


OPTIMIZATION_MODES: Dict[str, OptimizationMode] = {
    'efficiency': OptimizationMode(
        name='efficiency', label='Optimize Efficiency',
        description='Maximize total radiated efficiency across all ports',
        w_avg_band_eff=0.15, w_min_band_eff=0.25,
        w_avg_port_eff=0.15, w_min_port_eff=0.25,
        w_coupling=0.10,
        w_comp_loss=0.05,
        w_complexity=0.05,
    ),
    'return_loss': OptimizationMode(
        name='return_loss', label='Optimize Return Loss',
        description='Minimize |S11| — classic impedance match',
        w_avg_band_eff=0.05, w_min_band_eff=0.05,
        w_avg_port_eff=0.25, w_min_port_eff=0.35,
        w_coupling=0.10,
        w_comp_loss=0.10,
        w_complexity=0.10,
    ),
    'balanced': OptimizationMode(
        name='balanced', label='Balanced',
        description='Equal emphasis on efficiency, coupling, and simplicity',
        w_avg_band_eff=0.20, w_min_band_eff=0.20,
        w_avg_port_eff=0.15, w_min_port_eff=0.15,
        w_coupling=0.10,
        w_comp_loss=0.10,
        w_complexity=0.10,
    ),
    'worst_case': OptimizationMode(
        name='worst_case', label='Worst-Case Focused',
        description='Prioritise worst-performing port across all bands',
        w_avg_band_eff=0.10, w_min_band_eff=0.35,
        w_avg_port_eff=0.05, w_min_port_eff=0.30,
        w_coupling=0.10,
        w_comp_loss=0.05,
        w_complexity=0.05,
    ),
    'average_focused': OptimizationMode(
        name='average_focused', label='Average Focused',
        description='Maximise average (not worst) performance',
        w_avg_band_eff=0.35, w_min_band_eff=0.10,
        w_avg_port_eff=0.25, w_min_port_eff=0.05,
        w_coupling=0.10,
        w_comp_loss=0.10,
        w_complexity=0.05,
    ),
    'low_cost': OptimizationMode(
        name='low_cost', label='Low-Cost / Robust',
        description='Penalise component count and lossy parts more heavily',
        w_avg_band_eff=0.15, w_min_band_eff=0.20,
        w_avg_port_eff=0.10, w_min_port_eff=0.10,
        w_coupling=0.10,
        w_comp_loss=0.15,
        w_complexity=0.20,
    ),
}


def get_optimization_mode(name: str) -> OptimizationMode:
    """Get a predefined optimization mode by name."""
    if name in OPTIMIZATION_MODES:
        return OPTIMIZATION_MODES[name]
    # Default to balanced
    return OPTIMIZATION_MODES['balanced']


# ── Per-component loss estimation ──────────────────────────────────────

def estimate_component_loss_power(
    comp_s: np.ndarray,
    incident_wave_ratio: float = 1.0,
) -> float:
    """
    Estimate the fraction of incident power dissipated in a 2-port component
    from its S-parameters.

    For a 2-port network:
        P_diss / P_inc = 1 - |S11|² - |S21|²

    This assumes the component port 2 sees a matched load (which is
    approximately true when cascaded through a low-loss network).

    Args:
        comp_s: 2×2 S-parameter matrix of the component
        incident_wave_ratio: Fraction of total incident power reaching this
                             component (0-1). Default 1.0.

    Returns:
        Fraction of incident power dissipated in the component [0, 1]
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
    Sum the estimated power dissipation across all matching components.

    For series components, nearly all incident power flows through them,
    so the loss fraction directly applies.
    For shunt components, only a fraction of the power flows through them,
    so we apply a heuristic scaling.

    Args:
        component_s_params: List of (2×2 S-matrix, connection_type) tuples.

    Returns:
        Total estimated fractional power loss [0, 1]
    """
    total_loss = 0.0
    for i, (comp_s, conn_type) in enumerate(component_s_params):
        # Shunt components carry less total power than series components
        if conn_type == 'shunt':
            incident_ratio = 0.3  # heuristic: ~30% of power through shunt
        elif conn_type == 'parallel':
            incident_ratio = 0.5  # half the power through this branch
        else:  # series
            incident_ratio = 1.0

        loss = estimate_component_loss_power(comp_s, incident_ratio)
        total_loss += loss

    return min(total_loss, 1.0)


# ── Unified scoring ────────────────────────────────────────────────────

@dataclass
class ScoreInput:
    """All metrics needed to compute the unified score for one solution."""
    # Per-port efficiency at each band frequency point
    # shape: (n_ports,) — average across band frequencies
    avg_port_efficiency: np.ndarray

    # shape: (n_ports,) — minimum across band frequencies per port
    min_port_efficiency: np.ndarray

    # Global band efficiency — the average across ALL evaluated band points
    avg_band_efficiency: float

    # Worst single band-point across all ports
    min_band_efficiency: float

    # Coupling loss — max |Sji|² across all port pairs
    max_coupling_loss: float = 0.0

    # Estimated component loss fraction
    component_loss: float = 0.0

    # Number of matching components used
    component_count: int = 0

    # Tolerance sensitivity (placeholder for future Monte Carlo)
    tolerance_sensitivity: float = 0.0


def compute_unified_score(
    inp: ScoreInput,
    mode: OptimizationMode,
) -> float:
    """
    Compute the unified score for a solution given an optimization mode.

    Returns a value in [0, 1] (higher = better).
    """
    # Compute each term
    avg_port_term = float(np.mean(inp.avg_port_efficiency))
    min_port_term = float(np.min(inp.min_port_efficiency))
    avg_band_term = inp.avg_band_efficiency
    min_band_term = inp.min_band_efficiency

    # Penalties
    coupling_penalty = inp.max_coupling_loss
    comp_loss_penalty = inp.component_loss
    # Normalise component count: assume max reasonable = 8
    complexity_penalty = min(inp.component_count / 8.0, 1.0)

    # Weighted sum
    score = (
        + mode.w_avg_band_eff * avg_band_term
        + mode.w_min_band_eff * min_band_term
        + mode.w_avg_port_eff * avg_port_term
        + mode.w_min_port_eff * min_port_term
        - mode.w_coupling * coupling_penalty
        - mode.w_comp_loss * comp_loss_penalty
        - mode.w_complexity * complexity_penalty
    )

    # Normalise to [0, 1] assuming max raw score ≈ sum of all positive weights
    total_positive = (
        mode.w_avg_band_eff + mode.w_min_band_eff
        + mode.w_avg_port_eff + mode.w_min_port_eff
    )
    total_negative = (
        mode.w_coupling + mode.w_comp_loss + mode.w_complexity
    )
    if total_positive + total_negative > 0:
        score = (score + total_negative) / (total_positive + total_negative)
    else:
        score = 0.0

    return float(np.clip(score, 0.0, 1.0))


def compute_score_from_metrics(
    port_metrics: Dict[int, dict],
    component_s_params: List[Tuple[np.ndarray, str]],
    component_count: int,
    mode: OptimizationMode,
    n_ports: int,
) -> float:
    """
    Convenience function: compute score directly from evaluation output.

    Args:
        port_metrics: Dict from evaluate_joint_solution() result
        component_s_params: List of (S-matrix, connection_type) for each component
        component_count: Number of matching components
        mode: Optimization mode
        n_ports: Total number of DUT ports

    Returns:
        Unified score in [0, 1]
    """
    # Gather per-port efficiency values
    avg_port_eff = []
    min_port_eff = []
    for i in range(n_ports):
        pm = port_metrics.get(i, {})
        # Use total efficiency if available, fall back to mismatch efficiency
        avg_port_eff.append(pm.get('total_efficiency', pm.get('mismatch_efficiency', 0)))
        min_port_eff.append(pm.get('total_efficiency', pm.get('mismatch_efficiency', 0)))

    # Compute global band metrics
    all_total_effs = [pm.get('total_efficiency', pm.get('mismatch_efficiency', 0))
                      for pm in port_metrics.values()]
    all_mismatch_effs = [pm.get('mismatch_efficiency', 0) for pm in port_metrics.values()]

    avg_band_efficiency = float(np.mean(all_total_effs)) if all_total_effs else 0.0
    min_band_efficiency = float(np.min(all_total_effs)) if all_total_effs else 0.0

    max_coupling = max(
        (pm.get('coupling_loss', 0) for pm in port_metrics.values()),
        default=0.0,
    )

    comp_loss = estimate_total_component_loss(component_s_params)

    inp = ScoreInput(
        avg_port_efficiency=np.array(avg_port_eff),
        min_port_efficiency=np.array(min_port_eff),
        avg_band_efficiency=avg_band_efficiency,
        min_band_efficiency=min_band_efficiency,
        max_coupling_loss=max_coupling,
        component_loss=comp_loss,
        component_count=component_count,
    )

    return compute_unified_score(inp, mode)
