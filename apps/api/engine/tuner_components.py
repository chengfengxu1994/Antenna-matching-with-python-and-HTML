"""
Tunable component models for RF matching/tuning.

Supports:
  - VariableCapacitor: Model of a tunable capacitor (like Murata LXR series)
    with discrete capacitance states and S-parameter interpolation
  - SwitchBranch: Model of a switch branch (SP2T/SP3T/SP4T)
    with per-state component networks
  - IdealTunableL: Ideal variable inductor (for theoretical upper-bound search)
  - TunableComponentCollection: Collection of tunable components for a TuningPlan

These models allow Optenni-style "tuner" search where:
  - Fixed components are shared across all states
  - Variable components change value per state
  - Switch branches provide different matching topologies per state
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum


class TunableType(str, Enum):
    """Type of tunable element."""
    VARIABLE_CAPACITOR = "variable_capacitor"
    VARIABLE_INDUCTOR = "variable_inductor"
    SWITCH = "switch"
    FIXED = "fixed"


@dataclass
class TunableState:
    """
    One discrete state of a tunable component.

    Attributes:
        value: The capacitance (pF) or inductance (nH) value
        label: Optional human-readable label
        s_matrix_func: Optional function freq_hz → 2x2 S-matrix
    """
    value: float
    label: str = ""
    s_matrix_func: Optional[Callable] = None


@dataclass
class VariableCapacitor:
    """
    Model of a tunable variable capacitor.

    Supports:
    - Discrete capacitance states (realistic for switched capacitor banks)
    - Continuous range with step size (for ideal search)
    - S-parameter interpolation from real component data

    Attributes:
        name: Component name/identifier
        values_pf: List of capacitance values in pF for discrete states
        continuous_range: (min_pf, max_pf) for continuous mode
        step_pf: Step size for continuous mode
        q_factor: Optional Q factor for ideal model
        # If using real S-parameters, provide a mapping: value_pf → 2x2 S-matrix
        s_params_at_freq: Optional callable(freq_hz, value_pf) → 2x2 S-matrix
    """
    name: str = "VC"
    values_pf: List[float] = field(default_factory=list)
    continuous_range: Optional[Tuple[float, float]] = None
    step_pf: float = 0.1
    q_factor: Optional[float] = None
    s_params_at_freq: Optional[Callable] = None

    def get_states(self) -> List[TunableState]:
        """Get all discrete states."""
        states = []
        for v in self.values_pf:
            label = f"{self.name}={v:.1f}pF"
            s_func = None
            if self.s_params_at_freq:
                s_func = lambda freq, val=v: self.s_params_at_freq(freq, val)
            states.append(TunableState(value=v, label=label, s_matrix_func=s_func))
        return states

    def get_ideal_s_matrix(self, freq_hz: float, value_pf: float) -> np.ndarray:
        """
        Compute ideal S-matrix for a capacitor at given frequency.

        Returns series-mode 2x2 S-matrix.
        """
        omega = 2 * np.pi * freq_hz
        Zc = 1.0 / (1j * omega * value_pf * 1e-12)

        # Add ESR from Q factor
        if self.q_factor and self.q_factor > 0:
            # Q = |Xc| / ESR
            xc = abs(1.0 / (omega * value_pf * 1e-12))
            esr = xc / self.q_factor
            Zc = Zc + esr

        Z0 = 50.0
        gamma = Zc / (Zc + 2 * Z0)
        thru = 2 * Z0 / (Zc + 2 * Z0)
        return np.array([[gamma, thru], [thru, gamma]], dtype=complex)


@dataclass
class SwitchBranch:
    """
    One branch of a switch-based tuner.

    Each branch corresponds to one switch position and contains
    a fixed matching network.

    Attributes:
        name: Branch name (e.g. "LB", "HB")
        components: List of (connection_type, component_value, component_type) tuples
        target_bands: Which bands this branch targets
    """
    name: str
    components: List[Tuple[str, float, str]] = field(default_factory=list)
    target_bands: List[List[float]] = field(default_factory=lambda: [[2400, 2500]])


@dataclass
class SwitchModel:
    """
    Model of a switch with multiple branches (SP2T, SP3T, etc.).

    Attributes:
        name: Switch name
        num_positions: Number of switch positions (2 for SP2T, etc.)
        branches: List of SwitchBranch, one per position
        insertion_loss_db: Insertion loss in dB (for ideal model)
        isolation_db: Isolation in dB (for ideal model)
    """
    name: str = "SW"
    num_positions: int = 2
    branches: List[SwitchBranch] = field(default_factory=list)
    insertion_loss_db: float = 0.3
    isolation_db: float = 25.0

    def get_ideal_s_matrix(self, freq_hz: float, active_branch: int) -> np.ndarray:
        """
        Compute ideal 2x2 S-matrix for the switch with one active branch.

        For a switch in the "on" state:
        - S21 = sqrt(10^(-IL/10)) = transmission coefficient
        - S11 = 0 (ideally matched)
        - S22 = 0 (ideally matched)

        For "off" branches (isolation):
        - S21 = sqrt(10^(-isolation/10))
        """
        il_linear = 10 ** (-self.insertion_loss_db / 10)
        # Simple model: matched through line with loss
        # |S21|^2 = IL_linear
        s21_mag = np.sqrt(il_linear)
        # Ideal switch: S11 = S22 = 0, S12 = S21
        return np.array([
            [0.0, s21_mag],
            [s21_mag, 0.0],
        ], dtype=complex)


@dataclass
class TunableComponentCollection:
    """
    Collection of tunable components for a complete TuningPlan.

    Attributes:
        variable_capacitors: {position_index: VariableCapacitor}
        switches: {position_index: SwitchModel}
        fixed_components: List of fixed (non-tunable) component descriptors
    """
    variable_capacitors: Dict[int, VariableCapacitor] = field(default_factory=dict)
    switches: Dict[int, SwitchModel] = field(default_factory=dict)
    fixed_components: List[dict] = field(default_factory=list)

    def get_all_states_combinations(self) -> List[Dict[int, float]]:
        """
        Get all combinations of tunable component states.

        Returns list of {position_index: value} dicts.
        This can grow exponentially — use with caution.
        """
        # Collect all state lists per position
        position_states = {}
        for pos, vc in self.variable_capacitors.items():
            position_states[pos] = vc.values_pf
        for pos, sw in self.switches.items():
            position_states[pos] = list(range(sw.num_positions))

        if not position_states:
            return [{}]

        # Generate all combinations
        from itertools import product
        keys = list(position_states.keys())
        values = list(position_states.values())
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations


# ── Convenience: build standard tunable capacitor banks ─────────────────

def make_standard_tunable_bank(
    name: str = "VC",
    c_min_pf: float = 0.5,
    c_max_pf: float = 10.0,
    num_states: int = 8,
    q_factor: Optional[float] = 50.0,
) -> VariableCapacitor:
    """
    Create a standard tunable capacitor bank with log-spaced states.

    Args:
        name: Component name
        c_min_pf: Minimum capacitance
        c_max_pf: Maximum capacitance
        num_states: Number of discrete states
        q_factor: Q factor for ideal model

    Returns:
        VariableCapacitor with num_states evenly log-spaced values
    """
    values = np.logspace(np.log10(c_min_pf), np.log10(c_max_pf), num_states)
    return VariableCapacitor(
        name=name,
        values_pf=[round(float(v), 2) for v in values],
        q_factor=q_factor,
    )
