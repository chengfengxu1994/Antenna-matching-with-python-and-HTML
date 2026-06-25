"""
Tuning data model for Optenni-style antenna tuning.

Key concepts:
  TuningPlan represents a full tuning configuration:
    - Multiple ports, each with its own band and topology constraints
    - Multiple tuning states (for tunable components or switch positions)
    - An objective (efficiency mode, weights)

  PortTuningSpec describes one port's tuning requirements.
  TuningState describes one configuration state (e.g. "LB_B17", "WiFi_2G").

Modes:
  - fixed_lc:    Traditional fixed LC matching network
  - tunable_c:   One or more variable capacitor positions
  - switch:      Switch with multiple states, each with its own LC branch
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class TuningMode(str, Enum):
    FIXED_LC = "fixed_lc"
    TUNABLE_C = "tunable_c"
    SWITCH = "switch"


@dataclass
class PortTuningSpec:
    """
    Tuning specification for one antenna port.

    Attributes:
        port_index: 0-based port index
        enabled: Whether to optimize this port
        bands_mhz: List of [start_mhz, stop_mhz] bands to target
        max_components: Maximum number of matching components
        topology_constraints: Optional list of topology names to restrict search
        component_series: Optional filter for component series (e.g. ['LQP03HQ', 'GJM03'])
        tunable_c_positions: List of positions (0-based) of variable capacitors
        switch_branches: Number of switch branches (for switch mode)
    """
    port_index: int
    enabled: bool = True
    bands_mhz: List[List[float]] = field(default_factory=lambda: [[2400, 2500]])
    max_components: int = 2
    topology_constraints: Optional[List[str]] = None
    component_series: Optional[List[str]] = None
    tunable_c_positions: Optional[List[int]] = None
    switch_branches: int = 1

    @property
    def center_freq_hz(self) -> float:
        """Center of the first band, in Hz."""
        if self.bands_mhz:
            return (self.bands_mhz[0][0] + self.bands_mhz[0][1]) / 2.0 * 1e6
        return 2.45e9


@dataclass
class TuningState:
    """
    One state of a tunable tuning plan.

    For fixed_lc mode, there is typically one state.
    For tunable_c mode, each state has different cap values.
    For switch mode, each state represents one switch position.

    Attributes:
        name: Human-readable name (e.g. "LTE B7", "WiFi 2G")
        active_bands: Which bands are active in this state
        variable_values: {position_index: value_pf} for tunable capacitors
        switch_position: Which branch is active (for switch mode)
        fixed_components: Fixed LC components shared across states
    """
    name: str
    active_bands: List[List[float]] = field(default_factory=lambda: [[2400, 2500]])
    variable_values: Dict[int, float] = field(default_factory=dict)
    switch_position: int = 0
    fixed_components: List[dict] = field(default_factory=list)

    @property
    def center_freq_hz(self) -> float:
        """Center frequency of the first active band, in Hz."""
        if self.active_bands:
            return (self.active_bands[0][0] + self.active_bands[0][1]) / 2.0 * 1e6
        return 2.45e9


@dataclass
class ObjectiveConfig:
    """
    Tuning objective configuration.

    Attributes:
        primary: Primary optimization goal name
        avg_weight: Weight for average efficiency (single-port)
        min_weight: Weight for worst-case efficiency (single-port)
        coupling_penalty: Penalty weight for inter-port coupling (multi-port)
        component_count_penalty: Penalty weight for component count
    """
    primary: str = 'balanced'
    avg_weight: float = 0.55
    min_weight: float = 0.25
    coupling_penalty: float = 0.15
    component_count_penalty: float = 0.05


@dataclass
class TuningPlan:
    """
    Complete tuning plan: ports, states, objective.

    This is the top-level data structure equivalent to Optenni's
    "Tuning Plan" concept.

    Attributes:
        ports: List of per-port tuning specs
        states: List of tuning states (at least one)
        objective: Objective configuration
        mode: Tuning mode (fixed_lc, tunable_c, switch)
        search: Search configuration
    """
    ports: List[PortTuningSpec] = field(default_factory=list)
    states: List[TuningState] = field(default_factory=lambda: [TuningState(name="default")])
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    mode: TuningMode = TuningMode.FIXED_LC
    top_candidates_per_port: int = 10
    max_joint_combinations: int = 10000
    timeout_seconds: float = 120.0

    def get_enabled_ports(self) -> List[PortTuningSpec]:
        """Get only enabled ports."""
        return [p for p in self.ports if p.enabled]

    def get_all_bands_mhz(self) -> List[List[float]]:
        """Aggregate all unique bands from enabled ports."""
        bands = set()
        for p in self.get_enabled_ports():
            for b in p.bands_mhz:
                bands.add((b[0], b[1]))
        return sorted([[b[0], b[1]] for b in bands])

    def to_dict(self) -> dict:
        return {
            'mode': self.mode.value,
            'ports': [
                {
                    'port_index': p.port_index,
                    'enabled': p.enabled,
                    'bands_mhz': p.bands_mhz,
                    'max_components': p.max_components,
                    'topology_constraints': p.topology_constraints,
                    'component_series': p.component_series,
                    'tunable_c_positions': p.tunable_c_positions,
                    'switch_branches': p.switch_branches,
                }
                for p in self.ports
            ],
            'states': [
                {
                    'name': s.name,
                    'active_bands': s.active_bands,
                    'variable_values': s.variable_values,
                    'switch_position': s.switch_position,
                }
                for s in self.states
            ],
            'objective': {
                'primary': self.objective.primary,
                'avg_weight': self.objective.avg_weight,
                'min_weight': self.objective.min_weight,
                'coupling_penalty': self.objective.coupling_penalty,
                'component_count_penalty': self.objective.component_count_penalty,
            },
            'search': {
                'top_candidates_per_port': self.top_candidates_per_port,
                'max_joint_combinations': self.max_joint_combinations,
                'timeout_seconds': self.timeout_seconds,
            },
        }


# ── Search configuration ────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """Search algorithm configuration."""
    top_candidates_per_port: int = 10
    max_joint_combinations: int = 10000
    timeout_seconds: float = 120.0
    num_band_points: int = 5
    beam_width: int = 20
    target_rl_db: float = 10.0
    max_iterations: int = 5
