"""
RF Matching Optimizer — Multi-strategy optimization engine.

Problem: Given an N-port DUT (SNP), find the optimal combination of:
1. Port termination states (open/short/load/component)
2. Matching topology (how components connect)
3. Component selections (which Murata parts)

to minimize |S11| at the target frequency.

Strategies:
1. Analytic pre-computation: For simple topologies, analytically compute
   required impedance match → ideal L/C values → nearest Murata parts.
2. Two-stage search: Coarse ideal search → fine real-component search.
3. Branch-and-bound with beam search for larger topologies.
4. Parallel evaluation of independent combinations.
"""

import os
import numpy as np
import logging
import sys
from project_paths import ARTIFACTS_DIR

_log_dir = str(ARTIFACTS_DIR / 'output')
os.makedirs(_log_dir, exist_ok=True)

logger = logging.getLogger("rf_matching.optimizer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
    _h.setLevel(logging.INFO)
    logger.addHandler(_h)
    _fh = logging.FileHandler(
        os.path.join(_log_dir, 'optimizer.log'),
        mode='w', encoding='utf-8'
    )
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
    _fh.setLevel(logging.DEBUG)
    logger.addHandler(_fh)

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math

from .touchstone import TouchstoneData, interpolate_s_matrix
from .network import (
    terminate_ports, s_to_z, z_to_s,
    compute_s11_after_matching
)
from .component_lib import ComponentLibrary, ComponentInfo
from .topology import Topology, TopologyElement, ConnectionType
from .efficiency_data import EfficiencyData


class PortState(Enum):
    OPEN = 'open'       # Γ = 1
    SHORT = 'short'     # Γ = -1
    LOAD = 'load'       # Γ = 0 (matched)
    COMPONENT = 'component'  # Connected to a matching component


# Reflection coefficients for standard terminations
TERMINATION_GAMMA = {
    PortState.OPEN: 1.0 + 0j,
    PortState.SHORT: -1.0 + 0j,
    PortState.LOAD: 0.0 + 0j,
}


@dataclass
class ComponentChoice:
    """A specific component selection for one matching element."""
    position: int
    component: ComponentInfo
    connection_type: str  # 'series', 'shunt', 'parallel'
    port: int
    port2: Optional[int] = None


@dataclass
class MatchingSolution:
    """Complete matching solution."""
    topology: Topology
    port_states: Dict[int, PortState]  # port index → state
    component_choices: List[ComponentChoice]
    s11_magnitude: float  # |S11|
    s11_complex: complex  # full complex S11
    frequency_hz: float
    input_impedance: complex  # Zin at input port
    vswr: float
    return_loss_db: float
    # Multi-band evaluation results
    band_efficiency: Optional[List[dict]] = None  # Per-band efficiency data
    avg_band_efficiency: Optional[float] = None  # Average efficiency across all bands
    min_band_efficiency: Optional[float] = None  # Worst-case efficiency across bands
    balanced_score: Optional[float] = None  # min/avg balanced score
    avg_total_efficiency: Optional[float] = None  # Average total efficiency (η_rad × (1-|S11|²))
    min_total_efficiency: Optional[float] = None  # Min total efficiency across bands

    @property
    def s11_db(self) -> float:
        """Return loss in dB (positive = good)."""
        return -20 * np.log10(max(self.s11_magnitude, 1e-15))

    def to_dict(self) -> dict:
        return {
            'topology': self.topology.name,
            'port_states': {str(k): v.value for k, v in self.port_states.items()},
            'components': [
                {
                    'position': c.position,
                    'part_number': c.component.part_number,
                    'connection_type': c.connection_type,
                    'port': c.port,
                    'nominal_value': c.component.nominal_value,
                    'nominal_unit': c.component.nominal_unit,
                }
                for c in self.component_choices
            ],
            's11_magnitude': self.s11_magnitude,
            's11_db': self.s11_db,
            's11_complex': str(self.s11_complex),
            'frequency_hz': self.frequency_hz,
            'input_impedance': str(self.input_impedance),
            'vswr': self.vswr,
            'return_loss_db': self.return_loss_db,
            'band_efficiency': self.band_efficiency,
            'avg_band_efficiency': self.avg_band_efficiency,
            'min_band_efficiency': self.min_band_efficiency,
            'balanced_score': self.balanced_score,
            'avg_total_efficiency': self.avg_total_efficiency,
            'min_total_efficiency': self.min_total_efficiency,
        }


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    target_frequency_hz: float = 64e6  # 64 MHz
    target_s11_db: float = -20.0  # Target return loss in dB
    max_components: int = 4  # Max matching components
    reference_impedance: float = 50.0  # Z0
    beam_width: int = 10  # Number of best solutions to keep per beam search step
    coarse_search_div: int = 20  # Divisions for coarse ideal-value search
    fine_search_tolerance: float = 0.3  # 30% around ideal value for fine search
    max_combinations_to_evaluate: int = 500000  # Hard limit
    progressive_candidates_per_type: int = 40
    fine_candidates_per_position: int = 10
    full_exhaustive_max_components: int = 2  # Fully enumerate practical L networks
    use_multiprocessing: bool = True
    max_workers: int = 4
    timeout_seconds: float = 60.0
    # Multi-band evaluation
    bands_mhz: Optional[List[List[float]]] = None  # [[start, end], ...] in MHz
    num_band_points: int = 5  # Points per band for evaluation
    # Radiation efficiency data for total efficiency optimization
    radiation_efficiency: Optional[EfficiencyData] = None
    # Min/avg balance control: 0.0 = pure min, 1.0 = pure average, 0.5 = equal weight
    min_avg_balance: float = 0.5


def _dedup_components_by_value(parts: List[ComponentInfo]) -> List[ComponentInfo]:
    """Return one valid component per nominal value, sorted by value."""
    seen = {}
    for part in parts:
        value = getattr(part, "nominal_value", None)
        if value is not None and value > 0.01 and value not in seen:
            seen[value] = part
    return [part for _, part in sorted(seen.items(), key=lambda x: x[0])]


def _sample_components_by_value(parts: List[ComponentInfo], max_count: int = 50) -> List[ComponentInfo]:
    """Sample by value percentiles and avoid hard endpoints when possible."""
    unique = _dedup_components_by_value(parts)
    if len(unique) <= max_count:
        return unique

    values = np.array([part.nominal_value for part in unique], dtype=float)
    if max_count > 3 and len(unique) > max_count + 2:
        percentiles = np.linspace(
            100.0 / (max_count + 1),
            100.0 - 100.0 / (max_count + 1),
            max_count,
        )
    else:
        percentiles = np.linspace(0.0, 100.0, max_count)

    selected = []
    used = set()
    for percentile in percentiles:
        target = float(np.percentile(values, percentile))
        idx = min(
            (i for i in range(len(unique)) if i not in used),
            key=lambda i: abs(values[i] - target),
        )
        used.add(idx)
        selected.append(unique[idx])
    return sorted(selected, key=lambda part: part.nominal_value)


def _component_reactance_ohms(component: ComponentInfo, freq_hz: float) -> Optional[float]:
    """Return ideal nominal reactance magnitude for feasibility pre-filtering."""
    value = getattr(component, "nominal_value", None)
    comp_type = getattr(component, "component_type", "")
    if value is None or value <= 0 or freq_hz <= 0:
        return None
    omega = 2.0 * math.pi * freq_hz
    if comp_type == "inductor":
        return omega * value * 1e-9
    if comp_type == "capacitor":
        return 1.0 / max(omega * value * 1e-12, 1e-30)
    return None


def _edge_value_penalty(value: float, values: List[float]) -> float:
    """Small ranking penalty for library endpoints; zero in the interior."""
    if value is None or len(values) < 4:
        return 0.0
    v_min, v_max = values[0], values[-1]
    if v_max <= v_min:
        return 0.0
    pos = (value - v_min) / (v_max - v_min)
    edge_dist = min(pos, 1.0 - pos)
    if edge_dist > 0.15:
        return 0.0
    return 0.015 * (1.0 - edge_dist / 0.15) ** 2


def _solution_rank_key(solution: MatchingSolution, library: ComponentLibrary) -> float:
    """Rank by match quality with a mild endpoint tie-breaker."""
    return solution.s11_magnitude + _choices_edge_penalty(solution.component_choices, library)


def _solution_signature(solution: MatchingSolution) -> Tuple:
    """Stable signature for duplicate solutions from analytic and exhaustive paths."""
    return (
        solution.topology.name,
        tuple(
            (
                choice.position,
                choice.connection_type,
                choice.port,
                choice.port2,
                choice.component.part_number,
                getattr(choice.component, "s2p_filename", ""),
                getattr(choice.component, "zip_path", ""),
                choice.component.nominal_value,
                choice.component.nominal_unit,
            )
            for choice in solution.component_choices
        ),
    )


def _choices_edge_penalty(choices: List[ComponentChoice], library: ComponentLibrary) -> float:
    """Compute the endpoint penalty for a partial or complete component list."""
    inductor_values = [c.nominal_value for c in _dedup_components_by_value(library.inductors)]
    capacitor_values = [c.nominal_value for c in _dedup_components_by_value(library.capacitors)]
    penalty = 0.0
    for choice in choices:
        comp = choice.component
        values = inductor_values if comp.component_type == "inductor" else capacitor_values
        penalty += _edge_value_penalty(comp.nominal_value, values)
    return penalty


class MatchingOptimizer:
    """Main optimizer for RF impedance matching."""

    def __init__(self,
                 dut_data: TouchstoneData,
                 component_library: ComponentLibrary,
                 config: OptimizerConfig = None):
        self.dut = dut_data
        self.library = component_library
        self.config = config or OptimizerConfig()

        # Cache: S-matrix at target frequency
        self._dut_s_at_freq = self.dut.get_s_matrix_interpolated(
            self.config.target_frequency_hz
        )
        self._dut_z_at_freq = s_to_z(self._dut_s_at_freq, self.config.reference_impedance)

        # Component cache: pre-load S-parameters at target frequency
        self._component_cache: Dict[str, np.ndarray] = {}

    def _get_component_s(self, comp: ComponentInfo) -> np.ndarray:
        """Get the 2x2 S-matrix of a component at target frequency (cached)."""
        cache_key = "|".join((
            str(getattr(comp, "part_number", "")),
            str(getattr(comp, "zip_path", "")),
            str(getattr(comp, "s2p_filename", "")),
            str(getattr(comp, "db_id", "")),
        ))
        if cache_key not in self._component_cache:
            if hasattr(comp, "get_s_matrix_at_freq"):
                matrix = comp.get_s_matrix_at_freq(self.config.target_frequency_hz)
            elif hasattr(comp, "db_id") and hasattr(self.library, "get_s_matrix_at_freq"):
                matrix = self.library.get_s_matrix_at_freq(
                    comp.db_id, self.config.target_frequency_hz / 1e6
                )
            else:
                raise TypeError(f"component {getattr(comp, 'part_number', '?')} cannot provide S parameters")
            if matrix is None:
                raise ValueError(f"component {getattr(comp, 'part_number', '?')} has no S parameters at the target frequency")
            self._component_cache[cache_key] = matrix
        return self._component_cache[cache_key]

    def _sample_feasible_components(
        self,
        parts: List[ComponentInfo],
        max_count: int = 50,
    ) -> List[ComponentInfo]:
        """
        Sample components after dropping values with impractical reactance.

        The window is relative to Z0 so it scales across 64 MHz and GHz cases:
        extremely tiny reactance behaves like a short, extremely huge reactance
        behaves like an open. Keep a broad fallback so sparse libraries still work.
        """
        z0 = self.config.reference_impedance
        min_x = 0.05 * z0
        max_x = 20.0 * z0

        unique = _dedup_components_by_value(parts)
        feasible = [
            comp for comp in unique
            if (lambda x: x is not None and min_x <= x <= max_x)(
                _component_reactance_ohms(comp, self.config.target_frequency_hz)
            )
        ]
        if len(feasible) < min(max_count, 8):
            loose_min_x = 0.01 * z0
            loose_max_x = 50.0 * z0
            feasible = [
                comp for comp in unique
                if (lambda x: x is not None and loose_min_x <= x <= loose_max_x)(
                    _component_reactance_ohms(comp, self.config.target_frequency_hz)
                )
            ]
        if len(feasible) < 4:
            feasible = unique

        return _sample_components_by_value(feasible, max_count)

    def _primary_components_by_value(self, parts: List[ComponentInfo]) -> List[ComponentInfo]:
        """
        Return the library's selected primary component for each nominal value.

        The component library is responsible for ordering same-value parts by
        precision/tolerance. The optimizer consumes that primary view so it
        avoids repeated same-value calculations without hiding topology states.
        """
        return _dedup_components_by_value(parts)

    def _candidate_components_for_element(
        self,
        elem: TopologyElement,
        exhaustive: bool = True,
    ) -> List[ComponentInfo]:
        """Build candidate list for one topology element."""
        if exhaustive:
            if elem.component_type == 'inductor':
                return self._primary_components_by_value(self.library.inductors)
            if elem.component_type == 'capacitor':
                return self._primary_components_by_value(self.library.capacitors)
            return (
                self._primary_components_by_value(self.library.inductors) +
                self._primary_components_by_value(self.library.capacitors)
            )

        if elem.component_type == 'inductor':
            return self._sample_feasible_components(self.library.inductors, 80)
        if elem.component_type == 'capacitor':
            return self._sample_feasible_components(self.library.capacitors, 80)
        return (
            self._sample_feasible_components(self.library.inductors, 50) +
            self._sample_feasible_components(self.library.capacitors, 50)
        )

    def _apply_port_terminations(self, S: np.ndarray,
                                  port_states: Dict[int, PortState]) -> np.ndarray:
        """Apply port terminations (open/short/load) to reduce the S-matrix."""
        terminations = {}
        for port, state in port_states.items():
            if state in TERMINATION_GAMMA:
                terminations[port] = TERMINATION_GAMMA[state]

        if terminations:
            return terminate_ports(S, terminations)
        return S

    def _evaluate_solution(self,
                           S_dut_terminated: np.ndarray,
                           component_choices: List[ComponentChoice],
                           output_port: int = 0) -> complex:
        """
        Evaluate S11 for a specific set of component choices.
        Returns complex S11 at output_port.
        """
        S = S_dut_terminated.copy()

        for choice in component_choices:
            comp_s = self._get_component_s(choice.component)

            try:
                if choice.connection_type == 'series':
                    from .network import _embed_series_on_port
                    S = _embed_series_on_port(S, comp_s, choice.port)
                elif choice.connection_type == 'shunt':
                    from .network import _embed_shunt_to_ground
                    S = _embed_shunt_to_ground(S, comp_s, choice.port)
                elif choice.connection_type == 'parallel':
                    from .network import connect_2port_to_multiport
                    S = connect_2port_to_multiport(S, comp_s, choice.port, choice.port2)
            except np.linalg.LinAlgError:
                return complex(1.0, 0.0)  # Bad combination → total reflection

        return S[output_port, output_port]

    def _analytic_l_match(self, Z_load: complex, Z0: float = 50.0) -> List[Tuple[float, float, str, str]]:
        """
        Analytically compute ideal L/C values for L-network matching.

        For Z_load = R + jX:
        - If R < Z0: Use shunt element first (on load side), then series
          Series element Xs = ±√(R*(Z0-R)) - X
          Shunt element Bp = ±√((Z0-R)/R) / Z0
        - If R > Z0: Use series element first, then shunt
          Series element Xs = (X ± √(R*Z0 - R^2 + X^2*R/Z0)) / (R/Z0)
          Shunt element Bp = ±√(R*Z0 - R^2 + X^2*R/Z0) / (R*Z0 - R^2 + X^2)

        Returns list of (L_nH, C_pF, series_type, shunt_type) tuples.
        """
        R = Z_load.real
        X = Z_load.imag
        omega = 2 * np.pi * self.config.target_frequency_hz

        solutions = []

        # Handle both cases: R < Z0 and R > Z0
        for case in ['low_r', 'high_r']:
            if case == 'low_r' and R < Z0:
                # Shunt-first (on load side)
                # Q = sqrt(Z0/R - 1)
                if R <= 0:
                    continue
                Q = math.sqrt(Z0 / R - 1.0)

                for sign_b in [1, -1]:  # Shunt susceptance sign
                    B = sign_b * Q / Z0
                    Xs = -X + sign_b * math.sqrt(R * (Z0 - R))

                    for sign_x in [1, -1]:
                        Xs_val = -X + sign_x * math.sqrt(R * (Z0 - R))
                        B_val = sign_b * Q / Z0

                        # Convert to L/C
                        # Series: X > 0 → inductor, X < 0 → capacitor
                        # Shunt: B > 0 → capacitor, B < 0 → inductor
                        if Xs_val > 0:
                            L_nH = (Xs_val / omega) * 1e9
                            solutions.append((L_nH, None, 'L', 'C' if B_val > 0 else 'L'))
                        else:
                            C_pF = (-1.0 / (omega * Xs_val)) * 1e12
                            solutions.append((None, C_pF, 'C', 'C' if B_val > 0 else 'L'))

            elif case == 'high_r' and R >= Z0:
                # Series-first
                if R <= 0:
                    continue
                for sign_b in [1, -1]:
                    disc = R * Z0 - R * R + X * X * R / Z0
                    if disc < 0:
                        continue

                    for sign_x in [1, -1]:
                        Xs = (X + sign_x * math.sqrt(disc)) / (R / Z0)

                        # Shunt B = ±√(...)/(...)
                        denom = R * Z0 - R * R + X * X
                        if abs(denom) < 1e-15:
                            continue
                        B = sign_b * math.sqrt(R * Z0 - R * R + X * X * R / Z0) / denom

                        if Xs > 0:
                            L_nH = (Xs / omega) * 1e9
                            solutions.append((L_nH, None, 'L', 'C' if B > 0 else 'L'))
                        else:
                            C_pF = (-1.0 / (omega * Xs)) * 1e12
                            solutions.append((None, C_pF, 'C', 'C' if B > 0 else 'L'))

        return solutions

    def _analytic_pi_match(self, Z_load: complex, Z0: float = 50.0) -> List[dict]:
        """
        Analytic Pi-network matching: shunt - series - shunt.

        Returns list of dicts with ideal component values.
        """
        # Pi network is two L-networks back-to-back
        # First L: load → virtual R_intermediate → Z0
        # R_intermediate < min(R_load, Z0)

        omega = 2 * np.pi * self.config.target_frequency_hz
        R_load = Z_load.real
        X_load = Z_load.imag

        solutions = []

        # Choose intermediate resistance: Q determines bandwidth
        for Q_factor in [0.5, 1.0, 2.0, 3.0, 5.0]:
            R_virt = min(R_load, Z0) / (1 + Q_factor * Q_factor)
            if R_virt <= 0 or R_virt >= min(R_load, Z0):
                continue

            # First L: Z_load → R_virt
            l1_solutions = self._analytic_l_match(Z_load, R_virt)
            # Second L: R_virt → Z0
            l2_solutions = self._analytic_l_match(complex(R_virt, 0), Z0)

            for l1 in l1_solutions[:2]:
                for l2 in l2_solutions[:2]:
                    sol = {
                        'shunt1_type': l1[3],  # shunt element of first L
                        'series_type': l1[2] if l1[2] == 'L' else l2[2],
                        'shunt2_type': l2[3],
                    }
                    solutions.append(sol)

        return solutions

    def optimize_l_network(self,
                           port_states: Dict[int, PortState],
                           input_port: int = 0) -> List[MatchingSolution]:
        """
        Optimize L-network matching (2 components).
        Uses analytic formulas to find ideal values, then searches Murata parts.
        """
        # Terminate non-input ports
        S_terminated = self._apply_port_terminations(
            self._dut_s_at_freq.copy(), port_states
        )

        if S_terminated.shape[0] <= 1:
            # Only input port remains
            Z_load_complex = self.config.reference_impedance * (
                1 + S_terminated[0, 0]) / (1 - S_terminated[0, 0])
        else:
            # Map input_port to local index in the reduced S-matrix
            n_orig = self._dut_s_at_freq.shape[0]
            terminated_ports = {p for p in port_states}
            remaining = sorted([p for p in range(n_orig) if p not in terminated_ports])
            local_idx = remaining.index(input_port) if input_port in remaining else 0
            Z = s_to_z(S_terminated, self.config.reference_impedance)
            Z_load_complex = Z[local_idx, local_idx]

        # Analytic solutions
        analytic_sols = self._analytic_l_match(Z_load_complex)

        solutions = []

        from .topology import get_standard_topologies

        # For L-network, try all 4 variants
        l_topologies = [t for t in get_standard_topologies()
                       if t.num_components == 2 and 'L-Network' in t.name]

        for topo in l_topologies:
            for sol in analytic_sols:
                L_target, C_target, series_type, shunt_type = sol

                # Find nearest Murata parts
                if L_target is not None and L_target > 0:
                    nearest_l = self.library.find_nearest_inductor(L_target)
                    nearby_l = self.library.get_inductors_near(L_target, 1.0)  # ±100%
                else:
                    nearest_l = None
                    nearby_l = []

                if C_target is not None and C_target > 0:
                    nearest_c = self.library.find_nearest_capacitor(C_target)
                    nearby_c = self.library.get_capacitors_near(C_target, 1.0)
                else:
                    nearest_c = None
                    nearby_c = []

                # Build candidate component sets — deduplicate by nominal value
                def _dedup_by_value(parts, limit=30):
                    """Pick one representative per unique nominal value."""
                    return self._sample_feasible_components(parts, limit)

                l_candidates = _dedup_by_value(nearby_l) if nearby_l else ([nearest_l] if nearest_l else [])
                c_candidates = _dedup_by_value(nearby_c) if nearby_c else ([nearest_c] if nearest_c else [])

                if not l_candidates:
                    l_candidates = self._sample_feasible_components(self.library.inductors, 50)
                if not c_candidates:
                    c_candidates = self._sample_feasible_components(self.library.capacitors, 50)

                # Limit candidates
                l_candidates = l_candidates[:30]
                c_candidates = c_candidates[:30]

                # Evaluate combinations
                for l_comp in l_candidates:
                    for c_comp in c_candidates:
                        # Determine component order based on topology
                        choices = []
                        for elem in topo.elements:
                            if elem.component_type == 'inductor':
                                choices.append(ComponentChoice(
                                    position=elem.position,
                                    component=l_comp,
                                    connection_type=elem.connection_type.value,
                                    port=elem.port,
                                ))
                            else:
                                choices.append(ComponentChoice(
                                    position=elem.position,
                                    component=c_comp,
                                    connection_type=elem.connection_type.value,
                                    port=elem.port,
                                ))

                        s11 = self._evaluate_solution(S_terminated, choices)
                        mag = abs(s11)

                        if mag < 1.0:  # Valid solution
                            solutions.append(MatchingSolution(
                                topology=topo,
                                port_states=port_states,
                                component_choices=choices,
                                s11_magnitude=mag,
                                s11_complex=s11,
                                frequency_hz=self.config.target_frequency_hz,
                                input_impedance=self.config.reference_impedance * (1 + s11) / (1 - s11),
                                vswr=(1 + mag) / (1 - mag) if mag < 1 else float('inf'),
                                return_loss_db=-20 * np.log10(max(mag, 1e-15)),
                            ))

        solutions.sort(key=lambda s: _solution_rank_key(s, self.library))
        # Deduplicate by component values (not part numbers)
        seen_values = set()
        unique = []
        for s in solutions:
            key = (s.topology.name, tuple(
                (c.connection_type, c.component.nominal_value, c.component.nominal_unit)
                for c in s.component_choices
            ))
            if key not in seen_values:
                seen_values.add(key)
                unique.append(s)
        return unique[:20]

    def optimize_progressive(self,
                              topology: Topology,
                              port_states: Dict[int, PortState],
                              input_port: int = 0) -> List[MatchingSolution]:
        """
        Progressive refinement optimizer for handling large component counts.
        
        Stage 1: Coarse grid using only unique component values (~66 inductor vals, ~182 cap vals)
        Stage 2: For the best coarse solutions, fine-search with exact Murata parts
        Stage 3: Return best results
        
        This reduces search from K^N (Murata parts) to V^N (unique values) for coarse stage.
        For 5 components: 100^5=10B → 66^5=1.2B (still large) → use beam search on unique values.
        """
        S_terminated = self._apply_port_terminations(
            self._dut_s_at_freq.copy(), port_states
        )
        
        n_comp = topology.num_components
        
        # Stage 1: Build candidate lists using unique values only
        candidates_per_position = []
        for elem in topology.elements:
            if elem.component_type == 'inductor':
                comps = self._sample_feasible_components(
                    self.library.inductors, self.config.progressive_candidates_per_type
                )
            elif elem.component_type == 'capacitor':
                comps = self._sample_feasible_components(
                    self.library.capacitors, self.config.progressive_candidates_per_type
                )
            else:
                per_kind = max(4, self.config.progressive_candidates_per_type // 2)
                comps = (
                    self._sample_feasible_components(self.library.inductors, per_kind) +
                    self._sample_feasible_components(self.library.capacitors, per_kind)
                )
            
            candidates_per_position.append(comps)
        
        # Stage 1: Beam search on unique values
        beam_width = max(self.config.beam_width * 2, 20)  # Wider beam for coarse
        beam = [([], S_terminated.copy(), 1.0)]
        
        for pos_idx, candidates in enumerate(candidates_per_position):
            elem = topology.elements[pos_idx]
            new_beam = []
            
            for partial_choices, current_S, current_mag in beam:
                for comp in candidates:
                    choice = ComponentChoice(
                        position=pos_idx,
                        component=comp,
                        connection_type=elem.connection_type.value,
                        port=elem.port,
                        port2=elem.port2,
                    )
                    new_choices = partial_choices + [choice]
                    comp_s = self._get_component_s(comp)
                    
                    try:
                        if elem.connection_type == ConnectionType.SERIES:
                            from .network import _embed_series_on_port
                            new_S = _embed_series_on_port(current_S, comp_s, elem.port)
                        elif elem.connection_type == ConnectionType.SHUNT:
                            from .network import _embed_shunt_to_ground
                            new_S = _embed_shunt_to_ground(current_S, comp_s, elem.port)
                        else:
                            from .network import connect_2port_to_multiport
                            new_S = connect_2port_to_multiport(current_S, comp_s, elem.port, elem.port2)
                        
                        s11 = new_S[0, 0]
                        mag = abs(s11)
                        new_beam.append((new_choices, new_S, mag))
                    except np.linalg.LinAlgError:
                        continue
            
            # Keep top beam_width
            new_beam.sort(key=lambda x: x[2] + _choices_edge_penalty(x[0], self.library))
            beam = new_beam[:beam_width]
        
        # Stage 2: Fine refinement around best coarse solutions
        top_coarse = beam[:max(3, self.config.beam_width // 2)]
        solutions = []
        
        for coarse_choices, _, _ in top_coarse:
            # For each position, get nearby Murata parts from the best coarse value
            fine_candidates = []
            for pos_idx, coarse_choice in enumerate(coarse_choices):
                comp = coarse_choice.component
                if comp.component_type == 'inductor':
                    nearby = self.library.get_inductors_near(comp.nominal_value, 0.5)
                else:
                    nearby = self.library.get_capacitors_near(comp.nominal_value, 0.5)
                if not nearby:
                    nearby = [comp]
                fine_candidates.append(nearby[:self.config.fine_candidates_per_position])
            
            # Exhaustive search on narrowed candidates
            n_fine = len(fine_candidates)
            total_combos = 1
            for fc in fine_candidates:
                total_combos *= len(fc)
            
            if total_combos <= 10000:
                for combo in itertools.product(*fine_candidates):
                    choices = [
                        ComponentChoice(
                            position=topology.elements[i].position,
                            component=c,
                            connection_type=topology.elements[i].connection_type.value,
                            port=topology.elements[i].port,
                            port2=topology.elements[i].port2,
                        )
                        for i, c in enumerate(combo)
                    ]
                    s11 = self._evaluate_solution(S_terminated, choices)
                    mag = abs(s11)
                    if mag < 1.0:
                        solutions.append(self._make_solution(topology, port_states, choices, s11))
            else:
                # Beam search on fine candidates (with choice tracking)
                fine_beam = [(coarse_choices[:], S_terminated.copy(), 1.0)]
                for pos_idx, fine_comps in enumerate(fine_candidates):
                    elem = topology.elements[pos_idx]
                    new_fine_beam = []
                    for prev_choices, current_S, current_mag in fine_beam:
                        for comp in fine_comps:
                            choice = ComponentChoice(
                                position=pos_idx, component=comp,
                                connection_type=elem.connection_type.value,
                                port=elem.port, port2=elem.port2,
                            )
                            new_choices = prev_choices[:]
                            new_choices[pos_idx] = choice
                            comp_s = self._get_component_s(comp)
                            try:
                                if elem.connection_type == ConnectionType.SERIES:
                                    from .network import _embed_series_on_port
                                    new_S = _embed_series_on_port(current_S, comp_s, elem.port)
                                elif elem.connection_type == ConnectionType.SHUNT:
                                    from .network import _embed_shunt_to_ground
                                    new_S = _embed_shunt_to_ground(current_S, comp_s, elem.port)
                                else:
                                    from .network import connect_2port_to_multiport
                                    new_S = connect_2port_to_multiport(current_S, comp_s, elem.port, elem.port2)
                                new_fine_beam.append((new_choices, new_S, abs(new_S[0, 0])))
                            except np.linalg.LinAlgError:
                                continue
                    new_fine_beam.sort(key=lambda x: x[2] + _choices_edge_penalty(x[0], self.library))
                    fine_beam = new_fine_beam[:self.config.beam_width]
                
                # Build solutions from fine beam results with correct choices
                for fine_choices, best_S, best_mag in fine_beam[:5]:
                    if best_mag < 1.0:
                        solutions.append(MatchingSolution(
                            topology=topology, port_states=port_states,
                            component_choices=fine_choices,
                            s11_magnitude=best_mag,
                            s11_complex=best_S[0, 0],
                            frequency_hz=self.config.target_frequency_hz,
                            input_impedance=self.config.reference_impedance * (1 + best_S[0, 0]) / (1 - best_S[0, 0]),
                            vswr=(1 + best_mag) / (1 - best_mag) if best_mag < 1 else float('inf'),
                            return_loss_db=-20 * np.log10(max(best_mag, 1e-15)),
                        ))
        
        solutions.sort(key=lambda s: _solution_rank_key(s, self.library))
        return solutions[:self.config.beam_width]
    
    def _make_solution(self, topology, port_states, choices, s11):
        mag = abs(s11)
        return MatchingSolution(
            topology=topology,
            port_states=port_states,
            component_choices=choices,
            s11_magnitude=mag,
            s11_complex=s11,
            frequency_hz=self.config.target_frequency_hz,
            input_impedance=self.config.reference_impedance * (1 + s11) / (1 - s11),
            vswr=(1 + mag) / (1 - mag) if mag < 1 else float('inf'),
            return_loss_db=-20 * np.log10(max(mag, 1e-15)),
        )

    def evaluate_solution_across_bands(self, solution: MatchingSolution) -> MatchingSolution:
        """
        Evaluate a matching solution across configured frequency bands.
        
        Computes:
        - Mismatch efficiency: 1 - |S11|^2
        - Total efficiency: η_rad(f) × (1 - |S11|^2) if radiation efficiency data is available
        - Min/avg balanced score for ranking
        
        Modifies the solution in-place and returns it.
        """
        if not self.config.bands_mhz:
            return solution

        band_results = []
        all_mismatch_effs = []    # 1 - |S11|²
        all_total_effs = []       # η_rad × (1 - |S11|²)

        for band in self.config.bands_mhz:
            f_start_hz = band[0] * 1e6
            f_end_hz = band[1] * 1e6
            freqs_hz = np.linspace(f_start_hz, f_end_hz, self.config.num_band_points)

            efficiencies = []
            total_efficiencies = []
            s11_dbs = []
            freqs_ghz = []

            for freq_hz in freqs_hz:
                try:
                    S_dut = self.dut.get_s_matrix_interpolated(freq_hz)
                    S = S_dut.copy()

                    # Apply port terminations
                    terminations = {}
                    for port, pstate in solution.port_states.items():
                        if pstate in [PortState.OPEN, PortState.SHORT, PortState.LOAD]:
                            terminations[port] = TERMINATION_GAMMA[pstate]
                    if terminations:
                        from .network import terminate_ports
                        S = terminate_ports(S, terminations)

                    # Apply matching components
                    for choice in solution.component_choices:
                        comp_s = choice.component.get_s_matrix_at_freq(freq_hz)
                        if choice.connection_type == 'series':
                            from .network import _embed_series_on_port
                            S = _embed_series_on_port(S, comp_s, choice.port)
                        elif choice.connection_type == 'shunt':
                            from .network import _embed_shunt_to_ground
                            S = _embed_shunt_to_ground(S, comp_s, choice.port)
                        elif choice.connection_type == 'parallel':
                            from .network import connect_2port_to_multiport
                            S = connect_2port_to_multiport(S, comp_s, choice.port, choice.port2)

                    if S.shape[0] > 0:
                        s11 = S[0, 0]
                        mismatch_eff = 1.0 - abs(s11) ** 2
                        efficiencies.append(float(mismatch_eff))
                        s11_dbs.append(float(-20 * np.log10(max(abs(s11), 1e-15))))

                        # Total efficiency = η_rad × mismatch_eff
                        if self.config.radiation_efficiency is not None:
                            eta_rad = self.config.radiation_efficiency.get_efficiency_at(freq_hz)
                            total_eff = float(eta_rad * mismatch_eff)
                        else:
                            total_eff = float(mismatch_eff)
                        total_efficiencies.append(total_eff)
                    else:
                        efficiencies.append(0.0)
                        total_efficiencies.append(0.0)
                        s11_dbs.append(0.0)
                except Exception:
                    efficiencies.append(0.0)
                    total_efficiencies.append(0.0)
                    s11_dbs.append(0.0)

                freqs_ghz.append(freq_hz / 1e9)

            avg_eff = float(np.mean(efficiencies)) if efficiencies else 0.0
            avg_total = float(np.mean(total_efficiencies)) if total_efficiencies else 0.0
            min_total = float(np.min(total_efficiencies)) if total_efficiencies else 0.0
            all_mismatch_effs.extend(efficiencies)
            all_total_effs.extend(total_efficiencies)

            band_results.append({
                'band_mhz': band,
                'avg_efficiency': avg_eff,
                'avg_total_efficiency': avg_total,
                'min_total_efficiency': min_total,
                'efficiencies': efficiencies,
                'total_efficiencies': total_efficiencies,
                's11_dbs': s11_dbs,
                'frequencies_ghz': freqs_ghz,
            })

        # --- Compute aggregate metrics ---
        use_total = self.config.radiation_efficiency is not None
        score_effs = all_total_effs if use_total else all_mismatch_effs

        avg_score = float(np.mean(score_effs)) if score_effs else 0.0
        min_score = float(np.min(score_effs)) if score_effs else 0.0

        # Min/avg balance: balanced = balance * avg + (1 - balance) * min
        # balance=1.0 → pure average, balance=0.0 → pure worst-case
        balance = self.config.min_avg_balance
        balanced = balance * avg_score + (1.0 - balance) * min_score

        solution.band_efficiency = band_results
        solution.avg_band_efficiency = float(np.mean(all_mismatch_effs)) if all_mismatch_effs else 0.0
        solution.min_band_efficiency = float(np.min(all_mismatch_effs)) if all_mismatch_effs else 0.0
        solution.avg_total_efficiency = float(np.mean(all_total_effs)) if all_total_effs else 0.0
        solution.min_total_efficiency = float(np.min(all_total_effs)) if all_total_effs else 0.0
        solution.balanced_score = balanced

        return solution

    def evaluate_solutions_across_bands(self, solutions: List[MatchingSolution]) -> List[MatchingSolution]:
        """Evaluate all solutions across bands and re-sort by balanced score.
        
        The sorting key depends on configuration:
        - If radiation_efficiency data provided: sort by total efficiency balanced score
        - If min_avg_balance = 1.0: sort by average efficiency (pure average)
        - If min_avg_balance = 0.0: sort by min efficiency (pure worst-case)
        - Otherwise: sort by balanced score (weighted combination)
        """
        if not self.config.bands_mhz:
            return solutions
        for sol in solutions:
            self.evaluate_solution_across_bands(sol)

        # Sort by balanced_score (higher is better)
        solutions.sort(key=lambda s: s.balanced_score or 0.0, reverse=True)
        return solutions

    def optimize_with_topology(self,
                                topology: Topology,
                                port_states: Dict[int, PortState],
                                input_port: int = 0) -> List[MatchingSolution]:
        """
        Optimize matching using a specific topology.
        
        Uses progressive refinement for >3 components to handle combinatorial explosion.
        """
        n_comp = topology.num_components

        return self._exhaustive_search(
            self._apply_port_terminations(self._dut_s_at_freq.copy(), port_states),
            topology, port_states, input_port
        )

    def _exhaustive_search(self,
                            S_dut: np.ndarray,
                            topology: Topology,
                            port_states: Dict[int, PortState],
                            input_port: int = 0) -> List[MatchingSolution]:
        """Exhaustive search for small topologies (≤3 components)."""
        n_comp = topology.num_components
        solutions = []

        logger.info("Exhaustive search: topology=%s, n_comp=%d", topology.name, n_comp)

        candidates_per_position = []
        for elem in topology.elements:
            comps = self._candidate_components_for_element(elem, exhaustive=True)
            candidates_per_position.append(comps)

        # Log candidate details
        for i, (elem, comps) in enumerate(zip(topology.elements, candidates_per_position)):
            vals = sorted(set(c.nominal_value for c in comps))
            logger.info("  Position %d (%s, %s): %d candidates, values: %s",
                       i, elem.connection_type.value, elem.component_type,
                       len(comps), [round(v, 2) for v in vals[:10]])

        # Iterate through combinations
        total = 1
        for c in candidates_per_position:
            total *= len(c)

        logger.info("  Total combinations: %d", total)

        if total > self.config.max_combinations_to_evaluate:
            logger.warning(
                "  Full enumeration requires %d combinations, above configured limit %d. "
                "Use progressive search for this topology instead of truncating candidates.",
                total, self.config.max_combinations_to_evaluate,
            )
            return self.optimize_progressive(topology, port_states, input_port)

        count = 0
        best_mag = 1.0

        for combo in itertools.product(*candidates_per_position):
            count += 1

            choices = [
                ComponentChoice(
                    position=elem.position,
                    component=comp,
                    connection_type=elem.connection_type.value,
                    port=elem.port,
                    port2=elem.port2,
                )
                for elem, comp in zip(topology.elements, combo)
            ]

            s11 = self._evaluate_solution(S_dut, choices)
            mag = abs(s11)

            if mag < 1.0:
                solutions.append(MatchingSolution(
                    topology=topology,
                    port_states=port_states,
                    component_choices=choices,
                    s11_magnitude=mag,
                    s11_complex=s11,
                    frequency_hz=self.config.target_frequency_hz,
                    input_impedance=self.config.reference_impedance * (1 + s11) / (1 - s11),
                    vswr=(1 + mag) / (1 - mag) if mag < 1 else float('inf'),
                    return_loss_db=-20 * np.log10(max(mag, 1e-15)),
                ))

                if mag < best_mag:
                    best_mag = mag

        solutions.sort(key=lambda s: _solution_rank_key(s, self.library))
        logger.info("  Exhaustive done: %d sols, best=%.4f, evaluated=%d",
                   len(solutions), best_mag, count)
        if solutions:
            best = solutions[0]
            cs = ', '.join('%.1f%s' % (c.component.nominal_value, c.component.nominal_unit) for c in best.component_choices)
            logger.info("  Best: RL=%.1fdB %s [%s]", best.s11_db, best.topology.name, cs)
        return solutions[:self.config.beam_width]

    def _beam_search(self,
                     S_dut: np.ndarray,
                     topology: Topology,
                     port_states: Dict[int, PortState]) -> List[MatchingSolution]:
        """
        Beam search for larger topologies (>3 components).
        At each step, keep only the beam_width best partial solutions.
        """
        beam_width = self.config.beam_width
        solutions = []
        beam = [([], S_dut.copy(), 1.0)]  # (partial_choices, current_S, current_best_s11_mag)

        for elem in topology.elements:
            new_beam = []

            candidates = self._candidate_components_for_element(elem, exhaustive=False)

            for partial_choices, current_S, current_mag in beam:
                for comp in candidates:
                    choice = ComponentChoice(
                        position=elem.position,
                        component=comp,
                        connection_type=elem.connection_type.value,
                        port=elem.port,
                        port2=elem.port2,
                    )
                    new_choices = partial_choices + [choice]
                    comp_s = self._get_component_s(comp)

                    try:
                        if elem.connection_type == ConnectionType.SERIES:
                            from .network import _embed_series_on_port
                            new_S = _embed_series_on_port(current_S, comp_s, elem.port)
                        elif elem.connection_type == ConnectionType.SHUNT:
                            from .network import _embed_shunt_to_ground
                            new_S = _embed_shunt_to_ground(current_S, comp_s, elem.port)
                        else:
                            from .network import connect_2port_to_multiport
                            new_S = connect_2port_to_multiport(current_S, comp_s, elem.port, elem.port2)

                        s11 = new_S[0, 0]
                        mag = abs(s11)
                        new_beam.append((new_choices, new_S, mag))
                    except np.linalg.LinAlgError:
                        continue

            # Keep best beam_width
            new_beam.sort(key=lambda x: x[2] + sum(
                _edge_value_penalty(
                    choice.component.nominal_value,
                    [p.nominal_value for p in _dedup_components_by_value(
                        self.library.inductors
                        if choice.component.component_type == 'inductor'
                        else self.library.capacitors
                    )],
                )
                for choice in x[0]
            ))
            beam = new_beam[:beam_width]

        # Final solutions from beam
        for choices, S_final, mag in beam:
            s11 = S_final[0, 0]
            if mag < 1.0:
                solutions.append(MatchingSolution(
                    topology=topology,
                    port_states=port_states,
                    component_choices=choices,
                    s11_magnitude=mag,
                    s11_complex=s11,
                    frequency_hz=self.config.target_frequency_hz,
                    input_impedance=self.config.reference_impedance * (1 + s11) / (1 - s11),
                    vswr=(1 + mag) / (1 - mag) if mag < 1 else float('inf'),
                    return_loss_db=-20 * np.log10(max(mag, 1e-15)),
                ))

        solutions.sort(key=lambda s: _solution_rank_key(s, self.library))
        return solutions[:beam_width]

    def optimize_full(self,
                      port_states: Dict[int, PortState],
                      topologies: List[Topology] = None,
                      input_port: int = 0) -> List[MatchingSolution]:
        """
        Full optimization: try all specified topologies with given port states.
        Returns the best solutions across all topologies.
        """
        if topologies is None:
            from .topology import get_standard_topologies
            topologies = get_standard_topologies()
            # Filter by max components
            topologies = [t for t in topologies
                         if t.num_components <= self.config.max_components]

        all_solutions = []
        start_time = time.time()

        # Also try analytic L-network optimizer (guided by impedance)
        try:
            l_sols = self.optimize_l_network(port_states, input_port)
            all_solutions.extend(l_sols)
        except Exception:
            pass

        for topo in topologies:
            if time.time() - start_time > self.config.timeout_seconds:
                break

            sols = self.optimize_with_topology(topo, port_states, input_port)
            all_solutions.extend(sols)

        # Sort best first
        all_solutions.sort(key=lambda s: _solution_rank_key(s, self.library))
        deduped = []
        seen = set()
        for sol in all_solutions:
            sig = _solution_signature(sol)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(sol)
        top_solutions = deduped[:50]

        # Re-evaluate across bands if configured
        if self.config.bands_mhz:
            top_solutions = self.evaluate_solutions_across_bands(top_solutions)

        return top_solutions

    def optimize_all_port_configs(self,
                                   topologies: List[Topology] = None,
                                   input_port: int = 0) -> List[MatchingSolution]:
        """
        For multi-port DUTs: iterate over all possible port termination states
        for non-input ports. Each non-input port can be: open, short, or load.

        Returns best solutions across all configurations.
        """
        n_ports = self.dut.num_ports
        non_input_ports = [p for p in range(n_ports) if p != input_port]

        all_solutions = []
        start_time = time.time()

        # Generate all port state combinations
        # For each non-input port: OPEN, SHORT, LOAD (3 options)
        # For many ports, this grows as 3^(N-1). We limit to a few.
        max_configs = 3 ** min(len(non_input_ports), 4)  # Cap at 81 configs

        for state_combo in itertools.product(
            [PortState.OPEN, PortState.SHORT, PortState.LOAD],
            repeat=len(non_input_ports)
        ):
            if time.time() - start_time > self.config.timeout_seconds:
                break

            port_states = {}
            for port, state in zip(non_input_ports, state_combo):
                port_states[port] = state
            port_states[input_port] = PortState.COMPONENT  # Input has matching network

            sols = self.optimize_full(port_states, topologies, input_port)
            all_solutions.extend(sols)

        all_solutions.sort(key=lambda s: _solution_rank_key(s, self.library))
        top_solutions = all_solutions[:50]

        # Re-evaluate across bands if configured
        if self.config.bands_mhz:
            top_solutions = self.evaluate_solutions_across_bands(top_solutions)

        return top_solutions
