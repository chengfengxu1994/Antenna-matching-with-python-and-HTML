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
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import math
import numpy as np
import time
import logging
import re

from rfmatch_core import (
    Band as CoreBand,
    FrequencyConfiguration as CoreFrequencyConfiguration,
    IsolationTarget,
    MDIFModel,
    ModelPlacement,
    LumpedLossModel,
    LumpedModel,
    LazyComponentSpec,
    MatchingOptimizer as CoreMatchingOptimizer,
    MeasuredComponentOptimizer,
    Objective as CoreObjective,
    Problem as CoreProblem,
    S2PModel,
    SearchConfig as CoreSearchConfig,
    ComponentSpec,
    MeasuredSearchConfig,
    TunableMeasuredComponentOptimizer,
    TunableProblem as CoreTunableProblem,
    evaluate_isolation_targets,
    evaluate_tunable_physical,
    load_mdif,
    InputReactance,
    InputModelPlacement,
    SeriesReactance,
    SwitchSearchConfig,
    SwitchMeasuredComponentOptimizer,
    SwitchTunableOptimizer,
    SwitchTunableProblem,
    evaluate_loaded_switch_physical_power,
    evaluate_loaded_switch_power,
    load_component_model,
    preload_switch_state,
    build_model_circuit_topology,
    evaluate_physical_problem,
    monte_carlo_yield,
    monte_carlo_switch_yield,
    monte_carlo_tunable_yield,
    tolerance_summary,
    ToleranceModel,
    TransmissionLineModel,
    TransmissionLineStubModel,
    TransmissionLineOptimizer,
    LineSearchConfig,
    OptimizationCancelled,
    MicrostripDesignRules,
    MicrostripLineModel,
    MicrostripStubModel,
    PCBSubstrate,
    YieldCriteria,
    flip_s2p_ports,
    renormalize_s_parameters,
    deembed_s2p,
)

from .touchstone import TouchstoneData
from .component_lib import component_metadata, component_series_name
from .calibration_evidence import (
    multiport_calibration_reference,
    single_port_calibration_reference,
)
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
from .grid_s2p_narrowband import optimize_narrowband_grid, GridChoice

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

    @classmethod
    def from_dict(cls, value: dict) -> "PerPortTuningMetrics":
        fields = {
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value
        }
        return cls(**fields)


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
    # Runtime-only physical context for state-dependent manufacturing yield.
    yield_context: Optional[dict] = field(default=None, repr=False)

    # Sweep data (at selected solution)
    sweep_freqs_hz: List[float] = field(default_factory=list)

    # Metadata
    total_time_s: float = 0.0
    num_solutions_found: int = 0
    solution_index: int = 0          # which candidate this is (0 = best)
    efficiency_basis: str = "total_efficiency_estimate"
    isolation_targets: List[dict] = field(default_factory=list)
    isolation_penalty_db: float = 0.0
    isolation_constraints_passed: bool = True
    directed_isolation_db: Dict[str, dict] = field(default_factory=dict)
    tunable_states: Dict[str, str] = field(default_factory=dict)
    frequency_configurations: List[dict] = field(default_factory=list)
    maximum_power_balance_error: float = 0.0
    search_diagnostics: Optional[dict] = None

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
            'efficiency_basis': self.efficiency_basis,
            'isolation_targets': self.isolation_targets,
            'isolation_penalty_db': float(self.isolation_penalty_db),
            'isolation_constraints_passed': bool(self.isolation_constraints_passed),
            'directed_isolation_db': self.directed_isolation_db,
            'tunable_states': self.tunable_states,
            'frequency_configurations': self.frequency_configurations,
            'maximum_power_balance_error': float(self.maximum_power_balance_error),
            'search_diagnostics': self.search_diagnostics,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TuningResult":
        fields = {
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value and name not in {"per_port", "component_choices", "yield_context"}
        }
        result = cls(**fields)
        result.per_port = {
            int(port): PerPortTuningMetrics.from_dict(metrics)
            for port, metrics in value.get("per_port", {}).items()
        }
        # Runtime component objects are deliberately not reconstructed from an
        # untrusted project file. Saved band curves remain available for view;
        # an exact dense sweep requires rerunning the saved configuration.
        result.component_choices = {}
        result.yield_context = None
        return result


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
    metadata = component_metadata(comp) if comp is not None else {
        'manufacturer': '', 'package_code': '', 'tolerance_pct': None,
        'voltage_code': '', 'dielectric': '', 'tempco_ppm_per_c': None,
        'systematic_bias_pct': None, 'environment_metadata': {}, 'provenance': {},
    }
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
        'manufacturer': metadata['manufacturer'],
        'series': component_series_name(comp) if comp is not None else '',
        'package_code': metadata['package_code'],
        'tolerance_pct': metadata['tolerance_pct'],
        'voltage_code': metadata['voltage_code'],
        'dielectric': metadata['dielectric'],
        'tempco_ppm_per_c': metadata['tempco_ppm_per_c'],
        'systematic_bias_pct': metadata['systematic_bias_pct'],
        'environment_metadata': metadata['environment_metadata'],
        'metadata_provenance': metadata['provenance'],
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

    # JSON-safe request used to reproduce the current result set.
    last_tuning_request: dict = field(default_factory=dict)

    # "live" after optimization, "snapshot" after loading a project file.
    restoration_mode: str = "live"

    # Runtime-only measured-search state. It is deliberately excluded from
    # project serialization because it owns loaded models and NumPy arrays.
    search_checkpoint: Optional[dict] = field(default=None, repr=False)

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


def _reference_impedance_values(dut: TouchstoneData) -> np.ndarray:
    raw = getattr(dut, "port_impedances", None)
    if raw:
        values = np.asarray([complex(value) for value in raw], dtype=complex)
        if values.shape != (dut.num_ports,):
            raise ValueError("DUT port impedance count does not match its port count")
        if np.any(np.abs(values.imag) > 1e-12):
            raise ValueError("complex DUT reference impedances are not supported")
        real = values.real.astype(float)
    else:
        real = np.full(dut.num_ports, float(dut.reference_resistance), dtype=float)
    if not np.all(np.isfinite(real)) or np.any(real <= 0):
        raise ValueError("DUT reference impedances must be finite and positive")
    return real


def _reference_impedance(
    dut: TouchstoneData,
    ports: Optional[List[int]] = None,
) -> float | np.ndarray:
    """Return scalar or real per-port power-wave reference impedances."""
    real = _reference_impedance_values(dut)
    if ports is not None:
        real = real[np.asarray(ports, dtype=int)]
    return float(real[0]) if np.allclose(real, real[0]) else real


def _resolve_single_objective_weights(
    objective: str,
    within_band_average_weight: Optional[float] = None,
    across_band_average_weight: Optional[float] = None,
) -> tuple[float, float]:
    """Resolve Optenni-style min/average emphasis for a single-port search."""
    default_within = {
        "worst_case": 0.0,
        "average_efficiency": 1.0,
        "balanced": 0.05,
    }.get(objective, 0.05)
    within = default_within if within_band_average_weight is None else float(within_band_average_weight)
    across = 0.1 if across_band_average_weight is None else float(across_band_average_weight)
    if not np.isfinite(within) or not 0.0 <= within <= 1.0:
        raise ValueError("within-band average weight must be between zero and one")
    if not np.isfinite(across) or not 0.0 <= across <= 1.0:
        raise ValueError("across-band average weight must be between zero and one")
    return within, across


def _resolve_band_priorities(
    bands_mhz: List[List[float]],
    band_weights: Optional[List[float]] = None,
    port_weight: float = 1.0,
    *,
    require_positive: bool = True,
) -> tuple[list[float], list[float]]:
    """Validate product priority weights and return raw/effective band weights."""
    port_priority = float(port_weight)
    if not np.isfinite(port_priority) or not 0.0 <= port_priority <= 100.0:
        raise ValueError("port weight must be finite and between zero and 100")
    raw = [1.0] * len(bands_mhz) if band_weights is None else [float(value) for value in band_weights]
    if len(raw) != len(bands_mhz):
        raise ValueError("band weights must contain one value for every optimization band")
    if any(not np.isfinite(value) or not 0.0 <= value <= 100.0 for value in raw):
        raise ValueError("band weights must be finite and between zero and 100")
    effective = [port_priority * value for value in raw]
    if require_positive and effective and not any(value > 0.0 for value in effective):
        raise ValueError("at least one optimization band must have a positive effective weight")
    return raw, effective


def _resolve_generic_synthesis_loss(config: Optional[dict] = None) -> tuple[LumpedLossModel, dict]:
    """Validate generic topology-prior losses while preserving product defaults."""
    raw = dict(config or {})
    values = {
        "inductor_q": float(raw.get("inductor_q", 30.0)),
        "inductor_q_reference_hz": float(raw.get("inductor_q_reference_hz", 1e9)),
        "inductor_esr": float(raw.get("inductor_esr_ohm", raw.get("inductor_esr", 0.0))),
        "capacitor_esr": float(raw.get("capacitor_esr_ohm", raw.get("capacitor_esr", 0.4))),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("generic synthesis loss values must be finite")
    if values["inductor_q"] <= 0.0:
        raise ValueError("generic synthesis inductor Q must be positive")
    if values["inductor_q_reference_hz"] <= 0.0:
        raise ValueError("generic synthesis Q reference frequency must be positive")
    if values["inductor_esr"] < 0.0 or values["capacitor_esr"] < 0.0:
        raise ValueError("generic synthesis ESR values cannot be negative")
    return LumpedLossModel(**values), {
        "inductor_q": values["inductor_q"],
        "inductor_q_reference_hz": values["inductor_q_reference_hz"],
        "inductor_esr_ohm": values["inductor_esr"],
        "capacitor_esr_ohm": values["capacitor_esr"],
        "scope": "continuous_topology_prior_only",
    }


def _loss_aware_single_seed(
    dut: TouchstoneData,
    port_index: int,
    bands_mhz: List[List[float]],
    frequencies_hz: List[float],
    topologies,
    objective: str,
    effective_band_weights: Optional[List[float]] = None,
    within_band_average_weight: Optional[float] = None,
    across_band_average_weight: Optional[float] = None,
    generic_synthesis_loss: Optional[dict] = None,
    global_efficiency=None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
) -> dict:
    """Fast Optenni-style generic synthesis used to prioritize product search."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    matrices = []
    for frequency in frequencies:
        matrix = dut.get_s_matrix_interpolated(float(frequency))
        terminations = {
            port: 0.0 for port in range(dut.num_ports) if port != port_index
        }
        if terminations:
            matrix = terminate_ports(matrix, terminations)
        matrices.append(matrix)
    radiation_efficiency = {}
    efficiency_data = (per_port_efficiency or {}).get(port_index) or global_efficiency
    if efficiency_data is not None:
        radiation_efficiency[0] = efficiency_data.get_efficiency_array(frequencies)
    problem = CoreProblem(
        frequencies,
        np.asarray(matrices),
        {0: tuple(
            CoreBand(
                float(start) * 1e6,
                float(stop) * 1e6,
                weight=float(weight),
            )
            for (start, stop), weight in zip(
                bands_mhz,
                effective_band_weights or [1.0] * len(bands_mhz),
            )
        )},
        _reference_impedance(dut, [port_index]),
        radiation_efficiency,
    )
    signatures = []
    core_topologies = []
    for topology in topologies:
        signature = tuple(
            (
                element.connection_type.value,
                "L" if element.component_type == "inductor" else "C",
            )
            for element in topology.elements
        )
        if signature in signatures:
            continue
        signatures.append(signature)
        core_topologies.append(tuple(
            (connection, kind, 0) for connection, kind in signature
        ))
    within_weight, across_weight = _resolve_single_objective_weights(
        objective, within_band_average_weight, across_band_average_weight
    )
    loss_model, loss_diagnostics = _resolve_generic_synthesis_loss(generic_synthesis_loss)
    result = CoreMatchingOptimizer(
        problem,
        CoreObjective(
            within_band_average_weight=within_weight,
            across_band_average_weight=across_weight,
            port_average_weight=0.0,
        ),
        CoreSearchConfig(restarts=6, iterations=20, keep=20, seed=1),
        loss_model,
    ).optimize(core_topologies)
    best = result.best
    best_signature = tuple((item.connection, item.kind) for item in best.elements)
    return {
        "topology_signature": [list(item) for item in best_signature],
        "elements": [
            {
                "connection": item.connection,
                "kind": item.kind,
                "value_si": item.value,
            }
            for item in best.elements
        ],
        "score_db": best.score_db,
        "evaluations": result.evaluations,
        "frequency_points": len(frequencies),
        "loss_model": best.metrics["loss_model"],
        "requested_loss_model": loss_diagnostics,
        "maximum_power_balance_error": best.metrics["maximum_power_balance_error"],
    }


def core_s2p_layout_from_touchstone(
    layout: TouchstoneData,
    *,
    reverse_ports: bool = False,
    target_reference_impedance_ohm: float | None = None,
    left_fixture: tuple[TouchstoneData, bool] | None = None,
    right_fixture: tuple[TouchstoneData, bool] | None = None,
    maximum_deembedding_condition_number: float = 1e10,
) -> tuple[S2PModel, dict]:
    """Convert a two-port EM/VNA file and report passivity/reciprocity diagnostics."""
    if layout.num_ports != 2:
        raise ValueError(f"layout block {layout.filename!r} must be a two-port S2P file")
    frequencies = np.asarray(layout.frequencies, dtype=float)
    if len(frequencies) < 1 or np.any(~np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0):
        raise ValueError(f"layout block {layout.filename!r} requires finite increasing frequencies")
    matrices = np.asarray([
        layout.get_s_matrix(index) for index in range(len(frequencies))
    ], dtype=complex)
    if np.any(~np.isfinite(matrices)):
        raise ValueError(f"layout block {layout.filename!r} contains non-finite S-parameters")
    native_reference_impedance = float(layout.reference_resistance)
    if reverse_ports:
        matrices = flip_s2p_ports(matrices)
    effective_reference_impedance = native_reference_impedance
    if target_reference_impedance_ohm is not None:
        matrices = renormalize_s_parameters(
            matrices, native_reference_impedance, float(target_reference_impedance_ohm)
        )
        effective_reference_impedance = float(target_reference_impedance_ohm)
    fixture_diagnostics = {}

    def fixture_sweep(side: str, fixture_spec):
        if fixture_spec is None:
            return None
        fixture_data, fixture_reversed = fixture_spec
        fixture_model, fixture_diag = core_s2p_layout_from_touchstone(
            fixture_data,
            reverse_ports=fixture_reversed,
            target_reference_impedance_ohm=effective_reference_impedance,
        )
        fixture_frequencies = np.asarray(fixture_model.frequencies_hz, dtype=float)
        tolerance_hz = max(1.0, abs(float(frequencies[0])) * 1e-12, abs(float(frequencies[-1])) * 1e-12)
        if frequencies[0] < fixture_frequencies[0] - tolerance_hz or frequencies[-1] > fixture_frequencies[-1] + tolerance_hz:
            raise ValueError(
                f"{side} fixture {fixture_data.filename!r} does not cover layout frequencies "
                f"{frequencies[0]:g}..{frequencies[-1]:g} Hz"
            )
        fixture_diagnostics[side] = fixture_diag
        return np.asarray([fixture_model.at(float(frequency)) for frequency in frequencies])

    left_s = fixture_sweep("left", left_fixture)
    right_s = fixture_sweep("right", right_fixture)
    deembedding = None
    if left_s is not None or right_s is not None:
        matrices, deembedding = deembed_s2p(
            matrices,
            left_fixture=left_s,
            right_fixture=right_s,
            maximum_condition_number=float(maximum_deembedding_condition_number),
        )
    singular_values = np.asarray([
        np.linalg.svd(matrix, compute_uv=False)[0] for matrix in matrices
    ])
    maximum_singular = float(np.max(singular_values))
    maximum_reciprocity_error = float(np.max(np.abs(matrices[:, 0, 1] - matrices[:, 1, 0])))
    diagnostics = {
        "passive": maximum_singular <= 1.0 + 1e-6,
        "maximum_singular_value": maximum_singular,
        "maximum_power_gain_excess": max(0.0, maximum_singular**2 - 1.0),
        "maximum_reciprocity_error": maximum_reciprocity_error,
        "frequency_start_hz": float(frequencies[0]),
        "frequency_stop_hz": float(frequencies[-1]),
        "frequency_points": len(frequencies),
        "native_reference_impedance_ohm": native_reference_impedance,
        "reference_impedance_ohm": effective_reference_impedance,
        "renormalized": target_reference_impedance_ohm is not None,
        "ports_reversed": bool(reverse_ports),
        "deembedded": deembedding is not None,
        "deembedding": deembedding,
        "fixtures": fixture_diagnostics,
    }
    return S2PModel(
        layout.filename, frequencies, matrices,
        effective_reference_impedance,
    ), diagnostics


def run_manual_tuning_physical(
    dut: TouchstoneData,
    library,
    *,
    target_frequency_hz: float,
    input_port: int,
    port_states: List[dict],
    components: List[dict],
    sweep_start_hz: Optional[float] = None,
    sweep_stop_hz: Optional[float] = None,
    sweep_points: int = 201,
    use_snp_points: bool = False,
    sweep_frequencies_hz: Optional[List[float]] = None,
) -> dict:
    """Evaluate manual L/C/R, measured parts, lines and stubs in rfmatch-core."""
    if target_frequency_hz <= 0:
        raise ValueError("target_frequency_hz must be positive")
    termination_gamma = {"load": 0.0, "open": 1.0, "short": -1.0}
    terminations = {}
    for item in port_states:
        port = int(item.get("port_index", 0))
        state_name = str(item.get("state", "load")).lower()
        if port == input_port:
            raise ValueError("input_port cannot also be terminated")
        if not 0 <= port < dut.num_ports:
            raise ValueError(f"termination port {port} is outside the DUT")
        if state_name not in termination_gamma:
            raise ValueError(f"unsupported port termination {state_name!r}")
        terminations[port] = termination_gamma[state_name]
    remaining_ports = [port for port in range(dut.num_ports) if port not in terminations]
    if input_port not in remaining_ports:
        raise ValueError("input_port is terminated")
    port_map = {original: reduced for reduced, original in enumerate(remaining_ports)}
    reduced_input_port = port_map[input_port]

    if sweep_frequencies_hz is not None:
        sweep_frequencies = np.asarray(sorted(set(
            float(frequency) for frequency in sweep_frequencies_hz
            if np.isfinite(float(frequency)) and float(frequency) > 0
        )), dtype=float)
        if not len(sweep_frequencies):
            raise ValueError("manual sweep frequency list is empty")
    elif sweep_start_hz is not None or sweep_stop_hz is not None:
        if sweep_start_hz is None or sweep_stop_hz is None:
            raise ValueError("sweep_start_hz and sweep_stop_hz must be provided together")
        if sweep_start_hz <= 0 or sweep_stop_hz <= sweep_start_hz:
            raise ValueError("manual sweep bounds must be positive and increasing")
        if use_snp_points:
            sweep_frequencies = np.asarray([
                frequency for frequency in dut.frequencies
                if sweep_start_hz <= frequency <= sweep_stop_hz
            ], dtype=float)
            if not len(sweep_frequencies):
                raise ValueError("no DUT frequency points fall inside the requested sweep")
        else:
            if not 2 <= sweep_points <= 10001:
                raise ValueError("sweep_points must be between 2 and 10001")
            sweep_frequencies = np.linspace(sweep_start_hz, sweep_stop_hz, sweep_points)
    else:
        sweep_frequencies = np.asarray([target_frequency_hz], dtype=float)
    frequencies = np.asarray(sorted(set([
        float(target_frequency_hz), *map(float, sweep_frequencies)
    ])), dtype=float)

    matrices = []
    raw_matrices = []
    for frequency in frequencies:
        matrix = dut.get_s_matrix_interpolated(float(frequency))
        if terminations:
            matrix = terminate_ports(matrix, terminations)
        matrices.append(matrix)
        raw_matrices.append(matrix)

    placements = []
    component_payload = []
    for index, request in enumerate(components):
        if not isinstance(request, dict):
            raise ValueError(f"manual component #{index + 1} must be an object")
        component_type = str(request.get("comp_type", "inductor")).lower()
        connection = str(request.get("connection_type", "series")).lower()
        original_port = int(request.get("port", input_port))
        if original_port not in port_map:
            raise ValueError(f"manual component #{index + 1} targets a terminated port")
        if connection not in {"series", "shunt"}:
            raise ValueError(f"manual component #{index + 1} has an invalid connection")
        use_ideal = bool(request.get("use_ideal", True))
        value = float(request.get("value", 1.0))
        part_number = None

        if component_type in {"inductor", "capacitor", "resistor"}:
            kind = {"inductor": "L", "capacitor": "C", "resistor": "R"}[component_type]
            if use_ideal:
                scale = {"L": 1e-9, "C": 1e-12, "R": 1.0}[kind]
                if value <= 0:
                    raise ValueError(f"manual component #{index + 1} value must be positive")
                model = LumpedModel(
                    f"manual_{index + 1}_{kind}", kind, value * scale,
                    q=(float(request["q"]) if request.get("q") not in {None, ""} else None),
                    esr=float(request.get("esr_ohm", 0.0)),
                    q_reference_hz=(
                        float(request.get("q_reference_frequency_hz", target_frequency_hz))
                        if request.get("q") not in {None, ""} else None
                    ),
                )
            else:
                requested_part = str(request.get("part_number") or "").strip()
                candidates = (
                    getattr(library, "inductors", [])
                    if component_type == "inductor" else getattr(library, "capacitors", [])
                )
                nearest = next((
                    item for item in candidates
                    if requested_part and str(getattr(item, "part_number", "")).casefold() == requested_part.casefold()
                ), None)
                if requested_part and nearest is None:
                    raise ValueError(
                        f"manual component #{index + 1} measured part {requested_part!r} is not available in the active catalog"
                    )
                if nearest is None:
                    nearest = (
                        library.find_nearest_inductor(value)
                        if component_type == "inductor" and hasattr(library, "find_nearest_inductor")
                        else library.find_nearest_capacitor(value)
                        if component_type == "capacitor" and hasattr(library, "find_nearest_capacitor")
                        else None
                    )
                if nearest is None:
                    raise ValueError(f"no measured component found near {value:g} {component_type}")
                model = _core_s2p_from_library_component(nearest, library, frequencies)
                part_number = str(getattr(nearest, "part_number", model.name))
        elif component_type in {"transmission_line", "open_stub", "short_stub"}:
            line = TransmissionLineModel(
                str(request.get("name") or f"manual_TL{index + 1}"),
                float(request.get("characteristic_impedance_ohm", 50.0)),
                float(request.get("electrical_length_deg", value)),
                float(request.get("reference_frequency_hz", target_frequency_hz)),
                float(request.get("attenuation_db", 0.0)),
                float(request.get("loss_frequency_exponent", 0.5)),
            )
            if component_type == "transmission_line":
                if connection != "series":
                    raise ValueError("a through transmission line must use a series connection")
                model = line
            else:
                if connection != "shunt":
                    raise ValueError("a transmission-line stub must use a shunt connection")
                model = TransmissionLineStubModel(
                    line, "open" if component_type == "open_stub" else "short"
                )
        else:
            raise ValueError(f"unsupported manual component type {component_type!r}")

        placements.append(ModelPlacement(connection, port_map[original_port], model))
        component_payload.append({
            **request,
            "position": index,
            "comp_type": component_type,
            "connection_type": connection,
            "port": original_port,
            "part_number": part_number,
            "numeric_core": "rfmatch_core",
        })

    problem = CoreProblem(frequencies, np.asarray(matrices), {
        reduced_input_port: (CoreBand(float(frequencies[0]), float(frequencies[-1])),)
    }, _reference_impedance(dut, remaining_ports))
    topology = build_model_circuit_topology(len(remaining_ports), placements)
    physical = evaluate_physical_problem(problem, topology)
    target_index = int(np.argmin(np.abs(frequencies - target_frequency_hz)))
    target_s11 = physical.s_parameters[target_index, reduced_input_port, reduced_input_port]
    magnitude = float(abs(target_s11))
    target_raw_s11 = np.asarray(raw_matrices)[target_index, reduced_input_port, reduced_input_port]
    reference_values = np.asarray(problem.z0, dtype=float)
    reference_impedance = float(
        reference_values if reference_values.ndim == 0 else reference_values[reduced_input_port]
    )
    input_impedance = (
        reference_impedance * (1.0 + target_s11) / (1.0 - target_s11)
        if abs(1.0 - target_s11) > 1e-15 else complex(np.inf, np.inf)
    )
    return_loss = float(-20.0 * np.log10(max(magnitude, 1e-15)))
    raw_return_loss = float(-20.0 * np.log10(max(abs(target_raw_s11), 1e-15)))

    sweep_indices = [int(np.argmin(np.abs(frequencies - value))) for value in sweep_frequencies]
    sweep_s = physical.s_parameters[sweep_indices]
    sweep_s11 = sweep_s[:, reduced_input_port, reduced_input_port]
    accepted = np.maximum(0.0, 1.0 - np.abs(sweep_s11) ** 2)
    coupling = np.asarray([
        sum(abs(matrix[other, reduced_input_port]) ** 2 for other in range(matrix.shape[0]) if other != reduced_input_port)
        for matrix in sweep_s
    ])
    component_loss = physical.component_loss[sweep_indices, reduced_input_port]
    total = physical.dut_absorbed_power[sweep_indices, reduced_input_port]
    raw_s11 = np.asarray(raw_matrices)[sweep_indices, reduced_input_port, reduced_input_port]
    return {
        "s11_magnitude": magnitude,
        "s11_db": return_loss,
        "raw_s11_db": raw_return_loss,
        "return_loss_improvement_db": return_loss - raw_return_loss,
        "s11_real": float(target_s11.real),
        "s11_imag": float(target_s11.imag),
        "input_impedance_real": float(input_impedance.real),
        "input_impedance_imag": float(input_impedance.imag),
        "reference_impedance_ohm": reference_impedance,
        "vswr": float((1.0 + magnitude) / (1.0 - magnitude)) if magnitude < 1.0 else None,
        "frequency_hz": float(target_frequency_hz),
        "numeric_core": "rfmatch_core",
        "maximum_power_balance_error": float(np.max(np.abs(physical.power_balance_error))),
        "components": component_payload,
        "sweep": {
            "frequencies": sweep_frequencies.tolist(),
            "s11_magnitude": np.abs(sweep_s11).tolist(),
            "s11_db": (-20.0 * np.log10(np.maximum(np.abs(sweep_s11), 1e-15))).tolist(),
            "s11_real": sweep_s11.real.tolist(),
            "s11_imag": sweep_s11.imag.tolist(),
            "raw_magnitude": np.abs(raw_s11).tolist(),
            "raw_db": (-20.0 * np.log10(np.maximum(np.abs(raw_s11), 1e-15))).tolist(),
            "efficiency": {
                "accepted_pct": (100.0 * accepted).tolist(),
                "coupling_pct": (100.0 * coupling).tolist(),
                "component_loss_pct": (100.0 * component_loss).tolist(),
                "total_pct": (100.0 * total).tolist(),
            },
            "power_balance_error": physical.power_balance_error[
                sweep_indices, reduced_input_port
            ].tolist(),
        },
    }


def run_manual_yield_analysis_physical(
    dut: TouchstoneData,
    library,
    *,
    input_port: int,
    port_states: List[dict],
    components: List[dict],
    bands_mhz: List[List[float]],
    target_return_loss_db: float = 10.0,
    samples: int = 200,
    seed: int = 1,
    distribution: str = "uniform",
    confidence_level: float = 0.95,
    default_tolerance_pct: float = 5.0,
    batch_correlation: float = 0.0,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Monte Carlo robustness analysis for the exact ordered manual network."""
    if not 0 <= input_port < dut.num_ports:
        raise ValueError("manual yield input port is outside the DUT")
    if not bands_mhz:
        raise ValueError("manual yield requires at least one target band")
    termination_gamma = {"load": 0.0, "open": 1.0, "short": -1.0}
    terminations = {}
    for item in port_states:
        port = int(item.get("port_index", 0))
        state_name = str(item.get("state", "load")).lower()
        if port == input_port:
            raise ValueError("input_port cannot also be terminated")
        if not 0 <= port < dut.num_ports or state_name not in termination_gamma:
            raise ValueError("manual yield contains an invalid port termination")
        terminations[port] = termination_gamma[state_name]
    remaining_ports = [port for port in range(dut.num_ports) if port not in terminations]
    port_map = {original: reduced for reduced, original in enumerate(remaining_ports)}
    reduced_input_port = port_map[input_port]

    positive_frequencies = np.asarray(
        sorted(float(value) for value in dut.frequencies if float(value) > 0), dtype=float,
    )
    normalized_bands = []
    requested_frequencies = []
    for band_index, band in enumerate(bands_mhz):
        if len(band) != 2:
            raise ValueError(f"manual yield band #{band_index + 1} must contain start and stop")
        start_hz, stop_hz = float(band[0]) * 1e6, float(band[1]) * 1e6
        if stop_hz < start_hz or start_hz < positive_frequencies[0] or stop_hz > positive_frequencies[-1]:
            raise ValueError(f"manual yield band #{band_index + 1} is invalid or outside the DUT sweep")
        normalized_bands.append(CoreBand(start_hz, stop_hz))
        requested_frequencies.extend(np.linspace(start_hz, stop_hz, 17).tolist())
    requested_frequencies.extend(
        frequency for frequency in positive_frequencies
        if any(band.start_hz <= frequency <= band.stop_hz for band in normalized_bands)
    )
    frequencies = np.asarray(sorted(set(map(float, requested_frequencies))), dtype=float)
    matrices = []
    for frequency in frequencies:
        matrix = dut.get_s_matrix_interpolated(float(frequency))
        matrices.append(terminate_ports(matrix, terminations) if terminations else matrix)

    placements = []
    tolerance_payload = []
    for index, request in enumerate(components):
        component_type = str(request.get("comp_type", "inductor")).lower()
        connection = str(request.get("connection_type", "series")).lower()
        original_port = int(request.get("port", input_port))
        if connection not in {"series", "shunt"} or original_port not in port_map:
            raise ValueError(f"manual yield component #{index + 1} has an invalid placement")
        use_ideal = bool(request.get("use_ideal", True))
        tolerance_pct = float(request.get("tolerance_pct") or default_tolerance_pct)
        tolerance_fraction = tolerance_pct / 100.0
        value = float(request.get("value", 1.0))
        source = "request_component" if request.get("tolerance_pct") not in {None, ""} else "request_default"
        part_number = None
        variable = "value"
        if component_type in {"inductor", "capacitor", "resistor"} and use_ideal:
            kind = {"inductor": "L", "capacitor": "C", "resistor": "R"}[component_type]
            scale = {"L": 1e-9, "C": 1e-12, "R": 1.0}[kind]
            model = LumpedModel(
                f"manual_{index + 1}_{kind}", kind, value * scale,
                tolerance=tolerance_fraction,
                q=(float(request["q"]) if request.get("q") not in {None, ""} else None),
                esr=float(request.get("esr_ohm", 0.0)),
                q_reference_hz=(
                    float(request.get("q_reference_frequency_hz", frequencies[len(frequencies) // 2]))
                    if request.get("q") not in {None, ""} else None
                ),
            )
        elif component_type in {"inductor", "capacitor"}:
            requested_part = str(request.get("part_number") or "").strip()
            candidates = (
                getattr(library, "inductors", []) if component_type == "inductor"
                else getattr(library, "capacitors", [])
            ) if library is not None else []
            component = next((
                item for item in candidates
                if requested_part and str(getattr(item, "part_number", "")).casefold() == requested_part.casefold()
            ), None)
            if component is None:
                raise ValueError(f"manual yield measured part {requested_part!r} is not available")
            native_tolerance = getattr(component, "tolerance_pct", None)
            tolerance_fraction = (
                float(native_tolerance) / 100.0
                if native_tolerance is not None and float(native_tolerance) > 0 else tolerance_fraction
            )
            tolerance_pct = 100.0 * tolerance_fraction
            source = "component_metadata" if native_tolerance is not None else source
            model = _core_s2p_from_library_component(
                component, library, frequencies,
                default_tolerance_fraction=tolerance_fraction,
            )
            part_number = requested_part
        elif component_type in {"transmission_line", "open_stub", "short_stub"}:
            line = TransmissionLineModel(
                str(request.get("name") or f"manual_TL{index + 1}"),
                float(request.get("characteristic_impedance_ohm", 50.0)),
                float(request.get("electrical_length_deg", value)),
                float(request.get("reference_frequency_hz", frequencies[len(frequencies) // 2])),
                float(request.get("attenuation_db", 0.0)),
                float(request.get("loss_frequency_exponent", 0.5)),
                tolerance=tolerance_fraction,
            )
            model = line if component_type == "transmission_line" else TransmissionLineStubModel(
                line, "open" if component_type == "open_stub" else "short",
            )
            variable = "electrical_length"
        else:
            raise ValueError(f"manual yield does not support component type {component_type!r}")
        placements.append(ModelPlacement(connection, port_map[original_port], model))
        tolerance_payload.append({
            "position": index + 1,
            "part_number": part_number or str(getattr(model, "name", f"component_{index + 1}")),
            "component_type": component_type,
            "variable": variable,
            "tolerance_pct": tolerance_pct,
            "source": source,
        })

    problem = CoreProblem(
        frequencies, np.asarray(matrices),
        {reduced_input_port: tuple(normalized_bands)},
        _reference_impedance(dut, remaining_ports),
    )
    topology = build_model_circuit_topology(len(remaining_ports), placements)
    criteria = YieldCriteria(
        minimum_total_efficiency=1e-12,
        minimum_average_total_efficiency=1e-12,
        minimum_return_loss_db=float(target_return_loss_db),
    )
    tolerance_result = monte_carlo_yield(
        problem, topology, criteria,
        samples=samples, seed=seed, distribution=distribution,
        confidence_level=confidence_level,
        tolerance_model=ToleranceModel(batch_correlation=batch_correlation),
        progress_callback=progress_callback, cancel_check=cancel_check,
    )
    summary = tolerance_summary(tolerance_result)
    return_loss_percentiles = {
        str(percentile): float(np.percentile(
            tolerance_result.sample_minimum_return_loss_db, percentile,
        )) for percentile in (1, 5, 50, 95, 99)
    }
    risk_components = []
    worst_sample = summary["worst_sample"]
    for item, branch in zip(tolerance_payload, topology.branches):
        branch_name = str(branch.name)
        scale = float(worst_sample.get(branch_name, 1.0))
        risk_components.append({
            **item,
            "variation_key": branch_name,
            "worst_scale": scale,
            "worst_deviation_pct": 100.0 * (scale - 1.0),
        })
    risk_components.sort(key=lambda item: abs(item["worst_deviation_pct"]), reverse=True)
    nominal = evaluate_physical_problem(problem, topology)
    nominal_s11 = nominal.s_parameters[:, reduced_input_port, reduced_input_port]
    nominal_return_loss = -20.0 * np.log10(np.maximum(np.abs(nominal_s11), 1e-15))
    summary.update({
        "analysis_scope": "manual_exact_network_monte_carlo",
        "target_return_loss_db": float(target_return_loss_db),
        "frequency_points": len(frequencies),
        "nominal_worst_return_loss_db": float(np.min(nominal_return_loss)),
        "return_loss_percentiles_db": return_loss_percentiles,
        "risk_components": risk_components,
        "component_tolerances": tolerance_payload,
        "maximum_nominal_power_balance_error": float(np.max(np.abs(nominal.power_balance_error))),
    })
    return summary


def optimize_manual_network_physical(
    dut: TouchstoneData,
    library,
    *,
    target_frequency_hz: float,
    input_port: int,
    port_states: List[dict],
    components: List[dict],
    bands_mhz: List[List[float]],
    target_return_loss_db: float = 10.0,
    objective: str = "balanced",
    max_passes: int = 4,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Refine continuous values while preserving the exact manual topology.

    Measured S2P parts remain fixed. Ideal L/C/R values and transmission-line
    Z0/electrical length are optimized with deterministic log-coordinate
    descent and every trial is evaluated by the same physical manual core.
    """
    if objective not in {"worst", "average", "balanced"}:
        raise ValueError("manual refinement objective must be worst, average, or balanced")
    if not 1 <= int(max_passes) <= 8:
        raise ValueError("manual refinement max_passes must be between 1 and 8")
    if not bands_mhz:
        raise ValueError("manual refinement requires at least one target band")
    dut_frequencies = np.asarray(dut.frequencies, dtype=float)
    positive_dut_frequencies = dut_frequencies[dut_frequencies > 0]
    if not len(positive_dut_frequencies):
        raise ValueError("DUT has no positive frequency points")
    minimum_hz = float(np.min(positive_dut_frequencies))
    maximum_hz = float(np.max(positive_dut_frequencies))
    normalized_bands = []
    search_frequencies = []
    for index, band in enumerate(bands_mhz):
        if len(band) != 2:
            raise ValueError(f"manual refinement band #{index + 1} must contain start and stop")
        start_hz, stop_hz = float(band[0]) * 1e6, float(band[1]) * 1e6
        if not np.isfinite(start_hz) or not np.isfinite(stop_hz) or stop_hz < start_hz:
            raise ValueError(f"manual refinement band #{index + 1} is invalid")
        if start_hz < float(np.min(dut_frequencies)) or stop_hz > maximum_hz:
            raise ValueError(f"manual refinement band #{index + 1} is outside the DUT sweep")
        start_hz = max(start_hz, minimum_hz)
        normalized_bands.append((start_hz, stop_hz))
        search_frequencies.extend(np.linspace(start_hz, stop_hz, 17).tolist())
    search_frequencies = sorted(set(float(value) for value in search_frequencies))

    optimized = deepcopy(components)
    variables = []
    lumped_bounds = {
        "inductor": (0.05, 1000.0),
        "capacitor": (0.05, 10000.0),
        "resistor": (0.01, 10000.0),
    }
    for index, component in enumerate(optimized):
        component_type = str(component.get("comp_type", "")).lower()
        if component_type in lumped_bounds and bool(component.get("use_ideal", True)):
            variables.append((index, "value", lumped_bounds[component_type]))
        elif component_type in {"transmission_line", "open_stub", "short_stub"}:
            variables.extend([
                (index, "characteristic_impedance_ohm", (10.0, 150.0)),
                (index, "electrical_length_deg", (0.1, 360.0)),
            ])
    if not variables:
        raise ValueError("manual refinement has no continuous ideal or transmission-line variables")

    evaluation_count = 0
    probe_position_count = 1 if not optimized else 2
    topology_probe_budget = 4 * 9 * probe_position_count
    measured_probe_budget = 4 * 5
    measured_full_verification_budget = 4 * 2
    # Baseline + coordinate trials + local perturbations + topology probes + final full sweep.
    total_evaluations = (
        2 + 2 * len(variables) * (int(max_passes) + 1)
        + topology_probe_budget + measured_probe_budget + measured_full_verification_budget
    )
    cache = {}

    def score_result(result: dict) -> tuple[float, dict]:
        frequencies = np.asarray(result["sweep"]["frequencies"], dtype=float)
        return_loss = np.asarray(result["sweep"]["s11_db"], dtype=float)
        s11_real = np.asarray(result["sweep"]["s11_real"], dtype=float)
        s11_imag = np.asarray(result["sweep"]["s11_imag"], dtype=float)
        reference_impedance = float(result.get("reference_impedance_ohm", 50.0))
        values = []
        band_metrics = []
        for start_hz, stop_hz in normalized_bands:
            mask = (frequencies >= start_hz) & (frequencies <= stop_hz)
            selected = return_loss[mask]
            if not len(selected):
                raise ValueError("manual refinement search grid missed a target band")
            worst = float(np.min(selected))
            average = float(np.mean(selected))
            selected_indices = np.flatnonzero(mask)
            worst_index = int(selected_indices[int(np.argmin(selected))])
            gamma = complex(s11_real[worst_index], s11_imag[worst_index])
            impedance = (
                reference_impedance * (1.0 + gamma) / (1.0 - gamma)
                if abs(1.0 - gamma) > 1e-15 else None
            )
            values.extend(selected.tolist())
            band_metrics.append({
                "start_mhz": start_hz / 1e6,
                "stop_mhz": stop_hz / 1e6,
                "worst_return_loss_db": worst,
                "average_return_loss_db": average,
                "worst_frequency_hz": float(frequencies[worst_index]),
                "worst_gamma_real": float(gamma.real),
                "worst_gamma_imag": float(gamma.imag),
                "worst_impedance_real_ohm": float(impedance.real) if impedance is not None else None,
                "worst_impedance_imag_ohm": float(impedance.imag) if impedance is not None else None,
                "margin_db": worst - float(target_return_loss_db),
                "passes": worst >= float(target_return_loss_db),
            })
        worst = float(np.min(values))
        average = float(np.mean(values))
        score = worst if objective == "worst" else average if objective == "average" else 0.7 * worst + 0.3 * average
        return score, {
            "score_db": score,
            "worst_return_loss_db": worst,
            "average_return_loss_db": average,
            "margin_db": worst - float(target_return_loss_db),
            "passes_all_bands": worst >= float(target_return_loss_db),
            "bands": band_metrics,
        }

    def evaluate(candidate_components: List[dict]) -> tuple[dict, float, dict]:
        nonlocal evaluation_count
        if cancel_check is not None and cancel_check():
            raise OptimizationCancelled("manual fixed-topology refinement cancelled")
        signature = tuple(
            round(math.log(max(float(candidate_components[index].get(key, bounds[0])), 1e-30)), 12)
            for index, key, bounds in variables
        )
        if signature in cache:
            return cache[signature]
        result = run_manual_tuning_physical(
            dut, library,
            target_frequency_hz=target_frequency_hz,
            input_port=input_port,
            port_states=port_states,
            components=candidate_components,
            sweep_frequencies_hz=search_frequencies,
        )
        score, metrics = score_result(result)
        evaluation_count += 1
        cached = (result, score, metrics)
        cache[signature] = cached
        return cached

    baseline_result, baseline_score, baseline_metrics = evaluate(optimized)
    best_score = baseline_score
    best_metrics = baseline_metrics
    factors = np.geomspace(2.5, 1.08, int(max_passes))
    for pass_index, factor in enumerate(factors):
        for variable_index, (component_index, key, bounds) in enumerate(variables):
            current_value = float(optimized[component_index].get(key, bounds[0]))
            best_trial = None
            for direction in (-1.0, 1.0):
                trial = deepcopy(optimized)
                trial_value = float(np.clip(
                    current_value * (float(factor) ** direction), bounds[0], bounds[1]
                ))
                trial[component_index][key] = trial_value
                trial_result, trial_score, trial_metrics = evaluate(trial)
                if trial_score > best_score + 1e-9 and (
                    best_trial is None or trial_score > best_trial[1]
                ):
                    best_trial = (trial, trial_score, trial_metrics, trial_result)
            if best_trial is not None:
                optimized, best_score, best_metrics, _ = best_trial
            if progress_callback is not None:
                progress_callback({
                    "stage": "manual_refinement",
                    "current": min(evaluation_count, total_evaluations),
                    "total": total_evaluations,
                    "message": f"Refining variable {variable_index + 1}/{len(variables)} · pass {pass_index + 1}/{len(factors)}",
                    "best_score_db": best_score,
                    "best_worst_return_loss_db": best_metrics["worst_return_loss_db"],
                })

    sensitivity = []
    perturbation_factor = 1.1
    parameter_units = {
        "value": {
            "inductor": "nH", "capacitor": "pF", "resistor": "ohm",
        },
        "characteristic_impedance_ohm": "ohm",
        "electrical_length_deg": "deg",
    }
    for variable_index, (component_index, key, bounds) in enumerate(variables):
        component_type = str(optimized[component_index].get("comp_type", "")).lower()
        base_value = float(optimized[component_index].get(key, bounds[0]))
        lower_value = float(np.clip(base_value / perturbation_factor, bounds[0], bounds[1]))
        upper_value = float(np.clip(base_value * perturbation_factor, bounds[0], bounds[1]))
        local_trials = []
        for direction, trial_value in (("decrease", lower_value), ("increase", upper_value)):
            trial = deepcopy(optimized)
            trial[component_index][key] = trial_value
            _, trial_score, trial_metrics = evaluate(trial)
            local_trials.append((direction, trial_value, trial_score, trial_metrics))
        lower_trial, upper_trial = local_trials
        score_impact = max(
            abs(lower_trial[2] - best_score), abs(upper_trial[2] - best_score),
        )
        worst_impact = max(
            abs(lower_trial[3]["worst_return_loss_db"] - best_metrics["worst_return_loss_db"]),
            abs(upper_trial[3]["worst_return_loss_db"] - best_metrics["worst_return_loss_db"]),
        )
        best_local_trial = max(local_trials, key=lambda item: item[2])
        preferred_direction = (
            best_local_trial[0] if best_local_trial[2] > best_score + 1e-9 else "hold"
        )
        sensitivity.append({
            "component_index": component_index,
            "component_type": component_type,
            "parameter": key,
            "unit": parameter_units.get(key, {}).get(component_type, "")
            if isinstance(parameter_units.get(key), dict) else parameter_units.get(key, ""),
            "base_value": base_value,
            "perturbation_pct": 10.0,
            "lower_value": lower_value,
            "upper_value": upper_value,
            "lower_score_db": lower_trial[2],
            "upper_score_db": upper_trial[2],
            "score_impact_db": score_impact,
            "worst_return_loss_impact_db": worst_impact,
            "preferred_direction": preferred_direction,
        })
        if progress_callback is not None:
            progress_callback({
                "stage": "manual_sensitivity",
                "current": min(evaluation_count, total_evaluations),
                "total": total_evaluations,
                "message": f"Measuring sensitivity {variable_index + 1}/{len(variables)}",
                "best_score_db": best_score,
                "best_worst_return_loss_db": best_metrics["worst_return_loss_db"],
            })
    sensitivity.sort(key=lambda item: item["score_impact_db"], reverse=True)

    topology_probes = []
    topology_probe_evaluations = 0
    if not best_metrics["passes_all_bands"]:
        insertion_points = [(0, "dut_side")]
        if optimized:
            insertion_points.append((len(optimized), "source_side"))
        probe_specs = (
            ("series", "inductor", (0.05, 1000.0)),
            ("series", "capacitor", (0.05, 10000.0)),
            ("shunt", "inductor", (0.05, 1000.0)),
            ("shunt", "capacitor", (0.05, 10000.0)),
        )
        for insertion_index, location in insertion_points:
            for connection, component_type, bounds in probe_specs:
                best_probe = None
                for value in np.geomspace(bounds[0], bounds[1], 9):
                    if cancel_check is not None and cancel_check():
                        raise OptimizationCancelled("manual topology probe cancelled")
                    probe_component = {
                        "connection_type": connection,
                        "comp_type": component_type,
                        "value": float(value),
                        "use_ideal": True,
                        "port": input_port,
                        "reference_frequency_hz": target_frequency_hz,
                    }
                    trial = deepcopy(optimized)
                    trial.insert(insertion_index, probe_component)
                    trial_result = run_manual_tuning_physical(
                        dut, library,
                        target_frequency_hz=target_frequency_hz,
                        input_port=input_port,
                        port_states=port_states,
                        components=trial,
                        sweep_frequencies_hz=search_frequencies,
                    )
                    trial_score, trial_metrics = score_result(trial_result)
                    evaluation_count += 1
                    topology_probe_evaluations += 1
                    if best_probe is None or trial_score > best_probe[0]:
                        best_probe = (trial_score, trial_metrics, probe_component)
                if best_probe is not None and best_probe[0] > best_score + 1e-9:
                    topology_probes.append({
                        "insertion_index": insertion_index,
                        "location": location,
                        "connection_type": connection,
                        "component_type": component_type,
                        "component": best_probe[2],
                        "score_db": best_probe[0],
                        "score_improvement_db": best_probe[0] - best_score,
                        "worst_return_loss_db": best_probe[1]["worst_return_loss_db"],
                        "worst_return_loss_improvement_db": (
                            best_probe[1]["worst_return_loss_db"]
                            - best_metrics["worst_return_loss_db"]
                        ),
                        "passes_all_bands": best_probe[1]["passes_all_bands"],
                        "bands": best_probe[1]["bands"],
                    })
                if progress_callback is not None:
                    progress_callback({
                        "stage": "manual_topology_probe",
                        "current": min(evaluation_count, total_evaluations),
                        "total": total_evaluations,
                        "message": f"Probing {location} {connection} {component_type}",
                        "best_score_db": best_score,
                        "best_worst_return_loss_db": best_metrics["worst_return_loss_db"],
                    })
        topology_probes.sort(key=lambda item: item["score_improvement_db"], reverse=True)
        topology_probes = topology_probes[:4]

    verification_frequencies = sorted(set(
        positive_dut_frequencies.tolist() + search_frequencies
    ))
    final_result = run_manual_tuning_physical(
        dut, library,
        target_frequency_hz=target_frequency_hz,
        input_port=input_port,
        port_states=port_states,
        components=optimized,
        sweep_frequencies_hz=verification_frequencies,
    )
    optimized_full_score, optimized_full_metrics = score_result(final_result)
    evaluation_count += 1

    measured_probe_evaluations = 0
    measured_full_verification_evaluations = 0
    measured_probe_errors = 0
    if library is not None:
        for probe_index, probe in enumerate(topology_probes):
            source = (
                getattr(library, "inductors", [])
                if probe["component_type"] == "inductor"
                else getattr(library, "capacitors", [])
            ) or []
            target_value = float(probe["component"]["value"])
            ordered_parts = sorted(
                (part for part in source if float(getattr(part, "nominal_value", 0.0)) > 0),
                key=lambda part: (
                    abs(math.log(float(getattr(part, "nominal_value")) / target_value)),
                    str(getattr(part, "part_number", "")),
                ),
            )
            unique_parts = []
            seen_part_numbers = set()
            for part in ordered_parts:
                part_number = str(getattr(part, "part_number", "")).strip()
                if not part_number or part_number.casefold() in seen_part_numbers:
                    continue
                seen_part_numbers.add(part_number.casefold())
                unique_parts.append(part)
                if len(unique_parts) >= 5:
                    break
            measured_alternatives = []
            for part in unique_parts:
                if cancel_check is not None and cancel_check():
                    raise OptimizationCancelled("manual measured topology probe cancelled")
                part_number = str(getattr(part, "part_number"))
                nominal_value = float(getattr(part, "nominal_value"))
                measured_component = {
                    **probe["component"],
                    "value": nominal_value,
                    "use_ideal": False,
                    "part_number": part_number,
                }
                trial = deepcopy(optimized)
                trial.insert(probe["insertion_index"], measured_component)
                try:
                    measured_result = run_manual_tuning_physical(
                        dut, library,
                        target_frequency_hz=target_frequency_hz,
                        input_port=input_port,
                        port_states=port_states,
                        components=trial,
                        sweep_frequencies_hz=search_frequencies,
                    )
                    measured_score, measured_metrics = score_result(measured_result)
                except (OSError, RuntimeError, ValueError, KeyError):
                    measured_probe_errors += 1
                    continue
                evaluation_count += 1
                measured_probe_evaluations += 1
                if measured_score > best_score + 1e-9:
                    measured_alternatives.append({
                        "part_number": part_number,
                        "nominal_value": nominal_value,
                        "nominal_unit": str(getattr(part, "nominal_unit", "")),
                        "component": measured_component,
                        "score_db": measured_score,
                        "score_improvement_db": measured_score - best_score,
                        "ideal_score_delta_db": measured_score - probe["score_db"],
                        "worst_return_loss_db": measured_metrics["worst_return_loss_db"],
                        "worst_return_loss_improvement_db": (
                            measured_metrics["worst_return_loss_db"]
                            - best_metrics["worst_return_loss_db"]
                        ),
                        "passes_all_bands": measured_metrics["passes_all_bands"],
                        "bands": measured_metrics["bands"],
                    })
            measured_alternatives.sort(
                key=lambda item: item["score_improvement_db"], reverse=True,
            )
            verified_alternatives = []
            for alternative in measured_alternatives[:2]:
                if cancel_check is not None and cancel_check():
                    raise OptimizationCancelled("manual measured full-sweep verification cancelled")
                trial = deepcopy(optimized)
                trial.insert(probe["insertion_index"], alternative["component"])
                try:
                    verified_result = run_manual_tuning_physical(
                        dut, library,
                        target_frequency_hz=target_frequency_hz,
                        input_port=input_port,
                        port_states=port_states,
                        components=trial,
                        sweep_frequencies_hz=verification_frequencies,
                    )
                    verified_score, verified_metrics = score_result(verified_result)
                except (OSError, RuntimeError, ValueError, KeyError):
                    measured_probe_errors += 1
                    continue
                evaluation_count += 1
                measured_full_verification_evaluations += 1
                if verified_score > optimized_full_score + 1e-9:
                    verified_alternatives.append({
                        **alternative,
                        "coarse_score_db": alternative["score_db"],
                        "coarse_score_improvement_db": alternative["score_improvement_db"],
                        "score_db": verified_score,
                        "score_improvement_db": verified_score - optimized_full_score,
                        "worst_return_loss_db": verified_metrics["worst_return_loss_db"],
                        "worst_return_loss_improvement_db": (
                            verified_metrics["worst_return_loss_db"]
                            - optimized_full_metrics["worst_return_loss_db"]
                        ),
                        "passes_all_bands": verified_metrics["passes_all_bands"],
                        "bands": verified_metrics["bands"],
                        "verification": "full_dut_plus_band_grid",
                        "verification_points": len(verification_frequencies),
                    })
            verified_alternatives.sort(
                key=lambda item: item["score_improvement_db"], reverse=True,
            )
            probe["measured_alternatives"] = verified_alternatives
            if progress_callback is not None:
                progress_callback({
                    "stage": "manual_full_verification",
                    "current": min(evaluation_count, total_evaluations),
                    "total": total_evaluations,
                    "message": f"Full-sweep verification {probe_index + 1}/{len(topology_probes)}",
                    "best_score_db": best_score,
                    "best_worst_return_loss_db": best_metrics["worst_return_loss_db"],
                })

    return {
        "status": "ok",
        "mode": "manual_fixed_topology_refinement",
        "objective": objective,
        "variable_count": len(variables),
        "evaluations": evaluation_count,
        "improved": best_score > baseline_score + 1e-9,
        "score_improvement_db": best_score - baseline_score,
        "baseline": baseline_metrics,
        "optimized": best_metrics,
        "optimized_full": optimized_full_metrics,
        "verification_points": len(verification_frequencies),
        "sensitivity": sensitivity,
        "topology_probes": topology_probes,
        "topology_probe_evaluations": topology_probe_evaluations,
        "measured_probe_evaluations": measured_probe_evaluations,
        "measured_full_verification_evaluations": measured_full_verification_evaluations,
        "measured_probe_errors": measured_probe_errors,
        "components": optimized,
        "result": final_result,
    }


def run_tuning_transmission_line(
    dut: TouchstoneData,
    *,
    port_index: int,
    bands_mhz: List[List[float]],
    objective: str = "balanced",
    num_band_points: int = 10,
    search_config: Optional[dict] = None,
    timeout_seconds: float = 120.0,
    global_efficiency=None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    layout_blocks: Optional[List[dict]] = None,
) -> Dict[int, TuningResult]:
    """Automatically synthesize physical line/stub networks on one DUT port."""
    if not 0 <= port_index < dut.num_ports:
        raise ValueError("transmission-line synthesis port is outside the DUT")
    if not bands_mhz:
        raise ValueError("transmission-line synthesis requires at least one band")
    num_band_points = max(2, int(num_band_points))
    bands = []
    frequency_values = []
    for band in bands_mhz:
        if len(band) != 2 or min(band) <= 0 or band[0] == band[1]:
            raise ValueError("each transmission-line band requires two distinct positive MHz values")
        start_mhz, stop_mhz = sorted(map(float, band))
        bands.append(CoreBand(start_mhz * 1e6, stop_mhz * 1e6))
        frequency_values.extend(np.linspace(start_mhz * 1e6, stop_mhz * 1e6, num_band_points))
    frequencies = np.asarray(sorted(set(map(float, frequency_values))), dtype=float)
    matrices = []
    terminations = {port: 0.0 for port in range(dut.num_ports) if port != port_index}
    for frequency in frequencies:
        matrix = dut.get_s_matrix_interpolated(float(frequency))
        matrices.append(terminate_ports(matrix, terminations) if terminations else matrix)
    radiation = {}
    efficiency_data = (per_port_efficiency or {}).get(port_index) or global_efficiency
    if efficiency_data is not None:
        radiation[0] = efficiency_data.get_efficiency_array(frequencies)
    problem = CoreProblem(
        frequencies, np.asarray(matrices), {0: tuple(bands)},
        _reference_impedance(dut, [port_index]), radiation,
    )
    within_weight = {
        "worst_case": 0.0, "average_efficiency": 1.0, "balanced": 0.05,
    }.get(objective, 0.05)
    raw_config = dict(search_config or {})
    microstrip_config = dict(raw_config.get("microstrip") or {})
    microstrip_rules = None
    if microstrip_config.get("enabled", False):
        minimum_width = float(microstrip_config.get("minimum_width_mm", 0.1)) * 1e-3
        maximum_width = float(microstrip_config.get("maximum_width_mm", 10.0)) * 1e-3
        microstrip_rules = MicrostripDesignRules(
            PCBSubstrate(
                str(microstrip_config.get("substrate_name", "FR-4 engineering model")),
                float(microstrip_config.get("relative_permittivity", 4.5)),
                float(microstrip_config.get("substrate_height_mm", 1.6)) * 1e-3,
                float(microstrip_config.get("loss_tangent", 0.02)),
                float(microstrip_config.get("copper_thickness_um", 35.0)) * 1e-6,
                float(microstrip_config.get("copper_resistivity_ohm_m", 1.68e-8)),
                float(microstrip_config.get("copper_roughness_rms_um", 0.15)) * 1e-6,
            ),
            minimum_width,
            maximum_width,
            float(microstrip_config.get("width_tolerance_pct", 10.0)) / 100.0,
            float(microstrip_config.get("length_tolerance_pct", 0.0)) / 100.0,
            float(microstrip_config.get("substrate_height_tolerance_pct", 0.0)) / 100.0,
            float(microstrip_config.get("relative_permittivity_tolerance_pct", 0.0)) / 100.0,
        )
    dut_side_layouts = []
    connector_side_layouts = []
    layout_metadata = {}
    for item in layout_blocks or []:
        model = item.get("model")
        if not isinstance(model, S2PModel):
            raise ValueError("layout block is missing a valid S2P model")
        if model.name in layout_metadata:
            raise ValueError(f"duplicate layout block name {model.name!r}")
        model_frequencies = np.asarray(model.frequencies_hz, dtype=float)
        coverage_tolerance_hz = max(
            1.0,
            abs(float(frequencies[0])) * 1e-12,
            abs(float(frequencies[-1])) * 1e-12,
        )
        if (
            frequencies[0] < model_frequencies[0] - coverage_tolerance_hz
            or frequencies[-1] > model_frequencies[-1] + coverage_tolerance_hz
        ):
            raise ValueError(
                f"layout block {model.name!r} covers {model_frequencies[0]:g}..{model_frequencies[-1]:g} Hz "
                f"but active synthesis requires {frequencies[0]:g}..{frequencies[-1]:g} Hz"
            )
        placement = ModelPlacement("series", 0, model)
        location = str(item.get("location", "connector_side"))
        if location == "dut_side":
            dut_side_layouts.append(placement)
        elif location == "connector_side":
            connector_side_layouts.append(placement)
        else:
            raise ValueError(f"invalid layout block location {location!r}")
        layout_metadata[model.name] = item
    config = LineSearchConfig(
        characteristic_impedance_min_ohm=float(raw_config.get("characteristic_impedance_min_ohm", 20.0)),
        characteristic_impedance_max_ohm=float(raw_config.get("characteristic_impedance_max_ohm", 120.0)),
        electrical_length_min_deg=float(raw_config.get("electrical_length_min_deg", 1.0)),
        electrical_length_max_deg=float(raw_config.get("electrical_length_max_deg", 179.0)),
        attenuation_db=float(raw_config.get("attenuation_db", 0.0)),
        loss_frequency_exponent=float(raw_config.get("loss_frequency_exponent", 0.5)),
        topologies=tuple(raw_config.get("topologies") or LineSearchConfig().topologies),
        restarts=int(raw_config.get("restarts", 10)),
        iterations=int(raw_config.get("iterations", 24)),
        keep=max(1, min(50, int(raw_config.get("keep", 12)))),
        seed=int(raw_config.get("seed", 1)),
        timeout_seconds=float(timeout_seconds),
        maximum_evaluations=int(raw_config.get("maximum_evaluations", 10000)),
        microstrip_rules=microstrip_rules,
        fixed_dut_side=tuple(dut_side_layouts),
        fixed_connector_side=tuple(connector_side_layouts),
    )
    search = TransmissionLineOptimizer(
        problem, 0, float(np.mean(frequencies)),
        CoreObjective(
            within_weight, 0.1, 0.0,
            complexity_penalty_db=0.35 if objective == "low_cost" else 0.0,
        ), config,
        cancel_check=cancel_check, progress_callback=progress_callback,
    ).optimize()
    results = {}
    for solution_index, candidate in enumerate(search.candidates):
        metrics = candidate.metrics
        s11 = np.asarray(metrics["s_parameters"])[:, 0, 0]
        reflection = np.abs(s11)
        return_loss = -20.0 * np.log10(np.maximum(reflection, 1e-15))
        total = np.asarray(metrics["total_efficiency"])[:, 0]
        component_loss = np.asarray(metrics["component_loss"])[:, 0]
        absorbed = np.asarray(metrics["dut_absorbed_power"])[:, 0]
        center = int(np.argmin(np.abs(frequencies - np.mean(frequencies))))
        descriptions = candidate.components()
        for item in descriptions:
            item["port"] = port_index
            item["type"] = item["comp_type"]
            item["part"] = item["comp_type"]
            item["part_number"] = item["comp_type"]
            if item["comp_type"] == "layout_s2p":
                metadata = layout_metadata.get(item.get("part") or item.get("part_number"), {})
                # Candidate metadata initially uses the model name; preserve it
                # as the report-facing identity and attach traceability fields.
                model_name = candidate.placements[item["position"]].model.name
                metadata = layout_metadata.get(model_name, metadata)
                item["part"] = model_name
                item["part_number"] = model_name
                item.update({
                    "filename": metadata.get("filename", model_name),
                    "sha256": metadata.get("sha256"),
                    "location": metadata.get("location"),
                    "reverse_ports": metadata.get("reverse_ports", False),
                    "reference_impedance_mode": metadata.get("reference_impedance_mode", "native"),
                    "fixtures": metadata.get("fixtures", {}),
                    "passivity": metadata.get("passivity"),
                })
        per_port = PerPortTuningMetrics(
            port_index=port_index,
            s11_magnitude=float(reflection[center]),
            s11_db=float(return_loss[center]),
            accepted_efficiency=float(1.0 - reflection[center] ** 2),
            coupling_loss=0.0,
            component_loss=float(component_loss[center]),
            radiated_efficiency=float(absorbed[center]),
            total_efficiency=float(total[center]),
            components=descriptions,
            band_freqs_hz=frequencies.tolist(),
            band_s11_db=return_loss.tolist(),
            band_total_eff=total.tolist(),
        )
        radiation_efficiency = float(total[center] / absorbed[center]) if absorbed[center] > 0 else 0.0
        antenna_loss = float(absorbed[center] - total[center])
        system_power_balance = {
            "per_port": {str(port_index): {
                "incident": 1.0,
                "reflected": float(reflection[center] ** 2),
                "coupled": 0.0,
                "component_loss": float(component_loss[center]),
                "antenna_loss": antenna_loss,
                "radiated": float(total[center]),
                "accepted": float(1.0 - reflection[center] ** 2),
                "sum_check": float(reflection[center] ** 2 + component_loss[center] + antenna_loss + total[center]),
            }},
            "total_incident": 1.0,
            "total_reflected": float(reflection[center] ** 2),
            "total_coupled": 0.0,
            "total_component_loss": float(component_loss[center]),
            "total_antenna_loss": antenna_loss,
            "total_radiated": float(total[center]),
            "system_efficiency": float(total[center]),
        }
        result = TuningResult(
            port_indices=[port_index], mode="transmission_line", objective=objective,
            per_port={port_index: per_port}, system_score=float(candidate.score_db),
            avg_total_efficiency=float(np.mean(total)),
            min_total_efficiency=float(np.min(total)),
            total_component_loss=float(np.mean(component_loss)),
            total_component_count=len(descriptions),
            system_power_balance=system_power_balance,
            sweep_freqs_hz=frequencies.tolist(),
            total_time_s=search.elapsed_seconds,
            num_solutions_found=len(search.candidates),
            solution_index=solution_index,
            efficiency_basis="rfmatch_core_physical_transmission_line",
            maximum_power_balance_error=float(metrics["maximum_power_balance_error"]),
            search_diagnostics={
                "mode": "transmission_line_auto_synthesis",
                "numeric_core": "rfmatch_core",
                "topology": candidate.topology,
                "reference_frequency_hz": float(np.mean(frequencies)),
                "characteristic_impedance_ohm": candidate.characteristic_impedance_ohm,
                "line_length_deg": candidate.line_length_deg,
                "stub_length_deg": candidate.stub_length_deg,
                "stub_termination": candidate.stub_termination,
                "evaluations": search.evaluations,
                "elapsed_seconds": search.elapsed_seconds,
                "stopped_reason": search.stopped_reason,
                "frequency_points": len(frequencies),
                "radiation_efficiency_at_center": radiation_efficiency,
                "maximum_power_balance_error": float(metrics["maximum_power_balance_error"]),
                "search_bounds": raw_config,
                "physical_microstrip": microstrip_rules is not None,
                "layout_blocks": [
                    {
                        "filename": item.get("filename"),
                        "sha256": item.get("sha256"),
                        "location": item.get("location"),
                        "reverse_ports": item.get("reverse_ports", False),
                        "reference_impedance_mode": item.get("reference_impedance_mode", "native"),
                        "fixtures": item.get("fixtures", {}),
                        "passivity": item.get("passivity"),
                    }
                    for item in (layout_blocks or [])
                ],
            },
        )
        result.yield_context = {
            "kind": "transmission_line",
            "placements": candidate.placements,
            "original_port": port_index,
            "problem": problem,
        }
        results[solution_index] = result
    return results


# ── Service functions ───────────────────────────────────────────────────

_TOPOLOGY_CODE_PATTERN = re.compile(r"^(?:[SP][LC]){1,6}$")


def normalize_allowed_topology_codes(
    raw_codes: object,
    max_components: int,
    *,
    port_index: Optional[int] = None,
) -> Optional[frozenset[str]]:
    """Validate the product topology contract and return canonical codes.

    ``None`` means automatic topology search. ``0`` denotes the bare-DUT
    topology; every other code is an ordered DUT-outward series/shunt ladder.
    """
    location = f" for port {port_index + 1}" if port_index is not None else ""
    if raw_codes is None:
        return frozenset({"0"}) if max_components == 0 else None
    if not isinstance(raw_codes, (list, tuple, set, frozenset)) or not raw_codes:
        raise ValueError(f"allowed_topology_codes{location} must be a non-empty list")
    normalized: set[str] = set()
    for raw_code in raw_codes:
        if not isinstance(raw_code, str):
            raise ValueError(f"topology codes{location} must be strings")
        code = raw_code.strip().upper()
        if code != "0" and not _TOPOLOGY_CODE_PATTERN.fullmatch(code):
            raise ValueError(
                f"invalid topology code {raw_code!r}{location}; use 0 or ordered pairs such as SL, PC, PLSC"
            )
        depth = 0 if code == "0" else len(code) // 2
        if depth > max_components:
            raise ValueError(
                f"topology code {code}{location} exceeds max_components={max_components}"
            )
        normalized.add(code)
    if max_components == 0 and normalized != {"0"}:
        raise ValueError(f"a zero-component port{location} only permits topology code 0")
    return frozenset(normalized)

def run_tuning_single(
    dut: TouchstoneData,
    library: object,
    port_index: int,
    bands_mhz: List[List[float]],
    band_weights: Optional[List[float]] = None,
    port_weight: float = 1.0,
    max_components: int = 2,
    objective: str = "average_efficiency",
    within_band_average_weight: Optional[float] = None,
    across_band_average_weight: Optional[float] = None,
    generic_synthesis_loss: Optional[dict] = None,
    beam_width: int = 20,
    timeout_seconds: float = 60.0,
    num_band_points: int = 10,
    global_efficiency: Optional[object] = None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
    component_series: Optional[List[str]] = None,
    topology_filter: Optional[List[str]] = None,
    allowed_topology_codes: Optional[List[str]] = None,
    search_checkpoint: Optional[dict] = None,
    checkpoint_store: Optional[dict] = None,
    search_profile_timeout_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[int, TuningResult]:
    """
    Single-port tuning — the ONE entry point.

    Returns dict: {solution_index: TuningResult} for all candidates.
    The best solution is at index 0.
    """
    t_start = time.time()
    if not isinstance(max_components, int) or isinstance(max_components, bool):
        raise ValueError("maximum components must be an integer")
    if max_components < 0 or max_components > 6:
        raise ValueError("maximum components must be between zero and six")
    normalized_topology_codes = normalize_allowed_topology_codes(
        allowed_topology_codes, max_components, port_index=port_index
    )
    if not 0 <= int(port_index) < int(dut.num_ports):
        raise ValueError("port index is outside the DUT port range")
    if num_band_points < 2:
        raise ValueError("at least two frequency points per band are required")
    if not bands_mhz:
        raise ValueError("at least one optimization band is required")
    for band in bands_mhz:
        if (
            len(band) != 2
            or not np.all(np.isfinite(np.asarray(band, dtype=float)))
            or float(band[0]) <= 0.0
            or float(band[1]) < float(band[0])
        ):
            raise ValueError("each optimization band must be [positive start_mhz, stop_mhz]")
    resolved_band_weights, effective_band_weights = _resolve_band_priorities(
        bands_mhz, band_weights, port_weight
    )
    preset = get_objective_preset(objective)
    within_weight, across_weight = _resolve_single_objective_weights(
        objective, within_band_average_weight, across_band_average_weight
    )
    _, generic_loss_diagnostics = _resolve_generic_synthesis_loss(generic_synthesis_loss)
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

    # Use the loss-aware numeric core as a full-band topology prior. The
    # product optimizer still selects the actual measured/library parts.
    topos = get_standard_topologies()
    topos = [t for t in topos if t.num_components <= max_components]
    if topology_filter:
        allowed_topologies = {str(name) for name in topology_filter}
        topos = [topology for topology in topos if topology.name in allowed_topologies]
    if topology_filter and not topos:
        return {}
    loss_aware_seed = None
    seed_prioritized = False
    seed_topologies = [
        topology for topology in topos
        if 0 < topology.num_components <= min(max_components, 2)
    ]
    # The loss-aware continuous prior is useful, but it must not consume a
    # sub-second request before the anytime measured core can establish its
    # guaranteed baseline. Tiny budgets therefore enter measured search
    # directly; larger runs retain the calibrated prior.
    if band_freqs and seed_topologies and float(timeout_seconds) >= 0.25:
        try:
            loss_aware_seed = _loss_aware_single_seed(
                dut=dut,
                port_index=port_index,
                bands_mhz=bands_mhz,
                effective_band_weights=effective_band_weights,
                frequencies_hz=band_freqs,
                topologies=seed_topologies,
                objective=objective,
                within_band_average_weight=within_weight,
                across_band_average_weight=across_weight,
                generic_synthesis_loss=generic_loss_diagnostics,
                global_efficiency=global_efficiency,
                per_port_efficiency=per_port_efficiency,
            )
            seed_signature = tuple(
                tuple(item) for item in loss_aware_seed["topology_signature"]
            )

            def topology_signature(topology):
                return tuple(
                    (
                        element.connection_type.value,
                        "L" if element.component_type == "inductor" else "C",
                    )
                    for element in topology.elements
                )

            topos.sort(key=lambda topology: (
                topology_signature(topology) != seed_signature,
                topology.num_components,
                topology.name,
            ))
            seed_prioritized = bool(
                topos and topology_signature(topos[0]) == seed_signature
            )
        except Exception as exc:
            logger.warning("Loss-aware full-band synthesis seed failed: %s", exc)

    measured_candidates, measured_reason = _run_tuning_single_measured_core(
        dut=dut,
        library=library,
        port_index=port_index,
        bands_mhz=bands_mhz,
        band_weights=resolved_band_weights,
        port_weight=port_weight,
        effective_band_weights=effective_band_weights,
        frequencies_hz=band_freqs,
        max_components=max_components,
        objective=objective,
        within_band_average_weight=within_weight,
        across_band_average_weight=across_weight,
        generic_synthesis_loss=generic_loss_diagnostics,
        beam_width=beam_width,
        timeout_seconds=max(0.05, float(timeout_seconds) - (time.time() - t_start)),
        global_efficiency=global_efficiency,
        per_port_efficiency=per_port_efficiency,
        loss_aware_seed=loss_aware_seed,
        search_checkpoint=search_checkpoint,
        checkpoint_store=checkpoint_store,
        search_profile_timeout_seconds=search_profile_timeout_seconds,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        allowed_topology_codes=(
            frozenset(
                "".join(
                    ("S" if element.connection_type.value == "series" else "P")
                    + ("L" if element.component_type == "inductor" else "C")
                    for element in topology.elements
                )
                for topology in topos
            )
            if topology_filter else normalized_topology_codes
        ),
        include_zero_component=not bool(topology_filter),
    )
    if measured_candidates:
        return measured_candidates
    custom_priorities_requested = (
        not np.isclose(float(port_weight), 1.0)
        or any(not np.isclose(value, 1.0) for value in resolved_band_weights)
    )
    if topology_filter or normalized_topology_codes is not None or custom_priorities_requested:
        logger.warning(
            "Measured constraint-aware search returned no candidate: %s",
            measured_reason,
        )
        return {}

    # Run the measured/library optimizer at center frequency, with the
    # predicted topology first for bounded searches.
    config = OptimizerConfig(
        target_frequency_hz=center_freq,
        max_components=max_components,
        beam_width=beam_width,
        timeout_seconds=timeout_seconds,
        bands_mhz=bands_mhz,
        num_band_points=num_band_points,
    )
    opt = MatchingOptimizer(dut, library, config)

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
        result.search_diagnostics = {
            "numeric_core": "rfmatch_core",
            "loss_aware_ideal_seed": loss_aware_seed,
            "generic_synthesis_loss": generic_loss_diagnostics,
            "engine_topology_prioritized": seed_prioritized,
            "measured_physical_search": False,
            "measured_physical_fallback_reason": measured_reason,
        }
        candidates[idx] = result

    return candidates


def run_tuning_grid_s2p(
    dut: TouchstoneData,
    port_specs: List[dict],
    objective: str = "balanced",
    num_band_points: int = 1,
) -> Dict[int, TuningResult]:
    """
    Narrow-band direct S2P grid search.

    Efficiency fields are conservative matching metrics.  They are not
    radiation efficiency unless external efficiency data is applied elsewhere.
    """
    t_start = time.time()
    raw = optimize_narrowband_grid(
        dut=dut,
        port_specs=port_specs,
        num_band_points=num_band_points,
    )
    freqs_hz = [float(f) for f in raw["freqs_hz"]]
    per_port = {}
    component_choices = {}
    enabled_ports = [p for p in port_specs if p.get("enabled", True)]
    optimized_indices = {int(p.get("port_index", i)) for i, p in enumerate(enabled_ports)}

    total_eff_values = []
    for pinfo in raw["per_port"]:
        pi = int(pinfo["port"])
        components = []
        choices = []
        for pos, comp in enumerate(pinfo.get("components", [])):
            topo = pinfo.get("topology", [])
            connection = "shunt" if pos < len(topo) and topo[pos] == "P" else "series"
            choice = GridChoice(
                position=pos,
                component=comp,
                connection_type=connection,
                port=pi,
            )
            choices.append(choice)
            components.append(_component_choice_to_dict(choice))

        accepted = float(pinfo.get("accepted_efficiency", 0.0))
        gt_est = float(pinfo.get("transducer_gain_estimate", accepted))
        total_est = max(0.0, min(1.0, accepted))
        if pinfo.get("port_type") == "load":
            total_eff_values.append(total_est)
        pm = PerPortTuningMetrics(
            port_index=pi,
            s11_magnitude=float(pinfo.get("s11_mag", 1.0)),
            s11_db=float(-pinfo.get("s11_db", 0.0)) if pinfo.get("s11_db", 0.0) < 0 else float(pinfo.get("s11_db", 0.0)),
            accepted_efficiency=accepted,
            coupling_loss=0.0,
            component_loss=0.0,
            radiated_efficiency=gt_est,
            total_efficiency=total_est,
            components=components,
            band_freqs_hz=freqs_hz,
            band_s11_db=[float(-pinfo.get("s11_db", 0.0))],
            band_total_eff=[total_est],
        )
        per_port[pi] = pm
        component_choices[pi] = choices

    if not total_eff_values:
        return {}

    avg_eff = float(np.mean(total_eff_values))
    min_eff = float(np.min(total_eff_values))
    result = TuningResult(
        port_indices=sorted(optimized_indices),
        mode="grid_s2p",
        objective=objective,
        per_port=per_port,
        system_score=float(raw.get("combined_score", avg_eff)),
        avg_total_efficiency=avg_eff,
        min_total_efficiency=min_eff,
        total_component_count=sum(len(v) for v in component_choices.values()),
        component_choices=component_choices,
        total_time_s=time.time() - t_start,
        num_solutions_found=1,
        solution_index=0,
        efficiency_basis=raw.get("efficiency_basis", "accepted_power_estimate_not_radiation_efficiency"),
    )
    result.system_power_balance = {
        "system_efficiency": avg_eff,
        "basis": result.efficiency_basis,
        "note": "Grid S2P mode reports matching accepted-power estimates; load radiation efficiency is not inferred.",
        "library": raw.get("library", {}),
        "optimizer_time_s": raw.get("time_sec", 0.0),
    }
    return {0: result}


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
    isolation_targets: Optional[List[dict]] = None,
    search_checkpoint: Optional[dict] = None,
    checkpoint_store: Optional[dict] = None,
    search_profile_timeout_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
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
    normalized_port_specs = []
    topology_constraints_requested = False
    custom_priorities_requested = False
    enabled_effective_weights = []
    for raw_spec in port_specs:
        spec = dict(raw_spec)
        if spec.get("enabled", True):
            port = int(spec.get("port_index", 0))
            limit = int(spec.get("max_components", 2))
            bands = spec.get("bands_mhz") or [[2400.0, 2500.0]]
            raw_weights, effective_weights = _resolve_band_priorities(
                bands,
                spec.get("band_weights"),
                spec.get("port_weight", 1.0),
                require_positive=False,
            )
            spec["port_weight"] = float(spec.get("port_weight", 1.0))
            spec["band_weights"] = raw_weights
            spec["effective_band_weights"] = effective_weights
            enabled_effective_weights.extend(effective_weights)
            custom_priorities_requested = custom_priorities_requested or (
                not np.isclose(spec["port_weight"], 1.0)
                or any(not np.isclose(value, 1.0) for value in raw_weights)
            )
            normalized = normalize_allowed_topology_codes(
                spec.get("allowed_topology_codes"), limit, port_index=port
            )
            if normalized is not None:
                topology_constraints_requested = True
                spec["allowed_topology_codes"] = sorted(normalized)
        normalized_port_specs.append(spec)
    if enabled_effective_weights and not any(value > 0.0 for value in enabled_effective_weights):
        raise ValueError("at least one enabled optimization band must have a positive effective weight")
    port_specs = normalized_port_specs
    preset = get_objective_preset(objective)
    isolation_targets = isolation_targets or []
    core_isolation_targets = tuple(
        IsolationTarget(
            source_port=int(target['source_port']),
            destination_port=int(target['destination_port']),
            start_hz=float(target['band_mhz'][0]) * 1e6,
            stop_hz=float(target['band_mhz'][1]) * 1e6,
            maximum_db=float(target.get('maximum_db', -20.0)),
            weight=float(target.get('weight', 1.0)),
            average_weight=float(target.get('average_weight', 0.0)),
        )
        for target in isolation_targets
    )

    measured_fallback_reason = ""
    try:
        measured_results, measured_fallback_reason = _run_tuning_joint_measured_core(
            dut=dut,
            library=library,
            port_specs=port_specs,
            objective=objective,
            beam_width=beam_width,
            num_band_points=num_band_points,
            global_efficiency=global_efficiency,
            per_port_efficiency=per_port_efficiency,
            isolation_targets=isolation_targets,
            core_isolation_targets=core_isolation_targets,
            timeout_seconds=timeout_seconds,
            search_checkpoint=search_checkpoint,
            checkpoint_store=checkpoint_store,
            search_profile_timeout_seconds=search_profile_timeout_seconds,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        if measured_results:
            return measured_results
    except Exception as exc:  # the compatibility path must remain available
        if topology_constraints_requested or custom_priorities_requested:
            raise
        measured_fallback_reason = f"measured joint core failed: {exc}"
        logger.exception("Measured multi-port core failed; using legacy optimizer")

    if topology_constraints_requested or custom_priorities_requested:
        logger.warning(
            "Measured topology-constrained joint search returned no candidate: %s",
            measured_fallback_reason,
        )
        return {}

    # Build compatibility-engine configs from enabled ports only.
    active_port_specs = [item for item in port_specs if item.get('enabled', True)]
    port_configs = []
    all_bands_mhz = []
    for ps in active_port_specs:
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
    compatibility_timeout = (
        float(search_profile_timeout_seconds)
        if search_profile_timeout_seconds is not None
        else timeout_seconds
    )
    opt = JointMultiPortOptimizer(
        dut=dut,
        component_library=library,
        port_configs=port_configs,
        top_candidates_per_port=beam_width,
        timeout_seconds=compatibility_timeout,
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
    enabled_port_indices = set(ps.get('port_index') for ps in active_port_specs)

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
        for target in isolation_targets:
            band = target['band_mhz']
            band_freqs.extend(np.linspace(band[0] * 1e6, band[1] * 1e6, num_band_points).tolist())
        band_freqs = sorted(set(band_freqs))

        # Evaluate at each band frequency to get sweep data
        port_band_s11 = {pi: [] for pi in js.port_metrics}
        port_band_eff = {pi: [] for pi in js.port_metrics}
        isolation_freqs = []
        isolation_s_matrices = []

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
                isolation_freqs.append(freq_hz)
                isolation_s_matrices.append(sub_result['s_matrix'])
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

        isolation_evaluation = {
            'targets': [], 'penalty_db': 0.0, 'passed': True,
            'transmission_db': np.empty((0, n_ports, n_ports)),
        }
        if core_isolation_targets:
            if isolation_s_matrices:
                isolation_evaluation = evaluate_isolation_targets(
                    np.asarray(isolation_freqs),
                    np.asarray(isolation_s_matrices),
                    core_isolation_targets,
                )
            else:
                isolation_evaluation = {
                    'targets': [], 'penalty_db': 1e9, 'passed': False,
                    'transmission_db': np.empty((0, n_ports, n_ports)),
                }
        directed_isolation = {}
        transmission_db = isolation_evaluation['transmission_db']
        if len(transmission_db):
            directed_isolation = {
                f"S{destination + 1}{source + 1}": {
                    'worst_db': float(np.max(transmission_db[:, destination, source])),
                    'average_db': float(np.mean(transmission_db[:, destination, source])),
                }
                for source in range(n_ports)
                for destination in range(n_ports)
                if source != destination
            }

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
            isolation_targets=isolation_evaluation['targets'],
            isolation_penalty_db=isolation_evaluation['penalty_db'],
            isolation_constraints_passed=isolation_evaluation['passed'],
            directed_isolation_db=directed_isolation,
            efficiency_basis="legacy_joint_component_loss_estimate",
            search_diagnostics={
                "numeric_core": "legacy_joint",
                "measured_physical_search": False,
                "measured_physical_fallback_reason": (
                    measured_fallback_reason
                    or "library or request is not supported by measured joint search"
                ),
            },
        )
        candidates[idx] = result

    # Re-sort candidates by recomputed system_score (descending)
    if core_isolation_targets:
        sorted_items = sorted(
            candidates.items(),
            key=lambda x: (
                x[1].isolation_constraints_passed,
                -x[1].isolation_penalty_db,
                x[1].system_score,
            ),
            reverse=True,
        )
    else:
        sorted_items = sorted(candidates.items(), key=lambda x: x[1].system_score, reverse=True)
    candidates = {new_idx: item for new_idx, (_, item) in enumerate(sorted_items)}
    for new_idx, item in candidates.items():
        item.solution_index = new_idx

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
    global_efficiency: Optional[object] = None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
) -> dict:
    """
    Compute frequency sweep for a selected solution.

    Returns dict with:
      frequencies, s11_db, s11_magnitude, s11_real, s11_imag,
      raw_db, raw_magnitude, raw_real, raw_imag
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
    raw_real, raw_imag = [], []
    match_real, match_imag = [], []
    eff_total, eff_accepted, eff_coupling, eff_comp_loss, eff_radiation = [], [], [], [], []
    efficiency_data = (per_port_efficiency or {}).get(port_index) or global_efficiency
    radiation_values = (
        efficiency_data.get_efficiency_array(freqs)
        if efficiency_data is not None else np.ones(len(freqs), dtype=float)
    )

    for frequency_index, freq in enumerate(freqs):
        S_base = dut.get_s_matrix_interpolated(freq)

        # Raw S11
        gamma_raw = S_base[port_index, port_index] if port_index < N else 1.0 + 0.0j
        s11_raw = abs(gamma_raw)
        raw_db.append(float(-20 * np.log10(max(s11_raw, 1e-15))))
        raw_mag.append(float(s11_raw))
        raw_real.append(float(gamma_raw.real))
        raw_imag.append(float(gamma_raw.imag))

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

        gamma_matched = S[port_index, port_index] if port_index < S.shape[0] else 1.0 + 0.0j
        s11_m = abs(gamma_matched)
        match_db.append(float(-20 * np.log10(max(s11_m, 1e-15))))
        match_mag.append(float(s11_m))
        match_real.append(float(gamma_matched.real))
        match_imag.append(float(gamma_matched.imag))

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
            radiation_eff = float(np.clip(radiation_values[frequency_index], 0.0, 1.0))
            delivered_to_dut = max(0.0, accepted - coupling - comp_loss)
            total_eff = radiation_eff * delivered_to_dut

            eff_accepted.append(float(accepted * 100))
            eff_coupling.append(float(coupling * 100))
            eff_comp_loss.append(float(comp_loss * 100))
            eff_radiation.append(float(radiation_eff * 100))
            eff_total.append(float(total_eff * 100))

    result = {
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

    if include_efficiency:
        result["efficiency"] = {
            "accepted_pct": eff_accepted,
            "coupling_pct": eff_coupling,
            "component_loss_pct": eff_comp_loss,
            "radiation_pct": eff_radiation,
            "total_pct": eff_total,
            "basis": (
                "radiation_x_delivered_power"
                if efficiency_data is not None
                else "delivered_power_no_radiation_data"
            ),
        }

    return result


def compute_transmission_line_sweep(
    dut: TouchstoneData,
    result: TuningResult,
    *,
    start_hz: float,
    stop_hz: float,
    num_points: int = 201,
    use_snp_points: bool = True,
    global_efficiency=None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
) -> dict:
    """Recompute a live synthesized line/stub candidate on an arbitrary sweep."""
    context = result.yield_context or {}
    if context.get("kind") != "transmission_line":
        raise ValueError("selected result has no live transmission-line topology")
    original_port = int(context["original_port"])
    if use_snp_points:
        frequencies = np.asarray([
            frequency for frequency in dut.frequencies
            if start_hz <= frequency <= stop_hz
        ], dtype=float)
        if not len(frequencies):
            raise ValueError("no DUT frequency points fall inside the requested sweep")
    else:
        frequencies = np.linspace(start_hz, stop_hz, max(2, int(num_points)))
    terminations = {
        port: 0.0 for port in range(dut.num_ports) if port != original_port
    }
    raw = []
    matrices = []
    for frequency in frequencies:
        matrix = dut.get_s_matrix_interpolated(float(frequency))
        reduced = terminate_ports(matrix, terminations) if terminations else matrix
        matrices.append(reduced)
        raw.append(reduced[0, 0])
    radiation = {}
    efficiency_data = (per_port_efficiency or {}).get(original_port) or global_efficiency
    if efficiency_data is not None:
        radiation[0] = efficiency_data.get_efficiency_array(frequencies)
    problem = CoreProblem(
        frequencies, np.asarray(matrices),
        {0: (CoreBand(float(frequencies[0]), float(frequencies[-1])),)},
        _reference_impedance(dut, [original_port]), radiation,
    )
    topology = build_model_circuit_topology(1, context["placements"])
    physical = evaluate_physical_problem(problem, topology)
    s11 = physical.s_parameters[:, 0, 0]
    raw = np.asarray(raw)
    accepted = np.maximum(0.0, 1.0 - np.abs(s11) ** 2)
    return {
        "frequencies": frequencies.tolist(),
        "s11_db": (-20.0 * np.log10(np.maximum(np.abs(s11), 1e-15))).tolist(),
        "s11_magnitude": np.abs(s11).tolist(),
        "s11_real": s11.real.tolist(),
        "s11_imag": s11.imag.tolist(),
        "raw_db": (-20.0 * np.log10(np.maximum(np.abs(raw), 1e-15))).tolist(),
        "raw_magnitude": np.abs(raw).tolist(),
        "port_index": original_port,
        "efficiency": {
            "accepted_pct": (100.0 * accepted).tolist(),
            "coupling_pct": np.zeros(len(frequencies)).tolist(),
            "component_loss_pct": (100.0 * physical.component_loss[:, 0]).tolist(),
            "total_pct": (100.0 * physical.total_efficiency[:, 0]).tolist(),
        },
        "power_balance_error": physical.power_balance_error[:, 0].tolist(),
        "maximum_power_balance_error": float(np.max(np.abs(physical.power_balance_error))),
    }


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

def _core_s2p_from_library_component(
    component,
    library,
    frequencies_hz: np.ndarray,
    *,
    default_tolerance_fraction: float = 0.0,
) -> S2PModel:
    matrices = []
    for frequency in frequencies_hz:
        if hasattr(component, "get_s_matrix_at_freq"):
            matrix = component.get_s_matrix_at_freq(float(frequency))
        elif hasattr(component, "db_id") and hasattr(library, "get_s_matrix_at_freq"):
            matrix = library.get_s_matrix_at_freq(component.db_id, float(frequency) / 1e6)
        else:
            matrix = component.data.get_s_matrix_interpolated(float(frequency))
        if matrix is None:
            raise ValueError(
                f"no measured S-parameters for {getattr(component, 'part_number', 'component')} "
                f"at {frequency / 1e6:g} MHz"
            )
        matrices.append(np.asarray(matrix, dtype=complex))
    component_type = str(getattr(component, "component_type", "")).lower()
    kind = "L" if component_type == "inductor" else "C" if component_type == "capacitor" else None
    nominal = getattr(component, "nominal_value", None)
    nominal_value = (
        float(nominal) * (1e-9 if kind == "L" else 1e-12)
        if kind and nominal is not None else None
    )
    tolerance_pct = getattr(component, "tolerance_pct", None)
    tolerance = (
        float(tolerance_pct) / 100.0
        if tolerance_pct is not None and float(tolerance_pct) > 0
        else float(default_tolerance_fraction)
    )
    loaded_data = getattr(component, "_data", None)
    model_z0 = float(getattr(loaded_data, "reference_resistance", 50.0))
    environment_metadata = dict(getattr(component, "environment_metadata", {}) or {})
    environment_provenance = str(
        environment_metadata.get("evidence_level")
        or (getattr(component, "metadata_provenance", {}) or {}).get("tempco_ppm_per_c", "")
    )
    return S2PModel(
        name=str(getattr(component, "part_number", "measured component")),
        frequencies_hz=frequencies_hz,
        s_parameters=np.asarray(matrices),
        z0=model_z0,
        tolerance=tolerance,
        kind=kind,
        nominal_value=nominal_value,
        tempco_ppm_per_c=getattr(component, "tempco_ppm_per_c", None),
        systematic_bias_pct=getattr(component, "systematic_bias_pct", None),
        environment_provenance=environment_provenance,
    )


def _yield_component_environment(model, position: str, tolerance_model: ToleranceModel) -> dict:
    model = getattr(model, "model", model)
    kind = getattr(model, "kind", None)
    native_tempco = getattr(model, "tempco_ppm_per_c", None)
    native_bias = getattr(model, "systematic_bias_pct", None)
    global_tempco = (
        tolerance_model.inductor_tempco_ppm_per_c if kind == "L"
        else tolerance_model.capacitor_tempco_ppm_per_c if kind == "C" else 0.0
    )
    global_bias = (
        tolerance_model.inductor_bias_pct if kind == "L"
        else tolerance_model.capacitor_bias_pct if kind == "C" else 0.0
    )
    provenance = str(getattr(model, "environment_provenance", "") or "")
    return {
        "position": position,
        "part_number": str(getattr(model, "name", position)),
        "kind": kind,
        "tempco_ppm_per_c": float(native_tempco if native_tempco is not None else global_tempco),
        "tempco_source": provenance if native_tempco is not None else "request_global_fallback",
        "systematic_bias_pct": float(native_bias if native_bias is not None else global_bias),
        "bias_source": provenance if native_bias is not None else "request_global_fallback",
    }


def run_tuning_yield_analysis(
    dut: TouchstoneData,
    library,
    candidates: List[TuningResult],
    tuning_request: dict,
    *,
    solution_indices: Optional[List[int]] = None,
    samples: int = 200,
    seed: int = 1,
    distribution: str = "uniform",
    confidence_level: float = 0.95,
    minimum_total_efficiency: float = 0.5,
    minimum_average_total_efficiency: float = 0.0,
    minimum_return_loss_db: float = 6.0,
    default_tolerance_pct: float = 5.0,
    batch_correlation: float = 0.0,
    reference_temperature_c: float = 25.0,
    temperature_min_c: Optional[float] = None,
    temperature_max_c: Optional[float] = None,
    inductor_tempco_ppm_per_c: float = 0.0,
    capacitor_tempco_ppm_per_c: float = 0.0,
    inductor_bias_pct: float = 0.0,
    capacitor_bias_pct: float = 0.0,
    global_efficiency=None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
) -> dict:
    """Rank current measured fixed-network candidates by Monte Carlo yield."""
    if not candidates:
        raise ValueError("no tuning candidates are available")
    per_port_efficiency = per_port_efficiency or {}
    ports = [item for item in tuning_request.get("ports", []) if item.get("enabled", True)]
    if not ports:
        ports = [
            {"port_index": port, "bands_mhz": [[float(min(dut.frequencies)) / 1e6, float(max(dut.frequencies)) / 1e6]]}
            for port in candidates[0].port_indices
        ]
    bands_by_port = {
        int(item["port_index"]): tuple(
            CoreBand(float(band[0]) * 1e6, float(band[1]) * 1e6)
            for band in item.get("bands_mhz", [])
        )
        for item in ports
    }
    frequencies = np.asarray(sorted({
        float(frequency)
        for frequency in dut.frequencies
        if any(
            band.start_hz <= float(frequency) <= band.stop_hz
            for bands in bands_by_port.values() for band in bands
        )
    }), dtype=float)
    if not len(frequencies):
        frequencies = np.asarray(sorted({
            float(frequency)
            for candidate in candidates
            for metrics in candidate.per_port.values()
            for frequency in metrics.band_freqs_hz
        }), dtype=float)
    if not len(frequencies):
        raise ValueError("no DUT frequency points fall inside the configured bands")
    dut_s = np.asarray([
        dut.get_s_matrix_interpolated(float(frequency)) for frequency in frequencies
    ])
    radiation_efficiency = {}
    for port in bands_by_port:
        efficiency_data = per_port_efficiency.get(port) or global_efficiency
        if efficiency_data is not None:
            radiation_efficiency[port] = efficiency_data.get_efficiency_array(frequencies)
    problem = CoreProblem(
        frequencies, dut_s, bands_by_port, _reference_impedance(dut),
        radiation_efficiency,
    )
    criteria = YieldCriteria(
        minimum_total_efficiency=minimum_total_efficiency,
        minimum_return_loss_db=minimum_return_loss_db,
        minimum_average_total_efficiency=minimum_average_total_efficiency,
    )
    tolerance_model = ToleranceModel(
        batch_correlation=batch_correlation,
        reference_temperature_c=reference_temperature_c,
        temperature_min_c=temperature_min_c,
        temperature_max_c=temperature_max_c,
        inductor_tempco_ppm_per_c=inductor_tempco_ppm_per_c,
        capacitor_tempco_ppm_per_c=capacitor_tempco_ppm_per_c,
        inductor_bias_pct=inductor_bias_pct,
        capacitor_bias_pct=capacitor_bias_pct,
    )
    selected = set(solution_indices) if solution_indices is not None else set(range(len(candidates)))
    if any(index < 0 or index >= len(candidates) for index in selected):
        raise ValueError("solution index is outside the current candidate list")
    model_cache = {}
    analyses, unsupported = [], []
    for index, candidate in enumerate(candidates):
        if index not in selected:
            continue
        if candidate.mode == "transmission_line":
            context = candidate.yield_context
            if not context or not context.get("problem") or not context.get("placements"):
                unsupported.append({
                    "solution_index": index,
                    "reason": "transmission-line candidate has no live physical yield context; rerun synthesis",
                })
                continue
            line_problem = context["problem"]
            placements = tuple(context["placements"])
            topology = build_model_circuit_topology(
                line_problem.s_parameters.shape[1], placements
            )
            tolerance_result = monte_carlo_yield(
                line_problem, topology, criteria,
                samples=samples, seed=seed, distribution=distribution,
                confidence_level=confidence_level, tolerance_model=tolerance_model,
            )
            tolerance_payload = []
            for position, placement in enumerate(placements, start=1):
                model = placement.model
                line = model.line if isinstance(model, MicrostripStubModel) else model
                if isinstance(line, MicrostripLineModel):
                    for variable, value in (
                        ("trace_width", line.width_tolerance),
                        ("physical_length", line.length_tolerance),
                        ("substrate_height", line.substrate_height_tolerance),
                        ("relative_permittivity", line.relative_permittivity_tolerance),
                    ):
                        tolerance_payload.append({
                            "part_number": line.name,
                            "position": f"line_{position}",
                            "variable": variable,
                            "tolerance_pct": 100.0 * value,
                            "source": "microstrip_fabrication_config",
                        })
                else:
                    tolerance_payload.append({
                        "part_number": str(getattr(model, "name", f"line_{position}")),
                        "position": f"line_{position}",
                        "variable": "electrical_length",
                        "tolerance_pct": 100.0 * float(getattr(model, "tolerance", 0.0)),
                        "source": "line_model",
                    })
            summary = tolerance_summary(tolerance_result)
            summary.update({
                "solution_index": index,
                "nominal_score": candidate.system_score,
                "component_tolerances": tolerance_payload,
                "component_environment": [],
                "analysis_scope": "physical_transmission_line_manufacturing",
            })
            candidate.search_diagnostics = candidate.search_diagnostics or {}
            candidate.search_diagnostics["yield_analysis"] = summary
            analyses.append(summary)
            continue
        if candidate.mode == "switch":
            context = candidate.yield_context
            if not context:
                unsupported.append({
                    "solution_index": index,
                    "reason": "switch candidate has no live measured yield context; enable measured refinement and rerun optimization",
                })
                continue
            tolerance_result = monte_carlo_switch_yield(
                context["problem"],
                context["loaded_states"],
                context["branch_models"],
                context["input_elements"],
                context["state_by_configuration"],
                criteria,
                samples=samples,
                seed=seed,
                distribution=distribution,
                confidence_level=confidence_level,
                tolerance_model=tolerance_model,
            )
            summary = tolerance_summary(tolerance_result)
            summary.update({
                "solution_index": index,
                "nominal_score": candidate.system_score,
                "component_tolerances": context["component_tolerances"],
                "analysis_scope": "joint_switch_configurations",
                "component_environment": [
                    *(
                        _yield_component_environment(model, f"branch_{position + 1}", tolerance_model)
                        for position, model in enumerate(context["branch_models"])
                    ),
                    *(
                        _yield_component_environment(model, f"input_{position + 1}", tolerance_model)
                        for position, model in enumerate(context["input_elements"])
                    ),
                ],
            })
            candidate.search_diagnostics["yield_analysis"] = summary
            analyses.append(summary)
            continue
        if candidate.mode == "tunable":
            context = candidate.yield_context
            if not context:
                unsupported.append({
                    "solution_index": index,
                    "reason": "tunable candidate has no live measured yield context; rerun physical MDIF optimization",
                })
                continue
            tolerance_result = monte_carlo_tunable_yield(
                context["problem"],
                context["fixed_placements"],
                context["tuner"],
                context["state_by_configuration"],
                criteria,
                samples=samples,
                seed=seed,
                distribution=distribution,
                confidence_level=confidence_level,
                tolerance_model=tolerance_model,
            )
            summary = tolerance_summary(tolerance_result)
            summary.update({
                "solution_index": index,
                "nominal_score": candidate.system_score,
                "component_tolerances": context["component_tolerances"],
                "analysis_scope": "joint_tunable_configurations",
                "component_environment": [
                    _yield_component_environment(
                        placement.model, f"fixed_{position + 1}", tolerance_model
                    )
                    for position, placement in enumerate(context["fixed_placements"])
                ],
            })
            candidate.search_diagnostics = candidate.search_diagnostics or {}
            candidate.search_diagnostics["yield_analysis"] = summary
            analyses.append(summary)
            continue
        choices = [
            choice
            for port_choices in candidate.component_choices.values()
            for choice in port_choices
        ]
        if not choices:
            unsupported.append({
                "solution_index": index,
                "reason": "candidate has no live measured component models; rerun fixed-network optimization",
            })
            continue
        placements = []
        tolerance_payload = []
        for choice in choices:
            if choice.connection_type not in {"series", "shunt"}:
                raise ValueError(
                    f"yield analysis does not yet support {choice.connection_type!r} placements"
                )
            component = choice.component
            tolerance_pct = getattr(component, "tolerance_pct", None)
            tolerance_fraction = (
                float(tolerance_pct) / 100.0
                if tolerance_pct is not None and float(tolerance_pct) > 0
                else default_tolerance_pct / 100.0
            )
            key = (
                getattr(component, "part_number", id(component)), tolerance_fraction,
                tuple(frequencies),
            )
            if key not in model_cache:
                model_cache[key] = _core_s2p_from_library_component(
                    component,
                    library,
                    frequencies,
                    default_tolerance_fraction=tolerance_fraction,
                )
            placements.append(ModelPlacement(
                choice.connection_type, int(choice.port), model_cache[key]
            ))
            tolerance_payload.append({
                "part_number": str(getattr(component, "part_number", "component")),
                "tolerance_pct": 100.0 * tolerance_fraction,
                "source": "component_metadata" if tolerance_pct is not None else "request_default",
            })
        topology = build_model_circuit_topology(dut.num_ports, placements)
        tolerance_result = monte_carlo_yield(
            problem, topology, criteria,
            samples=samples, seed=seed, distribution=distribution,
            confidence_level=confidence_level,
            tolerance_model=tolerance_model,
        )
        summary = tolerance_summary(tolerance_result)
        summary.update({
            "solution_index": index,
            "nominal_score": candidate.system_score,
            "component_tolerances": tolerance_payload,
            "component_environment": [
                _yield_component_environment(
                    placement.model, f"component_{position + 1}", tolerance_model
                )
                for position, placement in enumerate(placements)
            ],
        })
        candidate.search_diagnostics["yield_analysis"] = summary
        analyses.append(summary)
    analyses.sort(key=lambda item: (
        item["yield_confidence_interval"][0],
        item["yield_fraction"],
        item["score_percentiles_db"]["5"],
        item["nominal_score"],
    ), reverse=True)
    for rank, item in enumerate(analyses, start=1):
        item["yield_rank"] = rank
    return {
        "status": "ok",
        "criteria": {
            "minimum_total_efficiency": minimum_total_efficiency,
            "minimum_average_total_efficiency": minimum_average_total_efficiency,
            "minimum_return_loss_db": minimum_return_loss_db,
        },
        "variation_model": tolerance_model.as_dict(),
        "frequency_points": len(frequencies),
        "ranked_candidates": analyses,
        "unsupported_candidates": unsupported,
    }


def run_tuning_tunable_mdif(
    dut: TouchstoneData,
    library,
    port_index: int,
    tuner_mdif_path: str,
    frequency_configurations: List[dict],
    fixed_components: List[dict],
    objective: str = "balanced",
) -> Dict[int, TuningResult]:
    """Evaluate a shared measured network and automatically select MDIF states."""
    frequencies, configurations, problem, core_objective = _build_core_tunable_problem(
        dut, port_index, frequency_configurations, objective
    )
    placements = []
    component_descriptions = []
    for item in fixed_components:
        connection = str(item.get("connection", "series")).lower()
        kind = str(item.get("kind", "")).upper()
        value = float(item.get("value", 0.0))
        if connection not in ("series", "shunt") or kind not in ("L", "C") or value <= 0:
            raise ValueError("tunable fixed components require series/shunt, L/C and a positive value")
        component = (
            library.find_nearest_inductor(value)
            if kind == "L"
            else library.find_nearest_capacitor(value)
        )
        if component is None:
            raise ValueError(f"no measured {kind} component is available near {value:g}")
        tolerance_pct = getattr(component, "tolerance_pct", None)
        tolerance_fraction = (
            float(tolerance_pct) / 100.0
            if tolerance_pct is not None and float(tolerance_pct) > 0
            else 0.05
        )
        model = _core_s2p_from_library_component(
            component,
            library,
            frequencies,
            default_tolerance_fraction=tolerance_fraction,
        )
        placements.append(ModelPlacement(connection, 0, model))
        component_descriptions.append({
            "position": len(component_descriptions),
            "part": getattr(component, "part_number", model.name),
            "part_number": getattr(component, "part_number", model.name),
            "type": "inductor" if kind == "L" else "capacitor",
            "comp_type": "inductor" if kind == "L" else "capacitor",
            "connection_type": connection,
            "port": 0,
            "nominal_value": float(getattr(component, "nominal_value", value)),
            "nominal_unit": "nH" if kind == "L" else "pF",
            "value": f"{getattr(component, 'nominal_value', value):g}{'nH' if kind == 'L' else 'pF'}",
            "tolerance_pct": 100.0 * tolerance_fraction,
            "tolerance_source": (
                "component_metadata"
                if tolerance_pct is not None and float(tolerance_pct) > 0
                else "request_default"
            ),
        })
    tuner: MDIFModel = load_mdif(tuner_mdif_path)
    evaluated = evaluate_tunable_physical(
        problem, tuple(placements), tuner, core_objective
    )
    return _tunable_result_from_evaluation(
        evaluated, tuner, configurations, frequency_configurations,
        frequencies, component_descriptions, objective,
        yield_context={
            "problem": problem,
            "fixed_placements": tuple(placements),
            "tuner": tuner,
            "state_by_configuration": dict(evaluated.state_by_configuration),
            "component_tolerances": [
                {
                    "position": f"fixed_{index + 1}",
                    "part_number": description["part_number"],
                    "tolerance_pct": description["tolerance_pct"],
                    "source": description["tolerance_source"],
                }
                for index, description in enumerate(component_descriptions)
            ],
        },
    )


def _build_core_tunable_problem(
    dut: TouchstoneData,
    port_index: int,
    frequency_configurations: List[dict],
    objective: str,
):
    if dut.num_ports != 1:
        raise ValueError("physical MDIF tunable mode currently supports a one-port DUT")
    if port_index != 0:
        raise ValueError("one-port MDIF evaluation requires port_index 0")
    frequencies = np.asarray(dut.frequencies, dtype=float)
    s_parameters = np.asarray(
        [dut.get_s_matrix(index) for index in range(len(frequencies))],
        dtype=complex,
    )
    configurations = []
    for item in frequency_configurations:
        bands = []
        for band in item.get("bands_mhz", []):
            if len(band) != 2 or min(band) <= 0 or band[0] == band[1]:
                raise ValueError("each tunable frequency band requires two distinct positive MHz values")
            bands.append(CoreBand(float(band[0]) * 1e6, float(band[1]) * 1e6))
        configurations.append(CoreFrequencyConfiguration(
            str(item.get("name", "")).strip(),
            {0: bands},
            float(item.get("weight", 1.0)),
        ))
    if not configurations:
        raise ValueError("physical MDIF tunable mode requires frequency_configurations")
    active_mask = np.zeros(len(frequencies), dtype=bool)
    for configuration in configurations:
        for bands in configuration.bands_by_port.values():
            for band in bands:
                active_mask |= band.mask(frequencies)
    if not np.any(active_mask):
        raise ValueError("tunable frequency configurations have no DUT frequency samples")
    frequencies = frequencies[active_mask]
    s_parameters = s_parameters[active_mask]
    base = CoreProblem(
        frequencies, s_parameters, {0: configurations[0].bands_by_port[0]},
        _reference_impedance(dut, [0]),
    )
    mode_weights = {
        "worst_case": (0.0, 0.0, 0.0, 0.0),
        "average_efficiency": (1.0, 1.0, 1.0, 1.0),
        "balanced": (0.05, 0.1, 0.1, 0.5),
    }
    within, across, ports, config_average = mode_weights.get(
        objective, mode_weights["balanced"]
    )
    core_objective = CoreObjective(within, across, ports)
    problem = CoreTunableProblem(base, tuple(configurations), config_average)
    return frequencies, configurations, problem, core_objective


def _tunable_result_from_evaluation(
    evaluated,
    tuner,
    configurations,
    frequency_configurations,
    frequencies,
    component_descriptions,
    objective,
    yield_context=None,
):
    active_efficiencies = []
    active_return_losses = []
    configuration_payload = []
    for item in evaluated.metrics["configurations"]:
        metrics = item["metrics"]
        for band_index, band in enumerate(configurations[len(configuration_payload)].bands_by_port[0]):
            mask = band.mask(frequencies)
            active_efficiencies.extend(np.asarray(metrics["total_efficiency"])[mask, 0].tolist())
            active_return_losses.extend(np.asarray(metrics["return_loss_db"])[mask, 0].tolist())
        configuration_payload.append({
            "name": item["name"],
            "state": item["state"],
            "score_db": float(item["score_db"]),
            "bands_mhz": frequency_configurations[len(configuration_payload)]["bands_mhz"],
        })
    efficiencies = np.asarray(active_efficiencies, dtype=float)
    returns = np.asarray(active_return_losses, dtype=float)
    average_efficiency = float(np.mean(efficiencies))
    minimum_efficiency = float(np.min(efficiencies))
    tuner_description = {
        "position": -1,
        "part": tuner.name,
        "part_number": tuner.name,
        "type": "tunable_capacitor",
        "comp_type": "tunable_capacitor",
        "connection_type": "series",
        "port": 0,
        "value": " / ".join(tuner_state for tuner_state in evaluated.state_by_configuration.values()),
        "states": evaluated.state_by_configuration,
    }
    per_port = PerPortTuningMetrics(
        port_index=0,
        s11_magnitude=float(10.0 ** (-float(np.min(returns)) / 20.0)),
        s11_db=float(np.min(returns)),
        accepted_efficiency=average_efficiency,
        radiated_efficiency=average_efficiency,
        component_loss=float(np.mean([
            np.mean(item["metrics"]["component_loss"][:, 0])
            for item in evaluated.metrics["configurations"]
        ])),
        total_efficiency=average_efficiency,
        components=[tuner_description, *component_descriptions],
        band_freqs_hz=[],
        band_s11_db=[],
        band_total_eff=[],
    )
    result = TuningResult(
        port_indices=[0],
        mode="tunable",
        objective=objective,
        per_port={0: per_port},
        system_score=float(evaluated.score_db),
        avg_total_efficiency=average_efficiency,
        min_total_efficiency=minimum_efficiency,
        total_component_loss=per_port.component_loss,
        total_component_count=len(component_descriptions) + 1,
        efficiency_basis="rfmatch_core_physical_mdif",
        tunable_states=evaluated.state_by_configuration,
        frequency_configurations=configuration_payload,
        maximum_power_balance_error=float(evaluated.metrics["maximum_power_balance_error"]),
    )
    result.yield_context = yield_context
    result.system_power_balance = {
        "system_efficiency": average_efficiency,
        "accepted_efficiency": average_efficiency + per_port.component_loss,
        "component_loss": per_port.component_loss,
        "maximum_balance_error": result.maximum_power_balance_error,
        "basis": result.efficiency_basis,
        "note": (
            "Average over every active MDIF configuration band; the tuner and "
            "shared fixed network are evaluated from measured S-parameters."
        ),
    }
    return {0: result}


def _load_adapter_component_model(
    component,
    frequencies_hz: np.ndarray,
    kind: str,
    value_si: float,
    tolerance: float,
) -> S2PModel:
    matrices = []
    for frequency in frequencies_hz:
        matrix = component.get_s_matrix_at_freq(float(frequency))
        if matrix is None:
            raise ValueError(
                f"component {component.part_number} has no S2P data at {frequency:g} Hz"
            )
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"component {component.part_number} returned an invalid S2P matrix"
            )
        matrices.append(matrix)
    # ComponentInfo loads its TouchstoneData on the first interpolation above.
    # Database adapters do not expose that object and use the conventional
    # 50-ohm fallback.
    loaded_data = getattr(component, "_data", None)
    model_z0 = float(getattr(loaded_data, "reference_resistance", 50.0))
    return S2PModel(
        component.part_number,
        np.asarray(frequencies_hz, dtype=float),
        np.asarray(matrices),
        model_z0,
        tolerance,
        kind,
        value_si,
    )


def _core_component_catalog(
    library,
    kind: str,
    frequencies_hz: Optional[np.ndarray] = None,
) -> list[ComponentSpec | LazyComponentSpec]:
    if library is None:
        return []
    components = (
        getattr(library, "inductors", None)
        if kind == "L"
        else getattr(library, "capacitors", None)
    ) or []
    unique = {}
    for component in components:
        if component.nominal_value <= 0:
            continue
        value_si = component.nominal_value * (1e-9 if kind == "L" else 1e-12)
        record = getattr(component, "_record", None)
        component_tolerance_pct = getattr(component, "tolerance_pct", None)
        record_tolerance_pct = getattr(record, "tolerance_pct", None)
        tolerance = float(
            getattr(component, "tolerance", 0.0)
            or (
                float(component_tolerance_pct) / 100.0
                if component_tolerance_pct is not None
                else float(record_tolerance_pct) / 100.0
                if record_tolerance_pct is not None
                else 0.05
            )
        )
        path = Path(str(getattr(component, "s2p_filename", "")))
        is_directory_model = (
            getattr(component, "zip_path", None) == "__DIR__" and path.is_file()
        )
        if is_directory_model:
            spec = ComponentSpec(
                component.part_number,
                kind,
                value_si,
                tolerance,
                path.parent.name,
                path.resolve(),
            )
        elif frequencies_hz is not None and hasattr(component, "get_s_matrix_at_freq"):
            frequencies = np.asarray(frequencies_hz, dtype=float).copy()
            provenance = "|".join(filter(None, (
                str(getattr(component, "zip_path", "")),
                str(getattr(component, "s2p_filename", "")),
                str(getattr(getattr(component, "_record", None), "id", "")),
            ))) or f"adapter:{type(component).__name__}:{component.part_number}"
            family = str(
                getattr(component, "series_name", "")
                or getattr(component, "family", "")
                or type(component).__name__
            )
            spec = LazyComponentSpec(
                component.part_number,
                kind,
                value_si,
                tolerance,
                family,
                provenance,
                lambda component=component, frequencies=frequencies,
                       kind=kind, value_si=value_si, tolerance=tolerance:
                    _load_adapter_component_model(
                        component, frequencies, kind, value_si, tolerance
                    ),
            )
        else:
            continue
        provenance = str(
            getattr(spec, "source_path", "")
            or getattr(spec, "provenance", "")
        )
        identity = (spec.name, provenance)
        previous = unique.get(identity)
        if previous is None or (spec.tolerance, spec.family) < (
            previous.tolerance, previous.family
        ):
            unique[identity] = spec
    return sorted(
        unique.values(),
        key=lambda item: (
            item.value, item.tolerance, item.name,
            str(getattr(item, "source_path", "") or getattr(item, "provenance", "")),
        ),
    )


def _run_tuning_joint_measured_core(
    dut: TouchstoneData,
    library,
    port_specs: List[dict],
    objective: str,
    beam_width: int,
    num_band_points: int,
    global_efficiency=None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
    isolation_targets: Optional[List[dict]] = None,
    core_isolation_targets: Tuple[IsolationTarget, ...] = (),
    timeout_seconds: float = 120.0,
    search_checkpoint: Optional[dict] = None,
    checkpoint_store: Optional[dict] = None,
    search_profile_timeout_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[Dict[int, TuningResult], str]:
    """Adapt product joint tuning to the full-matrix measured-S2P core."""
    enabled_specs = [item for item in port_specs if item.get("enabled", True)]
    if len(enabled_specs) < 2:
        return {}, "joint measured search requires at least two enabled ports"
    ports = [int(item.get("port_index", 0)) for item in enabled_specs]
    if len(set(ports)) != len(ports):
        return {}, "each enabled port may appear only once"
    if any(port < 0 or port >= dut.num_ports for port in ports):
        return {}, "an enabled port is outside the DUT port range"

    max_components_by_port = {
        int(item.get("port_index", 0)): int(item.get("max_components", 2))
        for item in enabled_specs
    }
    if any(limit < 0 for limit in max_components_by_port.values()):
        return {}, "maximum components cannot be negative"
    if any(limit > 6 for limit in max_components_by_port.values()):
        return {}, "measured physical joint synthesis currently supports up to six components per port"
    allowed_topology_codes_by_port = {}
    for item in enabled_specs:
        port = int(item.get("port_index", 0))
        try:
            allowed = normalize_allowed_topology_codes(
                item.get("allowed_topology_codes"),
                max_components_by_port[port],
                port_index=port,
            )
        except ValueError as exc:
            return {}, str(exc)
        if allowed is not None:
            allowed_topology_codes_by_port[port] = allowed

    points_per_band = max(1, int(num_band_points))
    frequencies_list: list[float] = []
    bands_by_port = {}
    bands_mhz_by_port = {}
    priority_weights_by_port = {}
    for item in enabled_specs:
        port = int(item.get("port_index", 0))
        bands_mhz = item.get("bands_mhz") or [[2400.0, 2500.0]]
        if not bands_mhz:
            return {}, f"port {port + 1} has no active band"
        raw_band_weights, effective_band_weights = _resolve_band_priorities(
            bands_mhz,
            item.get("band_weights"),
            item.get("port_weight", 1.0),
            require_positive=False,
        )
        parsed_bands = []
        for band, effective_weight in zip(bands_mhz, effective_band_weights):
            if len(band) != 2:
                return {}, f"port {port + 1} contains an invalid band"
            start_mhz, stop_mhz = sorted((float(band[0]), float(band[1])))
            parsed_bands.append(CoreBand(
                start_mhz * 1e6,
                stop_mhz * 1e6,
                weight=float(effective_weight),
            ))
            frequencies_list.extend(
                np.linspace(start_mhz * 1e6, stop_mhz * 1e6, points_per_band).tolist()
            )
        bands_by_port[port] = tuple(parsed_bands)
        bands_mhz_by_port[port] = [
            [band.start_hz / 1e6, band.stop_hz / 1e6] for band in parsed_bands
        ]
        priority_weights_by_port[port] = {
            "port_weight": float(item.get("port_weight", 1.0)),
            "band_weights": [float(value) for value in raw_band_weights],
            "effective_band_weights": [float(value) for value in effective_band_weights],
        }
    for target in isolation_targets or []:
        start_mhz, stop_mhz = sorted(map(float, target["band_mhz"]))
        frequencies_list.extend(
            np.linspace(start_mhz * 1e6, stop_mhz * 1e6, points_per_band).tolist()
        )
    frequencies = np.asarray(sorted(set(frequencies_list)), dtype=float)
    if len(frequencies) == 0:
        return {}, "joint measured search has no active frequency samples"

    if any(max_components_by_port.values()):
        inductors = _core_component_catalog(library, "L", frequencies)
        capacitors = _core_component_catalog(library, "C", frequencies)
    else:
        inductors, capacitors = (), ()
    available_component_kinds = frozenset(
        kind for kind, components in (("L", inductors), ("C", capacitors))
        if components
    )

    started = time.perf_counter()
    dut_matrices = np.asarray([
        dut.get_s_matrix_interpolated(float(frequency))
        for frequency in frequencies
    ])
    radiation_efficiency = {}
    for port in ports:
        efficiency_data = (per_port_efficiency or {}).get(port) or global_efficiency
        if efficiency_data is not None:
            radiation_efficiency[port] = efficiency_data.get_efficiency_array(frequencies)
    problem = CoreProblem(
        frequencies,
        dut_matrices,
        bands_by_port,
        _reference_impedance(dut),
        radiation_efficiency,
        core_isolation_targets,
    )
    within_weight = {
        "worst_case": 0.0,
        "average_efficiency": 1.0,
        "balanced": 0.05,
    }.get(objective, 0.05)
    # Output depth and independent-port shortlist width are different knobs.
    # Letting a top-50 UI request create 50^N joint combinations makes runtime
    # explode without calibrated recall benefit; official exhaustive and saved
    # winner baselines support a bounded 8–12 candidate port frontier.
    per_port_keep = max(8, min(12, int(beam_width)))
    profile_timeout = float(
        timeout_seconds
        if search_profile_timeout_seconds is None
        else search_profile_timeout_seconds
    )
    if profile_timeout < 30:
        search_profile = "time_budgeted"
        ideal_restarts, ideal_iterations, ideal_keep = 2, 6, 16
        refine_seeds = 2
    elif profile_timeout < 60:
        search_profile = "balanced"
        ideal_restarts, ideal_iterations, ideal_keep = 2, 8, 24
        refine_seeds = 3
    else:
        search_profile = "thorough"
        ideal_restarts, ideal_iterations, ideal_keep = 4, 12, 32
        refine_seeds = 4
    deadline = max(
        started + max(0.0, float(timeout_seconds)),
        time.perf_counter() + 0.05,
    )
    should_cancel = lambda: (
        time.perf_counter() >= deadline
        or (cancel_check is not None and cancel_check())
    )
    topology_beam_width = max(8, min(128, int(beam_width) * 2))
    deep_discrete_topology_seeds = min(32, max(4, int(beam_width)))
    complete_topology_constraint = (
        len(allowed_topology_codes_by_port) == len(ports)
        and all(
            any(code != "0" for code in allowed_topology_codes_by_port[port])
            for port in ports
        )
    )
    constrained_coupled_search = bool(
        complete_topology_constraint
        and profile_timeout >= 60
        and max(max_components_by_port.values()) <= 2
    )
    automatic_topology_search = bool(
        not allowed_topology_codes_by_port
        and profile_timeout >= 150
        and len(ports) <= 3
        and max(max_components_by_port.values()) <= 2
    )
    coupled_block_search = bool(
        constrained_coupled_search or automatic_topology_search
    )
    if coupled_block_search:
        # Keep the constrained coupled profile identical to the checked-in
        # full-catalog Optenni replay.  A wider independent-port frontier does
        # not imply better joint recall: it changes the sole block-refinement
        # seed and previously displaced the saved BOM despite doing more work.
        per_port_keep = 13 if automatic_topology_search else 8
        ideal_restarts, ideal_iterations, ideal_keep = 2, 8, 32
    constrained_topologies_per_port = max(
        (
            len([
                code for code in allowed_topology_codes_by_port.get(port, ())
                if code != "0"
            ])
            for port in ports
        ),
        default=0,
    )
    core_objective = CoreObjective(
            within_band_average_weight=within_weight,
            across_band_average_weight=0.1,
            port_average_weight=0.1,
        )
    search_config = MeasuredSearchConfig(
            ideal_restarts=ideal_restarts,
            ideal_iterations=ideal_iterations,
            ideal_keep=ideal_keep,
            nearest_parts=2,
            per_port_keep=per_port_keep,
            result_keep=max(50, int(beam_width)),
            joint_refine_seeds=1 if coupled_block_search else refine_seeds,
            joint_refine_passes=3 if coupled_block_search else 1,
            joint_refine_neighbors=8,
            joint_refine_variants_per_value=2 if coupled_block_search else 1,
            joint_refine_beam_width=(4 if automatic_topology_search else 8)
            if coupled_block_search else 1,
            joint_refine_port_blocks=coupled_block_search,
            joint_ideal_topologies_per_port=(
                13 if automatic_topology_search
                else min(2, constrained_topologies_per_port)
                if constrained_coupled_search else 0
            ),
            joint_ideal_combination_keep=(48 if automatic_topology_search else 8),
            joint_ideal_rank_combinations=automatic_topology_search,
            joint_ideal_diverse_combinations=automatic_topology_search,
            joint_ideal_refine_topology_neighbors=automatic_topology_search,
            joint_ideal_growth_refine_keep=(8 if automatic_topology_search else 0),
            joint_ideal_growth_refine_restarts=8,
            joint_ideal_growth_refine_iterations=20,
            joint_ideal_growth_refine_nearest_parts=(
                1 if automatic_topology_search else 2
            ),
            joint_ideal_restarts=(1 if automatic_topology_search else 8)
            if coupled_block_search else 1,
            joint_ideal_iterations=(8 if automatic_topology_search else 20)
            if coupled_block_search else 6,
            joint_ideal_keep=(2 if automatic_topology_search else 4)
            if coupled_block_search else 2,
            joint_ideal_nearest_parts=(1 if automatic_topology_search else 2),
            seed=1,
            max_components_per_port=max(max_components_by_port.values()),
            max_components_by_port=max_components_by_port,
            topology_beam_width=topology_beam_width,
            deep_discrete_topology_seeds=deep_discrete_topology_seeds,
            allowed_topology_codes_by_port=(
                allowed_topology_codes_by_port or None
            ),
            available_component_kinds=available_component_kinds,
        )
    resumed = False
    search_optimizer = None
    if search_checkpoint is not None and search_checkpoint.get("kind") == "joint_measured_s2p":
        candidate_optimizer = search_checkpoint.get("optimizer")
        if isinstance(candidate_optimizer, MeasuredComponentOptimizer):
            search_optimizer = candidate_optimizer
            previous_config = search_optimizer.config
            previous_ideal_profile = (
                previous_config.ideal_restarts,
                previous_config.ideal_iterations,
                previous_config.ideal_keep,
                previous_config.topology_beam_width,
                previous_config.allowed_topology_codes_by_port,
                previous_config.available_component_kinds,
            )
            next_ideal_profile = (
                search_config.ideal_restarts,
                search_config.ideal_iterations,
                search_config.ideal_keep,
                search_config.topology_beam_width,
                search_config.allowed_topology_codes_by_port,
                search_config.available_component_kinds,
            )
            if previous_ideal_profile != next_ideal_profile:
                search_optimizer.ideal_candidates_by_port.clear()
            search_optimizer.config = search_config
            search_optimizer.cancel_check = should_cancel
            search_optimizer.progress_callback = progress_callback
            resumed = True
    if search_optimizer is None:
        search_optimizer = MeasuredComponentOptimizer(
            problem,
            inductors,
            capacitors,
            core_objective,
            search_config,
            cancel_check=should_cancel,
            progress_callback=progress_callback,
        )
    prior_ideal_evaluations = search_optimizer.ideal_evaluations
    prior_physical_evaluations = search_optimizer.physical_evaluations
    prior_exact_cache_entries = len(search_optimizer.evaluation_cache)
    search = search_optimizer.optimize()
    if checkpoint_store is not None:
        checkpoint_store.update({
            "kind": "joint_measured_s2p",
            "optimizer": search_optimizer,
            "resumed": resumed,
            "prior_ideal_evaluations": prior_ideal_evaluations,
            "prior_physical_evaluations": prior_physical_evaluations,
            "prior_exact_cache_entries": prior_exact_cache_entries,
        })
    selected = [
        candidate
        for candidate in search.candidates
        if all(
            sum(item.port == port for item in candidate.placements) <= limit
            for port, limit in max_components_by_port.items()
        )
    ][:max(1, int(beam_width))]
    if not selected:
        return {}, "measured joint search produced no candidate within the per-port limits"

    originals = {
        component.part_number: component
        for component in [
            *((getattr(library, "inductors", None) or []) if library is not None else []),
            *((getattr(library, "capacitors", None) or []) if library is not None else []),
        ]
    }
    backend_names = sorted({
        "adapter" if isinstance(item, LazyComponentSpec) else "file"
        for item in [*inductors, *capacitors]
    })
    calibration_reference = multiport_calibration_reference()
    results = {}
    for index, candidate in enumerate(selected):
        metrics = candidate.metrics
        matched = np.asarray(metrics["s_parameters"], dtype=complex)
        total_efficiency = np.asarray(metrics["total_efficiency"], dtype=float)
        component_loss = np.asarray(metrics["component_loss"], dtype=float)
        dut_absorbed = np.asarray(metrics["dut_absorbed_power"], dtype=float)
        choices_by_port = {port: [] for port in ports}
        mapping_failed = False
        for position, placement in enumerate(candidate.placements):
            component = originals.get(placement.component.name)
            if component is None:
                mapping_failed = True
                break
            choices_by_port[placement.port].append(GridChoice(
                position=position,
                component=component,
                connection_type=placement.connection,
                port=placement.port,
            ))
        if mapping_failed:
            continue

        per_port = {}
        active_total, active_coupling, active_component_loss = [], [], []
        for port in ports:
            mask = np.zeros(len(frequencies), dtype=bool)
            for band in bands_by_port[port]:
                mask |= band.mask(frequencies)
            port_frequencies = frequencies[mask]
            reflection = np.abs(matched[mask, port, port])
            return_loss_db = -20.0 * np.log10(np.maximum(reflection, 1e-15))
            port_component_loss = component_loss[mask, port]
            port_absorbed = dut_absorbed[mask, port]
            port_total = total_efficiency[mask, port]
            port_coupling = np.maximum(
                0.0, 1.0 - reflection ** 2 - port_component_loss - port_absorbed
            )
            center_frequency = np.mean([
                value for band in bands_by_port[port]
                for value in (band.start_hz, band.stop_hz)
            ])
            center = int(np.argmin(np.abs(port_frequencies - center_frequency)))
            choices = choices_by_port[port]
            per_port[port] = PerPortTuningMetrics(
                port_index=port,
                s11_magnitude=float(reflection[center]),
                s11_db=float(return_loss_db[center]),
                accepted_efficiency=float(1.0 - reflection[center] ** 2),
                coupling_loss=float(port_coupling[center]),
                component_loss=float(port_component_loss[center]),
                radiated_efficiency=float(port_absorbed[center]),
                total_efficiency=float(port_total[center]),
                components=[_component_choice_to_dict(choice) for choice in choices],
                band_freqs_hz=port_frequencies.tolist(),
                band_s11_db=return_loss_db.tolist(),
                band_total_eff=port_total.tolist(),
            )
            active_total.extend(port_total.tolist())
            active_coupling.extend(port_coupling.tolist())
            active_component_loss.extend(port_component_loss.tolist())

        transmission_db = np.asarray(metrics["transmission_db"], dtype=float)
        directed_isolation = {
            f"S{destination + 1}{source + 1}": {
                "worst_db": float(np.max(transmission_db[:, destination, source])),
                "average_db": float(np.mean(transmission_db[:, destination, source])),
            }
            for source in range(dut.num_ports)
            for destination in range(dut.num_ports)
            if source != destination
        }
        evaluated_targets = []
        for target_index, target_result in enumerate(metrics["isolation_targets"]):
            enriched = dict(target_result)
            if target_index < len(isolation_targets or []):
                enriched["band_mhz"] = list(
                    (isolation_targets or [])[target_index]["band_mhz"]
                )
            evaluated_targets.append(enriched)

        result = TuningResult(
            port_indices=ports,
            mode="joint",
            objective=objective,
            per_port=per_port,
            system_score=float(candidate.score_db),
            avg_total_efficiency=float(np.mean(active_total)),
            min_total_efficiency=float(np.min(active_total)),
            avg_coupling_loss=float(np.mean(active_coupling)),
            max_coupling_loss=float(np.max(active_coupling)),
            total_component_loss=float(np.mean(active_component_loss)),
            total_component_count=sum(map(len, choices_by_port.values())),
            component_choices=choices_by_port,
            sweep_freqs_hz=frequencies.tolist(),
            total_time_s=time.perf_counter() - started,
            num_solutions_found=len(selected),
            solution_index=index,
            efficiency_basis=(
                "rfmatch_core_physical_measured_s2p_joint"
                if any(max_components_by_port.values())
                else "rfmatch_core_physical_bare_dut_joint"
            ),
            isolation_targets=evaluated_targets,
            isolation_penalty_db=float(metrics["isolation_penalty_db"]),
            isolation_constraints_passed=bool(metrics["isolation_constraints_passed"]),
            directed_isolation_db=directed_isolation,
            maximum_power_balance_error=float(metrics["maximum_power_balance_error"]),
            search_diagnostics={
                "numeric_core": "rfmatch_core",
                "measured_physical_search": bool(
                    any(max_components_by_port.values()) and available_component_kinds
                ),
                "bare_dut_core_baseline": not any(max_components_by_port.values()),
                "available_component_kinds": sorted(available_component_kinds),
                "component_catalog_search_unavailable": bool(
                    any(max_components_by_port.values())
                    and not available_component_kinds
                ),
                "search_mode": "joint_full_matrix",
                "search_profile": search_profile,
                "objective_weights": {
                    "within_band_average": within_weight,
                    "across_band_average": 0.1,
                    "port_average": 0.1,
                },
                "priority_weights_by_port": {
                    str(port): {**values, "semantics": "band_margin_multiplier"}
                    for port, values in priority_weights_by_port.items()
                },
                "ideal_evaluations": search.ideal_evaluations,
                "physical_evaluations": search.physical_evaluations,
                "stage_physical_evaluations": search.stage_physical_evaluations,
                "component_models_loaded": search.loaded_component_models,
                "component_catalog_size": {
                    "inductors": len(inductors), "capacitors": len(capacitors)
                },
                "component_model_backends": backend_names,
                "reference_impedances_ohm": _reference_impedance_values(dut).tolist(),
                "active_frequency_points": len(frequencies),
                "maximum_components_searched": max(max_components_by_port.values()),
                "maximum_components_by_port": {
                    str(port): limit for port, limit in max_components_by_port.items()
                },
                "allowed_topology_codes_by_port": {
                    str(port): sorted(codes)
                    for port, codes in allowed_topology_codes_by_port.items()
                },
                "topology_search": "progressive_alternating_ladder" if max(max_components_by_port.values()) > 4 else "calibrated_fixed_grammar",
                "topology_beam_width": topology_beam_width if max(max_components_by_port.values()) > 4 else None,
                "deep_discrete_topology_seeds": deep_discrete_topology_seeds if max(max_components_by_port.values()) > 4 else None,
                "coupled_ideal_topology_search": coupled_block_search,
                "automatic_topology_deep_search": automatic_topology_search,
                "joint_refine_port_blocks": coupled_block_search,
                "joint_refine_beam_width": search_config.joint_refine_beam_width,
                "joint_refine_variants_per_value": search_config.joint_refine_variants_per_value,
                "bands_mhz_by_port": {
                    str(port): bands for port, bands in bands_mhz_by_port.items()
                },
                "per_port_keep": per_port_keep,
                "topology_code": candidate.topology_code,
                "timeout_seconds_requested": float(timeout_seconds),
                "search_profile_timeout_seconds": profile_timeout,
                "search_truncated": search.truncated,
                "termination_reason": search.termination_reason,
                "checkpoint_reused": resumed,
                "checkpoint_prior_ideal_evaluations": prior_ideal_evaluations,
                "checkpoint_prior_physical_evaluations": prior_physical_evaluations,
                "checkpoint_prior_exact_cache_entries": prior_exact_cache_entries,
                "calibration_reference": calibration_reference,
            },
        )
        result.system_power_balance = {
            "system_efficiency": result.avg_total_efficiency,
            "component_loss": result.total_component_loss,
            "maximum_balance_error": result.maximum_power_balance_error,
            "basis": result.efficiency_basis,
            "note": (
                "Full multi-port DUT coupling and measured matching-component "
                "loss are solved together over every active frequency point."
            ),
        }
        results[index] = result
    if not results:
        return {}, "measured joint candidates could not be mapped to runtime components"
    return results, ""


def _run_tuning_single_measured_core(
    dut: TouchstoneData,
    library,
    port_index: int,
    bands_mhz: List[List[float]],
    band_weights: List[float],
    port_weight: float,
    effective_band_weights: List[float],
    frequencies_hz: List[float],
    max_components: int,
    objective: str,
    within_band_average_weight: Optional[float],
    across_band_average_weight: Optional[float],
    generic_synthesis_loss: dict,
    beam_width: int,
    timeout_seconds: float = 60.0,
    global_efficiency=None,
    per_port_efficiency: Optional[Dict[int, object]] = None,
    loss_aware_seed: Optional[dict] = None,
    search_checkpoint: Optional[dict] = None,
    checkpoint_store: Optional[dict] = None,
    search_profile_timeout_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    allowed_topology_codes: Optional[frozenset[str]] = None,
    include_zero_component: bool = True,
) -> Tuple[Dict[int, TuningResult], str]:
    """Run exact full-band S2P ranking when the library exposes local models."""
    started = time.perf_counter()
    if max_components > 6:
        return {}, "measured physical synthesis currently supports up to six components"
    if max_components < 0:
        return {}, "maximum components cannot be negative"
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if max_components == 0:
        inductors, capacitors = (), ()
    else:
        inductors = _core_component_catalog(library, "L", frequencies)
        capacitors = _core_component_catalog(library, "C", frequencies)
    available_component_kinds = frozenset(
        kind for kind, components in (("L", inductors), ("C", capacitors))
        if components
    )

    matrices = [
        dut.get_s_matrix_interpolated(float(frequency))
        for frequency in frequencies
    ]
    radiation_efficiency = {}
    efficiency_data = (per_port_efficiency or {}).get(port_index) or global_efficiency
    if efficiency_data is not None:
        radiation_efficiency[port_index] = efficiency_data.get_efficiency_array(frequencies)
    problem = CoreProblem(
        frequencies,
        np.asarray(matrices),
        {port_index: tuple(
            CoreBand(
                float(start) * 1e6,
                float(stop) * 1e6,
                weight=float(weight),
            )
            for (start, stop), weight in zip(bands_mhz, effective_band_weights)
        )},
        _reference_impedance(dut),
        radiation_efficiency,
    )
    within_weight, across_weight = _resolve_single_objective_weights(
        objective, within_band_average_weight, across_band_average_weight
    )
    topology_beam_width = max(8, min(128, int(beam_width) * 2))
    deep_discrete_topology_seeds = min(32, max(4, int(beam_width)))
    profile_timeout = float(
        timeout_seconds
        if search_profile_timeout_seconds is None
        else search_profile_timeout_seconds
    )
    if max_components > 4:
        search_profile = "progressive_deep"
        # Deep searches use one deterministic nested backbone. More time only
        # expands additional exact discrete neighborhoods/refinement seeds, so
        # a longer budget remains a superset of a shorter run.
        ideal_restarts, ideal_iterations, ideal_keep = 1, 3, 12
        if profile_timeout >= 30:
            deep_discrete_topology_seeds = 32
            refine_seeds = 4
        elif profile_timeout >= 15:
            deep_discrete_topology_seeds = min(32, max(8, int(beam_width) * 2))
            refine_seeds = 2
        else:
            refine_seeds = 1
    elif profile_timeout < 10:
        search_profile = "time_budgeted"
        ideal_restarts, ideal_iterations, ideal_keep = 1, 3, 12
        refine_seeds = 1
    elif profile_timeout < 30:
        search_profile = "balanced"
        ideal_restarts, ideal_iterations, ideal_keep = 2, 6, 24
        refine_seeds = 2
    else:
        search_profile = "thorough"
        ideal_restarts, ideal_iterations, ideal_keep = 4, 12, 32
        refine_seeds = 4
    deadline = max(
        started + max(0.0, float(timeout_seconds)),
        time.perf_counter() + 0.05,
    )
    should_cancel = lambda: (
        time.perf_counter() >= deadline
        or (cancel_check is not None and cancel_check())
    )
    core_objective = CoreObjective(
            within_band_average_weight=within_weight,
            across_band_average_weight=across_weight,
            port_average_weight=0.0,
        )
    search_config = MeasuredSearchConfig(
            ideal_restarts=ideal_restarts,
            ideal_iterations=ideal_iterations,
            ideal_keep=ideal_keep,
            nearest_parts=2,
            per_port_keep=max(8, beam_width),
            result_keep=max(20, beam_width),
            joint_refine_seeds=refine_seeds,
            joint_refine_passes=1,
            # A two-part ladder often needs both values to move together.
            # Coordinate-only refinement can get trapped when the ideal-to-
            # measured seed lands several catalog steps from the physical
            # optimum (the official Optenni Quick Start is one such case).
            # A bounded 12x12 port-block neighborhood is still inexpensive,
            # but preserves discovery of those coupled value moves.
            joint_refine_neighbors=12 if max_components == 2 else 8,
            joint_refine_port_blocks=max_components == 2,
            joint_refine_port_block_max_components=2,
            seed=1,
            max_components_per_port=max_components,
            topology_beam_width=topology_beam_width,
            deep_discrete_topology_seeds=deep_discrete_topology_seeds,
            allowed_topology_codes=allowed_topology_codes,
            include_zero_component=include_zero_component,
            available_component_kinds=available_component_kinds,
        )
    resumed = False
    search_optimizer = None
    if search_checkpoint is not None and search_checkpoint.get("kind") == "single_measured_s2p":
        candidate_optimizer = search_checkpoint.get("optimizer")
        if isinstance(candidate_optimizer, MeasuredComponentOptimizer):
            search_optimizer = candidate_optimizer
            previous_config = search_optimizer.config
            previous_ideal_profile = (
                previous_config.ideal_restarts,
                previous_config.ideal_iterations,
                previous_config.ideal_keep,
                previous_config.topology_beam_width,
                previous_config.allowed_topology_codes,
                previous_config.include_zero_component,
                previous_config.available_component_kinds,
            )
            next_ideal_profile = (
                search_config.ideal_restarts,
                search_config.ideal_iterations,
                search_config.ideal_keep,
                search_config.topology_beam_width,
                search_config.allowed_topology_codes,
                search_config.include_zero_component,
                search_config.available_component_kinds,
            )
            if previous_ideal_profile != next_ideal_profile:
                search_optimizer.ideal_candidates_by_port.clear()
            search_optimizer.config = search_config
            search_optimizer.cancel_check = should_cancel
            search_optimizer.progress_callback = progress_callback
            resumed = True
    if search_optimizer is None:
        search_optimizer = MeasuredComponentOptimizer(
            problem,
            inductors,
            capacitors,
            core_objective,
            search_config,
            cancel_check=should_cancel,
            progress_callback=progress_callback,
        )
    prior_ideal_evaluations = search_optimizer.ideal_evaluations
    prior_physical_evaluations = search_optimizer.physical_evaluations
    prior_exact_cache_entries = len(search_optimizer.evaluation_cache)
    search = search_optimizer.optimize()
    if checkpoint_store is not None:
        checkpoint_store.update({
            "kind": "single_measured_s2p",
            "optimizer": search_optimizer,
            "resumed": resumed,
            "prior_ideal_evaluations": prior_ideal_evaluations,
            "prior_physical_evaluations": prior_physical_evaluations,
            "prior_exact_cache_entries": prior_exact_cache_entries,
        })
    selected = [
        candidate for candidate in search.candidates
        if len(candidate.placements) <= max_components
    ][:beam_width]
    if not selected:
        return {}, "measured physical synthesis produced no candidate within the component limit"

    originals = {
        component.part_number: component
        for component in [
            *(getattr(library, "inductors", []) or []),
            *(getattr(library, "capacitors", []) or []),
        ]
    }
    calibration_reference = single_port_calibration_reference()
    results = {}
    for index, candidate in enumerate(selected):
        metrics = candidate.metrics
        total_efficiency = np.asarray(
            metrics["total_efficiency"], dtype=float
        )[:, port_index]
        s11_magnitude = np.abs(
            np.asarray(metrics["s_parameters"])[:, port_index, port_index]
        )
        return_loss_db = -20.0 * np.log10(np.maximum(s11_magnitude, 1e-15))
        component_loss = np.asarray(
            metrics["component_loss"], dtype=float
        )[:, port_index]
        dut_absorbed = np.asarray(
            metrics["dut_absorbed_power"], dtype=float
        )[:, port_index]
        coupling_loss = np.maximum(
            0.0,
            1.0 - s11_magnitude ** 2 - component_loss - dut_absorbed,
        )
        center = int(np.argmin(np.abs(frequencies - np.mean(frequencies))))
        choices = []
        for position, placement in enumerate(candidate.placements):
            component = originals.get(placement.component.name)
            if component is None:
                continue
            choices.append(GridChoice(
                position=position,
                component=component,
                connection_type=placement.connection,
                port=port_index,
            ))
        if len(choices) != len(candidate.placements):
            continue
        per_port = PerPortTuningMetrics(
            port_index=port_index,
            s11_magnitude=float(s11_magnitude[center]),
            s11_db=float(return_loss_db[center]),
            accepted_efficiency=float(1.0 - s11_magnitude[center] ** 2),
            coupling_loss=float(coupling_loss[center]),
            component_loss=float(component_loss[center]),
            radiated_efficiency=float(dut_absorbed[center]),
            total_efficiency=float(total_efficiency[center]),
            components=[_component_choice_to_dict(choice) for choice in choices],
            band_freqs_hz=frequencies.tolist(),
            band_s11_db=return_loss_db.tolist(),
            band_total_eff=total_efficiency.tolist(),
        )
        result = TuningResult(
            port_indices=[port_index],
            mode="single",
            objective=objective,
            per_port={port_index: per_port},
            system_score=float(candidate.score_db),
            avg_total_efficiency=float(np.mean(total_efficiency)),
            min_total_efficiency=float(np.min(total_efficiency)),
            avg_coupling_loss=float(np.mean(coupling_loss)),
            max_coupling_loss=float(np.max(coupling_loss)),
            total_component_loss=float(np.mean(component_loss)),
            total_component_count=len(choices),
            component_choices={port_index: choices},
            total_time_s=time.perf_counter() - started,
            num_solutions_found=len(selected),
            solution_index=index,
            efficiency_basis=(
                "rfmatch_core_physical_bare_dut"
                if max_components == 0
                else "rfmatch_core_physical_measured_s2p"
            ),
            maximum_power_balance_error=float(metrics["maximum_power_balance_error"]),
            search_diagnostics={
                "numeric_core": "rfmatch_core",
                "physical_core_evaluation": True,
                "loss_aware_ideal_seed": loss_aware_seed,
                "generic_synthesis_loss": generic_synthesis_loss,
                "measured_physical_search": max_components > 0 and bool(available_component_kinds),
                "bare_dut_core_baseline": max_components == 0,
                "available_component_kinds": sorted(available_component_kinds),
                "search_profile": search_profile,
                "objective_weights": {
                    "within_band_average": within_weight,
                    "across_band_average": across_weight,
                    "port_average": 0.0,
                },
                "priority_weights": {
                    "port_weight": float(port_weight),
                    "band_weights": [float(value) for value in band_weights],
                    "effective_band_weights": [
                        float(value) for value in effective_band_weights
                    ],
                    "semantics": "band_margin_multiplier",
                },
                "allowed_topology_codes_by_port": (
                    {str(port_index): sorted(allowed_topology_codes)}
                    if allowed_topology_codes is not None else {}
                ),
                "component_catalog_search_unavailable": (
                    max_components > 0 and not available_component_kinds
                ),
                "ideal_evaluations": search.ideal_evaluations,
                "physical_evaluations": search.physical_evaluations,
                "stage_physical_evaluations": search.stage_physical_evaluations,
                "joint_refine_neighbors": search_config.joint_refine_neighbors,
                "joint_refine_port_blocks": search_config.joint_refine_port_blocks,
                "joint_refine_port_block_max_components": (
                    search_config.joint_refine_port_block_max_components
                ),
                "component_models_loaded": search.loaded_component_models,
                "component_catalog_size": {
                    "inductors": len(inductors),
                    "capacitors": len(capacitors),
                },
                "reference_impedances_ohm": _reference_impedance_values(dut).tolist(),
                "component_model_backends": sorted({
                    "adapter" if isinstance(item, LazyComponentSpec) else "file"
                    for item in [*inductors, *capacitors]
                }),
                "active_frequency_points": len(frequencies),
                "maximum_components_searched": max_components,
                "topology_search": "progressive_alternating_ladder" if max_components > 4 else "calibrated_fixed_grammar",
                "topology_beam_width": topology_beam_width if max_components > 4 else None,
                "deep_discrete_topology_seeds": deep_discrete_topology_seeds if max_components > 4 else None,
                "timeout_seconds_requested": float(timeout_seconds),
                "search_profile_timeout_seconds": profile_timeout,
                "search_truncated": search.truncated,
                "termination_reason": search.termination_reason,
                "topology_code": candidate.topology_code,
                "checkpoint_reused": resumed,
                "checkpoint_prior_ideal_evaluations": prior_ideal_evaluations,
                "checkpoint_prior_physical_evaluations": prior_physical_evaluations,
                "checkpoint_prior_exact_cache_entries": prior_exact_cache_entries,
                "calibration_reference": calibration_reference,
            },
        )
        result.system_power_balance = {
            "system_efficiency": result.avg_total_efficiency,
            "component_loss": result.total_component_loss,
            "maximum_balance_error": result.maximum_power_balance_error,
            "basis": result.efficiency_basis,
            "note": "Full-band DUT and matching components evaluated from measured S-parameters.",
        }
        results[index] = result
    if not results:
        return {}, "measured candidates could not be mapped back to runtime library components"
    return results, ""


def run_tuning_tunable_mdif_auto(
    dut: TouchstoneData,
    library,
    port_index: int,
    tuner_mdif_path: str,
    frequency_configurations: List[dict],
    objective: str = "balanced",
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[int, TuningResult]:
    """Automatically synthesize topology, measured fixed parts and MDIF states."""
    started = time.perf_counter()
    _, _, problem, core_objective = _build_core_tunable_problem(
        dut, port_index, frequency_configurations, objective
    )
    inductors = _core_component_catalog(library, "L")
    capacitors = _core_component_catalog(library, "C")
    if not inductors or not capacitors:
        raise ValueError(
            "automatic MDIF synthesis requires directory-backed measured inductor and capacitor S2P models"
        )
    search = TunableMeasuredComponentOptimizer(
        problem,
        load_mdif(tuner_mdif_path),
        inductors,
        capacitors,
        core_objective,
        MeasuredSearchConfig(
            ideal_restarts=4,
            ideal_iterations=12,
            ideal_keep=24,
            nearest_parts=6,
            result_keep=20,
            joint_refine_seeds=0,
            joint_refine_passes=0,
            seed=1,
        ),
        exact_seed_keep=16,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    ).optimize()
    fixed_components = [
        {
            "connection": placement.connection,
            "kind": placement.component.kind,
            "value": placement.component.value * (1e9 if placement.component.kind == "L" else 1e12),
        }
        for placement in search.best.placements
    ]
    results = run_tuning_tunable_mdif(
        dut,
        library,
        port_index,
        tuner_mdif_path,
        frequency_configurations,
        fixed_components,
        objective,
    )
    result = results[0]
    result.total_time_s = time.perf_counter() - started
    result.num_solutions_found = len(search.candidates)
    result.search_diagnostics = {
        "mode": "tunable_mdif_auto_synthesis",
        "ideal_evaluations": search.ideal_evaluations,
        "exact_physical_evaluations": search.exact_physical_evaluations,
        "tuner_state_precomputations": search.tuner_state_precomputations,
        "ideal_frequency_points": search.ideal_frequency_points,
        "component_models_loaded": search.loaded_component_models,
        "full_dut_frequency_points": len(dut.frequencies),
        "active_frequency_points": len(problem.base_problem.frequencies_hz),
        "fixed_network": [
            {
                "connection": placement.connection,
                "kind": placement.component.kind,
                "part_number": placement.component.name,
                "value": placement.component.value_display,
            }
            for placement in search.best.placements
        ],
    }
    return results


def run_tuning_switch_mdif_auto(
    dut: TouchstoneData,
    port_index: int,
    switch_mdif_path: str,
    frequency_configurations: List[dict],
    objective: str = "balanced",
    state_options_by_configuration: Optional[Dict[str, List[str]]] = None,
    library=None,
    measured_refine: bool = False,
    max_input_components: int = 2,
    progress_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[int, TuningResult]:
    """Synthesize switch networks and return one real solution per complexity."""
    started = time.perf_counter()
    frequencies, configurations, tunable_problem, core_objective = _build_core_tunable_problem(
        dut, port_index, frequency_configurations, objective
    )
    switch = load_mdif(switch_mdif_path)
    problem = SwitchTunableProblem(
        frequencies, tunable_problem.base_problem.s_parameters[:, 0, 0],
        tuple(configurations), _reference_impedance(dut, [0]),
        tunable_problem.configuration_average_weight,
        state_options_by_configuration or {},
    )
    optimizer = SwitchTunableOptimizer(
        problem, switch, core_objective,
        SwitchSearchConfig(restarts=2, iterations=10, keep=8, seed=1),
        progress_callback=progress_callback, cancel_check=cancel_check,
    )
    search = optimizer.optimize_full_network(
        max_input_elements=max_input_components, coarse_iterations=3
    )
    measured_search = None
    if measured_refine:
        if library is None:
            raise ValueError("measured switch refinement requires a configured component library")
        inductors = _core_component_catalog(library, "L")
        capacitors = _core_component_catalog(library, "C")
        if not inductors or not capacitors:
            raise ValueError(
                "measured switch refinement requires directory-backed 0402 inductor and capacitor S2P models"
            )
        measured_search = SwitchMeasuredComponentOptimizer(
            optimizer, inductors, capacitors,
            nearest_parts=3, ideal_seed_keep=3, result_keep=8,
        ).optimize(search.candidates)

    source_candidates = measured_search.candidates if measured_search else search.candidates
    best_by_complexity = {}
    for candidate in source_candidates:
        inputs = candidate.input_components if measured_search else candidate.input_reactances
        complexity = len(inputs)
        current = best_by_complexity.get(complexity)
        if current is None or candidate.score_db > current.score_db:
            best_by_complexity[complexity] = candidate
    selected_candidates = sorted(best_by_complexity.values(), key=lambda item: item.score_db, reverse=True)
    selected_states = {
        state_label
        for candidate in selected_candidates
        for state_label in candidate.state_by_configuration.values()
    }
    loaded_by_state = {
        state.label: preload_switch_state(
            frequencies, problem.dut_s11, state,
            common_port=int(switch.metadata.get("commonPort", "1")) - 1,
            z0=problem.z0,
        )
        for state in switch.states
        if state.label in selected_states
    }
    component_model_cache = {}

    def component_model(component):
        key = component.source_path.resolve()
        if key not in component_model_cache:
            component_model_cache[key] = load_component_model(component)
        return component_model_cache[key]

    def candidate_network(candidate):
        if measured_search:
            branches = [
                (item.kind, item.value, item.name) for item in candidate.branch_components
            ]
            inputs = [
                (connection, item.kind, item.value, item.name)
                for connection, item in candidate.input_components
            ]
        else:
            branches = [
                (item.kind, item.value, f"ideal {item.kind}")
                for item in candidate.branch_reactances
            ]
            inputs = [
                (item.connection, item.kind, item.value, f"ideal {item.kind}")
                for item in candidate.input_reactances
            ]
        return branches, inputs

    best_score = max(candidate.score_db for candidate in selected_candidates)
    minimum_input_count = min(
        len(candidate.input_components if measured_search else candidate.input_reactances)
        for candidate in selected_candidates
    )
    alternative_payload = []
    for solution_index, candidate in enumerate(selected_candidates):
        branches, inputs = candidate_network(candidate)
        if solution_index == 0:
            recommendation_role = "best_performance"
        elif len(inputs) == minimum_input_count:
            recommendation_role = "simplest_bom"
        else:
            recommendation_role = "performance_complexity_compromise"
        alternative_payload.append({
            "solution_index": solution_index,
            "input_component_count": len(inputs),
            "score_db": candidate.score_db,
            "score_delta_from_best_db": candidate.score_db - best_score,
            "recommendation_role": recommendation_role,
            "state_by_configuration": candidate.state_by_configuration,
            "branch_network": [
                {"branch": index + 1, "kind": kind, "value_si": value, "part_number": part}
                for index, (kind, value, part) in enumerate(branches)
            ],
            "input_network": [
                {"connection": connection, "kind": kind, "value_si": value, "part_number": part}
                for connection, kind, value, part in inputs
            ],
        })

    def build_result(candidate, solution_index: int) -> TuningResult:
        active_efficiencies, active_accepted = [], []
        active_switch_loss, active_matching_loss = [], []
        active_return_losses, active_balance = [], []
        chart_frequencies, chart_return_losses, chart_efficiencies = [], [], []
        configuration_payload = []
        if measured_search:
            branch_models = [component_model(item) for item in candidate.branch_components]
            input_models = [
                InputModelPlacement(connection, component_model(item))
                for connection, item in candidate.input_components
            ]
        for configuration, source in zip(configurations, frequency_configurations):
            state_label = candidate.state_by_configuration[configuration.name]
            if measured_search:
                power = evaluate_loaded_switch_physical_power(
                    loaded_by_state[state_label], branch_models, input_elements=input_models
                )
            else:
                power = evaluate_loaded_switch_power(
                    loaded_by_state[state_label], candidate.branch_reactances,
                    input_reactances=candidate.input_reactances,
                )
            return_loss = -20.0 * np.log10(np.maximum(np.abs(power.input_gamma), 1e-15))
            mask = np.zeros(len(frequencies), dtype=bool)
            for band in configuration.bands_by_port[0]:
                mask |= band.mask(frequencies)
            active_efficiencies.extend(power.dut_absorbed_power[mask].tolist())
            active_accepted.extend(power.input_accepted_power[mask].tolist())
            active_switch_loss.extend(power.switch_loss[mask].tolist())
            active_matching_loss.extend(power.matching_network_loss[mask].tolist())
            active_return_losses.extend(return_loss[mask].tolist())
            active_balance.extend(power.power_balance_error[mask].tolist())
            chart_frequencies.extend(frequencies[mask].tolist())
            chart_return_losses.extend(return_loss[mask].tolist())
            chart_efficiencies.extend(power.dut_absorbed_power[mask].tolist())
            score_item = next(
                item for item in candidate.metrics["configurations"]
                if item["name"] == configuration.name
            )
            configuration_payload.append({
                "name": configuration.name,
                "state": state_label,
                "score_db": float(score_item["score_db"]),
                "bands_mhz": source["bands_mhz"],
            })

        branches, inputs = candidate_network(candidate)
        components = [{
            "position": -1, "part": switch.name, "part_number": switch.name,
            "type": "multi_throw_switch", "comp_type": "multi_throw_switch",
            "connection_type": "common_port_to_dut", "port": 0,
            "value": " / ".join(candidate.state_by_configuration.values()),
            "states": candidate.state_by_configuration,
        }]
        for index, (kind, value_si, part_number) in enumerate(branches):
            scale, unit = (1e9, "nH") if kind == "L" else (1e12, "pF")
            components.append({
                "position": index, "branch": index + 1,
                "part": part_number, "part_number": part_number,
                "type": "inductor" if kind == "L" else "capacitor",
                "comp_type": "inductor" if kind == "L" else "capacitor",
                "connection_type": "series_to_switch_throw", "port": 0,
                "nominal_value": value_si * scale, "nominal_unit": unit,
                "value": f"{value_si * scale:.4g}{unit}",
            })
        for index, (connection, kind, value_si, part_number) in enumerate(inputs, start=len(branches)):
            scale, unit = (1e9, "nH") if kind == "L" else (1e12, "pF")
            components.append({
                "position": index, "part": part_number, "part_number": part_number,
                "type": "inductor" if kind == "L" else "capacitor",
                "comp_type": "inductor" if kind == "L" else "capacitor",
                "connection_type": connection, "port": 0,
                "nominal_value": value_si * scale, "nominal_unit": unit,
                "value": f"{value_si * scale:.4g}{unit}",
            })
        efficiencies = np.asarray(active_efficiencies)
        accepted = np.asarray(active_accepted)
        returns = np.asarray(active_return_losses)
        component_loss = float(np.mean(
            np.asarray(active_switch_loss) + np.asarray(active_matching_loss)
        ))
        maximum_balance = float(np.max(np.abs(active_balance)))
        per_port = PerPortTuningMetrics(
            port_index=0,
            s11_magnitude=float(10.0 ** (-float(np.min(returns)) / 20.0)),
            s11_db=float(np.min(returns)),
            accepted_efficiency=float(np.mean(accepted)),
            radiated_efficiency=float(np.mean(efficiencies)),
            component_loss=component_loss,
            total_efficiency=float(np.mean(efficiencies)),
            components=components,
            band_freqs_hz=chart_frequencies,
            band_s11_db=chart_return_losses,
            band_total_eff=chart_efficiencies,
        )
        diagnostics = {
            "mode": "switch_mdif_measured_synthesis" if measured_search else "switch_mdif_auto_synthesis",
            "candidate_label": f"{len(inputs)} input component{'s' if len(inputs) != 1 else ''}",
            "input_component_count": len(inputs),
            "evaluations": search.evaluations,
            "physical_evaluations": measured_search.physical_evaluations if measured_search else 0,
            "component_models_loaded": measured_search.loaded_component_models if measured_search else 0,
            "switch_state_precomputations": search.state_precomputations,
            "active_frequency_points": len(frequencies),
            "maximum_power_balance_error": maximum_balance,
            "maximum_switch_model_gain": candidate.metrics["maximum_switch_model_gain"],
            "branch_network": alternative_payload[solution_index]["branch_network"],
            "input_network": alternative_payload[solution_index]["input_network"],
            "average_matching_network_loss": float(np.mean(active_matching_loss)),
            "max_input_components": max_input_components,
            "complexity_alternatives": alternative_payload,
            "calibration_reference": {
                "artifact": "artifacts/benchmarks/optenni-switch-tuning-baseline.json",
                "status": "reference_only_not_request_specific",
                "case": "Optenni 10.6 impedance tuning using a switch",
                "tutorial_document_sha256": "9ae7c208a4eba0ac8b1e5c43d2ed158c8f6f6e3a8d0de6d476b90d75c8bee52c",
                "verified_pages": [12, 13, 16],
                "reference_design_roles": [
                    "best_performance",
                    "simplified_near_best",
                    "reduced_bom_near_equivalent",
                ],
            },
        }
        result = TuningResult(
            port_indices=[0], mode="switch", objective=objective,
            per_port={0: per_port}, system_score=float(candidate.score_db),
            avg_total_efficiency=float(np.mean(efficiencies)),
            min_total_efficiency=float(np.min(efficiencies)),
            total_component_count=len(components), total_component_loss=component_loss,
            efficiency_basis="rfmatch_core_switch_mdif_wave_power",
            tunable_states=candidate.state_by_configuration,
            frequency_configurations=configuration_payload,
            num_solutions_found=len(selected_candidates), solution_index=solution_index,
            search_diagnostics=diagnostics,
            maximum_power_balance_error=maximum_balance,
        )
        if measured_search:
            tolerance_components = [
                {
                    "position": f"branch_{index + 1}",
                    "part_number": item.name,
                    "tolerance_pct": 100.0 * item.tolerance,
                    "source": "component_metadata",
                }
                for index, item in enumerate(candidate.branch_components)
            ] + [
                {
                    "position": f"input_{index + 1}",
                    "part_number": item.name,
                    "tolerance_pct": 100.0 * item.tolerance,
                    "source": "component_metadata",
                }
                for index, (_, item) in enumerate(candidate.input_components)
            ]
            result.yield_context = {
                "problem": problem,
                "loaded_states": loaded_by_state,
                "branch_models": tuple(branch_models),
                "input_elements": tuple(input_models),
                "state_by_configuration": dict(candidate.state_by_configuration),
                "component_tolerances": tolerance_components,
            }
        result.system_power_balance = {
            "system_efficiency": result.avg_total_efficiency,
            "accepted_efficiency": float(np.mean(accepted)),
            "component_loss": component_loss,
            "maximum_balance_error": maximum_balance,
            "basis": result.efficiency_basis,
            "note": "Full switch-port wave reconstruction separates DUT absorption and component loss.",
        }
        return result

    results = {
        index: build_result(candidate, index)
        for index, candidate in enumerate(selected_candidates)
    }
    elapsed = time.perf_counter() - started
    for result in results.values():
        result.total_time_s = elapsed
    return results


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
