"""Independent RF matching optimization kernel."""

__version__ = "0.3.0"

from .models import Band, Branch, Candidate, CircuitTopology, Component, Element, IsolationTarget, LumpedLossModel, LumpedModel, Objective, Problem, S2PModel
from .components import ComponentSpec, LazyComponentSpec, MeasuredComponentSpec, component_model_key, component_sha256, load_coilcraft_0402cs_catalog, load_coilcraft_0402hp_catalog, load_component_model, load_murata_gjm15_catalog, load_murata_gqm18_catalog
from .optimizer import MatchingOptimizer, OptimizationCancelled, OptimizationResult, SearchConfig
from .evaluator import evaluate_isolation_targets
from .physical import CircuitEvaluation, PhysicalSweep, evaluate_circuit, evaluate_physical_problem
from .physical_optimizer import (
    PORT_TOPOLOGY_PATTERNS,
    MeasuredCandidate,
    MeasuredComponentOptimizer,
    MeasuredOptimizationResult,
    MeasuredPlacement,
    ModelPlacement,
    MeasuredSearchConfig,
    build_circuit_topology,
    build_model_circuit_topology,
    evaluate_measured_candidate,
)
from .multi_scenario import (
    MultiScenarioMatchingOptimizer,
    MultiScenarioProblem,
    ScenarioProblem,
    SharedMeasuredComponentOptimizer,
    SharedMeasuredOptimizationResult,
    evaluate_measured_multi_scenario,
    evaluate_physical_multi_scenario,
    evaluate_multi_scenario,
)
from .optenni_opr import parse_optenni_opr
from .optenni_native import OptenniNativeComparison, OptenniNetworkReplay, OptenniPlotExport, compare_optenni_native_plot, load_optenni_plot_export, replay_one_port_dut_through_network
from .tolerance import ToleranceModel, ToleranceResult, YieldCriteria, monte_carlo_yield, rank_measured_candidates_by_yield, tolerance_summary
from .tolerance_golden import OptenniToleranceExport, load_optenni_tolerance_export, summarize_optenni_tolerance
from .state_tolerance import monte_carlo_switch_yield, monte_carlo_tunable_yield
from .touchstone import Touchstone, load_s2p_model, load_touchstone, parse_touchstone_text
from .network import cascade_s2p, chain_to_s2p, deembed_s2p, flip_s2p_ports, renormalize_s_parameters, s2p_to_chain
from .transmission_line import TransmissionLineModel, TransmissionLineStubModel
from .line_optimizer import LINE_TOPOLOGIES, LineSearchConfig, LineSynthesisCandidate, LineSynthesisResult, TransmissionLineOptimizer
from .microstrip import PCBSubstrate, MicrostripProperties, MicrostripDesignRules, MicrostripVariation, MicrostripLineModel, MicrostripStubModel, microstrip_quasi_static, microstrip_properties, solve_microstrip_width
from .mdif import MDIFModel, MDIFState, load_mdif
from .switch import InputModelPlacement, InputReactance, LoadedSwitchState, SeriesReactance, SwitchPowerSweep, evaluate_loaded_switch_physical_power, evaluate_loaded_switch_power, evaluate_loaded_switch_state, evaluate_switched_matching, preload_switch_state, reduce_switch_with_series_branches
from .switch_optimizer import MeasuredSwitchCandidate, MeasuredSwitchOptimizationResult, SwitchCandidate, SwitchMeasuredComponentOptimizer, SwitchOptimizationResult, SwitchSearchConfig, SwitchTunableOptimizer, SwitchTunableProblem, standard_switch_input_topologies
from .tunable import FrequencyConfiguration, TunableCandidate, TunableProblem, TunableSearchResult, evaluate_tunable_physical, load_measured_placements, rank_tunable_fixed_networks
from .tunable_optimizer import TunableMeasuredCandidate, TunableMeasuredComponentOptimizer, TunableMeasuredOptimizationResult
from .search_validation import ExhaustiveMeasuredResult, SearchRecallReport, exhaustive_measured_joint_search, exhaustive_measured_search, measured_candidate_signature, measured_search_recall, measured_topology_signature

__all__ = [
    "Band", "Branch", "Candidate", "CircuitTopology", "Component", "Element", "IsolationTarget", "LumpedLossModel", "LumpedModel", "Objective", "Problem", "S2PModel",
    "ComponentSpec", "LazyComponentSpec", "MeasuredComponentSpec", "component_model_key", "component_sha256", "load_coilcraft_0402cs_catalog", "load_coilcraft_0402hp_catalog", "load_component_model", "load_murata_gjm15_catalog", "load_murata_gqm18_catalog",
    "MatchingOptimizer", "OptimizationResult", "SearchConfig",
    "evaluate_isolation_targets",
    "PORT_TOPOLOGY_PATTERNS", "MeasuredCandidate", "MeasuredComponentOptimizer", "MeasuredOptimizationResult", "MeasuredPlacement", "ModelPlacement", "MeasuredSearchConfig",
    "build_circuit_topology", "build_model_circuit_topology", "evaluate_measured_candidate",
    "ScenarioProblem", "MultiScenarioProblem", "MultiScenarioMatchingOptimizer",
    "SharedMeasuredComponentOptimizer", "SharedMeasuredOptimizationResult",
    "evaluate_multi_scenario", "evaluate_measured_multi_scenario", "evaluate_physical_multi_scenario",
    "parse_optenni_opr",
    "OptenniPlotExport", "OptenniNetworkReplay", "OptenniNativeComparison", "load_optenni_plot_export", "replay_one_port_dut_through_network", "compare_optenni_native_plot",
    "CircuitEvaluation", "PhysicalSweep", "evaluate_circuit", "evaluate_physical_problem", "ToleranceModel", "ToleranceResult", "YieldCriteria", "monte_carlo_yield", "rank_measured_candidates_by_yield", "tolerance_summary",
    "OptenniToleranceExport", "load_optenni_tolerance_export", "summarize_optenni_tolerance",
    "monte_carlo_switch_yield", "monte_carlo_tunable_yield",
    "Touchstone", "load_s2p_model", "load_touchstone", "parse_touchstone_text",
    "cascade_s2p", "chain_to_s2p", "deembed_s2p", "flip_s2p_ports", "renormalize_s_parameters", "s2p_to_chain",
    "TransmissionLineModel", "TransmissionLineStubModel",
    "LINE_TOPOLOGIES", "LineSearchConfig", "LineSynthesisCandidate", "LineSynthesisResult", "TransmissionLineOptimizer",
    "PCBSubstrate", "MicrostripProperties", "MicrostripDesignRules", "MicrostripVariation", "MicrostripLineModel", "MicrostripStubModel", "microstrip_quasi_static", "microstrip_properties", "solve_microstrip_width",
    "MDIFModel", "MDIFState", "load_mdif",
    "InputModelPlacement", "InputReactance", "LoadedSwitchState", "SeriesReactance", "SwitchPowerSweep", "evaluate_loaded_switch_physical_power", "evaluate_loaded_switch_power", "evaluate_loaded_switch_state", "evaluate_switched_matching", "preload_switch_state", "reduce_switch_with_series_branches",
    "SwitchCandidate", "SwitchOptimizationResult", "SwitchSearchConfig", "SwitchTunableOptimizer", "SwitchTunableProblem",
    "MeasuredSwitchCandidate", "MeasuredSwitchOptimizationResult", "SwitchMeasuredComponentOptimizer",
    "standard_switch_input_topologies",
    "FrequencyConfiguration", "TunableCandidate", "TunableProblem", "TunableSearchResult", "evaluate_tunable_physical", "load_measured_placements", "rank_tunable_fixed_networks",
    "TunableMeasuredCandidate", "TunableMeasuredComponentOptimizer", "TunableMeasuredOptimizationResult",
    "OptimizationCancelled",
    "ExhaustiveMeasuredResult", "SearchRecallReport", "exhaustive_measured_joint_search", "exhaustive_measured_search",
    "measured_candidate_signature", "measured_search_recall", "measured_topology_signature",
]
