"""Hierarchical topology synthesis using measured S2P component models."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Callable, Literal, Sequence

import numpy as np

from .components import (
    MeasuredComponentSpec,
    component_model_key,
    load_component_model,
)
from .evaluator import evaluate, score_sweep
from .models import Branch, Candidate, CircuitTopology, Element, LumpedModel, Objective, Problem, S2PModel
from .transmission_line import TransmissionLineModel, TransmissionLineStubModel
from .microstrip import MicrostripLineModel, MicrostripStubModel
from .optimizer import MatchingOptimizer, OptimizationCancelled, SearchConfig
from .physical import evaluate_physical_problem


# The eight two-component topology codes observed in Optenni Lab 4.3, every
# zero/one-component option, and the practical 3/4-element Pi, T, and ladder
# families exposed by the product. P=parallel(shunt), S=series.
PORT_TOPOLOGY_PATTERNS: tuple[tuple[tuple[str, str], ...], ...] = (
    (),
    (("series", "L"),),
    (("shunt", "L"),),
    (("series", "C"),),
    (("shunt", "C"),),
    (("shunt", "L"), ("series", "L")),       # PLSL
    (("series", "L"), ("shunt", "L")),       # SLPL
    (("shunt", "L"), ("series", "C")),       # PLSC
    (("series", "L"), ("shunt", "C")),       # SLPC
    (("shunt", "C"), ("shunt", "L")),        # PCPL
    (("series", "C"), ("shunt", "L")),       # SCPL
    (("series", "C"), ("series", "L")),      # SCSL
    (("shunt", "C"), ("series", "L")),       # PCSL
    (("shunt", "C"), ("series", "L"), ("shunt", "C")),   # PCSLPC
    (("shunt", "L"), ("series", "C"), ("shunt", "L")),   # PLSCPL
    (("series", "L"), ("shunt", "C"), ("series", "L")),  # SLPCSL
    (("series", "C"), ("shunt", "L"), ("series", "C")),  # SCPLSC
    (
        ("series", "L"), ("shunt", "C"),
        ("series", "L"), ("shunt", "C"),
    ),  # SLPCSLPC
    (
        ("series", "C"), ("shunt", "L"),
        ("series", "C"), ("shunt", "L"),
    ),  # SCPLSCPL
)


@dataclass(frozen=True)
class MeasuredPlacement:
    connection: Literal["series", "shunt"]
    port: int
    component: MeasuredComponentSpec


@dataclass(frozen=True)
class ModelPlacement:
    """A physical placement whose already-loaded model can come from any adapter."""

    connection: Literal["series", "shunt"]
    port: int
    model: LumpedModel | S2PModel | TransmissionLineModel | TransmissionLineStubModel | MicrostripLineModel | MicrostripStubModel


@dataclass
class MeasuredCandidate:
    placements: tuple[MeasuredPlacement, ...]
    score_db: float
    metrics: dict = field(default_factory=dict)

    @property
    def topology_code(self) -> str:
        if not self.placements:
            return "0"
        return "".join(
            ("S" if item.connection == "series" else "P") + item.component.kind
            for item in self.placements
        )


@dataclass(frozen=True)
class MeasuredSearchConfig:
    ideal_restarts: int = 4
    ideal_iterations: int = 12
    ideal_keep: int = 32
    nearest_parts: int = 2
    per_port_keep: int = 8
    result_keep: int = 50
    joint_refine_seeds: int = 2
    joint_refine_passes: int = 2
    joint_refine_neighbors: int | None = None
    joint_refine_variants_per_value: int = 1
    joint_refine_beam_width: int = 1
    joint_refine_port_blocks: bool = False
    joint_refine_port_block_max_components: int = 2
    joint_ideal_topologies_per_port: int = 0
    joint_ideal_combination_keep: int = 8
    joint_ideal_rank_combinations: bool = False
    joint_ideal_diverse_combinations: bool = False
    joint_ideal_refine_topology_neighbors: bool = False
    joint_ideal_growth_refine_keep: int = 0
    joint_ideal_growth_refine_restarts: int = 4
    joint_ideal_growth_refine_iterations: int = 16
    joint_ideal_growth_refine_nearest_parts: int = 2
    joint_ideal_restarts: int = 1
    joint_ideal_iterations: int = 6
    joint_ideal_keep: int = 2
    joint_ideal_nearest_parts: int = 2
    seed: int = 1
    max_components_per_port: int = 2
    max_components_by_port: dict[int, int] | None = None
    topology_beam_width: int = 8
    deep_nearest_parts: int = 1
    deep_discrete_topology_seeds: int = 8
    allowed_topology_codes: frozenset[str] | None = None
    allowed_topology_codes_by_port: dict[int, frozenset[str]] | None = None
    include_zero_component: bool = True
    available_component_kinds: frozenset[str] | None = None


@dataclass
class MeasuredOptimizationResult:
    candidates: list[MeasuredCandidate]
    per_port_candidates: dict[int, list[MeasuredCandidate]]
    ideal_evaluations: int
    physical_evaluations: int
    loaded_component_models: int
    stage_physical_evaluations: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    termination_reason: str | None = None

    @property
    def best(self) -> MeasuredCandidate:
        if not self.candidates:
            raise ValueError("measured optimization produced no candidates")
        return self.candidates[0]


def topology_code(placements: Sequence[MeasuredPlacement]) -> str:
    if not placements:
        return "0"
    return "".join(
        ("S" if item.connection == "series" else "P") + item.component.kind
        for item in placements
    )


def build_circuit_topology(
    port_count: int,
    placements: Sequence[MeasuredPlacement],
    model_cache: dict[object, S2PModel] | None = None,
) -> CircuitTopology:
    """Convert ordered per-port placements into a physical nodal topology."""
    model_cache = model_cache if model_cache is not None else {}
    loaded: list[ModelPlacement] = []
    for placement in placements:
        key = component_model_key(placement.component)
        model = model_cache.get(key)
        if model is None:
            model = load_component_model(placement.component)
            model_cache[key] = model
        loaded.append(ModelPlacement(placement.connection, placement.port, model))
    return build_model_circuit_topology(port_count, loaded)


def build_model_circuit_topology(
    port_count: int,
    placements: Sequence[ModelPlacement],
) -> CircuitTopology:
    """Build the same DUT-outward topology from in-memory lumped/S2P models."""
    external = tuple(f"p{port + 1}" for port in range(port_count))
    grouped = {port: [] for port in range(port_count)}
    for placement in placements:
        if placement.port not in grouped:
            raise ValueError(f"invalid placement port {placement.port}")
        grouped[placement.port].append(placement)

    dut_nodes: list[str] = []
    branches: list[Branch] = []
    branch_index = 0
    for port in range(port_count):
        # Candidate element order follows network.apply_elements: DUT outward.
        # Nodal construction walks from the external connector toward the DUT,
        # so the physical traversal must use the reverse order.
        port_placements = list(reversed(grouped[port]))
        series_total = sum(item.connection == "series" for item in port_placements)
        dut_node = external[port] if series_total == 0 else f"dut{port + 1}"
        dut_nodes.append(dut_node)
        current = external[port]
        series_seen = 0
        for placement in port_placements:
            branch_index += 1
            model = placement.model
            if isinstance(model, (TransmissionLineModel, MicrostripLineModel)) and placement.connection != "series":
                raise ValueError("a through transmission line must use a series placement")
            if isinstance(model, (TransmissionLineStubModel, MicrostripStubModel)) and placement.connection != "shunt":
                raise ValueError("a transmission-line stub must use a shunt placement")
            name = f"b{branch_index}_p{port + 1}_{model.name}"
            if placement.connection == "shunt":
                branches.append(Branch(name, current, None, model))
            else:
                series_seen += 1
                next_node = dut_node if series_seen == series_total else f"p{port + 1}_n{series_seen}"
                branches.append(Branch(name, current, next_node, model))
                current = next_node
        if current != dut_node:
            raise AssertionError("series chain does not reach its DUT node")
    return CircuitTopology(external, tuple(branches), tuple(dut_nodes))


def _as_ideal_candidate(placements: Sequence[MeasuredPlacement]) -> Candidate:
    return Candidate([
        Element(item.connection, item.component.kind, item.port, item.component.value, item.component.name)
        for item in placements
    ])


def evaluate_measured_candidate(
    problem: Problem,
    placements: Sequence[MeasuredPlacement],
    objective: Objective,
    model_cache: dict[object, S2PModel] | None = None,
) -> MeasuredCandidate:
    topology = build_circuit_topology(problem.s_parameters.shape[1], placements, model_cache)
    sweep = evaluate_physical_problem(problem, topology)
    scored = score_sweep(problem, _as_ideal_candidate(placements), objective, sweep.s_parameters, sweep.total_efficiency)
    metrics = dict(scored.metrics)
    metrics.update({
        "s_parameters": sweep.s_parameters,
        "component_loss": sweep.component_loss,
        "dut_absorbed_power": sweep.dut_absorbed_power,
        "power_balance_error": sweep.power_balance_error,
        "maximum_power_balance_error": float(np.max(np.abs(sweep.power_balance_error))),
    })
    return MeasuredCandidate(tuple(placements), scored.score_db, metrics)


def _nearest(
    specs: Sequence[MeasuredComponentSpec], value: float, count: int
) -> list[MeasuredComponentSpec]:
    if not specs:
        raise ValueError("measured component catalog is empty")
    # Initial continuous-to-discrete snapping must cover different nominal
    # values. Multiple S2P variants at one nominal value are explored later by
    # variant-aware physical refinement; allowing them to occupy every nearest
    # slot would collapse value-space coverage before joint search begins.
    representative_by_value: dict[float, MeasuredComponentSpec] = {}
    for spec in specs:
        previous = representative_by_value.get(spec.value)
        if previous is None or (spec.tolerance, spec.name) < (
            previous.tolerance, previous.name
        ):
            representative_by_value[spec.value] = spec
    return sorted(
        representative_by_value.values(),
        key=lambda item: (
            abs(np.log(item.value / value)), item.tolerance, item.name
        ),
    )[:count]


def _candidate_signature(candidate: MeasuredCandidate) -> tuple:
    return tuple((item.connection, item.port, item.component.name) for item in candidate.placements)


def _unique_sorted(candidates: Sequence[MeasuredCandidate]) -> list[MeasuredCandidate]:
    unique: dict[tuple, MeasuredCandidate] = {}
    for candidate in candidates:
        signature = _candidate_signature(candidate)
        if signature not in unique or candidate.score_db > unique[signature].score_db:
            unique[signature] = candidate
    return sorted(unique.values(), key=lambda item: (-item.score_db, _candidate_signature(item)))


def _select_diverse(
    candidates: Sequence[MeasuredCandidate],
    keep: int,
    score_first_fraction: float = 0.0,
) -> list[MeasuredCandidate]:
    ordered = _unique_sorted(candidates)
    selected: list[MeasuredCandidate] = []
    selected_signatures: set[tuple] = set()
    seen_topologies: set[str] = set()
    score_slots = min(keep, max(0, int(round(keep * score_first_fraction))))
    for candidate in ordered[:score_slots]:
        selected.append(candidate)
        selected_signatures.add(_candidate_signature(candidate))
        seen_topologies.add(topology_code(candidate.placements))
    for candidate in ordered:
        code = topology_code(candidate.placements)
        signature = _candidate_signature(candidate)
        if code not in seen_topologies and signature not in selected_signatures:
            selected.append(candidate)
            selected_signatures.add(_candidate_signature(candidate))
            seen_topologies.add(code)
            if len(selected) == keep:
                return selected
    for candidate in ordered:
        signature = _candidate_signature(candidate)
        if signature not in selected_signatures:
            selected.append(candidate)
            selected_signatures.add(signature)
            if len(selected) == keep:
                break
    return selected


class MeasuredComponentOptimizer:
    """Ideal seed search → measured per-port refinement → joint physical ranking."""

    def __init__(
        self,
        problem: Problem,
        inductors: Sequence[MeasuredComponentSpec],
        capacitors: Sequence[MeasuredComponentSpec],
        objective: Objective | None = None,
        config: MeasuredSearchConfig | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ):
        self.problem = problem
        self.library = {"L": tuple(inductors), "C": tuple(capacitors)}
        self.objective = objective or Objective()
        self.config = config or MeasuredSearchConfig()
        self.cancel_check = cancel_check
        self.progress_callback = progress_callback
        self.model_cache: dict[object, S2PModel] = {}
        self.evaluation_cache: dict[tuple, MeasuredCandidate] = {}
        # Runtime-only continuation state.  Once an ideal topology search has
        # completed, a later call can reuse its deterministic frontier and all
        # exact measured evaluations instead of paying for them again.
        self.ideal_candidates_by_port: dict[int, list[Candidate]] = {}
        self._problem_refs: dict[int, Problem] = {}
        self.subproblems_by_port: dict[int, Problem] = {}
        self.ideal_evaluations = 0
        self.physical_evaluations = 0
        self.truncated = False
        self.termination_reason: str | None = None

    def _check_cancel(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise OptimizationCancelled("measured component optimization cancelled")

    def _truncate(self, stage: str) -> None:
        self.truncated = True
        self.termination_reason = f"cancelled during {stage}"

    def _emit_progress(self, stage: str, current: int, total: int, **details) -> None:
        if self.progress_callback is not None:
            payload = {
                "stage": stage,
                "current": int(current),
                "total": int(total),
                "ideal_evaluations": self.ideal_evaluations,
                "physical_evaluations": self.physical_evaluations,
            }
            payload.update(details)
            labels = {
                "ideal_search": "Ideal topology search",
                "measured_expansion": "Measured S2P candidate expansion",
                "deep_discrete_expansion": "Deep discrete-part expansion",
                "per_port": "Per-port candidate search",
                "joint": "Full-matrix joint ranking",
                "joint_ideal_search": "Coupled ideal topology search",
                "refine": "Exact physical refinement",
            }
            port = payload.get("port")
            prefix = f"Port {int(port) + 1}: " if port is not None else ""
            payload.setdefault("message", prefix + labels.get(stage, stage.replace("_", " ").title()))
            self.progress_callback(payload)

    def _evaluate(self, problem: Problem, placements: Sequence[MeasuredPlacement], objective: Objective) -> MeasuredCandidate:
        self._check_cancel()
        key = (id(problem), tuple(
            (item.connection, item.port, item.component.name)
            for item in placements
        ))
        self._problem_refs[id(problem)] = problem
        cached = self.evaluation_cache.get(key)
        if cached is not None:
            return cached
        self.physical_evaluations += 1
        evaluated = evaluate_measured_candidate(
            problem, placements, objective, self.model_cache
        )
        self.evaluation_cache[key] = evaluated
        return evaluated

    def _allowed_topology_codes(self, port: int) -> frozenset[str] | None:
        """Return the effective whitelist for one port.

        A per-port entry takes precedence over the legacy global whitelist.
        Missing per-port entries intentionally fall back to the global value so
        existing single-port and programmatic integrations remain compatible.
        """
        if (
            self.config.allowed_topology_codes_by_port is not None
            and port in self.config.allowed_topology_codes_by_port
        ):
            return self.config.allowed_topology_codes_by_port[port]
        return self.config.allowed_topology_codes

    def _zero_component_allowed(self, port: int) -> bool:
        if not self.config.include_zero_component:
            return False
        allowed = self._allowed_topology_codes(port)
        return allowed is None or "0" in allowed

    def _per_port(self, port: int) -> list[MeasuredCandidate]:
        max_components = self.config.max_components_per_port
        if self.config.max_components_by_port is not None:
            max_components = self.config.max_components_by_port.get(
                port, max_components
            )
        if max_components < 0:
            raise ValueError("maximum components per port cannot be negative")
        if self.config.nearest_parts < 1 or self.config.deep_nearest_parts < 1:
            raise ValueError("nearest part counts must be positive")
        if self.config.deep_discrete_topology_seeds < 0:
            raise ValueError("deep_discrete_topology_seeds cannot be negative")
        allowed_topology_codes = self._allowed_topology_codes(port)
        if len(self.problem.bands_by_port) == 1 and port in self.problem.bands_by_port:
            # A one-port product search already is the full problem. Reusing the
            # object lets exact measured evaluations survive into joint ranking
            # instead of being repeated under a different cache identity.
            subproblem = self.problem
        else:
            subproblem = self.subproblems_by_port.get(port)
            if subproblem is None:
                subproblem = Problem(
                    self.problem.frequencies_hz,
                    self.problem.s_parameters,
                    {port: self.problem.bands_by_port[port]},
                    self.problem.z0,
                    self.problem.radiation_efficiency,
                    tuple(
                        target
                        for target in self.problem.isolation_targets
                        if port in (target.source_port, target.destination_port)
                    ),
                )
                self.subproblems_by_port[port] = subproblem
        cached_ideal = self.ideal_candidates_by_port.get(port)
        if cached_ideal is not None:
            ideal_candidates = list(cached_ideal)
            self._emit_progress("ideal_search", 1, 1, port=port, checkpoint_reused=True)
        else:
            self._emit_progress("ideal_search", 0, 1, port=port, checkpoint_reused=False)
            available_kinds = self.config.available_component_kinds
            if available_kinds is None:
                available_kinds = frozenset(
                    kind for kind, components in self.library.items() if components
                )
            ideal_patterns = [
                [(connection, kind, port) for connection, kind in pattern]
                for pattern in PORT_TOPOLOGY_PATTERNS
                if pattern and len(pattern) <= max_components
                and all(kind in available_kinds for _connection, kind in pattern)
                and (
                    allowed_topology_codes is None
                    or "".join(
                        ("S" if connection == "series" else "P") + kind
                        for connection, kind in pattern
                    ) in allowed_topology_codes
                )
            ]
            ideal_optimizer = None
            ideal_candidates = []
            if ideal_patterns:
                ideal_optimizer = MatchingOptimizer(
                    subproblem,
                    self.objective,
                    SearchConfig(
                        restarts=self.config.ideal_restarts,
                        iterations=self.config.ideal_iterations,
                        keep=self.config.ideal_keep,
                        seed=self.config.seed + port,
                    ),
                    cancel_check=self.cancel_check,
                )
                ideal_candidates = ideal_optimizer.optimize(ideal_patterns).candidates
            progressive_optimizer = None
            if max_components > max(len(pattern) for pattern in PORT_TOPOLOGY_PATTERNS):
                progressive_optimizer = MatchingOptimizer(
                    subproblem,
                    self.objective,
                    SearchConfig(
                        restarts=max(1, min(2, self.config.ideal_restarts)),
                        iterations=self.config.ideal_iterations,
                        keep=max(
                            self.config.ideal_keep,
                            self.config.topology_beam_width * max_components,
                        ),
                        seed=self.config.seed + port,
                    ),
                    cancel_check=self.cancel_check,
                )
                discovered = progressive_optimizer.discover_ladder_topologies(
                    port,
                    max_components,
                    topology_beam_width=self.config.topology_beam_width,
                ).candidates
                ideal_candidates.extend(
                    candidate for candidate in discovered
                    if len(candidate.elements) > 4
                    and all(item.kind in available_kinds for item in candidate.elements)
                    and (
                        allowed_topology_codes is None
                        or "".join(
                            ("S" if item.connection == "series" else "P") + item.kind
                            for item in candidate.elements
                        ) in allowed_topology_codes
                    )
                )
                ranked_ideal = sorted(
                    ideal_candidates, key=lambda candidate: candidate.score_db, reverse=True
                )
                best_by_depth = {}
                deepest_by_topology = {}
                for candidate in ranked_ideal:
                    best_by_depth.setdefault(len(candidate.elements), candidate)
                    if len(candidate.elements) == max_components:
                        topology = tuple(
                            (item.connection, item.kind) for item in candidate.elements
                        )
                        deepest_by_topology.setdefault(topology, candidate)
                mandatory = list(deepest_by_topology.values())
                mandatory_ids = {id(candidate) for candidate in mandatory}
                mandatory.extend(
                    candidate for candidate in best_by_depth.values()
                    if id(candidate) not in mandatory_ids
                )
                mandatory = mandatory[:self.config.ideal_keep]
                mandatory_ids = {id(candidate) for candidate in mandatory}
                ideal_candidates = mandatory + [
                    candidate for candidate in ranked_ideal if id(candidate) not in mandatory_ids
                ][:max(0, self.config.ideal_keep - len(mandatory))]
            zero_component_allowed = self._zero_component_allowed(port)
            if zero_component_allowed:
                ideal_candidates.append(evaluate(subproblem, Candidate([]), self.objective))
            self.ideal_evaluations += (
                ideal_optimizer.evaluations if ideal_optimizer is not None else 0
            ) + (progressive_optimizer.evaluations if progressive_optimizer is not None else 0) + int(
                zero_component_allowed
            )
            self.ideal_candidates_by_port[port] = list(ideal_candidates)
            self._emit_progress("ideal_search", 1, 1, port=port, checkpoint_reused=False)

        measured: list[MeasuredCandidate] = []
        deep_seeds: list[tuple[Candidate, MeasuredCandidate]] = []
        try:
            for seed_index, seed in enumerate(ideal_candidates):
                if not seed.elements:
                    measured.append(self._evaluate(subproblem, (), self.objective))
                    self._emit_progress(
                        "measured_expansion", seed_index + 1, len(ideal_candidates), port=port,
                    )
                    continue
                nearest_count = (
                    self.config.deep_nearest_parts
                    if len(seed.elements) > 4 else self.config.nearest_parts
                )
                choices = [
                    _nearest(self.library[element.kind], element.value, nearest_count)
                    for element in seed.elements
                ]
                seed_measured = []
                for selected in itertools.product(*choices):
                    placements = tuple(
                        MeasuredPlacement(element.connection, port, spec)
                        for element, spec in zip(seed.elements, selected)
                    )
                    evaluated = self._evaluate(subproblem, placements, self.objective)
                    measured.append(evaluated)
                    seed_measured.append(evaluated)
                if len(seed.elements) > 4 and seed_measured:
                    deep_seeds.append((seed, max(seed_measured, key=lambda item: item.score_db)))
                self._emit_progress(
                    "measured_expansion", seed_index + 1, len(ideal_candidates), port=port,
                )
            if self.config.deep_discrete_topology_seeds > 0:
                deep_seeds.sort(key=lambda item: item[1].score_db, reverse=True)
                selected_deep_seeds = deep_seeds[:self.config.deep_discrete_topology_seeds]
                for seed_index, (seed, _) in enumerate(selected_deep_seeds):
                    choices = [
                        _nearest(self.library[element.kind], element.value, self.config.nearest_parts)
                        for element in seed.elements
                    ]
                    for selected in itertools.product(*choices):
                        placements = tuple(
                            MeasuredPlacement(element.connection, port, spec)
                            for element, spec in zip(seed.elements, selected)
                        )
                        measured.append(self._evaluate(subproblem, placements, self.objective))
                    self._emit_progress(
                        "deep_discrete_expansion", seed_index + 1,
                        len(selected_deep_seeds), port=port,
                    )
        except OptimizationCancelled:
            self._truncate(f"per-port measured expansion for port {port}")
            if not measured:
                raise
        return _select_diverse(
            measured,
            self.config.per_port_keep,
            score_first_fraction=0.5 if max_components > 4 else 0.0,
        )

    def _coordinate_refine(self, seed: MeasuredCandidate) -> list[MeasuredCandidate]:
        """Refine real parts and retain every already-paid-for physical trial.

        Keeping only the final coordinate-descent winner wastes useful exact
        evaluations and severely reduces top-k part-number recall. The global
        result limit still bounds the externally returned candidate list.
        """
        explored = [seed]
        if self.config.joint_refine_variants_per_value < 1:
            raise ValueError("joint_refine_variants_per_value must be positive")
        if self.config.joint_refine_beam_width < 1:
            raise ValueError("joint_refine_beam_width must be positive")

        def neighbor_components(placement: MeasuredPlacement):
            components = self.library[placement.component.kind]
            value_count = self.config.joint_refine_neighbors
            if value_count is None:
                return components
            by_value: dict[float, list[MeasuredComponentSpec]] = {}
            for component in components:
                by_value.setdefault(component.value, []).append(component)
            selected_values = sorted(
                by_value,
                key=lambda value: abs(np.log(value / placement.component.value)),
            )[:value_count]
            return tuple(
                component
                for value in selected_values
                for component in sorted(
                    by_value[value], key=lambda item: (item.tolerance, item.name)
                )[: self.config.joint_refine_variants_per_value]
            )

        frontier = [seed]
        for _ in range(self.config.joint_refine_passes):
            previous_signatures = {
                _candidate_signature(candidate) for candidate in frontier
            }
            pass_candidates: list[MeasuredCandidate] = []
            for origin in frontier:
                # Evaluate complete neighborhoods around fixed origins. A beam
                # wider than one preserves temporarily worse port-block moves
                # that can become superior after another coupled port changes.
                groups: list[tuple[int, ...]] = []
                if self.config.joint_refine_port_blocks:
                    for port in dict.fromkeys(
                        item.port for item in origin.placements
                    ):
                        indices = tuple(
                            index for index, item in enumerate(origin.placements)
                            if item.port == port
                        )
                        if len(indices) <= self.config.joint_refine_port_block_max_components:
                            groups.append(indices)
                        else:
                            groups.extend((index,) for index in indices)
                else:
                    groups = [(index,) for index in range(len(origin.placements))]

                for indices in groups:
                    choices = [
                        neighbor_components(origin.placements[index])
                        for index in indices
                    ]
                    for selected in itertools.product(*choices):
                        if all(
                            component == origin.placements[index].component
                            for index, component in zip(indices, selected)
                        ):
                            continue
                        trial = list(origin.placements)
                        for index, component in zip(indices, selected):
                            placement = origin.placements[index]
                            trial[index] = MeasuredPlacement(
                                placement.connection, placement.port, component
                            )
                        pass_candidates.append(
                            self._evaluate(self.problem, tuple(trial), self.objective)
                        )
            explored.extend(pass_candidates)
            frontier = _unique_sorted([*frontier, *pass_candidates])[
                : self.config.joint_refine_beam_width
            ]
            if {
                _candidate_signature(candidate) for candidate in frontier
            } == previous_signatures:
                break
        return _unique_sorted(explored)

    def _joint_ideal_expansion(
        self, per_port: dict[int, list[MeasuredCandidate]]
    ) -> list[MeasuredCandidate]:
        """Create full-matrix continuous seeds before measured joint ranking.

        Independent port shortlists are efficient but can prune networks whose
        value only appears after the other coupled ports are matched.  This
        bounded bridge retains a few topology shapes per port, optimizes all
        their ideal values together on the complete DUT matrix, and snaps those
        coupled optima to measured parts.
        """
        topology_keep = self.config.joint_ideal_topologies_per_port
        if topology_keep <= 0 or not per_port:
            return []
        if self.config.joint_ideal_combination_keep < 1:
            raise ValueError("joint_ideal_combination_keep must be positive")
        if self.config.joint_ideal_keep < 1:
            raise ValueError("joint_ideal_keep must be positive")
        if self.config.joint_ideal_nearest_parts < 1:
            raise ValueError("joint_ideal_nearest_parts must be positive")

        topology_options = []
        for port in self.problem.bands_by_port:
            ideal_by_topology = {}
            for ideal in self.ideal_candidates_by_port.get(port, ()):
                if not ideal.elements:
                    continue
                code = "".join(
                    ("S" if item.connection == "series" else "P") + item.kind
                    for item in ideal.elements
                )
                previous = ideal_by_topology.get(code)
                if previous is None or ideal.score_db > previous.score_db:
                    ideal_by_topology[code] = ideal
            seen = set()
            options = []
            for candidate in per_port.get(port, ()):
                topology = tuple(
                    (item.connection, item.component.kind, port)
                    for item in candidate.placements
                )
                if not topology or topology in seen:
                    continue
                ideal = ideal_by_topology.get(topology_code(candidate.placements))
                if ideal is None:
                    continue
                seen.add(topology)
                options.append((topology, candidate, ideal))
                if len(options) >= topology_keep:
                    break
            if not options:
                return []
            topology_options.append(options)

        measured: list[MeasuredCandidate] = []
        raw_combinations = itertools.product(*topology_options)
        if self.config.joint_ideal_rank_combinations:
            ranked_combinations = []
            for combination in raw_combinations:
                ideal_elements = tuple(itertools.chain.from_iterable(
                    item[2].elements for item in combination
                ))
                screened = evaluate(
                    self.problem, Candidate(list(ideal_elements)), self.objective
                )
                self.ideal_evaluations += 1
                ranked_combinations.append((
                    screened.score_db,
                    tuple(item[0] for item in combination),
                ))
            ranked_combinations.sort(key=lambda item: item[0], reverse=True)
            if self.config.joint_ideal_diverse_combinations:
                keep = self.config.joint_ideal_combination_keep
                selected = ranked_combinations[:max(1, keep // 4)]
                selected_combinations = {item[1] for item in selected}
                anchor = ranked_combinations[0][1]
                one_port_neighbors = [
                    item for item in ranked_combinations
                    if item[1] not in selected_combinations
                    and sum(
                        left != right
                        for left, right in zip(item[1], anchor)
                    ) == 1
                ]
                for item in one_port_neighbors:
                    if len(selected) >= keep:
                        break
                    selected.append(item)
                    selected_combinations.add(item[1])
                covered = {
                    (port_index, topology)
                    for _score, combination in selected
                    for port_index, topology in enumerate(combination)
                }
                while len(selected) < keep:
                    remaining = [
                        item for item in ranked_combinations
                        if item[1] not in selected_combinations
                    ]
                    if not remaining:
                        break
                    best = max(
                        remaining,
                        key=lambda item: (
                            sum(
                                (port_index, topology) not in covered
                                for port_index, topology in enumerate(item[1])
                            ),
                            item[0],
                        ),
                    )
                    selected.append(best)
                    selected_combinations.add(best[1])
                    covered.update(
                        (port_index, topology)
                        for port_index, topology in enumerate(best[1])
                    )
                combinations = [item[1] for item in selected]
            else:
                combinations = [
                    combination for _score, combination in ranked_combinations[
                        : self.config.joint_ideal_combination_keep
                    ]
                ]
        else:
            combinations = [
                tuple(item[0] for item in combination)
                for combination in itertools.islice(
                    raw_combinations,
                    self.config.joint_ideal_combination_keep,
                )
            ]
        optimized_scores = []

        def expand_combination(
            combination, combination_index, total, *,
            restarts=None, iterations=None, nearest_parts=None,
        ):
            self._check_cancel()
            topology = tuple(itertools.chain.from_iterable(combination))
            optimizer = MatchingOptimizer(
                self.problem,
                self.objective,
                SearchConfig(
                    restarts=(
                        self.config.joint_ideal_restarts
                        if restarts is None else restarts
                    ),
                    iterations=(
                        self.config.joint_ideal_iterations
                        if iterations is None else iterations
                    ),
                    keep=max(self.config.joint_ideal_keep, 2),
                    seed=self.config.seed + 10_000 + combination_index,
                ),
                cancel_check=self.cancel_check,
            )
            continuous = optimizer.optimize([topology]).candidates[
                : self.config.joint_ideal_keep
            ]
            self.ideal_evaluations += optimizer.evaluations
            expanded = []
            for seed in continuous:
                choices = [
                    _nearest(
                        self.library[element.kind],
                        element.value,
                        (
                            self.config.joint_ideal_nearest_parts
                            if nearest_parts is None else nearest_parts
                        ),
                    )
                    for element in seed.elements
                ]
                for selected in itertools.product(*choices):
                    placements = tuple(
                        MeasuredPlacement(element.connection, element.port, spec)
                        for element, spec in zip(seed.elements, selected)
                    )
                    evaluated = self._evaluate(
                        self.problem, placements, self.objective
                    )
                    measured.append(evaluated)
                    expanded.append(evaluated)
            self._emit_progress(
                "joint_ideal_search", combination_index + 1, total
            )
            return max(
                (candidate.score_db for candidate in expanded),
                default=float("-inf"),
            )

        for combination_index, combination in enumerate(combinations):
            optimized_scores.append((
                expand_combination(
                    combination, combination_index, len(combinations)
                ),
                combination,
            ))

        if (
            self.config.joint_ideal_refine_topology_neighbors
            and optimized_scores
        ):
            anchor = max(optimized_scores, key=lambda item: item[0])[1]
            already = set(combinations)
            neighbors = []
            for port_index, options in enumerate(topology_options):
                for option in options:
                    trial = list(anchor)
                    trial[port_index] = option[0]
                    trial = tuple(trial)
                    if trial not in already:
                        neighbors.append(trial)
                        already.add(trial)
            neighbor_scores = []
            for neighbor_index, combination in enumerate(neighbors):
                score = expand_combination(
                    combination,
                    len(combinations) + neighbor_index,
                    len(combinations) + len(neighbors),
                )
                neighbor_scores.append((score, combination))
            if self.config.joint_ideal_growth_refine_keep > 0:
                coupled_anchor = max(
                    [*optimized_scores, *neighbor_scores],
                    key=lambda item: item[0],
                )[1]
                growth = [
                    item for item in neighbor_scores
                    if sum(
                        left != right
                        for left, right in zip(item[1], coupled_anchor)
                    ) == 1
                    and sum(len(topology) for topology in item[1])
                    > sum(len(topology) for topology in coupled_anchor)
                ]
                growth.sort(key=lambda item: item[0], reverse=True)
                growth = growth[:self.config.joint_ideal_growth_refine_keep]
                for growth_index, (_score, combination) in enumerate(growth):
                    expand_combination(
                        combination,
                        len(combinations) + len(neighbors) + growth_index,
                        len(combinations) + len(neighbors) + len(growth),
                        restarts=self.config.joint_ideal_growth_refine_restarts,
                        iterations=self.config.joint_ideal_growth_refine_iterations,
                        nearest_parts=(
                            self.config.joint_ideal_growth_refine_nearest_parts
                        ),
                    )
        return _unique_sorted(measured)

    def optimize(self) -> MeasuredOptimizationResult:
        # A checkpoint may call optimize again with a fresh cooperative
        # deadline. Truncation describes the current slice, not the lifetime
        # of the optimizer object.
        self.truncated = False
        self.termination_reason = None
        # Always establish one valid full-problem result before optional work so
        # a cooperative deadline can still return a usable partial answer.
        baseline_key = (id(self.problem), ())
        baseline = self.evaluation_cache.get(baseline_key)
        if baseline is None:
            # The no-network solve is the bounded grace result guaranteed even
            # when the cooperative deadline is already expired.
            self.physical_evaluations += 1
            baseline = evaluate_measured_candidate(
                self.problem, (), self.objective, self.model_cache
            )
            self.evaluation_cache[baseline_key] = baseline
        stage_physical_evaluations = {
            "baseline": 1 if self.physical_evaluations == 1 else 0,
        }
        stage_start = self.physical_evaluations
        per_port: dict[int, list[MeasuredCandidate]] = {}
        ordered_ports = list(self.problem.bands_by_port)
        for index, port in enumerate(ordered_ports):
            self._emit_progress("per_port", index, len(ordered_ports))
            try:
                per_port[port] = self._per_port(port)
            except OptimizationCancelled:
                self._truncate(f"per-port search for port {port}")
                break
        self._emit_progress("per_port", len(per_port), len(ordered_ports))
        stage_physical_evaluations["per_port"] = (
            self.physical_evaluations - stage_start
        )

        baseline_allowed = all(
            self._zero_component_allowed(port) for port in ordered_ports
        )
        joint: list[MeasuredCandidate] = [baseline] if baseline_allowed else []
        stage_start = self.physical_evaluations
        if len(per_port) == len(ordered_ports) and not self.truncated:
            try:
                joint.extend(self._joint_ideal_expansion(per_port))
            except OptimizationCancelled:
                self._truncate("coupled ideal topology search")
        stage_physical_evaluations["joint_ideal"] = (
            self.physical_evaluations - stage_start
        )
        stage_start = self.physical_evaluations
        if self.truncated and len(ordered_ports) == 1 and ordered_ports[0] in per_port:
            # _per_port reused the exact full Problem in the single-port case,
            # so every completed candidate is already final-quality work.
            joint.extend(per_port[ordered_ports[0]])
        elif len(per_port) < len(ordered_ports):
            # One grace evaluation combines the best completed port seeds. It
            # bypasses the expired check deliberately and is bounded to one
            # exact solve, preserving useful work without restarting search.
            placements = tuple(itertools.chain.from_iterable(
                per_port[port][0].placements for port in ordered_ports if port in per_port
            ))
            if placements:
                self.physical_evaluations += 1
                grace = evaluate_measured_candidate(
                    self.problem, placements, self.objective, self.model_cache
                )
                self.evaluation_cache[(id(self.problem), tuple(
                    (item.connection, item.port, item.component.name)
                    for item in placements
                ))] = grace
                joint.append(grace)
        else:
            combinations = itertools.product(*(per_port[port] for port in ordered_ports))
            total_combinations = int(np.prod([
                len(per_port[port]) for port in ordered_ports
            ]))
            for index, combination in enumerate(combinations):
                try:
                    placements = tuple(itertools.chain.from_iterable(
                        candidate.placements for candidate in combination
                    ))
                    joint.append(self._evaluate(
                        self.problem, placements, self.objective
                    ))
                except OptimizationCancelled:
                    self._truncate("joint ranking")
                    break
                if index % 25 == 0:
                    self._emit_progress("joint", index + 1, total_combinations)
        stage_physical_evaluations["joint_ranking"] = (
            self.physical_evaluations - stage_start
        )
        initial = _unique_sorted(joint)
        refined = []
        stage_start = self.physical_evaluations
        if not self.truncated:
            for index, candidate in enumerate(
                initial[: self.config.joint_refine_seeds]
            ):
                try:
                    refined.extend(self._coordinate_refine(candidate))
                except OptimizationCancelled:
                    self._truncate("joint refinement")
                    break
                self._emit_progress(
                    "refine", index + 1,
                    min(len(initial), self.config.joint_refine_seeds),
                )
        stage_physical_evaluations["joint_refinement"] = (
            self.physical_evaluations - stage_start
        )
        candidates = _unique_sorted([*initial, *refined])[: self.config.result_keep]
        return MeasuredOptimizationResult(
            candidates,
            per_port,
            self.ideal_evaluations,
            self.physical_evaluations,
            len(self.model_cache),
            stage_physical_evaluations,
            self.truncated,
            self.termination_reason,
        )
