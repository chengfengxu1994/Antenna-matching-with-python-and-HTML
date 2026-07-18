"""Joint topology, value, and state optimization for multi-throw switches."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Callable, Mapping, Sequence

import numpy as np

from .components import ComponentSpec, component_model_key, component_sha256, load_component_model
from .mdif import MDIFModel
from .models import Objective
from .switch import InputModelPlacement, InputReactance, LoadedSwitchState, SeriesReactance, evaluate_loaded_switch_physical_power, evaluate_loaded_switch_power, preload_switch_state
from .tunable import FrequencyConfiguration
from .tunable_optimizer import OptimizationCancelled


@dataclass(frozen=True)
class SwitchTunableProblem:
    frequencies_hz: np.ndarray
    dut_s11: np.ndarray
    configurations: tuple[FrequencyConfiguration, ...]
    z0: float = 50.0
    configuration_average_weight: float = 0.5
    state_options_by_configuration: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        dut = np.asarray(self.dut_s11, dtype=complex)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "dut_s11", dut)
        if frequencies.ndim != 1 or dut.shape != frequencies.shape or len(frequencies) == 0:
            raise ValueError("frequencies_hz and dut_s11 must be non-empty equal-length vectors")
        if np.any(np.diff(frequencies) <= 0) or np.any(frequencies <= 0):
            raise ValueError("frequencies must be positive and strictly increasing")
        if not self.configurations:
            raise ValueError("at least one frequency configuration is required")
        if not 0.0 <= self.configuration_average_weight <= 1.0:
            raise ValueError("configuration_average_weight must be between 0 and 1")
        for configuration in self.configurations:
            if set(configuration.bands_by_port) != {0}:
                raise ValueError("switch tuner currently supports the one-port DUT configuration")
            for band in configuration.bands_by_port[0]:
                if not np.any(band.mask(frequencies)):
                    raise ValueError(f"configuration band {band} has no frequency samples")
        names = {configuration.name for configuration in self.configurations}
        unknown = set(self.state_options_by_configuration) - names
        if unknown:
            raise ValueError(f"state options name unknown configurations: {sorted(unknown)}")
        if any(not tuple(options) for options in self.state_options_by_configuration.values()):
            raise ValueError("configured state option lists must not be empty")


@dataclass(frozen=True)
class SwitchSearchConfig:
    restarts: int = 4
    iterations: int = 16
    initial_log_step: float = 1.0
    keep: int = 20
    seed: int = 1


@dataclass
class SwitchCandidate:
    branch_reactances: tuple[SeriesReactance, ...]
    input_reactances: tuple[InputReactance, ...]
    state_by_configuration: dict[str, str]
    score_db: float
    metrics: dict = field(default_factory=dict)


@dataclass
class SwitchOptimizationResult:
    candidates: list[SwitchCandidate]
    evaluations: int
    state_precomputations: int

    @property
    def best(self) -> SwitchCandidate:
        if not self.candidates:
            raise ValueError("switch optimization produced no candidates")
        return self.candidates[0]


@dataclass
class MeasuredSwitchCandidate:
    branch_components: tuple[ComponentSpec, ...]
    input_components: tuple[tuple[str, ComponentSpec], ...]
    state_by_configuration: dict[str, str]
    score_db: float
    metrics: dict = field(default_factory=dict)


@dataclass
class MeasuredSwitchOptimizationResult:
    candidates: list[MeasuredSwitchCandidate]
    physical_evaluations: int
    loaded_component_models: int

    @property
    def best(self) -> MeasuredSwitchCandidate:
        if not self.candidates:
            raise ValueError("measured switch optimization produced no candidates")
        return self.candidates[0]


def _blend_min_average(values: np.ndarray, average_weight: float) -> float:
    weight = float(np.clip(average_weight, 0.0, 1.0))
    return (1.0 - weight) * float(np.min(values)) + weight * float(np.mean(values))


def standard_switch_input_topologies(max_elements: int = 2) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Ordered 0–2 element series/shunt L/C synthesis-block topologies."""
    if not 0 <= max_elements <= 2:
        raise ValueError("switch input synthesis currently supports 0–2 elements")
    element_types = tuple(itertools.product(("series", "shunt"), ("L", "C")))
    topologies: list[tuple[tuple[str, str], ...]] = [()]
    if max_elements >= 1:
        topologies.extend((item,) for item in element_types)
    if max_elements >= 2:
        topologies.extend(tuple(items) for items in itertools.product(element_types, repeat=2))
    return tuple(topologies)


class SwitchTunableOptimizer:
    """Deterministic multi-start optimizer with state selection inside scoring."""

    def __init__(
        self,
        problem: SwitchTunableProblem,
        switch: MDIFModel,
        objective: Objective | None = None,
        config: SwitchSearchConfig | None = None,
        *,
        common_port: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.problem = problem
        self.switch = switch
        self.objective = objective or Objective()
        self.config = config or SwitchSearchConfig()
        if not switch.states:
            raise ValueError("switch MDIF has no states")
        port_counts = {state.n_ports for state in switch.states}
        if len(port_counts) != 1:
            raise ValueError("all switch states must have the same port count")
        metadata_port = int(switch.metadata.get("commonPort", "1")) - 1
        self.common_port = metadata_port if common_port is None else common_port
        if not 0 <= self.common_port < next(iter(port_counts)):
            raise ValueError("common_port is outside the switch model")
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self._emit_progress("switch_state_precompute", 0, len(switch.states), "Preparing switch states")
        loaded_states = []
        for index, state in enumerate(switch.states):
            self._check_cancel()
            loaded_states.append(preload_switch_state(
                problem.frequencies_hz,
                problem.dut_s11,
                state,
                common_port=self.common_port,
                z0=problem.z0,
            ))
            self._emit_progress("switch_state_precompute", index + 1, len(switch.states), state.label)
        self.loaded_states = tuple(loaded_states)
        available_states = {state.label for state in self.loaded_states}
        requested_states = {
            label
            for options in problem.state_options_by_configuration.values()
            for label in options
        }
        missing_states = requested_states - available_states
        if missing_states:
            raise ValueError(f"configured switch states are unavailable: {sorted(missing_states)}")
        self.evaluations = 0

    def _check_cancel(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise OptimizationCancelled("switch optimization cancelled")

    def _emit_progress(self, stage: str, current: int, total: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback({
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
                "evaluations": getattr(self, "evaluations", 0),
            })

    @staticmethod
    def _bounds(kind: str) -> tuple[float, float]:
        return (1e-10, 1e-7) if kind == "L" else (1e-13, 1e-10)

    def _score_trace(
        self,
        gamma: np.ndarray,
        configuration: FrequencyConfiguration,
        total_efficiency: np.ndarray,
    ) -> tuple[float, list[dict]]:
        efficiency_db = 10.0 * np.log10(np.maximum(total_efficiency, 1e-15))
        return_loss_db = -20.0 * np.log10(np.maximum(np.abs(gamma), 1e-15))
        band_scores = []
        metrics = []
        for band in configuration.bands_by_port[0]:
            mask = band.mask(self.problem.frequencies_hz)
            values = efficiency_db[mask]
            score = _blend_min_average(values, self.objective.within_band_average_weight)
            cost = (score - band.target_db) * band.weight
            if self.objective.impedance_target_db is not None and self.objective.impedance_weight > 0:
                margin = float(np.min(return_loss_db[mask])) + self.objective.impedance_target_db
                cost += self.objective.impedance_weight * min(0.0, margin)
            band_scores.append(cost)
            metrics.append({
                "start_hz": band.start_hz,
                "stop_hz": band.stop_hz,
                "minimum_efficiency_db": float(np.min(values)),
                "average_efficiency_db": float(np.mean(values)),
                "minimum_return_loss_db": float(np.min(return_loss_db[mask])),
            })
        return _blend_min_average(np.asarray(band_scores), self.objective.across_band_average_weight), metrics

    def evaluate(
        self,
        branch_reactances: Sequence[SeriesReactance],
        input_reactances: Sequence[InputReactance],
    ) -> SwitchCandidate:
        self._check_cancel()
        self.evaluations += 1
        state_sweeps = {
            loaded.label: evaluate_loaded_switch_power(
                loaded, branch_reactances, input_reactances=input_reactances
            )
            for loaded in self.loaded_states
        }
        assignments, final_score, metrics = self._select_state_sweeps(
            state_sweeps, len(branch_reactances) + len(input_reactances)
        )
        return SwitchCandidate(
            tuple(branch_reactances),
            tuple(input_reactances),
            assignments,
            final_score,
            metrics,
        )

    def _select_state_sweeps(self, state_sweeps: dict, complexity: int):
        selected = []
        assignments: dict[str, str] = {}
        for configuration in self.problem.configurations:
            options = []
            allowed = set(self.problem.state_options_by_configuration.get(configuration.name, state_sweeps))
            for label, sweep in state_sweeps.items():
                if label not in allowed:
                    continue
                score, bands = self._score_trace(
                    sweep.input_gamma, configuration, sweep.dut_absorbed_power
                )
                options.append((score, label, bands, sweep))
            score, label, bands, sweep = max(options, key=lambda item: (item[0], item[1]))
            assignments[configuration.name] = label
            selected.append({
                "name": configuration.name,
                "weight": configuration.weight,
                "state": label,
                "score_db": score,
                "bands": bands,
                "average_switch_loss": float(np.mean(sweep.switch_loss)),
                "average_matching_network_loss": float(np.mean(sweep.matching_network_loss)),
                "maximum_switch_model_gain": float(np.max(np.maximum(0.0, -sweep.switch_loss))),
                "maximum_power_balance_error": float(np.max(np.abs(sweep.power_balance_error))),
            })
        scores = np.asarray([item["score_db"] for item in selected])
        weights = np.asarray([item["weight"] for item in selected])
        if not np.any(weights > 0):
            weights = np.ones_like(weights)
        average = float(np.average(scores, weights=weights))
        worst = float(np.min(scores))
        blend = self.problem.configuration_average_weight
        final_score = (1.0 - blend) * worst + blend * average - self.objective.complexity_penalty_db * complexity
        return assignments, final_score, {
                "configurations": selected,
                "weighted_average_score_db": average,
                "worst_configuration_score_db": worst,
                "configuration_average_weight": blend,
                "maximum_power_balance_error": max(
                    item["maximum_power_balance_error"] for item in selected
                ),
                "maximum_switch_model_gain": max(
                    item["maximum_switch_model_gain"] for item in selected
                ),
                "average_matching_network_loss": float(np.mean([
                    item["average_matching_network_loss"] for item in selected
                ])),
            }

    def _candidate_from_log_values(
        self,
        branch_kinds: Sequence[str],
        input_topology: Sequence[tuple[str, str]],
        log_values: np.ndarray,
    ) -> SwitchCandidate:
        values = np.exp(log_values)
        branch_count = len(branch_kinds)
        branches = tuple(
            SeriesReactance(kind, float(value)) for kind, value in zip(branch_kinds, values[:branch_count])
        )
        fixed = tuple(
            InputReactance(connection, kind, float(value))
            for (connection, kind), value in zip(input_topology, values[branch_count:])
        )
        return self.evaluate(branches, fixed)

    def _optimize_topology(
        self,
        branch_kinds: tuple[str, ...],
        input_topology: tuple[tuple[str, str], ...],
        topology_seed: int,
        *,
        restarts: int | None = None,
        iterations: int | None = None,
    ) -> list[SwitchCandidate]:
        kinds = (*branch_kinds, *(kind for _, kind in input_topology))
        bounds = np.asarray([np.log(self._bounds(kind)) for kind in kinds])
        rng = np.random.default_rng(topology_seed)
        engineering_seed = np.asarray([
            np.log(10e-9 if kind == "L" else 1e-12) for kind in kinds
        ])
        restart_count = self.config.restarts if restarts is None else restarts
        iteration_count = self.config.iterations if iterations is None else iterations
        seeds = [engineering_seed]
        if restart_count >= 2:
            seeds.append(np.mean(bounds, axis=1))
        seeds.extend(rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(max(0, restart_count - len(seeds))))
        results = []
        for seed in seeds:
            point = np.asarray(seed, dtype=float)
            best = self._candidate_from_log_values(branch_kinds, input_topology, point)
            step = self.config.initial_log_step
            for _ in range(iteration_count):
                improved = False
                for position in range(len(point)):
                    for direction in (-1.0, 1.0):
                        trial_point = point.copy()
                        trial_point[position] = np.clip(
                            trial_point[position] + direction * step,
                            bounds[position, 0],
                            bounds[position, 1],
                        )
                        trial = self._candidate_from_log_values(branch_kinds, input_topology, trial_point)
                        if trial.score_db > best.score_db:
                            point, best, improved = trial_point, trial, True
                if not improved:
                    step *= 0.5
                    if step < 1e-3:
                        break
            results.append(best)
        return results

    def optimize(
        self,
        branch_topologies: Sequence[Sequence[str]] | None = None,
        input_topologies: Sequence[Sequence[tuple[str, str]]] = ((), (("series", "L"),)),
    ) -> SwitchOptimizationResult:
        throw_count = self.loaded_states[0].throws_z.shape[1]
        if branch_topologies is None:
            branch_topologies = tuple(itertools.product(("C", "L"), repeat=throw_count))
        normalized_branches = [tuple(item) for item in branch_topologies]
        normalized_inputs = [tuple(item) for item in input_topologies]
        if any(len(item) != throw_count or any(kind not in ("L", "C") for kind in item) for item in normalized_branches):
            raise ValueError("every branch topology must provide L/C for each switch throw")
        if any(
            any(connection not in ("series", "shunt") or kind not in ("L", "C") for connection, kind in item)
            for item in normalized_inputs
        ):
            raise ValueError("input topologies must contain series/shunt L/C pairs")
        candidates = []
        topologies = list(itertools.product(normalized_branches, normalized_inputs))
        self._emit_progress("switch_topology_search", 0, len(topologies), "Searching switch topologies")
        for topology_index, (branches, fixed) in enumerate(topologies):
            self._check_cancel()
            candidates.extend(self._optimize_topology(branches, fixed, self.config.seed + topology_index))
            self._emit_progress(
                "switch_topology_search", topology_index + 1, len(topologies),
                f"branches={''.join(branches)}, input={fixed or 'none'}",
            )
        unique: dict[tuple, SwitchCandidate] = {}
        for candidate in candidates:
            signature = (
                tuple((item.kind, round(np.log(item.value), 10)) for item in candidate.branch_reactances),
                tuple((item.connection, item.kind, round(np.log(item.value), 10)) for item in candidate.input_reactances),
                tuple(sorted(candidate.state_by_configuration.items())),
            )
            if signature not in unique or candidate.score_db > unique[signature].score_db:
                unique[signature] = candidate
        all_ranked = sorted(unique.values(), key=lambda item: item.score_db, reverse=True)
        ranked = list(all_ranked[: self.config.keep])
        # Preserve the best 0/1/2-input-component alternative even when a
        # lower-complexity circuit falls outside the global top-K.
        represented = {len(candidate.input_reactances) for candidate in ranked}
        for candidate in all_ranked:
            complexity = len(candidate.input_reactances)
            if complexity not in represented:
                ranked.append(candidate)
                represented.add(complexity)
        ranked.sort(key=lambda item: item.score_db, reverse=True)
        self._emit_progress("complete", len(topologies), len(topologies), "Switch optimization complete")
        return SwitchOptimizationResult(ranked, self.evaluations, len(self.loaded_states))

    def optimize_full_network(
        self,
        *,
        max_input_elements: int = 2,
        branch_beam: int = 3,
        topology_beam: int = 8,
        coarse_iterations: int = 5,
    ) -> SwitchOptimizationResult:
        """Beam-search all branch types and the ordered 0–2 element input block."""
        if branch_beam <= 0 or topology_beam <= 0 or coarse_iterations <= 0:
            raise ValueError("switch full-network beam limits must be positive")
        throw_count = self.loaded_states[0].throws_z.shape[1]
        all_branches = tuple(itertools.product(("C", "L"), repeat=throw_count))
        # A one-series-L reference discovers promising throw types without
        # multiplying the first stage by every fixed-network topology.
        branch_stage = []
        self._emit_progress("switch_branch_discovery", 0, len(all_branches), "Discovering branch types")
        for index, branches in enumerate(all_branches):
            branch_stage.extend(self._optimize_topology(
                branches, (("series", "L"),), self.config.seed + index,
                restarts=1, iterations=coarse_iterations,
            ))
            self._emit_progress("switch_branch_discovery", index + 1, len(all_branches), "Branch topology")
        branch_ranked = sorted(branch_stage, key=lambda item: item.score_db, reverse=True)
        selected_branches = []
        for candidate in branch_ranked:
            signature = tuple(item.kind for item in candidate.branch_reactances)
            if signature not in selected_branches:
                selected_branches.append(signature)
            if len(selected_branches) >= branch_beam:
                break

        input_topologies = standard_switch_input_topologies(max_input_elements)
        pairs = list(itertools.product(selected_branches, input_topologies))
        coarse = []
        self._emit_progress("switch_input_topology_search", 0, len(pairs), "Searching 0–2 element input networks")
        for index, (branches, fixed) in enumerate(pairs):
            coarse.extend(self._optimize_topology(
                tuple(branches), tuple(fixed), self.config.seed + 1000 + index,
                restarts=1, iterations=coarse_iterations,
            ))
            self._emit_progress("switch_input_topology_search", index + 1, len(pairs), "Input topology")
        coarse_ranked = sorted(coarse, key=lambda item: item.score_db, reverse=True)
        selected_pairs = []
        for candidate in coarse_ranked:
            pair = (
                tuple(item.kind for item in candidate.branch_reactances),
                tuple((item.connection, item.kind) for item in candidate.input_reactances),
            )
            if pair not in selected_pairs:
                selected_pairs.append(pair)
            if len(selected_pairs) >= topology_beam:
                break

        refined = []
        self._emit_progress("switch_topology_refine", 0, len(selected_pairs), "Refining switch topologies")
        for index, (branches, fixed) in enumerate(selected_pairs):
            refined.extend(self._optimize_topology(
                branches, fixed, self.config.seed + 2000 + index
            ))
            self._emit_progress("switch_topology_refine", index + 1, len(selected_pairs), "Refined topology")
        all_candidates = [*coarse_ranked, *refined]
        unique = {}
        for candidate in all_candidates:
            signature = (
                tuple((item.kind, round(np.log(item.value), 10)) for item in candidate.branch_reactances),
                tuple((item.connection, item.kind, round(np.log(item.value), 10)) for item in candidate.input_reactances),
                tuple(sorted(candidate.state_by_configuration.items())),
            )
            if signature not in unique or candidate.score_db > unique[signature].score_db:
                unique[signature] = candidate
        all_ranked = sorted(unique.values(), key=lambda item: item.score_db, reverse=True)
        ranked = list(all_ranked[: self.config.keep])
        represented = {len(candidate.input_reactances) for candidate in ranked}
        for candidate in all_ranked:
            complexity = len(candidate.input_reactances)
            if complexity not in represented:
                ranked.append(candidate)
                represented.add(complexity)
        ranked.sort(key=lambda item: item.score_db, reverse=True)
        self._emit_progress("complete", len(selected_pairs), len(selected_pairs), "Full switch network optimization complete")
        return SwitchOptimizationResult(ranked, self.evaluations, len(self.loaded_states))


class SwitchMeasuredComponentOptimizer:
    """Snap ideal switch candidates to vendor S2P parts and rerank physically."""

    def __init__(
        self,
        ideal_optimizer: SwitchTunableOptimizer,
        inductors: Sequence[ComponentSpec],
        capacitors: Sequence[ComponentSpec],
        *,
        nearest_parts: int = 3,
        ideal_seed_keep: int = 3,
        result_keep: int = 20,
    ) -> None:
        if nearest_parts <= 0 or ideal_seed_keep <= 0 or result_keep <= 0:
            raise ValueError("measured switch search limits must be positive")
        self.ideal_optimizer = ideal_optimizer
        self.catalog = {
            "L": tuple(sorted((item for item in inductors if item.kind == "L"), key=lambda item: (item.value, item.name))),
            "C": tuple(sorted((item for item in capacitors if item.kind == "C"), key=lambda item: (item.value, item.name))),
        }
        if not self.catalog["L"] or not self.catalog["C"]:
            raise ValueError("measured switch search requires inductor and capacitor catalogs")
        self.nearest_parts = nearest_parts
        self.ideal_seed_keep = ideal_seed_keep
        self.result_keep = result_keep
        self.model_cache = {}
        self.physical_evaluations = 0

    def _nearest(self, kind: str, value: float) -> tuple[ComponentSpec, ...]:
        return tuple(sorted(
            self.catalog[kind],
            key=lambda item: (abs(np.log(item.value / value)), item.tolerance, item.name),
        )[: self.nearest_parts])

    def _model(self, component: ComponentSpec):
        key = component_model_key(component)
        if key not in self.model_cache:
            self.model_cache[key] = load_component_model(component)
        return self.model_cache[key]

    def _evaluate(
        self,
        branch_components: tuple[ComponentSpec, ...],
        input_components: tuple[tuple[str, ComponentSpec], ...],
    ) -> MeasuredSwitchCandidate:
        optimizer = self.ideal_optimizer
        optimizer._check_cancel()
        self.physical_evaluations += 1
        branch_models = tuple(self._model(item) for item in branch_components)
        input_models = tuple(
            InputModelPlacement(connection, self._model(item))
            for connection, item in input_components
        )
        state_sweeps = {
            loaded.label: evaluate_loaded_switch_physical_power(
                loaded, branch_models, input_elements=input_models
            )
            for loaded in optimizer.loaded_states
        }
        assignments, score, metrics = optimizer._select_state_sweeps(
            state_sweeps, len(branch_components) + len(input_components)
        )
        metrics["maximum_component_model_nonpassivity"] = max(
            max(0.0, -float(np.min(sweep.matching_network_loss)))
            for sweep in state_sweeps.values()
        )
        return MeasuredSwitchCandidate(
            branch_components, input_components, assignments, score, metrics
        )

    def optimize(self, ideal_candidates: Sequence[SwitchCandidate]) -> MeasuredSwitchOptimizationResult:
        seeds = list(ideal_candidates[: self.ideal_seed_keep])
        if not seeds:
            raise ValueError("measured switch search requires ideal candidates")
        represented_complexities = {len(candidate.input_reactances) for candidate in seeds}
        for candidate in ideal_candidates:
            complexity = len(candidate.input_reactances)
            if complexity not in represented_complexities:
                seeds.append(candidate)
                represented_complexities.add(complexity)
        combinations = {}
        for seed in seeds:
            choices = [self._nearest(item.kind, item.value) for item in seed.branch_reactances]
            choices.extend(self._nearest(item.kind, item.value) for item in seed.input_reactances)
            for selected in itertools.product(*choices):
                branch_count = len(seed.branch_reactances)
                branches = tuple(selected[:branch_count])
                inputs = tuple(
                    (reactance.connection, component)
                    for reactance, component in zip(seed.input_reactances, selected[branch_count:])
                )
                signature = (
                    tuple(item.name for item in branches),
                    tuple((connection, item.name) for connection, item in inputs),
                )
                combinations[signature] = (branches, inputs)
        total = len(combinations)
        self.ideal_optimizer._emit_progress(
            "switch_measured_refine", 0, total, "Evaluating measured switch parts"
        )
        evaluated = []
        for index, (branches, inputs) in enumerate(combinations.values()):
            evaluated.append(self._evaluate(branches, inputs))
            self.ideal_optimizer._emit_progress(
                "switch_measured_refine", index + 1, total, "Measured switch candidate"
            )
        all_ranked = sorted(evaluated, key=lambda item: item.score_db, reverse=True)
        ranked = list(all_ranked[: self.result_keep])
        represented = {len(candidate.input_components) for candidate in ranked}
        for candidate in all_ranked:
            complexity = len(candidate.input_components)
            if complexity not in represented:
                ranked.append(candidate)
                represented.add(complexity)
        ranked.sort(key=lambda item: item.score_db, reverse=True)
        for candidate in ranked:
            candidate.metrics["component_sha256"] = {
                item.name: component_sha256(item)
                for item in (*candidate.branch_components, *(part for _, part in candidate.input_components))
            }
        return MeasuredSwitchOptimizationResult(
            ranked, self.physical_evaluations, len(self.model_cache)
        )
