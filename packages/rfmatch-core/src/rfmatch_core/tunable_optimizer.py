"""Automatic shared-network synthesis for measured multi-state tuners."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .components import ComponentSpec
from .models import Candidate, Objective, Problem, S2PModel
from .optimizer import MatchingOptimizer, OptimizationCancelled, SearchConfig
from .physical import evaluate_physical_problem
from .physical_optimizer import (
    MeasuredPlacement,
    MeasuredSearchConfig,
    ModelPlacement,
    PORT_TOPOLOGY_PATTERNS,
    build_model_circuit_topology,
)
from .mdif import MDIFModel
from .tunable import TunableCandidate, TunableProblem, evaluate_tunable_physical, load_measured_placements


@dataclass
class TunableMeasuredCandidate:
    placements: tuple[MeasuredPlacement, ...]
    evaluation: TunableCandidate

    @property
    def score_db(self) -> float:
        return self.evaluation.score_db


@dataclass
class TunableMeasuredOptimizationResult:
    candidates: list[TunableMeasuredCandidate]
    ideal_evaluations: int
    exact_physical_evaluations: int
    loaded_component_models: int
    tuner_state_precomputations: int = 0
    ideal_frequency_points: int = 0

    @property
    def best(self) -> TunableMeasuredCandidate:
        if not self.candidates:
            raise ValueError("tunable measured optimization produced no candidates")
        return self.candidates[0]


def _placement_signature(placements: Sequence[MeasuredPlacement]) -> tuple:
    return tuple(
        (item.connection, item.port, item.component.name)
        for item in placements
    )


class _TunableIdealOptimizer(MatchingOptimizer):
    """Fast ideal network search with state selection inside every evaluation."""

    def __init__(self, owner: "TunableMeasuredComponentOptimizer"):
        self.owner = owner
        self.problem = owner.problem.base_problem
        self.objective = owner.objective
        self.config = SearchConfig(
            restarts=owner.config.ideal_restarts,
            iterations=owner.config.ideal_iterations,
            keep=owner.config.ideal_keep,
            seed=owner.config.seed,
        )
        self.evaluations = 0

    def _evaluate(self, elements):
        self.owner._check_cancel()
        self.evaluations += 1
        if self.evaluations == 1 or self.evaluations % 100 == 0:
            self.owner._emit_progress(
                "ideal_search",
                self.evaluations,
                0,
                "Evaluating topology/value candidates",
            )
        selected = []
        for configuration in self.owner.problem.configurations:
            state_candidates = [
                self.owner._evaluate_effective(configuration.name, state.label, elements)
                for state in self.owner.tuner.states
            ]
            selected.append((configuration, max(state_candidates, key=lambda item: item.score_db)))
        scores = np.asarray([candidate.score_db for _, candidate in selected], dtype=float)
        weights = np.asarray([configuration.weight for configuration, _ in selected], dtype=float)
        if not np.any(weights > 0):
            weights = np.ones_like(weights)
        average = float(np.average(scores, weights=weights))
        worst = float(np.min(scores))
        blend = self.owner.problem.configuration_average_weight
        return Candidate(
            list(elements),
            (1.0 - blend) * worst + blend * average,
            {
                "state_by_configuration": {
                    configuration.name: candidate.metrics["tuner_state"]
                    for configuration, candidate in selected
                },
                "weighted_average_score_db": average,
                "worst_configuration_score_db": worst,
            },
        )


class TunableMeasuredComponentOptimizer:
    """Surrogate ideal search followed by exact tuner/state physical ranking.

    The tuner-only network is first folded into one effective DUT per frequency
    configuration. Existing shared-network synthesis can then search topology
    and measured fixed parts quickly. Final candidates are always re-evaluated
    against the original DUT and every MDIF state, so surrogate approximations
    never leak into reported scores or state assignments.
    """

    def __init__(
        self,
        problem: TunableProblem,
        tuner: MDIFModel,
        inductors: Sequence[ComponentSpec],
        capacitors: Sequence[ComponentSpec],
        objective: Objective | None = None,
        config: MeasuredSearchConfig | None = None,
        port: int = 0,
        tuner_connection: str = "series",
        exact_seed_keep: int = 20,
        ideal_points_per_band: int = 9,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        if problem.base_problem.s_parameters.shape[1] != 1 or port != 0:
            raise ValueError("automatic tunable synthesis currently supports one-port DUTs")
        if tuner_connection not in ("series", "shunt"):
            raise ValueError("tuner_connection must be 'series' or 'shunt'")
        if exact_seed_keep <= 0:
            raise ValueError("exact_seed_keep must be positive")
        if ideal_points_per_band < 2:
            raise ValueError("ideal_points_per_band must be at least 2")
        self.problem = problem
        self.tuner = tuner
        self.inductors = tuple(inductors)
        self.capacitors = tuple(capacitors)
        self.objective = objective or Objective()
        self.config = config or MeasuredSearchConfig()
        self.port = port
        self.tuner_connection = tuner_connection
        self.exact_seed_keep = exact_seed_keep
        self.ideal_points_per_band = ideal_points_per_band
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.model_cache: dict[Path, S2PModel] = {}
        self._effective: dict[tuple[str, str], Problem] = {}
        self._tuner_sweeps = {}

    def _check_cancel(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise OptimizationCancelled("tunable synthesis was cancelled")

    def _emit_progress(self, stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
        if self.progress_callback is not None:
            self.progress_callback({
                "stage": stage,
                "current": int(current),
                "total": int(total),
                "message": message,
            })

    def _effective_problem(self, configuration, state_label: str, sweep) -> Problem:
        base = self.problem.base_problem
        sample_indices = set()
        for bands in configuration.bands_by_port.values():
            for band in bands:
                lo, hi = sorted((band.start_hz, band.stop_hz))
                for target in np.linspace(lo, hi, self.ideal_points_per_band):
                    sample_indices.add(int(np.argmin(np.abs(base.frequencies_hz - target))))
        indices = np.asarray(sorted(sample_indices), dtype=int)
        sampled_s = sweep.s_parameters[indices]
        accepted = np.maximum(1.0 - np.abs(sampled_s[:, 0, 0]) ** 2, 1e-15)
        effective_radiation = np.clip(sweep.total_efficiency[indices, 0] / accepted, 0.0, 1.0)
        return Problem(
            base.frequencies_hz[indices],
            sampled_s,
            dict(configuration.bands_by_port),
            base.z0,
            {self.port: effective_radiation},
            base.isolation_targets,
        )

    def _prepare_effective_problems(self) -> None:
        self._effective.clear()
        self._tuner_sweeps.clear()
        for state in self.tuner.states:
            self._check_cancel()
            topology = build_model_circuit_topology(
                1,
                (ModelPlacement(self.tuner_connection, self.port, state.as_s2p_model()),),
            )
            self._tuner_sweeps[state.label] = evaluate_physical_problem(
                self.problem.base_problem, topology
            )
            self._emit_progress(
                "tuner_state_precompute",
                len(self._tuner_sweeps),
                len(self.tuner.states),
                state.label,
            )
        for configuration in self.problem.configurations:
            for state in self.tuner.states:
                self._effective[(configuration.name, state.label)] = self._effective_problem(
                    configuration, state.label, self._tuner_sweeps[state.label]
                )

    def _evaluate_effective(self, configuration_name: str, state_label: str, elements) -> Candidate:
        from .evaluator import evaluate

        candidate = evaluate(
            self._effective[(configuration_name, state_label)],
            Candidate(list(elements)),
            self.objective,
        )
        candidate.metrics["tuner_state"] = state_label
        return candidate

    def optimize(self) -> TunableMeasuredOptimizationResult:
        self._check_cancel()
        self._emit_progress("preparing", message="Preparing measured tuner states")
        self._prepare_effective_problems()
        patterns = [
            [(connection, kind, self.port) for connection, kind in pattern]
            for pattern in PORT_TOPOLOGY_PATTERNS
            if pattern
        ]
        ideal_optimizer = _TunableIdealOptimizer(self)
        self._emit_progress("ideal_search", message="Searching ideal topologies and values")
        ideal = ideal_optimizer.optimize(patterns).candidates
        self._check_cancel()
        self._emit_progress(
            "measured_shortlist",
            ideal_optimizer.evaluations,
            ideal_optimizer.evaluations,
            "Snapping continuous values to measured parts",
        )

        measured_seeds: list[tuple[tuple[MeasuredPlacement, ...], float]] = [
            ((), ideal_optimizer._evaluate([]).score_db)
        ]
        for seed in ideal[: self.exact_seed_keep]:
            self._check_cancel()
            choices = []
            for element in seed.elements:
                catalog = self.inductors if element.kind == "L" else self.capacitors
                if not catalog:
                    raise ValueError(f"measured component catalog has no {element.kind}")
                choices.append(sorted(
                    catalog,
                    key=lambda item: (
                        abs(np.log(item.value / element.value)), item.tolerance, item.name
                    ),
                )[: self.config.nearest_parts])
            for selected in itertools.product(*choices):
                self._check_cancel()
                placements = tuple(
                    MeasuredPlacement(element.connection, self.port, component)
                    for element, component in zip(seed.elements, selected)
                )
                nominal = ideal_optimizer._evaluate([
                    type(element)(
                        element.connection,
                        component.kind,
                        element.port,
                        component.value,
                        component.name,
                    )
                    for element, component in zip(seed.elements, selected)
                ])
                measured_seeds.append((placements, nominal.score_db))
        unique_scored: dict[tuple, tuple[tuple[MeasuredPlacement, ...], float]] = {}
        for placements, nominal_score in measured_seeds:
            key = _placement_signature(placements)
            previous = unique_scored.get(key)
            if previous is None or nominal_score > previous[1]:
                unique_scored[key] = (placements, nominal_score)
        shortlisted = sorted(
            unique_scored.values(),
            key=lambda item: (-item[1], _placement_signature(item[0])),
        )[: self.exact_seed_keep]
        unique = {_placement_signature(seed): seed for seed, _ in shortlisted}
        exact: list[TunableMeasuredCandidate] = []
        total_exact = len(unique)
        self._emit_progress("exact_physical", 0, total_exact, "Running full measured S-parameter verification")
        for index, key in enumerate(sorted(unique), start=1):
            self._check_cancel()
            placements = unique[key]
            loaded = load_measured_placements(placements, self.model_cache)
            evaluation = evaluate_tunable_physical(
                self.problem,
                loaded,
                self.tuner,
                self.objective,
                tuner_port=self.port,
                tuner_connection=self.tuner_connection,
            )
            exact.append(TunableMeasuredCandidate(placements, evaluation))
            self._emit_progress("exact_physical", index, total_exact, "Verified measured candidate")
        exact.sort(key=lambda item: (-item.score_db, _placement_signature(item.placements)))
        self._emit_progress("complete", total_exact, total_exact, "Tunable synthesis complete")
        return TunableMeasuredOptimizationResult(
            exact,
            ideal_optimizer.evaluations,
            len(exact),
            len(self.model_cache),
            len(self._tuner_sweeps),
            max(len(problem.frequencies_hz) for problem in self._effective.values()),
        )
