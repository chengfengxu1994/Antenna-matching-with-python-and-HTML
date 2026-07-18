"""Shared matching-network evaluation and synthesis across DUT scenarios.

The scenario axis represents multiple measurements of the same RF feed (for
example free-space, cover and hand states).  A candidate therefore keeps one
ordered topology and one set of values/part numbers for every scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .components import ComponentSpec
from .evaluator import evaluate, score_sweep
from .models import Candidate, CircuitTopology, Component, Element, Objective, Problem, S2PModel
from .optimizer import MatchingOptimizer, OptimizationResult, SearchConfig
from .physical_optimizer import (
    PORT_TOPOLOGY_PATTERNS,
    MeasuredCandidate,
    MeasuredPlacement,
    MeasuredSearchConfig,
    evaluate_measured_candidate,
)
from .physical import evaluate_physical_problem


@dataclass(frozen=True)
class ScenarioProblem:
    name: str
    problem: Problem
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("scenario weight must be a finite non-negative number")


@dataclass(frozen=True)
class MultiScenarioProblem:
    scenarios: tuple[ScenarioProblem, ...]
    scenario_average_weight: float = 0.5

    def __post_init__(self) -> None:
        if len(self.scenarios) < 2:
            raise ValueError("multi-scenario problem requires at least two scenarios")
        if not 0.0 <= self.scenario_average_weight <= 1.0:
            raise ValueError("scenario_average_weight must be between 0 and 1")
        port_counts = {item.problem.s_parameters.shape[1] for item in self.scenarios}
        if len(port_counts) != 1:
            raise ValueError("all scenarios must describe the same number of ports")

    @classmethod
    def from_mode(
        cls,
        scenarios: Sequence[ScenarioProblem],
        mode: Literal["balanced", "average", "worst_case"] = "balanced",
    ) -> "MultiScenarioProblem":
        weights = {"balanced": 0.5, "average": 1.0, "worst_case": 0.0}
        if mode not in weights:
            raise ValueError(f"unknown multi-scenario objective mode: {mode}")
        return cls(tuple(scenarios), weights[mode])


def _aggregate_metrics(
    problem: MultiScenarioProblem,
    names_and_candidates: Sequence[tuple[ScenarioProblem, Candidate | MeasuredCandidate]],
) -> tuple[float, dict]:
    scores = np.asarray([candidate.score_db for _, candidate in names_and_candidates], dtype=float)
    weights = np.asarray([scenario.weight for scenario, _ in names_and_candidates], dtype=float)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    weighted_score = float(np.average(scores, weights=weights))
    worst_score = float(np.min(scores))
    average_weight = problem.scenario_average_weight
    score = (1.0 - average_weight) * worst_score + average_weight * weighted_score

    scenario_metrics = []
    scenario_average_efficiencies = []
    minimum_efficiencies = []
    minimum_return_losses = []
    for scenario, candidate in names_and_candidates:
        metrics = candidate.metrics
        efficiency = np.asarray(metrics["total_efficiency"], dtype=float)
        return_loss = np.asarray(metrics["return_loss_db"], dtype=float)
        scenario_average_efficiencies.append(float(np.mean(efficiency)))
        minimum_efficiencies.append(float(np.min(efficiency)))
        minimum_return_losses.append(float(np.min(return_loss)))
        scenario_metrics.append({
            "name": scenario.name,
            "weight": scenario.weight,
            "score_db": float(candidate.score_db),
            "average_total_efficiency": float(np.mean(efficiency)),
            "minimum_total_efficiency": float(np.min(efficiency)),
            "minimum_return_loss_db": float(np.min(return_loss)),
            "metrics": metrics,
        })

    summary_weights = np.asarray([item.weight for item, _ in names_and_candidates], dtype=float)
    if not np.any(summary_weights > 0):
        summary_weights = np.ones_like(summary_weights)
    return score, {
        "scenarios": scenario_metrics,
        "weighted_average_score_db": weighted_score,
        "worst_scenario_score_db": worst_score,
        "scenario_average_weight": average_weight,
        "average_total_efficiency": float(np.average(scenario_average_efficiencies, weights=summary_weights)),
        "minimum_total_efficiency": float(np.min(minimum_efficiencies)),
        "minimum_return_loss_db": float(np.min(minimum_return_losses)),
        "scenario_constraints_passed": all(
            item.metrics.get("isolation_constraints_passed", True)
            for _, item in names_and_candidates
        ),
    }


def evaluate_multi_scenario(
    problem: MultiScenarioProblem,
    candidate: Candidate,
    objective: Objective,
) -> Candidate:
    """Evaluate one ideal topology/value set unchanged on every scenario."""
    evaluated = [
        (scenario, evaluate(scenario.problem, Candidate(list(candidate.elements)), objective))
        for scenario in problem.scenarios
    ]
    score, metrics = _aggregate_metrics(problem, evaluated)
    candidate.score_db = score
    candidate.metrics = metrics
    return candidate


class MultiScenarioMatchingOptimizer(MatchingOptimizer):
    """Continuous/discrete synthesis using the shared-scenario objective."""

    def __init__(
        self,
        problem: MultiScenarioProblem,
        objective: Objective | None = None,
        config: SearchConfig | None = None,
    ):
        self.problem = problem
        self.objective = objective or Objective()
        self.config = config or SearchConfig()
        self.evaluations = 0

    def _evaluate(self, elements: list[Element]) -> Candidate:
        self.evaluations += 1
        return evaluate_multi_scenario(self.problem, Candidate(elements), self.objective)


def evaluate_measured_multi_scenario(
    problem: MultiScenarioProblem,
    placements: Sequence[MeasuredPlacement],
    objective: Objective,
    model_cache: dict[Path, S2PModel] | None = None,
) -> MeasuredCandidate:
    """Evaluate one ordered measured-part network unchanged on every scenario."""
    evaluated = [
        (
            scenario,
            evaluate_measured_candidate(
                scenario.problem, placements, objective, model_cache
            ),
        )
        for scenario in problem.scenarios
    ]
    score, metrics = _aggregate_metrics(problem, evaluated)
    metrics["maximum_power_balance_error"] = max(
        candidate.metrics["maximum_power_balance_error"]
        for _, candidate in evaluated
    )
    return MeasuredCandidate(tuple(placements), score, metrics)


def evaluate_physical_multi_scenario(
    problem: MultiScenarioProblem,
    topology: CircuitTopology,
    candidate: Candidate,
    objective: Objective,
) -> Candidate:
    """Score one in-memory physical topology across every shared scenario."""
    evaluated = []
    for scenario in problem.scenarios:
        sweep = evaluate_physical_problem(scenario.problem, topology)
        scored = score_sweep(
            scenario.problem,
            Candidate(list(candidate.elements)),
            objective,
            sweep.s_parameters,
            sweep.total_efficiency,
        )
        scored.metrics.update({
            "s_parameters": sweep.s_parameters,
            "component_loss": sweep.component_loss,
            "dut_absorbed_power": sweep.dut_absorbed_power,
            "power_balance_error": sweep.power_balance_error,
            "maximum_power_balance_error": float(np.max(np.abs(sweep.power_balance_error))),
        })
        evaluated.append((scenario, scored))
    score, metrics = _aggregate_metrics(problem, evaluated)
    metrics["maximum_power_balance_error"] = max(
        item.metrics["maximum_power_balance_error"] for _, item in evaluated
    )
    candidate.score_db = score
    candidate.metrics = metrics
    return candidate


@dataclass
class SharedMeasuredOptimizationResult:
    candidates: list[MeasuredCandidate]
    ideal_evaluations: int
    physical_evaluations: int
    loaded_component_models: int

    @property
    def best(self) -> MeasuredCandidate:
        if not self.candidates:
            raise ValueError("shared measured optimization produced no candidates")
        return self.candidates[0]


def _measured_signature(candidate: MeasuredCandidate) -> tuple:
    return tuple(
        (item.connection, item.port, item.component.name)
        for item in candidate.placements
    )


def _unique_measured(candidates: Sequence[MeasuredCandidate]) -> list[MeasuredCandidate]:
    unique: dict[tuple, MeasuredCandidate] = {}
    for candidate in candidates:
        signature = _measured_signature(candidate)
        if signature not in unique or candidate.score_db > unique[signature].score_db:
            unique[signature] = candidate
    return sorted(unique.values(), key=lambda item: (-item.score_db, _measured_signature(item)))


def _diverse_refine_seeds(
    candidates: Sequence[MeasuredCandidate], keep: int
) -> list[MeasuredCandidate]:
    """Retain the best seed from different topology classes before duplicates."""
    ordered = _unique_measured(candidates)
    selected: list[MeasuredCandidate] = []
    signatures: set[tuple] = set()
    topology_codes: set[str] = set()
    for candidate in ordered:
        if candidate.topology_code not in topology_codes:
            selected.append(candidate)
            signatures.add(_measured_signature(candidate))
            topology_codes.add(candidate.topology_code)
            if len(selected) == keep:
                return selected
    for candidate in ordered:
        signature = _measured_signature(candidate)
        if signature not in signatures:
            selected.append(candidate)
            signatures.add(signature)
            if len(selected) == keep:
                break
    return selected


class SharedMeasuredComponentOptimizer:
    """Ideal shared seed search followed by measured-part refinement."""

    def __init__(
        self,
        problem: MultiScenarioProblem,
        inductors: Sequence[ComponentSpec],
        capacitors: Sequence[ComponentSpec],
        objective: Objective | None = None,
        config: MeasuredSearchConfig | None = None,
        port: int = 0,
    ):
        port_count = problem.scenarios[0].problem.s_parameters.shape[1]
        if not 0 <= port < port_count:
            raise ValueError("shared matching port is outside the scenario port matrix")
        self.problem = problem
        self.library = {"L": tuple(inductors), "C": tuple(capacitors)}
        self.objective = objective or Objective()
        self.config = config or MeasuredSearchConfig()
        self.port = port
        self.model_cache: dict[Path, S2PModel] = {}
        self.physical_evaluations = 0

    def _evaluate(self, placements: Sequence[MeasuredPlacement]) -> MeasuredCandidate:
        self.physical_evaluations += 1
        return evaluate_measured_multi_scenario(
            self.problem, placements, self.objective, self.model_cache
        )

    def _coordinate_refine(self, seed: MeasuredCandidate) -> MeasuredCandidate:
        best = seed
        for _ in range(self.config.joint_refine_passes):
            improved = False
            for index, placement in enumerate(best.placements):
                local = best
                for component in self.library[placement.component.kind]:
                    if component == placement.component:
                        continue
                    trial = list(best.placements)
                    trial[index] = MeasuredPlacement(placement.connection, self.port, component)
                    evaluated = self._evaluate(trial)
                    if evaluated.score_db > local.score_db:
                        local = evaluated
                if local.score_db > best.score_db:
                    best = local
                    improved = True
            if not improved:
                break
        return best

    def optimize(self) -> SharedMeasuredOptimizationResult:
        patterns = [
            [(connection, kind, self.port) for connection, kind in pattern]
            for pattern in PORT_TOPOLOGY_PATTERNS
            if pattern
        ]
        ideal_optimizer = MultiScenarioMatchingOptimizer(
            self.problem,
            self.objective,
            SearchConfig(
                restarts=self.config.ideal_restarts,
                iterations=self.config.ideal_iterations,
                keep=self.config.ideal_keep,
                seed=self.config.seed,
            ),
        )
        ideal = ideal_optimizer.optimize(patterns).candidates
        ideal.append(evaluate_multi_scenario(self.problem, Candidate([]), self.objective))

        measured: list[MeasuredCandidate] = []
        for seed in ideal:
            if not seed.elements:
                measured.append(self._evaluate(()))
                continue
            choices = []
            for element in seed.elements:
                parts = self.library[element.kind]
                if not parts:
                    raise ValueError(f"measured component catalog has no {element.kind}")
                choices.append(sorted(
                    parts,
                    key=lambda item: (
                        abs(np.log(item.value / element.value)), item.tolerance, item.name
                    ),
                )[: self.config.nearest_parts])
            for selected in itertools.product(*choices):
                measured.append(self._evaluate(tuple(
                    MeasuredPlacement(element.connection, self.port, component)
                    for element, component in zip(seed.elements, selected)
                )))

        initial = _unique_measured(measured)
        refine_seeds = _diverse_refine_seeds(
            initial, self.config.joint_refine_seeds
        )
        refined = [
            self._coordinate_refine(candidate)
            for candidate in refine_seeds
        ]
        candidates = _unique_measured([*initial, *refined])[: self.config.result_keep]
        return SharedMeasuredOptimizationResult(
            candidates=candidates,
            ideal_evaluations=ideal_optimizer.evaluations + 1,
            physical_evaluations=self.physical_evaluations,
            loaded_component_models=len(self.model_cache),
        )
