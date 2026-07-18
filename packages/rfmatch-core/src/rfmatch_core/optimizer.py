from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Callable, Sequence

import numpy as np

from .evaluator import evaluate, evaluate_lumped_physical
from .models import Candidate, Component, Element, LumpedLossModel, Objective, Problem


class OptimizationCancelled(RuntimeError):
    """Raised when a caller cooperatively stops an optimization search."""


@dataclass(frozen=True)
class SearchConfig:
    restarts: int = 12
    iterations: int = 30
    initial_log_step: float = 1.0
    discrete_neighbors: int = 3
    keep: int = 20
    seed: int = 1


@dataclass
class OptimizationResult:
    candidates: list[Candidate]
    evaluations: int

    @property
    def best(self) -> Candidate:
        return self.candidates[0]


class MatchingOptimizer:
    """Continuous multi-start search followed by discrete neighborhood search."""

    def __init__(
        self,
        problem: Problem,
        objective: Objective | None = None,
        config: SearchConfig | None = None,
        loss_model: LumpedLossModel | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        self.problem = problem
        self.objective = objective or Objective()
        self.config = config or SearchConfig()
        self.loss_model = loss_model
        self.cancel_check = cancel_check
        self.evaluations = 0

    def _check_cancel(self) -> None:
        cancel_check = getattr(self, "cancel_check", None)
        if cancel_check is not None and cancel_check():
            raise OptimizationCancelled("matching optimization cancelled")

    def _evaluate(self, elements: list[Element]) -> Candidate:
        self._check_cancel()
        self.evaluations += 1
        candidate = Candidate(elements)
        if self.loss_model is not None:
            return evaluate_lumped_physical(
                self.problem, candidate, self.objective, self.loss_model
            )
        return evaluate(self.problem, candidate, self.objective)

    @staticmethod
    def _bounds(kind: str) -> tuple[float, float]:
        return (1e-11, 1e-6) if kind == "L" else (1e-14, 1e-8)

    def optimize_continuous(self, topology: Sequence[tuple[str, str, int]]) -> list[Candidate]:
        rng = np.random.default_rng(self.config.seed)
        bounds = np.asarray([np.log(self._bounds(kind)) for _, kind, _ in topology])
        seeds = [np.mean(bounds, axis=1)]
        seeds.extend(rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(max(0, self.config.restarts - 1)))
        results = []
        for seed in seeds:
            self._check_cancel()
            results.append(self._optimize_seed(topology, seed, bounds))
        return self._unique_sorted(results)

    def _optimize_seed(self, topology, seed, bounds=None, iterations: int | None = None) -> Candidate:
        bounds = np.asarray(bounds if bounds is not None else [
            np.log(self._bounds(kind)) for _, kind, _ in topology
        ])
        point = np.asarray(seed, dtype=float)
        best = self._from_log_values(topology, point)
        step = self.config.initial_log_step
        iteration_count = self.config.iterations if iterations is None else max(0, int(iterations))
        for _ in range(iteration_count):
            self._check_cancel()
            improved = False
            for pos in range(len(point)):
                for direction in (-1.0, 1.0):
                    trial_point = point.copy()
                    trial_point[pos] = np.clip(
                        trial_point[pos] + direction * step, bounds[pos, 0], bounds[pos, 1]
                    )
                    trial = self._from_log_values(topology, trial_point)
                    if trial.score_db > best.score_db:
                        point, best, improved = trial_point, trial, True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        return best

    def discover_ladder_topologies(
        self,
        port: int,
        maximum_components: int,
        topology_beam_width: int = 12,
    ) -> OptimizationResult:
        """Progressively discover alternating series/shunt L/C ladders up to a depth."""
        if maximum_components < 1:
            return OptimizationResult([], self.evaluations)
        if topology_beam_width < 1:
            raise ValueError("topology_beam_width must be positive")
        frontier = []
        retained = []
        for connection, kind in itertools.product(("series", "shunt"), ("L", "C")):
            topology = ((connection, kind, port),)
            candidate = self.optimize_continuous(topology)[0]
            frontier.append((topology, candidate))
        for depth in range(1, maximum_components + 1):
            frontier.sort(key=lambda item: item[1].score_db, reverse=True)
            frontier = frontier[:topology_beam_width]
            retained.extend(candidate for _, candidate in frontier)
            if depth == maximum_components:
                break
            expanded = []
            for topology, parent in frontier:
                next_connection = "shunt" if topology[-1][0] == "series" else "series"
                for kind in ("L", "C"):
                    child = topology + ((next_connection, kind, port),)
                    bounds = np.asarray([np.log(self._bounds(item[1])) for item in child])
                    seed = np.asarray(
                        [np.log(element.value) for element in parent.elements]
                        + [float(np.mean(bounds[-1]))]
                    )
                    # Parent values have already converged at the previous
                    # depth. A bounded incremental refinement avoids repeating
                    # the full cold-start budget for every child topology.
                    child_iterations = max(1, (self.config.iterations + 3) // 4)
                    expanded.append((child, self._optimize_seed(
                        child, seed, bounds, iterations=child_iterations,
                    )))
            frontier = expanded
        return OptimizationResult(self._unique_sorted(retained)[: self.config.keep], self.evaluations)

    def _from_log_values(self, topology, log_values) -> Candidate:
        elements = [Element(connection, kind, port, float(np.exp(value))) for (connection, kind, port), value in zip(topology, log_values)]
        return self._evaluate(elements)

    def snap_to_library(self, continuous: Sequence[Candidate], library: Sequence[Component]) -> list[Candidate]:
        by_kind = {kind: sorted((c for c in library if c.kind == kind), key=lambda c: c.value) for kind in ("L", "C")}
        results = []
        for seed in continuous:
            self._check_cancel()
            choices = []
            for element in seed.elements:
                parts = by_kind[element.kind]
                if not parts:
                    raise ValueError(f"component library has no {element.kind}")
                nearest = sorted(parts, key=lambda c: abs(np.log(c.value / element.value)))[: self.config.discrete_neighbors]
                choices.append(nearest)
            for combo in itertools.product(*choices):
                elements = [Element(old.connection, part.kind, old.port, part.value, part.name) for old, part in zip(seed.elements, combo)]
                results.append(self._evaluate(elements))
        return self._unique_sorted(results)

    def optimize(self, topologies: Sequence[Sequence[tuple[str, str, int]]], library: Sequence[Component] | None = None) -> OptimizationResult:
        all_candidates = []
        for topology in topologies:
            self._check_cancel()
            continuous = self.optimize_continuous(topology)
            all_candidates.extend(self.snap_to_library(continuous[:3], library) if library else continuous)
        return OptimizationResult(self._unique_sorted(all_candidates)[: self.config.keep], self.evaluations)

    @staticmethod
    def _unique_sorted(candidates: Sequence[Candidate]) -> list[Candidate]:
        unique = {}
        for candidate in candidates:
            signature = tuple((e.connection, e.kind, e.port, round(np.log(e.value), 10)) for e in candidate.elements)
            if signature not in unique or candidate.score_db > unique[signature].score_db:
                unique[signature] = candidate
        return sorted(unique.values(), key=lambda c: c.score_db, reverse=True)
