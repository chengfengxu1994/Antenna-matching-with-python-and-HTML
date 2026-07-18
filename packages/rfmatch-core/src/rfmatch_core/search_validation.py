"""Exhaustive calibration tools for measured-component search quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from typing import Sequence

from .components import MeasuredComponentSpec
from .models import Objective, Problem, S2PModel
from .physical_optimizer import (
    PORT_TOPOLOGY_PATTERNS,
    MeasuredCandidate,
    MeasuredOptimizationResult,
    MeasuredPlacement,
    evaluate_measured_candidate,
)


@dataclass
class ExhaustiveMeasuredResult:
    candidates: list[MeasuredCandidate]
    physical_evaluations: int
    loaded_component_models: int

    @property
    def best(self) -> MeasuredCandidate:
        if not self.candidates:
            raise ValueError("exhaustive measured search produced no candidates")
        return self.candidates[0]


@dataclass(frozen=True)
class SearchRecallReport:
    top_k: int
    exhaustive_candidate_count: int
    heuristic_candidate_count: int
    exact_matches: int
    exact_top_k_recall: float
    topology_matches: int
    topology_top_k_recall: float
    best_score_gap_db: float
    best_exhaustive_rank_found: int | None
    exhaustive_physical_evaluations: int
    heuristic_physical_evaluations: int
    exhaustive_loaded_component_models: int
    heuristic_loaded_component_models: int

    def to_dict(self) -> dict:
        return asdict(self)


def measured_candidate_signature(candidate: MeasuredCandidate) -> tuple:
    """Stable exact signature: DUT-outward connection, port, kind, and part name."""
    return tuple(
        (
            placement.connection,
            placement.port,
            placement.component.kind,
            placement.component.name,
        )
        for placement in candidate.placements
    )


def measured_topology_signature(candidate: MeasuredCandidate) -> tuple:
    return tuple(
        (placement.connection, placement.port, placement.component.kind)
        for placement in candidate.placements
    )


def _unique_sorted(candidates: Sequence[MeasuredCandidate]) -> list[MeasuredCandidate]:
    unique: dict[tuple, MeasuredCandidate] = {}
    for candidate in candidates:
        signature = measured_candidate_signature(candidate)
        previous = unique.get(signature)
        if previous is None or candidate.score_db > previous.score_db:
            unique[signature] = candidate
    return sorted(
        unique.values(),
        key=lambda item: (-item.score_db, measured_candidate_signature(item)),
    )


def _product(values) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def exhaustive_measured_search(
    problem: Problem,
    inductors: Sequence[MeasuredComponentSpec],
    capacitors: Sequence[MeasuredComponentSpec],
    objective: Objective | None = None,
    *,
    max_components_per_port: int = 2,
    max_evaluations: int = 100_000,
) -> ExhaustiveMeasuredResult:
    """Enumerate every allowed part assignment for one actively matched port.

    The full DUT matrix is retained, so coupling and terminated-port absorption
    are still evaluated physically. This intentionally targets small catalogs;
    the explicit limit prevents accidental combinatorial production workloads.
    """
    if len(problem.bands_by_port) != 1:
        raise ValueError("use exhaustive_measured_joint_search for multiple active matching ports")
    return exhaustive_measured_joint_search(
        problem,
        inductors,
        capacitors,
        objective,
        max_components_per_port=max_components_per_port,
        max_evaluations=max_evaluations,
    )


def _port_placement_options(port: int, library: dict, max_components: int):
    options: list[tuple[MeasuredPlacement, ...]] = []
    for pattern in PORT_TOPOLOGY_PATTERNS:
        if len(pattern) > max_components:
            continue
        if not pattern:
            options.append(())
            continue
        choices = [library[kind] for _, kind in pattern]
        for selected in itertools.product(*choices):
            options.append(tuple(
                MeasuredPlacement(connection, port, component)
                for (connection, _), component in zip(pattern, selected)
            ))
    return options


def exhaustive_measured_joint_search(
    problem: Problem,
    inductors: Sequence[MeasuredComponentSpec],
    capacitors: Sequence[MeasuredComponentSpec],
    objective: Objective | None = None,
    *,
    max_components_per_port: int = 2,
    max_evaluations: int = 100_000,
) -> ExhaustiveMeasuredResult:
    """Enumerate the complete coupled assignment space across active ports."""
    if not problem.bands_by_port:
        raise ValueError("exhaustive measured calibration requires an active matching port")
    if not 0 <= max_components_per_port <= 4:
        raise ValueError("max_components_per_port must be between zero and four")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    library = {"L": tuple(inductors), "C": tuple(capacitors)}
    patterns = [
        pattern for pattern in PORT_TOPOLOGY_PATTERNS
        if len(pattern) <= max_components_per_port
    ]
    per_port_count = sum(
        1 if not pattern else _product(len(library[kind]) for _, kind in pattern)
        for pattern in patterns
    )
    expected = per_port_count ** len(problem.bands_by_port)
    if expected > max_evaluations:
        raise ValueError(
            f"exhaustive measured calibration requires {expected} evaluations, "
            f"above max_evaluations={max_evaluations}"
        )
    if any(not library[kind] for pattern in patterns for _, kind in pattern):
        raise ValueError("measured component catalog is missing a required L or C family")

    objective = objective or Objective()
    model_cache: dict[object, S2PModel] = {}
    candidates: list[MeasuredCandidate] = []
    port_options = [
        _port_placement_options(port, library, max_components_per_port)
        for port in problem.bands_by_port
    ]
    for selected_by_port in itertools.product(*port_options):
        placements = tuple(itertools.chain.from_iterable(selected_by_port))
        candidates.append(
            evaluate_measured_candidate(problem, placements, objective, model_cache)
        )
    return ExhaustiveMeasuredResult(
        _unique_sorted(candidates),
        len(candidates),
        len(model_cache),
    )


def measured_search_recall(
    heuristic: MeasuredOptimizationResult,
    exhaustive: ExhaustiveMeasuredResult,
    *,
    top_k: int = 10,
) -> SearchRecallReport:
    """Compare the returned heuristic list with the exact exhaustive ranking."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    exact_top = exhaustive.candidates[:top_k]
    if not exact_top:
        raise ValueError("exhaustive ranking is empty")
    effective_k = len(exact_top)
    heuristic_exact = {
        measured_candidate_signature(candidate) for candidate in heuristic.candidates
    }
    heuristic_topologies = {
        measured_topology_signature(candidate) for candidate in heuristic.candidates
    }
    exact_matches = sum(
        measured_candidate_signature(candidate) in heuristic_exact
        for candidate in exact_top
    )
    topology_matches = sum(
        measured_topology_signature(candidate) in heuristic_topologies
        for candidate in exact_top
    )
    ranked_signatures = {
        measured_candidate_signature(candidate): index + 1
        for index, candidate in enumerate(exhaustive.candidates)
    }
    found_ranks = [
        ranked_signatures[signature]
        for signature in heuristic_exact
        if signature in ranked_signatures
    ]
    heuristic_best_score = max(
        (candidate.score_db for candidate in heuristic.candidates),
        default=float("-inf"),
    )
    return SearchRecallReport(
        top_k=effective_k,
        exhaustive_candidate_count=len(exhaustive.candidates),
        heuristic_candidate_count=len(heuristic.candidates),
        exact_matches=exact_matches,
        exact_top_k_recall=exact_matches / effective_k,
        topology_matches=topology_matches,
        topology_top_k_recall=topology_matches / effective_k,
        best_score_gap_db=max(0.0, exhaustive.best.score_db - heuristic_best_score),
        best_exhaustive_rank_found=min(found_ranks) if found_ranks else None,
        exhaustive_physical_evaluations=exhaustive.physical_evaluations,
        heuristic_physical_evaluations=heuristic.physical_evaluations,
        exhaustive_loaded_component_models=exhaustive.loaded_component_models,
        heuristic_loaded_component_models=heuristic.loaded_component_models,
    )
