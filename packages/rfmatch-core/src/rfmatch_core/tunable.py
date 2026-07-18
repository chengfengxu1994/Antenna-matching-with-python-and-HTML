"""Physical evaluation of one shared network with a selectable MDIF state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .components import load_component_model
from .evaluator import score_sweep
from .mdif import MDIFModel, MDIFState
from .models import Band, Candidate, Objective, Problem, S2PModel
from .physical import evaluate_physical_problem
from .physical_optimizer import MeasuredPlacement, ModelPlacement, build_model_circuit_topology


@dataclass(frozen=True)
class FrequencyConfiguration:
    """A named operating configuration containing one or more active bands."""

    name: str
    bands_by_port: Mapping[int, Sequence[Band]]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("frequency configuration name must not be empty")
        if not self.bands_by_port or not any(self.bands_by_port.values()):
            raise ValueError("frequency configuration requires at least one band")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("frequency configuration weight must be finite and non-negative")


@dataclass(frozen=True)
class TunableProblem:
    base_problem: Problem
    configurations: tuple[FrequencyConfiguration, ...]
    configuration_average_weight: float = 0.5

    def __post_init__(self) -> None:
        if not self.configurations:
            raise ValueError("tunable problem requires at least one frequency configuration")
        if not 0.0 <= self.configuration_average_weight <= 1.0:
            raise ValueError("configuration_average_weight must be between 0 and 1")
        port_count = self.base_problem.s_parameters.shape[1]
        for configuration in self.configurations:
            for port, bands in configuration.bands_by_port.items():
                if not 0 <= port < port_count:
                    raise ValueError(f"invalid configuration port {port}")
                for band in bands:
                    if not np.any(band.mask(self.base_problem.frequencies_hz)):
                        raise ValueError(f"configuration band {band} has no DUT frequency samples")


@dataclass
class TunableCandidate:
    fixed_placements: tuple[ModelPlacement, ...]
    state_by_configuration: dict[str, str]
    score_db: float
    metrics: dict = field(default_factory=dict)


@dataclass
class TunableSearchResult:
    """Deterministically ranked shared fixed networks and their MDIF states."""

    candidates: list[TunableCandidate]
    physical_evaluations: int

    @property
    def best(self) -> TunableCandidate:
        if not self.candidates:
            raise ValueError("tunable search produced no candidates")
        return self.candidates[0]


def _configuration_problem(problem: TunableProblem, configuration: FrequencyConfiguration) -> Problem:
    base = problem.base_problem
    return Problem(
        base.frequencies_hz,
        base.s_parameters,
        dict(configuration.bands_by_port),
        base.z0,
        base.radiation_efficiency,
        base.isolation_targets,
    )


def _state_evaluation(
    problem: Problem,
    fixed_placements: Sequence[ModelPlacement],
    tuner_state: MDIFState,
    objective: Objective,
    tuner_port: int,
    tuner_connection: str,
    physical_sweep=None,
) -> Candidate:
    placements = (
        ModelPlacement(tuner_connection, tuner_port, tuner_state.as_s2p_model()),
        *fixed_placements,
    )
    if physical_sweep is None:
        topology = build_model_circuit_topology(problem.s_parameters.shape[1], placements)
        physical_sweep = evaluate_physical_problem(problem, topology)
    sweep = physical_sweep
    candidate = score_sweep(
        problem,
        Candidate([]),
        objective,
        sweep.s_parameters,
        sweep.total_efficiency,
    )
    # score_sweep sees no ideal Elements; account for every physical branch here.
    candidate.score_db -= objective.complexity_penalty_db * len(placements)
    candidate.metrics.update({
        "s_parameters": sweep.s_parameters,
        "component_loss": sweep.component_loss,
        "dut_absorbed_power": sweep.dut_absorbed_power,
        "power_balance_error": sweep.power_balance_error,
        "maximum_power_balance_error": float(np.max(np.abs(sweep.power_balance_error))),
        "tuner_state": tuner_state.label,
    })
    return candidate


def evaluate_tunable_physical(
    problem: TunableProblem,
    fixed_placements: Sequence[ModelPlacement],
    tuner: MDIFModel,
    objective: Objective | None = None,
    state_by_configuration: Mapping[str, str | float] | None = None,
    tuner_port: int = 0,
    tuner_connection: str = "series",
) -> TunableCandidate:
    """Choose or evaluate one real tuner state for every frequency configuration.

    Placement order is DUT outward.  The tuner is therefore placed closest to
    the DUT, followed by the caller's shared fixed network.
    """
    if tuner_connection not in ("series", "shunt"):
        raise ValueError("tuner_connection must be 'series' or 'shunt'")
    port_count = problem.base_problem.s_parameters.shape[1]
    if not 0 <= tuner_port < port_count:
        raise ValueError("tuner port is outside the DUT port matrix")
    objective = objective or Objective()
    requested = dict(state_by_configuration or {})
    unknown = set(requested) - {item.name for item in problem.configurations}
    if unknown:
        raise ValueError(f"state assignment names unknown configurations: {sorted(unknown)}")

    selected: list[tuple[FrequencyConfiguration, Candidate]] = []
    assignments: dict[str, str] = {}
    physical_sweeps = {}
    for configuration in problem.configurations:
        configured_problem = _configuration_problem(problem, configuration)
        if configuration.name in requested:
            states = (tuner.state(requested[configuration.name]),)
        else:
            states = tuner.states
        evaluated = []
        for state in states:
            if state.label not in physical_sweeps:
                placements = (
                    ModelPlacement(tuner_connection, tuner_port, state.as_s2p_model()),
                    *fixed_placements,
                )
                topology = build_model_circuit_topology(port_count, placements)
                physical_sweeps[state.label] = evaluate_physical_problem(
                    problem.base_problem, topology
                )
            evaluated.append(_state_evaluation(
                configured_problem,
                fixed_placements,
                state,
                objective,
                tuner_port,
                tuner_connection,
                physical_sweeps[state.label],
            ))
        best = max(evaluated, key=lambda item: item.score_db)
        assignments[configuration.name] = best.metrics["tuner_state"]
        selected.append((configuration, best))

    scores = np.asarray([candidate.score_db for _, candidate in selected], dtype=float)
    weights = np.asarray([configuration.weight for configuration, _ in selected], dtype=float)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    average_score = float(np.average(scores, weights=weights))
    worst_score = float(np.min(scores))
    average_weight = problem.configuration_average_weight
    score = (1.0 - average_weight) * worst_score + average_weight * average_score
    configuration_metrics = []
    for configuration, candidate in selected:
        total_efficiency = np.asarray(candidate.metrics["total_efficiency"])
        configuration_metrics.append({
            "name": configuration.name,
            "weight": configuration.weight,
            "state": candidate.metrics["tuner_state"],
            "score_db": float(candidate.score_db),
            "average_total_efficiency": float(np.mean(total_efficiency)),
            "minimum_total_efficiency": float(np.min(total_efficiency)),
            "maximum_power_balance_error": candidate.metrics["maximum_power_balance_error"],
            "metrics": candidate.metrics,
        })
    return TunableCandidate(
        tuple(fixed_placements),
        assignments,
        score,
        {
            "configurations": configuration_metrics,
            "weighted_average_score_db": average_score,
            "worst_configuration_score_db": worst_score,
            "configuration_average_weight": average_weight,
            "maximum_power_balance_error": max(
                item["maximum_power_balance_error"] for item in configuration_metrics
            ),
            "physical_sweep_evaluations": len(physical_sweeps),
        },
    )


def rank_tunable_fixed_networks(
    problem: TunableProblem,
    candidate_networks: Sequence[Sequence[ModelPlacement]],
    tuner: MDIFModel,
    objective: Objective | None = None,
    result_keep: int = 20,
    tuner_port: int = 0,
    tuner_connection: str = "series",
) -> TunableSearchResult:
    """Jointly rank fixed-network candidates and automatic MDIF assignments.

    Candidate placement order is DUT outward. Duplicate networks are evaluated
    once and ties are resolved by a stable physical signature, making the result
    reproducible regardless of the caller's candidate order.
    """
    if result_keep <= 0:
        raise ValueError("result_keep must be positive")

    def signature(placements: Sequence[ModelPlacement]) -> tuple:
        return tuple(
            (
                placement.connection,
                placement.port,
                placement.model.name,
                getattr(placement.model, "kind", "S2P"),
                float(getattr(placement.model, "value", 0.0)),
            )
            for placement in placements
        )

    unique = {signature(network): tuple(network) for network in candidate_networks}
    evaluated = [
        evaluate_tunable_physical(
            problem,
            unique[key],
            tuner,
            objective,
            tuner_port=tuner_port,
            tuner_connection=tuner_connection,
        )
        for key in sorted(unique)
    ]
    evaluated.sort(key=lambda candidate: (-candidate.score_db, signature(candidate.fixed_placements)))
    return TunableSearchResult(evaluated[:result_keep], len(evaluated))


def load_measured_placements(
    placements: Sequence[MeasuredPlacement],
    model_cache: dict[Path, S2PModel] | None = None,
) -> tuple[ModelPlacement, ...]:
    """Load measured fixed placements once for tunable-state evaluation."""
    model_cache = model_cache if model_cache is not None else {}
    loaded = []
    for placement in placements:
        path = placement.component.source_path
        model = model_cache.get(path)
        if model is None:
            model = load_component_model(placement.component)
            model_cache[path] = model
        loaded.append(ModelPlacement(placement.connection, placement.port, model))
    return tuple(loaded)
