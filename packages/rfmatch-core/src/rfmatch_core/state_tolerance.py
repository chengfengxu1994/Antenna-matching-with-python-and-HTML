from __future__ import annotations

from copy import deepcopy
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

from .mdif import MDIFModel
from .models import S2PModel
from .physical import evaluate_physical_problem
from .physical_optimizer import ModelPlacement, build_model_circuit_topology
from .switch import (
    InputModelPlacement,
    InputReactance,
    LoadedSwitchState,
    SeriesReactance,
    evaluate_loaded_switch_physical_power,
)
from .switch_optimizer import SwitchTunableProblem
from .tolerance import (
    ToleranceModel,
    ToleranceResult,
    YieldCriteria,
    _geometric_mean_efficiency,
    _sample_model_variations,
    _variation_record,
    _variation_model_description,
)
from .tunable import TunableProblem


def _scale(
    tolerance: float,
    rng: np.random.Generator,
    distribution: str,
) -> float:
    if tolerance <= 0:
        return 1.0
    if distribution == "uniform":
        return float(rng.uniform(1.0 - tolerance, 1.0 + tolerance))
    if distribution == "normal":
        return float(np.clip(
            rng.normal(1.0, tolerance / 3.0),
            1.0 - tolerance,
            1.0 + tolerance,
        ))
    raise ValueError("distribution must be 'uniform' or 'normal'")


def _tolerance(item: SeriesReactance | S2PModel | InputReactance | InputModelPlacement) -> float:
    model = item.model if isinstance(item, InputModelPlacement) else item
    value = float(getattr(model, "tolerance", 0.0))
    if isinstance(model, S2PModel) and value > 0 and (
        model.kind not in {"L", "C"} or model.nominal_value is None
    ):
        raise ValueError(
            "S2P tolerance requires kind and nominal_value metadata; "
            "the complete S-parameter matrix is never scaled"
        )
    return value


def _kind(item: SeriesReactance | S2PModel | InputReactance | InputModelPlacement) -> str | None:
    model = item.model if isinstance(item, InputModelPlacement) else item
    return getattr(model, "kind", None)


def _environment_value(
    item: SeriesReactance | S2PModel | InputReactance | InputModelPlacement,
    field_name: str,
) -> float | None:
    model = item.model if isinstance(item, InputModelPlacement) else item
    value = getattr(model, field_name, None)
    return None if value is None else float(value)


def _model(item):
    return item.model if isinstance(item, InputModelPlacement) else item


def _margin_db(value: float, target: float) -> float:
    return float(10.0 * np.log10(max(value, 1e-15) / max(target, 1e-15)))


def monte_carlo_switch_yield(
    problem: SwitchTunableProblem,
    loaded_states: Mapping[str, LoadedSwitchState],
    branch_models: Sequence[SeriesReactance | S2PModel],
    input_elements: Sequence[InputReactance | InputModelPlacement],
    state_by_configuration: Mapping[str, str],
    criteria: YieldCriteria,
    *,
    samples: int = 500,
    seed: int = 1,
    distribution: str = "uniform",
    confidence_level: float = 0.95,
    tolerance_model: ToleranceModel | None = None,
) -> ToleranceResult:
    """Joint manufacturing yield across every switch state/configuration.

    One component draw is shared across all states. A sample passes only when
    every configuration satisfies its minimum efficiency, dB-averaged
    efficiency and return-loss targets.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if distribution not in {"uniform", "normal"}:
        raise ValueError("distribution must be 'uniform' or 'normal'")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if not 0.0 <= criteria.minimum_total_efficiency <= 1.0:
        raise ValueError("minimum_total_efficiency must be between 0 and 1")
    if not 0.0 <= criteria.minimum_average_total_efficiency <= 1.0:
        raise ValueError("minimum_average_total_efficiency must be between 0 and 1")
    if criteria.minimum_return_loss_db < 0:
        raise ValueError("minimum_return_loss_db must be non-negative")
    configuration_names = {item.name for item in problem.configurations}
    if set(state_by_configuration) != configuration_names:
        raise ValueError("state assignment must contain every configuration exactly once")
    selected_labels = set(state_by_configuration.values())
    missing = selected_labels - set(loaded_states)
    if missing:
        raise ValueError(f"loaded switch states are missing: {sorted(missing)}")
    for label in selected_labels:
        loaded = loaded_states[label]
        if not np.allclose(
            loaded.frequencies_hz, problem.frequencies_hz, rtol=1e-12, atol=1e-6
        ):
            raise ValueError(f"loaded state {label!r} uses a different frequency grid")

    sampled_models = [*map(_model, branch_models), *map(_model, input_elements)]
    variation_names = [
        *(f"branch_{index + 1}" for index in range(len(branch_models))),
        *(f"input_{index + 1}" for index in range(len(input_elements))),
    ]
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    variations: list[dict[str, float]] = []
    minimum_efficiencies: list[float] = []
    average_efficiencies: list[float] = []
    minimum_return_losses: list[float] = []
    configuration_passes = {
        configuration.name: 0 for configuration in problem.configurations
    }
    passed = 0
    for _ in range(samples):
        all_scales, environment = _sample_model_variations(
            sampled_models, rng, distribution, tolerance_model,
        )
        branch_scales = all_scales[:len(branch_models)]
        input_scales = all_scales[len(branch_models):]
        variation = {**_variation_record(variation_names, all_scales), **environment}
        sweeps = {
            label: evaluate_loaded_switch_physical_power(
                loaded_states[label],
                branch_models,
                input_elements=input_elements,
                branch_scales=branch_scales,
                input_scales=input_scales,
            )
            for label in selected_labels
        }
        configuration_minimums = []
        configuration_averages = []
        configuration_return_losses = []
        for configuration in problem.configurations:
            sweep = sweeps[state_by_configuration[configuration.name]]
            mask = np.logical_or.reduce([
                band.mask(problem.frequencies_hz)
                for band in configuration.bands_by_port[0]
            ])
            efficiency = np.asarray(sweep.dut_absorbed_power)[mask]
            return_loss = -20.0 * np.log10(
                np.maximum(np.abs(np.asarray(sweep.input_gamma)[mask]), 1e-15)
            )
            minimum_efficiency = float(np.min(efficiency))
            average_efficiency = _geometric_mean_efficiency(efficiency)
            minimum_return_loss = float(np.min(return_loss))
            configuration_minimums.append(minimum_efficiency)
            configuration_averages.append(average_efficiency)
            configuration_return_losses.append(minimum_return_loss)
            configuration_pass = (
                minimum_efficiency >= criteria.minimum_total_efficiency
                and average_efficiency >= criteria.minimum_average_total_efficiency
                and minimum_return_loss >= criteria.minimum_return_loss_db
            )
            configuration_passes[configuration.name] += int(configuration_pass)
        minimum_efficiency = min(configuration_minimums)
        # The worst configuration average prevents one easy state from hiding a bad one.
        average_efficiency = min(configuration_averages)
        minimum_return_loss = min(configuration_return_losses)
        score = min(
            _margin_db(minimum_efficiency, criteria.minimum_total_efficiency),
            _margin_db(average_efficiency, criteria.minimum_average_total_efficiency),
            minimum_return_loss - criteria.minimum_return_loss_db,
        )
        sample_pass = (
            minimum_efficiency >= criteria.minimum_total_efficiency
            and average_efficiency >= criteria.minimum_average_total_efficiency
            and minimum_return_loss >= criteria.minimum_return_loss_db
        )
        passed += int(sample_pass)
        scores.append(score)
        variations.append(variation)
        minimum_efficiencies.append(minimum_efficiency)
        average_efficiencies.append(average_efficiency)
        minimum_return_losses.append(minimum_return_loss)

    score_array = np.asarray(scores)
    fraction = passed / samples
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z_value**2 / samples
    center = (fraction + z_value**2 / (2.0 * samples)) / denominator
    half_width = z_value * np.sqrt(
        fraction * (1.0 - fraction) / samples
        + z_value**2 / (4.0 * samples**2)
    ) / denominator
    return ToleranceResult(
        samples=samples,
        passed_samples=passed,
        yield_fraction=fraction,
        yield_confidence_interval=(
            float(min(fraction, max(0.0, center - half_width))),
            float(max(fraction, min(1.0, center + half_width))),
        ),
        confidence_level=confidence_level,
        seed=seed,
        distribution=distribution,
        score_percentiles_db={
            percentile: float(np.percentile(score_array, percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
        worst_sample=variations[int(np.argmin(score_array))],
        sample_scores_db=score_array,
        sample_minimum_efficiency_db=10.0 * np.log10(
            np.maximum(np.asarray(minimum_efficiencies), 1e-15)
        ),
        sample_average_efficiency_db=10.0 * np.log10(
            np.maximum(np.asarray(average_efficiencies), 1e-15)
        ),
        sample_minimum_return_loss_db=np.asarray(minimum_return_losses),
        configuration_yield_fraction={
            name: count / samples for name, count in configuration_passes.items()
        },
        variation_model=_variation_model_description(tolerance_model, sampled_models),
    )


def monte_carlo_tunable_yield(
    problem: TunableProblem,
    fixed_placements: Sequence[ModelPlacement],
    tuner: MDIFModel,
    state_by_configuration: Mapping[str, str | float],
    criteria: YieldCriteria,
    *,
    tuner_port: int = 0,
    tuner_connection: str = "series",
    samples: int = 500,
    seed: int = 1,
    distribution: str = "uniform",
    confidence_level: float = 0.95,
    tolerance_model: ToleranceModel | None = None,
) -> ToleranceResult:
    """Joint yield of one shared fixed network across selected tuner states."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if distribution not in {"uniform", "normal"}:
        raise ValueError("distribution must be 'uniform' or 'normal'")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if tuner_connection not in {"series", "shunt"}:
        raise ValueError("tuner_connection must be 'series' or 'shunt'")
    if not 0.0 <= criteria.minimum_total_efficiency <= 1.0:
        raise ValueError("minimum_total_efficiency must be between 0 and 1")
    if not 0.0 <= criteria.minimum_average_total_efficiency <= 1.0:
        raise ValueError("minimum_average_total_efficiency must be between 0 and 1")
    if criteria.minimum_return_loss_db < 0:
        raise ValueError("minimum_return_loss_db must be non-negative")
    names = {configuration.name for configuration in problem.configurations}
    if set(state_by_configuration) != names:
        raise ValueError("state assignment must contain every configuration exactly once")

    selected_states = {
        configuration.name: tuner.state(state_by_configuration[configuration.name])
        for configuration in problem.configurations
    }
    unique_states = {state.label: state for state in selected_states.values()}
    unique_fixed = tuple(
        ModelPlacement(
            placement.connection,
            placement.port,
            deepcopy(placement.model),
        )
        for index, placement in enumerate(fixed_placements)
    )
    topologies = {}
    fixed_branch_names = {}
    for label, state in unique_states.items():
        placements = (
            ModelPlacement(tuner_connection, tuner_port, state.as_s2p_model()),
            *unique_fixed,
        )
        topology = build_model_circuit_topology(
            problem.base_problem.s_parameters.shape[1], placements
        )
        topologies[label] = topology
        branch_by_model = {id(branch.model): branch.name for branch in topology.branches}
        try:
            names_for_fixed = [
                branch_by_model[id(placement.model)] for placement in unique_fixed
            ]
        except KeyError as exc:
            raise ValueError("unable to map fixed tuner placements to physical branches")
        fixed_branch_names[label] = names_for_fixed

    fixed_models = [placement.model for placement in fixed_placements]
    variation_names = [f"fixed_{index + 1}" for index in range(len(fixed_placements))]
    rng = np.random.default_rng(seed)
    scores = []
    variations = []
    minimum_efficiencies = []
    average_efficiencies = []
    minimum_return_losses = []
    configuration_passes = {
        configuration.name: 0 for configuration in problem.configurations
    }
    passed = 0
    for _ in range(samples):
        scales, environment = _sample_model_variations(
            fixed_models, rng, distribution, tolerance_model,
        )
        sweeps = {}
        for label, topology in topologies.items():
            variation = dict(zip(fixed_branch_names[label], scales))
            sweeps[label] = evaluate_physical_problem(
                problem.base_problem, topology, variation
            )
        configuration_minimums = []
        configuration_averages = []
        configuration_return_losses = []
        for configuration in problem.configurations:
            sweep = sweeps[selected_states[configuration.name].label]
            point_efficiencies = []
            point_return_losses = []
            for port, bands in configuration.bands_by_port.items():
                mask = np.logical_or.reduce([
                    band.mask(problem.base_problem.frequencies_hz) for band in bands
                ])
                point_efficiencies.extend(sweep.total_efficiency[mask, port])
                point_return_losses.extend(
                    -20.0 * np.log10(np.maximum(
                        np.abs(sweep.s_parameters[mask, port, port]), 1e-15
                    ))
                )
            minimum_efficiency = float(np.min(point_efficiencies))
            average_efficiency = _geometric_mean_efficiency(point_efficiencies)
            minimum_return_loss = float(np.min(point_return_losses))
            configuration_minimums.append(minimum_efficiency)
            configuration_averages.append(average_efficiency)
            configuration_return_losses.append(minimum_return_loss)
            configuration_passes[configuration.name] += int(
                minimum_efficiency >= criteria.minimum_total_efficiency
                and average_efficiency >= criteria.minimum_average_total_efficiency
                and minimum_return_loss >= criteria.minimum_return_loss_db
            )
        minimum_efficiency = min(configuration_minimums)
        average_efficiency = min(configuration_averages)
        minimum_return_loss = min(configuration_return_losses)
        score = min(
            _margin_db(minimum_efficiency, criteria.minimum_total_efficiency),
            _margin_db(average_efficiency, criteria.minimum_average_total_efficiency),
            minimum_return_loss - criteria.minimum_return_loss_db,
        )
        sample_pass = (
            minimum_efficiency >= criteria.minimum_total_efficiency
            and average_efficiency >= criteria.minimum_average_total_efficiency
            and minimum_return_loss >= criteria.minimum_return_loss_db
        )
        passed += int(sample_pass)
        scores.append(score)
        variations.append({**_variation_record(variation_names, scales), **environment})
        minimum_efficiencies.append(minimum_efficiency)
        average_efficiencies.append(average_efficiency)
        minimum_return_losses.append(minimum_return_loss)

    score_array = np.asarray(scores)
    fraction = passed / samples
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z_value**2 / samples
    center = (fraction + z_value**2 / (2.0 * samples)) / denominator
    half_width = z_value * np.sqrt(
        fraction * (1.0 - fraction) / samples
        + z_value**2 / (4.0 * samples**2)
    ) / denominator
    return ToleranceResult(
        samples=samples,
        passed_samples=passed,
        yield_fraction=fraction,
        yield_confidence_interval=(
            float(min(fraction, max(0.0, center - half_width))),
            float(max(fraction, min(1.0, center + half_width))),
        ),
        confidence_level=confidence_level,
        seed=seed,
        distribution=distribution,
        score_percentiles_db={
            percentile: float(np.percentile(score_array, percentile))
            for percentile in (1, 5, 50, 95, 99)
        },
        worst_sample=variations[int(np.argmin(score_array))],
        sample_scores_db=score_array,
        sample_minimum_efficiency_db=10.0 * np.log10(
            np.maximum(np.asarray(minimum_efficiencies), 1e-15)
        ),
        sample_average_efficiency_db=10.0 * np.log10(
            np.maximum(np.asarray(average_efficiencies), 1e-15)
        ),
        sample_minimum_return_loss_db=np.asarray(minimum_return_losses),
        configuration_yield_fraction={
            name: count / samples for name, count in configuration_passes.items()
        },
        variation_model=_variation_model_description(tolerance_model, fixed_models),
    )
