from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np

from .models import CircuitTopology, LumpedModel, Problem, S2PModel
from .microstrip import MicrostripLineModel, MicrostripStubModel, MicrostripVariation
from .physical import evaluate_circuit
from .optimizer import OptimizationCancelled


@dataclass(frozen=True)
class YieldCriteria:
    minimum_total_efficiency: float = 0.0
    minimum_return_loss_db: float = 0.0
    minimum_average_total_efficiency: float = 0.0


@dataclass(frozen=True)
class ToleranceModel:
    """Manufacturing and operating-environment variation assumptions.

    ``batch_correlation`` applies a Gaussian copula to component value draws,
    preserving the requested uniform or truncated-normal marginal distribution.
    A configured L/C bias shifts every component of that kind systematically.
    Temperature is sampled once per assembly and therefore affects every part
    coherently according to its L/C temperature coefficient.
    """

    batch_correlation: float = 0.0
    reference_temperature_c: float = 25.0
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    inductor_tempco_ppm_per_c: float = 0.0
    capacitor_tempco_ppm_per_c: float = 0.0
    inductor_bias_pct: float = 0.0
    capacitor_bias_pct: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.batch_correlation <= 1.0:
            raise ValueError("batch_correlation must be between 0 and 1")
        if (self.temperature_min_c is None) != (self.temperature_max_c is None):
            raise ValueError("temperature_min_c and temperature_max_c must be provided together")
        if self.temperature_min_c is not None:
            if self.temperature_min_c < -273.15:
                raise ValueError("temperature_min_c cannot be below absolute zero")
            if self.temperature_max_c < self.temperature_min_c:
                raise ValueError("temperature_max_c must be at least temperature_min_c")
        if self.inductor_bias_pct <= -100.0:
            raise ValueError("inductor_bias_pct must keep the component value positive")
        if self.capacitor_bias_pct <= -100.0:
            raise ValueError("capacitor_bias_pct must keep the component value positive")

    @property
    def temperature_enabled(self) -> bool:
        return self.temperature_min_c is not None

    def as_dict(self) -> dict:
        return {
            "batch_correlation": self.batch_correlation,
            "reference_temperature_c": self.reference_temperature_c,
            "temperature_min_c": self.temperature_min_c,
            "temperature_max_c": self.temperature_max_c,
            "inductor_tempco_ppm_per_c": self.inductor_tempco_ppm_per_c,
            "capacitor_tempco_ppm_per_c": self.capacitor_tempco_ppm_per_c,
            "inductor_bias_pct": self.inductor_bias_pct,
            "capacitor_bias_pct": self.capacitor_bias_pct,
            "correlation_method": "gaussian_copula",
            "temperature_scope": "one_shared_draw_per_assembly",
            "bias_scope": "systematic_by_component_kind",
        }


@dataclass
class ToleranceResult:
    samples: int
    passed_samples: int
    yield_fraction: float
    yield_confidence_interval: tuple[float, float]
    confidence_level: float
    seed: int
    distribution: str
    score_percentiles_db: dict[float, float]
    worst_sample: dict[str, float]
    sample_scores_db: np.ndarray
    sample_minimum_efficiency_db: np.ndarray
    sample_average_efficiency_db: np.ndarray
    sample_minimum_return_loss_db: np.ndarray
    configuration_yield_fraction: dict[str, float] = field(default_factory=dict)
    variation_model: dict = field(default_factory=dict)


def _geometric_mean_efficiency(values) -> float:
    """Optenni ave eff.: arithmetic mean in dB, geometric mean in linear units."""
    array = np.asarray(values, dtype=float)
    if not array.size:
        return 0.0
    return float(np.exp(np.mean(np.log(np.maximum(array, 1e-15)))))


def _variation(topology: CircuitTopology, rng: np.random.Generator, distribution: str) -> dict[str, float]:
    result = {}
    for branch in topology.branches:
        if isinstance(branch.model, S2PModel):
            tolerance = float(branch.model.tolerance)
            if tolerance > 0 and (
                branch.model.kind not in {"L", "C"}
                or branch.model.nominal_value is None
            ):
                raise ValueError(
                    "S2P tolerance requires kind and nominal_value metadata; "
                    "the complete S-parameter matrix is never scaled"
                )
        else:
            tolerance = float(getattr(branch.model, "tolerance", 0.0))
        if tolerance <= 0:
            result[branch.name] = 1.0
        elif distribution == "uniform":
            result[branch.name] = rng.uniform(1.0 - tolerance, 1.0 + tolerance)
        elif distribution == "normal":
            result[branch.name] = np.clip(rng.normal(1.0, tolerance / 3.0), 1.0 - tolerance, 1.0 + tolerance)
        else:
            raise ValueError("distribution must be 'uniform' or 'normal'")
    return result


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _sample_component_scales(
    tolerances: Sequence[float],
    kinds: Sequence[str | None],
    rng: np.random.Generator,
    distribution: str,
    tolerance_model: ToleranceModel | None,
    tempco_overrides_ppm_per_c: Sequence[float | None] | None = None,
    bias_overrides_pct: Sequence[float | None] | None = None,
) -> tuple[list[float], dict[str, float]]:
    """Draw one assembly while preserving legacy independent sampling exactly."""
    model = tolerance_model or ToleranceModel()
    if len(tolerances) != len(kinds):
        raise ValueError("tolerances and kinds must have the same length")
    if tempco_overrides_ppm_per_c is None:
        tempco_overrides_ppm_per_c = [None] * len(kinds)
    if bias_overrides_pct is None:
        bias_overrides_pct = [None] * len(kinds)
    if len(tempco_overrides_ppm_per_c) != len(kinds):
        raise ValueError("tempco overrides and kinds must have the same length")
    if len(bias_overrides_pct) != len(kinds):
        raise ValueError("bias overrides and kinds must have the same length")

    if model.batch_correlation == 0.0:
        manufacturing = []
        for tolerance in tolerances:
            if tolerance <= 0:
                manufacturing.append(1.0)
            elif distribution == "uniform":
                manufacturing.append(float(rng.uniform(1.0 - tolerance, 1.0 + tolerance)))
            elif distribution == "normal":
                manufacturing.append(float(np.clip(
                    rng.normal(1.0, tolerance / 3.0),
                    1.0 - tolerance,
                    1.0 + tolerance,
                )))
            else:
                raise ValueError("distribution must be 'uniform' or 'normal'")
    else:
        shared = float(rng.normal())
        shared_weight = sqrt(model.batch_correlation)
        individual_weight = sqrt(1.0 - model.batch_correlation)
        manufacturing = []
        for tolerance in tolerances:
            if tolerance <= 0:
                manufacturing.append(1.0)
                continue
            latent = shared_weight * shared + individual_weight * float(rng.normal())
            if distribution == "uniform":
                manufacturing.append(1.0 + tolerance * (2.0 * _normal_cdf(latent) - 1.0))
            elif distribution == "normal":
                manufacturing.append(float(np.clip(
                    1.0 + tolerance * latent / 3.0,
                    1.0 - tolerance,
                    1.0 + tolerance,
                )))
            else:
                raise ValueError("distribution must be 'uniform' or 'normal'")

    temperature = model.reference_temperature_c
    environment: dict[str, float] = {}
    if model.temperature_enabled:
        temperature = float(rng.uniform(model.temperature_min_c, model.temperature_max_c))
    delta_temperature = temperature - model.reference_temperature_c
    if model.temperature_enabled:
        environment = {
            "temperature_c": float(temperature),
            "temperature_delta_c": float(delta_temperature),
        }
    scales = []
    for value, kind, tempco_override, bias_override in zip(
        manufacturing, kinds, tempco_overrides_ppm_per_c, bias_overrides_pct
    ):
        bias_pct = float(bias_override) if bias_override is not None else (
            model.inductor_bias_pct if kind == "L"
            else model.capacitor_bias_pct if kind == "C"
            else 0.0
        )
        bias_scale = 1.0 + bias_pct / 100.0
        coefficient = float(tempco_override) if tempco_override is not None else (
            model.inductor_tempco_ppm_per_c if kind == "L"
            else model.capacitor_tempco_ppm_per_c if kind == "C"
            else 0.0
        )
        temperature_scale = 1.0 + coefficient * 1e-6 * delta_temperature
        if temperature_scale <= 0:
            raise ValueError("temperature coefficient produces a non-positive component value")
        scales.append(float(value * bias_scale * temperature_scale))
    return scales, environment


def _microstrip_line(model):
    if isinstance(model, MicrostripLineModel):
        return model
    if isinstance(model, MicrostripStubModel):
        return model.line
    return None


def _sample_model_variations(
    models: Sequence[object],
    rng: np.random.Generator,
    distribution: str,
    tolerance_model: ToleranceModel | None,
) -> tuple[list[float | MicrostripVariation], dict[str, float]]:
    """Sample scalar components and explicit PCB geometry in one assembly draw."""
    tolerances: list[float] = []
    kinds: list[str | None] = []
    tempcos: list[float | None] = []
    biases: list[float | None] = []
    descriptors: list[tuple[str, int]] = []
    for model in models:
        line = _microstrip_line(model)
        if line is not None:
            values = (
                line.width_tolerance,
                line.length_tolerance,
                line.substrate_height_tolerance,
                line.relative_permittivity_tolerance,
            )
            start = len(tolerances)
            tolerances.extend(map(float, values))
            kinds.extend([None] * 4)
            tempcos.extend([None] * 4)
            biases.extend([None] * 4)
            descriptors.append(("microstrip", start))
            continue
        tolerance = float(getattr(model, "tolerance", 0.0))
        if isinstance(model, S2PModel) and tolerance > 0 and (
            model.kind not in {"L", "C"} or model.nominal_value is None
        ):
            raise ValueError(
                "S2P tolerance requires kind and nominal_value metadata; "
                "the complete S-parameter matrix is never scaled"
            )
        descriptors.append(("scalar", len(tolerances)))
        tolerances.append(tolerance)
        kinds.append(getattr(model, "kind", None))
        tempcos.append(getattr(model, "tempco_ppm_per_c", None))
        biases.append(getattr(model, "systematic_bias_pct", None))

    values, environment = _sample_component_scales(
        tolerances, kinds, rng, distribution, tolerance_model, tempcos, biases,
    )
    variations: list[float | MicrostripVariation] = []
    for kind, start in descriptors:
        if kind == "scalar":
            variations.append(values[start])
        else:
            variations.append(MicrostripVariation(*values[start:start + 4]))
    return variations, environment


def _variation_record(
    names: Sequence[str], values: Sequence[float | MicrostripVariation],
) -> dict[str, float]:
    """Flatten structured PCB draws into stable JSON/report field names."""
    result: dict[str, float] = {}
    for name, value in zip(names, values):
        if isinstance(value, MicrostripVariation):
            result[f"{name}.trace_width"] = value.width_scale
            result[f"{name}.physical_length"] = value.length_scale
            result[f"{name}.substrate_height"] = value.substrate_height_scale
            result[f"{name}.relative_permittivity"] = value.relative_permittivity_scale
        else:
            result[name] = float(value)
    return result


def _variation_model_description(
    tolerance_model: ToleranceModel | None, models: Sequence[object],
) -> dict:
    result = (tolerance_model or ToleranceModel()).as_dict()
    if any(_microstrip_line(model) is not None for model in models):
        result.update({
            "microstrip_variables": [
                "trace_width", "physical_length", "substrate_height", "relative_permittivity",
            ],
            "microstrip_scope": "one_shared_manufactured_board_draw_across_states",
        })
    return result


def monte_carlo_yield(
    problem: Problem,
    topology: CircuitTopology,
    criteria: YieldCriteria,
    samples: int = 500,
    seed: int = 1,
    distribution: str = "uniform",
    confidence_level: float = 0.95,
    tolerance_model: ToleranceModel | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ToleranceResult:
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
    rng = np.random.default_rng(seed)
    scores, passed, variations = [], 0, []
    minimum_efficiencies, average_efficiencies, minimum_return_losses = [], [], []
    active_points = [
        (fi, port)
        for fi in range(len(problem.frequencies_hz))
        for port, bands in problem.bands_by_port.items()
        if any(band.mask(problem.frequencies_hz)[fi] for band in bands)
    ]
    branch_models = [branch.model for branch in topology.branches]
    branch_names = [branch.name for branch in topology.branches]
    progress_interval = max(1, samples // 50)
    for sample_index in range(samples):
        if cancel_check is not None and cancel_check():
            raise OptimizationCancelled("Monte Carlo yield analysis cancelled")
        scale_values, environment = _sample_model_variations(
            branch_models, rng, distribution, tolerance_model,
        )
        scale = dict(zip(branch_names, scale_values))
        point_eff, point_rl = [], []
        evaluations = [
            evaluate_circuit(problem.s_parameters[fi], topology, frequency, problem.z0, scale)
            for fi, frequency in enumerate(problem.frequencies_hz)
        ]
        for fi, port in active_points:
            solved = evaluations[fi]
            eta_rad = problem.radiation_efficiency.get(
                port, np.ones(len(problem.frequencies_hz))
            )[fi]
            point_eff.append(float(eta_rad) * solved.dut_absorbed_power[port])
            point_rl.append(
                -20.0 * np.log10(max(abs(solved.s_parameters[port, port]), 1e-15))
            )
        min_eff = min(point_eff, default=0.0)
        average_eff = _geometric_mean_efficiency(point_eff)
        min_rl = min(point_rl, default=0.0)
        efficiency_margin = 10.0 * np.log10(max(min_eff, 1e-15) / max(criteria.minimum_total_efficiency, 1e-15))
        average_efficiency_margin = 10.0 * np.log10(
            max(average_eff, 1e-15)
            / max(criteria.minimum_average_total_efficiency, 1e-15)
        )
        score = min(
            efficiency_margin,
            average_efficiency_margin,
            min_rl - criteria.minimum_return_loss_db,
        )
        scores.append(score)
        minimum_efficiencies.append(min_eff)
        average_efficiencies.append(average_eff)
        minimum_return_losses.append(min_rl)
        variations.append({**_variation_record(branch_names, scale_values), **environment})
        passed += (
            min_eff >= criteria.minimum_total_efficiency
            and average_eff >= criteria.minimum_average_total_efficiency
            and min_rl >= criteria.minimum_return_loss_db
        )
        if progress_callback is not None and (
            sample_index == samples - 1 or (sample_index + 1) % progress_interval == 0
        ):
            progress_callback({
                "stage": "manual_yield",
                "current": sample_index + 1,
                "total": samples,
                "yield_fraction": passed / (sample_index + 1),
                "worst_return_loss_db": float(min(minimum_return_losses)),
            })
    scores_array = np.asarray(scores)
    worst = int(np.argmin(scores_array))
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    fraction = passed / samples
    denominator = 1.0 + z_value**2 / samples
    center = (fraction + z_value**2 / (2.0 * samples)) / denominator
    half_width = z_value * np.sqrt(
        fraction * (1.0 - fraction) / samples + z_value**2 / (4.0 * samples**2)
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
        score_percentiles_db={p: float(np.percentile(scores_array, p)) for p in (1, 5, 50, 95, 99)},
        worst_sample=variations[worst],
        sample_scores_db=scores_array,
        sample_minimum_efficiency_db=10.0 * np.log10(
            np.maximum(np.asarray(minimum_efficiencies), 1e-15)
        ),
        sample_average_efficiency_db=10.0 * np.log10(
            np.maximum(np.asarray(average_efficiencies), 1e-15)
        ),
        sample_minimum_return_loss_db=np.asarray(minimum_return_losses),
        variation_model=_variation_model_description(tolerance_model, branch_models),
    )


def tolerance_summary(result: ToleranceResult) -> dict:
    """Strict-JSON summary suitable for candidate metrics and API responses."""
    return {
        "samples": int(result.samples),
        "passed_samples": int(result.passed_samples),
        "yield_fraction": float(result.yield_fraction),
        "yield_confidence_interval": [
            float(value) for value in result.yield_confidence_interval
        ],
        "confidence_level": float(result.confidence_level),
        "seed": int(result.seed),
        "distribution": str(result.distribution),
        "score_percentiles_db": {
            str(int(percentile)): float(value)
            for percentile, value in result.score_percentiles_db.items()
        },
        "worst_sample": {
            str(name): float(value) for name, value in result.worst_sample.items()
        },
        "configuration_yield_fraction": {
            str(name): float(value)
            for name, value in result.configuration_yield_fraction.items()
        },
        "variation_model": result.variation_model,
    }


def rank_measured_candidates_by_yield(
    problem: Problem,
    candidates,
    criteria: YieldCriteria,
    *,
    samples: int = 500,
    seed: int = 1,
    distribution: str = "uniform",
    confidence_level: float = 0.95,
    tolerance_model: ToleranceModel | None = None,
):
    """Evaluate every measured candidate and rank by robust yield evidence.

    Ranking uses Wilson lower confidence bound first, then point yield, the
    5th-percentile margin, and finally nominal score. This prevents a noisy
    small-sample 100% estimate from outranking a statistically stronger result.
    """
    from .physical_optimizer import build_circuit_topology

    ranked = []
    model_cache = {}
    for candidate in candidates:
        topology = build_circuit_topology(
            problem.s_parameters.shape[1], candidate.placements, model_cache
        )
        result = monte_carlo_yield(
            problem,
            topology,
            criteria,
            samples=samples,
            seed=seed,
            distribution=distribution,
            confidence_level=confidence_level,
            tolerance_model=tolerance_model,
        )
        summary = tolerance_summary(result)
        candidate.metrics["yield_analysis"] = summary
        ranked.append(candidate)
    return sorted(
        ranked,
        key=lambda candidate: (
            candidate.metrics["yield_analysis"]["yield_confidence_interval"][0],
            candidate.metrics["yield_analysis"]["yield_fraction"],
            candidate.metrics["yield_analysis"]["score_percentiles_db"]["5"],
            candidate.score_db,
        ),
        reverse=True,
    )
