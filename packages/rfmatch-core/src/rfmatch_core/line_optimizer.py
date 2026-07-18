"""Deterministic bounded synthesis for physical transmission lines and stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Sequence

import numpy as np

from .evaluator import score_sweep
from .models import Candidate, Element, Objective, Problem, S2PModel
from .optimizer import OptimizationCancelled
from .physical import evaluate_physical_problem
from .physical_optimizer import ModelPlacement, build_model_circuit_topology
from .transmission_line import TransmissionLineModel, TransmissionLineStubModel
from .microstrip import (
    MicrostripDesignRules,
    MicrostripLineModel,
    MicrostripStubModel,
)


LINE_TOPOLOGIES = (
    "through_line",
    "open_stub",
    "short_stub",
    "connector_line_open_stub_dut",
    "connector_line_short_stub_dut",
    "connector_open_stub_line_dut",
    "connector_short_stub_line_dut",
)


@dataclass(frozen=True)
class LineSearchConfig:
    characteristic_impedance_min_ohm: float = 20.0
    characteristic_impedance_max_ohm: float = 120.0
    electrical_length_min_deg: float = 1.0
    electrical_length_max_deg: float = 179.0
    attenuation_db: float = 0.0
    loss_frequency_exponent: float = 0.5
    topologies: tuple[str, ...] = LINE_TOPOLOGIES
    restarts: int = 10
    iterations: int = 24
    initial_step_fraction: float = 0.25
    minimum_step_fraction: float = 2e-4
    keep: int = 12
    seed: int = 1
    timeout_seconds: float | None = None
    maximum_evaluations: int = 10000
    microstrip_rules: MicrostripDesignRules | None = None
    fixed_dut_side: tuple[ModelPlacement, ...] = ()
    fixed_connector_side: tuple[ModelPlacement, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.characteristic_impedance_min_ohm < self.characteristic_impedance_max_ohm:
            raise ValueError("characteristic-impedance bounds must be positive and increasing")
        if not 0 < self.electrical_length_min_deg < self.electrical_length_max_deg < 180:
            raise ValueError("electrical-length bounds must lie strictly between 0 and 180 degrees")
        if self.attenuation_db < 0 or self.loss_frequency_exponent < 0:
            raise ValueError("line loss parameters must be non-negative")
        if not self.topologies or any(item not in LINE_TOPOLOGIES for item in self.topologies):
            raise ValueError("line search contains an unsupported topology")
        if self.restarts <= 0 or self.iterations <= 0 or self.keep <= 0:
            raise ValueError("line search counts must be positive")
        if not 0 < self.initial_step_fraction <= 1 or self.minimum_step_fraction <= 0:
            raise ValueError("line search step fractions must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.maximum_evaluations <= 0:
            raise ValueError("maximum_evaluations must be positive")
        if self.microstrip_rules is not None and self.attenuation_db != 0:
            raise ValueError("parametric attenuation_db cannot be combined with physical microstrip loss")
        for placement in (*self.fixed_dut_side, *self.fixed_connector_side):
            if placement.connection != "series" or not isinstance(placement.model, S2PModel):
                raise ValueError("fixed layout blocks must be series S2P placements")


@dataclass
class LineSynthesisCandidate:
    topology: str
    characteristic_impedance_ohm: float
    line_length_deg: float
    stub_length_deg: float | None
    stub_termination: str | None
    score_db: float
    placements: tuple[ModelPlacement, ...] = field(repr=False)
    metrics: dict = field(default_factory=dict, repr=False)

    def components(self) -> list[dict]:
        result = []
        for position, placement in enumerate(self.placements):
            model = placement.model
            if isinstance(model, S2PModel):
                result.append({
                    "position": position,
                    "connection_type": placement.connection,
                    "port": placement.port,
                    "comp_type": "layout_s2p",
                    "type": "layout_s2p",
                    "part": model.name,
                    "part_number": model.name,
                    "value": "Imported measured/EM S2P layout block",
                    "physical_model": "measured_s2p_layout",
                    "reference_impedance_ohm": model.z0,
                    "frequency_start_hz": float(np.min(model.frequencies_hz)),
                    "frequency_stop_hz": float(np.max(model.frequencies_hz)),
                    "frequency_points": len(model.frequencies_hz),
                })
                continue
            is_stub = isinstance(model, (TransmissionLineStubModel, MicrostripStubModel))
            line = model.line if is_stub else model
            is_microstrip = isinstance(line, MicrostripLineModel)
            if is_microstrip:
                reference_frequency = float(line.design_reference_frequency_hz)
                properties = line.properties_at(reference_frequency)
                characteristic_impedance = properties.characteristic_impedance_ohm
                electrical_length = float(line.design_electrical_length_deg)
                attenuation_db = properties.attenuation_db_per_m * line.length_m
            else:
                reference_frequency = line.reference_frequency_hz
                characteristic_impedance = line.characteristic_impedance_ohm
                electrical_length = line.electrical_length_deg
                attenuation_db = line.attenuation_db
            result.append({
                "position": position,
                "connection_type": placement.connection,
                "port": placement.port,
                "comp_type": (
                    f"{model.termination}_stub"
                    if is_stub
                    else "transmission_line"
                ),
                "characteristic_impedance_ohm": characteristic_impedance,
                "electrical_length_deg": electrical_length,
                "reference_frequency_hz": reference_frequency,
                "attenuation_db": attenuation_db,
                "loss_frequency_exponent": None if is_microstrip else line.loss_frequency_exponent,
                "value": (
                    f"{line.width_m * 1e3:.4f} mm x {line.length_m * 1e3:.4f} mm; "
                    f"Z0 {characteristic_impedance:.3f} ohm; {electrical_length:.3f} deg"
                    if is_microstrip else
                    f"Z0 {characteristic_impedance:.3f} ohm; {electrical_length:.3f} deg"
                ),
                **({
                    "physical_model": "microstrip_hammerstad_kirschning_wheeler",
                    "width_m": line.width_m,
                    "length_m": line.length_m,
                    "effective_permittivity": properties.effective_permittivity,
                    "conductor_loss_db": _microstrip_loss_db(line, reference_frequency, "conductor"),
                    "dielectric_loss_db": _microstrip_loss_db(line, reference_frequency, "dielectric"),
                    "substrate": {
                        "name": line.substrate.name,
                        "relative_permittivity": line.substrate.relative_permittivity,
                        "height_m": line.substrate.height_m,
                        "loss_tangent": line.substrate.loss_tangent,
                        "copper_thickness_m": line.substrate.copper_thickness_m,
                        "copper_roughness_rms_m": line.substrate.copper_roughness_rms_m,
                    },
                    "model_warnings": list(properties.warnings),
                    "manufacturing_tolerances_pct": {
                        "trace_width": 100.0 * line.width_tolerance,
                        "physical_length": 100.0 * line.length_tolerance,
                        "substrate_height": 100.0 * line.substrate_height_tolerance,
                        "relative_permittivity": 100.0 * line.relative_permittivity_tolerance,
                    },
                } if is_microstrip else {"physical_model": "parameterized_uniform_line"}),
            })
        return result


def _microstrip_loss_db(line: MicrostripLineModel, frequency_hz: float, kind: str) -> float:
    properties = line.properties_at(frequency_hz)
    attenuation = (
        properties.conductor_attenuation_np_per_m
        if kind == "conductor" else properties.dielectric_attenuation_np_per_m
    )
    return float(20.0 / np.log(10.0) * attenuation * line.length_m)


def _stub(line, termination: str):
    if isinstance(line, MicrostripLineModel):
        return MicrostripStubModel(line, termination)
    return TransmissionLineStubModel(line, termination)


@dataclass
class LineSynthesisResult:
    candidates: list[LineSynthesisCandidate]
    evaluations: int
    elapsed_seconds: float
    stopped_reason: str

    @property
    def best(self) -> LineSynthesisCandidate:
        if not self.candidates:
            raise ValueError("line synthesis produced no valid candidates")
        return self.candidates[0]


class TransmissionLineOptimizer:
    """Multi-start pattern search over line impedance and electrical lengths."""

    def __init__(
        self,
        problem: Problem,
        port: int,
        reference_frequency_hz: float,
        objective: Objective | None = None,
        config: LineSearchConfig | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        if not 0 <= port < problem.s_parameters.shape[1]:
            raise ValueError("line synthesis port is outside the DUT")
        if reference_frequency_hz <= 0:
            raise ValueError("reference_frequency_hz must be positive")
        self.problem = problem
        self.port = port
        self.reference_frequency_hz = float(reference_frequency_hz)
        self.objective = objective or Objective()
        self.config = config or LineSearchConfig()
        self.cancel_check = cancel_check
        self.progress_callback = progress_callback
        self.evaluations = 0
        self._started = 0.0
        self._stopped_reason = "completed"

    def _check_stop(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise OptimizationCancelled("transmission-line synthesis cancelled")
        if self.evaluations >= self.config.maximum_evaluations:
            self._stopped_reason = "evaluation_limit"
            raise StopIteration
        if (
            self.config.timeout_seconds is not None
            and time.monotonic() - self._started >= self.config.timeout_seconds
        ):
            self._stopped_reason = "timeout"
            raise StopIteration

    @staticmethod
    def _parameter_count(topology: str) -> int:
        return 3 if "line_" in topology and "_stub_" in topology else 2

    def _decode(self, topology: str, point: np.ndarray) -> tuple[float, float, float | None]:
        z0 = self.config.characteristic_impedance_min_ohm + point[0] * (
            self.config.characteristic_impedance_max_ohm
            - self.config.characteristic_impedance_min_ohm
        )
        length_span = self.config.electrical_length_max_deg - self.config.electrical_length_min_deg
        first_length = self.config.electrical_length_min_deg + point[1] * length_span
        second_length = (
            self.config.electrical_length_min_deg + point[2] * length_span
            if len(point) == 3 else None
        )
        return float(z0), float(first_length), None if second_length is None else float(second_length)

    def _placements(self, topology: str, point: np.ndarray) -> tuple[ModelPlacement, ...]:
        z0, first_length, second_length = self._decode(topology, point)

        def line(name: str, length: float):
            if self.config.microstrip_rules is not None:
                rules = self.config.microstrip_rules
                return MicrostripLineModel.from_electrical_design(
                    name, rules.substrate, z0, length, self.reference_frequency_hz,
                    rules.minimum_width_m, rules.maximum_width_m,
                    rules.width_tolerance, rules.length_tolerance,
                    rules.substrate_height_tolerance,
                    rules.relative_permittivity_tolerance,
                )
            return TransmissionLineModel(
                name, z0, length, self.reference_frequency_hz,
                self.config.attenuation_db,
                self.config.loss_frequency_exponent,
            )

        if topology == "through_line":
            generated = (ModelPlacement("series", self.port, line("TL", first_length)),)
        elif topology in {"open_stub", "short_stub"}:
            termination = topology.removesuffix("_stub")
            generated = (ModelPlacement(
                "shunt", self.port,
                _stub(line("STUB", first_length), termination),
            ),)
        else:
            termination = "open" if "open" in topology else "short"
            through = ModelPlacement("series", self.port, line("TL", first_length))
            stub = ModelPlacement(
                "shunt", self.port,
                _stub(line("STUB", float(second_length)), termination),
            )
            # Placements are DUT-outward. Names describe connector-to-DUT order.
            generated = (stub, through) if topology.startswith("connector_line_") else (through, stub)
        return self.config.fixed_dut_side + generated + self.config.fixed_connector_side

    def _evaluate(self, topology: str, point: np.ndarray) -> LineSynthesisCandidate | None:
        self._check_stop()
        self.evaluations += 1
        try:
            placements = self._placements(topology, point)
            physical_topology = build_model_circuit_topology(
                self.problem.s_parameters.shape[1], placements
            )
            sweep = evaluate_physical_problem(self.problem, physical_topology)
            scoring_elements = [
                Element(placement.connection, "L", placement.port, 1.0, placement.model.name)
                for placement in placements
            ]
            scored = score_sweep(
                self.problem, Candidate(scoring_elements), self.objective,
                sweep.s_parameters, sweep.total_efficiency,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None
        z0, line_length, stub_length = self._decode(topology, point)
        metrics = dict(scored.metrics)
        metrics.update({
            "s_parameters": sweep.s_parameters,
            "component_loss": sweep.component_loss,
            "dut_absorbed_power": sweep.dut_absorbed_power,
            "power_balance_error": sweep.power_balance_error,
            "maximum_power_balance_error": float(np.max(np.abs(sweep.power_balance_error))),
        })
        return LineSynthesisCandidate(
            topology, z0, line_length, stub_length,
            "open" if "open" in topology else "short" if "short" in topology else None,
            scored.score_db, placements, metrics,
        )

    @staticmethod
    def _better(candidate, current) -> bool:
        return candidate is not None and (current is None or candidate.score_db > current.score_db)

    def optimize(self) -> LineSynthesisResult:
        self._started = time.monotonic()
        rng = np.random.default_rng(self.config.seed)
        results: list[LineSynthesisCandidate] = []
        stop = False
        for topology_index, topology in enumerate(self.config.topologies):
            dimensions = self._parameter_count(topology)
            seeds = [np.full(dimensions, 0.5)]
            seeds.extend(rng.random(dimensions) for _ in range(self.config.restarts - 1))
            for restart_index, seed in enumerate(seeds):
                try:
                    point = np.asarray(seed, dtype=float)
                    best = self._evaluate(topology, point)
                    step = self.config.initial_step_fraction
                    for _ in range(self.config.iterations):
                        self._check_stop()
                        improved = False
                        for position in range(dimensions):
                            for direction in (-1.0, 1.0):
                                trial_point = point.copy()
                                trial_point[position] = np.clip(
                                    trial_point[position] + direction * step, 0.0, 1.0
                                )
                                trial = self._evaluate(topology, trial_point)
                                if self._better(trial, best):
                                    point, best, improved = trial_point, trial, True
                        if not improved:
                            step *= 0.5
                            if step < self.config.minimum_step_fraction:
                                break
                    if best is not None:
                        results.append(best)
                    if self.progress_callback is not None:
                        self.progress_callback({
                            "stage": "transmission_line_synthesis",
                            "topology": topology,
                            "topology_index": topology_index + 1,
                            "topologies_total": len(self.config.topologies),
                            "restart": restart_index + 1,
                            "restarts_total": len(seeds),
                            "evaluations": self.evaluations,
                            "best_score_db": max((item.score_db for item in results), default=None),
                        })
                except StopIteration:
                    stop = True
                    break
            if stop:
                break
        unique = {}
        for candidate in results:
            signature = (
                candidate.topology,
                round(candidate.characteristic_impedance_ohm, 6),
                round(candidate.line_length_deg, 6),
                None if candidate.stub_length_deg is None else round(candidate.stub_length_deg, 6),
            )
            if signature not in unique or candidate.score_db > unique[signature].score_db:
                unique[signature] = candidate
        ordered = sorted(unique.values(), key=lambda item: item.score_db, reverse=True)
        return LineSynthesisResult(
            ordered[: self.config.keep], self.evaluations,
            time.monotonic() - self._started, self._stopped_reason,
        )
