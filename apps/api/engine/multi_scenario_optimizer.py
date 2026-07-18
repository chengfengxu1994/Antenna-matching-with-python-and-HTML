"""Shared-network optimization across multiple measured antenna scenarios.

Every candidate uses the same topology and physical component part numbers for
all DUT files.  This is intentionally different from multi-port optimization:
the scenario axis represents grip, enclosure, head/hand, or other measurements
of the same feed, rather than ports that may each receive a separate network.
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence
import math
import time

import numpy as np

from .component_lib import ComponentInfo
from .efficiency_data import EfficiencyData
from .network import _embed_series_on_port, _embed_shunt_to_ground
from .topology import ConnectionType, Topology
from .touchstone import TouchstoneData

from rfmatch_core import (
    Band,
    Candidate,
    Element,
    LumpedModel,
    ModelPlacement,
    MultiScenarioProblem,
    MultiScenarioMatchingOptimizer,
    OptimizationCancelled,
    Objective,
    Problem,
    S2PModel,
    SearchConfig,
    ScenarioProblem,
    build_model_circuit_topology,
    evaluate_physical_multi_scenario,
)


@dataclass
class Scenario:
    filename: str
    dut: TouchstoneData
    weight: float = 1.0
    efficiency: Optional[EfficiencyData] = None
    efficiency_kind: str = "radiation"  # radiation | total


def ideal_component_s(comp_type: str, value: float, freq_hz: float, z0: float = 50.0) -> np.ndarray:
    """Lossless series two-port model used by the manual evaluator."""
    omega = 2.0 * np.pi * freq_hz
    if comp_type == "inductor":
        impedance = 1j * omega * value * 1e-9
    elif comp_type == "capacitor":
        impedance = 1.0 / (1j * omega * max(value, 1e-15) * 1e-12)
    else:
        raise ValueError("component type must be inductor or capacitor")
    denom = 2.0 * z0 + impedance
    return np.array(
        [[impedance / denom, 2.0 * z0 / denom],
         [2.0 * z0 / denom, impedance / denom]], dtype=complex,
    )


class MultiScenarioOptimizer:
    def __init__(
        self,
        scenarios: Sequence[Scenario],
        library,
        bands_mhz: Sequence[Sequence[float]],
        input_port: int = 0,
        num_band_points: int = 7,
        objective: str = "balanced",
        beam_width: int = 20,
        timeout_seconds: float = 120.0,
        max_candidates_per_position: int = 24,
    ):
        if len(scenarios) < 2:
            raise ValueError("multi-scenario optimization requires at least two SNP files")
        if not bands_mhz:
            raise ValueError("at least one frequency band is required")
        self.scenarios = list(scenarios)
        self.library = library
        self.bands_mhz = [list(b) for b in bands_mhz]
        self.input_port = input_port
        self.num_band_points = max(2, int(num_band_points))
        self.objective = objective
        self.beam_width = max(1, int(beam_width))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._progress_callback = None
        self._cancel_check = None
        self.max_candidates_per_position = max(4, int(max_candidates_per_position))
        self.physical_evaluations = 0
        self.model_builds = 0
        self.search_truncated = False
        self._model_cache: Dict[str, S2PModel] = {}
        self._candidate_cache: Dict[str, List[ComponentInfo]] = {}
        self._candidate_universe: Dict[str, List[ComponentInfo]] = {}
        self._candidate_stats: Dict[str, dict] = {}
        self.topologies_requested = 0
        self.topologies_screened = 0
        self.topologies_started = 0
        self.topologies_completed = 0
        self.screen_evaluations = 0
        self.local_refinement_evaluations = 0
        self.ideal_ranking_evaluations = 0
        self.ideal_ranked_topologies = 0
        self.frequencies_hz = np.unique(np.concatenate([
            np.linspace(float(b[0]) * 1e6, float(b[1]) * 1e6, self.num_band_points)
            for b in self.bands_mhz
        ]))

        for scenario in self.scenarios:
            if self.input_port >= scenario.dut.num_ports:
                raise ValueError(
                    f"input port {self.input_port + 1} does not exist in {scenario.filename}"
                )

        aliases = {
            "average_efficiency": "average",
            "worst": "worst_case",
        }
        mode = aliases.get(self.objective, self.objective)
        if mode not in {"average", "worst_case", "balanced"}:
            raise ValueError(f"unknown multi-scenario objective: {self.objective}")
        blend = {"average": 1.0, "worst_case": 0.0, "balanced": 0.5}[mode]
        scenario_problems = []
        for scenario in self.scenarios:
            matrices = np.asarray([
                scenario.dut.get_s_matrix_interpolated(float(frequency))
                for frequency in self.frequencies_hz
            ])
            radiation = np.asarray([
                self._radiation_efficiency(
                    scenario,
                    float(frequency),
                    matrices[index, self.input_port, self.input_port],
                )
                for index, frequency in enumerate(self.frequencies_hz)
            ])
            core_problem = Problem(
                self.frequencies_hz,
                matrices,
                {self.input_port: [
                    Band(float(start) * 1e6, float(stop) * 1e6)
                    for start, stop in self.bands_mhz
                ]},
                float(scenario.dut.reference_resistance),
                {self.input_port: radiation},
            )
            scenario_problems.append(ScenarioProblem(
                scenario.filename, core_problem, max(float(scenario.weight), 0.0)
            ))
        self.core_problem = MultiScenarioProblem(tuple(scenario_problems), blend)
        self.core_objective = Objective(
            within_band_average_weight=blend,
            across_band_average_weight=blend,
            port_average_weight=blend,
        )

    def _component_s(self, component: ComponentInfo, freq_hz: float) -> np.ndarray:
        matrix = component.get_s_matrix_at_freq(freq_hz)
        if matrix is None:
            return ideal_component_s(
                component.component_type, component.nominal_value, freq_hz
            )
        return matrix

    def _apply(self, matrix: np.ndarray, connection: str, component_s: np.ndarray) -> np.ndarray:
        if connection == ConnectionType.SERIES.value:
            return _embed_series_on_port(matrix, component_s, self.input_port)
        if connection == ConnectionType.SHUNT.value:
            return _embed_shunt_to_ground(matrix, component_s, self.input_port)
        raise ValueError(f"unsupported shared-network connection: {connection}")

    def _radiation_efficiency(self, scenario: Scenario, freq_hz: float, raw_gamma: complex) -> float:
        if scenario.efficiency is None:
            return 1.0
        measured = scenario.efficiency.get_efficiency_at(freq_hz)
        if scenario.efficiency_kind == "total":
            raw_mismatch = max(1.0 - abs(raw_gamma) ** 2, 1e-6)
            return float(np.clip(measured / raw_mismatch, 0.0, 1.0))
        return float(np.clip(measured, 0.0, 1.0))

    def evaluate(self, components: Sequence[dict], topology_name: str = "Manual") -> dict:
        """Evaluate physical or ideal parts with the rfmatch-core power solver."""
        self.physical_evaluations += 1
        placements = []
        ideal_elements = []
        for index, spec in enumerate(components):
            component_type = spec["component_type"]
            kind = "L" if component_type == "inductor" else "C"
            component = spec.get("component")
            if component is None:
                value = float(spec["value"])
                value_si = value * (1e-9 if kind == "L" else 1e-12)
                model = LumpedModel(f"ideal-{kind}-{index + 1}", kind, value_si)
                name = model.name
            else:
                name = component.part_number
                cache_key = f"{component.component_type}:{name}"
                model = self._model_cache.get(cache_key)
                if model is None:
                    matrices = np.asarray([
                        self._component_s(component, float(frequency))
                        for frequency in self.frequencies_hz
                    ])
                    model = S2PModel(name, self.frequencies_hz, matrices)
                    self._model_cache[cache_key] = model
                    self.model_builds += 1
                value_si = float(component.nominal_value) * (1e-9 if kind == "L" else 1e-12)
            connection = spec["connection_type"]
            placements.append(ModelPlacement(connection, self.input_port, model))
            ideal_elements.append(Element(connection, kind, self.input_port, value_si, name))

        port_count = self.core_problem.scenarios[0].problem.s_parameters.shape[1]
        topology = build_model_circuit_topology(port_count, placements)
        scored = evaluate_physical_multi_scenario(
            self.core_problem, topology, Candidate(ideal_elements), self.core_objective
        )

        scenario_results = []
        mismatch_averages = []
        for source, core_item in zip(self.scenarios, scored.metrics["scenarios"]):
            metrics = core_item["metrics"]
            matched = metrics["s_parameters"]
            gamma = matched[:, self.input_port, self.input_port]
            mismatch = np.clip(1.0 - np.abs(gamma) ** 2, 0.0, 1.0)
            total_efficiency = metrics["total_efficiency"][:, self.input_port]
            component_loss = metrics["component_loss"][:, self.input_port]
            radiation = self.core_problem.scenarios[len(scenario_results)].problem.radiation_efficiency[self.input_port]
            points = []
            for index, frequency in enumerate(self.frequencies_hz):
                points.append({
                    "frequency_hz": float(frequency),
                    "s11_real": float(gamma[index].real),
                    "s11_imag": float(gamma[index].imag),
                    "s11_db": float(20.0 * np.log10(max(abs(gamma[index]), 1e-15))),
                    "return_loss_db": float(metrics["return_loss_db"][index, self.input_port]),
                    "mismatch_efficiency": float(mismatch[index]),
                    "radiation_efficiency": float(radiation[index]),
                    "component_loss": float(component_loss[index]),
                    "total_efficiency": float(total_efficiency[index]),
                })
            mismatch_averages.append((float(np.mean(mismatch)), max(source.weight, 0.0)))
            scenario_results.append({
                "filename": source.filename,
                "weight": source.weight,
                "efficiency_kind": source.efficiency_kind if source.efficiency else None,
                "score_db": float(core_item["score_db"]),
                "avg_total_efficiency": float(np.mean(total_efficiency)),
                "min_total_efficiency": float(np.min(total_efficiency)),
                "avg_mismatch_efficiency": float(np.mean(mismatch)),
                "min_return_loss_db": float(np.min(metrics["return_loss_db"][:, self.input_port])),
                "points": points,
            })

        mismatch_weights = [weight for _, weight in mismatch_averages]
        if not any(weight > 0 for weight in mismatch_weights):
            mismatch_weights = [1.0] * len(mismatch_weights)
        return {
            "topology": topology_name,
            "score": float(10.0 ** (scored.score_db / 10.0)),
            "score_db": float(scored.score_db),
            "weighted_average_score_db": float(scored.metrics["weighted_average_score_db"]),
            "worst_scenario_score_db": float(scored.metrics["worst_scenario_score_db"]),
            "avg_total_efficiency": float(scored.metrics["average_total_efficiency"]),
            "min_total_efficiency": float(scored.metrics["minimum_total_efficiency"]),
            "avg_mismatch_efficiency": float(np.average(
                [value for value, _ in mismatch_averages], weights=mismatch_weights
            )),
            "min_return_loss_db": float(scored.metrics["minimum_return_loss_db"]),
            "maximum_power_balance_error": float(scored.metrics["maximum_power_balance_error"]),
            "scenarios": scenario_results,
        }

    def _candidates(self, component_type: str) -> List[ComponentInfo]:
        cached = self._candidate_cache.get(component_type)
        if cached is not None:
            return cached
        source = self.library.inductors if component_type == "inductor" else self.library.capacitors
        # Keep one physical part per value, reject parts that are effectively an
        # open/short in the requested bands, then sample uniformly in log(value).
        # Linear percentiles are badly biased by libraries containing 10 nF MLCCs
        # alongside sub-pF RF capacitors.
        by_value = {}
        center_hz = float(np.sqrt(self.frequencies_hz[0] * self.frequencies_hz[-1]))
        omega = 2.0 * np.pi * center_hz
        for component in source:
            value = float(getattr(component, 'nominal_value', 0.0) or 0.0)
            if value <= 0 or value in by_value:
                continue
            reactance = omega * value * 1e-9 if component_type == 'inductor' else 1.0 / (omega * value * 1e-12)
            # Preserve near-through and near-open physical parts: when an exact
            # component count is requested they can legitimately be the robust
            # solution.  Only discard numerically degenerate extremes.
            if 0.001 <= reactance <= 10000.0:
                by_value[value] = component
        ordered = [by_value[value] for value in sorted(by_value)]
        self._candidate_universe[component_type] = ordered
        if len(ordered) <= self.max_candidates_per_position:
            selected = ordered
        else:
            logs = np.log10([component.nominal_value for component in ordered])
            targets = np.linspace(logs[0], logs[-1], self.max_candidates_per_position)
            selected = []
            used = set()
            for target in targets:
                index = min((i for i in range(len(ordered)) if i not in used), key=lambda i: abs(logs[i] - target))
                used.add(index)
                selected.append(ordered[index])
            selected = sorted(selected, key=lambda component: component.nominal_value)
        self._candidate_cache[component_type] = selected
        self._candidate_stats[component_type] = {
            "catalog_models": len(source),
            "viable_unique_values": len(ordered),
            "selected_values": len(selected),
            "selection_truncated": len(ordered) > len(selected),
        }
        return selected

    @staticmethod
    def _spec(element, component: ComponentInfo) -> dict:
        return {
            "position": element.position,
            "connection_type": element.connection_type.value,
            "component_type": component.component_type,
            "component": component,
        }

    @staticmethod
    def _attach_components(metrics: dict, specs: Sequence[dict]) -> dict:
        metrics["components"] = [{
            "position": spec["position"],
            "connection_type": spec["connection_type"],
            "component_type": spec["component_type"],
            "part_number": spec["component"].part_number,
            "nominal_value": spec["component"].nominal_value,
            "nominal_unit": spec["component"].nominal_unit,
        } for spec in specs]
        return metrics

    @staticmethod
    def _uniform_subset(items: Sequence[ComponentInfo], count: int) -> List[ComponentInfo]:
        if len(items) <= count:
            return list(items)
        indices = np.linspace(0, len(items) - 1, count).round().astype(int)
        return [items[int(index)] for index in indices]

    def _neighbors(self, component: ComponentInfo, radius: int = 2) -> List[ComponentInfo]:
        universe = self._candidate_universe.get(component.component_type)
        if universe is None:
            self._candidates(component.component_type)
            universe = self._candidate_universe[component.component_type]
        center = min(
            range(len(universe)),
            key=lambda index: abs(universe[index].nominal_value - component.nominal_value),
        )
        return universe[max(0, center - radius):min(len(universe), center + radius + 1)]

    def _screen_topologies(
        self, topologies: Sequence[Topology], started: float
    ) -> tuple[List[Topology], List[dict]]:
        """Give every topology a small physical search before deep allocation."""
        if len(topologies) <= 1:
            return list(topologies), []
        deadline = min(self.timeout_seconds * 0.30, 12.0)
        ranked = []
        candidates = []
        for topology_index, topology in enumerate(topologies):
            self._checkpoint("physical_screen", topology_index, len(topologies),
                             f"Screening topology {topology.name}")
            if time.monotonic() - started >= deadline:
                break
            pools = [
                self._uniform_subset(self._candidates(element.component_type), 5)
                for element in topology.elements
            ]
            best = None
            for components in product(*pools):
                self._checkpoint("physical_screen", topology_index, len(topologies),
                                 f"Screening topology {topology.name}")
                if time.monotonic() - started >= deadline:
                    break
                specs = [
                    self._spec(element, component)
                    for element, component in zip(topology.elements, components)
                ]
                before = self.physical_evaluations
                try:
                    metrics = self.evaluate(specs, topology.name)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    continue
                self.screen_evaluations += self.physical_evaluations - before
                if best is None or metrics["score"] > best[1]["score"]:
                    best = (specs, metrics)
            if best is not None:
                self.topologies_screened += 1
                ranked.append((best[1]["score"], topology))
                candidates.append(self._attach_components(best[1], best[0]))
        ranked_names = {topology.name for _, topology in ranked}
        ordered = [topology for _, topology in sorted(ranked, key=lambda item: item[0], reverse=True)]
        ordered.extend(topology for topology in topologies if topology.name not in ranked_names)
        return ordered, candidates

    def _checkpoint(self, stage: str, current: int, total: int, message: str) -> None:
        if self._cancel_check is not None and self._cancel_check():
            raise OptimizationCancelled("multi-scenario optimization cancelled")
        if self._progress_callback is not None:
            self._progress_callback({
                "stage": stage,
                "current": int(current),
                "total": int(total),
                "message": message,
                "physical_evaluations": self.physical_evaluations,
            })

    @staticmethod
    def _topology_signature(topology: Topology) -> tuple[tuple[str, str], ...]:
        return tuple(
            (element.connection_type.value,
             "L" if element.component_type == "inductor" else "C")
            for element in topology.elements
        )

    def _ideal_rank_topologies(self, topologies: Sequence[Topology]) -> List[Topology]:
        """Rank topology allocation using cheap continuous lossless synthesis."""
        if len(topologies) <= 1:
            return list(topologies)
        patterns = [
            [(connection, kind, self.input_port) for connection, kind in self._topology_signature(topology)]
            for topology in topologies
        ]
        result = MultiScenarioMatchingOptimizer(
            self.core_problem,
            self.core_objective,
            SearchConfig(restarts=3, iterations=12, keep=max(32, len(topologies) * 2), seed=7),
        ).optimize(patterns)
        self.ideal_ranking_evaluations = result.evaluations
        scores = {}
        for candidate in result.candidates:
            signature = tuple((element.connection, element.kind) for element in candidate.elements)
            scores[signature] = max(scores.get(signature, -math.inf), candidate.score_db)
        self.ideal_ranked_topologies = len(scores)
        return sorted(
            topologies,
            key=lambda topology: scores.get(self._topology_signature(topology), -math.inf),
            reverse=True,
        )

    def optimize(
        self, topologies: Sequence[Topology], result_limit: int = 20,
        progress_callback=None, cancel_check=None,
    ) -> List[dict]:
        started = time.monotonic()
        self._progress_callback = progress_callback
        self._cancel_check = cancel_check
        self.topologies_requested = len(topologies)
        self._checkpoint("ideal_screen", 0, len(topologies), "Ranking ideal topology families")
        ideal_order = self._ideal_rank_topologies(topologies)
        self._checkpoint("physical_screen", 0, len(topologies), "Screening physical topologies")
        screen_order, completed = self._screen_topologies(topologies, started)
        screen_index = {topology.name: index for index, topology in enumerate(screen_order)}
        ideal_index = {topology.name: index for index, topology in enumerate(ideal_order)}
        ordered_topologies = sorted(
            topologies,
            key=lambda topology: (
                ideal_index.get(topology.name, len(topologies)),
                screen_index.get(topology.name, len(topologies)),
            ),
        )
        for topology_index, topology in enumerate(ordered_topologies):
            self._checkpoint("physical_refine", topology_index, len(ordered_topologies),
                             f"Optimizing topology {topology.name}")
            if time.monotonic() - started >= self.timeout_seconds:
                self.search_truncated = True
                break
            self.topologies_started += 1
            beam = [([], None)]
            for element in topology.elements:
                next_beam = []
                for choices, _ in beam:
                    for component in self._candidates(element.component_type):
                        self._checkpoint("physical_refine", topology_index, len(ordered_topologies),
                                         f"Optimizing topology {topology.name}")
                        specs = choices + [self._spec(element, component)]
                        try:
                            metrics = self.evaluate(specs, topology.name)
                        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                            continue
                        next_beam.append((specs, metrics))
                    if time.monotonic() - started >= self.timeout_seconds:
                        self.search_truncated = True
                        break
                next_beam.sort(key=lambda item: item[1]["score"], reverse=True)
                # For two-element networks, retain every first-position value so
                # the second element can compensate it; the sampled Cartesian
                # product remains small and avoids premature greedy pruning.
                if topology.num_components <= 2 and element.position == 0:
                    beam = next_beam
                else:
                    beam = next_beam[:self.beam_width]
                if not beam or time.monotonic() - started >= self.timeout_seconds:
                    if time.monotonic() - started >= self.timeout_seconds:
                        self.search_truncated = True
                    break

            # Refine around the best coarse values in the complete unique-value
            # family. This recovers standard values omitted by bounded log sampling.
            refined = []
            seen_refinement = set()
            if beam and all(len(specs) == topology.num_components for specs, _ in beam):
                for seed_specs, _ in beam[:min(5, len(beam))]:
                    pools = [self._neighbors(spec["component"]) for spec in seed_specs]
                    for components in product(*pools):
                        self._checkpoint("physical_refine", topology_index, len(ordered_topologies),
                                         f"Refining topology {topology.name}")
                        if time.monotonic() - started >= self.timeout_seconds:
                            self.search_truncated = True
                            break
                        signature = tuple(component.part_number for component in components)
                        if signature in seen_refinement:
                            continue
                        seen_refinement.add(signature)
                        specs = [
                            {**seed, "component": component}
                            for seed, component in zip(seed_specs, components)
                        ]
                        before = self.physical_evaluations
                        try:
                            metrics = self.evaluate(specs, topology.name)
                        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                            continue
                        self.local_refinement_evaluations += self.physical_evaluations - before
                        refined.append((specs, metrics))
                    if self.search_truncated:
                        break
            beam = sorted([*beam, *refined], key=lambda item: item[1]["score"], reverse=True)[:self.beam_width]

            for specs, metrics in beam:
                if len(specs) != topology.num_components:
                    continue
                completed.append(self._attach_components(metrics, specs))
            if beam and all(len(specs) == topology.num_components for specs, _ in beam):
                self.topologies_completed += 1

        self._checkpoint("physical_refine", len(ordered_topologies), len(ordered_topologies),
                         "Physical topology search complete")

        completed.sort(key=lambda item: item["score"], reverse=True)
        seen = set()
        unique = []
        for item in completed:
            signature = (item["topology"], tuple(c["part_number"] for c in item["components"]))
            if signature not in seen:
                seen.add(signature)
                unique.append(item)
        return unique[:result_limit]

    def verify_solutions(
        self, solutions: Sequence[dict], num_band_points: int,
        progress_callback=None, cancel_check=None,
    ) -> tuple[List[dict], dict]:
        """Rescore final physical networks on a denser independent grid."""
        points = max(2, int(num_band_points))
        if not solutions:
            return [], {"physical_evaluations": 0, "component_models_built": 0,
                        "band_points": points, "frequency_points": 0}
        verifier = MultiScenarioOptimizer(
            scenarios=self.scenarios,
            library=self.library,
            bands_mhz=self.bands_mhz,
            input_port=self.input_port,
            num_band_points=points,
            objective=self.objective,
            beam_width=1,
            timeout_seconds=max(self.timeout_seconds, 1.0),
            max_candidates_per_position=self.max_candidates_per_position,
        )
        by_name = {
            component.part_number.lower(): component
            for component in [*self.library.inductors, *self.library.capacitors]
        }
        verified = []
        for solution_index, solution in enumerate(solutions):
            if cancel_check is not None and cancel_check():
                raise OptimizationCancelled("multi-scenario verification cancelled")
            if progress_callback is not None:
                progress_callback({
                    "stage": "verification", "current": solution_index,
                    "total": len(solutions),
                    "message": f"Verifying solution {solution_index + 1}/{len(solutions)}",
                    "physical_evaluations": self.physical_evaluations,
                })
            specs = []
            for component in solution.get("components", []):
                physical = by_name.get(str(component["part_number"]).lower())
                if physical is None:
                    raise ValueError(f"verification component not found: {component['part_number']}")
                specs.append({
                    "position": component["position"],
                    "connection_type": component["connection_type"],
                    "component_type": component["component_type"],
                    "component": physical,
                })
            result = verifier.evaluate(specs, solution["topology"])
            result["search_estimate_score_db"] = solution["score_db"]
            verified.append(self._attach_components(result, specs))
        if progress_callback is not None:
            progress_callback({
                "stage": "verification", "current": len(solutions),
                "total": len(solutions), "message": "Independent verification complete",
                "physical_evaluations": self.physical_evaluations,
            })
        verified.sort(key=lambda item: item["score"], reverse=True)
        diagnostics = verifier.diagnostics()
        diagnostics["band_points"] = points
        diagnostics["frequency_points"] = len(verifier.frequencies_hz)
        return verified, diagnostics

    def diagnostics(self) -> dict:
        return {
            "physical_evaluations": self.physical_evaluations,
            "component_models_built": self.model_builds,
            "component_model_cache_entries": len(self._model_cache),
            "candidate_coverage": self._candidate_stats,
            "topologies_requested": self.topologies_requested,
            "topologies_screened": self.topologies_screened,
            "topologies_started": self.topologies_started,
            "topologies_completed": self.topologies_completed,
            "screen_evaluations": self.screen_evaluations,
            "local_refinement_evaluations": self.local_refinement_evaluations,
            "ideal_ranking_evaluations": self.ideal_ranking_evaluations,
            "ideal_ranked_topologies": self.ideal_ranked_topologies,
            "search_truncated": self.search_truncated,
        }


def find_component(library, part_number: str):
    needle = part_number.strip().lower()
    for component in list(library.inductors) + list(library.capacitors):
        if component.part_number.lower() == needle:
            return component
    return None
