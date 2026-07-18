from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from .models import Band, Candidate, IsolationTarget, Objective, Problem
from .optimizer import MatchingOptimizer, SearchConfig
from .components import component_sha256, load_coilcraft_0402cs_catalog, load_coilcraft_0402hp_catalog, load_component_model, load_murata_gjm15_catalog, load_murata_gqm18_catalog
from .physical_optimizer import PORT_TOPOLOGY_PATTERNS, MeasuredComponentOptimizer, MeasuredPlacement, MeasuredSearchConfig
from .multi_scenario import MultiScenarioMatchingOptimizer, MultiScenarioProblem, ScenarioProblem, SharedMeasuredComponentOptimizer, evaluate_multi_scenario
from .touchstone import load_touchstone
from .mdif import load_mdif
from .tunable import FrequencyConfiguration, TunableProblem, evaluate_tunable_physical, load_measured_placements
from .tunable_optimizer import TunableMeasuredComponentOptimizer
from .switch import InputModelPlacement, SeriesReactance, evaluate_loaded_switch_physical_power, evaluate_switched_matching
from .switch_optimizer import SwitchMeasuredComponentOptimizer, SwitchSearchConfig, SwitchTunableOptimizer, SwitchTunableProblem


CASES = {
    "quick-start": (Path("1 - START HERE/1a - QUICK START.pdf"), Path("1 - START HERE/measured_antenna.s1p"), [(2.50e9, 2.69e9)]),
    "optimization-settings": (Path("3 - Optimization settings/Optenni Lab Optimization Settings.pdf"), Path("3 - Optimization settings/measured_antenna.s1p"), [(1.7e9, 2.5e9)]),
    "radiation-efficiency": (Path("9 - Radiation efficiency/Radiation Efficiency Tutorial.pdf"), Path("9 - Radiation efficiency/Radiation_Efficiency_Tutorial.s1p"), [(880e6, 960e6), (1710e6, 2155e6), (2300e6, 2400e6)]),
}

MULTIPORT_CASES = {
    "multiantenna-different-bands": (
        Path("4 - Multiantenna system/4.1 Multiantenna system at different bands/3_antennas.s3p"),
        {0: [(2.50e9, 2.69e9)], 1: [(1.92e9, 2.17e9)], 2: [(1.215e9, 1.30e9)]},
    ),
}

MULTI_SCENARIO_CASES = {
    "multiple-impedance-configurations": (
        (
            Path("7 - Multiple impedance configurations/free_space.s1p"),
            Path("7 - Multiple impedance configurations/cover.s1p"),
            Path("7 - Multiple impedance configurations/cover_w_spacer.s1p"),
        ),
        ((2.4e9, 2.483e9), (2.5e9, 2.69e9)),
    ),
}

TUNABLE_CASES = {
    "variable-capacitor": (
        Path("10 - Tunable antennas/10.5 Impedance tuning using a variable capacitor/Variable_capacitor_Tutorial_antenna.s1p"),
        Path("10 - Tunable antennas/10.5 Impedance tuning using a variable capacitor/Variable_capacitor_tutorial.mdif"),
    ),
}

SWITCH_CASES = {
    "switch-tuning": (
        Path("10 - Tunable antennas/10.6 Impedance tuning using a switch/Switch_Tuner_Tutorial.s1p"),
        Path("10 - Tunable antennas/10.6 Impedance tuning using a switch/SP3T_ideal.mdif"),
        Path("10 - Tunable antennas/10.6 Impedance tuning using a switch/tutorial_SP2T.mdif"),
    ),
}


def _radiation_efficiency(root: Path, name: str, frequencies_hz: np.ndarray) -> dict[int, np.ndarray]:
    if name != "radiation-efficiency":
        return {}
    path = root / "9 - Radiation efficiency/radiation_efficiency.txt"
    rows = np.loadtxt(path, comments="%")
    return {0: np.interp(frequencies_hz, rows[:, 0] * 1e6, rows[:, 1])}


def run_case(root: Path, name: str) -> dict:
    _, data_path, bands = CASES[name]
    data = load_touchstone(root / data_path)
    mask = np.zeros(len(data.frequencies_hz), dtype=bool)
    for lo, hi in bands:
        mask |= (data.frequencies_hz >= lo) & (data.frequencies_hz <= hi)
    frequencies = data.frequencies_hz[mask]
    problem = Problem(frequencies, data.s_parameters[mask], {0: [Band(a, b) for a, b in bands]}, data.z0, _radiation_efficiency(root, name, frequencies))
    topologies = [[("shunt", "C", 0), ("series", "L", 0)], [("shunt", "L", 0), ("series", "C", 0)]]
    result = MatchingOptimizer(problem, Objective(), SearchConfig(restarts=8, iterations=20)).optimize(topologies)
    return {"case": name, "score_db": result.best.score_db, "evaluations": result.evaluations, "elements": [(e.connection, e.kind, e.value) for e in result.best.elements]}


def _resampled_indices(frequencies_hz: np.ndarray, bands_by_port: dict[int, list[tuple[float, float]]], points_per_band: int = 5) -> np.ndarray:
    indices: set[int] = set()
    for bands in bands_by_port.values():
        for start, stop in bands:
            targets = np.linspace(start, stop, points_per_band)
            indices.update(int(np.argmin(np.abs(frequencies_hz - target))) for target in targets)
    return np.asarray(sorted(indices), dtype=int)


def _directed_isolation_summary(transmission_db: np.ndarray) -> dict[str, dict[str, float]]:
    n_ports = transmission_db.shape[1]
    return {
        f"S{destination + 1}{source + 1}": {
            "worst_db": float(np.max(transmission_db[:, destination, source])),
            "average_db": float(np.mean(transmission_db[:, destination, source])),
        }
        for source in range(n_ports)
        for destination in range(n_ports)
        if source != destination
    }


def run_multiport_case(root: Path, name: str, isolation_targets: tuple[IsolationTarget, ...] = ()) -> dict:
    """Run the first deterministic ideal-LC baseline for an official multiport case."""
    data_path, bands = MULTIPORT_CASES[name]
    data = load_touchstone(root / data_path)
    sample = _resampled_indices(data.frequencies_hz, bands)
    problem = Problem(
        data.frequencies_hz[sample],
        data.s_parameters[sample],
        {port: [Band(start, stop) for start, stop in port_bands] for port, port_bands in bands.items()},
        data.z0,
        isolation_targets=isolation_targets,
    )
    per_port_patterns = [
        (("shunt", "C"), ("series", "L")),
        (("shunt", "L"), ("series", "C")),
    ]
    topologies = []
    for selected in itertools.product(per_port_patterns, repeat=len(bands)):
        topology = []
        for port, pattern in enumerate(selected):
            topology.extend((connection, kind, port) for connection, kind in pattern)
        topologies.append(topology)
    result = MatchingOptimizer(
        problem,
        Objective(port_average_weight=0.1),
        SearchConfig(restarts=4, iterations=12, keep=20, seed=1),
    ).optimize(topologies)
    best = result.best
    return {
        "case": name,
        "score_db": best.score_db,
        "evaluations": result.evaluations,
        "frequency_points": len(problem.frequencies_hz),
        "port_scores_db": [float(value) for value in best.metrics["port_scores_db"]],
        "directed_isolation_db": _directed_isolation_summary(best.metrics["transmission_db"]),
        "isolation_targets": best.metrics["isolation_targets"],
        "elements": [(element.connection, element.kind, element.port, element.value) for element in best.elements],
    }


def run_multiport_measured_case(
    root: Path,
    name: str,
    component_root: Path,
    config: MeasuredSearchConfig | None = None,
    isolation_targets: tuple[IsolationTarget, ...] = (),
) -> dict:
    """Synthesize a 0–2 element network per port using measured vendor S2P parts."""
    data_path, bands = MULTIPORT_CASES[name]
    data = load_touchstone(root / data_path)
    sample = _resampled_indices(data.frequencies_hz, bands)
    problem = Problem(
        data.frequencies_hz[sample],
        data.s_parameters[sample],
        {port: [Band(start, stop) for start, stop in port_bands] for port, port_bands in bands.items()},
        data.z0,
        isolation_targets=isolation_targets,
    )
    inductors = load_coilcraft_0402hp_catalog(component_root / "Inductors" / "Coilcraft Inductors 0402hp")
    capacitors = load_murata_gqm18_catalog(component_root / "Capacitors" / "Murata Capacitors gqm18")
    if not inductors or not capacitors:
        raise ValueError(f"no supported measured component catalogs found below {component_root}")
    result = MeasuredComponentOptimizer(
        problem,
        inductors,
        capacitors,
        Objective(port_average_weight=0.1),
        config,
    ).optimize()
    best = result.best
    return {
        "case": name,
        "mode": "measured-s2p",
        "score_db": best.score_db,
        "evaluations": result.ideal_evaluations + result.physical_evaluations,
        "ideal_evaluations": result.ideal_evaluations,
        "physical_evaluations": result.physical_evaluations,
        "component_models_loaded": result.loaded_component_models,
        "frequency_points": len(problem.frequencies_hz),
        "port_scores_db": [float(value) for value in best.metrics["port_scores_db"]],
        "directed_isolation_db": _directed_isolation_summary(best.metrics["transmission_db"]),
        "isolation_targets": best.metrics["isolation_targets"],
        "maximum_power_balance_error": best.metrics["maximum_power_balance_error"],
        "mean_component_loss": float(np.mean(best.metrics["component_loss"])),
        "elements": [
            {
                "connection": placement.connection,
                "kind": placement.component.kind,
                "port": placement.port,
                "name": placement.component.name,
                "family": placement.component.family,
                "value_si": placement.component.value,
                "model_sha256": component_sha256(placement.component),
            }
            for placement in best.placements
        ],
    }


def _shared_scenario_problem(root: Path, name: str) -> MultiScenarioProblem:
    paths, bands = MULTI_SCENARIO_CASES[name]
    scenarios = []
    for path in paths:
        data = load_touchstone(root / path)
        mask = np.zeros(len(data.frequencies_hz), dtype=bool)
        for start, stop in bands:
            mask |= (data.frequencies_hz >= start) & (data.frequencies_hz <= stop)
        if not np.any(mask):
            raise ValueError(f"no tutorial samples fall inside the configured bands for {path}")
        problem = Problem(
            data.frequencies_hz[mask],
            data.s_parameters[mask],
            {0: [Band(start, stop) for start, stop in bands]},
            data.z0,
        )
        scenarios.append(ScenarioProblem(path.stem, problem))
    return MultiScenarioProblem.from_mode(tuple(scenarios), "worst_case")


def _shared_scenario_summary(best) -> list[dict]:
    return [
        {
            "name": item["name"],
            "weight": float(item["weight"]),
            "score_db": float(item["score_db"]),
            "minimum_total_efficiency_db": float(10.0 * np.log10(max(item["minimum_total_efficiency"], 1e-15))),
            "average_total_efficiency_db": float(10.0 * np.log10(max(item["average_total_efficiency"], 1e-15))),
            "minimum_return_loss_db": float(item["minimum_return_loss_db"]),
        }
        for item in best.metrics["scenarios"]
    ]


def run_multi_scenario_case(root: Path, name: str) -> dict:
    """Run an ideal-LC shared-network baseline for an official impedance-state case."""
    problem = _shared_scenario_problem(root, name)
    patterns = [
        [(connection, kind, 0) for connection, kind in pattern]
        for pattern in PORT_TOPOLOGY_PATTERNS
        if pattern
    ]
    objective = Objective(
        within_band_average_weight=0.0,
        across_band_average_weight=0.0,
        port_average_weight=0.0,
    )
    optimizer = MultiScenarioMatchingOptimizer(
        problem,
        objective,
        SearchConfig(restarts=8, iterations=20, keep=30, seed=7),
    )
    result = optimizer.optimize(patterns)
    unmatched = evaluate_multi_scenario(problem, Candidate([]), objective)
    best = max([*result.candidates, unmatched], key=lambda item: item.score_db)
    return {
        "case": name,
        "mode": "shared-ideal-lc",
        "score_db": best.score_db,
        "evaluations": result.evaluations + 1,
        "frequency_points_by_scenario": [
            len(item.problem.frequencies_hz) for item in problem.scenarios
        ],
        "weighted_average_score_db": best.metrics["weighted_average_score_db"],
        "worst_scenario_score_db": best.metrics["worst_scenario_score_db"],
        "minimum_total_efficiency_db": float(10.0 * np.log10(max(best.metrics["minimum_total_efficiency"], 1e-15))),
        "average_total_efficiency_db": float(10.0 * np.log10(max(best.metrics["average_total_efficiency"], 1e-15))),
        "scenarios": _shared_scenario_summary(best),
        "elements": [(element.connection, element.kind, element.port, element.value) for element in best.elements],
    }


def run_multi_scenario_measured_case(
    root: Path,
    name: str,
    component_root: Path,
    config: MeasuredSearchConfig | None = None,
) -> dict:
    """Synthesize one shared network using the exact tutorial component families."""
    problem = _shared_scenario_problem(root, name)
    inductors = load_coilcraft_0402cs_catalog(
        component_root / "Inductors" / "Coilcraft Inductors 0402cs"
    )
    capacitors = [
        item for item in load_murata_gjm15_catalog(
            component_root / "Capacitors" / "Murata Capacitors gjm15",
            prefer_loosest_tolerance=True,
        )
        if 0.5e-12 <= item.value <= 100e-12
    ]
    if not inductors or not capacitors:
        raise ValueError(f"tutorial component catalogs are unavailable below {component_root}")
    objective = Objective(
        within_band_average_weight=0.0,
        across_band_average_weight=0.0,
        port_average_weight=0.0,
    )
    result = SharedMeasuredComponentOptimizer(
        problem, inductors, capacitors, objective, config
    ).optimize()
    best = result.best
    scenario_component_losses = [
        float(np.mean(item["metrics"]["component_loss"]))
        for item in best.metrics["scenarios"]
    ]
    return {
        "case": name,
        "mode": "shared-measured-s2p",
        "score_db": best.score_db,
        "evaluations": result.ideal_evaluations + result.physical_evaluations,
        "ideal_evaluations": result.ideal_evaluations,
        "physical_evaluations": result.physical_evaluations,
        "component_models_loaded": result.loaded_component_models,
        "frequency_points_by_scenario": [
            len(item.problem.frequencies_hz) for item in problem.scenarios
        ],
        "weighted_average_score_db": best.metrics["weighted_average_score_db"],
        "worst_scenario_score_db": best.metrics["worst_scenario_score_db"],
        "minimum_total_efficiency_db": float(10.0 * np.log10(max(best.metrics["minimum_total_efficiency"], 1e-15))),
        "average_total_efficiency_db": float(10.0 * np.log10(max(best.metrics["average_total_efficiency"], 1e-15))),
        "scenarios": _shared_scenario_summary(best),
        "maximum_power_balance_error": best.metrics["maximum_power_balance_error"],
        "mean_component_loss": float(np.mean(scenario_component_losses)),
        "elements": [
            {
                "connection": placement.connection,
                "kind": placement.component.kind,
                "port": placement.port,
                "name": placement.component.name,
                "family": placement.component.family,
                "value_si": placement.component.value,
                "tolerance": placement.component.tolerance,
                "model_sha256": component_sha256(placement.component),
            }
            for placement in best.placements
        ],
    }


def run_tunable_variable_capacitor_case(
    root: Path,
    name: str,
    component_root: Path,
) -> dict:
    """Replay the official variable-capacitor reference with measured fixed parts."""
    antenna_path, tuner_path = TUNABLE_CASES[name]
    data = load_touchstone(root / antenna_path)
    mask = (data.frequencies_hz >= 700e6) & (data.frequencies_hz <= 2170e6)
    base = Problem(
        data.frequencies_hz[mask],
        data.s_parameters[mask],
        {0: [Band(704e6, 2170e6)]},
        data.z0,
    )
    configurations = (
        FrequencyConfiguration("Set 1", {0: [Band(704e6, 746e6), Band(1920e6, 2170e6)]}),
        FrequencyConfiguration("Set 2", {0: [Band(791e6, 862e6), Band(1920e6, 2170e6)]}),
        FrequencyConfiguration("Set 3", {0: [Band(880e6, 960e6), Band(1920e6, 2170e6)]}),
    )
    problem = TunableProblem(base, configurations, configuration_average_weight=0.5)
    inductors = load_coilcraft_0402cs_catalog(
        component_root / "Inductors" / "Coilcraft Inductors 0402cs"
    )
    capacitors = load_murata_gjm15_catalog(
        component_root / "Capacitors" / "Murata Capacitors gjm15",
        unique_values=False,
    )
    inductor = next((item for item in inductors if abs(item.value - 15e-9) < 1e-18), None)
    capacitor = next(
        (
            item for item in capacitors
            if item.name.upper() == "GJM1555C1H2R8WB01"
        ),
        None,
    )
    if inductor is None or capacitor is None:
        raise ValueError("official 0402CS 15 nH / GJM15 2.8 pF parts are unavailable")
    measured = (
        # Ordered DUT outward: tuner is inserted first by the evaluator.
        MeasuredPlacement("series", 0, capacitor),
        MeasuredPlacement("series", 0, inductor),
    )
    fixed = load_measured_placements(measured)
    tuner = load_mdif(root / tuner_path)
    best = evaluate_tunable_physical(problem, fixed, tuner, Objective())
    return {
        "case": name,
        "mode": "tunable-mdif-measured-s2p",
        "score_db": best.score_db,
        "frequency_points": len(base.frequencies_hz),
        "state_by_configuration": best.state_by_configuration,
        "weighted_average_score_db": best.metrics["weighted_average_score_db"],
        "worst_configuration_score_db": best.metrics["worst_configuration_score_db"],
        "maximum_power_balance_error": best.metrics["maximum_power_balance_error"],
        "configurations": [
            {
                "name": item["name"],
                "state": item["state"],
                "score_db": item["score_db"],
                "band_minimum_efficiency_db": [
                    metric["minimum_efficiency_db"]
                    for metric in item["metrics"]["bands"].values()
                ],
                "maximum_power_balance_error": item["maximum_power_balance_error"],
            }
            for item in best.metrics["configurations"]
        ],
        "fixed_elements": [
            {
                "connection": placement.connection,
                "kind": placement.component.kind,
                "port": placement.port,
                "name": placement.component.name,
                "family": placement.component.family,
                "value_si": placement.component.value,
                "tolerance": placement.component.tolerance,
                "model_sha256": component_sha256(placement.component),
            }
            for placement in measured
        ],
        "tuner": {
            "name": tuner.name,
            "states": [state.label for state in tuner.states],
            "frequency_points_per_state": [len(state.frequencies_hz) for state in tuner.states],
        },
    }


def run_tunable_variable_capacitor_synthesis_case(
    root: Path,
    name: str,
    component_root: Path,
) -> dict:
    """Automatically synthesize the official variable-C shared fixed network."""
    antenna_path, tuner_path = TUNABLE_CASES[name]
    data = load_touchstone(root / antenna_path)
    mask = (data.frequencies_hz >= 700e6) & (data.frequencies_hz <= 2170e6)
    base = Problem(
        data.frequencies_hz[mask], data.s_parameters[mask],
        {0: [Band(704e6, 2170e6)]}, data.z0,
    )
    problem = TunableProblem(base, (
        FrequencyConfiguration("Set 1", {0: [Band(704e6, 746e6), Band(1920e6, 2170e6)]}),
        FrequencyConfiguration("Set 2", {0: [Band(791e6, 862e6), Band(1920e6, 2170e6)]}),
        FrequencyConfiguration("Set 3", {0: [Band(880e6, 960e6), Band(1920e6, 2170e6)]}),
    ), configuration_average_weight=0.5)
    inductors = load_coilcraft_0402cs_catalog(
        component_root / "Inductors" / "Coilcraft Inductors 0402cs"
    )
    capacitors = load_murata_gjm15_catalog(
        component_root / "Capacitors" / "Murata Capacitors gjm15"
    )
    search = TunableMeasuredComponentOptimizer(
        problem,
        load_mdif(root / tuner_path),
        inductors,
        capacitors,
        Objective(),
        MeasuredSearchConfig(
            ideal_restarts=4,
            ideal_iterations=12,
            ideal_keep=24,
            nearest_parts=6,
            result_keep=20,
            joint_refine_seeds=0,
            joint_refine_passes=0,
            seed=1,
        ),
        exact_seed_keep=16,
    ).optimize()
    best = search.best
    return {
        "case": name,
        "mode": "tunable-mdif-auto-synthesis",
        "score_db": best.score_db,
        "state_by_configuration": best.evaluation.state_by_configuration,
        "weighted_average_score_db": best.evaluation.metrics["weighted_average_score_db"],
        "worst_configuration_score_db": best.evaluation.metrics["worst_configuration_score_db"],
        "maximum_power_balance_error": best.evaluation.metrics["maximum_power_balance_error"],
        "ideal_evaluations": search.ideal_evaluations,
        "exact_physical_evaluations": search.exact_physical_evaluations,
        "tuner_state_precomputations": search.tuner_state_precomputations,
        "ideal_frequency_points": search.ideal_frequency_points,
        "component_models_loaded": search.loaded_component_models,
        "fixed_elements": [
            {
                "connection": placement.connection,
                "kind": placement.component.kind,
                "port": placement.port,
                "name": placement.component.name,
                "family": placement.component.family,
                "value_si": placement.component.value,
                "tolerance": placement.component.tolerance,
                "model_sha256": component_sha256(placement.component),
            }
            for placement in best.placements
        ],
    }


def run_switch_tutorial_case(root: Path, name: str) -> dict:
    """Replay the documented reference switch topologies from tutorial 10.6."""
    antenna_path, sp3t_path, sp2t_path = SWITCH_CASES[name]
    antenna = load_touchstone(root / antenna_path)
    sp3t = load_mdif(root / sp3t_path)
    sp2t = load_mdif(root / sp2t_path)
    configurations = (
        ("Set 1", ((704e6, 746e6), (1920e6, 2170e6)), "100", "all on"),
        ("Set 2", ((791e6, 862e6), (1920e6, 2170e6)), "010", "RFC-RF1"),
        ("Set 3", ((880e6, 960e6), (1920e6, 2170e6)), "001", "RFC-RF2"),
    )
    def variant(
        label: str,
        switch,
        branches: tuple[SeriesReactance, ...],
        input_series: tuple[SeriesReactance, ...],
        state_index: int,
        *,
        tutorial_page: int,
        design_role: str,
    ) -> dict:
        results = []
        for configuration, bands, sp3_state, sp2_state in configurations:
            state_label = (sp3_state, sp2_state)[state_index]
            mask = np.zeros(len(antenna.frequencies_hz), dtype=bool)
            for start, stop in bands:
                mask |= (antenna.frequencies_hz >= start) & (antenna.frequencies_hz <= stop)
            frequencies = antenna.frequencies_hz[mask]
            gamma = evaluate_switched_matching(
                frequencies,
                antenna.s_parameters[mask, 0, 0],
                switch.state(state_label),
                branches,
                input_series_reactances=input_series,
                z0=antenna.z0,
            )
            return_loss = -20.0 * np.log10(np.maximum(np.abs(gamma), 1e-15))
            centers = tuple((start + stop) / 2.0 for start, stop in bands)
            center_dut = np.interp(centers, antenna.frequencies_hz, antenna.s_parameters[:, 0, 0].real) + 1j * np.interp(
                centers, antenna.frequencies_hz, antenna.s_parameters[:, 0, 0].imag
            )
            center_gamma = evaluate_switched_matching(
                centers, center_dut, switch.state(state_label), branches,
                input_series_reactances=input_series, z0=antenna.z0,
            )
            results.append({
                "name": configuration,
                "state": state_label,
                "minimum_return_loss_db": float(np.min(return_loss)),
                "mean_return_loss_db": float(np.mean(return_loss)),
                "center_return_loss_db": [
                    float(value) for value in -20.0 * np.log10(np.maximum(np.abs(center_gamma), 1e-15))
                ],
            })
        return {
            "name": label,
            "tutorial_page": tutorial_page,
            "design_role": design_role,
            "input_series": [
                {"kind": item.kind, "value_si": item.value} for item in input_series
            ],
            "branches": [
                {"throw": index + 1, "kind": item.kind, "value_si": item.value}
                for index, item in enumerate(branches)
            ],
            "configurations": results,
        }

    return {
        "case": name,
        "mode": "switch-mdif-tutorial-replay",
        "reference_source": {
            "document": "Impedance tuning using a switch Tutorial.pdf",
            "verified_pages": [12, 13, 16],
        },
        "variants": [
            variant("SP3T best topology #1", sp3t, (
                SeriesReactance("L", 1e-9),
                SeriesReactance("C", 2.6e-12),
                SeriesReactance("C", 1.2e-12),
            ), (
                SeriesReactance("L", 15e-9),
                SeriesReactance("C", 2e-12),
            ), 0, tutorial_page=12, design_role="best_performance"),
            variant("SP3T simplified topology #4", sp3t, (
                SeriesReactance("C", 2.3e-12),
                SeriesReactance("C", 1.2e-12),
                SeriesReactance("C", 0.8e-12),
            ), (SeriesReactance("L", 13e-9),), 0,
                tutorial_page=13, design_role="simplified_near_best"),
            variant("SP2T optimized topology #5", sp2t, (
                SeriesReactance("C", 1.2e-12),
                SeriesReactance("C", 0.8e-12),
            ), (SeriesReactance("L", 13e-9),), 1,
                tutorial_page=16, design_role="reduced_bom_near_equivalent"),
        ],
    }


def run_switch_tutorial_synthesis_case(root: Path, name: str) -> dict:
    """Automatically select switch-branch L/C types, values, and states."""
    antenna_path, sp3t_path, sp2t_path = SWITCH_CASES[name]
    antenna = load_touchstone(root / antenna_path)
    configurations = (
        FrequencyConfiguration("Set 1", {0: (Band(704e6, 746e6), Band(1920e6, 2170e6))}),
        FrequencyConfiguration("Set 2", {0: (Band(791e6, 862e6), Band(1920e6, 2170e6))}),
        FrequencyConfiguration("Set 3", {0: (Band(880e6, 960e6), Band(1920e6, 2170e6))}),
    )
    active = np.zeros(len(antenna.frequencies_hz), dtype=bool)
    for configuration in configurations:
        for band in configuration.bands_by_port[0]:
            active |= band.mask(antenna.frequencies_hz)
    search_config = SwitchSearchConfig(restarts=2, iterations=10, keep=8, seed=1)

    def synthesize(label: str, switch_path: Path, state_options: dict[str, tuple[str, ...]]) -> dict:
        problem = SwitchTunableProblem(
            antenna.frequencies_hz[active],
            antenna.s_parameters[active, 0, 0],
            configurations,
            antenna.z0,
            state_options_by_configuration=state_options,
        )
        result = SwitchTunableOptimizer(
            problem, load_mdif(root / switch_path), config=search_config
        ).optimize(input_topologies=[(("series", "L"),)])
        best = result.best
        return {
            "name": label,
            "score_db": best.score_db,
            "state_by_configuration": best.state_by_configuration,
            "branch_reactances": [
                {"kind": item.kind, "value_si": item.value}
                for item in best.branch_reactances
            ],
            "input_reactances": [
                {"connection": item.connection, "kind": item.kind, "value_si": item.value}
                for item in best.input_reactances
            ],
            "configuration_scores_db": {
                item["name"]: item["score_db"] for item in best.metrics["configurations"]
            },
            "maximum_power_balance_error": best.metrics["maximum_power_balance_error"],
            "maximum_switch_model_gain": best.metrics["maximum_switch_model_gain"],
            "evaluations": result.evaluations,
            "state_precomputations": result.state_precomputations,
        }

    return {
        "case": name,
        "mode": "switch-mdif-auto-synthesis",
        "active_frequency_points": int(np.sum(active)),
        "search": {
            "branch_types": "all L/C combinations",
            "input_topologies": ["series L"],
            "restarts": search_config.restarts,
            "iterations": search_config.iterations,
        },
        "variants": [
            synthesize("SP3T fixed tutorial states", sp3t_path, {
                "Set 1": ("100",), "Set 2": ("010",), "Set 3": ("001",),
            }),
            synthesize("SP2T optimized states", sp2t_path, {}),
        ],
    }


def run_switch_tutorial_measured_synthesis_case(
    root: Path, name: str, component_root: Path, *, full_network: bool = False
) -> dict:
    """Snap the automatic SP2T solution to real 0402CS/GJM15 S2P parts."""
    antenna_path, _, sp2t_path = SWITCH_CASES[name]
    antenna = load_touchstone(root / antenna_path)
    configurations = (
        FrequencyConfiguration("Set 1", {0: (Band(704e6, 746e6), Band(1920e6, 2170e6))}),
        FrequencyConfiguration("Set 2", {0: (Band(791e6, 862e6), Band(1920e6, 2170e6))}),
        FrequencyConfiguration("Set 3", {0: (Band(880e6, 960e6), Band(1920e6, 2170e6))}),
    )
    active = np.zeros(len(antenna.frequencies_hz), dtype=bool)
    for configuration in configurations:
        for band in configuration.bands_by_port[0]:
            active |= band.mask(antenna.frequencies_hz)
    problem = SwitchTunableProblem(
        antenna.frequencies_hz[active], antenna.s_parameters[active, 0, 0],
        configurations, antenna.z0,
    )
    optimizer = SwitchTunableOptimizer(
        problem,
        load_mdif(root / sp2t_path),
        config=SwitchSearchConfig(restarts=2, iterations=10, keep=8, seed=1),
    )
    ideal = (
        optimizer.optimize_full_network(max_input_elements=2, coarse_iterations=3)
        if full_network
        else optimizer.optimize(input_topologies=[(("series", "L"),)])
    )
    measured = SwitchMeasuredComponentOptimizer(
        optimizer,
        load_coilcraft_0402cs_catalog(
            component_root / "Inductors" / "Coilcraft Inductors 0402cs"
        ),
        load_murata_gjm15_catalog(
            component_root / "Capacitors" / "Murata Capacitors gjm15"
        ),
        nearest_parts=3,
        ideal_seed_keep=3 if full_network else 2,
        result_keep=8,
    ).optimize(ideal.candidates)
    best = measured.best
    branch_models = tuple(load_component_model(item) for item in best.branch_components)
    input_models = tuple(
        InputModelPlacement(connection, load_component_model(item))
        for connection, item in best.input_components
    )
    loaded_by_label = {loaded.label: loaded for loaded in optimizer.loaded_states}
    configuration_curves = []
    for configuration in configurations:
        state_label = best.state_by_configuration[configuration.name]
        power = evaluate_loaded_switch_physical_power(
            loaded_by_label[state_label], branch_models, input_elements=input_models
        )
        mask = np.zeros(len(problem.frequencies_hz), dtype=bool)
        for band in configuration.bands_by_port[0]:
            mask |= band.mask(problem.frequencies_hz)
        configuration_curves.append({
            "configuration": configuration.name,
            "state": state_label,
            "frequency_hz": problem.frequencies_hz[mask].tolist(),
            "s11_db": (
                20.0 * np.log10(np.maximum(np.abs(power.input_gamma[mask]), 1e-15))
            ).tolist(),
            "total_efficiency": power.dut_absorbed_power[mask].tolist(),
            "accepted_efficiency": power.input_accepted_power[mask].tolist(),
            "switch_loss": power.switch_loss[mask].tolist(),
            "matching_network_loss": power.matching_network_loss[mask].tolist(),
        })
    return {
        "case": name,
        "mode": "switch-mdif-full-network-measured-synthesis" if full_network else "switch-mdif-measured-synthesis",
        "variant": "SP2T optimized states",
        "score_db": best.score_db,
        "state_by_configuration": best.state_by_configuration,
        "branch_components": [
            {
                "branch": index + 1,
                "kind": item.kind,
                "part_number": item.name,
                "value_si": item.value,
                "family": item.family,
                "model_sha256": component_sha256(item),
            }
            for index, item in enumerate(best.branch_components)
        ],
        "input_components": [
            {
                "position": index + 1,
                "connection": connection,
                "kind": item.kind,
                "part_number": item.name,
                "value_si": item.value,
                "family": item.family,
                "model_sha256": component_sha256(item),
            }
            for index, (connection, item) in enumerate(best.input_components)
        ],
        "ideal_evaluations": ideal.evaluations,
        "physical_evaluations": measured.physical_evaluations,
        "component_models_loaded": measured.loaded_component_models,
        "active_frequency_points": int(np.sum(active)),
        "average_matching_network_loss": best.metrics["average_matching_network_loss"],
        "maximum_power_balance_error": best.metrics["maximum_power_balance_error"],
        "maximum_switch_model_gain": best.metrics["maximum_switch_model_gain"],
        "maximum_component_model_nonpassivity": best.metrics["maximum_component_model_nonpassivity"],
        "configuration_curves": configuration_curves,
        "complexity_alternatives": [
            {
                "input_component_count": count,
                "score_db": next(
                    candidate.score_db
                    for candidate in measured.candidates
                    if len(candidate.input_components) == count
                ),
            }
            for count in sorted({len(candidate.input_components) for candidate in measured.candidates})
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial-root", type=Path, required=True)
    parser.add_argument("--case", choices=sorted(set(CASES) | set(MULTIPORT_CASES) | set(MULTI_SCENARIO_CASES) | set(SWITCH_CASES)), default="quick-start")
    args = parser.parse_args()
    runner = run_switch_tutorial_case if args.case in SWITCH_CASES else run_multiport_case if args.case in MULTIPORT_CASES else run_multi_scenario_case if args.case in MULTI_SCENARIO_CASES else run_case
    print(runner(args.tutorial_root, args.case))


if __name__ == "__main__":
    main()
