from __future__ import annotations

import numpy as np

from .models import Candidate, LumpedLossModel, LumpedModel, Objective, Problem
from .network import apply_elements_sweep


def _blend_min_average(values_db: np.ndarray, average_weight: float) -> float:
    if len(values_db) == 0:
        return float("-inf")
    w = float(np.clip(average_weight, 0.0, 1.0))
    return (1.0 - w) * float(np.min(values_db)) + w * float(np.mean(values_db))


def evaluate_isolation_targets(
    frequencies_hz: np.ndarray,
    matched: np.ndarray,
    targets,
) -> dict:
    """Evaluate directed S_destination,source constraints for any matched sweep."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    matched = np.asarray(matched, dtype=complex)
    if matched.ndim != 3 or matched.shape[0] != len(frequencies_hz) or matched.shape[1] != matched.shape[2]:
        raise ValueError("matched S parameters must have shape (frequency, port, port)")
    transmission_db = 20.0 * np.log10(np.maximum(np.abs(matched), 1e-15))
    penalty_total_db = 0.0
    metrics = []
    for target in targets:
        mask = target.mask(frequencies_hz)
        if not np.any(mask):
            raise ValueError(f"isolation target {target} has no frequency samples")
        values_db = transmission_db[mask, target.destination_port, target.source_port]
        violations_db = np.maximum(0.0, values_db - target.maximum_db)
        penalty_db = target.weight * (
            (1.0 - target.average_weight) * float(np.max(violations_db))
            + target.average_weight * float(np.mean(violations_db))
        )
        penalty_total_db += penalty_db
        metrics.append({
            "source_port": target.source_port,
            "destination_port": target.destination_port,
            "maximum_allowed_db": target.maximum_db,
            "worst_transmission_db": float(np.max(values_db)),
            "average_transmission_db": float(np.mean(values_db)),
            "penalty_db": penalty_db,
            "passed": bool(np.max(values_db) <= target.maximum_db),
        })
    return {
        "transmission_db": transmission_db,
        "targets": metrics,
        "penalty_db": penalty_total_db,
        "passed": all(item["passed"] for item in metrics),
    }


def evaluate(problem: Problem, candidate: Candidate, objective: Objective) -> Candidate:
    n_freq, n_ports, _ = problem.s_parameters.shape
    matched = apply_elements_sweep(problem.s_parameters, candidate.elements, problem.frequencies_hz, problem.z0)
    power = np.abs(matched) ** 2
    outgoing_power = np.sum(power, axis=1)
    radiation = np.ones((n_freq, n_ports), dtype=float)
    for port, values in problem.radiation_efficiency.items():
        radiation[:, port] = np.asarray(values, dtype=float)
    total_eff = radiation * np.maximum(0.0, 1.0 - outgoing_power)
    return score_sweep(problem, candidate, objective, matched, total_eff)


def evaluate_lumped_physical(
    problem: Problem,
    candidate: Candidate,
    objective: Objective,
    loss_model: LumpedLossModel,
) -> Candidate:
    """Score continuous generic L/C values with explicit component losses."""
    if (
        problem.s_parameters.shape[1:] == (1, 1)
        and all(element.port == 0 for element in candidate.elements)
    ):
        frequencies = problem.frequencies_hz
        omega = 2.0 * np.pi * frequencies
        z0 = float(np.asarray(problem.z0))
        dut_gamma = problem.s_parameters[:, 0, 0]
        dut_impedance = z0 * (1.0 + dut_gamma) / (1.0 - dut_gamma)
        element_impedances = []
        input_impedance = dut_impedance.copy()
        for element in candidate.elements:
            if element.kind == "L":
                resistance = loss_model.inductor_esr
                if loss_model.inductor_q is not None:
                    resistance += (
                        2.0 * np.pi * loss_model.inductor_q_reference_hz
                        * element.value / loss_model.inductor_q
                    )
                impedance = resistance + 1j * omega * element.value
            else:
                impedance = (
                    loss_model.capacitor_esr
                    + 1.0 / (1j * omega * element.value)
                )
            element_impedances.append(impedance)
            if element.connection == "series":
                input_impedance = input_impedance + impedance
            else:
                input_impedance = 1.0 / (
                    1.0 / input_impedance + 1.0 / impedance
                )
        gamma = (input_impedance - z0) / (input_impedance + z0)
        voltage = np.sqrt(z0) * (1.0 + gamma)
        current = (1.0 - gamma) / np.sqrt(z0)
        for element, impedance in reversed(list(zip(candidate.elements, element_impedances))):
            if element.connection == "series":
                voltage = voltage - current * impedance
            else:
                current = current - voltage / impedance
        dut_absorbed = np.abs(voltage) ** 2 * np.real(1.0 / dut_impedance)
        radiation = problem.radiation_efficiency.get(0, np.ones(len(frequencies)))
        total_efficiency = (np.asarray(radiation) * dut_absorbed)[:, None]
        accepted = 1.0 - np.abs(gamma) ** 2
        component_loss = (accepted - dut_absorbed)[:, None]
        scored = score_sweep(
            problem,
            candidate,
            objective,
            gamma[:, None, None],
            total_efficiency,
        )
        scored.metrics.update({
            "s_parameters": gamma[:, None, None],
            "component_loss": component_loss,
            "dut_absorbed_power": dut_absorbed[:, None],
            "power_balance_error": np.zeros((len(frequencies), 1)),
            "maximum_power_balance_error": 0.0,
            "loss_model": {
                "inductor_q": loss_model.inductor_q,
                "inductor_q_reference_hz": loss_model.inductor_q_reference_hz,
                "inductor_esr": loss_model.inductor_esr,
                "capacitor_esr": loss_model.capacitor_esr,
            },
        })
        return scored

    from .physical import evaluate_physical_problem
    from .physical_optimizer import ModelPlacement, build_model_circuit_topology

    placements = []
    for index, element in enumerate(candidate.elements):
        if element.kind == "L":
            model = LumpedModel(
                f"L{index + 1}",
                "L",
                element.value,
                q=loss_model.inductor_q,
                esr=loss_model.inductor_esr,
                q_reference_hz=(
                    loss_model.inductor_q_reference_hz
                    if loss_model.inductor_q is not None
                    else None
                ),
            )
        else:
            model = LumpedModel(
                f"C{index + 1}", "C", element.value,
                esr=loss_model.capacitor_esr,
            )
        placements.append(ModelPlacement(element.connection, element.port, model))
    topology = build_model_circuit_topology(
        problem.s_parameters.shape[1], placements
    )
    sweep = evaluate_physical_problem(problem, topology)
    scored = score_sweep(
        problem, candidate, objective, sweep.s_parameters, sweep.total_efficiency
    )
    scored.metrics.update({
        "s_parameters": sweep.s_parameters,
        "component_loss": sweep.component_loss,
        "dut_absorbed_power": sweep.dut_absorbed_power,
        "power_balance_error": sweep.power_balance_error,
        "maximum_power_balance_error": float(np.max(np.abs(sweep.power_balance_error))),
        "loss_model": {
            "inductor_q": loss_model.inductor_q,
            "inductor_q_reference_hz": loss_model.inductor_q_reference_hz,
            "inductor_esr": loss_model.inductor_esr,
            "capacitor_esr": loss_model.capacitor_esr,
        },
    })
    return scored


def score_sweep(
    problem: Problem,
    candidate: Candidate,
    objective: Objective,
    matched: np.ndarray,
    total_eff: np.ndarray,
) -> Candidate:
    """Score precomputed matched S parameters and total-efficiency arrays."""
    matched = np.asarray(matched, dtype=complex)
    total_eff = np.asarray(total_eff, dtype=float)
    expected = problem.s_parameters.shape
    if matched.shape != expected:
        raise ValueError(f"matched S parameters must have shape {expected}")
    if total_eff.shape != expected[:2]:
        raise ValueError(f"total_efficiency must have shape {expected[:2]}")
    diagonal = np.diagonal(matched, axis1=1, axis2=2)
    return_loss = -20.0 * np.log10(np.maximum(np.abs(diagonal), 1e-15))
    isolation = evaluate_isolation_targets(problem.frequencies_hz, matched, problem.isolation_targets)
    transmission_db = isolation["transmission_db"]

    port_scores, band_metrics = [], {}
    for port, bands in problem.bands_by_port.items():
        scores = []
        for bi, band in enumerate(bands):
            mask = band.mask(problem.frequencies_hz)
            if not np.any(mask):
                raise ValueError(f"band {band} has no frequency samples")
            eff_db = 10.0 * np.log10(np.maximum(total_eff[mask, port], 1e-15))
            performance = _blend_min_average(eff_db, objective.within_band_average_weight)
            cost = (performance - band.target_db) * band.weight
            if objective.impedance_target_db is not None and objective.impedance_weight > 0:
                margin = float(np.min(return_loss[mask, port])) + objective.impedance_target_db
                cost += objective.impedance_weight * min(0.0, margin)
            scores.append(cost)
            band_metrics[(port, bi)] = {
                "minimum_efficiency_db": float(np.min(eff_db)),
                "average_efficiency_db": float(np.mean(eff_db)),
                "minimum_return_loss_db": float(np.min(return_loss[mask, port])),
                "weight": float(band.weight),
                "cost_db": cost,
            }
        port_scores.append(_blend_min_average(np.asarray(scores), objective.across_band_average_weight))

    isolation_penalty_db = isolation["penalty_db"]
    isolation_metrics = isolation["targets"]

    candidate.score_db = (
        _blend_min_average(np.asarray(port_scores), objective.port_average_weight)
        - objective.complexity_penalty_db * len(candidate.elements)
        - isolation_penalty_db
    )
    candidate.metrics = {
        "total_efficiency": total_eff,
        "return_loss_db": return_loss,
        "transmission_db": transmission_db,
        "bands": band_metrics,
        "port_scores_db": port_scores,
        "isolation_targets": isolation_metrics,
        "isolation_penalty_db": isolation_penalty_db,
        "isolation_constraints_passed": isolation["passed"],
    }
    return candidate
