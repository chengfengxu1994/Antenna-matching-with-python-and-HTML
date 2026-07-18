from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Branch, CircuitTopology, LumpedModel, S2PModel
from .network import s_to_z, z_to_s
from .transmission_line import TransmissionLineModel, TransmissionLineStubModel
from .microstrip import MicrostripLineModel, MicrostripStubModel, MicrostripVariation


@dataclass
class CircuitEvaluation:
    s_parameters: np.ndarray
    component_loss: np.ndarray
    dut_absorbed_power: np.ndarray
    power_balance_error: np.ndarray


@dataclass
class PhysicalSweep:
    s_parameters: np.ndarray
    total_efficiency: np.ndarray
    component_loss: np.ndarray
    dut_absorbed_power: np.ndarray
    power_balance_error: np.ndarray


def _s_to_y(s: np.ndarray, z0: float | np.ndarray) -> np.ndarray:
    # Compute Y directly. Converting through Z and then inverting fails for
    # perfectly floating series elements: their common-mode Z is infinite,
    # while the admittance matrix is finite (but singular) and well-defined.
    s = np.asarray(s, dtype=complex)
    count = s.shape[0]
    z0_values = np.full(count, float(z0)) if np.ndim(z0) == 0 else np.asarray(z0, dtype=float)
    inverse_root = np.diag(1.0 / np.sqrt(z0_values))
    identity = np.eye(count, dtype=complex)
    return inverse_root @ (identity - s) @ np.linalg.solve(identity + s, inverse_root)


def _lumped_impedance(model: LumpedModel, frequency_hz: float, scale: float = 1.0) -> complex:
    value = model.value * scale
    omega = 2.0 * np.pi * frequency_hz
    if model.kind == "R":
        return complex(value)
    if model.kind == "L":
        reactance = omega * value
        q_reactance = (
            2.0 * np.pi * model.q_reference_hz * value
            if model.q_reference_hz is not None
            else reactance
        )
        resistance = model.esr + (q_reactance / model.q if model.q else 0.0)
        return resistance + 1j * reactance
    reactance = -1.0 / (omega * value)
    resistance = model.esr + (abs(reactance) / model.q if model.q else 0.0)
    return resistance + 1j * reactance


def measured_component_admittance(
    model: S2PModel, frequency_hz: float, scale: float = 1.0
) -> np.ndarray:
    """Return a measured two-port Y matrix with a perturbed nominal L/C value.

    The measured matrix is interpreted as a general pi network. The common
    series branch is extracted from the two transfer admittances; only its
    ideal nominal reactance is shifted. Port shunts, ESR, asymmetry and the
    remaining measured residual stay unchanged. This avoids the nonphysical
    practice of multiplying the complete S matrix by a scalar.
    """
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("component variation scale must be finite and positive")
    measured_y = _s_to_y(model.at(frequency_hz), model.z0)
    if np.isclose(scale, 1.0, rtol=0.0, atol=1e-15):
        return measured_y
    if model.kind not in {"L", "C"} or model.nominal_value is None:
        raise ValueError(
            f"measured component {model.name!r} requires kind and nominal_value metadata for tolerance variation"
        )
    series_admittance = -0.5 * (measured_y[0, 1] + measured_y[1, 0])
    if abs(series_admittance) < 1e-18:
        raise ValueError(f"measured component {model.name!r} has no extractable series branch")
    omega = 2.0 * np.pi * frequency_hz
    value = float(model.nominal_value)
    if model.kind == "L":
        nominal_reactance = 1j * omega * value
        varied_reactance = 1j * omega * value * scale
    else:
        nominal_reactance = 1.0 / (1j * omega * value)
        varied_reactance = 1.0 / (1j * omega * value * scale)
    varied_series_admittance = 1.0 / (
        1.0 / series_admittance + varied_reactance - nominal_reactance
    )
    delta = varied_series_admittance - series_admittance
    varied_y = measured_y.copy()
    varied_y[0, 0] += delta
    varied_y[1, 1] += delta
    varied_y[0, 1] -= delta
    varied_y[1, 0] -= delta
    return varied_y


def _nodes(topology: CircuitTopology) -> list[str]:
    nodes = list(topology.external_nodes)
    for node in topology.dut_nodes:
        if node not in nodes:
            nodes.append(node)
    for branch in topology.branches:
        for node in (branch.node_a, branch.node_b):
            if node is not None and node not in nodes:
                nodes.append(node)
    return nodes


def _stamp_branch(y: np.ndarray, branch: Branch, indices: dict[str, int], frequency_hz: float, scale: float | MicrostripVariation) -> tuple[list[int], np.ndarray]:
    a = indices[branch.node_a]
    if isinstance(branch.model, (TransmissionLineStubModel, MicrostripStubModel)):
        if branch.node_b is not None:
            raise ValueError("a transmission-line stub must connect one node to ground")
        admittance = branch.model.input_admittance(frequency_hz, scale)
        stamp = np.asarray([[admittance]], dtype=complex)
        y[a, a] += admittance
        return [a], stamp
    if isinstance(branch.model, (TransmissionLineModel, MicrostripLineModel)):
        if branch.node_b is None:
            raise ValueError("a through transmission line requires two circuit nodes")
        b = indices[branch.node_b]
        stamp = branch.model.admittance(frequency_hz, scale)
        y[np.ix_([a, b], [a, b])] += stamp
        return [a, b], stamp
    if isinstance(branch.model, LumpedModel):
        admittance = 1.0 / _lumped_impedance(branch.model, frequency_hz, scale)
        if branch.node_b is None:
            y[a, a] += admittance
            return [a], np.array([[admittance]], dtype=complex)
        b = indices[branch.node_b]
        stamp = admittance * np.array([[1, -1], [-1, 1]], dtype=complex)
        y[np.ix_([a, b], [a, b])] += stamp
        return [a, b], stamp
    model_y = measured_component_admittance(branch.model, frequency_hz, scale)
    if branch.node_b is None:
        # For a measured two-terminal component represented by an S2P series
        # fixture, a shunt placement grounds port 2 and presents Y11 at port 1.
        stamp = model_y[:1, :1]
        y[a, a] += stamp[0, 0]
        return [a], stamp
    b = indices[branch.node_b]
    stamp = model_y
    y[np.ix_([a, b], [a, b])] += stamp
    return [a, b], stamp


def evaluate_circuit(
    dut_s: np.ndarray,
    topology: CircuitTopology,
    frequency_hz: float,
    z0: float | np.ndarray = 50.0,
    variation: dict[str, float | MicrostripVariation] | None = None,
) -> CircuitEvaluation:
    """Solve a physical nodal circuit and account for each component's real power."""
    variation = variation or {}
    n_external = len(topology.external_nodes)
    if dut_s.shape != (n_external, n_external):
        raise ValueError("DUT port count must equal topology external node count")
    nodes = _nodes(topology)
    indices = {name: pos for pos, name in enumerate(nodes)}
    y_full = np.zeros((len(nodes), len(nodes)), dtype=complex)
    dut_nodes = topology.dut_nodes or topology.external_nodes
    dut_indices = [indices[node] for node in dut_nodes]
    y_full[np.ix_(dut_indices, dut_indices)] += _s_to_y(dut_s, z0)
    stamps = []
    for branch in topology.branches:
        branch_nodes, stamp = _stamp_branch(y_full, branch, indices, frequency_hz, variation.get(branch.name, 1.0))
        stamps.append((branch.name, branch_nodes, stamp))

    if len(nodes) > n_external:
        ee = y_full[:n_external, :n_external]
        ei = y_full[:n_external, n_external:]
        ie = y_full[n_external:, :n_external]
        ii = y_full[n_external:, n_external:]
        y_external = ee - ei @ np.linalg.solve(ii, ie)
    else:
        y_external = y_full
    s_external = z_to_s(np.linalg.inv(y_external), z0)

    z0_values = np.full(n_external, float(z0)) if np.ndim(z0) == 0 else np.asarray(z0, dtype=float)
    component_loss = np.zeros((n_external, len(topology.branches)), dtype=float)
    dut_absorbed = np.zeros(n_external, dtype=float)
    balance_error = np.zeros(n_external, dtype=float)
    root = np.sqrt(z0_values)
    for driven in range(n_external):
        incident = np.zeros(n_external, dtype=complex)
        incident[driven] = 1.0
        rhs = np.zeros(len(nodes), dtype=complex)
        rhs[:n_external] = 2.0 * incident / root
        boundary = np.zeros_like(y_full)
        boundary[np.arange(n_external), np.arange(n_external)] = 1.0 / z0_values
        voltages = np.linalg.solve(y_full + boundary, rhs)
        for ci, (_, branch_nodes, stamp) in enumerate(stamps):
            v = voltages[branch_nodes]
            component_loss[driven, ci] = max(0.0, float(np.real(np.vdot(v, stamp @ v))))
        reflected = float(np.sum(np.abs(s_external[:, driven]) ** 2))
        component_total = float(np.sum(component_loss[driven]))
        dut_absorbed[driven] = max(0.0, 1.0 - reflected - component_total)
        balance_error[driven] = 1.0 - reflected - component_total - dut_absorbed[driven]
    return CircuitEvaluation(s_external, component_loss, dut_absorbed, balance_error)


def evaluate_physical_problem(
    problem,
    topology: CircuitTopology,
    variation: dict[str, float | MicrostripVariation] | None = None,
) -> PhysicalSweep:
    """Evaluate one physical topology over every Problem frequency."""
    n_freq = len(problem.frequencies_hz)
    n_ports = problem.s_parameters.shape[1]
    s = np.empty_like(problem.s_parameters)
    component_loss = np.zeros((n_freq, n_ports), dtype=float)
    absorbed = np.zeros((n_freq, n_ports), dtype=float)
    total_efficiency = np.zeros((n_freq, n_ports), dtype=float)
    errors = np.zeros((n_freq, n_ports), dtype=float)
    for fi, frequency in enumerate(problem.frequencies_hz):
        result = evaluate_circuit(
            problem.s_parameters[fi], topology, frequency, problem.z0, variation
        )
        s[fi] = result.s_parameters
        component_loss[fi] = np.sum(result.component_loss, axis=1)
        absorbed[fi] = result.dut_absorbed_power
        errors[fi] = result.power_balance_error
        for port in range(n_ports):
            eta_rad = problem.radiation_efficiency.get(port, np.ones(n_freq))[fi]
            total_efficiency[fi, port] = float(eta_rad) * absorbed[fi, port]
    return PhysicalSweep(s, total_efficiency, component_loss, absorbed, errors)
