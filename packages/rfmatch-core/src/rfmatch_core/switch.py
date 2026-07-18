"""Evaluation primitives for multi-throw switch matching networks.

The switch common port is connected to the DUT.  Each remaining throw is
reached from one shared RF input node through a series reactance.  This is the
topology used by Optenni Lab's official SP2T/SP3T impedance-tuning tutorial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .mdif import MDIFState
from .models import S2PModel
from .network import s_to_z, terminate
from .physical import measured_component_admittance


@dataclass(frozen=True)
class SeriesReactance:
    kind: Literal["L", "C"]
    value: float

    def impedance(self, frequency_hz: float) -> complex:
        if frequency_hz <= 0:
            raise ValueError("frequency must be positive")
        if self.value <= 0:
            raise ValueError("reactance value must be positive")
        omega = 2.0 * np.pi * frequency_hz
        return 1j * omega * self.value if self.kind == "L" else 1.0 / (1j * omega * self.value)


@dataclass(frozen=True)
class InputReactance:
    connection: Literal["series", "shunt"]
    kind: Literal["L", "C"]
    value: float

    def impedance(self, frequency_hz: float) -> complex:
        return SeriesReactance(self.kind, self.value).impedance(frequency_hz)


@dataclass(frozen=True)
class InputModelPlacement:
    connection: Literal["series", "shunt"]
    model: S2PModel


@dataclass(frozen=True)
class LoadedSwitchState:
    """One MDIF state reduced to its throw-port Z matrices for one DUT sweep."""

    label: str
    frequencies_hz: np.ndarray
    throws_z: np.ndarray
    z0: float
    switch_s: np.ndarray
    dut_gamma: np.ndarray
    common_port: int
    throw_ports: tuple[int, ...]


@dataclass(frozen=True)
class SwitchPowerSweep:
    input_gamma: np.ndarray
    input_accepted_power: np.ndarray
    dut_absorbed_power: np.ndarray
    switch_loss: np.ndarray
    matching_network_loss: np.ndarray
    power_balance_error: np.ndarray


def preload_switch_state(
    frequencies_hz: Sequence[float],
    dut_s11: Sequence[complex],
    switch_state: MDIFState,
    *,
    common_port: int = 0,
    z0: float = 50.0,
) -> LoadedSwitchState:
    """Terminate the common port once so candidate searches can reuse it."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    dut = np.asarray(dut_s11, dtype=complex)
    if frequencies.ndim != 1 or dut.shape != frequencies.shape:
        raise ValueError("frequencies_hz and dut_s11 must be equal-length vectors")
    switch_sweep = switch_state.sweep_at(frequencies)
    if not 0 <= common_port < switch_state.n_ports:
        raise ValueError("common_port is outside the switch S-parameter matrix")
    kept = [port for port in range(switch_state.n_ports) if port != common_port]
    throws_s = np.empty((len(frequencies), len(kept), len(kept)), dtype=complex)
    for index, (switch_s, gamma) in enumerate(zip(switch_sweep, dut)):
        throws_s[index], _ = terminate(switch_s, {common_port: gamma})
    return LoadedSwitchState(
        switch_state.label,
        frequencies,
        s_to_z(throws_s, z0),
        z0,
        switch_sweep,
        dut,
        common_port,
        tuple(kept),
    )


def _apply_input_network(
    impedance: np.ndarray | complex,
    frequencies_hz: np.ndarray,
    elements: Sequence[InputReactance],
) -> np.ndarray:
    result = np.asarray(impedance, dtype=complex).copy()
    omega = 2.0 * np.pi * frequencies_hz
    for element in elements:
        if element.value <= 0:
            raise ValueError("reactance value must be positive")
        reactance = 1j * omega * element.value if element.kind == "L" else 1.0 / (1j * omega * element.value)
        if element.connection == "series":
            result += reactance
        else:
            result = 1.0 / (1.0 / result + 1.0 / reactance)
    return result


def evaluate_loaded_switch_state(
    loaded: LoadedSwitchState,
    branch_reactances: Sequence[SeriesReactance],
    *,
    input_reactances: Sequence[InputReactance] = (),
) -> np.ndarray:
    """Vectorized S11 evaluation using a pre-terminated switch state."""
    if loaded.throws_z.ndim != 3 or loaded.throws_z.shape[0] != len(loaded.frequencies_hz):
        raise ValueError("loaded switch state has an invalid matrix sweep")
    throw_count = loaded.throws_z.shape[1]
    if loaded.throws_z.shape[2] != throw_count or len(branch_reactances) != throw_count:
        raise ValueError("one branch reactance is required for every switch throw")
    omega = 2.0 * np.pi * loaded.frequencies_hz
    branch_z = np.empty((len(loaded.frequencies_hz), throw_count), dtype=complex)
    for index, element in enumerate(branch_reactances):
        if element.value <= 0:
            raise ValueError("reactance value must be positive")
        branch_z[:, index] = 1j * omega * element.value if element.kind == "L" else 1.0 / (1j * omega * element.value)
    matrices = loaded.throws_z.copy()
    diagonal = np.arange(throw_count)
    matrices[:, diagonal, diagonal] += branch_z
    input_admittance = np.sum(np.linalg.inv(matrices), axis=(1, 2))
    input_impedance = np.divide(
        1.0,
        input_admittance,
        out=np.full_like(input_admittance, np.inf + 0j),
        where=np.abs(input_admittance) >= 1e-30,
    )
    input_impedance = _apply_input_network(input_impedance, loaded.frequencies_hz, input_reactances)
    return (input_impedance - loaded.z0) / (input_impedance + loaded.z0)


def evaluate_loaded_switch_power(
    loaded: LoadedSwitchState,
    branch_reactances: Sequence[SeriesReactance],
    *,
    input_reactances: Sequence[InputReactance] = (),
) -> SwitchPowerSweep:
    """Reconstruct switch waves and separate DUT absorption from switch loss."""
    throw_count = loaded.throws_z.shape[1]
    if len(branch_reactances) != throw_count:
        raise ValueError("one branch reactance is required for every switch throw")
    frequencies = loaded.frequencies_hz
    omega = 2.0 * np.pi * frequencies
    branch_z = np.empty((len(frequencies), throw_count), dtype=complex)
    for index, element in enumerate(branch_reactances):
        if element.value <= 0:
            raise ValueError("reactance value must be positive")
        branch_z[:, index] = 1j * omega * element.value if element.kind == "L" else 1.0 / (1j * omega * element.value)
    matrices = loaded.throws_z.copy()
    diagonal = np.arange(throw_count)
    matrices[:, diagonal, diagonal] += branch_z
    inverse = np.linalg.inv(matrices)
    input_admittance = np.sum(inverse, axis=(1, 2))
    node_impedance = np.divide(
        1.0,
        input_admittance,
        out=np.full_like(input_admittance, np.inf + 0j),
        where=np.abs(input_admittance) >= 1e-30,
    )
    input_impedance = _apply_input_network(node_impedance, frequencies, input_reactances)
    input_gamma = (input_impedance - loaded.z0) / (input_impedance + loaded.z0)

    root_z0 = np.sqrt(loaded.z0)
    voltage = root_z0 * (1.0 + input_gamma)
    current = (1.0 - input_gamma) / root_z0
    # Reverse the DUT-outward ladder to recover the shared branch-node waves.
    for element in reversed(input_reactances):
        reactance = 1j * omega * element.value if element.kind == "L" else 1.0 / (1j * omega * element.value)
        if element.connection == "series":
            voltage = voltage - current * reactance
        else:
            current = current - voltage / reactance
    node_voltage = voltage

    throw_currents = np.einsum("fij,fj->fi", inverse, np.broadcast_to(node_voltage[:, None], (len(frequencies), throw_count)))
    throw_voltages = node_voltage[:, None] - throw_currents * branch_z
    a_throw = (throw_voltages + loaded.z0 * throw_currents) / (2.0 * root_z0)

    common = loaded.common_port
    s_common_throws = loaded.switch_s[:, common, :][:, loaded.throw_ports]
    numerator = np.einsum("fi,fi->f", s_common_throws, a_throw)
    denominator = 1.0 - loaded.switch_s[:, common, common] * loaded.dut_gamma
    b_common = numerator / denominator
    a_common = loaded.dut_gamma * b_common
    a = np.empty((len(frequencies), throw_count + 1), dtype=complex)
    a[:, common] = a_common
    a[:, loaded.throw_ports] = a_throw
    b = np.einsum("fij,fj->fi", loaded.switch_s, a)

    accepted = 1.0 - np.abs(input_gamma) ** 2
    dut_absorbed = np.abs(b[:, common]) ** 2 - np.abs(a[:, common]) ** 2
    # Preserve the signed value. Some vendor/"ideal" files are microscopically
    # non-passive; clipping negative loss would hide that and break conservation.
    switch_loss = np.sum(np.abs(a) ** 2, axis=1) - np.sum(np.abs(b) ** 2, axis=1)
    balance = accepted - dut_absorbed - switch_loss
    return SwitchPowerSweep(
        input_gamma,
        accepted,
        dut_absorbed,
        switch_loss,
        np.zeros_like(switch_loss),
        balance,
    )


def _s_to_y_matrix(s: np.ndarray, z0: float) -> np.ndarray:
    count = s.shape[0]
    inverse_root = np.eye(count, dtype=complex) / np.sqrt(z0)
    identity = np.eye(count, dtype=complex)
    return inverse_root @ (identity - s) @ np.linalg.solve(identity + s, inverse_root)


def _series_model_y_sweep(
    model: SeriesReactance | S2PModel,
    frequencies_hz: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("component variation scale must be finite and positive")
    if isinstance(model, SeriesReactance):
        omega = 2.0 * np.pi * frequencies_hz
        value = model.value * scale
        impedance = 1j * omega * value if model.kind == "L" else 1.0 / (1j * omega * value)
        admittance = 1.0 / impedance
        pattern = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=complex)
        return admittance[:, None, None] * pattern
    return np.asarray([
        measured_component_admittance(model, float(frequency), scale)
        for frequency in frequencies_hz
    ])


def evaluate_loaded_switch_physical_power(
    loaded: LoadedSwitchState,
    branch_models: Sequence[SeriesReactance | S2PModel],
    *,
    input_elements: Sequence[InputReactance | InputModelPlacement] = (),
    branch_scales: Sequence[float] | None = None,
    input_scales: Sequence[float] | None = None,
) -> SwitchPowerSweep:
    """Solve a switch with ideal or measured two-port matching components.

    A nodal Y matrix retains the shared input node, every throw, and all series
    ladder nodes. Measured S2P branches are stamped directly, so ESR,
    self-resonance, and fixture asymmetry participate in the coupled solution.
    """
    frequencies = loaded.frequencies_hz
    throw_count = loaded.throws_z.shape[1]
    if len(branch_models) != throw_count:
        raise ValueError("one branch model is required for every switch throw")
    branch_scales = tuple(
        (1.0,) * len(branch_models) if branch_scales is None else branch_scales
    )
    input_scales = tuple(
        (1.0,) * len(input_elements) if input_scales is None else input_scales
    )
    if len(branch_scales) != len(branch_models):
        raise ValueError("branch_scales must match branch_models")
    if len(input_scales) != len(input_elements):
        raise ValueError("input_scales must match input_elements")
    series_input_count = sum(
        1
        for item in input_elements
        if (item.connection if isinstance(item, (InputReactance, InputModelPlacement)) else "") == "series"
    )
    node_count = 1 + throw_count + series_input_count
    shared = 0
    throws = list(range(1, throw_count + 1))
    y_full = np.zeros((len(frequencies), node_count, node_count), dtype=complex)
    switch_y = np.linalg.inv(loaded.throws_z)
    for row, row_node in enumerate(throws):
        for col, col_node in enumerate(throws):
            y_full[:, row_node, col_node] += switch_y[:, row, col]

    stamps: list[tuple[tuple[int, ...], np.ndarray]] = []
    for index, model in enumerate(branch_models):
        stamp = _series_model_y_sweep(model, frequencies, float(branch_scales[index]))
        nodes = (shared, throws[index])
        for row, row_node in enumerate(nodes):
            for col, col_node in enumerate(nodes):
                y_full[:, row_node, col_node] += stamp[:, row, col]
        stamps.append((nodes, stamp))

    current_node = shared
    next_node = 1 + throw_count
    for element_index, element in enumerate(input_elements):
        connection = element.connection
        scale = float(input_scales[element_index])
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("component variation scale must be finite and positive")
        if isinstance(element, InputReactance):
            if element.value <= 0:
                raise ValueError("reactance value must be positive")
            if connection == "series":
                model = SeriesReactance(element.kind, element.value)
                stamp = _series_model_y_sweep(model, frequencies, scale)
            else:
                omega = 2.0 * np.pi * frequencies
                value = element.value * scale
                impedance = 1j * omega * value if element.kind == "L" else 1.0 / (1j * omega * value)
                stamp = (1.0 / impedance)[:, None, None]
        else:
            model_y = _series_model_y_sweep(element.model, frequencies, scale)
            stamp = model_y if connection == "series" else model_y[:, :1, :1]
        if connection == "series":
            outer_node = next_node
            next_node += 1
            nodes = (outer_node, current_node)
            current_node = outer_node
        else:
            nodes = (current_node,)
        for row, row_node in enumerate(nodes):
            for col, col_node in enumerate(nodes):
                y_full[:, row_node, col_node] += stamp[:, row, col]
        stamps.append((nodes, stamp))
    external = current_node

    internal = [node for node in range(node_count) if node != external]
    if internal:
        yee = y_full[:, external, external]
        yei = y_full[:, external, internal]
        yie = y_full[:, internal, external]
        yii = y_full[:, internal, :][:, :, internal]
        correction = np.einsum("fi,fi->f", yei, np.linalg.solve(yii, yie))
        input_admittance = yee - correction
    else:
        input_admittance = y_full[:, external, external]
    input_impedance = 1.0 / input_admittance
    input_gamma = (input_impedance - loaded.z0) / (input_impedance + loaded.z0)

    boundary = y_full.copy()
    boundary[:, external, external] += 1.0 / loaded.z0
    rhs = np.zeros((len(frequencies), node_count), dtype=complex)
    rhs[:, external] = 2.0 / np.sqrt(loaded.z0)
    voltages = np.linalg.solve(boundary, rhs)
    throw_voltages = voltages[:, throws]
    throw_currents = np.einsum("fij,fj->fi", switch_y, throw_voltages)
    root_z0 = np.sqrt(loaded.z0)
    a_throw = (throw_voltages + loaded.z0 * throw_currents) / (2.0 * root_z0)

    common = loaded.common_port
    s_common_throws = loaded.switch_s[:, common, :][:, loaded.throw_ports]
    b_common = np.einsum("fi,fi->f", s_common_throws, a_throw) / (
        1.0 - loaded.switch_s[:, common, common] * loaded.dut_gamma
    )
    a_common = loaded.dut_gamma * b_common
    a = np.empty((len(frequencies), throw_count + 1), dtype=complex)
    a[:, common] = a_common
    a[:, loaded.throw_ports] = a_throw
    b = np.einsum("fij,fj->fi", loaded.switch_s, a)

    matching_loss = np.zeros(len(frequencies), dtype=float)
    for nodes, stamp in stamps:
        component_voltages = voltages[:, nodes]
        component_currents = np.einsum("fij,fj->fi", stamp, component_voltages)
        matching_loss += np.real(np.sum(component_voltages * np.conj(component_currents), axis=1))
    accepted = 1.0 - np.abs(input_gamma) ** 2
    dut_absorbed = np.abs(b[:, common]) ** 2 - np.abs(a[:, common]) ** 2
    switch_loss = np.sum(np.abs(a) ** 2, axis=1) - np.sum(np.abs(b) ** 2, axis=1)
    balance = accepted - dut_absorbed - switch_loss - matching_loss
    return SwitchPowerSweep(
        input_gamma, accepted, dut_absorbed, switch_loss, matching_loss, balance
    )


def reduce_switch_with_series_branches(
    switch_s: np.ndarray,
    *,
    common_port: int,
    dut_gamma: complex,
    branch_impedances: Sequence[complex],
    input_series_impedance: complex = 0j,
    z0: float = 50.0,
) -> complex:
    """Return input reflection after loading a switch and tying its throws.

    The DUT terminates ``common_port``.  The other switch ports each receive a
    series impedance and are then tied to one input node.  In Z/Y form, tying
    ports means equal voltages and summed currents, so ``Yin = 1.T @ Y @ 1``.
    """
    switch_s = np.asarray(switch_s, dtype=complex)
    if switch_s.ndim != 2 or switch_s.shape[0] != switch_s.shape[1]:
        raise ValueError("switch_s must be a square S-parameter matrix")
    n_ports = switch_s.shape[0]
    if n_ports < 2 or not 0 <= common_port < n_ports:
        raise ValueError("common_port is outside the switch S-parameter matrix")
    if len(branch_impedances) != n_ports - 1:
        raise ValueError("one branch impedance is required for every switch throw")
    if z0 <= 0:
        raise ValueError("z0 must be positive")

    throws_s, _ = terminate(switch_s, {common_port: complex(dut_gamma)})
    throws_z = s_to_z(throws_s, z0)
    throws_z[np.diag_indices_from(throws_z)] += np.asarray(branch_impedances, dtype=complex)
    throws_y = np.linalg.inv(throws_z)
    input_admittance = np.sum(throws_y)
    if abs(input_admittance) < 1e-30:
        input_impedance = complex(np.inf)
    else:
        input_impedance = 1.0 / input_admittance + input_series_impedance
    if np.isinf(input_impedance):
        return 1.0 + 0j
    return (input_impedance - z0) / (input_impedance + z0)


def evaluate_switched_matching(
    frequencies_hz: Sequence[float],
    dut_s11: Sequence[complex],
    switch_state: MDIFState,
    branch_reactances: Sequence[SeriesReactance],
    *,
    common_port: int = 0,
    input_series_reactances: Sequence[SeriesReactance] = (),
    input_reactances: Sequence[InputReactance] = (),
    z0: float = 50.0,
) -> np.ndarray:
    """Evaluate S11 for a switched matching topology over frequency."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    dut = np.asarray(dut_s11, dtype=complex)
    if frequencies.ndim != 1 or dut.shape != frequencies.shape:
        raise ValueError("frequencies_hz and dut_s11 must be equal-length vectors")
    if switch_state.n_ports - 1 != len(branch_reactances):
        raise ValueError("branch_reactances must match the number of switch throws")
    if input_series_reactances and input_reactances:
        raise ValueError("use input_series_reactances or input_reactances, not both")
    fixed = tuple(input_reactances) or tuple(
        InputReactance("series", item.kind, item.value) for item in input_series_reactances
    )
    loaded = preload_switch_state(
        frequencies, dut, switch_state, common_port=common_port, z0=z0
    )
    return evaluate_loaded_switch_state(loaded, branch_reactances, input_reactances=fixed)
