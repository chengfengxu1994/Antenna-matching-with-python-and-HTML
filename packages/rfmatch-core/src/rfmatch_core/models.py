from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from .transmission_line import TransmissionLineModel, TransmissionLineStubModel
    from .microstrip import MicrostripLineModel, MicrostripStubModel


@dataclass(frozen=True)
class Band:
    start_hz: float
    stop_hz: float
    target_db: float = 0.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_hz) or not np.isfinite(self.stop_hz):
            raise ValueError("band bounds must be finite")
        if not np.isfinite(self.target_db):
            raise ValueError("band target must be finite")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("band weight must be finite and non-negative")

    def mask(self, frequencies_hz: np.ndarray) -> np.ndarray:
        lo, hi = sorted((self.start_hz, self.stop_hz))
        return (frequencies_hz >= lo) & (frequencies_hz <= hi)


@dataclass(frozen=True)
class IsolationTarget:
    """Maximum directed transmission S_destination,source over a frequency band."""

    source_port: int
    destination_port: int
    start_hz: float
    stop_hz: float
    maximum_db: float = -20.0
    weight: float = 1.0
    average_weight: float = 0.0

    def mask(self, frequencies_hz: np.ndarray) -> np.ndarray:
        lo, hi = sorted((self.start_hz, self.stop_hz))
        return (frequencies_hz >= lo) & (frequencies_hz <= hi)


@dataclass(frozen=True)
class Element:
    connection: Literal["series", "shunt"]
    kind: Literal["L", "C"]
    port: int
    value: float
    name: str = ""


@dataclass(frozen=True)
class Component:
    name: str
    kind: Literal["L", "C"]
    value: float
    tolerance: float = 0.0
    q: float | None = None
    esr: float | None = None


@dataclass(frozen=True)
class Objective:
    within_band_average_weight: float = 0.05
    across_band_average_weight: float = 0.10
    port_average_weight: float = 0.10
    complexity_penalty_db: float = 0.0
    impedance_target_db: float | None = None
    impedance_weight: float = 0.0


@dataclass(frozen=True)
class LumpedLossModel:
    """Generic-component loss assumptions used during continuous synthesis."""

    inductor_q: float | None = None
    inductor_q_reference_hz: float = 1e9
    inductor_esr: float = 0.0
    capacitor_esr: float = 0.0

    def __post_init__(self) -> None:
        if self.inductor_q is not None and self.inductor_q <= 0:
            raise ValueError("inductor_q must be positive")
        if self.inductor_q_reference_hz <= 0:
            raise ValueError("inductor_q_reference_hz must be positive")
        if self.inductor_esr < 0 or self.capacitor_esr < 0:
            raise ValueError("component ESR must be non-negative")


@dataclass
class Problem:
    frequencies_hz: np.ndarray
    s_parameters: np.ndarray
    bands_by_port: dict[int, Sequence[Band]]
    z0: float | np.ndarray = 50.0
    radiation_efficiency: dict[int, np.ndarray] = field(default_factory=dict)
    isolation_targets: Sequence[IsolationTarget] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.frequencies_hz = np.asarray(self.frequencies_hz, dtype=float)
        self.s_parameters = np.asarray(self.s_parameters, dtype=complex)
        if self.s_parameters.ndim != 3 or self.s_parameters.shape[0] != len(self.frequencies_hz) or self.s_parameters.shape[1] != self.s_parameters.shape[2]:
            raise ValueError("s_parameters must have shape (frequency, port, port)")
        z0 = np.asarray(self.z0, dtype=float)
        if z0.ndim == 0:
            if not np.isfinite(z0) or z0 <= 0:
                raise ValueError("z0 must be finite and positive")
            self.z0 = float(z0)
        elif z0.shape == (self.s_parameters.shape[1],):
            if not np.all(np.isfinite(z0)) or np.any(z0 <= 0):
                raise ValueError("per-port z0 values must be finite and positive")
            self.z0 = z0.copy()
        else:
            raise ValueError("z0 must be scalar or contain one value per port")
        for port in self.bands_by_port:
            if not 0 <= port < self.s_parameters.shape[1]:
                raise ValueError(f"invalid port index {port}")
        n_ports = self.s_parameters.shape[1]
        for target in self.isolation_targets:
            if not 0 <= target.source_port < n_ports or not 0 <= target.destination_port < n_ports:
                raise ValueError("isolation target port outside the S-parameter matrix")
            if target.source_port == target.destination_port:
                raise ValueError("isolation target requires two distinct ports")
            if target.weight < 0:
                raise ValueError("isolation target weight must be non-negative")
            if not 0.0 <= target.average_weight <= 1.0:
                raise ValueError("isolation target average_weight must be between 0 and 1")


@dataclass
class Candidate:
    elements: list[Element]
    score_db: float = float("-inf")
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class S2PModel:
    name: str
    frequencies_hz: np.ndarray
    s_parameters: np.ndarray
    z0: float = 50.0
    tolerance: float = 0.0
    kind: Literal["L", "C"] | None = None
    nominal_value: float | None = None
    tempco_ppm_per_c: float | None = None
    systematic_bias_pct: float | None = None
    environment_provenance: str = ""

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("S2PModel tolerance must be non-negative")
        if (self.kind is None) != (self.nominal_value is None):
            raise ValueError("S2PModel kind and nominal_value must be provided together")
        if self.nominal_value is not None and self.nominal_value <= 0:
            raise ValueError("S2PModel nominal_value must be positive")
        if self.tempco_ppm_per_c is not None and not np.isfinite(self.tempco_ppm_per_c):
            raise ValueError("S2PModel tempco_ppm_per_c must be finite")
        if self.systematic_bias_pct is not None and (
            not np.isfinite(self.systematic_bias_pct) or self.systematic_bias_pct <= -100.0
        ):
            raise ValueError("S2PModel systematic_bias_pct must keep the component value positive")

    def at(self, frequency_hz: float) -> np.ndarray:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        data = np.asarray(self.s_parameters, dtype=complex)
        if data.shape != (len(frequencies), 2, 2):
            raise ValueError("S2PModel requires shape (frequency, 2, 2)")
        out = np.empty((2, 2), dtype=complex)
        for row in range(2):
            for col in range(2):
                out[row, col] = np.interp(frequency_hz, frequencies, data[:, row, col].real) + 1j * np.interp(frequency_hz, frequencies, data[:, row, col].imag)
        return out


@dataclass(frozen=True)
class LumpedModel:
    name: str
    kind: Literal["L", "C", "R"]
    value: float
    tolerance: float = 0.0
    q: float | None = None
    esr: float = 0.0
    q_reference_hz: float | None = None


@dataclass(frozen=True)
class Branch:
    name: str
    node_a: str
    node_b: str | None
    model: LumpedModel | S2PModel | TransmissionLineModel | TransmissionLineStubModel | MicrostripLineModel | MicrostripStubModel


@dataclass(frozen=True)
class CircuitTopology:
    external_nodes: tuple[str, ...]
    branches: tuple[Branch, ...]
    dut_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.external_nodes)) != len(self.external_nodes):
            raise ValueError("external node names must be unique")
        if self.dut_nodes and len(self.dut_nodes) != len(self.external_nodes):
            raise ValueError("dut_nodes must have one node per external port")
        if self.dut_nodes and len(set(self.dut_nodes)) != len(self.dut_nodes):
            raise ValueError("DUT node names must be unique")
        if self.dut_nodes:
            for index, node in enumerate(self.dut_nodes):
                if node in self.external_nodes and node != self.external_nodes[index]:
                    raise ValueError("a DUT port may alias only its corresponding external port")
        for branch in self.branches:
            if branch.node_a == branch.node_b:
                raise ValueError(f"branch {branch.name} connects a node to itself")
