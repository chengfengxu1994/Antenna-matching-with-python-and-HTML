"""Parameterized transmission-line and shunt-stub models for physical node graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .models import S2PModel


_DB_PER_NEPER = 20.0 / np.log(10.0)


@dataclass(frozen=True)
class TransmissionLineModel:
    """Uniform reciprocal line defined by electrical length at one frequency."""

    name: str
    characteristic_impedance_ohm: float
    electrical_length_deg: float
    reference_frequency_hz: float
    attenuation_db: float = 0.0
    loss_frequency_exponent: float = 0.5
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.characteristic_impedance_ohm <= 0:
            raise ValueError("characteristic_impedance_ohm must be positive")
        if self.reference_frequency_hz <= 0:
            raise ValueError("reference_frequency_hz must be positive")
        if self.electrical_length_deg <= 0:
            raise ValueError("electrical_length_deg must be positive")
        if self.attenuation_db < 0:
            raise ValueError("attenuation_db must be non-negative")
        if self.loss_frequency_exponent < 0:
            raise ValueError("loss_frequency_exponent must be non-negative")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    def propagation_length(self, frequency_hz: float, scale: float = 1.0) -> complex:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("transmission-line scale must be finite and positive")
        ratio = float(frequency_hz) / self.reference_frequency_hz
        alpha = self.attenuation_db / _DB_PER_NEPER * ratio ** self.loss_frequency_exponent * scale
        beta = np.deg2rad(self.electrical_length_deg) * ratio * scale
        return complex(alpha, beta)

    def abcd(self, frequency_hz: float, scale: float = 1.0) -> np.ndarray:
        propagation = self.propagation_length(frequency_hz, scale)
        cosine, sine = np.cosh(propagation), np.sinh(propagation)
        impedance = self.characteristic_impedance_ohm
        return np.asarray([
            [cosine, impedance * sine],
            [sine / impedance, cosine],
        ], dtype=complex)

    def admittance(self, frequency_hz: float, scale: float = 1.0) -> np.ndarray:
        propagation = self.propagation_length(frequency_hz, scale)
        sine = np.sinh(propagation)
        if abs(sine) < 1e-15:
            raise ValueError("transmission line is singular at an integer half wavelength")
        diagonal = np.cosh(propagation) / (self.characteristic_impedance_ohm * sine)
        transfer = -1.0 / (self.characteristic_impedance_ohm * sine)
        return np.asarray([[diagonal, transfer], [transfer, diagonal]], dtype=complex)

    def s_parameters(self, frequency_hz: float, reference_impedance_ohm: float = 50.0, scale: float = 1.0) -> np.ndarray:
        if reference_impedance_ohm <= 0:
            raise ValueError("reference_impedance_ohm must be positive")
        a, b, c, d = self.abcd(frequency_hz, scale).ravel()
        denominator = a + b / reference_impedance_ohm + c * reference_impedance_ohm + d
        determinant = a * d - b * c
        return np.asarray([
            [(a + b / reference_impedance_ohm - c * reference_impedance_ohm - d) / denominator,
             2.0 * determinant / denominator],
            [2.0 / denominator,
             (-a + b / reference_impedance_ohm - c * reference_impedance_ohm + d) / denominator],
        ], dtype=complex)

    def as_s2p_model(self, frequencies_hz, reference_impedance_ohm: float = 50.0) -> S2PModel:
        frequencies = np.asarray(frequencies_hz, dtype=float)
        matrices = np.asarray([
            self.s_parameters(float(frequency), reference_impedance_ohm)
            for frequency in frequencies
        ])
        return S2PModel(self.name, frequencies, matrices, reference_impedance_ohm)


@dataclass(frozen=True)
class TransmissionLineStubModel:
    """Open- or short-circuited shunt stub using the same line definition."""

    line: TransmissionLineModel
    termination: Literal["open", "short"] = "open"

    def __post_init__(self) -> None:
        if self.termination not in {"open", "short"}:
            raise ValueError("termination must be 'open' or 'short'")

    @property
    def name(self) -> str:
        return self.line.name

    @property
    def tolerance(self) -> float:
        return self.line.tolerance

    def input_admittance(self, frequency_hz: float, scale: float = 1.0) -> complex:
        tangent = np.tanh(self.line.propagation_length(frequency_hz, scale))
        impedance = self.line.characteristic_impedance_ohm
        if self.termination == "open":
            return complex(tangent / impedance)
        if abs(tangent) < 1e-15:
            raise ValueError("shorted stub is singular at an integer half wavelength")
        return complex(1.0 / (impedance * tangent))
