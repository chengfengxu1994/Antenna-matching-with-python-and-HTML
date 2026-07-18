"""Manufacturable microstrip geometry models with dispersion and loss.

Equations follow Hammerstad/Jensen (IEEE MTT-S, 1980), Kirschning/Jansen
(Electronics Letters, 1982), and Wheeler's incremental-inductance loss rule.
The implementation is intentionally dependency-free beyond NumPy and is
cross-validated against the independent scikit-rf/Qucs implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .models import S2PModel


_C0 = 299_792_458.0
_MU0 = 1.25663706212e-6
_EPSILON0 = 8.8541878128e-12
_Z_FREE_SPACE = np.sqrt(_MU0 / _EPSILON0)
_DB_PER_NEPER = 20.0 / np.log(10.0)


@dataclass(frozen=True)
class PCBSubstrate:
    name: str
    relative_permittivity: float
    height_m: float
    loss_tangent: float = 0.0
    copper_thickness_m: float = 35e-6
    copper_resistivity_ohm_m: float = 1.68e-8
    copper_roughness_rms_m: float = 0.15e-6

    def __post_init__(self) -> None:
        if self.relative_permittivity <= 1.0:
            raise ValueError("microstrip relative_permittivity must exceed 1")
        if self.height_m <= 0:
            raise ValueError("microstrip substrate height must be positive")
        if self.loss_tangent < 0:
            raise ValueError("microstrip loss_tangent must be non-negative")
        if self.copper_thickness_m < 0:
            raise ValueError("copper thickness must be non-negative")
        if self.copper_resistivity_ohm_m <= 0:
            raise ValueError("copper resistivity must be positive")
        if self.copper_roughness_rms_m < 0:
            raise ValueError("copper roughness must be non-negative")


@dataclass(frozen=True)
class MicrostripProperties:
    frequency_hz: float
    characteristic_impedance_ohm: float
    effective_permittivity: float
    effective_width_m: float
    conductor_attenuation_np_per_m: float
    dielectric_attenuation_np_per_m: float
    phase_constant_rad_per_m: float
    skin_depth_m: float
    warnings: tuple[str, ...] = ()

    @property
    def attenuation_np_per_m(self) -> float:
        return self.conductor_attenuation_np_per_m + self.dielectric_attenuation_np_per_m

    @property
    def attenuation_db_per_m(self) -> float:
        return _DB_PER_NEPER * self.attenuation_np_per_m


@dataclass(frozen=True)
class MicrostripDesignRules:
    substrate: PCBSubstrate
    minimum_width_m: float
    maximum_width_m: float
    width_tolerance: float = 0.0
    length_tolerance: float = 0.0
    substrate_height_tolerance: float = 0.0
    relative_permittivity_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.minimum_width_m < self.maximum_width_m:
            raise ValueError("microstrip design-rule widths must be positive and increasing")
        for name, value in (
            ("width", self.width_tolerance),
            ("length", self.length_tolerance),
            ("substrate height", self.substrate_height_tolerance),
            ("relative permittivity", self.relative_permittivity_tolerance),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"microstrip {name} tolerance must be between 0 and 1")


@dataclass(frozen=True)
class MicrostripVariation:
    """One manufactured PCB draw, expressed as positive scale factors."""

    width_scale: float = 1.0
    length_scale: float = 1.0
    substrate_height_scale: float = 1.0
    relative_permittivity_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("width", self.width_scale),
            ("length", self.length_scale),
            ("substrate height", self.substrate_height_scale),
            ("relative permittivity", self.relative_permittivity_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"microstrip {name} scale must be finite and positive")


def _microstrip_variation(value: float | MicrostripVariation) -> MicrostripVariation:
    # Backward-compatible scalar variation now has the manufacturing meaning
    # used by Optenni's msWidthToler: trace width, not electrical length.
    if isinstance(value, MicrostripVariation):
        return value
    return MicrostripVariation(width_scale=float(value))


def _hammerstad_z_air(normalized_width: float) -> float:
    u = float(normalized_width)
    factor = 6.0 + (2.0 * np.pi - 6.0) * np.exp(-(30.666 / u) ** 0.7528)
    return float(
        _Z_FREE_SPACE / (2.0 * np.pi)
        * np.log(factor / u + np.sqrt(1.0 + (2.0 / u) ** 2))
    )


def _hammerstad_ab(normalized_width: float, relative_permittivity: float) -> tuple[float, float]:
    u, er = float(normalized_width), float(relative_permittivity)
    a = (
        1.0
        + np.log((u**4 + (u / 52.0) ** 2) / (u**4 + 0.432)) / 49.0
        + np.log(1.0 + (u / 18.1) ** 3) / 18.7
    )
    b = 0.564 * ((er - 0.9) / (er + 3.0)) ** 0.053
    return float(a), float(b)


def _hammerstad_effective_permittivity(normalized_width: float, relative_permittivity: float) -> float:
    a, b = _hammerstad_ab(normalized_width, relative_permittivity)
    er, u = float(relative_permittivity), float(normalized_width)
    return float((er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b))


def microstrip_quasi_static(substrate: PCBSubstrate, width_m: float) -> tuple[float, float, float]:
    """Return Hammerstad/Jensen Z0, effective permittivity, and effective width."""
    if width_m <= 0:
        raise ValueError("microstrip width must be positive")
    u = float(width_m / substrate.height_m)
    normalized_thickness = substrate.copper_thickness_m / substrate.height_m
    delta_u_air = 0.0
    if normalized_thickness > 0:
        delta_u_air = normalized_thickness / np.pi * np.log(
            1.0
            + 4.0 * np.e / normalized_thickness
            * np.tanh(np.sqrt(6.517 * u)) ** 2
        )
    delta_u_dielectric = delta_u_air * (
        1.0 + 1.0 / np.cosh(np.sqrt(substrate.relative_permittivity - 1.0))
    ) / 2.0
    u_air = u + delta_u_air
    u_dielectric = u + delta_u_dielectric
    z_air_dielectric_width = _hammerstad_z_air(u_dielectric)
    z_air_air_width = _hammerstad_z_air(u_air)
    base_effective = _hammerstad_effective_permittivity(
        u_dielectric, substrate.relative_permittivity
    )
    impedance = z_air_dielectric_width / np.sqrt(base_effective)
    effective_permittivity = base_effective * (
        z_air_air_width / z_air_dielectric_width
    ) ** 2
    return (
        float(impedance),
        float(effective_permittivity),
        float(u_dielectric * substrate.height_m),
    )


def _kirschning_effective_permittivity(
    normalized_width: float,
    normalized_frequency: float,
    relative_permittivity: float,
    quasi_static_effective_permittivity: float,
) -> float:
    u, fn, er, ee = (
        float(normalized_width), float(normalized_frequency),
        float(relative_permittivity), float(quasi_static_effective_permittivity),
    )
    p1 = 0.27488 + (0.6315 + 0.525 / (1.0 + 0.0157 * fn) ** 20) * u - 0.065683 * np.exp(-8.7513 * u)
    p2 = 0.33622 * (1.0 - np.exp(-0.03442 * er))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1.0 - np.exp(-(fn / 38.7) ** 4.97))
    p4 = 1.0 + 2.751 * (1.0 - np.exp(-(er / 15.916) ** 8))
    pf = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    return float(er - (er - ee) / (1.0 + pf))


def _kirschning_impedance(
    normalized_width: float,
    normalized_frequency: float,
    relative_permittivity: float,
    quasi_static_effective_permittivity: float,
    effective_permittivity: float,
    quasi_static_impedance: float,
) -> float:
    u, fn, er, ee0, ee, z0 = map(float, (
        normalized_width, normalized_frequency, relative_permittivity,
        quasi_static_effective_permittivity, effective_permittivity,
        quasi_static_impedance,
    ))
    r1 = min(0.03891 * er**1.4, 20.0)
    r2 = min(0.2671 * u**7, 20.0)
    r3 = 4.766 * np.exp(-3.228 * u**0.641)
    r4 = 0.016 + (0.0514 * er) ** 4.524
    r5 = (fn / 28.843) ** 12
    r6 = min(22.20 * u**1.92, 20.0)
    r7 = 1.206 - 0.3144 * np.exp(-r1) * (1.0 - np.exp(-r2))
    r8 = 1.0 + 1.275 * (1.0 - np.exp(-0.004625 * r3 * er**1.674 * (fn / 18.365) ** 2.745))
    r9 = (
        5.086 * r4 * r5 / (0.3838 + 0.386 * r4)
        * np.exp(-r6) / (1.0 + 1.2992 * r5)
        * (er - 1.0) ** 6 / (1.0 + 10.0 * (er - 1.0) ** 6)
    )
    r10 = 0.00044 * er**2.136 + 0.0184
    r11 = (fn / 19.47) ** 6 / (1.0 + 0.0962 * (fn / 19.47) ** 6)
    r12 = 1.0 / (1.0 + 0.00245 * u**2)
    r13 = 0.9408 * ee**r8 - 0.9603
    r14 = (0.9408 - r9) * ee0**r8 - 0.9603
    r15 = 0.707 * r10 * (fn / 12.3) ** 1.097
    r16 = 1.0 + 0.0503 * er**2 * r11 * (1.0 - np.exp(-(u / 15.0) ** 6))
    r17 = r7 * (
        1.0 - 1.1241 * r12 / r16
        * np.exp(-0.026 * fn**1.15656 - r15)
    )
    return float(z0 * (r13 / r14) ** r17)


def microstrip_properties(substrate: PCBSubstrate, width_m: float, frequency_hz: float) -> MicrostripProperties:
    if frequency_hz <= 0:
        raise ValueError("microstrip frequency must be positive")
    quasi_z, quasi_ee, effective_width = microstrip_quasi_static(substrate, width_m)
    normalized_width = effective_width / substrate.height_m
    normalized_frequency = frequency_hz * substrate.height_m * 1e-6  # GHz-mm
    effective_permittivity = _kirschning_effective_permittivity(
        normalized_width, normalized_frequency,
        substrate.relative_permittivity, quasi_ee,
    )
    impedance = _kirschning_impedance(
        normalized_width, normalized_frequency,
        substrate.relative_permittivity, quasi_ee,
        effective_permittivity, quasi_z,
    )
    skin_depth = np.sqrt(
        substrate.copper_resistivity_ohm_m / (np.pi * frequency_hz * _MU0)
    )
    conductor = 0.0
    if substrate.copper_thickness_m > 0:
        surface_resistance = np.sqrt(
            np.pi * frequency_hz * _MU0 * substrate.copper_resistivity_ohm_m
        )
        current_factor = np.exp(-1.2 * (impedance / _Z_FREE_SPACE) ** 0.7)
        roughness_factor = 1.0 + 2.0 / np.pi * np.arctan(
            1.4 * (substrate.copper_roughness_rms_m / skin_depth) ** 2
        )
        conductor = surface_resistance / (impedance * width_m) * current_factor * roughness_factor
    dielectric = (
        np.pi * substrate.relative_permittivity
        / (substrate.relative_permittivity - 1.0)
        * (effective_permittivity - 1.0) / np.sqrt(effective_permittivity)
        * substrate.loss_tangent * frequency_hz / _C0
    )
    warnings = []
    ratio = width_m / substrate.height_m
    if not 0.1 <= ratio <= 100.0:
        warnings.append("width_to_height_outside_recommended_range")
    if substrate.copper_thickness_m and substrate.copper_thickness_m < 3.0 * skin_depth:
        warnings.append("copper_thinner_than_three_skin_depths")
    return MicrostripProperties(
        float(frequency_hz), impedance, effective_permittivity, effective_width,
        float(conductor), float(dielectric),
        float(2.0 * np.pi * frequency_hz * np.sqrt(effective_permittivity) / _C0),
        float(skin_depth), tuple(warnings),
    )


def solve_microstrip_width(
    substrate: PCBSubstrate,
    target_impedance_ohm: float,
    frequency_hz: float,
    minimum_width_m: float,
    maximum_width_m: float,
    *,
    relative_tolerance: float = 1e-9,
    maximum_iterations: int = 100,
) -> float:
    """Invert the dispersive impedance model using a monotonic bisection."""
    if target_impedance_ohm <= 0:
        raise ValueError("target microstrip impedance must be positive")
    if not 0 < minimum_width_m < maximum_width_m:
        raise ValueError("microstrip width bounds must be positive and increasing")
    narrow_z = microstrip_properties(substrate, minimum_width_m, frequency_hz).characteristic_impedance_ohm
    wide_z = microstrip_properties(substrate, maximum_width_m, frequency_hz).characteristic_impedance_ohm
    if not wide_z <= target_impedance_ohm <= narrow_z:
        raise ValueError(
            f"target impedance {target_impedance_ohm:g} ohm is outside the manufacturable "
            f"range {wide_z:g}..{narrow_z:g} ohm"
        )
    low, high = float(minimum_width_m), float(maximum_width_m)
    for _ in range(maximum_iterations):
        middle = 0.5 * (low + high)
        impedance = microstrip_properties(substrate, middle, frequency_hz).characteristic_impedance_ohm
        if abs(impedance - target_impedance_ohm) <= relative_tolerance * target_impedance_ohm:
            return middle
        if impedance > target_impedance_ohm:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@dataclass(frozen=True)
class MicrostripLineModel:
    name: str
    substrate: PCBSubstrate
    width_m: float
    length_m: float
    tolerance: float = 0.0
    design_reference_frequency_hz: float | None = None
    design_electrical_length_deg: float | None = None
    length_tolerance: float = 0.0
    substrate_height_tolerance: float = 0.0
    relative_permittivity_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.length_m <= 0:
            raise ValueError("microstrip width and length must be positive")
        for name, value in (
            ("width", self.tolerance),
            ("length", self.length_tolerance),
            ("substrate height", self.substrate_height_tolerance),
            ("relative permittivity", self.relative_permittivity_tolerance),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"microstrip {name} tolerance must be between 0 and 1")
        if self.design_reference_frequency_hz is not None and self.design_reference_frequency_hz <= 0:
            raise ValueError("microstrip design reference frequency must be positive")
        if self.design_electrical_length_deg is not None and self.design_electrical_length_deg <= 0:
            raise ValueError("microstrip design electrical length must be positive")

    @property
    def width_tolerance(self) -> float:
        return self.tolerance

    def properties_at(
        self, frequency_hz: float,
        variation: float | MicrostripVariation = 1.0,
    ) -> MicrostripProperties:
        varied = _microstrip_variation(variation)
        substrate = replace(
            self.substrate,
            relative_permittivity=self.substrate.relative_permittivity * varied.relative_permittivity_scale,
            height_m=self.substrate.height_m * varied.substrate_height_scale,
        )
        return microstrip_properties(substrate, self.width_m * varied.width_scale, frequency_hz)

    def propagation_length(
        self, frequency_hz: float,
        variation: float | MicrostripVariation = 1.0,
    ) -> complex:
        varied = _microstrip_variation(variation)
        properties = self.properties_at(frequency_hz, varied)
        length = self.length_m * varied.length_scale
        return complex(properties.attenuation_np_per_m * length, properties.phase_constant_rad_per_m * length)

    def admittance(self, frequency_hz: float, variation: float | MicrostripVariation = 1.0) -> np.ndarray:
        varied = _microstrip_variation(variation)
        properties = self.properties_at(frequency_hz, varied)
        propagation = self.propagation_length(frequency_hz, varied)
        sine = np.sinh(propagation)
        if abs(sine) < 1e-15:
            raise ValueError("microstrip is singular at an integer half wavelength")
        diagonal = np.cosh(propagation) / (properties.characteristic_impedance_ohm * sine)
        transfer = -1.0 / (properties.characteristic_impedance_ohm * sine)
        return np.asarray([[diagonal, transfer], [transfer, diagonal]], dtype=complex)

    def s_parameters(self, frequency_hz: float, reference_impedance_ohm: float = 50.0, variation: float | MicrostripVariation = 1.0) -> np.ndarray:
        varied = _microstrip_variation(variation)
        propagation = self.propagation_length(frequency_hz, varied)
        impedance = self.properties_at(frequency_hz, varied).characteristic_impedance_ohm
        a = d = np.cosh(propagation)
        b = impedance * np.sinh(propagation)
        c = np.sinh(propagation) / impedance
        denominator = a + b / reference_impedance_ohm + c * reference_impedance_ohm + d
        return np.asarray([
            [(a + b / reference_impedance_ohm - c * reference_impedance_ohm - d) / denominator, 2.0 / denominator],
            [2.0 / denominator, (-a + b / reference_impedance_ohm - c * reference_impedance_ohm + d) / denominator],
        ], dtype=complex)

    def as_s2p_model(self, frequencies_hz, reference_impedance_ohm: float = 50.0) -> S2PModel:
        frequencies = np.asarray(frequencies_hz, dtype=float)
        return S2PModel(
            self.name, frequencies,
            np.asarray([self.s_parameters(float(f), reference_impedance_ohm) for f in frequencies]),
            reference_impedance_ohm,
        )

    @classmethod
    def from_electrical_design(
        cls,
        name: str,
        substrate: PCBSubstrate,
        target_impedance_ohm: float,
        electrical_length_deg: float,
        reference_frequency_hz: float,
        minimum_width_m: float,
        maximum_width_m: float,
        width_tolerance: float = 0.0,
        length_tolerance: float = 0.0,
        substrate_height_tolerance: float = 0.0,
        relative_permittivity_tolerance: float = 0.0,
    ) -> "MicrostripLineModel":
        if electrical_length_deg <= 0:
            raise ValueError("microstrip electrical length must be positive")
        width = solve_microstrip_width(
            substrate, target_impedance_ohm, reference_frequency_hz,
            minimum_width_m, maximum_width_m,
        )
        beta = microstrip_properties(substrate, width, reference_frequency_hz).phase_constant_rad_per_m
        length = np.deg2rad(electrical_length_deg) / beta
        return cls(
            name, substrate, width, float(length), width_tolerance,
            float(reference_frequency_hz), float(electrical_length_deg),
            length_tolerance, substrate_height_tolerance,
            relative_permittivity_tolerance,
        )


@dataclass(frozen=True)
class MicrostripStubModel:
    line: MicrostripLineModel
    termination: Literal["open", "short"] = "open"

    def __post_init__(self) -> None:
        if self.termination not in {"open", "short"}:
            raise ValueError("microstrip stub termination must be open or short")

    @property
    def name(self) -> str:
        return self.line.name

    @property
    def tolerance(self) -> float:
        return self.line.tolerance

    def input_admittance(self, frequency_hz: float, variation: float | MicrostripVariation = 1.0) -> complex:
        varied = _microstrip_variation(variation)
        tangent = np.tanh(self.line.propagation_length(frequency_hz, varied))
        impedance = self.line.properties_at(frequency_hz, varied).characteristic_impedance_ohm
        if self.termination == "open":
            return complex(tangent / impedance)
        if abs(tangent) < 1e-15:
            raise ValueError("shorted microstrip stub is singular at an integer half wavelength")
        return complex(1.0 / (impedance * tangent))
