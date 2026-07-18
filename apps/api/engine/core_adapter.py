"""Explicit boundary between the legacy Web engine and rfmatch-core.

Only lossless data conversion belongs here. Numerical behavior is migrated in
small steps and cross-validated before any product endpoint switches engines.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from rfmatch_core.models import Band, Problem
from rfmatch_core.touchstone import Touchstone

from .efficiency_data import EfficiencyData
from .touchstone import TouchstoneData


def to_core_touchstone(data: TouchstoneData) -> Touchstone:
    matrices = np.asarray([data.get_s_matrix(index) for index in range(len(data.frequencies))], dtype=complex)
    if data.port_impedances:
        impedances = np.asarray([complex(value) for value in data.port_impedances], dtype=complex)
        if np.any(np.abs(impedances.imag) > 1e-12):
            raise ValueError("complex per-port reference impedances are not supported")
        real_impedances = impedances.real.astype(float)
        z0 = (
            float(real_impedances[0])
            if np.allclose(real_impedances, real_impedances[0])
            else real_impedances
        )
    else:
        z0 = float(data.reference_resistance)
    return Touchstone(np.asarray(data.frequencies, dtype=float), matrices, z0)


def to_core_problem(
    data: TouchstoneData,
    bands_by_port_hz: dict[int, Iterable[tuple[float, float]]],
    radiation_efficiency: dict[int, EfficiencyData] | None = None,
) -> Problem:
    converted = to_core_touchstone(data)
    efficiency = {
        port: values.get_efficiency_array(converted.frequencies_hz)
        for port, values in (radiation_efficiency or {}).items()
    }
    return Problem(
        converted.frequencies_hz,
        converted.s_parameters,
        {port: [Band(start, stop) for start, stop in bands] for port, bands in bands_by_port_hz.items()},
        converted.z0,
        efficiency,
    )
