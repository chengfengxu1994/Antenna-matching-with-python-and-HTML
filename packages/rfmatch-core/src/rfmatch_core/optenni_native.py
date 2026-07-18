"""Parse and replay native numeric exports from Optenni Lab.

Optenni's plot text export contains scalar dB curves, while its circuit-view
Touchstone export contains the complex two-port matching network.  Together
with the original one-port DUT they form an independent, point-by-point
cross-software oracle instead of a comparison against rounded UI values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .touchstone import Touchstone


@dataclass(frozen=True)
class OptenniPlotExport:
    frequencies_hz: np.ndarray
    s11_db: np.ndarray
    total_efficiency_db: np.ndarray
    metadata: dict[str, str]


@dataclass(frozen=True)
class OptenniNetworkReplay:
    frequencies_hz: np.ndarray
    input_reflection: np.ndarray
    s11_db: np.ndarray
    total_efficiency: np.ndarray
    total_efficiency_db: np.ndarray
    source_port: int
    load_port: int


@dataclass(frozen=True)
class OptenniNativeComparison:
    points: int
    source_port: int
    load_port: int
    maximum_s11_error_db: float
    rms_s11_error_db: float
    maximum_efficiency_error_db: float
    rms_efficiency_error_db: float


def load_optenni_plot_export(path: str | Path) -> OptenniPlotExport:
    """Load Optenni's tab-delimited nominal plot export.

    The parser intentionally requires the native ``Frequency [GHz]``, ``S11``
    and ``Total efficiency`` columns.  Tolerance exports with duplicate column
    names belong to :func:`load_optenni_tolerance_export` instead.
    """
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%"):
            key, separator, value = line[1:].partition(":")
            if separator:
                metadata[key.strip()] = value.strip().strip('"')
            continue
        data_lines.append(raw)
    if not data_lines:
        raise ValueError("Optenni plot export contains no tabular data")

    reader = csv.DictReader(data_lines, delimiter="\t")
    required = {"Frequency [GHz]", "S11", "Total efficiency"}
    columns = set(reader.fieldnames or [])
    if not required <= columns:
        raise ValueError(
            "Optenni plot export is missing columns: "
            + ", ".join(sorted(required - columns))
        )
    if len(reader.fieldnames or []) != len(columns):
        raise ValueError(
            "duplicate Optenni plot columns require the tolerance-export parser"
        )
    rows = list(reader)
    if not rows:
        raise ValueError("Optenni plot export contains no frequency rows")
    try:
        frequencies_hz = np.asarray(
            [float(row["Frequency [GHz]"]) * 1e9 for row in rows], dtype=float
        )
        s11_db = np.asarray([float(row["S11"]) for row in rows], dtype=float)
        efficiency_db = np.asarray(
            [float(row["Total efficiency"]) for row in rows], dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Optenni plot export contains non-numeric data") from exc
    if not all(np.all(np.isfinite(values)) for values in (
        frequencies_hz, s11_db, efficiency_db
    )):
        raise ValueError("Optenni plot export contains non-finite values")
    if np.any(frequencies_hz <= 0) or np.any(np.diff(frequencies_hz) <= 0):
        raise ValueError("Optenni plot frequencies must be positive and increasing")
    if np.any(s11_db > 1e-9) or np.any(efficiency_db > 1e-9):
        raise ValueError("Optenni passive S11 and efficiency dB values cannot exceed 0")
    return OptenniPlotExport(frequencies_hz, s11_db, efficiency_db, metadata)


def replay_one_port_dut_through_network(
    dut: Touchstone,
    network: Touchstone,
    *,
    source_port: int = 0,
    load_port: int = 1,
) -> OptenniNetworkReplay:
    """Terminate a two-port matching network in a one-port DUT.

    ``source_port`` and ``load_port`` refer to the matching-network Touchstone
    ports.  Total efficiency is the transducer power delivered to the DUT.  It
    therefore includes mismatch and matching-network dissipation and assumes
    the DUT's accepted power is useful, matching Optenni's no-radiation-file
    Quick Start definition.
    """
    if dut.s_parameters.shape[1:] != (1, 1):
        raise ValueError("Optenni native replay requires a one-port DUT")
    if network.s_parameters.shape[1:] != (2, 2):
        raise ValueError("Optenni native replay requires a two-port network")
    if {source_port, load_port} != {0, 1}:
        raise ValueError("source_port and load_port must be distinct 0/1 ports")
    if len(dut.frequencies_hz) != len(network.frequencies_hz) or not np.allclose(
        dut.frequencies_hz,
        network.frequencies_hz,
        rtol=0.0,
        atol=np.maximum(1e-6, np.abs(dut.frequencies_hz) * 1e-12),
    ):
        raise ValueError("DUT and matching-network frequency grids must match")

    gamma_load = dut.s_parameters[:, 0, 0]
    s = network.s_parameters
    denominator = 1.0 - s[:, load_port, load_port] * gamma_load
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("matching-network termination is singular")
    gamma_input = (
        s[:, source_port, source_port]
        + s[:, source_port, load_port]
        * gamma_load
        * s[:, load_port, source_port]
        / denominator
    )
    total_efficiency = (
        np.abs(s[:, load_port, source_port]) ** 2
        * np.maximum(0.0, 1.0 - np.abs(gamma_load) ** 2)
        / np.abs(denominator) ** 2
    )
    if np.any(total_efficiency < -1e-12) or np.any(total_efficiency > 1.0 + 1e-8):
        raise ValueError("replayed total efficiency violates passive power bounds")
    total_efficiency = np.clip(total_efficiency, 0.0, 1.0)
    return OptenniNetworkReplay(
        dut.frequencies_hz.copy(),
        gamma_input,
        20.0 * np.log10(np.maximum(np.abs(gamma_input), 1e-15)),
        total_efficiency,
        10.0 * np.log10(np.maximum(total_efficiency, 1e-15)),
        source_port,
        load_port,
    )


def compare_optenni_native_plot(
    plot: OptenniPlotExport,
    replay: OptenniNetworkReplay,
) -> OptenniNativeComparison:
    """Compare every native Optenni row against a complex network replay."""
    if len(plot.frequencies_hz) != len(replay.frequencies_hz) or not np.allclose(
        plot.frequencies_hz,
        replay.frequencies_hz,
        rtol=0.0,
        atol=np.maximum(1e-6, np.abs(plot.frequencies_hz) * 1e-12),
    ):
        raise ValueError("Optenni plot and replay frequency grids must match")
    s11_error = replay.s11_db - plot.s11_db
    efficiency_error = replay.total_efficiency_db - plot.total_efficiency_db
    return OptenniNativeComparison(
        len(plot.frequencies_hz),
        replay.source_port,
        replay.load_port,
        float(np.max(np.abs(s11_error))),
        float(np.sqrt(np.mean(s11_error ** 2))),
        float(np.max(np.abs(efficiency_error))),
        float(np.sqrt(np.mean(efficiency_error ** 2))),
    )
