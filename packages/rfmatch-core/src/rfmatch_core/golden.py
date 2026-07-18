from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEGACY_REQUIRED_COLUMNS = {
    "frequency_hz",
    "port",
    "s11_real",
    "s11_imag",
    "total_efficiency",
}
MATRIX_REQUIRED_COLUMNS = {
    "frequency_hz",
    "source_port",
    "destination_port",
    "s_real",
    "s_imag",
}


@dataclass(frozen=True)
class GoldenPoint:
    """One complex S-parameter sample and optional driven-port power metrics.

    Ports are zero-based internally.  ``s_parameter`` follows the standard
    S_ij convention: destination/output port i, source/driven port j.
    """

    frequency_hz: float
    source_port: int
    destination_port: int
    s_parameter: complex
    total_efficiency: float | None = None
    component_loss: float | None = None

    @property
    def port(self) -> int:
        """Legacy alias for diagonal-only callers."""
        return self.source_port

    @property
    def s11(self) -> complex:
        """Legacy alias retained for the original diagonal CSV format."""
        return self.s_parameter


@dataclass
class GoldenComparison:
    rows: list[dict]
    maximum_s_error: float
    maximum_efficiency_error: float | None
    maximum_component_loss_error: float | None
    passed: bool


@dataclass(frozen=True)
class GoldenDatasetSummary:
    rows: int
    ports: tuple[int, ...]
    s_parameter_pairs: tuple[tuple[int, int], ...]
    frequency_min_hz: float
    frequency_max_hz: float
    includes_complete_s_matrix: bool
    includes_total_efficiency: bool
    includes_component_loss: bool


def _optional_float(row: dict[str, str], name: str) -> float | None:
    value = (row.get(name) or "").strip()
    return float(value) if value else None


def _one_based_port(row: dict[str, str], name: str) -> int:
    port = int(row[name])
    if port < 1:
        raise ValueError(f"golden CSV {name} must be a 1-based positive integer")
    return port - 1


def validate_golden_points(points: list[GoldenPoint]) -> GoldenDatasetSummary:
    """Validate an Optenni export before it is accepted as a regression oracle."""
    if not points:
        raise ValueError("golden CSV contains no data rows")

    seen: set[tuple[float, int, int]] = set()
    last_frequency_by_pair: dict[tuple[int, int], float] = {}
    efficiency_by_excitation: dict[tuple[float, int], float] = {}
    loss_by_excitation: dict[tuple[float, int], float] = {}
    ports: set[int] = set()
    pairs: set[tuple[int, int]] = set()

    for point in points:
        if point.source_port < 0 or point.destination_port < 0:
            raise ValueError("golden ports must be zero-based non-negative integers internally")
        values = [point.frequency_hz, point.s_parameter.real, point.s_parameter.imag]
        if point.total_efficiency is not None:
            values.append(point.total_efficiency)
        if point.component_loss is not None:
            values.append(point.component_loss)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("golden CSV contains a non-finite value")
        if point.frequency_hz <= 0:
            raise ValueError("golden CSV frequencies must be positive")
        if point.total_efficiency is not None and not 0.0 <= point.total_efficiency <= 1.0:
            raise ValueError("golden CSV total_efficiency must be between 0 and 1")
        if point.component_loss is not None and not 0.0 <= point.component_loss <= 1.0:
            raise ValueError("golden CSV component_loss must be between 0 and 1")

        pair = (point.source_port, point.destination_port)
        key = (point.frequency_hz, *pair)
        if key in seen:
            raise ValueError(
                "duplicate golden row for frequency "
                f"{point.frequency_hz} Hz and S{point.destination_port + 1}{point.source_port + 1}"
            )
        previous = last_frequency_by_pair.get(pair)
        if previous is not None and point.frequency_hz <= previous:
            raise ValueError(
                "golden frequencies must increase strictly within "
                f"S{point.destination_port + 1}{point.source_port + 1}"
            )

        excitation = (point.frequency_hz, point.source_port)
        if point.total_efficiency is not None:
            previous_efficiency = efficiency_by_excitation.get(excitation)
            if previous_efficiency is not None and not np.isclose(
                previous_efficiency, point.total_efficiency, rtol=0.0, atol=1e-15
            ):
                raise ValueError("inconsistent total_efficiency for the same frequency and source_port")
            efficiency_by_excitation[excitation] = point.total_efficiency
        if point.component_loss is not None:
            previous_loss = loss_by_excitation.get(excitation)
            if previous_loss is not None and not np.isclose(
                previous_loss, point.component_loss, rtol=0.0, atol=1e-15
            ):
                raise ValueError("inconsistent component_loss for the same frequency and source_port")
            loss_by_excitation[excitation] = point.component_loss

        seen.add(key)
        ports.update(pair)
        pairs.add(pair)
        last_frequency_by_pair[pair] = point.frequency_hz

    frequencies = sorted({point.frequency_hz for point in points})
    expected_pairs = {(source, destination) for source in ports for destination in ports}
    complete = pairs == expected_pairs and all(
        (frequency, source, destination) in seen
        for frequency in frequencies
        for source, destination in expected_pairs
    )
    return GoldenDatasetSummary(
        rows=len(points),
        ports=tuple(sorted(ports)),
        s_parameter_pairs=tuple(sorted(pairs)),
        frequency_min_hz=frequencies[0],
        frequency_max_hz=frequencies[-1],
        includes_complete_s_matrix=complete,
        includes_total_efficiency=bool(efficiency_by_excitation),
        includes_component_loss=bool(loss_by_excitation),
    )


def load_golden_csv(path: str | Path) -> list[GoldenPoint]:
    """Load either the complete-matrix schema or the legacy diagonal schema."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        is_matrix = MATRIX_REQUIRED_COLUMNS <= columns
        is_legacy = LEGACY_REQUIRED_COLUMNS <= columns
        if not is_matrix and not is_legacy:
            matrix_missing = sorted(MATRIX_REQUIRED_COLUMNS - columns)
            legacy_missing = sorted(LEGACY_REQUIRED_COLUMNS - columns)
            raise ValueError(
                "golden CSV does not match a supported schema; "
                f"matrix columns missing: {matrix_missing}; legacy columns missing: {legacy_missing}"
            )

        points: list[GoldenPoint] = []
        for row in reader:
            frequency_hz = float(row["frequency_hz"])
            loss = _optional_float(row, "component_loss")
            if is_matrix:
                source = _one_based_port(row, "source_port")
                destination = _one_based_port(row, "destination_port")
                efficiency = _optional_float(row, "total_efficiency")
                s_parameter = complex(float(row["s_real"]), float(row["s_imag"]))
            else:
                source = destination = _one_based_port(row, "port")
                efficiency = float(row["total_efficiency"])
                s_parameter = complex(float(row["s11_real"]), float(row["s11_imag"]))
            points.append(
                GoldenPoint(
                    frequency_hz,
                    source,
                    destination,
                    s_parameter,
                    efficiency,
                    loss,
                )
            )
    validate_golden_points(points)
    return points


def compare_golden(
    points: list[GoldenPoint],
    frequencies_hz: np.ndarray,
    s_parameters: np.ndarray,
    total_efficiency: np.ndarray,
    component_loss: np.ndarray | None = None,
    s_tolerance: float = 1e-3,
    efficiency_tolerance: float = 1e-3,
    component_loss_tolerance: float = 1e-3,
) -> GoldenComparison:
    validate_golden_points(points)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    s_parameters = np.asarray(s_parameters, dtype=complex)
    total_efficiency = np.asarray(total_efficiency, dtype=float)
    if (
        s_parameters.ndim != 3
        or s_parameters.shape[0] != len(frequencies_hz)
        or s_parameters.shape[1] != s_parameters.shape[2]
    ):
        raise ValueError("computed s_parameters must have shape (frequency, port, port)")
    if total_efficiency.shape != s_parameters.shape[:2]:
        raise ValueError("computed total_efficiency must have shape (frequency, port)")
    if component_loss is not None:
        component_loss = np.asarray(component_loss, dtype=float)
        if component_loss.shape != s_parameters.shape[:2]:
            raise ValueError("computed component_loss must have shape (frequency, port)")

    rows: list[dict] = []
    s_errors: list[float] = []
    efficiency_errors: list[float] = []
    loss_errors: list[float] = []
    for point in points:
        if point.source_port >= s_parameters.shape[1] or point.destination_port >= s_parameters.shape[1]:
            raise ValueError(
                "golden S-parameter pair "
                f"S{point.destination_port + 1}{point.source_port + 1} is not present in computed data"
            )
        matches = np.flatnonzero(
            np.isclose(
                frequencies_hz,
                point.frequency_hz,
                rtol=0.0,
                atol=max(1e-6, abs(point.frequency_hz) * 1e-12),
            )
        )
        if not len(matches):
            raise ValueError(f"no computed frequency matches {point.frequency_hz} Hz")
        if len(matches) > 1:
            raise ValueError(f"multiple computed frequencies match {point.frequency_hz} Hz")
        fi = int(matches[0])
        predicted_s = s_parameters[fi, point.destination_port, point.source_port]
        s_error = float(abs(predicted_s - point.s_parameter))

        efficiency_error = None
        if point.total_efficiency is not None:
            predicted_efficiency = float(total_efficiency[fi, point.source_port])
            efficiency_error = abs(predicted_efficiency - point.total_efficiency)
            efficiency_errors.append(efficiency_error)

        loss_error = None
        if point.component_loss is not None:
            if component_loss is None:
                raise ValueError("golden data includes component_loss but computed data does not")
            predicted_loss = float(component_loss[fi, point.source_port])
            loss_error = abs(predicted_loss - point.component_loss)
            loss_errors.append(loss_error)

        s_errors.append(s_error)
        rows.append(
            {
                "frequency_hz": point.frequency_hz,
                "source_port": point.source_port,
                "destination_port": point.destination_port,
                "s_parameter": f"S{point.destination_port + 1}{point.source_port + 1}",
                "s_error": s_error,
                "efficiency_error": efficiency_error,
                "component_loss_error": loss_error,
            }
        )

    max_s = max(s_errors)
    max_efficiency = max(efficiency_errors) if efficiency_errors else None
    max_loss = max(loss_errors) if loss_errors else None
    passed = (
        max_s <= s_tolerance
        and (max_efficiency is None or max_efficiency <= efficiency_tolerance)
        and (max_loss is None or max_loss <= component_loss_tolerance)
    )
    return GoldenComparison(rows, max_s, max_efficiency, max_loss, passed)


def compare_physical_to_golden(problem, topology, golden_path: str | Path, **tolerances) -> GoldenComparison:
    """Run the physical sweep and compare every exported Optenni row."""
    from .physical import evaluate_physical_problem

    sweep = evaluate_physical_problem(problem, topology)
    return compare_golden(
        load_golden_csv(golden_path),
        problem.frequencies_hz,
        sweep.s_parameters,
        sweep.total_efficiency,
        sweep.component_loss,
        **tolerances,
    )
