from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SwitchGoldenPoint:
    configuration: str
    frequency_hz: float
    s11_db: float
    total_efficiency: float | None = None


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _find_column(columns: dict[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    return None


def load_switch_export_csv(
    path: str | Path, *, default_configuration: str | None = None
) -> list[SwitchGoldenPoint]:
    """Load a canonical or common Optenni curve export.

    Frequency may be in Hz/MHz/GHz. S11 must be in dB. Total efficiency is
    optional and may be linear, percent, or dB.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        lines = handle.readlines()
        metadata = [line.strip() for line in lines if line.lstrip().startswith("%")]
        data_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("%")]
        if not data_lines:
            raise ValueError("switch CSV contains no header or data rows")
        try:
            delimiter = csv.Sniffer().sniff("".join(data_lines[:4]), delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.DictReader(data_lines, delimiter=delimiter)
        columns = {_column_key(name): name for name in (reader.fieldnames or [])}
        configuration_column = _find_column(
            columns, ("configuration", "config", "set", "state_set")
        )
        frequency_columns = (
            (_find_column(columns, ("frequency_hz", "freq_hz", "frequency")), 1.0),
            (_find_column(columns, ("frequency_mhz", "freq_mhz")), 1e6),
            (_find_column(columns, ("frequency_ghz", "freq_ghz")), 1e9),
        )
        frequency_column, frequency_scale = next(
            ((name, scale) for name, scale in frequency_columns if name), (None, None)
        )
        s11_column = _find_column(
            columns, ("s11_db", "s_11_db", "return_loss_db", "reflection_db", "s11")
        )
        linear_efficiency = _find_column(
            columns, ("total_efficiency", "total_efficiency_linear", "efficiency")
        )
        percent_efficiency = _find_column(
            columns, ("total_efficiency_pct", "total_efficiency_percent", "efficiency_pct")
        )
        db_efficiency = _find_column(
            columns, ("total_efficiency_db", "efficiency_db")
        )
        generic_efficiency_is_db = bool(
            linear_efficiency
            and any("y axis" in line.lower() and "[db]" in line.lower() for line in metadata)
        )
        if not frequency_column or not s11_column:
            raise ValueError(
                "switch CSV requires frequency_hz/frequency_mhz/frequency_ghz and s11_db"
            )
        if not configuration_column and not default_configuration:
            raise ValueError(
                "switch CSV requires a configuration column or --configuration"
            )

        points = []
        for row_number, row in enumerate(reader, start=2):
            configuration = (
                (row.get(configuration_column) or "").strip()
                if configuration_column
                else str(default_configuration)
            )
            if not configuration:
                raise ValueError(f"row {row_number} has an empty configuration")
            frequency_hz = float(row[frequency_column]) * float(frequency_scale)
            s11_db = float(row[s11_column])
            efficiency = None
            if linear_efficiency and (row.get(linear_efficiency) or "").strip():
                raw_efficiency = float(row[linear_efficiency])
                efficiency = (
                    10.0 ** (raw_efficiency / 10.0)
                    if generic_efficiency_is_db
                    else raw_efficiency
                )
            elif percent_efficiency and (row.get(percent_efficiency) or "").strip():
                efficiency = float(row[percent_efficiency]) / 100.0
            elif db_efficiency and (row.get(db_efficiency) or "").strip():
                efficiency = 10.0 ** (float(row[db_efficiency]) / 10.0)
            values = [frequency_hz, s11_db]
            if efficiency is not None:
                values.append(efficiency)
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"row {row_number} contains a non-finite value")
            if frequency_hz <= 0:
                raise ValueError(f"row {row_number} frequency must be positive")
            if efficiency is not None and not 0.0 <= efficiency <= 1.0:
                raise ValueError(f"row {row_number} total efficiency must be between 0 and 1")
            points.append(SwitchGoldenPoint(configuration, frequency_hz, s11_db, efficiency))
    if not points:
        raise ValueError("switch CSV contains no data rows")
    seen = set()
    for point in points:
        key = (point.configuration, point.frequency_hz)
        if key in seen:
            raise ValueError(
                f"duplicate switch CSV row for {point.configuration} at {point.frequency_hz} Hz"
            )
        seen.add(key)
    return points


def compare_switch_export(
    points: list[SwitchGoldenPoint],
    configuration_curves: list[dict],
    *,
    s11_tolerance_db: float = 0.05,
    efficiency_tolerance: float = 0.005,
) -> dict:
    """Compare exported rows to stored full-physical switch curves."""
    references = {item["configuration"]: item for item in configuration_curves}
    rows = []
    for point in points:
        if point.configuration not in references:
            raise ValueError(f"unknown configuration {point.configuration!r}")
        reference = references[point.configuration]
        frequencies = np.asarray(reference["frequency_hz"], dtype=float)
        if len(frequencies) < 2 or np.any(np.diff(frequencies) <= 0):
            raise ValueError(f"reference frequencies for {point.configuration} are not increasing")
        margin = max(1e-6, abs(point.frequency_hz) * 1e-12)
        if point.frequency_hz < frequencies[0] - margin or point.frequency_hz > frequencies[-1] + margin:
            raise ValueError(
                f"{point.configuration} frequency {point.frequency_hz} Hz is outside the reference range"
            )
        predicted_s11 = float(np.interp(
            point.frequency_hz, frequencies, np.asarray(reference["s11_db"], dtype=float)
        ))
        s11_error = abs(predicted_s11 - point.s11_db)
        efficiency_error = None
        predicted_efficiency = None
        if point.total_efficiency is not None:
            predicted_efficiency = float(np.interp(
                point.frequency_hz,
                frequencies,
                np.asarray(reference["total_efficiency"], dtype=float),
            ))
            efficiency_error = abs(predicted_efficiency - point.total_efficiency)
        rows.append({
            "configuration": point.configuration,
            "frequency_hz": point.frequency_hz,
            "export_s11_db": point.s11_db,
            "rfmatch_s11_db": predicted_s11,
            "s11_error_db": s11_error,
            "export_total_efficiency": point.total_efficiency,
            "rfmatch_total_efficiency": predicted_efficiency,
            "efficiency_error": efficiency_error,
        })
    s11_errors = np.asarray([row["s11_error_db"] for row in rows], dtype=float)
    efficiency_errors = np.asarray([
        row["efficiency_error"] for row in rows if row["efficiency_error"] is not None
    ], dtype=float)
    maximum_efficiency_error = (
        float(np.max(efficiency_errors)) if len(efficiency_errors) else None
    )
    passed = bool(
        float(np.max(s11_errors)) <= s11_tolerance_db
        and (
            maximum_efficiency_error is None
            or maximum_efficiency_error <= efficiency_tolerance
        )
    )
    return {
        "passed": passed,
        "rows": len(rows),
        "configurations": sorted({point.configuration for point in points}),
        "maximum_s11_error_db": float(np.max(s11_errors)),
        "rms_s11_error_db": float(np.sqrt(np.mean(s11_errors ** 2))),
        "maximum_efficiency_error": maximum_efficiency_error,
        "rms_efficiency_error": (
            float(np.sqrt(np.mean(efficiency_errors ** 2)))
            if len(efficiency_errors) else None
        ),
        "s11_tolerance_db": s11_tolerance_db,
        "efficiency_tolerance": efficiency_tolerance,
        "details": rows,
    }
