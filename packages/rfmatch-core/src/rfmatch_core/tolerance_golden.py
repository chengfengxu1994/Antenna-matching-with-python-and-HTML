from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OptenniToleranceExport:
    """Native Optenni tolerance-plot export with duplicate sample columns."""

    frequencies_hz: np.ndarray
    nominal_s11_db: np.ndarray
    s11_variants_db: np.ndarray
    nominal_total_efficiency: np.ndarray
    total_efficiency_variants: np.ndarray
    metadata: dict[str, str]

    @property
    def samples(self) -> int:
        return int(self.s11_variants_db.shape[1])


def _metadata_and_data_lines(path: Path) -> tuple[dict[str, str], list[str]]:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            body = stripped[1:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"')
        else:
            data_lines.append(line)
    return metadata, data_lines


def _frequency_scale(header: str) -> float:
    normalized = header.lower().replace(" ", "")
    if "[ghz]" in normalized or normalized.endswith("_ghz"):
        return 1e9
    if "[mhz]" in normalized or normalized.endswith("_mhz"):
        return 1e6
    if "[khz]" in normalized or normalized.endswith("_khz"):
        return 1e3
    if "[hz]" in normalized or normalized.endswith("_hz"):
        return 1.0
    raise ValueError(f"cannot determine frequency unit from {header!r}")


def load_optenni_tolerance_export(path: str | Path) -> OptenniToleranceExport:
    """Load Optenni's tab-delimited tolerance plot export.

    ``csv.DictReader`` cannot be used here because Optenni intentionally repeats
    the same tolerance-data header once for every Monte Carlo evaluation.
    """
    export_path = Path(path)
    metadata, data_lines = _metadata_and_data_lines(export_path)
    if len(data_lines) < 2:
        raise ValueError("Optenni tolerance export contains no data rows")
    rows = list(csv.reader(data_lines, delimiter="\t"))
    header = [value.strip().strip('"') for value in rows[0]]
    try:
        efficiency_index = header.index("Total efficiency")
    except ValueError as exc:
        raise ValueError("Optenni tolerance export has no Total efficiency column") from exc
    if len(header) < 5 or header[1] != "S11":
        raise ValueError("Optenni tolerance export requires Frequency and S11 columns")

    s11_sample_columns = list(range(2, efficiency_index))
    efficiency_sample_columns = list(range(efficiency_index + 1, len(header)))
    if not s11_sample_columns or len(s11_sample_columns) != len(efficiency_sample_columns):
        raise ValueError("Optenni S11 and efficiency tolerance sample counts differ")
    if any(header[index] != "S11 tolerance data" for index in s11_sample_columns):
        raise ValueError("unexpected column inside the Optenni S11 tolerance block")
    if any(
        header[index] != "Total efficiency tolerance data"
        for index in efficiency_sample_columns
    ):
        raise ValueError("unexpected column inside the Optenni efficiency tolerance block")

    expected_columns = len(header)
    numeric_rows = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"row {row_number} has {len(row)} columns; expected {expected_columns}"
            )
        try:
            numeric_rows.append([float(value) for value in row])
        except ValueError as exc:
            raise ValueError(f"row {row_number} contains a non-numeric value") from exc
    values = np.asarray(numeric_rows, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Optenni tolerance export contains non-finite values")

    frequencies_hz = values[:, 0] * _frequency_scale(header[0])
    if np.any(frequencies_hz <= 0) or np.any(np.diff(frequencies_hz) <= 0):
        raise ValueError("Optenni tolerance frequencies must be positive and increasing")

    efficiency_is_db = "[db]" in metadata.get("Y axis", "").lower()
    nominal_efficiency = values[:, efficiency_index]
    efficiency_variants = values[:, efficiency_sample_columns]
    if efficiency_is_db:
        nominal_efficiency = 10.0 ** (nominal_efficiency / 10.0)
        efficiency_variants = 10.0 ** (efficiency_variants / 10.0)
    if (
        np.any(nominal_efficiency < 0.0)
        or np.any(nominal_efficiency > 1.0 + 1e-9)
        or np.any(efficiency_variants < 0.0)
        or np.any(efficiency_variants > 1.0 + 1e-9)
    ):
        raise ValueError("Optenni total efficiency must be between 0 and 1")

    return OptenniToleranceExport(
        frequencies_hz=frequencies_hz,
        nominal_s11_db=values[:, 1],
        s11_variants_db=values[:, s11_sample_columns],
        nominal_total_efficiency=nominal_efficiency,
        total_efficiency_variants=efficiency_variants,
        metadata=metadata,
    )


def summarize_optenni_tolerance(
    export: OptenniToleranceExport,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    *,
    minimum_total_efficiency: float = 0.0,
    minimum_average_total_efficiency: float = 0.0,
    minimum_return_loss_db: float = 0.0,
) -> dict:
    """Reproduce Optenni's joint minimum/average-efficiency yield decision."""
    if minimum_frequency_hz > maximum_frequency_hz:
        raise ValueError("minimum_frequency_hz must not exceed maximum_frequency_hz")
    for name, value in (
        ("minimum_total_efficiency", minimum_total_efficiency),
        ("minimum_average_total_efficiency", minimum_average_total_efficiency),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if minimum_return_loss_db < 0.0:
        raise ValueError("minimum_return_loss_db must be non-negative")
    band = (
        (export.frequencies_hz >= minimum_frequency_hz)
        & (export.frequencies_hz <= maximum_frequency_hz)
    )
    if not np.any(band):
        raise ValueError("selected band contains no exported frequency points")

    variants = export.total_efficiency_variants[band]
    minimum_efficiency = np.min(variants, axis=0)
    # Optenni's "ave eff." is averaged after conversion to dB.  Returning the
    # equivalent geometric mean keeps the public criteria in linear units.
    average_efficiency = np.exp(np.mean(np.log(np.maximum(variants, 1e-15)), axis=0))
    minimum_return_loss = np.min(-export.s11_variants_db[band], axis=0)
    passed = (
        (minimum_efficiency >= minimum_total_efficiency)
        & (average_efficiency >= minimum_average_total_efficiency)
        & (minimum_return_loss >= minimum_return_loss_db)
    )

    nominal_efficiency = export.nominal_total_efficiency[band]
    nominal_return_loss = -export.nominal_s11_db[band]
    to_db = lambda value: float(10.0 * np.log10(max(float(value), 1e-15)))
    percentiles = (0, 1, 5, 50, 95, 99, 100)
    return {
        "samples": export.samples,
        "passed_samples": int(np.sum(passed)),
        "yield_fraction": float(np.mean(passed)),
        "band_points": int(np.sum(band)),
        "nominal_minimum_total_efficiency_db": to_db(np.min(nominal_efficiency)),
        "nominal_average_total_efficiency_db": float(np.mean(
            10.0 * np.log10(np.maximum(nominal_efficiency, 1e-15))
        )),
        "nominal_minimum_return_loss_db": float(np.min(nominal_return_loss)),
        "minimum_total_efficiency_percentiles_db": {
            str(p): to_db(value)
            for p, value in zip(percentiles, np.percentile(minimum_efficiency, percentiles))
        },
        "average_total_efficiency_percentiles_db": {
            str(p): to_db(value)
            for p, value in zip(percentiles, np.percentile(average_efficiency, percentiles))
        },
        "minimum_return_loss_percentiles_db": {
            str(p): float(value)
            for p, value in zip(percentiles, np.percentile(minimum_return_loss, percentiles))
        },
    }
