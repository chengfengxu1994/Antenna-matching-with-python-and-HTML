"""Metadata-first loaders for measured two-port component libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Callable, Literal

import numpy as np

from .models import S2PModel
from .touchstone import load_s2p_model


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: Literal["L", "C"]
    value: float
    tolerance: float
    family: str
    source_path: Path

    @property
    def value_display(self) -> str:
        scale, unit = (1e9, "nH") if self.kind == "L" else (1e12, "pF")
        return f"{self.value * scale:g} {unit}"


@dataclass(frozen=True)
class LazyComponentSpec:
    """Measured part whose S2P model is supplied lazily by an external adapter.

    This keeps ZIP and database-backed product catalogs metadata-first: only
    parts reached by hierarchical refinement materialize a frequency sweep.
    """

    name: str
    kind: Literal["L", "C"]
    value: float
    tolerance: float
    family: str
    provenance: str
    model_loader: Callable[[], S2PModel] = field(repr=False, compare=False)

    @property
    def value_display(self) -> str:
        scale, unit = (1e9, "nH") if self.kind == "L" else (1e12, "pF")
        return f"{self.value * scale:g} {unit}"


MeasuredComponentSpec = ComponentSpec | LazyComponentSpec


def _parse_nh(code: str) -> float:
    code = code.upper()
    if re.fullmatch(r"\d+N\d+", code):
        whole, fractional = code.split("N", 1)
        return float(f"{whole}.{fractional}")
    if re.fullmatch(r"\d+N", code):
        return float(code[:-1])
    raise ValueError(f"unsupported inductor value code: {code}")


def _parse_pf(code: str) -> float:
    code = code.upper()
    if "R" in code:
        whole, fractional = code.split("R", 1)
        return float(f"{whole or '0'}.{fractional}")
    if re.fullmatch(r"\d{3}", code):
        return float(int(code[:2]) * 10 ** int(code[2]))
    raise ValueError(f"unsupported capacitor value code: {code}")


def load_coilcraft_0402hp_catalog(directory: str | Path) -> list[ComponentSpec]:
    """Read the standard 1–51 nH 0402HP S2P filenames without loading their data."""
    directory = Path(directory)
    records: list[ComponentSpec] = []
    for path in sorted(directory.glob("*.s2p")):
        match = re.fullmatch(r"04HP(\d+N\d*|\d+N)", path.stem, re.IGNORECASE)
        if not match:
            continue
        value_nh = _parse_nh(match.group(1))
        records.append(ComponentSpec(path.stem, "L", value_nh * 1e-9, 0.05, "Coilcraft 0402HP", path.resolve()))
    return sorted(records, key=lambda item: (item.value, item.name))


def load_coilcraft_0402cs_catalog(directory: str | Path) -> list[ComponentSpec]:
    """Read the Coilcraft 0402CS family used by the Optenni multi-state tutorial."""
    directory = Path(directory)
    records: list[ComponentSpec] = []
    for path in sorted(directory.glob("*.[sS]2[pP]")):
        match = re.fullmatch(r"04CS(\d+N\d*|\d+N)", path.stem, re.IGNORECASE)
        if not match:
            continue
        value_nh = _parse_nh(match.group(1))
        records.append(ComponentSpec(
            path.stem, "L", value_nh * 1e-9, 0.05,
            "Coilcraft 0402CS", path.resolve(),
        ))
    return sorted(records, key=lambda item: (item.value, item.name))


_TOLERANCE = {
    "B": (None, 0.1),
    "C": (None, 0.25),
    "D": (None, 0.5),
    "F": (0.01, None),
    "G": (0.02, None),
    "J": (0.05, None),
    "K": (0.10, None),
    "M": (0.20, None),
    "W": (None, 0.05),
}


def load_murata_gqm18_catalog(directory: str | Path, *, unique_values: bool = True) -> list[ComponentSpec]:
    """Read GQM18 part values and prefer the tightest available tolerance per value."""
    directory = Path(directory)
    records: list[ComponentSpec] = []
    pattern = re.compile(r"(\dR\d|R\d{2}|\d{3})([BCDFGHJKM])B12$", re.IGNORECASE)
    for path in sorted(directory.glob("*.s2p")):
        match = pattern.search(path.stem)
        if not match:
            continue
        value_pf = _parse_pf(match.group(1))
        relative, absolute = _TOLERANCE[match.group(2).upper()]
        tolerance = relative if relative is not None else absolute / value_pf
        records.append(ComponentSpec(path.stem, "C", value_pf * 1e-12, tolerance, "Murata GQM18", path.resolve()))
    if unique_values:
        best: dict[float, ComponentSpec] = {}
        for record in records:
            previous = best.get(record.value)
            if previous is None or (record.tolerance, record.name) < (previous.tolerance, previous.name):
                best[record.value] = record
        records = list(best.values())
    return sorted(records, key=lambda item: (item.value, item.tolerance, item.name))


def load_murata_gjm15_catalog(
    directory: str | Path,
    *,
    unique_values: bool = True,
    prefer_loosest_tolerance: bool = False,
) -> list[ComponentSpec]:
    """Read Murata GJM15 values, including absolute B/C/D/W tolerances."""
    directory = Path(directory)
    records: list[ComponentSpec] = []
    pattern = re.compile(r"(\dR\d|R\d{2}|\d{3})([BCDFGHJKMW])B01$", re.IGNORECASE)
    for path in sorted(directory.glob("*.[sS]2[pP]")):
        if not path.stem.upper().startswith("GJM15"):
            continue
        match = pattern.search(path.stem)
        if not match:
            continue
        value_pf = _parse_pf(match.group(1))
        relative, absolute = _TOLERANCE[match.group(2).upper()]
        tolerance = relative if relative is not None else absolute / value_pf
        records.append(ComponentSpec(
            path.stem, "C", value_pf * 1e-12, tolerance,
            "Murata GJM15", path.resolve(),
        ))
    if unique_values:
        selected: dict[float, ComponentSpec] = {}
        for record in records:
            previous = selected.get(record.value)
            key = (-record.tolerance, record.name) if prefer_loosest_tolerance else (record.tolerance, record.name)
            previous_key = None if previous is None else (
                (-previous.tolerance, previous.name)
                if prefer_loosest_tolerance else (previous.tolerance, previous.name)
            )
            if previous is None or key < previous_key:
                selected[record.value] = record
        records = list(selected.values())
    return sorted(records, key=lambda item: (item.value, item.tolerance, item.name))


def component_model_key(spec: MeasuredComponentSpec):
    if isinstance(spec, ComponentSpec):
        return spec.source_path.resolve()
    return ("lazy", spec.provenance, spec.name, spec.kind, spec.value)


def load_component_model(spec: MeasuredComponentSpec) -> S2PModel:
    if isinstance(spec, LazyComponentSpec):
        model = spec.model_loader()
        return S2PModel(
            spec.name,
            model.frequencies_hz,
            model.s_parameters,
            model.z0,
            spec.tolerance,
            spec.kind,
            spec.value,
        )
    return load_s2p_model(
        spec.source_path,
        name=spec.name,
        tolerance=spec.tolerance,
        kind=spec.kind,
        nominal_value=spec.value,
    )


def component_sha256(spec: MeasuredComponentSpec) -> str:
    digest = hashlib.sha256()
    if isinstance(spec, LazyComponentSpec):
        model = load_component_model(spec)
        digest.update(np.asarray(model.frequencies_hz, dtype="<f8").tobytes())
        digest.update(np.asarray(model.s_parameters, dtype="<c16").tobytes())
        digest.update(np.asarray([model.z0], dtype="<f8").tobytes())
        return digest.hexdigest()
    with spec.source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
