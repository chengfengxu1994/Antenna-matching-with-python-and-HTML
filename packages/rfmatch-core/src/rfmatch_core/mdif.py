"""Parser and state model for MDIF multi-state N-port components.

The initial contract intentionally supports the ACDATA dialect used by RF
component vendors and Optenni Lab: one ``VAR`` declaration followed by one
Touchstone-like S-parameter block per state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .models import S2PModel


_VAR_RE = re.compile(
    r"^VAR\s+(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+(?P<unit>\S+))?\s*$",
    re.IGNORECASE,
)
_VAR_ANY_RE = re.compile(
    r"^VAR\s+(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<(?P<name>[A-Za-z_]\w*)>(?P<value>[^<]*)</(?P=name)>")
_ROOT_ATTRIBUTE_RE = re.compile(r"<ComponentMetaData\s+(?P<attributes>[^>]*)>", re.IGNORECASE)
_ATTRIBUTE_RE = re.compile(r"(?P<name>[A-Za-z_]\w*)\s*=\s*\"(?P<value>[^\"]*)\"")


@dataclass(frozen=True)
class MDIFState:
    variable: str
    value: float | str
    unit: str | None
    frequencies_hz: np.ndarray
    s_parameters: np.ndarray
    z0: float = 50.0

    @property
    def label(self) -> str:
        value = f"{self.value:g}" if isinstance(self.value, float) else self.value
        return f"{value} {self.unit}" if self.unit else value

    @property
    def n_ports(self) -> int:
        return int(self.s_parameters.shape[1])

    def as_s2p_model(self, name: str | None = None, tolerance: float = 0.0) -> S2PModel:
        if self.n_ports != 2:
            raise ValueError(f"MDIF state has {self.n_ports} ports; a two-port model is required")
        return S2PModel(
            name=name or f"{self.variable}={self.label}",
            frequencies_hz=self.frequencies_hz,
            s_parameters=self.s_parameters,
            z0=self.z0,
            tolerance=tolerance,
        )

    def at(self, frequency_hz: float) -> np.ndarray:
        return self.sweep_at([frequency_hz])[0]

    def sweep_at(self, frequencies_hz: Sequence[float]) -> np.ndarray:
        """Interpolate the complete N-port matrix at multiple frequencies."""
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        requested = np.asarray(frequencies_hz, dtype=float)
        if requested.ndim != 1:
            raise ValueError("requested frequencies must be a vector")
        data = np.asarray(self.s_parameters, dtype=complex)
        if data.shape != (len(frequencies), self.n_ports, self.n_ports):
            raise ValueError("MDIF state S-parameters have an invalid shape")
        out = np.empty((len(requested), self.n_ports, self.n_ports), dtype=complex)
        for row in range(self.n_ports):
            for col in range(self.n_ports):
                values = data[:, row, col]
                out[:, row, col] = np.interp(requested, frequencies, values.real) + 1j * np.interp(
                    requested, frequencies, values.imag
                )
        return out


@dataclass(frozen=True)
class MDIFModel:
    name: str
    states: tuple[MDIFState, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    def state(self, selector: str | float) -> MDIFState:
        """Select a state by label (``8 pF``), numeric value, or categorical text."""
        if isinstance(selector, str):
            normalized = " ".join(selector.strip().lower().split())
            for state in self.states:
                candidates = {
                    state.label.lower(),
                    f"{state.value}",
                }
                if isinstance(state.value, float):
                    candidates.add(f"{state.value:g}")
                if normalized in candidates:
                    return state
            try:
                numeric = float(normalized.split()[0])
            except (ValueError, IndexError):
                numeric = None
        else:
            numeric = float(selector)
        if numeric is not None:
            matches = [
                state
                for state in self.states
                if isinstance(state.value, float) and np.isclose(state.value, numeric, rtol=1e-12, atol=0.0)
            ]
            if len(matches) == 1:
                return matches[0]
        available = ", ".join(state.label for state in self.states)
        raise KeyError(f"MDIF state {selector!r} not found; available states: {available}")


def _parse_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in lines:
        text = raw.lstrip("! ").strip()
        root = _ROOT_ATTRIBUTE_RE.search(text)
        if root:
            for match in _ATTRIBUTE_RE.finditer(root.group("attributes")):
                metadata[match.group("name")] = match.group("value")
        tag = _TAG_RE.search(text)
        if tag:
            metadata[tag.group("name")] = tag.group("value").strip()
    return metadata


def _decode_block(
    chunks: list[list[float]], option: list[str], n_ports: int | None
) -> tuple[np.ndarray, np.ndarray, float]:
    upper = [token.upper() for token in option]
    if "S" not in upper:
        raise ValueError("MDIF ACDATA currently supports S-parameters only")
    unit = next((item for item in ("HZ", "KHZ", "MHZ", "GHZ") if item in upper), "GHZ")
    fmt = next((item for item in ("RI", "MA", "DB") if item in upper), "MA")
    z0 = float(option[upper.index("R") + 1]) if "R" in upper else 50.0
    scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[unit]
    if n_ports is None:
        if not chunks:
            raise ValueError("MDIF block contains no numeric records")
        pair_count = (len(chunks[0]) - 1) / 2
        inferred = int(round(np.sqrt(pair_count)))
        if inferred < 1 or 1 + 2 * inferred * inferred != len(chunks[0]):
            raise ValueError("cannot infer MDIF port count; add ComponentMetaData nPorts")
        n_ports = inferred
    record_width = 1 + 2 * n_ports * n_ports
    records: list[list[float]] = []
    pending: list[float] = []
    for chunk in chunks:
        pending.extend(chunk)
        while len(pending) >= record_width:
            records.append(pending[:record_width])
            pending = pending[record_width:]
    if pending:
        raise ValueError(f"incomplete {n_ports}-port MDIF record")
    data = np.asarray(records, dtype=float)
    if data.ndim != 2 or data.shape[1] != record_width:
        raise ValueError(f"each {n_ports}-port MDIF record must contain frequency plus {record_width - 1} values")
    if not np.all(np.isfinite(data)):
        raise ValueError("MDIF block contains a non-finite number")
    frequencies = data[:, 0] * scale
    if len(frequencies) == 0 or np.any(np.diff(frequencies) <= 0):
        raise ValueError("MDIF state frequencies must be strictly increasing")
    pairs = data[:, 1:].reshape(-1, n_ports * n_ports, 2)
    if fmt == "RI":
        values = pairs[..., 0] + 1j * pairs[..., 1]
    else:
        magnitude = pairs[..., 0] if fmt == "MA" else 10.0 ** (pairs[..., 0] / 20.0)
        values = magnitude * np.exp(1j * np.deg2rad(pairs[..., 1]))
    # Touchstone/MDIF ordering is column-major: S11,S21,S12,S22.
    s_parameters = values.reshape(-1, n_ports, n_ports).transpose(0, 2, 1)
    return frequencies, s_parameters, z0


def load_mdif(path: str | Path, name: str | None = None) -> MDIFModel:
    """Load a multi-state two-port component from an MDIF ACDATA file."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = _parse_metadata(lines)
    states: list[MDIFState] = []
    pending: tuple[str, float | str, str | None] | None = None
    declared_ports = int(metadata["nPorts"]) if "nPorts" in metadata else None
    categorical_states = metadata.get("isSwitch") == "1"
    index = 0
    while index < len(lines):
        line = lines[index].split("!", 1)[0].strip()
        index += 1
        if not line:
            continue
        var = _VAR_RE.match(line)
        any_var = _VAR_ANY_RE.match(line)
        if any_var:
            if pending is not None:
                raise ValueError("MDIF VAR declaration is missing its ACDATA block")
            raw_value = any_var.group("value").strip()
            if var and not categorical_states:
                pending = (var.group("name"), float(var.group("value")), var.group("unit"))
            else:
                pending = (any_var.group("name"), raw_value, None)
            continue
        if line.upper() != "BEGIN ACDATA":
            continue
        if pending is None:
            raise ValueError("MDIF ACDATA block has no preceding VAR declaration")
        option: list[str] | None = None
        rows: list[list[float]] = []
        ended = False
        while index < len(lines):
            block_line = lines[index].split("!", 1)[0].strip()
            index += 1
            if not block_line:
                continue
            if block_line.upper() == "END":
                ended = True
                break
            if block_line.startswith("#"):
                if option is not None:
                    raise ValueError("MDIF state contains multiple option lines")
                option = block_line[1:].split()
                continue
            try:
                rows.append([float(token) for token in block_line.split()])
            except ValueError as exc:
                raise ValueError(f"invalid numeric MDIF record: {block_line}") from exc
        if not ended:
            raise ValueError("unterminated MDIF ACDATA block")
        if option is None:
            raise ValueError("MDIF state is missing its Touchstone option line")
        frequencies, s_parameters, z0 = _decode_block(rows, option, declared_ports)
        variable, value, unit = pending
        state_key = (variable.lower(), " ".join(str(value).lower().split()), unit)
        if any(
            (state.variable.lower(), " ".join(str(state.value).lower().split()), state.unit) == state_key
            for state in states
        ):
            raise ValueError(f"duplicate MDIF state {value} {unit or ''}".rstrip())
        states.append(MDIFState(variable, value, unit, frequencies, s_parameters, z0))
        pending = None
    if pending is not None:
        raise ValueError("MDIF VAR declaration is missing its ACDATA block")
    if not states:
        raise ValueError("MDIF file contains no state ACDATA blocks")
    if len({state.variable.lower() for state in states}) != 1:
        raise ValueError("all MDIF states must use the same variable name")
    component_name = name or metadata.get("ComponentName") or source.stem
    return MDIFModel(component_name, tuple(states), metadata)
