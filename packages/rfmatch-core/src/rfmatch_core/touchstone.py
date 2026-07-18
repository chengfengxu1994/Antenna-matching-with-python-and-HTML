from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass
class Touchstone:
    frequencies_hz: np.ndarray
    s_parameters: np.ndarray
    z0: float | np.ndarray
    frequency_unit: str = "HZ"
    data_format: str = "RI"
    parameter_type: str = "S"
    comments: tuple[str, ...] = ()


def parse_touchstone_text(content: str, filename: str = "unknown") -> Touchstone:
    """Strictly parse real-reference Touchstone 1.x/2.0 S data.

    The parser accepts wrapped multi-port records, inline comments and the
    Touchstone 2.0 ``[Network Data]`` section, matrix subsets, two-port order,
    and real per-port references. Unsupported parameter domains and mixed-mode
    data fail explicitly instead of being silently assigned the wrong meaning.
    """
    match = re.search(r"\.s(\d+)p$", filename, re.IGNORECASE)
    declared_ports = None
    declared_frequencies = None
    declared_references: list[float] | None = None
    version = None
    matrix_format = "FULL"
    two_port_order = "21_12"
    comments: list[str] = []
    option = None
    tokens: list[str] = []
    has_network_section = bool(re.search(r"^\s*\[Network Data\]", content, re.I | re.M))
    in_network_data = not has_network_section
    collecting_references = False

    for raw in content.splitlines():
        before_comment, separator, comment = raw.partition("!")
        if separator and comment.strip():
            comments.append(comment.strip())
        line = before_comment.strip()
        if not line:
            continue
        if line.startswith("#"):
            collecting_references = False
            option = line[1:].upper().split()
            continue
        keyword = re.match(r"^\[([^]]+)\]\s*(.*)$", line)
        if keyword:
            name, value = keyword.group(1).strip().lower(), keyword.group(2).strip()
            collecting_references = False
            if name == "version":
                version = value
            elif name == "number of ports":
                declared_ports = int(value)
            elif name == "number of frequencies":
                declared_frequencies = int(value)
            elif name == "network data":
                in_network_data = True
            elif name == "end":
                in_network_data = False
            elif name == "reference":
                collecting_references = True
                declared_references = []
                if value:
                    try:
                        declared_references.extend(float(item) for item in value.split())
                    except ValueError as exc:
                        raise ValueError(
                            "Touchstone [Reference] accepts only real-valued impedances"
                        ) from exc
            elif name == "matrix format":
                matrix_format = value.upper()
                if matrix_format not in {"FULL", "LOWER", "UPPER"}:
                    raise ValueError("[Matrix Format] must be Full, Lower, or Upper")
            elif name == "two-port data order":
                two_port_order = value.upper()
                if two_port_order not in {"21_12", "12_21"}:
                    raise ValueError("[Two-Port Data Order] must be 21_12 or 12_21")
            elif name == "mixed-mode order":
                raise ValueError("mixed-mode Touchstone data is not supported")
            continue
        if collecting_references:
            try:
                declared_references.extend(float(item) for item in line.split())
            except ValueError as exc:
                raise ValueError(
                    "Touchstone [Reference] accepts only real-valued impedances"
                ) from exc
            continue
        if in_network_data:
            tokens.extend(line.split())

    if not match and declared_ports is None:
        raise ValueError(f"cannot determine port count from {filename}")
    n_ports = int(declared_ports if declared_ports is not None else match.group(1))
    if match and declared_ports is not None and int(match.group(1)) != n_ports:
        raise ValueError("Touchstone filename and [Number of Ports] disagree")
    if n_ports < 1:
        raise ValueError("Touchstone port count must be positive")
    if two_port_order != "21_12" and n_ports != 2:
        raise ValueError("[Two-Port Data Order] is valid only for two-port data")
    if not option or "S" not in option:
        raise ValueError("only S-parameter Touchstone files are supported")
    parameter_tokens = {"S", "Y", "Z", "H", "G"}.intersection(option)
    if parameter_tokens != {"S"}:
        raise ValueError("only S-parameter Touchstone files are supported")
    unit = next((u for u in ("HZ", "KHZ", "MHZ", "GHZ") if u in option), "GHZ")
    fmt = next((f for f in ("RI", "MA", "DB") if f in option), "MA")
    z0 = float(option[option.index("R") + 1]) if "R" in option else 50.0
    if not np.isfinite(z0) or z0 <= 0:
        raise ValueError("Touchstone reference resistance must be positive and finite")
    if declared_references is not None:
        references = np.asarray(declared_references, dtype=float)
        if len(references) != n_ports:
            raise ValueError("[Reference] must contain exactly one real value per port")
        if not np.all(np.isfinite(references)) or np.any(references <= 0):
            raise ValueError("Touchstone per-port references must be positive and finite")
        z0 = (
            float(references[0])
            if np.allclose(references, references[0], rtol=0.0, atol=1e-12)
            else references.copy()
        )
    scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[unit]
    value_count = (
        n_ports * n_ports
        if matrix_format == "FULL"
        else n_ports * (n_ports + 1) // 2
    )
    width = 1 + 2 * value_count
    if not tokens:
        raise ValueError("Touchstone file contains no network data")
    if len(tokens) % width:
        raise ValueError("incomplete Touchstone data record")
    try:
        rows = np.asarray(tokens, dtype=float).reshape(-1, width)
    except ValueError as exc:
        raise ValueError("Touchstone network data contains a non-numeric token") from exc
    if not np.all(np.isfinite(rows)):
        raise ValueError("Touchstone network data must be finite")
    frequencies_hz = rows[:, 0] * scale
    if np.any(frequencies_hz < 0) or np.any(np.diff(frequencies_hz) <= 0):
        raise ValueError("Touchstone frequencies must be non-negative and strictly increasing")
    if declared_frequencies is not None and len(rows) != declared_frequencies:
        raise ValueError(
            f"[Number of Frequencies] declares {declared_frequencies}, found {len(rows)}"
        )
    values = rows[:, 1:].reshape(-1, value_count, 2)
    if fmt == "RI":
        complex_values = values[..., 0] + 1j * values[..., 1]
    else:
        magnitude = values[..., 0] if fmt == "MA" else 10.0 ** (values[..., 0] / 20.0)
        complex_values = magnitude * np.exp(1j * np.deg2rad(values[..., 1]))
    if matrix_format == "FULL":
        if n_ports == 2 and two_port_order == "21_12":
            coordinates = [(0, 0), (1, 0), (0, 1), (1, 1)]
        else:
            # Touchstone 1.x/2.0 uses row-wise matrix order for 3+ ports;
            # 12_21 selects the same natural row order for two-port data.
            coordinates = [
                (row, column)
                for row in range(n_ports)
                for column in range(n_ports)
            ]
    elif n_ports == 2:
        # The specification defines both triangular two-port encodings as
        # N11, N21, N22 and reconstructs N12 from symmetry.
        coordinates = [(0, 0), (1, 0), (1, 1)]
    elif matrix_format == "LOWER":
        coordinates = [
            (row, column)
            for row in range(n_ports)
            for column in range(row + 1)
        ]
    else:
        coordinates = [
            (row, column)
            for row in range(n_ports)
            for column in range(row, n_ports)
        ]
    s = np.zeros((len(rows), n_ports, n_ports), dtype=complex)
    for index, (row, column) in enumerate(coordinates):
        s[:, row, column] = complex_values[:, index]
        if matrix_format != "FULL" and row != column:
            s[:, column, row] = complex_values[:, index]
    return Touchstone(
        frequencies_hz, s, z0, unit, fmt, "S", tuple(comments)
    )


def load_touchstone(path: str | Path) -> Touchstone:
    """Read a Touchstone S-parameter file in RI/MA/DB format."""
    path = Path(path)
    return parse_touchstone_text(
        path.read_text(encoding="utf-8", errors="replace"), path.name
    )


def load_s2p_model(
    path: str | Path,
    name: str | None = None,
    tolerance: float = 0.0,
    kind: str | None = None,
    nominal_value: float | None = None,
):
    from .models import S2PModel
    data = load_touchstone(path)
    if data.s_parameters.shape[1:] != (2, 2):
        raise ValueError("component model must be an S2P file")
    return S2PModel(
        name or Path(path).stem,
        data.frequencies_hz,
        data.s_parameters,
        data.z0,
        tolerance,
        kind,
        nominal_value,
    )
