"""
Touchstone SNP/S2P file parser.

Supports Touchstone v1 and v2 formats through the authoritative core parser:
- Frequency units: Hz, kHz, MHz, GHz
- Parameter domain: S
- Data formats: RI (real-imaginary), MA (magnitude-angle in degrees), DB (dB-angle)
- Real scalar and per-port reference impedances
- Touchstone 2.0 Full/Lower/Upper matrices and explicit two-port data order

Touchstone 2.0 does not define complex reference impedances. Mixed-mode data
is rejected explicitly until a separately verified conversion is available.
"""

import os
import re
import hashlib
import zipfile
import io
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rfmatch_core.touchstone import parse_touchstone_text as _parse_core_touchstone
from rfmatch_core.network import s_to_z as _core_s_to_z


@dataclass
class TouchstoneData:
    """Parsed Touchstone file data."""
    filename: str
    frequency_unit: str  # Hz, kHz, MHz, GHz
    parameter_type: str  # S, Y, Z, H, G
    data_format: str  # RI, MA, DB
    reference_resistance: float  # R in ohms (default 50)
    num_ports: int
    frequencies: List[float] = field(default_factory=list)
    # S-parameters: dict mapping (i, j) to complex array over frequencies
    # Indices are 1-based for S-parameter convention
    sparameters: dict = field(default_factory=dict)
    comments: List[str] = field(default_factory=list)
    # Per-port complex reference impedance if present
    port_impedances: Optional[List[complex]] = None

    @property
    def n_ports(self):
        return self.num_ports

    def get_s(self, i: int, j: int, freq_idx: int) -> complex:
        """Get S-parameter S_ij at frequency index."""
        return self.sparameters[(i, j)][freq_idx]

    def get_s_matrix(self, freq_idx: int) -> np.ndarray:
        """Get full S-matrix at frequency index."""
        n = self.num_ports
        S = np.zeros((n, n), dtype=complex)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                S[i - 1, j - 1] = self.get_s(i, j, freq_idx)
        return S
    def get_z_matrix(
        self, freq_idx: int, Z0: float | np.ndarray | None = None
    ) -> np.ndarray:
        """Convert S-matrix to Z-matrix at frequency index."""
        S = self.get_s_matrix(freq_idx)
        if Z0 is None:
            Z0 = (
                np.asarray([complex(value).real for value in self.port_impedances])
                if self.port_impedances else self.reference_resistance
            )
        return _core_s_to_z(S, Z0)

    def find_freq_index(self, target_freq_hz: float) -> int:
        """Find the index of the closest frequency to target (in Hz)."""
        if not self.frequencies:
            raise ValueError("No frequency data")
        freqs = np.array(self.frequencies)
        return int(np.argmin(np.abs(freqs - target_freq_hz)))

    def interpolate_s(self, i: int, j: int, target_freq_hz: float) -> complex:
        """Linearly interpolate S-parameter at target frequency."""
        freqs = np.array(self.frequencies)
        vals = np.array([self.get_s(i, j, k) for k in range(len(freqs))])

        # Find bracketing points
        if target_freq_hz <= freqs[0]:
            return vals[0]
        if target_freq_hz >= freqs[-1]:
            return vals[-1]

        idx = np.searchsorted(freqs, target_freq_hz)
        if idx == 0:
            return vals[0]

        f0, f1 = freqs[idx - 1], freqs[idx]
        v0, v1 = vals[idx - 1], vals[idx]
        t = (target_freq_hz - f0) / (f1 - f0)

        # Interpolate real and imaginary separately
        real = v0.real + t * (v1.real - v0.real)
        imag = v0.imag + t * (v1.imag - v0.imag)
        return complex(real, imag)

    def get_s_matrix_interpolated(self, target_freq_hz: float) -> np.ndarray:
        """Get full S-matrix interpolated at target frequency."""
        n = self.num_ports
        S = np.zeros((n, n), dtype=complex)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                S[i - 1, j - 1] = self.interpolate_s(i, j, target_freq_hz)
        return S


def touchstone_network_sha256(data: TouchstoneData) -> str:
    """Hash the electrical network while ignoring comments and text formatting."""
    frequency_count = len(data.frequencies)
    port_count = int(data.num_ports)
    matrices = np.empty((frequency_count, port_count, port_count), dtype="<c16")
    for frequency_index in range(frequency_count):
        matrices[frequency_index] = data.get_s_matrix(frequency_index)
    references = np.asarray(
        data.port_impedances
        or [complex(data.reference_resistance)] * port_count,
        dtype="<c16",
    )
    digest = hashlib.sha256()
    digest.update(b"rfmatch.touchstone-network.v1\0")
    digest.update(np.asarray([port_count, frequency_count], dtype="<i8").tobytes())
    digest.update(np.asarray(data.frequencies, dtype="<f8").tobytes())
    digest.update(references.tobytes())
    digest.update(matrices.tobytes())
    return digest.hexdigest()


# Frequency unit to Hz multiplier
FREQ_MULTIPLIERS = {
    'HZ': 1.0,
    'KHZ': 1e3,
    'MHZ': 1e6,
    'GHZ': 1e9,
}


def _parse_freq_unit(token: str) -> str:
    token = token.upper()
    if token in FREQ_MULTIPLIERS:
        return token
    # Handle lowercase or mixed case
    for key in FREQ_MULTIPLIERS:
        if token == key:
            return key
    raise ValueError(f"Unknown frequency unit: {token}")


def _detect_num_ports_from_filename(filename: str) -> int:
    """Detect number of ports from Touchstone file extension."""
    import re
    match = re.search(r'\.s(\d+)p$', filename.lower())
    if match:
        return int(match.group(1))
    return 0


def _parse_touchstone_legacy(content: str, filename: str = "unknown") -> TouchstoneData:
    """Parse a Touchstone file from string content.

    Handles multi-line data for multi-port files. According to Touchstone spec:
    - Maximum 4 data pairs per line (8 numbers)
    - Multi-port data wraps across multiple lines per frequency point
    """
    lines = content.strip().split('\n')
    data = TouchstoneData(filename=filename, frequency_unit='GHZ',
                         parameter_type='S', data_format='MA',
                         reference_resistance=50.0, num_ports=0)

    # Guess number of ports from filename
    num_ports_guess = _detect_num_ports_from_filename(filename)

    option_line_found = False
    freq_unit_mult = 1.0
    param_format = None  # 'RI', 'MA', 'DB'

    # Collect non-comment, non-option lines as data lines
    data_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if line_stripped.startswith('!'):
            data.comments.append(line_stripped[1:].strip())
            continue

        if line_stripped.startswith('#'):
            option_line_found = True
            tokens = line_stripped.split()
            idx = 1
            if idx < len(tokens):
                data.frequency_unit = _parse_freq_unit(tokens[idx])
                freq_unit_mult = FREQ_MULTIPLIERS[data.frequency_unit]
                idx += 1
            if idx < len(tokens):
                param = tokens[idx].upper()
                if param in ('S', 'Y', 'Z', 'H', 'G'):
                    data.parameter_type = param
                idx += 1
            if idx < len(tokens):
                fmt = tokens[idx].upper()
                if fmt in ('RI', 'MA', 'DB'):
                    data.data_format = fmt
                    param_format = fmt
                idx += 1
            if idx < len(tokens) and tokens[idx].upper() == 'R':
                idx += 1
                if idx < len(tokens):
                    data.reference_resistance = float(tokens[idx])
                    idx += 1
            continue

        if not option_line_found:
            continue

        # Collect data lines
        data_lines.append(line_stripped)

    if not data_lines:
        return data

    # Determine num_ports: from filename hint, or from first data point
    # Collect all tokens from data lines
    all_tokens = []
    line_boundaries = []  # track where each original line starts in all_tokens
    for dl in data_lines:
        tokens = dl.split()
        line_boundaries.append((len(all_tokens), len(all_tokens) + len(tokens)))
        all_tokens.extend(tokens)

    if not all_tokens:
        return data

    # Parse all token values
    numeric_values = []
    freq_indices = []  # indices in numeric_values where frequency values are
    for idx, t in enumerate(all_tokens):
        try:
            numeric_values.append(float(t))
        except ValueError:
            numeric_values.append(None)

    # Determine number of ports if not known
    if num_ports_guess > 0:
        data.num_ports = num_ports_guess
    else:
        # Heuristic: check how many numbers follow the first frequency
        # The number of S-parameter values determines port count
        # For N ports: N^2 complex values = 2*N^2 real numbers
        pass  # Will be determined below

    # Parse frequency points: each starts with a frequency on a new line
    # For multi-port, data for a frequency spans multiple lines
    n = data.num_ports
    expected_per_freq = 2 * n * n  # total real numbers per frequency
    expected_with_freq = 1 + expected_per_freq  # including frequency value

    # Strategy: scan through all_tokens, identifying frequency values
    # (they appear at the start of each original data line, but only the first
    # line of a frequency point is a new frequency)

    # Better strategy: walk through data_lines and accumulate values
    # until we have enough for one complete frequency point

    import math

    # First pass: collect all numbers from all data lines
    all_numbers = [float(t) for t in all_tokens if t.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit() or _is_float(t)]

    # Find frequencies: they are the values where a new frequency point begins
    # For multi-line data, we need to guess ports first

    # Use filename hint first; if not available, try to detect
    if data.num_ports == 0:
        # Look at the number of values in the first few data lines
        # to estimate port count
        first_line_nums = [float(t) for t in data_lines[0].split() if _is_float(t)]
        if len(first_line_nums) > 1:
            n_vals = len(first_line_nums) - 1  # exclude frequency
            # n_vals should be 2*N^2 for a complete point, or fewer for wrapped lines
            # Typical: S2P has 8 values per freq, S3P has 18, S4P has 32
            possible_ports = {1: 2, 2: 8, 3: 18, 4: 32, 5: 50, 6: 72}
            for ports, count in possible_ports.items():
                if n_vals == count:
                    data.num_ports = ports
                    break
            else:
                # Guess based on partial data
                data.num_ports = 1  # default to 1-port

    n = data.num_ports
    if n == 0:
        n = 1
        data.num_ports = 1

    expected_per_freq = 2 * n * n

    # Now parse frequency points by iterating through data lines
    # Accumulate parameter values for current frequency point
    current_freq = None
    current_values = []
    freq_idx = 0

    for dl in data_lines:
        tokens = dl.split()
        if not tokens:
            continue
        try:
            freq_raw = float(tokens[0])
        except ValueError:
            continue

        # Is this a frequency value (start of new point) or a continuation?
        # If current_values is empty or full, this is a new frequency
        if current_freq is None or len(current_values) >= expected_per_freq:
            # Save previous point if exists
            if current_freq is not None and len(current_values) >= expected_per_freq:
                data.frequencies.append(current_freq)
                _add_sparams(data, current_values[:expected_per_freq], n)
                freq_idx += 1

            # Start new frequency point
            current_freq = freq_raw * freq_unit_mult
            current_values = []
            # Add remaining tokens from this line
            for t in tokens[1:]:
                try:
                    current_values.append(float(t))
                except ValueError:
                    pass
        else:
            # Continuation of current frequency point
            for t in tokens:
                try:
                    current_values.append(float(t))
                except ValueError:
                    pass

    # Don't forget the last point
    if current_freq is not None and len(current_values) >= expected_per_freq:
        data.frequencies.append(current_freq)
        _add_sparams(data, current_values[:expected_per_freq], n)

    return data


def parse_touchstone(content: str, filename: str = "unknown") -> TouchstoneData:
    """Parse product Touchstone input through the authoritative core parser."""
    parsed = _parse_core_touchstone(content, filename)
    n_ports = int(parsed.s_parameters.shape[1])
    sparameters = {
        (destination + 1, source + 1): parsed.s_parameters[:, destination, source].tolist()
        for destination in range(n_ports)
        for source in range(n_ports)
    }
    z0_values = np.asarray(parsed.z0, dtype=float)
    scalar_z0 = float(z0_values) if z0_values.ndim == 0 else float(z0_values[0])
    port_z0 = (
        np.full(n_ports, scalar_z0, dtype=float)
        if z0_values.ndim == 0 else z0_values
    )
    return TouchstoneData(
        filename=filename,
        frequency_unit=parsed.frequency_unit,
        parameter_type=parsed.parameter_type,
        data_format=parsed.data_format,
        reference_resistance=scalar_z0,
        num_ports=n_ports,
        frequencies=parsed.frequencies_hz.tolist(),
        sparameters=sparameters,
        comments=list(parsed.comments),
        port_impedances=[complex(value) for value in port_z0],
    )


def _is_float(s: str) -> bool:
    """Check if a string can be parsed as float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _add_sparams(data: TouchstoneData, values: list, n: int):
    """Add one Touchstone record using the standard column-major order.

    Touchstone stores S11, S21, ... first (responses at every row for source
    port 1), then S12, S22, ... . Iterating source/column first is essential
    for non-reciprocal and measured multiport networks.
    """
    idx_p = 0
    for jj in range(1, n + 1):
        for ii in range(1, n + 1):
            if idx_p + 1 >= len(values):
                break
            if data.data_format == 'RI':
                real = values[idx_p]
                imag = values[idx_p + 1]
                val = complex(real, imag)
            elif data.data_format == 'MA':
                mag = values[idx_p]
                angle_deg = values[idx_p + 1]
                angle_rad = np.deg2rad(angle_deg)
                val = mag * np.exp(1j * angle_rad)
            elif data.data_format == 'DB':
                db = values[idx_p]
                angle_deg = values[idx_p + 1]
                mag = 10 ** (db / 20.0)
                angle_rad = np.deg2rad(angle_deg)
                val = mag * np.exp(1j * angle_rad)
            else:
                val = complex(values[idx_p], values[idx_p + 1])

            if (ii, jj) not in data.sparameters:
                data.sparameters[(ii, jj)] = []
            data.sparameters[(ii, jj)].append(val)
            idx_p += 2


def load_touchstone_file(filepath: str | os.PathLike[str]) -> TouchstoneData:
    """Load a Touchstone file from disk."""
    filepath = os.fspath(filepath)
    # Determine file extension for port count hint
    ext = filepath.lower().rsplit('.', 1)[-1] if '.' in filepath else ''
    # .s1p, .s2p, .s3p, .s4p etc.

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    data = parse_touchstone(content, filename=filepath)
    return data


def load_s2p_from_zip(zip_path: str, s2p_filename: str) -> TouchstoneData:
    """Load an S2P file from within a ZIP archive."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(s2p_filename) as f:
            content = f.read().decode('utf-8', errors='replace')
    data = parse_touchstone(content, filename=s2p_filename)
    return data


def list_s2p_in_zip(zip_path: str) -> List[str]:
    """List all .s2p files in a ZIP archive."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return [name for name in zf.namelist() if name.lower().endswith('.s2p')]


def interpolate_s_matrix(data: TouchstoneData, target_freq_hz: float) -> np.ndarray:
    """Extract or interpolate the S-matrix at a given frequency."""
    return data.get_s_matrix_interpolated(target_freq_hz)
