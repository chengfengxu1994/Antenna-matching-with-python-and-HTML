"""
Radiation efficiency data parser and interpolator.

Supports the Optenni Lab format:
    % Comment lines
    Freq_MHz  LinearEfficiency
    700       0.833106
    740       0.896147
    ...

Also supports:
    - Frequency in Hz, kHz, MHz, GHz (auto-detected or specified)
    - Efficiency in linear (0-1) or dB (auto-detected)
    - CSV, TSV, and space-separated formats
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EfficiencyData:
    """Parsed radiation efficiency data with interpolation."""
    frequencies_hz: np.ndarray   # Frequencies in Hz (sorted)
    efficiency_linear: np.ndarray  # Linear efficiency values (0-1)
    source: str = ""              # Source filename or description

    @property
    def freq_min_hz(self) -> float:
        return float(self.frequencies_hz[0]) if len(self.frequencies_hz) > 0 else 0.0

    @property
    def freq_max_hz(self) -> float:
        return float(self.frequencies_hz[-1]) if len(self.frequencies_hz) > 0 else 0.0

    def get_efficiency_at(self, freq_hz: float) -> float:
        """Interpolate radiation efficiency at a given frequency (Hz).
        Returns linear efficiency (0-1). Clamps at boundaries."""
        if len(self.frequencies_hz) == 0:
            return 1.0  # No data → assume 100% radiation efficiency

        freqs = self.frequencies_hz
        effs = self.efficiency_linear

        # Clamp at boundaries
        if freq_hz <= freqs[0]:
            return float(effs[0])
        if freq_hz >= freqs[-1]:
            return float(effs[-1])

        # Linear interpolation
        idx = np.searchsorted(freqs, freq_hz)
        if idx == 0:
            return float(effs[0])

        f0, f1 = freqs[idx - 1], freqs[idx]
        e0, e1 = effs[idx - 1], effs[idx]
        t = (freq_hz - f0) / (f1 - f0)
        return float(e0 + t * (e1 - e0))

    def get_efficiency_array(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Get interpolated efficiency at multiple frequencies.
        Returns array of linear efficiency values."""
        return np.array([self.get_efficiency_at(f) for f in freqs_hz])

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'freq_min_hz': self.freq_min_hz,
            'freq_max_hz': self.freq_max_hz,
            'num_points': len(self.frequencies_hz),
            'efficiency_min': float(np.min(self.efficiency_linear)) if len(self.efficiency_linear) > 0 else 0,
            'efficiency_max': float(np.max(self.efficiency_linear)) if len(self.efficiency_linear) > 0 else 0,
        }


def _detect_freq_unit(values: np.ndarray) -> float:
    """Auto-detect frequency unit from the magnitude of values.
    Returns multiplier to convert to Hz."""
    if len(values) == 0:
        return 1e6  # Default to MHz

    max_val = np.max(np.abs(values))
    if max_val < 100:
        return 1e9    # GHz (values < 100)
    elif max_val < 100000:
        return 1e6    # MHz (values < 100k)
    elif max_val < 100000000:
        return 1e3    # kHz
    else:
        return 1.0    # Hz


def _detect_efficiency_format(values: np.ndarray) -> str:
    """Auto-detect whether efficiency is linear or dB.
    Linear: values between 0 and 1 (typically 0.5-1.0)
    dB: values typically negative (e.g., -1 to -10)
    """
    if len(values) == 0:
        return 'linear'

    min_val = np.min(values)
    max_val = np.max(values)

    # If any value is negative or all values > 1, it's likely dB
    if min_val < -0.01 or max_val > 1.05:
        return 'db'

    return 'linear'


def parse_efficiency_data(
    content: str,
    filename: str = "unknown",
    freq_unit: Optional[str] = None,  # 'hz', 'khz', 'mhz', 'ghz' or None for auto
    eff_format: Optional[str] = None,  # 'linear', 'db' or None for auto
) -> EfficiencyData:
    """
    Parse efficiency data from a text file.

    Supported formats:
    - Optenni Lab: "% Freq(MHz) eff(lin)" followed by tab/space-separated data
    - Simple CSV: freq,efficiency per line
    - Auto-detects frequency unit and efficiency format

    Args:
        content: File content as string
        filename: Source filename for logging
        freq_unit: Force frequency unit ('hz','khz','mhz','ghz') or None for auto-detect
        eff_format: Force 'linear' or 'db' or None for auto-detect

    Returns:
        EfficiencyData object
    """
    lines = content.strip().split('\n')

    freqs = []
    effs = []

    # Parse header hints
    header_freq_unit = None
    header_eff_format = None

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Parse comment headers for hints
        if line.startswith('%') or line.startswith('!') or line.startswith('#'):
            lower = line.lower()
            # Detect frequency unit from header
            if 'ghz' in lower or '(ghz)' in lower:
                header_freq_unit = 'ghz'
            elif 'mhz' in lower or '(mhz)' in lower:
                header_freq_unit = 'mhz'
            elif 'khz' in lower or '(khz)' in lower:
                header_freq_unit = 'khz'
            elif 'hz' in lower:
                header_freq_unit = 'hz'

            # Detect efficiency format from header
            if 'db' in lower or '(db)' in lower:
                header_eff_format = 'db'
            elif 'lin' in lower:
                header_eff_format = 'linear'

            continue

        # Try to parse data line
        # Split by tab, comma, or multiple spaces
        import re
        parts = re.split(r'[\t,]+|\s{2,}', line.strip())
        if len(parts) < 2:
            # Try single space split
            parts = line.split()
        if len(parts) < 2:
            continue

        try:
            freq_val = float(parts[0])
            eff_val = float(parts[1])
            freqs.append(freq_val)
            effs.append(eff_val)
        except (ValueError, IndexError):
            continue

    if not freqs:
        raise ValueError(f"No valid efficiency data found in {filename}")

    freqs = np.array(freqs)
    effs = np.array(effs)

    # Sort by frequency
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    effs = effs[sort_idx]

    # Determine frequency unit
    eff_freq_unit = freq_unit or header_freq_unit
    if eff_freq_unit:
        unit_map = {'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}
        freq_hz = freqs * unit_map.get(eff_freq_unit.lower(), 1.0)
    else:
        # Auto-detect
        freq_hz = freqs * _detect_freq_unit(freqs)

    # Determine efficiency format
    eff_fmt = eff_format or header_eff_format
    if eff_fmt is None:
        eff_fmt = _detect_efficiency_format(effs)

    if eff_fmt == 'db':
        # Convert dB to linear: linear = 10^(dB/10)
        eff_linear = np.power(10.0, effs / 10.0)
        # Clamp to [0, 1]
        eff_linear = np.clip(eff_linear, 0.0, 1.0)
    else:
        eff_linear = effs
        # If values look like percentages (all > 1), convert
        if np.all(eff_linear > 1.0) and np.max(eff_linear) <= 100.0:
            eff_linear = eff_linear / 100.0

    # Clamp to valid range
    eff_linear = np.clip(eff_linear, 0.0, 1.0)

    return EfficiencyData(
        frequencies_hz=freq_hz,
        efficiency_linear=eff_linear,
        source=filename,
    )


def load_efficiency_file(filepath: str) -> EfficiencyData:
    """Load efficiency data from a file path."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    import os
    return parse_efficiency_data(content, filename=os.path.basename(filepath))
