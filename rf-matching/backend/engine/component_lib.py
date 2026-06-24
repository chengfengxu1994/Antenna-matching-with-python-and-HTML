"""
Component library: load and manage Murata S2P component data.

Handles:
- Scanning ZIP files in the Murata directory
- Extracting component metadata (part number, type, nominal value)
- Caching parsed S2P data for quick access
- Fast frequency lookup and interpolation
"""

import os
import re
import zipfile
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .touchstone import parse_touchstone, TouchstoneData, FREQ_MULTIPLIERS


@dataclass
class ComponentInfo:
    """Metadata for a single Murata component."""
    part_number: str
    s2p_filename: str
    zip_path: str
    component_type: str  # 'inductor' or 'capacitor'
    nominal_value: float  # in nH for inductors, pF for capacitors
    nominal_unit: str  # 'nH' or 'pF'
    # Cached S-parameter data (lazy loaded)
    _data: Optional[TouchstoneData] = field(default=None, repr=False)

    @property
    def data(self) -> TouchstoneData:
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> TouchstoneData:
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.s2p_filename) as f:
                content = f.read().decode('utf-8', errors='replace')
        return parse_touchstone(content, filename=self.s2p_filename)

    def get_s_matrix_at_freq(self, freq_hz: float) -> np.ndarray:
        """Get 2x2 S-matrix interpolated at target frequency."""
        return self.data.get_s_matrix_interpolated(freq_hz)

    @property
    def label(self) -> str:
        return f"{self.part_number} ({self.nominal_value} {self.nominal_unit})"


@dataclass
class ComponentLibrary:
    """Complete Murata component library."""
    inductors: List[ComponentInfo] = field(default_factory=list)
    capacitors: List[ComponentInfo] = field(default_factory=list)
    # Fast lookup: by nominal value
    _inductors_by_value: Dict[float, List[ComponentInfo]] = field(default_factory=dict)
    _capacitors_by_value: Dict[float, List[ComponentInfo]] = field(default_factory=dict)

    @property
    def all_components(self) -> List[ComponentInfo]:
        return self.inductors + self.capacitors

    def add_component(self, comp: ComponentInfo):
        if comp.component_type == 'inductor':
            self.inductors.append(comp)
            v = comp.nominal_value
            if v not in self._inductors_by_value:
                self._inductors_by_value[v] = []
            self._inductors_by_value[v].append(comp)
        else:
            self.capacitors.append(comp)
            v = comp.nominal_value
            if v not in self._capacitors_by_value:
                self._capacitors_by_value[v] = []
            self._capacitors_by_value[v].append(comp)

    def get_unique_inductor_values(self) -> List[float]:
        return sorted(self._inductors_by_value.keys())

    def get_unique_capacitor_values(self) -> List[float]:
        return sorted(self._capacitors_by_value.keys())

    def get_inductors_near(self, target_nh: float, tolerance: float = 0.5) -> List[ComponentInfo]:
        """Get inductors with nominal value within tolerance of target (nH)."""
        result = []
        for val, comps in self._inductors_by_value.items():
            if abs(val - target_nh) / max(target_nh, 1e-9) <= tolerance:
                result.extend(comps)
        return result

    def get_capacitors_near(self, target_pf: float, tolerance: float = 0.5) -> List[ComponentInfo]:
        """Get capacitors with nominal value within tolerance of target (pF)."""
        result = []
        for val, comps in self._capacitors_by_value.items():
            if abs(val - target_pf) / max(target_pf, 1e-9) <= tolerance:
                result.extend(comps)
        return result

    def find_nearest_inductor(self, target_nh: float) -> Optional[ComponentInfo]:
        """Find the inductor closest to the target value."""
        values = self.get_unique_inductor_values()
        if not values:
            return None
        nearest_val = min(values, key=lambda v: abs(v - target_nh))
        return self._inductors_by_value[nearest_val][0]

    def find_nearest_capacitor(self, target_pf: float) -> Optional[ComponentInfo]:
        """Find the capacitor closest to the target value."""
        values = self.get_unique_capacitor_values()
        if not values:
            return None
        nearest_val = min(values, key=lambda v: abs(v - target_pf))
        return self._capacitors_by_value[nearest_val][0]


# Murata part number patterns
# Inductor: LQP03TN10NH02 → 10N = 10 nH, LQG15HH2N0S02 → 2N0 = 2.0 nH
# Capacitor: GRM1555C1H101JA01 → 101 = 10*10^1 = 100 pF

def parse_murata_part(part_number: str) -> Tuple[str, float, str]:
    """
    Parse a Murata part number to extract type and nominal value.

    Returns:
        (type, value, unit) where type is 'inductor' or 'capacitor'
    """
    pn = part_number.upper().strip()

    # Inductor patterns: typically contain N, NH, or UH in the value section
    # LQ* series: inductors
    # LQW* series: wire-wound inductors
    inductor_prefixes = ['LQP', 'LQG', 'LQW', 'LQH']

    is_inductor = any(pn.startswith(p) for p in inductor_prefixes)

    if is_inductor:
        value, unit = _parse_inductor_value(pn)
        return ('inductor', value, unit)
    else:
        # Capacitor: GRM, GJM, GCM, etc. (MLCC)
        value, unit = _parse_capacitor_value(pn)
        return ('capacitor', value, unit)


def _parse_inductor_value(pn: str) -> Tuple[float, str]:
    """
    Parse inductor value from Murata part number.
    Examples: LQP03TN10NH02 → 10 nH
              LQG15HH2N0S02 → 2.0 nH
              LQP03TN0N6B02 → 0.6 nH
    """
    # Look for patterns like: XXN, XNXX, XNX, XXNH, XXUH
    # Common: digits followed by N (nH) or U (uH)
    # The value encoding varies by series

    # Pattern: digits + 'N' + optional digits (for nH)
    # or digits + 'U' + optional digits (for uH)
    match = re.search(r'(\d+R?\d*)(N|NH|UH|U)(\d*[A-Z]?)', pn)
    if match:
        value_str = match.group(1).replace('R', '.')
        unit_code = match.group(2)[0]  # N or U
        try:
            base = float(value_str)
            if unit_code == 'N':
                return (base, 'nH')
            elif unit_code == 'U':
                return (base * 1000.0, 'nH')  # convert uH to nH
        except ValueError:
            pass

    # Fallback: try other patterns
    # LQP03TN0N6B02 → 0N6 = 0.6 nH
    match = re.search(r'(\d+)N(\d+)', pn)
    if match:
        whole = int(match.group(1))
        frac = int(match.group(2))
        return (float(f"{whole}.{frac}"), 'nH')

    return (0.0, 'nH')


def _parse_capacitor_value(pn: str) -> Tuple[float, str]:
    """
    Parse capacitor value from Murata part number.
    Examples: GJM0335C1E1R0BB01 → 1R0 = 1.0 pF
              GJM0335C1ER40BB01 → R40 = 0.40 pF
              GRM1555C1H101JA01 → 101 = 10*10^1 = 100 pF
              GRM155R71C104KA88 → 104 = 10*10^4 = 100000 pF
    """
    # Strategy 1: Find R-notation (e.g., 1R0, R40, 2R5)
    # This is common for sub-10pF values
    r_match = re.search(r'(\d*)R(\d+)', pn)
    if r_match:
        whole = r_match.group(1)
        frac = r_match.group(2)
        val = float(f"{whole if whole else '0'}.{frac}")
        if 0.01 < val < 1000:
            return (val, 'pF')

    # Strategy 2: Find 3-digit EIA code followed by a letter
    # Skip dimension codes (335, 155, 355, etc. at the start)
    matches = list(re.finditer(r'(\d{3})([A-Z])', pn))

    for match in matches:
        code = match.group(1)
        pos = match.start()
        # Skip if at the very start (dimension codes like 0335, 155)
        if pos < 6:
            continue
        try:
            digits = int(code[:2])
            multiplier = int(code[2])
            value_pf = digits * (10 ** multiplier)
            if 0.5 <= value_pf <= 100000000 and multiplier <= 6:
                return (value_pf, 'pF')
        except ValueError:
            continue

    # Strategy 3: R-notation at end
    r_match2 = re.search(r'(\d+)R(\d+)$', pn)
    if r_match2:
        return (float(f"{r_match2.group(1)}.{r_match2.group(2)}"), 'pF')

    return (0.0, 'pF')


def scan_murata_directory(murata_dir: str) -> ComponentLibrary:
    """
    Scan Murata directory for ZIP files containing S2P data.

    Directory structure:
        murata_dir/
            sparameter-inductor.../
                series1.zip
                series2.zip
            sparameter-mlcc.../
                series1.zip
                ...
    """
    library = ComponentLibrary()

    for root, dirs, files in os.walk(murata_dir):
        for f in files:
            if not f.lower().endswith('.zip'):
                continue

            zip_path = os.path.join(root, f)
            category = 'inductor' if 'inductor' in root.lower() else 'capacitor'

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if not name.lower().endswith('.s2p'):
                            continue

                        # Extract part number from filename (basename only, no directory)
                        part_number = os.path.splitext(os.path.basename(name))[0]
                        comp_type, nominal_value, nominal_unit = parse_murata_part(part_number)

                        # Override type based on directory structure
                        if comp_type != category:
                            comp_type = category

                        comp = ComponentInfo(
                            part_number=part_number,
                            s2p_filename=name,
                            zip_path=zip_path,
                            component_type=comp_type,
                            nominal_value=nominal_value,
                            nominal_unit=nominal_unit,
                        )
                        library.add_component(comp)

            except zipfile.BadZipFile:
                continue

    return library
