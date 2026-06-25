"""
Optenni ComponentLibrary Parser.

Parses the Optenni ComponentLibrary directory structure:
  Capacitors/index.xml  — series metadata for capacitors
  Inductors/index.xml   — series metadata for inductors
  Each series directory contains .s2p files and optionally index.xml / index.dat

Handles multi-vendor filename conventions:
  - AVX ACCU-P/C: C{size}SE{value}.s2p (EIA code / R notation, pF)
  - AVX ACCU-L/L: L{size}SE{value}.s2p (nH)
  - Coilcraft inductors: {size}{series}{value}.s2p (N notation / EIA-like)
  - Johanson caps: CAP_JOH_{size}..._{value}.s2p (explicit pF/nH)
  - Johanson inductors: IND_JOH_{size}..._{value}.s2p
  - Murata: Standard Murata part numbers → delegates to murata_parser
  - TDK caps: C{size}C0G...{value}.s2p (EIA code)
  - TDK inductors: {series}{value}.s2p (N notation)
  - Taiyo Yuden caps: {series}...{value}.s2p (EIA code)
  - Taiyo Yuden inductors: {series}_{value}_...s2p (N notation)
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from .murata_parser import (
    parse_murata_part_number, PartInfo, TOLERANCE_MAP,
    get_precision_rank, compute_tolerance
)


@dataclass
class OptenniSeriesInfo:
    """Series metadata from index.xml."""
    directory: str       # Folder name
    manufacturer: str    # e.g. "AVX", "Murata", "TDK"
    series: str          # e.g. "ACCU-P 01005", "GRM15"
    size_eia: str        # e.g. "01005", "0402", "0603"
    size_jis: str        # e.g. "0402", "1005", "1608"
    sizemm: str          # Physical size
    description: str     # Description text
    component_type: str  # "cap" or "ind"
    date: str            # Data date


@dataclass
class OptenniComponentData:
    """Parsed component data from the Optenni library."""
    part_number: str       # Derived part number (filename without .s2p)
    series_name: str       # Full series name from index.xml
    manufacturer: str      # Manufacturer
    component_type: str    # 'inductor' or 'capacitor'
    nominal_value: float   # In nH for inductors, pF for capacitors
    nominal_unit: str      # 'nH' or 'pF'
    tolerance_code: str    # Tolerance code
    tolerance_pct: float
    tolerance_abs: float
    size_code: str         # EIA size code
    value_str: str         # Raw value string from filename
    s2p_filename: str      # Relative path from series dir


# --- Top-level index.xml parser ---

def parse_optenni_index_xml(xml_path: str) -> List[OptenniSeriesInfo]:
    """Parse the top-level index.xml (Capacitors/index.xml or Inductors/index.xml)."""
    if not os.path.exists(xml_path):
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    series_list = []
    for elem in root.findall('ComponentSeries'):
        series = OptenniSeriesInfo(
            directory=elem.get('Directory', ''),
            manufacturer=elem.get('Manufacturer', ''),
            series=elem.get('Series', ''),
            size_eia=elem.get('sizeEIA', ''),
            size_jis=elem.get('sizeJIS', ''),
            sizemm=elem.get('sizemm', ''),
            description=elem.get('desc', ''),
            component_type=elem.get('type', ''),
            date=elem.get('date', ''),
        )
        series_list.append(series)
    
    return series_list


def parse_series_index_xml(xml_path: str) -> List[Dict]:
    """Parse a per-series index.xml with Component entries.
    
    Uses regex-based parsing because Optenni XML is sometimes malformed
    (e.g., missing spaces between attributes like RelTol="5"7-10-12").
    """
    if not os.path.exists(xml_path):
        return []
    
    with open(xml_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    components = []
    # Find all <Component .../> entries using regex (tolerates malformed XML)
    pattern = re.compile(
        r'<Component\s+'
        r'File="([^"]*)"\s+'
        r'Code="([^"]*)"\s+'
        r'Value="([^"]*)"\s+'
        r'AbsTol="([^"]*)"\s+'
        r'RelTol="([^"]*)"'
    )
    
    for match in pattern.finditer(content):
        file_val = match.group(1)
        code_val = match.group(2)
        value_str = match.group(3)
        abstol_str = match.group(4)
        reltol_str = match.group(5)
        
        # Clean reltol/abstol - sometimes date text gets appended
        reltol_str = re.sub(r'[^0-9.]', '', reltol_str)
        abstol_str = re.sub(r'[^0-9.]', '', abstol_str)
        
        try:
            value = float(value_str)
        except ValueError:
            continue
        
        comp = {
            'file': file_val,
            'code': code_val,
            'value': value,
            'abstol': abstol_str,
            'reltol': reltol_str,
            'max_current': '',
        }
        
        # Try to extract MaxCurrent from the rest of the line
        line_end = content[match.end():match.end()+100].split('>')[0]
        mc_match = re.search(r'MaxCurrent="([^"]*)"', line_end)
        if mc_match:
            comp['max_current'] = mc_match.group(1)
        
        components.append(comp)
    
    return components


# --- Vendor-specific filename parsers ---

def _parse_avx_accu_p(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse AVX ACCU-P capacitor filename: C0402SE0r1.s2p, C0402SE100.s2p"""
    name = filename.upper()
    if not name.endswith('.S2P'):
        name += '.S2P'
    basename = os.path.splitext(name)[0]
    
    # Pattern: C{size}SE{value}
    # size = 0402, 0603, etc.
    # value = 0r1 (0.1pF), 1r0 (1.0pF), 100 (10pF), 101 (100pF), etc.
    match = re.match(r'C(\d+)SE(.+)', basename)
    if not match:
        return None
    
    value_str = match.group(2)
    
    # Parse EIA / R value to pF
    value_pf = _parse_eia_value(value_str)
    if value_pf <= 0:
        return None
    
    # Extract EIA size code from the directory info
    size_code = series_info.size_eia
    
    # AVX ACCU-P caps are thin-film, tight tolerance
    # Default tolerance: typically B (±0.1pF) or F (±1%) depending on value
    tol_code, tol_pct, tol_abs = _guess_tolerance(value_pf, 'capacitor', 'AVX')
    
    return OptenniComponentData(
        part_number=basename,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type='capacitor',
        nominal_value=value_pf,
        nominal_unit='pF',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=value_str,
        s2p_filename=filename,
    )


def _parse_avx_accu_l(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse AVX ACCU-L inductor filename: L0402SE1r0.s2p, L0603SE10N.s2p"""
    name = filename.upper()
    basename = os.path.splitext(name)[0]
    
    # Pattern: L{size}SE{value}
    match = re.match(r'L(\d+)SE(.+)', basename)
    if not match:
        return None
    
    value_str = match.group(2)
    
    # Parse inductor value (nH)
    value_nh = _parse_inductor_value(value_str)
    if value_nh <= 0:
        return None
    
    size_code = series_info.size_eia
    
    # Default tolerance: typically G (±2%) or J (±5%)
    tol_code, tol_pct, tol_abs = _guess_tolerance(value_nh, 'inductor', 'AVX')
    
    return OptenniComponentData(
        part_number=basename,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type='inductor',
        nominal_value=value_nh,
        nominal_unit='nH',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=value_str,
        s2p_filename=filename,
    )


def _parse_coilcraft_inductor(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse Coilcraft inductor filenames:
    04CS10N.S2P, 04HP1N0.s2p, 08CS100.S2P, 0201HL-22N.s2p
    """
    name = os.path.splitext(filename)[0]
    
    # Pattern 1: {size}{series}{value}.S2P (e.g., 04CS10N, 04HP1N0, 08CS100)
    # size = 2-4 digits, series = 2-4 letters (non-greedy), value = alphanumeric
    match = re.match(r'(\d{2,4})([A-Za-z]{2,4}?)([A-Za-z0-9]+)', name)
    
    # Pattern 2: {series}-{value}.s2p (e.g., 0201HL-22N)
    if not match:
        match = re.match(r'([A-Za-z0-9]+)-([A-Za-z0-9]+)', name)
    
    if not match:
        return None
    
    value_str = match.group(len(match.groups()))
    
    # Parse inductor value
    value_nh = _parse_inductor_value(value_str)
    if value_nh <= 0:
        # Try as plain number (pF or nH)
        # Some Coilcraft caps use 3-digit codes like 100, 101 etc.
        val = _parse_eia_value(value_str)
        if val > 0:
            value_nh = val
        else:
            return None
    
    # Determine if inductor or capacitor based on series directory
    comp_type = 'inductor'  # default for Coilcraft inductors
    
    size_code = series_info.size_eia
    
    # Coilcraft typical tolerances
    tol_code, tol_pct, tol_abs = _guess_tolerance(value_nh, comp_type, 'Coilcraft')
    
    return OptenniComponentData(
        part_number=name,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type=comp_type,
        nominal_value=value_nh,
        nominal_unit='nH',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=value_str,
        s2p_filename=filename,
    )


def _parse_johanson(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse Johanson capacitor/inductor filenames:
    CAP_JOH_0402_001_R16_0p2pF_model-1.s2p
    IND_JOH_0201_001_F14_10nH_model.s2p
    """
    name = os.path.splitext(filename)[0]
    
    # Determine type from prefix
    if name.startswith('CAP_'):
        comp_type = 'capacitor'
        unit = 'pF'
    elif name.startswith('IND_'):
        comp_type = 'inductor'
        unit = 'nH'
    else:
        return None
    
    # Extract the value part - look for pattern like 0p2pF or 10nH
    # Johanson uses 'p' as decimal point in some values
    # Value patterns: 0p2pF (0.2 pF), 10pF (10 pF), 1p0nH (1.0 nH), 10nH (10 nH)
    
    val_match = re.search(r'(\d+(?:p\d+)?)(pF|nH)', name)
    if not val_match:
        return None
    
    val_str = val_match.group(1)
    val_unit = val_match.group(2)
    
    # Parse value, handling 'p' as decimal point
    val_str_clean = val_str.replace('p', '.')
    try:
        value = float(val_str_clean)
    except ValueError:
        return None
    
    # Determine actual unit
    if val_unit == 'pF':
        comp_type = 'capacitor'
        unit = 'pF'
    elif val_unit == 'nH':
        comp_type = 'inductor'
        unit = 'nH'
    else:
        return None
    
    size_code = series_info.size_eia
    tol_code, tol_pct, tol_abs = _guess_tolerance(value, comp_type, 'Johanson')
    
    return OptenniComponentData(
        part_number=name,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type=comp_type,
        nominal_value=value,
        nominal_unit=unit,
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=val_str,
        s2p_filename=filename,
    )


def _parse_tdk_capacitor(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse TDK capacitor filename: C1005C0G1H010B050BA.s2p
    Value is in EIA 3-digit code or R notation after voltage/temp codes.
    Format: C{size 4digits}{dielectric}{voltage}{value}{tol}{suffix}
    """
    name = os.path.splitext(filename)[0]
    
    # Skip the leading C + 4-digit size code (e.g., C0402, C1005, C1608)
    rest = name
    if rest.startswith('C'):
        rest = rest[1:]
    size_m = re.match(r'\d{4}', rest)
    if size_m:
        rest = rest[size_m.end():]  # now at dielectric/voltage
    
    # Try R notation first: e.g., R75 = 0.75 pF
    r_match = re.search(r'(\d*)R(\d+)([A-Z])', rest)
    if r_match:
        value_str = r_match.group(0)
        whole = r_match.group(1) or '0'
        frac = r_match.group(2)
        value_pf = float(f'{whole}.{frac}')
        tol_char = r_match.group(3)
        code = r_match.group(0)
    else:
        # Find EIA 3-digit code + tolerance letter
        eia_match = re.search(r'(\d{3})([A-Z])', rest)
        if not eia_match:
            return None
        code = eia_match.group(1)
        tol_char = eia_match.group(2)
        value_pf = _parse_eia_value(code)
        if value_pf <= 0:
            return None
    
    # Parse EIA code
    value_pf = _parse_eia_value(code)
    if value_pf <= 0:
        return None
    
    # Map TDK tolerance letters
    if tol_char in TOLERANCE_MAP:
        tol_code = tol_char
    else:
        tol_code = 'J'  # default ±5%
    
    tol_pct, tol_abs = compute_tolerance(tol_code, value_pf, 'capacitor')
    
    size_code = series_info.size_eia
    
    return OptenniComponentData(
        part_number=name,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type='capacitor',
        nominal_value=value_pf,
        nominal_unit='pF',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=code,
        s2p_filename=filename,
    )


def _parse_tdk_inductor(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse TDK inductor filename:
    MLG1005S0N3BT000.s2p, MHQ0402P_1N0_.s2p
    Value in N notation (0N3, 1N0, 10N) or R notation (R10).
    """
    name = os.path.splitext(filename)[0]
    
    # Skip the series prefix (letters + digits + letter, e.g., MLG1005S, MHQ0402P)
    rest = name
    series_m = re.match(r'[A-Za-z]+\d+[A-Za-z]?', rest)
    if series_m:
        rest = rest[series_m.end():]
    
    # Also strip leading undersores
    rest = rest.lstrip('_')
    
    # Try to find N/R notation value pattern (allow _ or letter as terminator)
    val_match = re.search(r'(\d*N\d+|\d+R\d+|R\d+|\d+N)([A-Z_])', rest)
    if not val_match:
        # Try alternative: value at start of rest, followed by letter or _
        val_match = re.search(r'(\d+\.?\d*)([A-Z_])', rest)
    
    if val_match:
        value_str = val_match.group(1)
        tol_char = val_match.group(2)
        # Clean up: if tol_char is underscore, use default tolerance
        if tol_char == '_':
            tol_char = 'J'
    else:
        # Last resort: find 3-digit code, but only look past the prefix
        num_match = re.search(r'(\d{3})([A-Z])', rest)
        if not num_match:
            return None
        value_str = num_match.group(1)
        tol_char = num_match.group(2)
    
    # Parse inductor value
    value_nh = _parse_inductor_value(value_str)
    if value_nh <= 0:
        # Try as EIA code (for higher values like 100 = 10nH)
        val = _parse_eia_value(value_str)
        if val > 0:
            value_nh = val
        else:
            return None
    
    if tol_char in TOLERANCE_MAP:
        tol_code = tol_char
    else:
        tol_code = 'J'  # default ±5%
    
    tol_pct, tol_abs = compute_tolerance(tol_code, value_nh, 'inductor')
    size_code = series_info.size_eia
    
    return OptenniComponentData(
        part_number=name,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type='inductor',
        nominal_value=value_nh,
        nominal_unit='nH',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=value_str,
        s2p_filename=filename,
    )


def _parse_taiyo_yuden_capacitor(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse Taiyo Yuden capacitor filenames:
    UMK105ABJ474_V-F.s2p, UMK105B7102_V-F.s2p
    EIA code embedded in filename.
    Format: {series}{dielectric}{valueEIA}{suffix}_{V-F|VHF|...}
    """
    name = os.path.splitext(filename)[0]
    
    # Use the base part before the first underscore (strip suffix)
    base = name.split('_')[0] if '_' in name else name
    
    # Try R notation first: e.g., 5R8 = 5.8 pF
    r_match = re.search(r'(\d+)R(\d+)', base)
    if r_match:
        whole = r_match.group(1)
        frac = r_match.group(2)
        value_pf = float(f'{whole}.{frac}')
        code = r_match.group(0)
    else:
        # Find the 3-digit EIA code that comes AFTER the dielectric character(s).
        # dielectrics are usually 1-3 letters (B, CG, CH, ABJ)
        # series prefix is letters+digits (e.g., UMK105, TMK042)
        # Strategy: find the LAST C4+ match: a letter followed by 4+ digits at end of base
        # Example: UMK105B7332 → dielecrtic='B', last 4 digits='7332', value='332'
        # Example: UMK105B7102 → dielecrtic='B', last 5 digits='7102', value='102'
        
        # Pattern: find dielectric boundary (letter then 3+ more chars)
        # The value EIA code is always 3 consecutive digits;
        # it may share digits with other info (e.g., 7332 has 733 and 332).
        # Use an overlapping approach: slide a 3-digit window backwards from end
        base_digits_only = re.sub(r'[^0-9]', '', base)
        
        code = None
        for end_pos in range(len(base_digits_only), 2, -1):
            candidate = base_digits_only[end_pos-3:end_pos]
            if len(candidate) == 3:
                val = _parse_eia_value(candidate)
                if 0.1 <= val <= 1e10:  # reasonable range
                    code = candidate
                    break
        
        if code is None:
            # Fallback: last 3-digit group (non-overlapping)
            digit_groups = re.findall(r'\d{3}', base)
            if not digit_groups:
                return None
            code = digit_groups[-1]
        
        value_pf = _parse_eia_value(code)
        if value_pf <= 0:
            return None
    
    size_code = series_info.size_eia
    tol_code = 'J'  # Taiyo Yuden defaults to ±5% typically
    tol_pct, tol_abs = compute_tolerance(tol_code, value_pf, 'capacitor')
    
    return OptenniComponentData(
        part_number=name,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type='capacitor',
        nominal_value=value_pf,
        nominal_unit='pF',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=code,
        s2p_filename=filename,
    )


def _parse_taiyo_yuden_inductor(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse Taiyo Yuden inductor filenames:
    HK1005_10N_-T.s2p, HKQ0402_1N0_-E.s2p, AQ105_1N0_.s2p
    Pattern: {series}_{value}_...s2p
    """
    name = os.path.splitext(filename)[0]
    
    # Extract value part between underscores
    # Pattern: SERIES_VALUE_-SUFFIX.s2p
    parts = name.split('_')
    if len(parts) < 2:
        return None
    
    # The value is typically in the second underscore-separated part
    value_str = parts[1]
    
    # Parse N notation value
    value_nh = _parse_inductor_value(value_str)
    if value_nh <= 0:
        # Try as plain EIA code
        val = _parse_eia_value(value_str)
        if val > 0:
            value_nh = val
        else:
            return None
    
    size_code = series_info.size_eia
    # Default tolerance
    tol_code = 'J'
    tol_pct, tol_abs = compute_tolerance(tol_code, value_nh, 'inductor')
    
    return OptenniComponentData(
        part_number=name,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type='inductor',
        nominal_value=value_nh,
        nominal_unit='nH',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        value_str=value_str,
        s2p_filename=filename,
    )


def _parse_murata_in_optenni(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse Murata part number found in Optenni library.
    Delegates to the existing murata_parser.
    """
    name = os.path.splitext(filename)[0]
    category = 'capacitor' if series_info.component_type == 'cap' else 'inductor'
    
    part_info = parse_murata_part_number(name, known_type=category)
    if part_info is None:
        return None
    
    return OptenniComponentData(
        part_number=part_info.part_number,
        series_name=series_info.series,
        manufacturer=series_info.manufacturer,
        component_type=part_info.component_type,
        nominal_value=part_info.nominal_value,
        nominal_unit=part_info.nominal_unit,
        tolerance_code=part_info.tolerance_code,
        tolerance_pct=part_info.tolerance_pct,
        tolerance_abs=part_info.tolerance_abs,
        size_code=part_info.size_code or series_info.size_eia,
        value_str=part_info.value_str,
        s2p_filename=filename,
    )


# --- Value parsing helpers ---

def _parse_eia_value(code: str) -> float:
    """Parse an EIA 3-digit capacitor code or R notation to pF.
    100 = 10pF, 101 = 100pF, 104 = 100000pF, 
    0R1 = 0.1pF, 1R0 = 1.0pF, R50 = 0.5pF
    """
    s = code.upper().strip()
    
    # R notation
    r_match = re.match(r'(\d*)R(\d+)', s)
    if r_match:
        whole = r_match.group(1) or '0'
        frac = r_match.group(2)
        return float(f'{whole}.{frac}')
    
    # 3-digit EIA
    if len(s) == 3 and s.isdigit():
        digits = int(s[:2])
        multiplier = int(s[2])
        return float(digits * (10 ** multiplier))
    
    # 2-digit (very small values)
    if len(s) == 2 and s.isdigit():
        return float(s)
    
    # Try as plain number
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_inductor_value(value_str: str) -> float:
    """Parse inductor value from various notations to nH.
    1N0 = 1.0 nH, 10N = 10 nH, 0N6 = 0.6 nH,
    R10 = 0.10 nH, 1R5 = 1.5 nH, 100 = 10 nH (EIA code)
    """
    s = value_str.upper().strip()
    
    # N notation: digit(s) + N + digit(s) or digit(s) + N
    n_match = re.match(r'(\d+)N(\d+)', s)
    if n_match:
        return float(f'{n_match.group(1)}.{n_match.group(2)}')
    
    n_int_match = re.match(r'(\d+)N$', s)
    if n_int_match:
        return float(n_int_match.group(1))
    
    # N at start: N{digits}
    n_start = re.match(r'N(\d+)', s)
    if n_start:
        return float(f'0.{n_start.group(1)}')
    
    # R notation: digit(s) + R + digit(s) or R + digit(s)
    r_match = re.match(r'(\d*)R(\d+)', s)
    if r_match:
        whole = r_match.group(1) or '0'
        frac = r_match.group(2)
        return float(f'{whole}.{frac}')
    
    # Plain digits - try EIA code (100 = 10 nH)
    if s.isdigit() and len(s) == 3:
        digits = int(s[:2])
        mult = int(s[2])
        return float(digits * (10 ** mult))
    
    # Try as plain number
    try:
        return float(s)
    except ValueError:
        return 0.0


def _guess_tolerance(value: float, comp_type: str, manufacturer: str) -> Tuple[str, float, float]:
    """Guess a reasonable tolerance based on manufacturer and value."""
    # Defaults
    if comp_type == 'capacitor':
        if value < 10:
            # Small caps use absolute tolerance
            tol_code = 'B'  # ±0.1pF
        else:
            tol_code = 'J'  # ±5%
    else:  # inductor
        if value < 10:
            tol_code = 'J'  # ±5%
        else:
            tol_code = 'J'  # ±5%
    
    tol_pct, tol_abs = compute_tolerance(tol_code, value, comp_type)
    return tol_code, tol_pct, tol_abs


# --- Per-series index.xml parser (HKQ0603W style) ---

def parse_per_series_xml_values(xml_path: str, 
                                 series_info: OptenniSeriesInfo) -> List[OptenniComponentData]:
    """Parse per-series index.xml with exact component entries (e.g., HKQ0603W)."""
    comps = parse_series_index_xml(xml_path)
    if not comps:
        return []
    
    results = []
    for c in comps:
        filename = c['file']
        value = c['value']
        code = c['code']
        abstol_str = c['abstol']
        reltol_str = c['reltol']
        
        # Determine component type
        comp_type = 'inductor' if series_info.component_type == 'ind' else 'capacitor'
        unit = 'nH' if comp_type == 'inductor' else 'pF'
        
        # Determine tolerance
        if reltol_str and reltol_str.strip():
            tol_pct = float(reltol_str)
            tol_code = _pct_to_tolerance_code(tol_pct)
            tol_abs = value * tol_pct / 100.0
        elif abstol_str and abstol_str.strip():
            tol_abs = float(abstol_str)
            tol_code = _abs_to_tolerance_code(tol_abs, value, comp_type)
            tol_pct = (tol_abs / value * 100.0) if value > 0 else 20.0
        else:
            tol_code = 'J'
            tol_pct, tol_abs = compute_tolerance(tol_code, value, comp_type)
        
        # Use code if available, else use filename stem as part number
        part_number = code if code else os.path.splitext(filename)[0]
        
        results.append(OptenniComponentData(
            part_number=part_number,
            series_name=series_info.series,
            manufacturer=series_info.manufacturer,
            component_type=comp_type,
            nominal_value=value,
            nominal_unit=unit,
            tolerance_code=tol_code,
            tolerance_pct=tol_pct,
            tolerance_abs=tol_abs,
            size_code=series_info.size_eia,
            value_str=str(value),
            s2p_filename=filename,
        ))
    
    return results


def _pct_to_tolerance_code(pct: float) -> str:
    """Convert percent tolerance to tolerance code."""
    mapping = {1.0: 'F', 2.0: 'G', 3.0: 'H', 5.0: 'J', 10.0: 'K', 20.0: 'M'}
    closest = min(mapping.keys(), key=lambda k: abs(k - pct))
    return mapping[closest]


def _abs_to_tolerance_code(abs_val: float, nominal: float, comp_type: str) -> str:
    """Convert absolute tolerance to tolerance code."""
    # Typical absolute tolerance codes
    if abs_val <= 0.02:
        return 'T'
    elif abs_val <= 0.05:
        return 'A'
    elif abs_val <= 0.1:
        return 'B'
    elif abs_val <= 0.25:
        return 'C'
    elif abs_val <= 0.5:
        return 'D'
    else:
        # Fallback to percentage-based
        if nominal > 0:
            pct = abs_val / nominal * 100
            return _pct_to_tolerance_code(pct)
        return 'J'


# --- Main dispatch: parse one .s2p filename in a series context ---

def parse_optenni_component(filename: str, series_info: OptenniSeriesInfo) -> Optional[OptenniComponentData]:
    """Parse a single .s2p filename from the Optenni library using the best strategy."""
    mfr = series_info.manufacturer.lower()
    
    # Murata parts - use existing parser
    if 'murata' in mfr:
        return _parse_murata_in_optenni(filename, series_info)
    
    # AVX
    if 'avx' in mfr:
        comp_type = series_info.component_type
        if comp_type == 'cap':
            return _parse_avx_accu_p(filename, series_info)
        else:
            return _parse_avx_accu_l(filename, series_info)
    
    # Johanson
    if 'johanson' in mfr:
        return _parse_johanson(filename, series_info)
    
    # Coilcraft
    if 'coilcraft' in mfr:
        return _parse_coilcraft_inductor(filename, series_info)
    
    # TDK
    if 'tdk' in mfr:
        comp_type = series_info.component_type
        if comp_type == 'cap':
            return _parse_tdk_capacitor(filename, series_info)
        else:
            return _parse_tdk_inductor(filename, series_info)
    
    # Taiyo Yuden
    if 'taiyo yuden' in mfr:
        comp_type = series_info.component_type
        if comp_type == 'cap':
            return _parse_taiyo_yuden_capacitor(filename, series_info)
        else:
            return _parse_taiyo_yuden_inductor(filename, series_info)
    
    # Fallback: try all parsers
    for parser in [_parse_murata_in_optenni, _parse_avx_accu_p, _parse_avx_accu_l,
                    _parse_johanson, _parse_coilcraft_inductor,
                    _parse_tdk_capacitor, _parse_tdk_inductor,
                    _parse_taiyo_yuden_capacitor, _parse_taiyo_yuden_inductor]:
        result = parser(filename, series_info)
        if result is not None:
            return result
    
    return None


# --- Directory scanning ---

def scan_optenni_library(library_root: str) -> List[Tuple[OptenniComponentData, str]]:
    """
    Scan the entire Optenni ComponentLibrary and return all parsed components.
    
    Returns:
        List of (OptenniComponentData, full_path_to_s2p) tuples
    """
    results = []
    
    # Parse top-level index.xml files
    for category in ['Capacitors', 'Inductors']:
        index_xml = os.path.join(library_root, category, 'index.xml')
        if not os.path.exists(index_xml):
            continue
        
        series_list = parse_optenni_index_xml(index_xml)
        
        for series_info in series_list:
            series_dir = os.path.join(library_root, category, series_info.directory)
            if not os.path.exists(series_dir):
                continue
            
            # Check for per-series index.xml
            series_xml = os.path.join(series_dir, 'index.xml')
            has_series_xml = os.path.exists(series_xml)
            
            if has_series_xml:
                # Parse exact values from series index.xml
                xml_comps = parse_per_series_xml_values(series_xml, series_info)
                for comp in xml_comps:
                    s2p_path = os.path.join(series_dir, comp.s2p_filename)
                    if os.path.exists(s2p_path):
                        results.append((comp, s2p_path))
                # Still fall through to scan remaining .s2p files not in XML
                xml_files = {c.s2p_filename for c in xml_comps}
            else:
                xml_files = set()
            
            # Scan .s2p files
            for fname in sorted(os.listdir(series_dir)):
                if not fname.lower().endswith('.s2p'):
                    continue
                if fname in xml_files:
                    continue  # Already parsed from XML
                
                comp = parse_optenni_component(fname, series_info)
                if comp is not None:
                    s2p_path = os.path.join(series_dir, fname)
                    results.append((comp, s2p_path))
    
    return results


# --- Convenience ---

def get_manufacturer_summary(comp_tuples: List[Tuple[OptenniComponentData, str]]) -> Dict:
    """Summary of components by manufacturer."""
    from collections import Counter
    mfr_count = Counter(c.manufacturer for c, _ in comp_tuples)
    type_count = Counter(c.component_type for c, _ in comp_tuples)
    
    return {
        'total': len(comp_tuples),
        'by_manufacturer': dict(mfr_count.most_common()),
        'by_type': dict(type_count),
    }


if __name__ == '__main__':
    # Test
    lib_root = r'C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary'
    
    print("Scanning Optenni ComponentLibrary...")
    print(f"  Root: {lib_root}")
    print()
    
    comps = scan_optenni_library(lib_root)
    
    summary = get_manufacturer_summary(comps)
    print(f"Total components parsed: {summary['total']}")
    print(f"By type: {summary['by_type']}")
    print()
    print("By manufacturer:")
    for mfr, count in summary['by_manufacturer'].items():
        print(f"  {mfr}: {count}")
    print()
    
    # Show some samples
    if comps:
        print("Sample components:")
        for comp, path in comps[:10]:
            print(f"  {comp.part_number:<35} {comp.component_type:<10} "
                  f"{comp.nominal_value:>10.3f} {comp.nominal_unit:<4} "
                  f"{comp.tolerance_code:>4} {comp.manufacturer:<15} "
                  f"{comp.series_name:<25}")
    else:
        print("No components parsed!")
