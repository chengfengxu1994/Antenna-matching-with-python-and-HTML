"""
Enhanced Murata Part Number Parser.

Accurately extracts:
- Series name (e.g., LQP03TN, GRM1555C)
- Component type (inductor/capacitor)
- Nominal value (nH for inductors, pF for capacitors)
- Tolerance code and absolute tolerance
- Size code
- Voltage characteristic (capacitors)
- Temperature characteristic (capacitors)

Tolerance codes (industry standard):
  B = ±0.1pF (caps <10pF) or ±0.1nH (inductors)
  C = ±0.25pF or ±0.2nH
  D = ±0.5pF
  F = ±1%
  G = ±2%
  H = ±3%  (for inductors) / ±? (for caps, rare)
  J = ±5%
  K = ±10%
  M = ±20%
  W = ±0.05nH (some inductors)
"""

import re
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class PartInfo:
    """Parsed Murata part number information."""
    part_number: str
    series: str           # e.g., 'LQP03TN', 'GRM1555C', 'LQW04AN'
    component_type: str   # 'inductor' or 'capacitor'
    nominal_value: float  # nH for inductors, pF for capacitors
    nominal_unit: str     # 'nH' or 'pF'
    tolerance_code: str   # B, C, D, F, G, H, J, K, M, W
    tolerance_pct: float  # Tolerance as percentage (e.g., 5.0 for J)
    tolerance_abs: float  # Absolute tolerance in nH or pF
    size_code: str        # e.g., '03', '15', '0402', '1608'
    voltage_code: str     # e.g., '1H', '2D' (capacitors only)
    value_str: str        # Raw value string from part number
    dielectric: str       # 'C0G', 'X7R', 'X5R', etc. (capacitors only)


# Tolerance code → (percentage, absolute_description)
TOLERANCE_MAP = {
    'B': (None, 0.1),     # ±0.1 (pF for caps, nH for small inductors)
    'C': (None, 0.25),    # ±0.25
    'D': (None, 0.5),     # ±0.5
    'F': (1.0, None),     # ±1%
    'G': (2.0, None),     # ±2%
    'H': (3.0, None),     # ±3%
    'J': (5.0, None),     # ±5%
    'K': (10.0, None),    # ±10%
    'M': (20.0, None),    # ±20%
    'W': (None, 0.05),    # ±0.05 (very tight)
    'Q': (None, 0.02),    # ±0.02 (some precision caps)
    'A': (None, 0.05),    # ±0.05 (some precision caps)
    'T': (None, 0.01),    # ±0.01 (ultra precision)
}

# Precision ranking (lower = better)
PRECISION_RANK = {
    'T': 0, 'Q': 1, 'A': 2, 'W': 3, 'B': 4, 'C': 5, 'D': 6,
    'F': 7, 'G': 8, 'H': 9, 'J': 10, 'K': 11, 'M': 12,
}

# Known inductor series prefixes
INDUCTOR_PREFIXES = ('LQP', 'LQG', 'LQW', 'LQH', 'LQM')

# Known capacitor series prefixes  
CAPACITOR_PREFIXES = ('GRM', 'GCM', 'GJM', 'GQM', 'GCQ', 'GCJ', 'GCE',
                      'GCH', 'GCD', 'GC3', 'GR3', 'GR4', 'GR7', 'GRJ',
                      'GRT', 'GXT', 'GGD', 'GGM', 'GA2', 'GA3', 'GJ4',
                      'KC3', 'KC9', 'KCA', 'KCM', 'KR3', 'KR9', 'KRM',
                      'KRT', 'LLC', 'LLL', 'ZRA', 'ZRB')


def compute_tolerance(tolerance_code: str, nominal_value: float, 
                      component_type: str) -> Tuple[float, float]:
    """
    Compute tolerance in percentage and absolute value.
    
    Returns (tolerance_pct, tolerance_abs) where one may be derived.
    """
    code = tolerance_code.upper()
    if code not in TOLERANCE_MAP:
        # Unknown tolerance, assume worst case
        return (20.0, nominal_value * 0.2)
    
    pct, abs_val = TOLERANCE_MAP[code]
    
    if pct is not None:
        tolerance_pct = pct
        tolerance_abs = nominal_value * pct / 100.0
    else:
        tolerance_abs = abs_val
        if nominal_value > 0:
            tolerance_pct = (abs_val / nominal_value) * 100.0
        else:
            tolerance_pct = 0.0
    
    return (tolerance_pct, tolerance_abs)


def _parse_inductor_value_and_tolerance(pn_after_prefix: str) -> Tuple[float, str, str, str]:
    """
    Parse inductor value from the part number suffix (after series prefix).
    
    Patterns:
    - 10NH02 → 10 nH, H tolerance
    - 0N6B02 → 0.6 nH, B tolerance
    - 9N7D00 → 9.7 nH, D tolerance
    - R22J02 → 0.22 nH, J tolerance
    - R10J00 → 0.10 nH, J tolerance
    - 1R5H02 → 1.5 nH, H tolerance
    - 1N0C00 → 1.0 nH, C tolerance
    - 2N2D00 → 2.2 nH, D tolerance
    - N40B02 → 0.40 nH, B tolerance (some series)
    - 4N7H02 → 4.7 nH, H tolerance
    
    Returns (value_nH, tolerance_code, value_str, suffix_after_tolerance)
    """
    s = pn_after_prefix.upper().strip()
    
    # Pattern 1: R notation at start (e.g., R22J, R10J) → decimal nH
    match = re.match(r'R(\d+)([A-Z])(.*)', s)
    if match:
        frac_str = match.group(1)
        tol_code = match.group(2)
        value = float(f'0.{frac_str}')
        return (value, tol_code, f'R{frac_str}', match.group(3))
    
    # Pattern 2: N notation - digit(s) + N + digit(s) (e.g., 0N6, 9N7, 1N0, 2N2)
    match = re.match(r'(\d+)N(\d+)([A-Z])(.*)', s)
    if match:
        whole = match.group(1)
        frac = match.group(2)
        tol_code = match.group(3)
        value = float(f'{whole}.{frac}')
        return (value, tol_code, f'{whole}N{frac}', match.group(4))
    
    # Pattern 3: N notation - just N + digit(s) (e.g., N40B)
    match = re.match(r'N(\d+)([A-Z])(.*)', s)
    if match:
        frac_str = match.group(1)
        tol_code = match.group(2)
        value = float(f'0.{frac_str}')
        return (value, tol_code, f'N{frac_str}', match.group(3))
    
    # Pattern 4: digit(s) + N (no fractional part, e.g., 10N, 33N) → integer nH
    # Must be followed by tolerance letter
    match = re.match(r'(\d+)N([A-Z])(.*)', s)
    if match:
        int_str = match.group(1)
        tol_code = match.group(2)
        value = float(int_str)
        return (value, tol_code, f'{int_str}N', match.group(3))
    
    # Pattern 5: digits + R + digits (e.g., 1R5)
    match = re.match(r'(\d+)R(\d+)([A-Z])(.*)', s)
    if match:
        whole = match.group(1)
        frac = match.group(2)
        tol_code = match.group(3)
        value = float(f'{whole}.{frac}')
        return (value, tol_code, f'{whole}R{frac}', match.group(4))
    
    # Pattern 6: digit(s) + UH notation (µH → nH)
    match = re.match(r'(\d+R?\d*)UH?([A-Z])(.*)', s)
    if match:
        val_str = match.group(1).replace('R', '.')
        tol_code = match.group(2)
        value = float(val_str) * 1000.0  # µH to nH
        return (value, tol_code, match.group(1), match.group(3))
    
    # Fallback: try to find any value pattern
    match = re.match(r'(\d+\.?\d*)([A-Z])(.*)', s)
    if match:
        val_str = match.group(1)
        tol_code = match.group(2)
        try:
            value = float(val_str)
            return (value, tol_code, val_str, match.group(3))
        except ValueError:
            pass
    
    return (0.0, '?', '', s)


def _parse_capacitor_eia_code(value_code: str) -> Tuple[float, str]:
    """
    Parse EIA capacitor value code.
    
    EIA 3-digit codes: ABx = A.B * 10^x pF
    - 100 = 10 pF
    - 101 = 100 pF
    - 104 = 100,000 pF = 100 nF
    - 223 = 22,000 pF = 22 nF
    - 472 = 4,700 pF
    
    R notation:
    - R50 = 0.50 pF
    - 1R0 = 1.0 pF
    - 2R2 = 2.2 pF
    
    2-digit codes (small caps):
    - 10 = 10 pF (sometimes used for very small values)
    
    Returns (value_pF, value_str)
    """
    vc = value_code.upper().strip()
    
    # R notation: digit(s) + R + digit(s) or R + digit(s)
    match = re.match(r'(\d*)R(\d+)', vc)
    if match:
        whole = match.group(1) or '0'
        frac = match.group(2)
        value = float(f'{whole}.{frac}')
        return (value, vc)
    
    # 3-digit EIA code
    if len(vc) == 3 and vc.isdigit():
        digits = int(vc[:2])
        multiplier = int(vc[2])
        value = digits * (10 ** multiplier)
        return (float(value), vc)
    
    # 2-digit code (rare, for very small caps)
    if len(vc) == 2 and vc.isdigit():
        return (float(vc), vc)
    
    # Try as a plain number
    try:
        return (float(vc), vc)
    except ValueError:
        return (0.0, vc)


def _extract_capacitor_value_from_part(pn: str) -> Tuple[float, str, str, str]:
    """
    Extract capacitor value code from a Murata MLCC part number.
    
    Strategy:
    1. Find the position after voltage/temp characteristic codes
    2. The value code (3-digit EIA or R notation) follows
    3. The tolerance letter follows the value code
    
    Voltage codes in Murata parts: digit + letter combinations after type code
    E.g., 1H, 1C, 1E, 2A, 2D, 3A, etc.
    """
    # Find all possible value code positions
    # The value code pattern: 3 digits followed by a letter, or contains R
    
    # First try: find 3-digit EIA code followed by a tolerance letter
    # Position must be after the first ~8 chars (past series+size+type+voltage)
    
    # More robust approach: find the value code by looking for the pattern
    # voltage_code(2char) + value_code(3digit or R notation) + tolerance_letter
    
    # Try pattern: ...Xvoltage(2) + value(3digit) + tolerance(1letter)...
    # Voltage pattern: digit + uppercase letter (but not at the very start)
    
    s = pn.upper()
    
    # Find R-notation value (e.g., 1R0, R50, 2R2)
    r_matches = list(re.finditer(r'(\d*R\d+)', s))
    for m in r_matches:
        pos = m.start()
        # Must be at least 6 chars in (past series+size)
        if pos >= 5:
            # Check that a tolerance letter follows
            end_pos = m.end()
            if end_pos < len(s) and s[end_pos] in TOLERANCE_MAP:
                return (0.0, m.group(1), s[end_pos], s[end_pos+1:])
    
    # Find 3-digit EIA code (digit digit digit letter)
    # The key: must be preceded by a voltage code (digit+letter at ~position 6-10)
    eia_matches = list(re.finditer(r'(\d{3})([A-Z])', s))
    
    best = None
    for m in eia_matches:
        code = m.group(1)
        tol_char = m.group(2)
        pos = m.start()
        
        # Must be at position 6+ (past series prefix + size + type)
        if pos < 5:
            continue
        
        # The tolerance character must be a known tolerance code
        if tol_char not in TOLERANCE_MAP:
            continue
        
        # Check: the 2 chars before the value code should be voltage/temp
        # Pattern: digit + letter (e.g., 1H, 2D, 1C, 80)
        if pos >= 2:
            pre2 = s[pos-2:pos]
            if pre2[0].isdigit() and pre2[1].isalpha():
                # This looks like voltage+temp code before value
                digits = int(code[:2])
                multiplier = int(code[2])
                # Validate: reasonable EIA code
                if 0 <= multiplier <= 9 and 1 <= digits <= 99:
                    value = digits * (10 ** multiplier)
                    if 0.1 <= value <= 1e9:  # reasonable capacitor range
                        return (float(value), code, tol_char, s[m.end()+1:])
        
        # Also accept without explicit voltage code check
        if best is None:
            digits = int(code[:2])
            multiplier = int(code[2])
            if 0 <= multiplier <= 6 and 1 <= digits <= 99:
                value = digits * (10 ** multiplier)
                if pos >= 6 and 1.0 <= value <= 1e8:
                    best = (float(value), code, tol_char, s[m.end()+1:])
    
    if best:
        return best
    
    return (0.0, '', '?', s)


def parse_murata_part_number(part_number: str, 
                             known_type: Optional[str] = None) -> Optional[PartInfo]:
    """
    Parse a Murata part number and extract all metadata.
    
    Args:
        part_number: The part number string (e.g., 'GRM1555C1H101JA01')
        known_type: Optional override for component type ('inductor'/'capacitor')
    
    Returns:
        PartInfo or None if parsing fails
    """
    pn = part_number.upper().strip()
    # Remove .s2p extension if present
    if pn.endswith('.S2P'):
        pn = pn[:-4]
    
    # Determine component type
    is_inductor = any(pn.startswith(p) for p in INDUCTOR_PREFIXES)
    is_capacitor = any(pn.startswith(p) for p in CAPACITOR_PREFIXES)
    
    if known_type:
        component_type = known_type
    elif is_inductor:
        component_type = 'inductor'
    elif is_capacitor:
        component_type = 'capacitor'
    else:
        # Try to guess from naming patterns
        component_type = 'capacitor'  # default
    
    if component_type == 'inductor':
        return _parse_inductor_part(pn)
    else:
        return _parse_capacitor_part(pn)


def _parse_inductor_part(pn: str) -> Optional[PartInfo]:
    """Parse an inductor part number."""
    # Find the series prefix
    series_prefix = ''
    for prefix in INDUCTOR_PREFIXES:
        if pn.startswith(prefix):
            series_prefix = prefix
            break
    
    if not series_prefix:
        return None
    
    # Extract size code: follows the prefix
    # LQP03TN → size = 03
    # LQW15AN → size = 15
    # LQG15HH → size = 15
    # LQW04AN → size = 04
    # LQW2BAN → size = 2B (letter in size for some series)
    # LQW2UAS → size = 2U
    rest = pn[len(series_prefix):]
    
    # Known alphanumeric sizes: 2B, 2U
    # Try alphanumeric size first (digit + letter)
    alphanum_match = re.match(r'(\d[A-Z])(?=[A-Z]{2})', rest)
    if alphanum_match:
        size_code = alphanum_match.group(1)
    else:
        # Standard 2-digit size
        size_match = re.match(r'(\d{2})', rest)
        size_code = size_match.group(1) if size_match else ''
    
    # After size code, there's a type/variant code (2 chars typically)
    # Type codes: TN, HQ, HV, HS, HZ, HH, HN, WH, WZ, AN, AS, CA, CN, etc.
    after_size = rest[len(size_code):]
    type_match = re.match(r'([A-Z]{2})', after_size)
    type_code = type_match.group(1) if type_match else ''
    
    # Full series = prefix + size + type
    series = series_prefix + size_code + type_code
    
    # Remaining is value + tolerance + suffix
    value_part = after_size[len(type_code):]
    
    # Parse value and tolerance
    value_nH, tol_code, value_str, suffix = _parse_inductor_value_and_tolerance(value_part)
    
    if value_nH <= 0 and not value_str:
        return None
    
    tol_pct, tol_abs = compute_tolerance(tol_code, value_nH, 'inductor')
    
    return PartInfo(
        part_number=pn,
        series=series,
        component_type='inductor',
        nominal_value=value_nH,
        nominal_unit='nH',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        voltage_code='',
        value_str=value_str,
        dielectric='',
    )


def _parse_capacitor_part(pn: str) -> Optional[PartInfo]:
    """Parse a capacitor part number using robust EIA code detection."""
    # Find the series prefix
    series_prefix = ''
    for prefix in CAPACITOR_PREFIXES:
        if pn.startswith(prefix):
            series_prefix = prefix
            break
    
    if not series_prefix:
        return None
    
    rest = pn[len(series_prefix):]
    
    # Size code: 3-4 digits (e.g., 011, 1555, 0402, 1608, 0225, 0335, 55)
    size_match = re.match(r'(\d{2,4})', rest)
    if size_match:
        size_code = size_match.group(1)
    else:
        size_code = ''
    
    after_size = rest[len(size_code):]
    
    # Type char: C, R (single char, at the start)
    type_char = ''
    if after_size and after_size[0].isalpha():
        type_char = after_size[0]
        after_type = after_size[1:]
    else:
        after_type = after_size
    
    # Series = prefix + size + type
    series = series_prefix + size_code + type_char
    
    # Robust value extraction: find the 3-digit EIA code or R-notation
    # that is followed by a known tolerance letter.
    # This works regardless of temp/voltage characteristic length.
    
    value_pf = 0.0
    value_code = ''
    tol_code = '?'
    remaining = after_type
    
    # Try R-notation first (e.g., 1R0, R50, 2R2, R70)
    r_matches = list(re.finditer(r'(\d*R\d+)([A-Z])', after_type))
    for m in r_matches:
        candidate_tol = m.group(2)
        if candidate_tol in TOLERANCE_MAP:
            value_code = m.group(1)
            tol_code = candidate_tol
            value_pf, _ = _parse_capacitor_eia_code(value_code)
            remaining = after_type[m.end():]
            break
    
    # Try 3-digit EIA code followed by known tolerance letter
    if value_pf == 0.0:
        eia_matches = list(re.finditer(r'(\d{3})([A-Z])', after_type))
        for m in eia_matches:
            candidate_tol = m.group(2)
            if candidate_tol in TOLERANCE_MAP:
                code = m.group(1)
                digits = int(code[:2])
                multiplier = int(code[2])
                # Validate reasonable EIA code
                if 0 <= multiplier <= 9 and 1 <= digits <= 99:
                    candidate_val = digits * (10 ** multiplier)
                    if 0.1 <= candidate_val <= 1e12:
                        value_pf = float(candidate_val)
                        value_code = code
                        tol_code = candidate_tol
                        remaining = after_type[m.end():]
                        break
    
    # Try 2-digit code + tolerance (for very small caps like 10pF)
    if value_pf == 0.0:
        m2 = re.search(r'(\d{2})([A-Z])', after_type)
        if m2 and m2.group(2) in TOLERANCE_MAP:
            code = m2.group(1)
            value_pf, _ = _parse_capacitor_eia_code(code)
            value_code = code
            tol_code = m2.group(2)
            remaining = after_type[m2.end():]
    
    # Extract voltage/temp characteristic from before the value code
    voltage_code = ''
    if value_code:
        val_pos = after_type.find(value_code)
        if val_pos > 0:
            voltage_code = after_type[:val_pos]
    
    # Detect dielectric
    dielectric = _guess_dielectric(pn, voltage_code)
    
    tol_pct, tol_abs = compute_tolerance(tol_code, value_pf, 'capacitor')
    
    return PartInfo(
        part_number=pn,
        series=series,
        component_type='capacitor',
        nominal_value=value_pf,
        nominal_unit='pF',
        tolerance_code=tol_code,
        tolerance_pct=tol_pct,
        tolerance_abs=tol_abs,
        size_code=size_code,
        voltage_code=voltage_code,
        value_str=value_code,
        dielectric=dielectric,
    )


def _extract_capacitor_value_from_part_full(remaining: str) -> Tuple[float, str, str, str]:
    """
    Extract capacitor value from the remaining part number after voltage code.
    
    Examples:
    - 101JA01 → (100, '101', 'J', 'A01')
    - 104ME01 → (100000, '104', 'M', 'E01')
    - 1R0BB01 → (1.0, '1R0', 'B', 'B01')
    - R70WB01 → (0.70, 'R70', 'W', 'B01')
    - 100GB01 → (10, '100', 'G', 'B01')
    """
    s = remaining.upper().strip()
    if not s:
        return (0.0, '', '?', '')
    
    # Try R-notation first
    r_match = re.match(r'(\d*R\d+)([A-Z])(.*)', s)
    if r_match:
        val_code = r_match.group(1)
        tol_code = r_match.group(2)
        suffix = r_match.group(3)
        value, _ = _parse_capacitor_eia_code(val_code)
        return (value, val_code, tol_code, suffix)
    
    # Try 3-digit EIA + tolerance
    eia_match = re.match(r'(\d{3})([A-Z])(.*)', s)
    if eia_match:
        code = eia_match.group(1)
        tol_char = eia_match.group(2)
        suffix = eia_match.group(3)
        if tol_char in TOLERANCE_MAP:
            value, _ = _parse_capacitor_eia_code(code)
            return (value, code, tol_char, suffix)
        else:
            # The 3-digit might not be the value code
            # Try next 3-digit
            pass
    
    # Try 2-digit code + tolerance (for very small caps like 10pF)
    m2 = re.match(r'(\d{2})([A-Z])(.*)', s)
    if m2 and m2.group(2) in TOLERANCE_MAP:
        code = m2.group(1)
        value, _ = _parse_capacitor_eia_code(code)
        return (value, code, m2.group(2), m2.group(3))
    
    return (0.0, '', '?', s)


def _guess_dielectric(part_number: str, voltage_code: str) -> str:
    """Guess dielectric type from part number characteristics."""
    pn = part_number.upper()
    
    # Temperature compensating dielectrics (C0G/NP0 type)
    # Voltage codes: 1C, 1H, 2C, 2D typically C0G
    c0g_codes = {'1C', '2C', '2D', '1B', '2B'}
    
    # High-K dielectrics
    # Voltage codes: 1E, 1J, 1A typically X5R/X7R
    x7r_codes = {'1E', '1J', '1A', '2A', '3A'}
    x5r_codes = {'1H', '0G', '0J'}
    
    # For GRM/GCM series, subdirectory sometimes indicates
    # But we can use the characteristic letter
    
    if voltage_code in c0g_codes:
        return 'C0G'
    elif voltage_code in x7r_codes:
        return 'X7R'
    elif voltage_code in x5r_codes:
        return 'X5R'
    
    # Check by characteristic code (second char of voltage code)
    if len(voltage_code) >= 2:
        char = voltage_code[1]
        if char in ('C', 'B'):
            return 'C0G'
        elif char in ('R', 'E', 'J', 'A'):
            return 'X7R'
        elif char in ('H', 'G'):
            return 'X5R'
    
    return ''


def get_precision_rank(tolerance_code: str) -> int:
    """Get precision rank (lower = higher precision)."""
    return PRECISION_RANK.get(tolerance_code.upper(), 99)


if __name__ == '__main__':
    # Test parser with known part numbers
    test_parts = [
        # Inductors
        ('LQP03TN10NH02', 'inductor', 10.0, 'nH', 'H'),
        ('LQP03TN0N6B02', 'inductor', 0.6, 'nH', 'B'),
        ('LQP03TNR22J02', 'inductor', 0.22, 'nH', 'J'),
        ('LQW15AN10NG00', 'inductor', 10.0, 'nH', 'G'),
        ('LQW04AN0N8C00', 'inductor', 0.8, 'nH', 'C'),
        ('LQW04AN9N7D00', 'inductor', 9.7, 'nH', 'D'),
        ('LQG15HH10NG02', 'inductor', 10.0, 'nH', 'G'),
        ('LQW15ANR10J00', 'inductor', 0.10, 'nH', 'J'),
        # Capacitors
        ('GRM1555C1H101JA01', 'capacitor', 100.0, 'pF', 'J'),
        ('GRM011C80E104ME01', 'capacitor', 100000.0, 'pF', 'M'),
        ('GJM0225C1C100GB01', 'capacitor', 10.0, 'pF', 'G'),
        ('GCM155R71H103KA55', 'capacitor', 10000.0, 'pF', 'K'),
        ('GCQ0335C1H1R0BB01', 'capacitor', 1.0, 'pF', 'B'),
        ('GCQ0335C1H100GB01', 'capacitor', 10.0, 'pF', 'G'),
        ('GJM1555C1HR70WB01', 'capacitor', 0.70, 'pF', 'W'),
    ]
    
    print(f"{'Part Number':<28} {'Type':<10} {'Value':>10} {'Unit':<4} {'Tol':>4} {'Series':<15}")
    print('-' * 80)
    
    for pn, expected_type, expected_val, expected_unit, expected_tol in test_parts:
        info = parse_murata_part_number(pn)
        if info:
            val_ok = abs(info.nominal_value - expected_val) < 0.01
            tol_ok = info.tolerance_code == expected_tol
            status = 'OK' if (val_ok and tol_ok) else 'FAIL'
            print(f"{info.part_number:<28} {info.component_type:<10} "
                  f"{info.nominal_value:>10.3f} {info.nominal_unit:<4} "
                  f"{info.tolerance_code:>4} {info.series:<15} {status}")
            if not val_ok or not tol_ok:
                print(f"  Expected: {expected_val} {expected_unit}, tol={expected_tol}")
        else:
            print(f"{pn:<28} PARSE FAILED")
