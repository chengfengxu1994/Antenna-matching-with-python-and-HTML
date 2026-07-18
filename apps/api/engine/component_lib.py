"""
Component library: load and manage real S2P component data.

Handles:
- Scanning ZIP files in the Murata directory
- Scanning extracted Optenni/third-party S2P component directories
- Extracting component metadata (part number, type, nominal value)
- Caching parsed S2P data for quick access
- Fast frequency lookup and interpolation
"""

import os
import re
import zipfile
import hashlib
import json
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
from .touchstone import parse_touchstone, TouchstoneData, FREQ_MULTIPLIERS


COMPONENT_CATALOG_FINGERPRINT_SCHEMA = "rfmatch.component-catalog.v3"
_SOURCE_DIGEST_CACHE: Dict[tuple, str] = {}
_SOURCE_DIGEST_CACHE_LOCK = threading.Lock()


def _cached_file_digest(path: str) -> str:
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    key = ("file", resolved, stat.st_size, stat.st_mtime_ns)
    with _SOURCE_DIGEST_CACHE_LOCK:
        cached = _SOURCE_DIGEST_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _SOURCE_DIGEST_CACHE_LOCK:
        _SOURCE_DIGEST_CACHE[key] = value
    return value


def _cached_zip_entry_digest(zip_path: str, entry_name: str) -> str:
    resolved = os.path.realpath(zip_path)
    stat = os.stat(resolved)
    with zipfile.ZipFile(resolved, "r") as archive:
        info = archive.getinfo(entry_name)
        key = (
            "zip_entry", resolved, stat.st_size, stat.st_mtime_ns,
            entry_name, info.CRC, info.file_size,
        )
        with _SOURCE_DIGEST_CACHE_LOCK:
            cached = _SOURCE_DIGEST_CACHE.get(key)
        if cached:
            return cached
        value = hashlib.sha256(archive.read(entry_name)).hexdigest()
    with _SOURCE_DIGEST_CACHE_LOCK:
        _SOURCE_DIGEST_CACHE[key] = value
    return value


def component_library_content_fingerprint(library: "ComponentLibrary") -> Dict[str, Any]:
    """Build a path-independent manifest hash from component identity and source bytes."""
    records = []
    unreadable_sources = 0
    source_keys = set()
    for component in getattr(library, "all_components", []) or []:
        source_kind = "file" if component.zip_path == "__DIR__" else "zip_entry"
        source_name = os.path.basename(str(component.s2p_filename).replace("\\", "/"))
        try:
            if source_kind == "file":
                source_digest = _cached_file_digest(component.s2p_filename)
                source_key = (source_kind, source_digest)
            else:
                source_digest = _cached_zip_entry_digest(
                    component.zip_path, component.s2p_filename
                )
                source_key = (source_kind, source_digest, source_name)
            source_keys.add(source_key)
        except (OSError, KeyError, zipfile.BadZipFile):
            source_digest = None
            unreadable_sources += 1
        records.append({
            "component_type": str(component.component_type),
            "part_number": str(component.part_number),
            "nominal_value": format(float(component.nominal_value), ".17g"),
            "nominal_unit": str(component.nominal_unit),
            "manufacturer": str(component.manufacturer or ""),
            "series": str(component.series or ""),
            "size_code": str(component.size_code or ""),
            "tolerance_pct": (
                None if component.tolerance_pct is None
                else format(float(component.tolerance_pct), ".17g")
            ),
            "voltage_code": str(component.voltage_code or ""),
            "dielectric": str(component.dielectric or ""),
            "tempco_ppm_per_c": (
                None if getattr(component, "tempco_ppm_per_c", None) is None
                else format(float(component.tempco_ppm_per_c), ".17g")
            ),
            "systematic_bias_pct": (
                None if getattr(component, "systematic_bias_pct", None) is None
                else format(float(component.systematic_bias_pct), ".17g")
            ),
            "source_kind": source_kind,
            "source_name": source_name,
            "source_sha256": source_digest,
        })
    records.sort(key=lambda item: json.dumps(
        item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))
    canonical = json.dumps(
        {"schema": COMPONENT_CATALOG_FINGERPRINT_SCHEMA, "components": records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": COMPONENT_CATALOG_FINGERPRINT_SCHEMA,
        "algorithm": "sha256",
        "digest": hashlib.sha256(canonical).hexdigest(),
        "component_count": len(records),
        "source_count": len(source_keys),
        "content_verified": unreadable_sources == 0,
        "unreadable_source_count": unreadable_sources,
    }


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
    manufacturer: str = ""
    series: str = ""
    size_code: str = ""
    tolerance_pct: Optional[float] = None
    voltage_code: str = ""
    dielectric: str = ""
    tempco_ppm_per_c: Optional[float] = None
    systematic_bias_pct: Optional[float] = None
    environment_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata_provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def data(self) -> TouchstoneData:
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> TouchstoneData:
        if self.zip_path == '__DIR__':
            with open(self.s2p_filename, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        else:
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


def parse_generic_s2p_part(part_number: str, known_type: Optional[str] = None) -> Tuple[str, float, str]:
    """
    Parse common S2P library names beyond Murata.

    Examples:
      04HP2N0 / 0402HP-2N0XJL -> 2.0 nH
      L0201SEr33              -> 0.33 nH
      GQM1885C2A1R0BB01       -> 1.0 pF
    """
    pn = part_number.upper().strip()
    comp_type = known_type
    if comp_type is None:
        if pn.startswith(('L', '04HP', '03HP', '06HP', '08HP')) or re.search(r'\dN\d|R\d+$', pn):
            comp_type = 'inductor'
        else:
            comp_type = 'capacitor'

    if comp_type == 'inductor':
        value, unit = _parse_inductor_value(pn)
        if value <= 0:
            value, unit = _parse_compact_inductor_value(pn)
        return ('inductor', value, unit)

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

    # Compact decimal notation must be handled before the generic "digits + N"
    # pattern, otherwise 6N2 would be read as 6 nH instead of 6.2 nH.
    match = re.search(r'(\d+)N(\d+)', pn)
    if match:
        return (float(f"{match.group(1)}.{match.group(2)}"), 'nH')

    match = re.search(r'N(\d+)', pn)
    if match:
        return (float(f"0.{match.group(1)}"), 'nH')

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
    return (0.0, 'nH')


def _parse_compact_inductor_value(pn: str) -> Tuple[float, str]:
    """Parse compact third-party inductor value codes such as 2N0, N47, r33."""
    match = re.search(r'(\d+)N(\d+)', pn)
    if match:
        return (float(f"{match.group(1)}.{match.group(2)}"), 'nH')

    match = re.search(r'N(\d+)', pn)
    if match:
        return (float(f"0.{match.group(1)}"), 'nH')

    match = re.search(r'R(\d+)$', pn, flags=re.IGNORECASE)
    if match:
        return (float(f"0.{match.group(1)}"), 'nH')

    match = re.search(r'(\d+)R(\d+)', pn, flags=re.IGNORECASE)
    if match:
        return (float(f"{match.group(1)}.{match.group(2)}"), 'nH')

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


def scan_s2p_directory(root_dir: str) -> ComponentLibrary:
    """
    Scan an extracted S2P library directory.

    This is used for Optenni's component library and other vendor folders. It
    keeps real measured/modelled S-parameters instead of substituting ideal LC
    parts.
    """
    library = ComponentLibrary()

    if not os.path.isdir(root_dir):
        return library

    for root, dirs, files in os.walk(root_dir):
        lower_root = root.lower()
        if 'inductor' in lower_root:
            known_type = 'inductor'
        elif 'capacitor' in lower_root:
            known_type = 'capacitor'
        else:
            known_type = None

        series_name = os.path.basename(root.rstrip("/\\")) or "Unclassified"
        manufacturer = _manufacturer_from_series_name(series_name)
        size_code = _package_from_text(series_name)

        for f in files:
            if not f.lower().endswith('.s2p'):
                continue

            full_path = os.path.join(root, f)
            part_number = os.path.splitext(os.path.basename(f))[0]
            comp_type, nominal_value, nominal_unit = parse_generic_s2p_part(part_number, known_type)
            if nominal_value <= 0:
                continue

            library.add_component(ComponentInfo(
                part_number=part_number,
                s2p_filename=full_path,
                zip_path='__DIR__',
                component_type=comp_type,
                nominal_value=nominal_value,
                nominal_unit=nominal_unit,
                manufacturer=manufacturer,
                series=series_name,
                size_code=size_code,
                metadata_provenance={
                    "manufacturer": "catalog_path" if manufacturer else "unknown",
                    "series": "catalog_path" if series_name != "Unclassified" else "unknown",
                    "package_code": "catalog_path" if size_code else "unknown",
                    "tolerance_pct": "unknown",
                    "voltage_code": "unknown",
                    "dielectric": "unknown",
                },
            ))

    return library


def merge_component_libraries(*libraries: ComponentLibrary) -> ComponentLibrary:
    """Merge component libraries while preserving one record per part/source."""
    merged = ComponentLibrary()
    seen = set()
    for lib in libraries:
        if lib is None:
            continue
        for comp in lib.all_components:
            key = (comp.part_number, comp.s2p_filename, comp.zip_path)
            if key in seen:
                continue
            seen.add(key)
            merged.add_component(comp)
    return merged


def filter_component_library(
    library,
    inductor_tokens: Optional[List[str]] = None,
    capacitor_tokens: Optional[List[str]] = None,
) -> ComponentLibrary:
    """Keep only components whose part/path/series text matches requested tokens."""
    filtered = ComponentLibrary()
    inductor_tokens = [t.upper() for t in (inductor_tokens or [])]
    capacitor_tokens = [t.upper() for t in (capacitor_tokens or [])]

    def matches(comp, tokens):
        if not tokens:
            return True
        text = " ".join(str(getattr(comp, attr, "")) for attr in (
            "part_number", "s2p_filename", "zip_path", "series"
        )).upper()
        return any(token in text for token in tokens)

    for comp in getattr(library, 'inductors', []) or []:
        if matches(comp, inductor_tokens):
            filtered.add_component(comp)

    for comp in getattr(library, 'capacitors', []) or []:
        if matches(comp, capacitor_tokens):
            filtered.add_component(comp)

    return filtered


def component_series_name(component) -> str:
    """Return a stable human-facing family name across file and DB adapters."""
    explicit = str(getattr(component, "series", "") or "").strip()
    if explicit:
        return explicit
    filename = str(getattr(component, "s2p_filename", "") or "")
    if filename:
        parent = os.path.basename(os.path.dirname(filename.rstrip("/\\")))
        if parent:
            return parent
    archive = str(getattr(component, "zip_path", "") or "")
    if archive and archive != "__DIR__":
        return os.path.splitext(os.path.basename(archive))[0]
    return "Unclassified"


def _manufacturer_from_series_name(series_name: str) -> str:
    """Extract only catalog-path evidence; do not pretend it is vendor DB data."""
    text = str(series_name or "").strip()
    lower = text.lower()
    known = (
        "Taiyo Yuden", "Coilcraft", "Johanson", "Murata", "TDK", "AVX",
        "Vishay", "Samsung", "Wurth", "Kemet", "Kyocera",
    )
    for manufacturer in known:
        if lower.startswith(manufacturer.lower()):
            return manufacturer
    return text.split()[0] if text and text != "Unclassified" else ""


def _package_from_text(text: str) -> str:
    match = re.search(
        r"(?<!\d)(01005|0201|0402|0603|0805|1005|1608|2012|2520)(?!\d)",
        str(text or "").upper(),
    )
    return match.group(1) if match else ""


def component_metadata(component) -> Dict[str, Any]:
    """Return normalized filter metadata together with field-level provenance."""
    provenance = dict(getattr(component, "metadata_provenance", {}) or {})
    manufacturer = str(getattr(component, "manufacturer", "") or "").strip()
    series_name = component_series_name(component)
    if not manufacturer and "manufacturer" not in provenance:
        manufacturer = _manufacturer_from_series_name(series_name)
        if manufacturer:
            provenance.setdefault("manufacturer", "catalog_path")
    package_code = str(
        getattr(component, "size_code", "")
        or getattr(component, "package_code", "")
        or ""
    ).strip()
    if not package_code and "package_code" not in provenance:
        package_code = _package_from_text(series_name)
        if package_code:
            provenance.setdefault("package_code", "catalog_path")
    tolerance = getattr(component, "tolerance_pct", None)
    try:
        tolerance = float(tolerance) if tolerance is not None else None
    except (TypeError, ValueError):
        tolerance = None
    voltage_code = str(getattr(component, "voltage_code", "") or "").strip().upper()
    dielectric = str(getattr(component, "dielectric", "") or "").strip().upper()
    tempco = getattr(component, "tempco_ppm_per_c", None)
    bias = getattr(component, "systematic_bias_pct", None)
    try:
        tempco = float(tempco) if tempco is not None else None
    except (TypeError, ValueError):
        tempco = None
    try:
        bias = float(bias) if bias is not None else None
    except (TypeError, ValueError):
        bias = None
    for field_name, value in (
        ("manufacturer", manufacturer),
        ("package_code", package_code),
        ("tolerance_pct", tolerance),
        ("voltage_code", voltage_code),
        ("dielectric", dielectric),
        ("tempco_ppm_per_c", tempco),
        ("systematic_bias_pct", bias),
    ):
        provenance.setdefault(field_name, "unknown" if value in (None, "") else "database")
    return {
        "manufacturer": manufacturer,
        "package_code": package_code,
        "tolerance_pct": tolerance,
        "voltage_code": voltage_code,
        "dielectric": dielectric,
        "tempco_ppm_per_c": tempco,
        "systematic_bias_pct": bias,
        "environment_metadata": dict(getattr(component, "environment_metadata", {}) or {}),
        "provenance": provenance,
    }


def filter_component_library_by_parameters(library, filters=None):
    """Apply procurement filters and return the catalog plus auditable counts.

    Unknown metadata is retained by default for backward compatibility. In
    strict mode (``exclude``), a component must have every requested field.
    """
    filters = dict(filters or {})
    manufacturers = {
        str(value).strip().casefold() for value in filters.get("manufacturers", [])
        if str(value).strip()
    }
    package_codes = {
        str(value).strip().upper() for value in filters.get("package_codes", [])
        if str(value).strip()
    }
    voltage_codes = {
        str(value).strip().upper() for value in filters.get("voltage_codes", [])
        if str(value).strip()
    }
    dielectrics = {
        str(value).strip().upper() for value in filters.get("dielectrics", [])
        if str(value).strip()
    }
    maximum_tolerance = filters.get("maximum_tolerance_pct")
    unknown_policy = filters.get("unknown_metadata_policy", "include")
    requested_fields = []
    if manufacturers:
        requested_fields.append("manufacturer")
    if package_codes:
        requested_fields.append("package_code")
    if maximum_tolerance is not None:
        requested_fields.append("tolerance_pct")
    if voltage_codes:
        requested_fields.append("voltage_code")
    if dielectrics:
        requested_fields.append("dielectric")

    result = ComponentLibrary()
    stats = {
        "input_components": 0,
        "matched_components": 0,
        "excluded_by_value": 0,
        "excluded_unknown": 0,
        "included_with_unknown": 0,
        "metadata_sources": {},
    }

    def matches(component) -> bool:
        stats["input_components"] += 1
        metadata = component_metadata(component)
        unknown = False
        mismatch = False
        for field_name in requested_fields:
            if (
                field_name in {"voltage_code", "dielectric"}
                and getattr(component, "component_type", "") != "capacitor"
            ):
                continue
            source = metadata["provenance"].get(field_name, "unknown")
            stats["metadata_sources"][source] = stats["metadata_sources"].get(source, 0) + 1
            value = metadata[field_name]
            if value in (None, "") or source == "unknown":
                unknown = True
                continue
            if field_name == "manufacturer" and value.casefold() not in manufacturers:
                mismatch = True
            elif field_name == "package_code" and value.upper() not in package_codes:
                mismatch = True
            elif field_name == "tolerance_pct" and value > float(maximum_tolerance):
                mismatch = True
            elif field_name == "voltage_code" and value.upper() not in voltage_codes:
                mismatch = True
            elif field_name == "dielectric" and value.upper() not in dielectrics:
                mismatch = True
        if mismatch:
            stats["excluded_by_value"] += 1
            return False
        if unknown and unknown_policy == "exclude":
            stats["excluded_unknown"] += 1
            return False
        if unknown:
            stats["included_with_unknown"] += 1
        stats["matched_components"] += 1
        return True

    for component in getattr(library, "inductors", []) or []:
        if matches(component):
            result.add_component(component)
    for component in getattr(library, "capacitors", []) or []:
        if matches(component):
            result.add_component(component)
    return result, stats


def component_series_id(component) -> str:
    prefix = "L" if getattr(component, "component_type", "") == "inductor" else "C"
    return f"{prefix}::{component_series_name(component)}"


def filter_component_library_by_series(library, selected_series: List[str]) -> ComponentLibrary:
    """Filter by exact typed family IDs while accepting legacy untyped names."""
    selected = {str(item).strip() for item in selected_series if str(item).strip()}
    filtered = ComponentLibrary()
    if not selected:
        return filtered

    def matches(component) -> bool:
        name = component_series_name(component)
        typed = component_series_id(component)
        if typed in selected or name in selected:
            return True
        # Compatibility for earlier UI values derived from part/path tokens.
        searchable = " ".join(str(getattr(component, attr, "")) for attr in (
            "part_number", "s2p_filename", "zip_path", "series"
        )).upper()
        return any("::" not in item and item.upper() in searchable for item in selected)

    for component in getattr(library, "inductors", []) or []:
        if matches(component):
            filtered.add_component(component)
    for component in getattr(library, "capacitors", []) or []:
        if matches(component):
            filtered.add_component(component)
    return filtered


class CompositeComponentLibrary:
    """Read-only view over multiple component library backends."""

    def __init__(self, *libraries):
        self.libraries = [lib for lib in libraries if lib is not None]

    @property
    def inductors(self):
        parts = []
        for lib in self.libraries:
            parts.extend(getattr(lib, 'inductors', []) or [])
        return parts

    @property
    def capacitors(self):
        parts = []
        for lib in self.libraries:
            parts.extend(getattr(lib, 'capacitors', []) or [])
        return parts

    @property
    def all_components(self):
        return self.inductors + self.capacitors

    def get_unique_inductor_values(self) -> List[float]:
        return sorted({c.nominal_value for c in self.inductors if c.nominal_value > 0})

    def get_unique_capacitor_values(self) -> List[float]:
        return sorted({c.nominal_value for c in self.capacitors if c.nominal_value > 0})

    def get_inductors_near(self, target_nh: float, tolerance: float = 0.5) -> List[ComponentInfo]:
        return [
            c for c in self.inductors
            if abs(c.nominal_value - target_nh) / max(target_nh, 1e-9) <= tolerance
        ]

    def get_capacitors_near(self, target_pf: float, tolerance: float = 0.5) -> List[ComponentInfo]:
        return [
            c for c in self.capacitors
            if abs(c.nominal_value - target_pf) / max(target_pf, 1e-9) <= tolerance
        ]

    def find_nearest_inductor(self, target_nh: float):
        parts = self.inductors
        return min(parts, key=lambda c: abs(c.nominal_value - target_nh)) if parts else None

    def find_nearest_capacitor(self, target_pf: float):
        parts = self.capacitors
        return min(parts, key=lambda c: abs(c.nominal_value - target_pf)) if parts else None
