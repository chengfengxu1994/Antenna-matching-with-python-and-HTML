"""
SQLite-backed ComponentLibrary adapter for the RF matching optimizer.

Drop-in replacement for the original ComponentLibrary that uses
the pre-computed SQLite database for dramatically faster queries.

Key optimizations:
1. Only primary (highest-precision) components used by default
2. Pre-computed S-parameters at standard RF frequencies (no ZIP extraction)
3. Pre-computed impedance, effective L/C, Q-factor at each frequency
4. Bulk queries for optimizer sweep (all components at a frequency in one query)
5. Falls back to on-demand ZIP loading for non-grid frequencies
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .component_lib import ComponentInfo, ComponentLibrary
from .murata_db import MurataDatabase, ComponentRecord
from .touchstone import TouchstoneData


@dataclass
class DBComponentInfo:
    """A component backed by the SQLite database."""
    db_id: int
    part_number: str
    series: str
    component_type: str
    nominal_value: float
    nominal_unit: str
    tolerance_code: str
    tolerance_pct: float
    size_code: str
    
    @property
    def label(self) -> str:
        return "%s (%s %s)" % (self.part_number, self.nominal_value, self.nominal_unit)


class MurataDBLibrary:
    """
    SQLite-backed component library.
    
    Provides the same interface as ComponentLibrary but backed by
    pre-computed database for fast queries. The optimizer can use
    this instead of loading all S2P data from ZIP files.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = MurataDatabase(db_path)
        self.db.connect()
        
        # Cache for frequently accessed data
        self._stats = None
        self._inductor_values = None
        self._capacitor_values = None
        self._inductors_cache = None
        self._capacitors_cache = None
    
    def close(self):
        """Close database connection."""
        self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @property
    def inductors(self):
        """Return primary inductors as list compatible with optimizer."""
        if self._inductors_cache is None:
            records = self.db.get_primary_inductors()
            # Filter out parsing errors: valid range 0.05-500 nH
            self._inductors_cache = [DBComponentAdapter(r, self) for r in records
                                     if 0.05 <= r.nominal_value <= 500]
        return self._inductors_cache
    
    @property
    def capacitors(self):
        """Return primary capacitors as list compatible with optimizer."""
        if self._capacitors_cache is None:
            records = self.db.get_primary_capacitors()
            # Filter out parsing errors: valid range 0.05-100000 pF
            self._capacitors_cache = [DBComponentAdapter(r, self) for r in records
                                      if 0.05 <= r.nominal_value <= 100000]
        return self._capacitors_cache
    
    @property
    def stats(self) -> Dict:
        if self._stats is None:
            self._stats = self.db.get_statistics()
        return self._stats
    
    # --- Component queries ---
    
    def get_inductors_near(self, target_nh: float, 
                           tolerance: float = 0.3) -> List[DBComponentInfo]:
        """Get primary inductors near a target value (nH)."""
        records = self.db.get_inductors_near(target_nh, tolerance, primary_only=True)
        return [self._record_to_info(r) for r in records]
    
    def get_capacitors_near(self, target_pf: float,
                            tolerance: float = 0.3) -> List[DBComponentInfo]:
        """Get primary capacitors near a target value (pF)."""
        records = self.db.get_capacitors_near(target_pf, tolerance, primary_only=True)
        return [self._record_to_info(r) for r in records]
    
    def find_nearest_inductor(self, target_nh: float) -> Optional[DBComponentInfo]:
        """Find the nearest inductor to a target value."""
        records = self.db.get_inductors_near(target_nh, tolerance=2.0, primary_only=True)
        if not records:
            return None
        nearest = min(records, key=lambda r: abs(r.nominal_value - target_nh))
        return self._record_to_info(nearest)
    
    def find_nearest_capacitor(self, target_pf: float) -> Optional[DBComponentInfo]:
        """Find the nearest capacitor to a target value."""
        records = self.db.get_capacitors_near(target_pf, tolerance=2.0, primary_only=True)
        if not records:
            return None
        nearest = min(records, key=lambda r: abs(r.nominal_value - target_pf))
        return self._record_to_info(nearest)
    
    def get_unique_inductor_values(self) -> List[float]:
        """Get sorted list of unique inductor values."""
        if self._inductor_values is None:
            self._inductor_values = self.db.get_unique_values('inductor')
        return self._inductor_values
    
    def get_unique_capacitor_values(self) -> List[float]:
        """Get sorted list of unique capacitor values."""
        if self._capacitor_values is None:
            self._capacitor_values = self.db.get_unique_values('capacitor')
        return self._capacitor_values
    
    # --- S-parameter access ---
    
    def get_s_matrix_at_freq(self, db_id: int, freq_mhz: float) -> Optional[np.ndarray]:
        """Get 2x2 S-matrix for a component at a frequency (from pre-computed data)."""
        return self.db.get_sparam_matrix(db_id, freq_mhz)
    
    def get_component_sparams(self, db_id: int, freq_mhz: float) -> Optional[Dict]:
        """Get full S-parameter data for a component at a frequency."""
        return self.db.get_component_sparams(db_id, freq_mhz)
    
    def get_component_derived(self, db_id: int, freq_mhz: float) -> Optional[Dict]:
        """Get derived parameters (Z, eff L/C, Q, SRF) at a frequency."""
        return self.db.get_component_derived(db_id, freq_mhz)
    
    def get_effective_value_at_freq(self, db_id: int, freq_mhz: float) -> Tuple[float, str]:
        """Get effective L/C value at a frequency."""
        derived = self.db.get_component_derived(db_id, freq_mhz)
        if derived:
            return (derived['eff_value'], derived['eff_unit'])
        return (0.0, '')
    
    # --- Bulk queries for optimizer ---
    
    def get_all_inductors_at_freq(self, freq_mhz: float) -> List[Dict]:
        """
        Get all primary inductors with derived data at a frequency.
        Returns list of dicts ready for the optimizer.
        """
        return self.db.get_all_primaries_with_derived_at_freq(freq_mhz, 'inductor')
    
    def get_all_capacitors_at_freq(self, freq_mhz: float) -> List[Dict]:
        """
        Get all primary capacitors with derived data at a frequency.
        Returns list of dicts ready for the optimizer.
        """
        return self.db.get_all_primaries_with_derived_at_freq(freq_mhz, 'capacitor')
    
    def get_all_components_at_freq(self, freq_mhz: float) -> Dict[str, List[Dict]]:
        """
        Get all primary components (both types) with derived data at a frequency.
        Returns {'inductors': [...], 'capacitors': [...]}.
        """
        return {
            'inductors': self.get_all_inductors_at_freq(freq_mhz),
            'capacitors': self.get_all_capacitors_at_freq(freq_mhz),
        }
    
    # --- Series info ---
    
    def get_series_summary(self) -> List[Dict]:
        """Get summary of all component series."""
        return self.db.export_series_summary()
    
    def get_series_for_value(self, comp_type: str, nominal_value: float,
                              tolerance_pct: float = 5.0) -> List[Dict]:
        """
        Get all series that have components near a nominal value with
        a tolerance better than the specified percentage.
        """
        c = self.db.cursor
        
        if comp_type == 'inductor':
            c.execute("""
                SELECT DISTINCT s.name, s.component_type, s.size_code, 
                       s.min_value, s.max_value, s.component_count
                FROM series s
                JOIN components c ON c.series_id = s.id
                WHERE c.component_type = 'inductor'
                  AND c.nominal_value BETWEEN ? AND ?
                  AND c.tolerance_pct <= ?
                ORDER BY s.name
            """, (nominal_value * 0.8, nominal_value * 1.2, tolerance_pct))
        else:
            c.execute("""
                SELECT DISTINCT s.name, s.component_type, s.size_code,
                       s.min_value, s.max_value, s.component_count
                FROM series s
                JOIN components c ON c.series_id = s.id
                WHERE c.component_type = 'capacitor'
                  AND c.nominal_value BETWEEN ? AND ?
                  AND c.tolerance_pct <= ?
                ORDER BY s.name
            """, (nominal_value * 0.8, nominal_value * 1.2, tolerance_pct))
        
        return [
            {
                'name': row[0], 'type': row[1], 'size': row[2],
                'min_value': row[3], 'max_value': row[4], 'count': row[5]
            }
            for row in c.fetchall()
        ]
    
    # --- Reporting ---
    
    def summary(self) -> str:
        """Get a human-readable summary of the database."""
        s = self.stats
        lines = [
            "Murata Component Database Summary",
            "=" * 40,
            "Total components: %d" % s['total_components'],
            "Primary (high-precision): %d" % s['primary_components'],
            "Total series: %d" % s['total_series'],
            "Frequency grid: %d points" % s['freq_grid_points'],
            "Pre-computed S-param records: %d" % s['sparam_records'],
            "Pre-computed derived records: %d" % s['derived_records'],
            "",
            "Inductors:",
            "  Primary: %d" % s.get('primary_inductors', 0),
            "  Unique values: %d" % s.get('inductor_unique_values', 0),
            "  Range: %.2f - %.1f nH" % (
                s.get('inductor_min_value', 0), s.get('inductor_max_value', 0)),
            "",
            "Capacitors:",
            "  Primary: %d" % s.get('primary_capacitors', 0),
            "  Unique values: %d" % s.get('capacitor_unique_values', 0),
            "  Range: %.2f - %.0f pF" % (
                s.get('capacitor_min_value', 0), s.get('capacitor_max_value', 0)),
        ]
        return "\n".join(lines)
    
    # --- Internal ---
    
    def _record_to_info(self, record: ComponentRecord) -> DBComponentInfo:
        """Convert a ComponentRecord to DBComponentInfo."""
        return DBComponentInfo(
            db_id=record.id,
            part_number=record.part_number,
            series=record.series,
            component_type=record.component_type,
            nominal_value=record.nominal_value,
            nominal_unit=record.nominal_unit,
            tolerance_code=record.tolerance_code,
            tolerance_pct=record.tolerance_pct,
            size_code=record.size_code,
        )


# --- Adapter for optimizer compatibility ---

class DBComponentAdapter:
    """
    Wraps a ComponentRecord to provide the same interface as ComponentInfo.
    The optimizer uses comp.get_s_matrix_at_freq(freq_hz) and comp.nominal_value etc.
    """
    def __init__(self, record, db_library):
        self._record = record
        self._db = db_library
        # Mirror attributes the optimizer expects
        self.part_number = record.part_number
        self.component_type = record.component_type
        self.nominal_value = record.nominal_value
        self.nominal_unit = record.nominal_unit
        self.s2p_filename = record.s2p_filename
        self.zip_path = record.zip_path

    def get_s_matrix_at_freq(self, freq_hz):
        """Get 2x2 S-matrix at arbitrary frequency (interpolated)."""
        freq_mhz = freq_hz / 1e6
        return self._db.get_s_matrix_at_freq(self._record.id, freq_mhz)

    @property
    def label(self):
        return "%s (%s %s)" % (self.part_number, self.nominal_value, self.nominal_unit)


# --- Factory function ---

def load_murata_db(db_path: str = None) -> MurataDBLibrary:
    """
    Load the Murata database. 
    
    If db_path is None, looks for the database in the default location
    relative to the Murata data directory.
    """
    if db_path is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Murata', 'murata_components.db'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'murata_components.db'),
            r'E:\RF matching\Murata\murata_components.db',
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                db_path = candidate
                break
    
    if db_path is None or not os.path.exists(db_path):
        raise FileNotFoundError(
            "Murata database not found. Run 'python -m engine.murata_db' to build it."
        )
    
    return MurataDBLibrary(db_path)
