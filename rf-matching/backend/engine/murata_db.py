"""
Murata Component SQLite Database - Optimized for RF matching engine.

Features:
1. Series-grouped component storage with accurate metadata
2. Precision-based deduplication (same nominal value -> use highest-precision)
3. Pre-computed S-parameters at standard RF frequencies
4. Pre-computed impedance (Z), effective L/C at each frequency
5. Fast query interface for the optimizer

Database schema:
  - series: series metadata
  - components: per-component metadata + tolerance info
  - freq_grid: the frequency points used for pre-computation
  - sparam_at_freq: pre-computed S-matrix entries per component per frequency
  - derived_at_freq: pre-computed Z, effective L/C, |S11|, |S21| per component
  - value_groups: nominal value -> best-precision component mapping
"""

import os
import re
import sys
import json
import time
import sqlite3
import zipfile
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .murata_parser import (
    parse_murata_part_number, PartInfo,
    get_precision_rank, TOLERANCE_MAP, PRECISION_RANK
)
from .optenni_parser import (
    scan_optenni_library, OptenniComponentData
)
from .touchstone import parse_touchstone, TouchstoneData, FREQ_MULTIPLIERS


# Standard RF frequency grid for pre-computation
DEFAULT_FREQ_GRID_MHZ = [
    # VHF
    30, 50, 64, 70, 88, 100, 108, 120, 150, 174, 200, 225,
    # UHF
    300, 400, 433, 450, 470, 500, 600, 700, 800, 850, 868, 900, 915, 960,
    # Low-band cellular / ISM
    1000, 1200, 1400, 1500, 1575, 1600, 1700, 1800, 1900,
    # Mid-band
    2000, 2100, 2200, 2300, 2400, 2450, 2500, 2600,
    # High-band / 5G sub-6
    2700, 2800, 3000, 3300, 3500, 3700, 3800, 4000, 4200, 4500, 4800, 5000,
    # WiFi 6E / mmWave start
    5200, 5500, 5800, 6000, 6500, 7000, 7500, 8000,
    # Higher harmonics
    9000, 10000, 12000, 15000, 20000,
]


@dataclass
class ComponentRecord:
    """A component record from the database."""
    id: int
    part_number: str
    series: str
    component_type: str
    nominal_value: float
    nominal_unit: str
    tolerance_code: str
    tolerance_pct: float
    tolerance_abs: float
    size_code: str
    voltage_code: str
    dielectric: str
    value_str: str
    s2p_filename: str
    zip_path: str
    is_primary: bool  # True = best precision for this nominal value


class MurataDatabase:
    """SQLite-backed Murata component library with pre-computed data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Open database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # --- Schema ---
    
    def create_schema(self):
        """Create all database tables."""
        c = self.cursor
        
        c.executescript("""
            CREATE TABLE IF NOT EXISTS series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                manufacturer TEXT DEFAULT '',
                component_type TEXT NOT NULL,
                size_code TEXT,
                component_count INTEGER DEFAULT 0,
                min_value REAL,
                max_value REAL
            );
            
            CREATE TABLE IF NOT EXISTS manufacturer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                country TEXT DEFAULT '',
                component_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                part_number TEXT UNIQUE NOT NULL,
                series_id INTEGER NOT NULL,
                component_type TEXT NOT NULL,
                nominal_value REAL NOT NULL,
                nominal_unit TEXT NOT NULL,
                tolerance_code TEXT,
                tolerance_pct REAL,
                tolerance_abs REAL,
                size_code TEXT,
                voltage_code TEXT,
                dielectric TEXT,
                value_str TEXT,
                s2p_filename TEXT NOT NULL,
                zip_path TEXT NOT NULL,
                precision_rank INTEGER DEFAULT 99,
                is_primary INTEGER DEFAULT 0,
                FOREIGN KEY (series_id) REFERENCES series(id)
            );
            
            CREATE TABLE IF NOT EXISTS freq_grid (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                freq_mhz REAL UNIQUE NOT NULL,
                freq_hz REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS sparam_at_freq (
                component_id INTEGER NOT NULL,
                freq_id INTEGER NOT NULL,
                s11_re REAL, s11_im REAL,
                s21_re REAL, s21_im REAL,
                s12_re REAL, s12_im REAL,
                s22_re REAL, s22_im REAL,
                PRIMARY KEY (component_id, freq_id),
                FOREIGN KEY (component_id) REFERENCES components(id),
                FOREIGN KEY (freq_id) REFERENCES freq_grid(id)
            );
            
            CREATE TABLE IF NOT EXISTS derived_at_freq (
                component_id INTEGER NOT NULL,
                freq_id INTEGER NOT NULL,
                z_in_re REAL, z_in_im REAL,
                z_trans_re REAL, z_trans_im REAL,
                eff_value REAL,
                eff_unit TEXT,
                s11_mag REAL,
                s11_db REAL,
                s21_mag REAL,
                s21_db REAL,
                q_factor REAL,
                self_resonant_mhz REAL,
                PRIMARY KEY (component_id, freq_id),
                FOREIGN KEY (component_id) REFERENCES components(id),
                FOREIGN KEY (freq_id) REFERENCES freq_grid(id)
            );
            
            CREATE TABLE IF NOT EXISTS value_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_type TEXT NOT NULL,
                nominal_value REAL NOT NULL,
                nominal_unit TEXT NOT NULL,
                primary_component_id INTEGER NOT NULL,
                best_tolerance_code TEXT,
                num_alternatives INTEGER DEFAULT 1,
                FOREIGN KEY (primary_component_id) REFERENCES components(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_comp_type_value
                ON components(component_type, nominal_value);
            CREATE INDEX IF NOT EXISTS idx_comp_series
                ON components(series_id);
            CREATE INDEX IF NOT EXISTS idx_comp_primary
                ON components(is_primary, component_type);
            CREATE INDEX IF NOT EXISTS idx_sparam_comp
                ON sparam_at_freq(component_id);
            CREATE INDEX IF NOT EXISTS idx_sparam_freq
                ON sparam_at_freq(freq_id);
            CREATE INDEX IF NOT EXISTS idx_derived_comp
                ON derived_at_freq(component_id);
            CREATE INDEX IF NOT EXISTS idx_derived_freq
                ON derived_at_freq(freq_id);
            CREATE INDEX IF NOT EXISTS idx_vg_type_value
                ON value_groups(component_type, nominal_value);
        """)
        self.conn.commit()
        
        # Migrate existing DB
        self._add_manufacturer_column()
    
    # --- Population ---
    
    def _add_manufacturer_column(self):
        """Add manufacturer column to series table if missing (migration)."""
        try:
            self.cursor.execute("ALTER TABLE series ADD COLUMN manufacturer TEXT DEFAULT ''")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    def populate_from_murata_dir(self, murata_dir, freq_grid_mhz=None, progress_callback=None):
        """Scan Murata directory and populate database."""
        if freq_grid_mhz is None:
            freq_grid_mhz = DEFAULT_FREQ_GRID_MHZ
        
        freq_grid_hz = [f * 1e6 for f in freq_grid_mhz]
        
        # Insert frequency grid
        self._insert_freq_grid(freq_grid_mhz, freq_grid_hz)
        
        # Scan all ZIPs
        all_parts = self._scan_zips(murata_dir)
        
        if progress_callback:
            progress_callback(0, len(all_parts), "Found %d S2P files" % len(all_parts))
        
        # Insert series and components
        series_map = {}
        c = self.cursor
        total = len(all_parts)
        
        for idx, (part_info, s2p_name, zip_path, category) in enumerate(all_parts):
            if part_info is None:
                continue
            
            sname = part_info.series
            if sname not in series_map:
                c.execute(
                    "INSERT OR IGNORE INTO series (name, manufacturer, component_type, size_code) VALUES (?, ?, ?, ?)",
                    (sname, 'Murata', part_info.component_type, part_info.size_code)
                )
                c.execute("SELECT id FROM series WHERE name = ?", (sname,))
                row = c.fetchone()
                series_map[sname] = row[0] if row else None
            
            series_id = series_map.get(sname)
            if series_id is None:
                continue
            
            precision_rank = get_precision_rank(part_info.tolerance_code)
            
            try:
                c.execute("""
                    INSERT OR IGNORE INTO components 
                    (part_number, series_id, component_type, nominal_value, nominal_unit,
                     tolerance_code, tolerance_pct, tolerance_abs, size_code, voltage_code,
                     dielectric, value_str, s2p_filename, zip_path, precision_rank, is_primary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    part_info.part_number, series_id, part_info.component_type,
                    part_info.nominal_value, part_info.nominal_unit,
                    part_info.tolerance_code, part_info.tolerance_pct, part_info.tolerance_abs,
                    part_info.size_code, part_info.voltage_code, part_info.dielectric,
                    part_info.value_str, s2p_name, zip_path, precision_rank
                ))
            except sqlite3.IntegrityError:
                pass
            
            if progress_callback and (idx + 1) % 5000 == 0:
                progress_callback(idx + 1, total, "Inserted %d/%d components" % (idx + 1, total))
        
        self.conn.commit()
        
        # Update series counts
        c.execute("""
            UPDATE series SET component_count = (
                SELECT COUNT(*) FROM components WHERE components.series_id = series.id
            )
        """)
        self.conn.commit()
        
        # Assign primaries
        self._assign_primaries()
        
        # Pre-compute S-parameters and derived values
        self._precompute_sparams(all_parts, freq_grid_hz, progress_callback)
        
        # Build value groups
        self._build_value_groups()
        
        # Update statistics
        self._update_statistics()
        
        if progress_callback:
            progress_callback(total, total, "Database population complete")
    
    def _scan_zips(self, murata_dir):
        """Scan ZIP files and parse part numbers. Also handles extracted directories."""
        all_parts = []
        
        for root, dirs, files in os.walk(murata_dir):
            # Check if this directory itself is an extracted ZIP (has .s2p files)
            s2p_files_here = [f for f in files if f.lower().endswith('.s2p')]
            if s2p_files_here:
                category = 'inductor' if 'inductor' in root.lower() else 'capacitor'
                for sf in s2p_files_here:
                    s2p_path = os.path.join(root, sf)
                    part_number = os.path.splitext(sf)[0]
                    part_info = parse_murata_part_number(part_number, known_type=category)
                    all_parts.append((part_info, s2p_path, '__DIR__', category))
                continue  # Don't recurse into extracted dirs
            
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
                            
                            part_number = os.path.splitext(os.path.basename(name))[0]
                            part_info = parse_murata_part_number(
                                part_number, known_type=category
                            )
                            all_parts.append((part_info, name, zip_path, category))
                except zipfile.BadZipFile:
                    continue
        
        return all_parts
    
    def _insert_freq_grid(self, freq_mhz, freq_hz):
        """Insert frequency grid points."""
        c = self.cursor
        for fmhz, fhz in zip(sorted(freq_mhz), sorted(freq_hz)):
            c.execute(
                "INSERT OR IGNORE INTO freq_grid (freq_mhz, freq_hz) VALUES (?, ?)",
                (fmhz, fhz)
            )
        self.conn.commit()
    
    def _assign_primaries(self):
        """Mark highest-precision component as primary for each nominal value."""
        c = self.cursor
        
        c.execute("UPDATE components SET is_primary = 0")
        
        c.execute("""
            UPDATE components SET is_primary = 1 
            WHERE id IN (
                SELECT c1.id 
                FROM components c1
                INNER JOIN (
                    SELECT component_type, nominal_value, MIN(precision_rank) as best_rank
                    FROM components
                    WHERE nominal_value > 0
                    GROUP BY component_type, nominal_value
                    HAVING COUNT(*) >= 1
                ) c2 ON c1.component_type = c2.component_type 
                    AND c1.nominal_value = c2.nominal_value 
                    AND c1.precision_rank = c2.best_rank
            )
        """)
        
        self.conn.commit()
        
        c.execute("SELECT COUNT(*) FROM components WHERE is_primary = 1")
        primary_count = c.fetchone()[0]
        c.execute("""
            SELECT COUNT(DISTINCT component_type || ':' || nominal_value) 
            FROM components WHERE nominal_value > 0
        """)
        unique_values = c.fetchone()[0]
        print("  Primary components: %d (covering %d unique values)" % (primary_count, unique_values))
    
    def _precompute_sparams(self, all_parts, freq_grid_hz, progress_callback=None):
        """Pre-compute S-parameters and derived values at grid frequencies."""
        c = self.cursor
        
        # Get freq_grid mapping
        c.execute("SELECT id, freq_hz FROM freq_grid ORDER BY freq_hz")
        freq_rows = c.fetchall()
        freq_map = {row[1]: row[0] for row in freq_rows}
        
        # Build lookup: part_number -> (s2p_name, zip_path)
        part_files = {}
        for part_info, s2p_name, zip_path, category in all_parts:
            if part_info:
                part_files[part_info.part_number] = (s2p_name, zip_path)
        
        # Get primary components
        c.execute("""
            SELECT id, part_number, component_type, nominal_value, nominal_unit
            FROM components WHERE is_primary = 1
        """)
        primary_comps = c.fetchall()
        
        total = len(primary_comps)
        computed = 0
        errors = 0
        
        # Cache: zip_path -> ZipFile handle (not kept open, just path)
        # We'll load files on demand but cache TouchstoneData in memory for current batch
        batch_size = 50
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = primary_comps[batch_start:batch_end]
            
            # Load batch of TouchstoneData
            ts_cache = {}  # part_number -> TouchstoneData
            
            for comp_row in batch:
                comp_id, part_number, comp_type, nominal_value, nominal_unit = comp_row
                
                files = part_files.get(part_number)
                if not files:
                    continue
                
                s2p_name, zip_path = files
                
                try:
                    if zip_path == '__DIR__':
                        # S2P file is directly on disk (extracted directory)
                        with open(s2p_name, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                    else:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            with zf.open(s2p_name) as f:
                                content = f.read().decode('utf-8', errors='replace')
                    ts_cache[part_number] = parse_touchstone(content, filename=s2p_name)
                except Exception:
                    errors += 1
            
            # Process batch
            for comp_row in batch:
                comp_id, part_number, comp_type, nominal_value, nominal_unit = comp_row
                
                ts_data = ts_cache.get(part_number)
                if ts_data is None:
                    continue
                
                try:
                    srf_hz = self._find_srf(ts_data)
                    
                    for freq_hz in freq_grid_hz:
                        freq_id = freq_map.get(freq_hz)
                        if freq_id is None:
                            continue
                        
                        S = ts_data.get_s_matrix_interpolated(freq_hz)
                        
                        s11 = S[0, 0]
                        s21 = S[0, 1]
                        s12 = S[1, 0]
                        s22 = S[1, 1]
                        
                        z_in, z_trans = self._compute_z_from_s(S)
                        
                        eff_value, eff_unit = self._compute_effective_lc(
                            z_in, freq_hz, comp_type, nominal_value, nominal_unit
                        )
                        
                        s11_mag = abs(s11)
                        s21_mag = abs(s21)
                        q_factor = self._compute_q_factor(z_in, freq_hz, comp_type)
                        
                        c.execute("""
                            INSERT OR REPLACE INTO sparam_at_freq 
                            (component_id, freq_id, s11_re, s11_im, s21_re, s21_im,
                             s12_re, s12_im, s22_re, s22_im)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            comp_id, freq_id,
                            s11.real, s11.imag, s21.real, s21.imag,
                            s12.real, s12.imag, s22.real, s22.imag
                        ))
                        
                        c.execute("""
                            INSERT OR REPLACE INTO derived_at_freq 
                            (component_id, freq_id, z_in_re, z_in_im, z_trans_re, z_trans_im,
                             eff_value, eff_unit, s11_mag, s11_db, s21_mag, s21_db,
                             q_factor, self_resonant_mhz)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            comp_id, freq_id,
                            z_in.real, z_in.imag, z_trans.real, z_trans.imag,
                            eff_value, eff_unit,
                            s11_mag, 20 * np.log10(max(s11_mag, 1e-15)),
                            s21_mag, 20 * np.log10(max(s21_mag, 1e-15)),
                            q_factor,
                            srf_hz / 1e6 if srf_hz else None
                        ))
                    
                    computed += 1
                except Exception as e:
                    errors += 1
            
            # Commit per batch
            self.conn.commit()
            
            if progress_callback:
                done = min(batch_end, total)
                progress_callback(done, total, "Pre-computed %d/%d (errors: %d)" % (done, total, errors))
        
        print("  Pre-computed: %d, errors: %d" % (computed, errors))
    
    def _compute_z_from_s(self, S):
        """
        Compute component impedance from S-matrix in series-mode connection.
        
        Murata S2P data uses series-mode measurement:
        Port1 --[Z_comp]-- Port2
        Port1 ---------- Port2 (ground)
        
        Z_component = 2 * Z0 * S11 / (1 - S11)
        """
        Z0 = 50.0
        S11 = S[0, 0]
        S21 = S[0, 1]
        S22 = S[1, 1]
        
        # Series-mode impedance extraction
        denom = 1.0 - S11
        if abs(denom) > 1e-15:
            z_in = 2 * Z0 * S11 / denom
        else:
            z_in = complex(1e15, 0)  # Very large (near short circuit)
        
        # Transfer impedance (Z21 from S21)
        denom2 = 1.0 - S21
        if abs(denom2) > 1e-15:
            z_trans = 2 * Z0 * S21 / denom2
        else:
            z_trans = complex(0, 0)
        
        return (z_in, z_trans)
    
    def _compute_effective_lc(self, z_in, freq_hz, comp_type, nominal_value, nominal_unit):
        """
        Compute effective inductance or capacitance from impedance.
        
        For series-mode measurement:
          Inductor: Z = R + jwL  -> L = Im(Z)/w
          Capacitor: Z = R - j/(wC) -> C = -1/(w*Im(Z))
        
        Note: Above SRF, inductors appear capacitive and vice versa.
        We report the effective value based on the sign of Im(Z).
        """
        omega = 2 * np.pi * freq_hz
        
        if omega <= 0:
            return (0.0, nominal_unit)
        
        x = z_in.imag  # Reactance
        
        if comp_type == 'inductor':
            # L_eff = X / omega (positive when inductive, negative when capacitive)
            l_h = x / omega
            l_nh = l_h * 1e9
            return (l_nh, 'nH')
        elif comp_type == 'capacitor':
            # C_eff = -1 / (omega * X) (positive when capacitive, negative when inductive)
            if abs(x) > 1e-15:
                c_f = -1.0 / (omega * x)
                c_pf = c_f * 1e12
                return (c_pf, 'pF')
        
        return (0.0, nominal_unit)
    
    def _compute_q_factor(self, z_in, freq_hz, comp_type):
        """
        Compute quality factor Q from impedance.
        Q = |Im(X)| / Re(Z)
        """
        r = z_in.real
        if r > 1e-15:
            return abs(z_in.imag) / r
        return 0.0
    
    def _find_srf(self, ts_data):
        """Find self-resonant frequency where |S21| is minimum."""
        min_s21_mag = float('inf')
        srf_freq = None
        
        for i, freq in enumerate(ts_data.frequencies):
            try:
                s21 = ts_data.get_s(1, 2, i)
                mag = abs(s21)
                if mag < min_s21_mag:
                    min_s21_mag = mag
                    srf_freq = freq
            except (IndexError, KeyError):
                continue
        
        return srf_freq
    
    def _build_value_groups(self):
        """Build value_groups: nominal value -> primary component mapping."""
        c = self.cursor
        
        c.execute("DELETE FROM value_groups")
        
        c.execute("""
            INSERT INTO value_groups 
            (component_type, nominal_value, nominal_unit, primary_component_id, 
             best_tolerance_code, num_alternatives)
            SELECT 
                c1.component_type,
                c1.nominal_value,
                c1.nominal_unit,
                c1.id,
                c1.tolerance_code,
                (SELECT COUNT(*) FROM components c2 
                 WHERE c2.component_type = c1.component_type 
                   AND c2.nominal_value = c1.nominal_value)
            FROM components c1
            WHERE c1.is_primary = 1 AND c1.nominal_value > 0
            ORDER BY c1.component_type, c1.nominal_value
        """)
        
        self.conn.commit()
    
    def _update_statistics(self):
        """Update series-level statistics."""
        c = self.cursor
        
        c.execute("""
            UPDATE series SET 
                min_value = (
                    SELECT MIN(nominal_value) FROM components 
                    WHERE components.series_id = series.id AND nominal_value > 0
                ),
                max_value = (
                    SELECT MAX(nominal_value) FROM components 
                    WHERE components.series_id = series.id AND nominal_value > 0
                )
        """)
        
        self.conn.commit()
    
    # --- Optenni Library Population ---
    
    def populate_from_optenni_dir(self, optenni_dir, freq_grid_mhz=None, progress_callback=None):
        """Scan Optenni ComponentLibrary and populate database (merge with existing)."""
        if freq_grid_mhz is None:
            freq_grid_mhz = DEFAULT_FREQ_GRID_MHZ
        
        freq_grid_hz = [f * 1e6 for f in freq_grid_mhz]
        
        # Insert frequency grid
        self._insert_freq_grid(freq_grid_mhz, freq_grid_hz)
        
        # Scan Optenni library
        if progress_callback:
            progress_callback(0, 0, "Scanning Optenni ComponentLibrary...")
        
        all_parts = scan_optenni_library(optenni_dir)
        
        if progress_callback:
            progress_callback(0, len(all_parts), "Found %d S2P files" % len(all_parts))
        
        # Insert series and components
        series_map = {}
        c = self.cursor
        total = len(all_parts)
        
        # Pre-populate series_map with existing series
        c.execute("SELECT id, name FROM series")
        for row in c.fetchall():
            series_map[row[1]] = row[0]
        
        inserted = 0
        skipped = 0
        
        for idx, (comp_data, s2p_path) in enumerate(all_parts):
            if comp_data is None:
                skipped += 1
                continue
            
            sname = comp_data.series_name
            if sname not in series_map:
                c.execute(
                    "INSERT OR IGNORE INTO series (name, manufacturer, component_type, size_code) VALUES (?, ?, ?, ?)",
                    (sname, comp_data.manufacturer, comp_data.component_type, comp_data.size_code)
                )
                c.execute("SELECT id FROM series WHERE name = ?", (sname,))
                row = c.fetchone()
                if row:
                    series_map[sname] = row[0]
            
            series_id = series_map.get(sname)
            if series_id is None:
                skipped += 1
                continue
            
            precision_rank = get_precision_rank(comp_data.tolerance_code)
            
            # Map tolerance codes consistently
            tolerance_code = comp_data.tolerance_code
            if tolerance_code not in TOLERANCE_MAP:
                tolerance_code = 'J'  # default ±5%
            
            try:
                c.execute("""
                    INSERT OR IGNORE INTO components 
                    (part_number, series_id, component_type, nominal_value, nominal_unit,
                     tolerance_code, tolerance_pct, tolerance_abs, size_code, voltage_code,
                     dielectric, value_str, s2p_filename, zip_path, precision_rank, is_primary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    comp_data.part_number, series_id, comp_data.component_type,
                    comp_data.nominal_value, comp_data.nominal_unit,
                    tolerance_code, comp_data.tolerance_pct, comp_data.tolerance_abs,
                    comp_data.size_code, '', '', comp_data.value_str,
                    comp_data.s2p_filename, s2p_path, precision_rank
                ))
                if self.cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                skipped += 1
            
            if progress_callback and (idx + 1) % 5000 == 0:
                progress_callback(idx + 1, total, 
                    "Optenni: inserted %d, skipped %d (total %d/%d)" % (
                        inserted, skipped, idx + 1, total))
        
        self.conn.commit()
        
        # Update series counts
        c.execute("""
            UPDATE series SET component_count = (
                SELECT COUNT(*) FROM components WHERE components.series_id = series.id
            )
        """)
        self.conn.commit()
        
        # Assign primaries (for new components only)
        # Re-run for all to ensure best precision wins across merged data
        self._assign_primaries()
        
        # Pre-compute S-parameters
        self._precompute_optenni_sparams(freq_grid_hz, progress_callback)
        
        # Rebuild value groups
        self._build_value_groups()
        
        # Update statistics
        self._update_statistics()
        
        if progress_callback:
            progress_callback(total, total, 
                "Optenni population complete: %d inserted, %d skipped" % (inserted, skipped))
    
    def _precompute_optenni_sparams(self, freq_grid_hz, progress_callback=None):
        """Pre-compute S-parameters for Optenni components that need it."""
        c = self.cursor
        
        # Build freq map
        c.execute("SELECT id, freq_hz FROM freq_grid")
        freq_map = {row[1]: row[0] for row in c.fetchall()}
        
        # Precompute for each primary component (or all new ones)
        c.execute("""
            SELECT c.id, c.part_number, c.component_type, 
                   c.nominal_value, c.nominal_unit, c.zip_path
            FROM components c
            WHERE c.is_primary = 1 
              AND NOT EXISTS (SELECT 1 FROM sparam_at_freq WHERE component_id = c.id)
        """)
        new_primaries = c.fetchall()
        
        total = len(new_primaries)
        if total == 0:
            return
        
        if progress_callback:
            progress_callback(0, total, "Pre-computing S-params for %d new components" % total)
        
        errors = 0
        batch = []
        
        for idx, row in enumerate(new_primaries):
            comp_id = row[0]
            part_number = row[1]
            comp_type = row[2]
            nominal_value = row[3]
            nominal_unit = row[4]
            s2p_path = row[5]  # zip_path from the query
            
            batch.append((comp_id, part_number, comp_type, nominal_value, nominal_unit, s2p_path))
            
            if len(batch) >= 100:
                self._precompute_batch(batch, freq_grid_hz, freq_map, errors)
                if progress_callback:
                    progress_callback(idx + 1, total, "Pre-computing S-params...")
                batch = []
        
        if batch:
            self._precompute_batch(batch, freq_grid_hz, freq_map, errors)
        
        if progress_callback:
            progress_callback(total, total, "S-param pre-computation done (%d errors)" % errors)
    
    def _precompute_batch(self, batch, freq_grid_hz, freq_map, errors):
        """Pre-compute S-params and derived values for a batch of components.
        
        Each batch entry: (comp_id, part_number, component_type, nominal_value, nominal_unit, s2p_path)
        """
        from .touchstone import load_touchstone_file as load_ts_file
        
        c = self.cursor
        
        for comp_id, part_number, comp_type, nominal_value, nominal_unit, s2p_path in batch:
            if not s2p_path or not os.path.exists(s2p_path):
                errors += 1
                continue
            
            try:
                ts_data = load_ts_file(s2p_path)
                srf_hz = self._find_srf(ts_data)
                
                for freq_hz in freq_grid_hz:
                    freq_id = freq_map.get(freq_hz)
                    if freq_id is None:
                        continue
                    
                    S = ts_data.get_s_matrix_interpolated(freq_hz)
                    
                    s11 = S[0, 0]
                    s21 = S[0, 1]
                    s12 = S[1, 0]
                    s22 = S[1, 1]
                    
                    z_in, z_trans = self._compute_z_from_s(S)
                    
                    # Effective value (L or C)
                    eff_value, eff_unit = self._compute_effective_lc(
                        z_in, freq_hz, comp_type, nominal_value, nominal_unit
                    )
                    
                    s11_mag = abs(s11)
                    s11_db = 20 * np.log10(s11_mag) if s11_mag > 0 else -200
                    s21_mag = abs(s21)
                    s21_db = 20 * np.log10(s21_mag) if s21_mag > 0 else -200
                    
                    # Q factor from Z
                    if abs(z_in.real) > 1e-12:
                        q_factor = abs(z_in.imag / z_in.real)
                    else:
                        q_factor = 0.0
                    
                    # Insert sparam
                    c.execute("""
                        INSERT OR IGNORE INTO sparam_at_freq
                        (component_id, freq_id, s11_re, s11_im, s21_re, s21_im, s12_re, s12_im, s22_re, s22_im)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (comp_id, freq_id,
                        s11.real, s11.imag, s21.real, s21.imag,
                        s12.real, s12.imag, s22.real, s22.imag))
                    
                    # Insert derived
                    srf_mhz = (srf_hz / 1e6) if srf_hz else None
                    c.execute("""
                        INSERT OR IGNORE INTO derived_at_freq
                        (component_id, freq_id, z_in_re, z_in_im, z_trans_re, z_trans_im,
                         eff_value, eff_unit, s11_mag, s11_db, s21_mag, s21_db,
                         q_factor, self_resonant_mhz)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (comp_id, freq_id,
                        z_in.real, z_in.imag, z_trans.real, z_trans.imag,
                        eff_value, eff_unit,
                        s11_mag, s11_db, s21_mag, s21_db,
                        q_factor, srf_mhz))
                
                self.conn.commit()
                
            except Exception as e:
                errors += 1
                continue
    
    # --- Query Interface ---
    
    def get_primary_inductors(self):
        """Get all primary (highest-precision) inductors."""
        return self._get_components("c.component_type = 'inductor' AND c.is_primary = 1")
    
    def get_primary_capacitors(self):
        """Get all primary (highest-precision) capacitors."""
        return self._get_components("c.component_type = 'capacitor' AND c.is_primary = 1")
    
    def get_inductors_near(self, target_nh, tolerance=0.3, primary_only=True):
        """Get inductors with nominal value within tolerance of target (nH)."""
        condition = "c.component_type = 'inductor' AND c.nominal_value > 0"
        if primary_only:
            condition += " AND c.is_primary = 1"
        condition += " AND ABS(c.nominal_value - %s) / MAX(%s, 1e-9) <= %s" % (
            target_nh, target_nh, tolerance)
        return self._get_components(condition)
    
    def get_capacitors_near(self, target_pf, tolerance=0.3, primary_only=True):
        """Get capacitors with nominal value within tolerance of target (pF)."""
        condition = "c.component_type = 'capacitor' AND c.nominal_value > 0"
        if primary_only:
            condition += " AND c.is_primary = 1"
        condition += " AND ABS(c.nominal_value - %s) / MAX(%s, 1e-9) <= %s" % (
            target_pf, target_pf, tolerance)
        return self._get_components(condition)
    
    def get_component_sparams(self, component_id, freq_mhz):
        """Get S-parameters for a component at a frequency (interpolated)."""
        c = self.cursor
        
        # Find two bracketing grid points
        c.execute("""
            SELECT sp.s11_re, sp.s11_im, sp.s21_re, sp.s21_im,
                   sp.s12_re, sp.s12_im, sp.s22_re, sp.s22_im, fg.freq_mhz
            FROM sparam_at_freq sp
            JOIN freq_grid fg ON sp.freq_id = fg.id
            WHERE sp.component_id = ?
            ORDER BY fg.freq_mhz
        """, (component_id,))
        
        rows = c.fetchall()
        if not rows:
            return None
        
        # If target is at or outside grid boundaries, return nearest
        freqs = np.array([r[8] for r in rows])
        
        if freq_mhz <= freqs[0]:
            r = rows[0]
            return {
                's11': complex(r[0], r[1]), 's21': complex(r[2], r[3]),
                's12': complex(r[4], r[5]), 's22': complex(r[6], r[7]),
                'freq_mhz': r[8],
            }
        if freq_mhz >= freqs[-1]:
            r = rows[-1]
            return {
                's11': complex(r[0], r[1]), 's21': complex(r[2], r[3]),
                's12': complex(r[4], r[5]), 's22': complex(r[6], r[7]),
                'freq_mhz': r[8],
            }
        
        # Find bracketing indices
        idx = np.searchsorted(freqs, freq_mhz)
        r0, r1 = rows[idx - 1], rows[idx]
        f0, f1 = r0[8], r1[8]
        
        # Interpolation weight
        t = (freq_mhz - f0) / (f1 - f0) if f1 != f0 else 0.0
        
        # Linear interpolation of real and imaginary parts
        s11 = complex(r0[0] + t * (r1[0] - r0[0]), r0[1] + t * (r1[1] - r0[1]))
        s21 = complex(r0[2] + t * (r1[2] - r0[2]), r0[3] + t * (r1[3] - r0[3]))
        s12 = complex(r0[4] + t * (r1[4] - r0[4]), r0[5] + t * (r1[5] - r0[5]))
        s22 = complex(r0[6] + t * (r1[6] - r0[6]), r0[7] + t * (r1[7] - r0[7]))
        
        return {
            's11': s11, 's21': s21, 's12': s12, 's22': s22,
            'freq_mhz': freq_mhz,
        }
    
    def get_component_derived(self, component_id, freq_mhz):
        """Get derived parameters for a component at a frequency (interpolated)."""
        c = self.cursor
        
        c.execute("""
            SELECT d.z_in_re, d.z_in_im, d.z_trans_re, d.z_trans_im,
                   d.eff_value, d.eff_unit, d.s11_mag, d.s11_db,
                   d.s21_mag, d.s21_db, d.q_factor, d.self_resonant_mhz,
                   fg.freq_mhz
            FROM derived_at_freq d
            JOIN freq_grid fg ON d.freq_id = fg.id
            WHERE d.component_id = ?
            ORDER BY fg.freq_mhz
        """, (component_id,))
        
        rows = c.fetchall()
        if not rows:
            return None
        
        freqs = np.array([r[12] for r in rows])
        
        if freq_mhz <= freqs[0]:
            r = rows[0]
        elif freq_mhz >= freqs[-1]:
            r = rows[-1]
        else:
            idx = np.searchsorted(freqs, freq_mhz)
            r0, r1 = rows[idx - 1], rows[idx]
            f0, f1 = r0[12], r1[12]
            t = (freq_mhz - f0) / (f1 - f0) if f1 != f0 else 0.0
            
            # Interpolate continuous values
            def lerp(a, b): return a + t * (b - a)
            
            return {
                'z_in': complex(lerp(r0[0], r1[0]), lerp(r0[1], r1[1])),
                'z_trans': complex(lerp(r0[2], r1[2]), lerp(r0[3], r1[3])),
                'eff_value': lerp(r0[4], r1[4]),
                'eff_unit': r0[5],
                's11_mag': lerp(r0[6], r1[6]),
                's11_db': lerp(r0[7], r1[7]),
                's21_mag': lerp(r0[8], r1[8]),
                's21_db': lerp(r0[9], r1[9]),
                'q_factor': lerp(r0[10], r1[10]),
                'self_resonant_mhz': r0[11],  # SRF doesn't interpolate
                'freq_mhz': freq_mhz,
            }
        
        return {
            'z_in': complex(r[0], r[1]),
            'z_trans': complex(r[2], r[3]),
            'eff_value': r[4], 'eff_unit': r[5],
            's11_mag': r[6], 's11_db': r[7],
            's21_mag': r[8], 's21_db': r[9],
            'q_factor': r[10], 'self_resonant_mhz': r[11],
            'freq_mhz': r[12],
        }
    
    def get_sparam_matrix(self, component_id, freq_mhz):
        """Get 2x2 S-matrix at a frequency as numpy array."""
        sp = self.get_component_sparams(component_id, freq_mhz)
        if sp:
            S = np.array([
                [sp['s11'], sp['s12']],
                [sp['s21'], sp['s22']]
            ], dtype=complex)
            return S
        return None
    
    def get_unique_values(self, comp_type):
        """Get sorted list of unique nominal values for a component type."""
        c = self.cursor
        c.execute("""
            SELECT DISTINCT nominal_value 
            FROM components 
            WHERE component_type = ? AND is_primary = 1 AND nominal_value > 0
            ORDER BY nominal_value
        """, (comp_type,))
        return [row[0] for row in c.fetchall()]
    
    def get_all_primaries_with_derived_at_freq(self, freq_mhz, comp_type=None):
        """
        Get all primary components with derived data at a frequency.
        Main query for the optimizer.
        """
        c = self.cursor
        
        type_filter = ""
        if comp_type:
            type_filter = "AND c.component_type = '%s'" % comp_type
        
        c.execute("""
            SELECT c.id, c.part_number, s.name, c.component_type,
                   c.nominal_value, c.nominal_unit, c.tolerance_code,
                   d.z_in_re, d.z_in_im, d.z_trans_re, d.z_trans_im,
                   d.eff_value, d.eff_unit, d.s11_mag, d.s11_db,
                   d.s21_mag, d.s21_db, d.q_factor, d.self_resonant_mhz,
                   fg.freq_mhz
            FROM components c
            JOIN series s ON c.series_id = s.id
            JOIN derived_at_freq d ON c.id = d.component_id
            JOIN freq_grid fg ON d.freq_id = fg.id
            WHERE c.is_primary = 1 %s
        """ % type_filter)
        
        results = []
        seen = set()
        
        for row in c.fetchall():
            comp_id = row[0]
            if comp_id in seen:
                continue
            seen.add(comp_id)
            
            results.append({
                'id': comp_id,
                'part_number': row[1],
                'series': row[2],
                'component_type': row[3],
                'nominal_value': row[4],
                'nominal_unit': row[5],
                'tolerance_code': row[6],
                'z_in': complex(row[7], row[8]),
                'z_trans': complex(row[9], row[10]),
                'eff_value': row[11],
                'eff_unit': row[12],
                's11_mag': row[13],
                's11_db': row[14],
                's21_mag': row[15],
                's21_db': row[16],
                'q_factor': row[17],
                'self_resonant_mhz': row[18],
                'freq_mhz': row[19],
            })
        
        return results
    
    def get_statistics(self):
        """Get database statistics."""
        c = self.cursor
        stats = {}
        
        c.execute("SELECT COUNT(*) FROM components")
        stats['total_components'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM components WHERE is_primary = 1")
        stats['primary_components'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM series")
        stats['total_series'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM freq_grid")
        stats['freq_grid_points'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM sparam_at_freq")
        stats['sparam_records'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM derived_at_freq")
        stats['derived_records'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM value_groups")
        stats['unique_values'] = c.fetchone()[0]
        
        for comp_type in ['inductor', 'capacitor']:
            c.execute("""
                SELECT COUNT(*) FROM components 
                WHERE component_type = '%s' AND is_primary = 1
            """ % comp_type)
            stats['primary_%ss' % comp_type] = c.fetchone()[0]
            
            c.execute("""
                SELECT MIN(nominal_value), MAX(nominal_value), 
                       COUNT(DISTINCT nominal_value)
                FROM components 
                WHERE component_type = '%s' AND is_primary = 1 AND nominal_value > 0
            """ % comp_type)
            row = c.fetchone()
            stats['%s_min_value' % comp_type] = row[0]
            stats['%s_max_value' % comp_type] = row[1]
            stats['%s_unique_values' % comp_type] = row[2]
        
        return stats
    
    def export_series_summary(self):
        """Export summary of all series."""
        c = self.cursor
        c.execute("""
            SELECT s.name, s.manufacturer, s.component_type, s.size_code, s.component_count,
                   s.min_value, s.max_value
            FROM series s
            ORDER BY s.manufacturer, s.component_type, s.name
        """)
        
        return [
            {
                'name': row[0], 'manufacturer': row[1], 'type': row[2], 'size': row[3],
                'count': row[4], 'min_value': row[5], 'max_value': row[6]
            }
            for row in c.fetchall()
        ]
    
    def _get_components(self, where_clause):
        """Get components matching a WHERE clause."""
        c = self.cursor
        c.execute("""
            SELECT c.id, c.part_number, s.name, c.component_type, c.nominal_value, c.nominal_unit,
                   c.tolerance_code, c.tolerance_pct, c.tolerance_abs, c.size_code, c.voltage_code,
                   c.dielectric, c.value_str, c.s2p_filename, c.zip_path, c.is_primary
            FROM components c
            JOIN series s ON c.series_id = s.id
            WHERE %s
            ORDER BY c.nominal_value
        """ % where_clause)
        
        results = []
        for row in c.fetchall():
            results.append(ComponentRecord(
                id=row[0], part_number=row[1], series=row[2],
                component_type=row[3], nominal_value=row[4], nominal_unit=row[5],
                tolerance_code=row[6], tolerance_pct=row[7], tolerance_abs=row[8],
                size_code=row[9], voltage_code=row[10], dielectric=row[11],
                value_str=row[12], s2p_filename=row[13], zip_path=row[14],
                is_primary=bool(row[15])
            ))
        
        return results


# --- Convenience ---

def build_database(murata_dir, db_path, freq_grid_mhz=None, progress_callback=None):
    """Build the complete Murata database from scratch."""
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = MurataDatabase(db_path)
    db.connect()
    db.create_schema()
    db.populate_from_murata_dir(murata_dir, freq_grid_mhz, progress_callback)
    
    return db


def build_full_database(murata_dir, optenni_dir, db_path, freq_grid_mhz=None, progress_callback=None):
    """Build a combined database from both Murata and Optenni libraries.
    
    If Murata ZIP files are available, imports from them.
    Otherwise copies existing murata_components.db as starting point.
    """
    import shutil
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = MurataDatabase(db_path)
    db.connect()
    db.create_schema()
    
    # Phase 1: Murata data
    if progress_callback:
        progress_callback(0, 0, "Phase 1: Importing Murata components...")
    
    # Check if Murata ZIPs exist
    has_zips = False
    if os.path.isdir(murata_dir):
        has_zips = any(f.lower().endswith('.zip') for f in os.listdir(murata_dir))
    
    if has_zips:
        db.populate_from_murata_dir(murata_dir, freq_grid_mhz, 
            lambda c, t, m: progress_callback(c, t, "Murata: " + m) if progress_callback else None)
    else:
        # Try copying existing murata_components.db
        legacy_db = os.path.join(murata_dir, 'murata_components.db')
        if os.path.isfile(legacy_db):
            if progress_callback:
                progress_callback(0, 0, "Murata: copying existing database...")
            db.close()
            shutil.copy2(legacy_db, db_path)
            db.conn = sqlite3.connect(db_path, check_same_thread=False)
            db.conn.execute("PRAGMA journal_mode=WAL")
            db.conn.execute("PRAGMA synchronous=NORMAL")
            db.conn.execute("PRAGMA cache_size=-64000")
            db.conn.row_factory = sqlite3.Row
            db.cursor = db.conn.cursor()
            db._add_manufacturer_column()
            db.cursor.execute("UPDATE series SET manufacturer='Murata' WHERE manufacturer='' OR manufacturer IS NULL")
            db.conn.commit()
            stats = db.get_statistics()
            if progress_callback:
                progress_callback(stats['total_components'], stats['total_components'],
                    "Murata: loaded %d components from existing DB" % stats['total_components'])
        else:
            if progress_callback:
                progress_callback(0, 0, "Murata: no ZIPs or existing DB found, skipping")
    
    # Phase 2: Optenni data (merge)
    if progress_callback:
        progress_callback(0, 0, "Phase 2: Merging Optenni components...")
    db.populate_from_optenni_dir(optenni_dir, freq_grid_mhz,
        lambda c, t, m: progress_callback(c, t, "Optenni: " + m) if progress_callback else None)
    
    return db


if __name__ == '__main__':
    """Build the database from command line."""
    
    import sys
    
    murata_dir = r'E:\RF matching\Murata'
    optenni_dir = r'C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary'
    db_path = r'E:\RF matching\Murata\full_components.db'
    
    # Check command-line args
    mode = 'full'  # 'murata', 'optenni', or 'full'
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ('murata', 'optenni', 'full'):
            print("Usage: python -m engine.murata_db [murata|optenni|full]")
            print("  Default: full (both Murata and Optenni)")
            sys.exit(1)
    
    # Override db_path for non-full modes
    if mode == 'murata':
        db_path = r'E:\RF matching\Murata\murata_components.db'
    elif mode == 'optenni':
        db_path = r'E:\RF matching\Murata\optenni_components.db'
    else:
        db_path = r'E:\RF matching\Murata\full_components.db'
    
    def progress(current, total, msg):
        pct = current / total * 100 if total > 0 else 0
        sys.stdout.write("\r  [%5.1f%%] %s" % (pct, msg))
        sys.stdout.flush()
        if current == total:
            print()
    
    t0 = time.time()
    
    if mode == 'murata':
        print("Building Murata-only database...")
        print("  Source: %s" % murata_dir)
        print("  Output: %s" % db_path)
        print()
        db = build_database(murata_dir, db_path, progress_callback=progress)
    elif mode == 'optenni':
        print("Building Optenni-only database...")
        print("  Source: %s" % optenni_dir)
        print("  Output: %s" % db_path)
        print()
        db = MurataDatabase(db_path)
        db.connect()
        db.create_schema()
        db.populate_from_optenni_dir(optenni_dir, progress_callback=progress)
    else:
        print("Building FULL component database (Murata + Optenni)...")
        print("  Murata:  %s" % murata_dir)
        print("  Optenni: %s" % optenni_dir)
        print("  Output:  %s" % db_path)
        print()
        db = build_full_database(murata_dir, optenni_dir, db_path, progress_callback=progress)
    
    t1 = time.time()
    
    print()
    print("Database built in %.1f seconds" % (t1 - t0))
    print()
    
    stats = db.get_statistics()
    print("=== Database Statistics ===")
    for key, value in stats.items():
        print("  %s: %s" % (key, value))
    
    print()
    print("=== Series Summary ===")
    series_list = db.export_series_summary()
    mfr_counts = {}
    for s in series_list:
        mfr = s.get('manufacturer', 'Unknown')
        mfr_counts[mfr] = mfr_counts.get(mfr, 0) + s['count']
    
    for mfr, count in sorted(mfr_counts.items(), key=lambda x: -x[1]):
        print("  %s: %d components" % (mfr, count))
    
    print()
    print("=== Sample Queries ===")
    
    try:
        ind_values = db.get_unique_values('inductor')
        cap_values = db.get_unique_values('capacitor')
        if ind_values:
            print("  Unique inductor values: %d (%.2f to %.1f nH)" % (
                len(ind_values), ind_values[0], ind_values[-1]))
        if cap_values:
            print("  Unique capacitor values: %d (%.2f to %.1f pF)" % (
                len(cap_values), cap_values[0], cap_values[-1]))
        
        # Find components near 10nH
        near_10nH = db.get_inductors_near(10.0, tolerance=0.1)
        print("\n  Inductors near 10nH (primary): %d" % len(near_10nH))
        for comp in near_10nH[:5]:
            print("    %s: %snH (%s, %s)" % (
                comp.part_number, comp.nominal_value, comp.tolerance_code, comp.series))
        
        # Find components near 100pF
        near_100pF = db.get_capacitors_near(100.0, tolerance=0.1)
        print("\n  Capacitors near 100pF (primary): %d" % len(near_100pF))
        for comp in near_100pF[:5]:
            print("    %s: %spF (%s, %s)" % (
                comp.part_number, comp.nominal_value, comp.tolerance_code, comp.series))
        
        # Test derived data
        if near_10nH:
            comp = near_10nH[0]
            derived_900 = db.get_component_derived(comp.id, 900.0)
            if derived_900:
                print("\n  %s at 900MHz:" % comp.part_number)
                print("    Z_in = %.2f + j%.2f ohm" % (
                    derived_900['z_in'].real, derived_900['z_in'].imag))
                print("    Effective L = %.2f %s" % (
                    derived_900['eff_value'], derived_900['eff_unit']))
                print("    |S11| = %.4f (%.1f dB)" % (
                    derived_900['s11_mag'], derived_900['s11_db']))
                print("    Q = %.1f" % derived_900['q_factor'])
                if derived_900['self_resonant_mhz']:
                    print("    SRF = %.0f MHz" % derived_900['self_resonant_mhz'])
        
        # Benchmark: time a bulk query
        t2 = time.time()
        all_ind = db.get_all_primaries_with_derived_at_freq(900.0, comp_type='inductor')
        t3 = time.time()
        print("\n  Bulk query: %d inductors with derived data at 900MHz in %.1f ms" % (
            len(all_ind), (t3 - t2) * 1000))
        
    except Exception as e:
        print("  Query test failed: %s" % e)
    
    db.close()
