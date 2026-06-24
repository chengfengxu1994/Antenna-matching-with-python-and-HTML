"""
Bandwidth Potential Assessment — Estimation of obtainable bandwidth.

Inspired by Optenni Lab's Bandwidth Potential tool (Rahola, EuCAP 2009).

The idea: for each center frequency f₀ in the antenna impedance data,
synthesize an optimal 2-component L-network matching circuit at f₀,
then sweep the full frequency range to find the bandwidth where
|S11| drops below a given threshold (e.g., -6 dB).

The result is a curve: obtainable relative bandwidth (%) vs center frequency.
This reveals "hidden" bandwidth that is not obvious from looking at the raw
impedance curve alone, and helps the designer quickly decide whether the
antenna is worth pursuing before committing to a full matching design.

Q factor estimation is also provided (Yaghjian & Best, IEEE TAP 2005).
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import time

from .touchstone import TouchstoneData
from .network import s_to_z


@dataclass
class BandwidthPotentialPoint:
    """Result at one center frequency."""
    center_freq_hz: float
    absolute_bandwidth_hz: float       # f_high - f_low
    relative_bandwidth_pct: float      # (f_high - f_low) / f0 * 100
    s11_at_center_db: float            # S11 at center after matching
    matching_L_nH: Optional[float] = None  # Ideal series inductance (nH)
    matching_C_pF: Optional[float] = None  # Ideal shunt capacitance (pF)
    f_low_hz: float = 0.0              # Lower -6dB edge
    f_high_hz: float = 0.0             # Upper -6dB edge
    q_factor: Optional[float] = None   # Estimated Q factor


@dataclass
class BandwidthPotentialResult:
    """Full bandwidth potential analysis result."""
    points: List[BandwidthPotentialPoint]
    s11_threshold_db: float            # The threshold used (e.g., -6 dB)
    reference_impedance: float         # Z0 (default 50)
    freq_min_hz: float
    freq_max_hz: float
    # Summary stats
    max_potential_pct: float = 0.0     # Best relative bandwidth found
    max_potential_freq_hz: float = 0.0 # Frequency of best bandwidth
    avg_potential_pct: float = 0.0     # Average over all points

    def to_dict(self) -> dict:
        return {
            's11_threshold_db': self.s11_threshold_db,
            'reference_impedance': self.reference_impedance,
            'freq_min_hz': self.freq_min_hz,
            'freq_max_hz': self.freq_max_hz,
            'max_potential_pct': self.max_potential_pct,
            'max_potential_freq_hz': self.max_potential_freq_hz,
            'avg_potential_pct': self.avg_potential_pct,
            'points': [
                {
                    'center_freq_hz': p.center_freq_hz,
                    'center_freq_mhz': p.center_freq_hz / 1e6,
                    'absolute_bandwidth_hz': p.absolute_bandwidth_hz,
                    'relative_bandwidth_pct': p.relative_bandwidth_pct,
                    's11_at_center_db': p.s11_at_center_db,
                    'f_low_hz': p.f_low_hz,
                    'f_high_hz': p.f_high_hz,
                    'matching_L_nH': p.matching_L_nH,
                    'matching_C_pF': p.matching_C_pF,
                    'q_factor': p.q_factor,
                }
                for p in self.points
            ]
        }


def _s11_db_threshold_to_linear(db: float) -> float:
    """Convert S11 threshold in dB to linear magnitude.
    e.g., -6 dB → 0.501, -10 dB → 0.316
    """
    return 10.0 ** (db / 20.0)


def _compute_l_network_matching(
    Z_load: complex,
    Z0: float,
    f_hz: float
) -> List[Tuple[Optional[float], Optional[float], str, str]]:
    """
    Compute all ideal 2-component L-network solutions to match Z_load to Z0.
    
    Returns list of (L_nH, C_pF, series_type, shunt_type) where:
    - L_nH: inductance in nH if series element is an inductor (None otherwise)
    - C_pF: capacitance in pF if series element is a capacitor (None otherwise)
    - series_type: 'L' or 'C'
    - shunt_type: 'L' or 'C'
    """
    R = Z_load.real
    X = Z_load.imag
    omega = 2 * math.pi * f_hz
    
    if R <= 0 or omega <= 0:
        return []
    
    solutions = []
    
    # Case 1: R < Z0 → shunt element first (on load side), then series
    if R < Z0:
        Q = math.sqrt(Z0 / R - 1.0)
        
        for sign in [1, -1]:
            # Shunt susceptance: B = ±Q/Z0
            B = sign * Q / Z0
            # Series reactance: Xs = -X ± sqrt(R*(Z0-R))
            Xs = -X + sign * math.sqrt(R * (Z0 - R))
            
            # Series element
            if Xs >= 0:
                # Inductor: Xs = ωL → L = Xs/ω
                L_nH = (Xs / omega) * 1e9
                series_val = L_nH
                series_type = 'L'
            else:
                # Capacitor: Xs = -1/(ωC) → C = -1/(ω*Xs)
                C_pF = (-1.0 / (omega * Xs)) * 1e12
                series_val = C_pF
                series_type = 'C'
            
            # Shunt element
            if B >= 0:
                # Capacitor: B = ωC → C = B/ω
                C_pF_shunt = (B / omega) * 1e12
                shunt_type = 'C'
            else:
                # Inductor: B = -1/(ωL) → L = -1/(ω*B)
                L_nH_shunt = (-1.0 / (omega * B)) * 1e9
                shunt_type = 'L'
            
            if series_type == 'L':
                solutions.append((series_val, None, 'L', shunt_type))
            else:
                solutions.append((None, series_val, 'C', shunt_type))
    
    # Case 2: R >= Z0 → series element first, then shunt
    if R >= Z0:
        for sign in [1, -1]:
            disc = R * Z0 - R * R + X * X * R / Z0
            if disc < 0:
                continue
            
            Xs = (X + sign * math.sqrt(disc)) / (R / Z0)
            denom = R * Z0 - R * R + X * X
            if abs(denom) < 1e-15:
                continue
            B = sign * math.sqrt(disc) / denom
            
            # Series element
            if Xs >= 0:
                L_nH = (Xs / omega) * 1e9
                series_val = L_nH
                series_type = 'L'
            else:
                C_pF = (-1.0 / (omega * Xs)) * 1e12
                series_val = C_pF
                series_type = 'C'
            
            # Shunt element
            if B >= 0:
                shunt_type = 'C'
            else:
                shunt_type = 'L'
            
            if series_type == 'L':
                solutions.append((series_val, None, 'L', shunt_type))
            else:
                solutions.append((None, series_val, 'C', shunt_type))
    
    return solutions


def _apply_ideal_l_network(
    Z_load_array: np.ndarray,   # Complex impedance at each frequency
    freqs_hz: np.ndarray,        # Frequency array (Hz)
    center_idx: int,             # Index of center frequency in freqs_hz
    L_nH: Optional[float],      # Series inductance (nH) or None
    C_pF: Optional[float],      # Series capacitance (pF) or None
    shunt_type: str,             # 'L' or 'C' for shunt element
    Z0: float = 50.0
) -> np.ndarray:
    """
    Apply an ideal L-network (designed at center frequency) across all frequencies.
    
    Circuit topology (for R < Z0 case):
        Port 0 ── [Series L or C] ──┬── Port 1 (to load)
                                     │
                               [Shunt C or L]
                                     │
                                    GND
    
    Returns S11 array (complex) at each frequency.
    """
    omega = 2 * math.pi * freqs_hz
    s11_array = np.zeros(len(freqs_hz), dtype=complex)
    
    # Get the shunt component value at center frequency
    omega_0 = 2 * math.pi * freqs_hz[center_idx]
    Z_center = Z_load_array[center_idx]
    R_center = Z_center.real
    X_center = Z_center.imag
    
    # Calculate shunt value from the matching solution
    if R_center < Z0:
        Q = math.sqrt(Z0 / R_center - 1.0)
    else:
        Q = 1.0  # Will be recalculated
    
    # Get shunt B at center
    solutions = _compute_l_network_matching(Z_center, Z0, freqs_hz[center_idx])
    if not solutions:
        return np.ones(len(freqs_hz), dtype=complex)  # Total reflection
    
    # Use first valid solution
    sol = solutions[0]
    L_s_nH, C_s_pF, series_t, shunt_t = sol
    
    # Series impedance at each frequency
    if series_t == 'L' and L_s_nH is not None:
        Z_series = 1j * omega * (L_s_nH * 1e-9)
    elif series_t == 'C' and C_s_pF is not None:
        Z_series = 1.0 / (1j * omega * (C_s_pF * 1e-12))
    else:
        return np.ones(len(freqs_hz), dtype=complex)
    
    # Shunt admittance at each frequency
    # We need to calculate the shunt value from the solution
    # Recalculate B at center
    if R_center < Z0:
        sign_b = 1  # Default to first solution
        B_center = sign_b * Q / Z0
    else:
        disc = R_center * Z0 - R_center**2 + X_center**2 * R_center / Z0
        if disc < 0:
            return np.ones(len(freqs_hz), dtype=complex)
        sign_b = 1
        denom = R_center * Z0 - R_center**2 + X_center**2
        if abs(denom) < 1e-15:
            return np.ones(len(freqs_hz), dtype=complex)
        B_center = sign_b * math.sqrt(disc) / denom
    
    if shunt_t == 'C':
        # B = ωC → C = B/ω₀
        C_shunt = B_center / omega_0
        Y_shunt = 1j * omega * C_shunt  # Y = jωC
    else:
        # B = -1/(ωL) → L = -1/(ω₀*B)
        if abs(B_center) < 1e-20:
            return np.ones(len(freqs_hz), dtype=complex)
        L_shunt = -1.0 / (omega_0 * B_center)
        Y_shunt = 1.0 / (1j * omega * L_shunt)  # Y = 1/(jωL)
    
    # Calculate S11 at each frequency
    for k in range(len(freqs_hz)):
        Z_L = Z_load_array[k]
        
        # Apply series element: Z_after_series = Z_L + Z_series
        Z_in_series = Z_L + Z_series[k]
        
        # Apply shunt element: Y_total = 1/Z_in_series + Y_shunt
        if abs(Z_in_series) < 1e-30:
            s11_array[k] = 1.0
            continue
        Y_total = 1.0 / Z_in_series + Y_shunt[k]
        
        # Convert to impedance
        if abs(Y_total) < 1e-30:
            s11_array[k] = 1.0
            continue
        Z_in = 1.0 / Y_total
        
        # S11 from Z_in
        s11_array[k] = (Z_in - Z0) / (Z_in + Z0)
    
    return s11_array


def _find_bandwidth(
    freqs_hz: np.ndarray,
    s11_db: np.ndarray,
    center_idx: int,
    threshold_db: float
) -> Tuple[float, float, float]:
    """
    Find the -XdB bandwidth around center_idx.
    
    Returns: (f_low_hz, f_high_hz, relative_bandwidth_pct)
    """
    n = len(freqs_hz)
    threshold_linear = _s11_db_threshold_to_linear(threshold_db)
    
    # Scan outward from center to find where |S11| exceeds threshold
    f_low_idx = 0
    f_high_idx = n - 1
    
    # Find lower edge
    for k in range(center_idx, -1, -1):
        if s11_db[k] > threshold_db:  # s11_db is negative, so "worse" means more positive
            f_low_idx = k
            break
    else:
        f_low_idx = 0
    
    # Find upper edge
    for k in range(center_idx, n):
        if s11_db[k] > threshold_db:
            f_high_idx = k
            break
    else:
        f_high_idx = n - 1
    
    f_low = freqs_hz[f_low_idx]
    f_high = freqs_hz[f_high_idx]
    f_center = freqs_hz[center_idx]
    
    bw = f_high - f_low
    rel_bw = (bw / f_center * 100.0) if f_center > 0 else 0.0
    
    return f_low, f_high, rel_bw


def _estimate_q_factor(
    Z_load: complex,
    f_hz: float,
    Z0: float = 50.0
) -> float:
    """
    Estimate the Q factor of the antenna at a given frequency.
    
    Uses the Yaghjian & Best method:
    Q ≈ ω * |dZ/dω| / (2 * R)
    
    For a simple RLC model: Q = |X| / R when X is the reactance.
    This is an approximation at a single frequency point.
    """
    R = Z_load.real
    X = Z_load.imag
    
    if R <= 0:
        return float('inf')
    
    # Simple Q estimate: |X|/R
    return abs(X) / R


def compute_bandwidth_potential(
    dut: TouchstoneData,
    s11_threshold_db: float = -6.0,
    reference_impedance: float = 50.0,
    freq_step_hz: Optional[float] = None,
    max_points: int = 500,
    progress_callback: Optional[callable] = None,
) -> BandwidthPotentialResult:
    """
    Compute the bandwidth potential curve for an antenna.
    
    For each center frequency f₀ in the impedance data:
    1. Compute Z_load(f₀)
    2. Synthesize optimal 2-component L-network to match to Z0
    3. Apply this matching across the full frequency range
    4. Find the bandwidth where |S11| < threshold
    
    Args:
        dut: Touchstone data of the antenna
        s11_threshold_db: S11 threshold in dB (e.g., -6.0 for -6dB)
        reference_impedance: System impedance (default 50 Ω)
        freq_step_hz: Optional frequency step for resampling
        max_points: Maximum number of center frequencies to evaluate
        progress_callback: Optional callback(progress_float, message_str)
    
    Returns:
        BandwidthPotentialResult with the potential curve
    """
    start_time = time.time()
    
    freqs = np.array(dut.frequencies)
    n_freqs = len(freqs)
    
    if n_freqs < 3:
        raise ValueError("Need at least 3 frequency points for bandwidth potential")
    
    # Subsample if too many points
    if n_freqs > max_points:
        indices = np.linspace(0, n_freqs - 1, max_points, dtype=int)
    else:
        indices = np.arange(n_freqs)
    
    # Pre-compute Z_load at all frequencies
    Z_load_all = np.zeros(n_freqs, dtype=complex)
    for k in range(n_freqs):
        S = dut.get_s_matrix(k)
        Z = s_to_z(S, reference_impedance)
        Z_load_all[k] = Z[0, 0]  # Port 0 impedance
    
    # Also get Z at all frequencies for the sweep
    # We'll use the full frequency array for bandwidth measurement
    Z_load_sweep = Z_load_all.copy()
    
    points = []
    total = len(indices)
    
    for progress_idx, center_idx in enumerate(indices):
        if progress_callback and progress_idx % 10 == 0:
            progress_callback(
                progress_idx / total,
                f"Computing bandwidth potential: {progress_idx}/{total}"
            )
        
        f0 = freqs[center_idx]
        Z0_load = Z_load_all[center_idx]
        
        # Skip if impedance is degenerate
        if Z0_load.real <= 0:
            points.append(BandwidthPotentialPoint(
                center_freq_hz=f0,
                absolute_bandwidth_hz=0.0,
                relative_bandwidth_pct=0.0,
                s11_at_center_db=0.0,
                q_factor=float('inf'),
            ))
            continue
        
        # Compute matching solutions at center frequency
        solutions = _compute_l_network_matching(Z0_load, reference_impedance, f0)
        
        if not solutions:
            points.append(BandwidthPotentialPoint(
                center_freq_hz=f0,
                absolute_bandwidth_hz=0.0,
                relative_bandwidth_pct=0.0,
                s11_at_center_db=0.0,
                q_factor=_estimate_q_factor(Z0_load, f0, reference_impedance),
            ))
            continue
        
        # Try all L-network solutions and pick the one with best bandwidth
        best_bw = 0.0
        best_point = None
        
        for sol in solutions:
            L_nH, C_pF, series_t, shunt_t = sol
            
            # Apply this matching circuit across all frequencies
            s11_complex = _apply_ideal_l_network(
                Z_load_sweep, freqs, center_idx,
                L_nH, C_pF, shunt_t,
                reference_impedance
            )
            
            s11_mag = np.abs(s11_complex)
            s11_db = 20 * np.log10(np.maximum(s11_mag, 1e-15))
            
            # Find bandwidth
            f_low, f_high, rel_bw = _find_bandwidth(
                freqs, s11_db, center_idx, s11_threshold_db
            )
            
            if rel_bw > best_bw:
                best_bw = rel_bw
                abs_bw = f_high - f_low
                s11_center_db = float(s11_db[center_idx])
                
                best_point = BandwidthPotentialPoint(
                    center_freq_hz=f0,
                    absolute_bandwidth_hz=abs_bw,
                    relative_bandwidth_pct=rel_bw,
                    s11_at_center_db=s11_center_db,
                    matching_L_nH=L_nH,
                    matching_C_pF=C_pF,
                    f_low_hz=f_low,
                    f_high_hz=f_high,
                    q_factor=_estimate_q_factor(Z0_load, f0, reference_impedance),
                )
        
        if best_point is not None:
            points.append(best_point)
        else:
            points.append(BandwidthPotentialPoint(
                center_freq_hz=f0,
                absolute_bandwidth_hz=0.0,
                relative_bandwidth_pct=0.0,
                s11_at_center_db=0.0,
                q_factor=_estimate_q_factor(Z0_load, f0, reference_impedance),
            ))
    
    if progress_callback:
        progress_callback(1.0, "Bandwidth potential computation complete")
    
    # Compute summary stats
    bw_values = [p.relative_bandwidth_pct for p in points if p.relative_bandwidth_pct > 0]
    
    if bw_values:
        max_bw = max(bw_values)
        max_bw_idx = bw_values.index(max_bw)
        max_bw_freq = points[max_bw_idx].center_freq_hz
        avg_bw = float(np.mean(bw_values))
    else:
        max_bw = 0.0
        max_bw_freq = 0.0
        avg_bw = 0.0
    
    result = BandwidthPotentialResult(
        points=points,
        s11_threshold_db=s11_threshold_db,
        reference_impedance=reference_impedance,
        freq_min_hz=float(freqs[0]),
        freq_max_hz=float(freqs[-1]),
        max_potential_pct=max_bw,
        max_potential_freq_hz=max_bw_freq,
        avg_potential_pct=avg_bw,
    )
    
    elapsed = time.time() - start_time
    if progress_callback:
        progress_callback(1.0, f"Done in {elapsed:.1f}s. "
                          f"Max bandwidth potential: {max_bw:.1f}% at {max_bw_freq/1e6:.1f} MHz")
    
    return result


def compute_bandwidth_potential_multi_threshold(
    dut: TouchstoneData,
    thresholds_db: List[float] = [-3.0, -6.0, -10.0],
    reference_impedance: float = 50.0,
    max_points: int = 300,
    progress_callback: Optional[callable] = None,
) -> Dict[float, BandwidthPotentialResult]:
    """
    Compute bandwidth potential for multiple S11 thresholds.
    
    This creates a "bandwidth map" — useful for quick visual assessment
    of how the antenna performs at different matching levels.
    
    Args:
        dut: Touchstone data
        thresholds_db: List of S11 thresholds (e.g., [-3, -6, -10])
        reference_impedance: Z0
        max_points: Max center frequencies per threshold
        progress_callback: Optional progress callback
    
    Returns:
        Dict mapping threshold_db → BandwidthPotentialResult
    """
    results = {}
    total_thresholds = len(thresholds_db)
    
    for i, threshold in enumerate(thresholds_db):
        def sub_progress(pct, msg):
            overall = (i + pct) / total_thresholds
            if progress_callback:
                progress_callback(overall, f"[{i+1}/{total_thresholds}] {msg}")
        
        results[threshold] = compute_bandwidth_potential(
            dut,
            s11_threshold_db=threshold,
            reference_impedance=reference_impedance,
            max_points=max_points,
            progress_callback=sub_progress,
        )
    
    return results
