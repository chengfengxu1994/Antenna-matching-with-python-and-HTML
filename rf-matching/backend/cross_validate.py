"""
Cross-validate database vs official Murata impedance data.
Uses S2P file directly since the ZIP may have been extracted.
"""
import sys, os
sys.path.insert(0, '.')
import numpy as np
from engine.touchstone import parse_touchstone
from engine.murata_parser import parse_murata_part_number

PART = 'GJM0335C0J270JB01'
Z_DIR = r'E:\RF matching\Murata\z'
S2P_PATH = r'E:\RF matching\Murata\sparameter-mlcc20260619152334\gjm-s-v76\Temperature_compensating\0603_0201\GJM0335C0J270JB01.s2p'

def load_csv(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    data[float(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    return data

# Part number parsing
info = parse_murata_part_number(PART)
print("=== Part Number Analysis ===")
print("  Part: %s" % PART)
print("  Series: %s" % info.series)
print("  Type: %s" % info.component_type)
print("  Nominal: %s %s" % (info.nominal_value, info.nominal_unit))
print("  Tolerance: %s (rank=%d)" % (info.tolerance_code, 
    __import__('engine.murata_parser', fromlist=['get_precision_rank']).get_precision_rank(info.tolerance_code)))
print("  Expected: 27 pF, tol=J (5%%)")

# Load official data
print("\n=== Loading Official Murata Data ===")
z_off = load_csv(os.path.join(Z_DIR, '%s_InProduction.csv' % PART))
r_off = load_csv(os.path.join(Z_DIR, '%s_InProduction (1).csv' % PART))
x_off = load_csv(os.path.join(Z_DIR, '%s_InProduction (2).csv' % PART))
q_off = load_csv(os.path.join(Z_DIR, '%s_InProduction (3).csv' % PART))
c_off = load_csv(os.path.join(Z_DIR, '%s_InProduction (6).csv' % PART))
print("  z=%d, r=%d, x=%d, q=%d, c=%d points" % (
    len(z_off), len(r_off), len(x_off), len(q_off), len(c_off)))

# Load S2P
with open(S2P_PATH, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
ts = parse_touchstone(content, filename=S2P_PATH)
print("  S2P: %d freq points, %.0f - %.0f Hz" % (
    len(ts.frequencies), ts.frequencies[0], ts.frequencies[-1]))

# Cross-validate
Z0 = 50.0
print("\n" + "="*120)
print("CROSS-VALIDATION: S2P (series-mode) vs Official Murata")
print("="*120)
print("%-14s | %-10s %-10s %-8s | %-10s %-10s %-8s | %-10s %-10s %-8s | %-8s %-8s %-8s | %s" % (
    "Freq(MHz)", "|Z|_off", "|Z|_S2P", "err%",
    "R_off", "R_S2P", "err%",
    "X_off", "X_S2P", "err%",
    "C_off(pF)", "C_S2P(pF)", "err%",
    "Status"))
print("-"*120)

freqs_hz = sorted(z_off.keys())
step = max(1, len(freqs_hz) // 25)
max_err_z = 0
max_err_c = 0
n_ok = 0

for freq_hz in freqs_hz[::step]:
    freq_mhz = freq_hz / 1e6
    
    # Official
    z_o = z_off.get(freq_hz, 0)
    r_o = r_off.get(freq_hz, 0)
    x_o = x_off.get(freq_hz, 0)
    c_o = c_off.get(freq_hz, 0) * 1e12  # to pF
    
    if z_o <= 0:
        continue
    
    # S2P (interpolated)
    if freq_hz < ts.frequencies[0] or freq_hz > ts.frequencies[-1]:
        continue
    
    S = ts.get_s_matrix_interpolated(freq_hz)
    Z_comp = 2 * Z0 * S[0,0] / (1 - S[0,0])
    z_s = abs(Z_comp)
    r_s = Z_comp.real
    x_s = Z_comp.imag
    
    omega = 2 * np.pi * freq_hz
    if abs(x_s) > 1e-15 and x_s < 0:
        c_s = -1.0 / (omega * x_s) * 1e12
    else:
        c_s = 0  # Above SRF, not capacitive
    
    err_z = abs(z_s - z_o) / z_o * 100
    err_c = abs(c_s - c_o) / max(c_o, 0.001) * 100 if c_o > 0 and c_s > 0 else 0
    
    max_err_z = max(max_err_z, err_z)
    if c_s > 0 and c_o > 0:
        max_err_c = max(max_err_c, err_c)
    
    status = "OK" if err_z < 2.0 else ("WARN" if err_z < 10.0 else "FAIL")
    if err_z < 2.0:
        n_ok += 1
    
    print("%-14.1f | %-10.4f %-10.4f %-8.4f | %-10.6f %-10.6f %-8.4f | %-10.4f %-10.4f %-8.4f | %-8.3f %-8.3f %-8.3f | %s" % (
        freq_mhz, z_o, z_s, err_z,
        r_o, r_s, abs(r_s - r_o)/max(r_o, 1e-10)*100,
        x_o, x_s, abs(x_s - x_o)/max(abs(x_o), 1e-10)*100,
        c_o, c_s, err_c,
        status))

print("-"*120)
print("\n=== SUMMARY ===")
print("  |Z| max error: %.4f%%" % max_err_z)
print("  C   max error: %.4f%%" % max_err_c)
print("  OK points (< 2%%): %d / %d" % (n_ok, len(freqs_hz[::step])))

# Specific checks at common frequencies
print("\n=== KEY FREQUENCY CHECK ===")
for target_mhz in [100, 500, 1000, 2400, 5000]:
    target_hz = target_mhz * 1e6
    # Find nearest official freq
    nearest_off = min(z_off.keys(), key=lambda f: abs(f - target_hz))
    z_o = z_off[nearest_off]
    c_o = c_off.get(nearest_off, 0) * 1e12
    q_o = q_off.get(nearest_off, 0)
    
    # S2P
    S = ts.get_s_matrix_interpolated(target_hz)
    Z_comp = 2 * Z0 * S[0,0] / (1 - S[0,0])
    z_s = abs(Z_comp)
    omega = 2 * np.pi * target_hz
    if Z_comp.imag < -1e-15:
        c_s = -1.0 / (omega * Z_comp.imag) * 1e12
    else:
        c_s = 0
    q_s = abs(Z_comp.imag) / Z_comp.real if Z_comp.real > 1e-15 else 0
    
    print("  %d MHz:" % target_mhz)
    print("    |Z|: off=%.3f, S2P=%.3f (err=%.2f%%)" % (z_o, z_s, abs(z_s-z_o)/z_o*100))
    if c_o > 0 and c_s > 0:
        print("    C:   off=%.3f pF, S2P=%.3f pF (err=%.2f%%)" % (c_o, c_s, abs(c_s-c_o)/c_o*100))
    if q_o > 0 and q_s > 0:
        print("    Q:   off=%.1f, S2P=%.1f (err=%.2f%%)" % (q_o, q_s, abs(q_s-q_o)/q_o*100))

print("\nDone.")
