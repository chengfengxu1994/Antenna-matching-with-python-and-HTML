"""Verify DB interpolation accuracy vs S2P at non-grid frequencies."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.murata_db_adapter import load_murata_db
from engine.touchstone import load_touchstone_file

db = load_murata_db(r"E:\RF matching\Murata\murata_components.db")

# Non-grid test frequencies
test_freqs = [2411.5, 2437.3, 2462.8, 5187.5, 5243.2, 5755.5, 897.3, 1842.7]

# Grid test frequencies
grid_freqs = [2400.0, 2450.0, 2500.0, 900.0, 1800.0, 5500.0]

# Get a sample inductor from DB
inds = db.db.get_inductors_near(10.0, tolerance=0.1, primary_only=True)
comp = inds[0]
print("Testing: %s (%.1f nH), DB ID=%d" % (comp.part_number, comp.nominal_value, comp.id))

# Load matching S2P file
import glob
s2p_path = None
for root, dirs, files in os.walk(r"E:\RF matching\Murata"):
    for f in files:
        if comp.part_number.lower() in f.lower() and f.endswith(".s2p"):
            s2p_path = os.path.join(root, f)
            break
    if s2p_path:
        break

if not s2p_path:
    print("S2P file not found!")
    db.close()
    sys.exit(1)

s2p_data = load_touchstone_file(s2p_path)
print("S2P: %s (%d freq points)" % (s2p_path, len(s2p_data.frequencies)))

# Non-grid comparison
print("\n" + "=" * 70)
print("NON-GRID FREQUENCIES (interpolation test)")
print("=" * 70)
print("%12s %20s %20s %12s %8s" % ("Freq MHz", "DB S11", "S2P S11", "Max Diff", "Match?"))
print("-" * 75)

max_diff_all = 0
for f in test_freqs:
    S_db = db.get_s_matrix_at_freq(comp.id, f)
    S_s2p = s2p_data.get_s_matrix_interpolated(f * 1e6)
    
    diff = np.max(np.abs(S_db - S_s2p))
    max_diff_all = max(max_diff_all, diff)
    match = "YES" if diff < 0.001 else ("CLOSE" if diff < 0.01 else "NO")
    
    db_s11 = "%+.6f%+.6fj" % (S_db[0,0].real, S_db[0,0].imag)
    s2p_s11 = "%+.6f%+.6fj" % (S_s2p[0,0].real, S_s2p[0,0].imag)
    
    print("%12.1f %20s %20s %12.6f %8s" % (f, db_s11, s2p_s11, diff, match))

print("\nMax diff (non-grid): %.6f" % max_diff_all)

# Grid comparison (should be exact)
print("\n" + "=" * 70)
print("GRID FREQUENCIES (should be exact)")
print("=" * 70)
for f in grid_freqs:
    S_db = db.get_s_matrix_at_freq(comp.id, f)
    S_s2p = s2p_data.get_s_matrix_interpolated(f * 1e6)
    diff = np.max(np.abs(S_db - S_s2p))
    print("%10.1f MHz: diff=%.8f" % (f, diff))

# Test capacitor too
print("\n" + "=" * 70)
print("CAPACITOR TEST (non-grid)")
print("=" * 70)
caps = db.db.get_capacitors_near(2.2, tolerance=0.1, primary_only=True)
if caps:
    ccomp = caps[0]
    print("Testing: %s (%.1f pF)" % (ccomp.part_number, ccomp.nominal_value))
    
    for root, dirs, files in os.walk(r"E:\RF matching\Murata"):
        for f in files:
            if ccomp.part_number.lower() in f.lower() and f.endswith(".s2p"):
                s2p_path_c = os.path.join(root, f)
                break
    
    if s2p_path_c:
        s2p_c = load_touchstone_file(s2p_path_c)
        for f in [2411.5, 5187.5, 897.3]:
            S_db = db.get_s_matrix_at_freq(ccomp.id, f)
            S_s2p = s2p_c.get_s_matrix_interpolated(f * 1e6)
            diff = np.max(np.abs(S_db - S_s2p))
            match = "YES" if diff < 0.001 else ("CLOSE" if diff < 0.01 else "NO")
            print("  %8.1f MHz: diff=%.6f %s" % (f, diff, match))

# Performance test
print("\n" + "=" * 70)
print("PERFORMANCE: interpolated DB query speed")
print("=" * 70)
import time

N = 1000
comp_id = comp.id

# Non-grid (interpolated)
t0 = time.time()
for _ in range(N):
    S = db.get_s_matrix_at_freq(comp_id, 2411.5)
t_interp = (time.time() - t0) / N * 1e6
print("  Interpolated (non-grid): %.1f us/call" % t_interp)

# Grid (nearest)
t0 = time.time()
for _ in range(N):
    S = db.get_s_matrix_at_freq(comp_id, 2450.0)
t_grid = (time.time() - t0) / N * 1e6
print("  Grid point:              %.1f us/call" % t_grid)

# S2P interpolation
t0 = time.time()
for _ in range(N):
    S = s2p_data.get_s_matrix_interpolated(2411.5e6)
t_s2p = (time.time() - t0) / N * 1e6
print("  S2P interpolation:       %.1f us/call" % t_s2p)

print("\n  DB interpolated vs S2P: %.2fx" % (t_s2p / max(t_interp, 0.001)))

db.close()
print("\nDone!")
