"""
Benchmark: SQLite DB adapter vs S2P file loading.
Compares load time, query time, and S-parameter accuracy.
"""
import time, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))
from project_paths import MURATA_DIR

DB_PATH = MURATA_DIR / "murata_components.db"

# ═══════════════════════════════════════════
# 1. Load DB adapter
# ═══════════════════════════════════════════
print("=" * 70)
print("BENCHMARK: SQLite DB vs S2P File Loading")
print("=" * 70)

print("\n--- SQLite DB Adapter ---")
t0 = time.time()
from engine.murata_db_adapter import load_murata_db
db_lib = load_murata_db(DB_PATH)
t_db_load = time.time() - t0
print(f"  Load time: {t_db_load*1000:.1f} ms")
print(db_lib.summary())

# ═══════════════════════════════════════════
# 2. Load S2P file library
# ═══════════════════════════════════════════
print("\n--- S2P File Library (scan_murata_directory) ---")
t0 = time.time()
from engine.component_lib import scan_murata_directory
s2p_lib = scan_murata_directory(MURATA_DIR)
t_s2p_load = time.time() - t0
print(f"  Load time: {t_s2p_load*1000:.1f} ms ({t_s2p_load:.2f}s)")
print(f"  Inductors: {len(s2p_lib.inductors)}")
print(f"  Capacitors: {len(s2p_lib.capacitors)}")
print(f"  Unique L values: {len(s2p_lib.get_unique_inductor_values())}")
print(f"  Unique C values: {len(s2p_lib.get_unique_capacitor_values())}")

print(f"\n  Speedup: {t_s2p_load / max(t_db_load, 0.001):.0f}x faster with DB")

# ═══════════════════════════════════════════
# 3. Query benchmark: find components near value
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("QUERY BENCHMARK: Find components near target values")
print("=" * 70)

test_values_L = [1.0, 2.2, 4.7, 10.0, 22.0, 47.0]
test_values_C = [0.5, 1.0, 2.2, 4.7, 10.0, 22.0, 47.0, 100.0]

# DB queries
print("\n--- DB queries ---")
t0 = time.time()
for v in test_values_L:
    results = db_lib.get_inductors_near(v, tolerance=0.3)
for v in test_values_C:
    results = db_lib.get_capacitors_near(v, tolerance=0.3)
t_db_query = time.time() - t0
print(f"  {len(test_values_L) + len(test_values_C)} queries: {t_db_query*1000:.1f} ms")

# S2P queries
print("\n--- S2P queries ---")
t0 = time.time()
for v in test_values_L:
    results = s2p_lib.get_inductors_near(v, tolerance=0.3)
for v in test_values_C:
    results = s2p_lib.get_capacitors_near(v, tolerance=0.3)
t_s2p_query = time.time() - t0
print(f"  {len(test_values_L) + len(test_values_C)} queries: {t_s2p_query*1000:.1f} ms")
print(f"  Speedup: {t_s2p_query / max(t_db_query, 0.001):.0f}x faster with DB")

# ═══════════════════════════════════════════
# 4. S-parameter access benchmark
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("S-PARAMETER ACCESS BENCHMARK")
print("=" * 70)

test_freqs_mhz = [700, 900, 1800, 2450, 3500, 5500]

# DB: get S-matrix at frequency (pre-computed)
print("\n--- DB: pre-computed S-matrix ---")
db_times = []
for freq in test_freqs_mhz:
    t0 = time.time()
    all_data = db_lib.get_all_components_at_freq(freq)
    dt = time.time() - t0
    db_times.append(dt)
    n_ind = len(all_data['inductors'])
    n_cap = len(all_data['capacitors'])
    print(f"  {freq:>5} MHz: {n_ind}L + {n_cap}C in {dt*1000:.1f} ms")

# S2P: load from ZIP + parse + interpolate
print("\n--- S2P: load from ZIP ---")
# Get a few sample components from the library
sample_L = s2p_lib.inductors[:10]
sample_C = s2p_lib.capacitors[:10]

s2p_times = []
for freq in test_freqs_mhz:
    t0 = time.time()
    for comp in sample_L:
        S = comp.get_s_matrix_at_freq(freq * 1e6)
    for comp in sample_C:
        S = comp.get_s_matrix_at_freq(freq * 1e6)
    dt = time.time() - t0
    s2p_times.append(dt)
    print(f"  {freq:>5} MHz: 20 components in {dt*1000:.1f} ms")

# Extrapolate to full library
avg_s2p_per_comp = np.mean(s2p_times) / 20
total_comps = len(s2p_lib.inductors) + len(s2p_lib.capacitors)
extrapolated = avg_s2p_per_comp * total_comps
print(f"\n  Extrapolated for full library ({total_comps} components): {extrapolated:.1f}s")
print(f"  DB bulk query: {np.mean(db_times)*1000:.1f} ms")
print(f"  Speedup: {extrapolated / max(np.mean(db_times), 0.001):.0f}x faster with DB")

# ═══════════════════════════════════════════
# 5. Accuracy comparison
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("ACCURACY: DB vs S2P S-parameter comparison")
print("=" * 70)

# Find common components
print("\nComparing S-parameters at 2450 MHz:")
mismatches = 0
comparisons = 0

for comp in s2p_lib.inductors[:20]:
    # Get from S2P
    S_s2p = comp.get_s_matrix_at_freq(2450e6)
    
    # Find in DB by part number
    db_results = db_lib.db.get_inductors_near(comp.nominal_value, tolerance=0.01)
    for db_comp in db_results:
        if db_comp.part_number == comp.part_number:
            S_db = db_lib.get_s_matrix_at_freq(db_comp.id, 2450.0)
            if S_db is not None:
                diff = np.max(np.abs(S_s2p - S_db))
                comparisons += 1
                if diff > 0.001:
                    mismatches += 1
                    print(f"  MISMATCH: {comp.part_number} max_diff={diff:.6f}")
            break

for comp in s2p_lib.capacitors[:20]:
    S_s2p = comp.get_s_matrix_at_freq(2450e6)
    db_results = db_lib.db.get_capacitors_near(comp.nominal_value, tolerance=0.01)
    for db_comp in db_results:
        if db_comp.part_number == comp.part_number:
            S_db = db_lib.get_s_matrix_at_freq(db_comp.id, 2450.0)
            if S_db is not None:
                diff = np.max(np.abs(S_s2p - S_db))
                comparisons += 1
                if diff > 0.001:
                    mismatches += 1
                    print(f"  MISMATCH: {comp.part_number} max_diff={diff:.6f}")
            break

print(f"\n  Compared: {comparisons} components")
print(f"  Mismatches (>0.001): {mismatches}")
print(f"  Accuracy: {(1 - mismatches/max(comparisons,1))*100:.1f}%")

# ═══════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  Load time:
    DB:    {t_db_load*1000:>8.1f} ms
    S2P:   {t_s2p_load*1000:>8.1f} ms  ({t_s2p_load/max(t_db_load,0.001):.0f}x slower)

  Component queries ({len(test_values_L) + len(test_values_C)} queries):
    DB:    {t_db_query*1000:>8.1f} ms
    S2P:   {t_s2p_query*1000:>8.1f} ms  ({t_s2p_query/max(t_db_query,0.001):.0f}x slower)

  Full library S-param access (per frequency):
    DB:    {np.mean(db_times)*1000:>8.1f} ms (all components at once)
    S2P:   {extrapolated*1000:>8.1f} ms (estimated, one by one)

  Accuracy: {(1 - mismatches/max(comparisons,1))*100:.1f}% match between DB and S2P

  RECOMMENDATION: Use DB adapter for production.
  - {t_s2p_load/max(t_db_load,0.001):.0f}x faster load
  - {extrapolated/max(np.mean(db_times),0.001):.0f}x faster per-frequency bulk query
  - Pre-computed derived parameters (Z, Q, SRF) available
""")

db_lib.close()
