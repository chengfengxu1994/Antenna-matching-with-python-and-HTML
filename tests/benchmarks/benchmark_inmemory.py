"""
Benchmark: In-memory S2P vs DB — pure computation speed.
Both fully loaded into memory before timing starts.
"""
import time, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))
from project_paths import MURATA_DIR, SNP_DIR

DB_PATH = MURATA_DIR / "murata_components.db"

from engine.touchstone import load_touchstone_file, TouchstoneData
from engine.component_lib import parse_murata_part
from engine.network import _embed_series_on_port, _embed_shunt_to_ground, s_to_y, y_to_s, terminate_ports

print("=" * 70)
print("IN-MEMORY BENCHMARK: S2P vs DB (pre-loaded)")
print("=" * 70)

# ═══════════════════════════════════════════
# Pre-load S2P files into memory
# ═══════════════════════════════════════════
print("\n[1. Loading S2P files into memory...]")

def scan_and_cache_s2p(murata_dir, max_per_type=500):
    """Scan directories, load S2P data, cache S-matrices at 2450 MHz."""
    L_parts = {}  # value -> (part_number, TouchstoneData)
    C_parts = {}
    
    t0 = time.time()
    count = 0
    for root, dirs, files in os.walk(murata_dir):
        for f in files:
            if not f.lower().endswith('.s2p'):
                continue
            filepath = os.path.join(root, f)
            part_number = os.path.splitext(f)[0]
            category = 'inductor' if 'inductor' in root.lower() else 'capacitor'
            _, nominal_value, _ = parse_murata_part(part_number)
            
            if nominal_value <= 0.01:
                continue
            
            # Skip if we already have this value (deduplicate)
            target_dict = L_parts if category == 'inductor' else C_parts
            if nominal_value in target_dict:
                continue
            if len(target_dict) >= max_per_type:
                continue
            
            try:
                data = load_touchstone_file(filepath)
                target_dict[nominal_value] = (part_number, data)
                count += 1
            except:
                pass
    
    t_load = time.time() - t0
    return L_parts, C_parts, t_load, count

t0 = time.time()
L_parts, C_parts, t_scan, n_files = scan_and_cache_s2p(MURATA_DIR, max_per_type=200)
t_s2p_total = time.time() - t0
print(f"  Scanned: {n_files} files in {t_scan:.2f}s")
print(f"  Unique L values: {len(L_parts)}")
print(f"  Unique C values: {len(C_parts)}")

# Pre-compute S-matrices at 2450 MHz
print("  Pre-computing S-matrices at 2450 MHz...")
t0 = time.time()
s2p_L_s = {}
for v, (pn, data) in L_parts.items():
    s2p_L_s[v] = data.get_s_matrix_interpolated(2450e6)
s2p_C_s = {}
for v, (pn, data) in C_parts.items():
    s2p_C_s[v] = data.get_s_matrix_interpolated(2450e6)
t_s2p_precalc = time.time() - t0
print(f"  Pre-calc: {t_s2p_precalc:.3f}s")

# ═══════════════════════════════════════════
# Pre-load DB data into memory
# ═══════════════════════════════════════════
print("\n[2. Loading DB into memory...]")

from engine.murata_db_adapter import load_murata_db

t0 = time.time()
db_lib = load_murata_db(DB_PATH)
t_db_load = time.time() - t0
print(f"  DB opened: {t_db_load*1000:.1f}ms")

# Bulk query: all primary components at 2450 MHz
t0 = time.time()
db_all_L = db_lib.get_all_inductors_at_freq(2450.0)
db_all_C = db_lib.get_all_capacitors_at_freq(2450.0)
t_db_bulk = time.time() - t0
print(f"  Bulk query: {len(db_all_L)}L + {len(db_all_C)}C in {t_db_bulk*1000:.1f}ms")

# Build in-memory dicts
db_L_s = {}
for d in db_all_L:
    v = d['nominal_value']
    if v not in db_L_s and v > 0.01:
        S = db_lib.get_s_matrix_at_freq(d['id'], 2450.0)
        if S is not None:
            db_L_s[v] = S
db_C_s = {}
for d in db_all_C:
    v = d['nominal_value']
    if v not in db_C_s and v > 0.01:
        S = db_lib.get_s_matrix_at_freq(d['id'], 2450.0)
        if S is not None:
            db_C_s[v] = S
print(f"  In-memory: {len(db_L_s)}L + {len(db_C_s)}C S-matrices")

# ═══════════════════════════════════════════
# Benchmark 1: Single S-matrix access
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("BENCH 1: Single S-matrix access (1000 iterations)")
print("=" * 70)

N = 1000

# S2P: interpolation from loaded TouchstoneData
s2p_sample_data = list(L_parts.values())[5][1]
t0 = time.time()
for _ in range(N):
    S = s2p_sample_data.get_s_matrix_interpolated(2450e6)
t_s2p_access = (time.time() - t0) / N * 1e6
print(f"  S2P (interpolation):  {t_s2p_access:.1f} us/call")

# DB: SQLite query
db_sample_id = db_all_L[5]['id'] if db_all_L else None
t0 = time.time()
for _ in range(N):
    S = db_lib.get_s_matrix_at_freq(db_sample_id, 2450.0)
t_db_access = (time.time() - t0) / N * 1e6
print(f"  DB  (SQLite query):   {t_db_access:.1f} us/call")

# In-memory: dict lookup
S_cached = list(s2p_L_s.values())[5]
t0 = time.time()
for _ in range(N):
    S = S_cached.copy()
t_mem_access = (time.time() - t0) / N * 1e6
print(f"  Mem (numpy copy):     {t_mem_access:.1f} us/call")

# ═══════════════════════════════════════════
# Benchmark 2: Optimizer inner loop
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("BENCH 2: Optimizer inner loop (4 patterns x N_L x N_C)")
print("=" * 70)

# Use same value count for fair comparison
n_L = min(len(s2p_L_s), len(db_L_s), 20)
n_C = min(len(s2p_C_s), len(db_C_s), 20)

s2p_L_vals = sorted(s2p_L_s.keys())[:n_L]
s2p_C_vals = sorted([v for v in s2p_C_s.keys() if v > 0.1])[:n_C]
db_L_vals = sorted(db_L_s.keys())[:n_L]
db_C_vals = sorted([v for v in db_C_s.keys() if v > 0.1])[:n_C]

total_evals = 4 * len(s2p_L_vals) * len(s2p_C_vals)
print(f"  Values: {n_L}L x {n_C}C x 4 patterns = {total_evals} evaluations")

# Build test S-matrix (6-port)
data_6p = load_touchstone_file(SNP_DIR / "SAR Head Hand and Phone.s6p")
S_test = data_6p.get_s_matrix_interpolated(2.45e9)

def inner_loop(S_base, L_vals, C_vals, L_s, C_s):
    """Run optimizer inner loop: find best L/C combination."""
    best_mag = 1.0
    best = None
    port = 0
    
    patterns = [
        ('series', 'L', 'shunt', 'C'),
        ('series', 'C', 'shunt', 'L'),
        ('shunt', 'L', 'series', 'C'),
        ('shunt', 'C', 'series', 'L'),
    ]
    
    for conn1, t1, conn2, t2 in patterns:
        v1s = L_vals if t1 == 'L' else C_vals
        v2s = L_vals if t2 == 'L' else C_vals
        s1d = L_s if t1 == 'L' else C_s
        s2d = L_s if t2 == 'L' else C_s
        
        for v1 in v1s:
            s1 = s1d.get(v1)
            if s1 is None: continue
            for v2 in v2s:
                s2 = s2d.get(v2)
                if s2 is None: continue
                
                S = S_base.copy()
                # First component
                if conn1 == 'series':
                    S = _embed_series_on_port(S, s1, port)
                else:
                    Y = s_to_y(S)
                    g = s1[0,0] + s1[0,1]*(-1)*s1[1,0]/(1+s1[1,1])
                    Y[port,port] += (1.0/50.0)*(1-g)/(1+g)
                    S = y_to_s(Y)
                # Second component
                if conn2 == 'series':
                    S = _embed_series_on_port(S, s2, port)
                else:
                    Y = s_to_y(S)
                    g = s2[0,0] + s2[0,1]*(-1)*s2[1,0]/(1+s2[1,1])
                    Y[port,port] += (1.0/50.0)*(1-g)/(1+g)
                    S = y_to_s(Y)
                
                mag = abs(S[0,0])
                if mag < best_mag:
                    best_mag = mag
                    best = (v1, v2, conn1, t1, conn2, t2)
    
    return best_mag, best

# Reduce to 1-port for fair comparison (terminate others)
S_1port = terminate_ports(S_test.copy(), {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})

print(f"\n  Terminated S[0,0] = {S_1port[0,0]:.4f} |S11|={abs(S_1port[0,0]):.4f}")

# S2P inner loop
print("\n  S2P inner loop...")
t0 = time.time()
mag_s2p, net_s2p = inner_loop(S_1port, s2p_L_vals, s2p_C_vals, s2p_L_s, s2p_C_s)
t_s2p_loop = time.time() - t0
print(f"  S2P: {t_s2p_loop*1000:.1f}ms | best |S11|={mag_s2p:.4f} | {net_s2p}")

# DB inner loop
print("  DB inner loop...")
t0 = time.time()
mag_db, net_db = inner_loop(S_1port, db_L_vals, db_C_vals, db_L_s, db_C_s)
t_db_loop = time.time() - t0
print(f"  DB:  {t_db_loop*1000:.1f}ms | best |S11|={mag_db:.4f} | {net_db}")

# ═══════════════════════════════════════════
# Benchmark 3: 6-port joint optimization
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("BENCH 3: Full 6-port joint optimization")
print("=" * 70)

def joint_opt(S_full, L_vals, C_vals, L_s, C_s, label):
    N = S_full.shape[0]
    results = {}
    t_total = 0
    
    for pi in range(N):
        term = {p: 0.0 for p in range(N) if p != pi}
        S_term = terminate_ports(S_full.copy(), term)
        
        t0 = time.time()
        mag, net = inner_loop(S_term, L_vals, C_vals, L_s, C_s)
        dt = time.time() - t0
        t_total += dt
        
        rl = -20 * np.log10(max(mag, 1e-15))
        results[pi] = {'mag': mag, 'rl': rl, 'time': dt}
        print(f"    P{pi+1}: {dt*1000:.1f}ms RL={rl:.1f}dB")
    
    return results, t_total

print("\n  S2P:")
s2p_res, s2p_joint = joint_opt(S_test, s2p_L_vals, s2p_C_vals, s2p_L_s, s2p_C_s, "S2P")

print(f"\n  DB:")
db_res, db_joint = joint_opt(S_test, db_L_vals, db_C_vals, db_L_s, db_C_s, "DB")

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY (all data pre-loaded in memory)")
print("=" * 70)
print(f"""
  Pre-load time:
    S2P (scan + cache):  {t_s2p_total:.2f}s
    DB  (open + bulk):   {(t_db_load + t_db_bulk):.2f}s

  Single S-matrix access:
    S2P interpolation:   {t_s2p_access:.1f} us
    DB  SQLite query:    {t_db_access:.1f} us
    Memory numpy copy:   {t_mem_access:.1f} us

  Inner loop ({total_evals} evaluations):
    S2P: {t_s2p_loop*1000:.1f}ms
    DB:  {t_db_loop*1000:.1f}ms
    Ratio: {t_s2p_loop/max(t_db_loop,0.001):.2f}x

  6-port joint optimization:
    S2P: {s2p_joint*1000:.1f}ms
    DB:  {db_joint*1000:.1f}ms
    Ratio: {s2p_joint/max(db_joint,0.001):.2f}x
""")

if s2p_joint < db_joint:
    print(f"  >>> S2P is {db_joint/s2p_joint:.2f}x FASTER (pre-loaded)")
else:
    print(f"  >>> DB is {s2p_joint/db_joint:.2f}x FASTER (pre-loaded)")

print("""
  ANALYSIS:
  - S2P interpolation: numpy array operations, no DB overhead
  - DB pre-computed: dict lookup, no interpolation needed
  - Matrix operations (_embed_series, s_to_y, y_to_s) dominate both
  - The S-parameter access method is negligible vs matrix math
""")

db_lib.close()
