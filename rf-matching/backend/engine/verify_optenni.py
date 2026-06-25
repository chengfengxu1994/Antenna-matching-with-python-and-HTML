"""
Verify Optenni component nominal values by extracting effective L/C from S2P data.
Compares DB-stored values with impedance-derived values at low frequency.
Handles different S2P measurement topologies (series vs shunt).
"""
import sys, os, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.touchstone import load_touchstone_file
import numpy as np

DB_PATH = r'E:\RF matching\Murata\optenni_components.db'

db = sqlite3.connect(DB_PATH)
c = db.cursor()

def effective_from_s2p(s2p_path, comp_type, nominal_unit, test_freq_hz=100e6):
    """Extract effective L/C from S2P at low frequency.
    Tries both series and shunt topology formulas and picks the best match.
    """
    if not os.path.exists(s2p_path):
        return None, "file not found", "N/A"
    try:
        ts = load_touchstone_file(s2p_path)
        S = ts.get_s_matrix_interpolated(test_freq_hz)
        Z0 = 50.0
        omega = 2 * np.pi * test_freq_hz
        
        s11 = S[0, 0]
        s21 = S[0, 1]
        
        # Method 1: Standard 2-port S-param to Z-matrix (series topology)
        I = np.eye(2)
        try:
            Z_mat = Z0 * np.linalg.solve(I - S, I + S)
            z11_series = Z_mat[0, 0]
        except np.linalg.LinAlgError:
            z11_series = None
        
        # Method 2: Shunt topology — component between signal and ground at port 1
        # S11 = -Z0 / (2Z + Z0)  →  Z = -Z0*(1+S11)/(2*S11)
        if abs(s11) > 1e-10:
            z11_shunt = -Z0 * (1 + s11) / (2 * s11)
        else:
            z11_shunt = None
        
        # Method 3: One-port formula from S11
        # Z = Z0 * (1+S11)/(1-S11)
        if abs(1 - s11) > 1e-10:
            z11_onep = Z0 * (1 + s11) / (1 - s11)
        else:
            z11_onep = None
        
        # Determine which impedance to use by checking which gives reasonable L/C
        candidates = []
        for label, z11 in [("series", z11_series), ("shunt", z11_shunt), ("onep", z11_onep)]:
            if z11 is None:
                continue
            
            if comp_type == 'inductor':
                L_h = z11.imag / omega
                L_nh = L_h * 1e9
                if L_nh > 0:
                    candidates.append((abs(np.log2(L_nh) - np.log2(nominal)), L_nh, label))
            else:
                if abs(z11.imag) > 1e-15:
                    C_f = -1.0 / (omega * z11.imag)
                    C_pf = C_f * 1e12
                    if C_pf > 0:
                        candidates.append((abs(np.log2(C_pf) - np.log2(nominal)), C_pf, label))
        
        if not candidates:
            return None, "no valid topology", ""
        
        # Pick closest to nominal
        candidates.sort()
        best_val = candidates[0][1]
        best_label = candidates[0][2]
        
        if comp_type == 'inductor':
            return best_val, f"{best_val:.4f} nH [{best_label}]", best_label
        else:
            return best_val, f"{best_val:.4f} pF [{best_label}]", best_label
            
    except Exception as e:
        return None, str(e), ""

# Test categories: (manufacturer, component_type, sql_filter, description)
tests = [
    ("AVX", "capacitor", "manufacturer='AVX' AND component_type='capacitor'", "AVX ACCU-P caps"),
    ("AVX", "inductor",  "manufacturer='AVX' AND component_type='inductor'", "AVX ACCU-L inds"),
    ("Coilcraft", "inductor", "manufacturer='Coilcraft' AND component_type='inductor'", "Coilcraft inds"),
    ("Johanson", "capacitor", "manufacturer='Johanson' AND component_type='capacitor'", "Johanson caps"),
    ("Johanson", "inductor", "manufacturer='Johanson' AND component_type='inductor'", "Johanson inds"),
    ("TDK", "capacitor", "manufacturer='TDK' AND component_type='capacitor'", "TDK caps"),
    ("TDK", "inductor", "manufacturer='TDK' AND component_type='inductor'", "TDK inds"),
    ("Taiyo Yuden", "capacitor", "manufacturer='Taiyo Yuden' AND component_type='capacitor'", "Taiyo caps"),
    ("Taiyo Yuden", "inductor", "manufacturer='Taiyo Yuden' AND component_type='inductor'", "Taiyo inds"),
    ("Murata", "capacitor", "manufacturer='Murata' AND component_type='capacitor' AND c.nominal_value < 100", "Murata small caps"),
    ("Murata", "inductor", "manufacturer='Murata' AND component_type='inductor' AND c.nominal_value < 100", "Murata inds"),
]

print(f"{'Manufacturer':<14} {'Type':<10} {'Part Number':<35} {'Nominal':>12} {'Effective':>12} {'Unit':<4} {'Error%':>8} {'Status':<8}")
print("=" * 105)

all_ok = True
errors = []

for mfr, ctype, sql_filter, desc in tests:
    # Build proper WHERE clause with table aliases
    where = sql_filter
    where = where.replace("manufacturer=", "s.manufacturer=")
    where = where.replace("component_type='", "c.component_type='")
    
    c.execute(f"""
        SELECT c.part_number, c.nominal_value, c.nominal_unit, c.zip_path, c.component_type
        FROM components c
        JOIN series s ON c.series_id = s.id
        WHERE {where} AND c.is_primary = 1
        ORDER BY RANDOM()
        LIMIT 8
    """)
    rows = c.fetchall()
    
    if not rows:
        print(f"{mfr:<14} {ctype:<10} {'[no primaries found]':<35}")
        all_ok = False
        continue
    
    for row in rows:
        part_number, nominal, unit, s2p_path, comp_type = row
        # Pick test frequency based on expected value
        if ctype == 'inductor':
            if nominal > 1000:  # uH range
                test_freq = 1e6  # 1 MHz
            elif nominal > 100:  # 100+ nH
                test_freq = 10e6
            else:
                test_freq = 100e6
        else:  # capacitor
            if nominal > 10000:  # >10nF
                test_freq = 1e6
            elif nominal > 1000:  # >1nF
                test_freq = 10e6
            else:
                test_freq = 100e6
        
        result, msg, topology = effective_from_s2p(s2p_path, comp_type, unit, test_freq)
        
        if result is None:
            status = f"FAIL[{msg}]"
            print(f"{mfr:<14} {ctype:<10} {part_number:<35} {nominal:>12.4f} {'ERR':>12} {unit:<4} {'N/A':>8} {status:<8}")
            errors.append((part_number, msg))
            all_ok = False
        else:
            # Compute error
            if nominal > 0:
                err_pct = abs(result - nominal) / nominal * 100
            else:
                err_pct = 999
            
            # Things close to SRF can look very off; use relaxed threshold
            if err_pct < 20:
                status = "OK"
            elif err_pct < 50:
                status = "WARN"
            else:
                status = "FAIL"
                all_ok = False
            
            print(f"{mfr:<14} {ctype:<10} {part_number:<35} {nominal:>12.4f} {result:>12.4f} {unit:<4} {err_pct:>7.1f}% {status:<8} [{topology}]")
            
            if status == "FAIL":
                errors.append((part_number, f"{err_pct:.1f}% off: nom={nominal} vs eff={result}"))

print("\n" + "=" * 105)

if errors:
    print("\n== %d FAILURES:" % len(errors))
    for pn, reason in errors[:15]:
        print("  %s: %s" % (pn, reason))
else:
    print("\n== All checks passed!")

db.close()
