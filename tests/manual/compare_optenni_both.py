"""Parse and compare both Optenni XML results (with and without radiation efficiency)."""
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

SNP_DIR = Path(__file__).resolve().parents[2] / "data" / "snp"

def parse_optenni_xml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    result = {
        'nports': int(root.get('nports', 0)),
        'date': root.get('SaveDate', ''),
        'ports': {},
    }
    
    for pd in root.findall('PortData'):
        port_num = int(pd.get('originalPort', 0))
        circuit = pd.find('Circuit')
        if circuit is None:
            continue
        
        # Circuit settings
        settings = {
            'inductor_series': circuit.get('inductorSeries', ''),
            'capacitor_series': circuit.get('capacitorSeries', ''),
            'inductor_directory': circuit.get('inductorDirectory', ''),
            'capacitor_directory': circuit.get('capacitorDirectory', ''),
            'cap_esr': circuit.get('cap_esr', '0'),
            'ind_q': circuit.get('ind_q', '0'),
            'indToler': circuit.get('indToler', '5'),
            'capToler': circuit.get('capToler', '5'),
        }
        
        # Parse components
        components = []
        series = circuit.find('Series_block')
        if series is not None:
            for child in series:
                if 'Inductor' in child.tag:
                    comp = child.find('.//ComponentFromSeries')
                    if comp is not None:
                        components.append({
                            'type': 'L',
                            'value': float(comp.get('value', 0)),
                            'code': comp.get('code', ''),
                            'label': comp.get('../../@componentLabel', child.find('..').get('componentLabel', '') if child.find('..') is not None else ''),
                            'tolerance': comp.get('absTol', ''),
                            'position': 'series' if 'series' in child.get('orientation', '') else 'shunt',
                        })
                elif 'Capacitor' in child.tag:
                    comp = child.find('.//ComponentFromSeries')
                    if comp is not None:
                        components.append({
                            'type': 'C',
                            'value': float(comp.get('value', 0)),
                            'code': comp.get('code', ''),
                            'label': comp.get('../../@componentLabel', ''),
                            'tolerance': comp.get('absTol', ''),
                            'position': 'series' if 'series' in child.get('orientation', '') else 'shunt',
                        })
                
                # Check nested parallelGrounded
                nested = child.find('Series_block')
                if nested is not None:
                    for nc in nested:
                        if 'Inductor' in nc.tag:
                            comp = nc.find('.//ComponentFromSeries')
                            if comp is not None:
                                components.append({
                                    'type': 'L',
                                    'value': float(comp.get('value', 0)),
                                    'code': comp.get('code', ''),
                                    'tolerance': comp.get('absTol', ''),
                                    'position': 'shunt',
                                })
                        elif 'Capacitor' in nc.tag:
                            comp = nc.find('.//ComponentFromSeries')
                            if comp is not None:
                                components.append({
                                    'type': 'C',
                                    'value': float(comp.get('value', 0)),
                                    'code': comp.get('code', ''),
                                    'tolerance': comp.get('absTol', ''),
                                    'position': 'shunt',
                                })
        
        # Parse S-parameters
        sparams = []
        cs = pd.find('CircuitSparameters')
        if cs is not None:
            for line in cs.text.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('!') or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        freq = float(parts[0])
                        s11_re, s11_im = float(parts[1]), float(parts[2])
                        s11_mag = np.sqrt(s11_re**2 + s11_im**2)
                        s11_db = 20 * np.log10(max(s11_mag, 1e-15))
                        sparams.append({
                            'freq_ghz': freq / 1e9,
                            's11_re': s11_re,
                            's11_im': s11_im,
                            's11_mag': s11_mag,
                            's11_db': s11_db,
                        })
                    except (ValueError, IndexError):
                        pass
        
        result['ports'][port_num] = {
            'settings': settings,
            'components': components,
            'sparams': sparams,
        }
    
    return result


# Parse both files
print("=" * 70)
print("OPTENNI LAB RESULTS COMPARISON")
print("=" * 70)

with_rad = parse_optenni_xml(SNP_DIR / "s6p.xml")
without_rad = parse_optenni_xml(SNP_DIR / "s6p without radiation eff.xml")

print(f"\nWith radiation eff: {with_rad['date']}, {with_rad['nports']}-port")
print(f"Without radiation eff: {without_rad['date']}, {without_rad['nports']}-port")

# Compare circuit settings
print("\n" + "=" * 70)
print("CIRCUIT SETTINGS")
print("=" * 70)
for label, data in [("WITH radiation eff", with_rad), ("WITHOUT radiation eff", without_rad)]:
    print(f"\n  {label}:")
    if data['ports']:
        p = list(data['ports'].values())[0]
        s = p['settings']
        print(f"    Inductor: {s['inductor_series']} ({s['inductor_directory']})")
        print(f"    Capacitor: {s['capacitor_series']} ({s['capacitor_directory']})")
        print(f"    Inductor tolerance: {s['indToler']}%")
        print(f"    Capacitor tolerance: {s['capToler']}%")
        print(f"    Inductor Q model: ind_q={s['ind_q']}")
        print(f"    Capacitor ESR model: cap_esr={s['cap_esr']}")

# Compare matching networks per port
print("\n" + "=" * 70)
print("MATCHING NETWORKS COMPARISON")
print("=" * 70)

all_ports = sorted(set(list(with_rad['ports'].keys()) + list(without_rad['ports'].keys())))

print(f"\n{'Port':<6} {'WITH rad eff':<40} {'WITHOUT rad eff':<40} {'Changed?'}")
print("-" * 100)

for p in all_ports:
    w_comps = with_rad['ports'].get(p, {}).get('components', [])
    wo_comps = without_rad['ports'].get(p, {}).get('components', [])
    
    w_str = " + ".join(f"{c['position']}:{c['type']}={c['value']}" for c in w_comps) if w_comps else "N/A"
    wo_str = " + ".join(f"{c['position']}:{c['type']}={c['value']}" for c in wo_comps) if wo_comps else "N/A"
    
    changed = "YES" if w_str != wo_str else "no"
    print(f"P{p:<5} {w_str:<40} {wo_str:<40} {changed}")

# Compare S11 at 2.45 GHz
print("\n" + "=" * 70)
print(f"S11 COMPARISON AT 2.4 GHz BAND")
print("=" * 70)

print(f"\n{'Port':<6} {'WITH rad eff S11':<18} {'WITHOUT rad eff S11':<18} {'Delta':<10}")
print("-" * 55)

for p in all_ports:
    w_data = with_rad['ports'].get(p, {}).get('sparams', [])
    wo_data = without_rad['ports'].get(p, {}).get('sparams', [])
    
    # Find 2.45 GHz point
    w_s11 = None
    wo_s11 = None
    
    for sp in w_data:
        if abs(sp['freq_ghz'] - 2.45) < 0.01:
            w_s11 = sp['s11_db']
            break
    
    for sp in wo_data:
        if abs(sp['freq_ghz'] - 2.45) < 0.01:
            wo_s11 = sp['s11_db']
            break
    
    if w_s11 is not None and wo_s11 is not None:
        delta = wo_s11 - w_s11
        print(f"P{p:<5} {w_s11:>10.2f} dB     {wo_s11:>10.2f} dB     {delta:>+6.2f} dB")
    else:
        print(f"P{p:<5} {'N/A':>10}         {'N/A':>10}         N/A")

# Band efficiency comparison
print("\n" + "=" * 70)
print("BAND EFFICIENCY (2.4-2.5 GHz)")
print("=" * 70)

for label, data in [("WITH radiation eff", with_rad), ("WITHOUT radiation eff", without_rad)]:
    print(f"\n  {label}:")
    for p in all_ports:
        sparams = data['ports'].get(p, {}).get('sparams', [])
        if not sparams:
            continue
        
        band_s11 = [sp for sp in sparams if 2.4 <= sp['freq_ghz'] <= 2.5]
        if not band_s11:
            continue
        
        s11_mags = [sp['s11_mag'] for sp in band_s11]
        effs = [(1 - m**2) * 100 for m in s11_mags]
        
        best_s11 = min(s11_mags)
        avg_eff = np.mean(effs)
        best_eff = max(effs)
        best_rl = -20 * np.log10(max(best_s11, 1e-15))
        
        print(f"    P{p}: Best RL={best_rl:.1f}dB | Avg Eff={avg_eff:.1f}% | Best Eff={best_eff:.1f}%")

# Key insight
print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("""
The "without radiation efficiency" version uses the SAME matching topology
(series L/C + shunt L/C to ground) but with DIFFERENT component values.

This is because:
- With radiation eff: Optenni optimizes TOTAL efficiency = radiation_eff × mismatch_eff
  → Prefers matching that preserves radiation pattern
- Without radiation eff: Optenni optimizes MISMATCH efficiency only = 1 - |S11|^2
  → Pure impedance matching, ignores radiation

For our engine: we only compute mismatch efficiency (1 - |S11|^2),
which matches the "without radiation eff" case. This is correct behavior
when no radiation efficiency data is provided.

The component differences confirm that radiation efficiency DOES affect
the optimal matching network selection.
""")
