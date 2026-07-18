"""Parse and compare Optenni Lab results with our engine results."""
import numpy as np
from pathlib import Path

SNP_DIR = Path(__file__).resolve().parents[2] / "data" / "snp"

# Parse Optenni txt
data = []
with open(SNP_DIR / "s6p.txt", "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("%") or line.startswith('"'):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 7:
            continue
        vals = [float(p.replace(",", ".")) for p in parts]
        data.append(vals)

data = np.array(data)
freqs = data[:, 0]  # GHz

band = (freqs >= 2.4) & (freqs <= 2.5)

print("=" * 60)
print("Optenni Lab Results at 2.4-2.5 GHz")
print("=" * 60)
print(f"Data points in band: {np.sum(band)}")

for pi in range(6):
    s11_col = pi + 1
    eff_col = pi + 7
    s11_band = data[band, s11_col]
    eff_band = data[band, eff_col]
    best_s11 = np.min(s11_band)
    avg_s11 = np.mean(s11_band)
    best_eff = np.max(eff_band)
    avg_eff = np.mean(eff_band)
    print(f"Port {pi+1}: S11 best={best_s11:.2f}dB avg={avg_s11:.2f}dB | Eff best={best_eff:.1f}% avg={avg_eff:.1f}%")

# Parse Optenni XML for component details
print("\n" + "=" * 60)
print("Optenni Matching Circuits (from XML)")
print("=" * 60)
import xml.etree.ElementTree as ET
tree = ET.parse(SNP_DIR / "s6p.xml")
root = tree.getroot()

for pd in root.findall("PortData"):
    port_num = pd.get("originalPort")
    circuit = pd.find("Circuit")
    if circuit is None:
        continue
    series = circuit.find("Series_block")
    if series is None:
        continue
    
    components = []
    for child in series:
        tag = child.tag
        if "Inductor" in tag:
            comp = child.find(".//ComponentFromSeries")
            if comp is not None:
                components.append(f"L={comp.get('value')}nH ({comp.get('code')})")
        elif "Capacitor" in tag:
            comp = child.find(".//ComponentFromSeries")
            if comp is not None:
                components.append(f"C={comp.get('value')}pF ({comp.get('code')})")
        
        # Check nested parallelGrounded
        nested = child.find("Series_block")
        if nested is not None:
            for nc in nested:
                if "Inductor" in nc.tag:
                    comp = nc.find(".//ComponentFromSeries")
                    if comp is not None:
                        components.append(f"L={comp.get('value')}nH ({comp.get('code')})")
                elif "Capacitor" in nc.tag:
                    comp = nc.find(".//ComponentFromSeries")
                    if comp is not None:
                        components.append(f"C={comp.get('value')}pF ({comp.get('code')})")
    
    print(f"Port {port_num}: {' + '.join(components)}")

# Isolation
print("\n" + "=" * 60)
print("Optenni Isolation at 2.45 GHz")
print("=" * 60)
idx2450 = np.argmin(np.abs(freqs - 2.45))
print(f"Freq: {freqs[idx2450]:.3f} GHz")
iso_pairs = [
    (13, "S21"), (14, "S31"), (15, "S32"), (16, "S41"), (17, "S42"), (18, "S43"),
    (19, "S51"), (20, "S52"), (21, "S53"), (22, "S54"),
    (23, "S61"), (24, "S62"), (25, "S63"), (26, "S64"), (27, "S65"),
]
for col, name in iso_pairs:
    if col < data.shape[1]:
        val = data[idx2450, col]
        print(f"  {name}: {val:.1f} dB")

# Comparison
print("\n" + "=" * 60)
print("COMPARISON: Optenni vs Our Engine")
print("=" * 60)
print(f"{'Port':<6} {'Optenni S11':<14} {'Our S11':<14} {'Optenni Eff':<14} {'Our Eff':<14}")
print("-" * 62)

our_results = {
    0: {"s11": -16.4, "eff": 97.7},
    1: {"s11": -2.6, "eff": 45.3},
    2: {"s11": -3.1, "eff": 51.2},
    3: {"s11": -16.3, "eff": 97.6},
    4: {"s11": -12.1, "eff": 93.9},
    5: {"s11": -15.7, "eff": 97.3},
}

for pi in range(6):
    s11_band = data[band, pi + 1]
    eff_band = data[band, pi + 7]
    opt_s11 = np.min(s11_band)
    opt_eff = np.max(eff_band)
    our = our_results[pi]
    print(f"P{pi+1:<5} {opt_s11:>8.1f}dB    {our['s11']:>8.1f}dB    {opt_eff:>8.1f}%     {our['eff']:>8.1f}%")

# Optenni XML component summary
print("\n" + "=" * 60)
print("Component Comparison")
print("=" * 60)
optenni_comps = {}
for pd in root.findall("PortData"):
    pn = pd.get("originalPort")
    circuit = pd.find("Circuit")
    if circuit is None:
        continue
    series = circuit.find("Series_block")
    if series is None:
        continue
    comps = []
    for child in series:
        if "Inductor" in child.tag:
            c = child.find(".//ComponentFromSeries")
            if c is not None:
                comps.append(f"L {c.get('value')}nH")
        elif "Capacitor" in child.tag:
            c = child.find(".//ComponentFromSeries")
            if c is not None:
                comps.append(f"C {c.get('value')}pF")
        nested = child.find("Series_block")
        if nested is not None:
            for nc in nested:
                if "Inductor" in nc.tag:
                    c = nc.find(".//ComponentFromSeries")
                    if c is not None:
                        comps.append(f"L {c.get('value')}nH")
                elif "Capacitor" in nc.tag:
                    c = nc.find(".//ComponentFromSeries")
                    if c is not None:
                        comps.append(f"C {c.get('value')}pF")
    optenni_comps[int(pn)] = comps

our_comps = {
    0: ["shunt 4.0nH", "series 33000pF"],
    1: ["shunt 1.5pF", "series 7.0nH"],
    2: ["shunt 1.5pF", "series 7.0nH"],
    3: ["shunt 1.1pF", "series 4.0nH"],
    4: ["shunt 1.0pF", "series 5.0nH"],
    5: ["shunt 2.0pF", "series 8.0nH"],
}

for pi in range(1, 7):
    opt_c = optenni_comps.get(pi, ["N/A"])
    our_c = our_comps.get(pi - 1, ["N/A"])
    print(f"Port {pi}: Optenni={', '.join(opt_c)} | Ours={', '.join(our_c)}")

print("\nKey differences:")
print("- Optenni uses LQP03HQ + GJM03 series (small, high-Q)")
print("- Our engine found larger value components (33000pF, 8nH)")
print("- Optenni's matching is more practical (smaller, realizable)")
print("- Our engine over-matches with high-value parasitics")
