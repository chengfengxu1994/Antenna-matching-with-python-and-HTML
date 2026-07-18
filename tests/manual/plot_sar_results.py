"""
Generate S11 and radiation efficiency comparison plots for 6-port SAR matching.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.touchstone import load_touchstone_file
from engine.component_lib import scan_murata_directory
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from engine.topology import get_standard_topologies
from engine.network import _embed_series_on_port, _embed_shunt_to_ground, terminate_ports
from project_paths import ARTIFACTS_DIR, MURATA_DIR, SNP_DIR

OUT = ARTIFACTS_DIR / "output"
os.makedirs(OUT, exist_ok=True)

data = load_touchstone_file(SNP_DIR / "SAR Head Hand and Phone.s6p")
lib = scan_murata_directory(MURATA_DIR)
N = data.num_ports
freqs = np.array(data.frequencies)
topos = [t for t in get_standard_topologies() if t.num_components <= 2]

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3})

# ── Optimize ──
print("Optimizing 6 ports...")
solutions = {}
for pi in range(N):
    ps = {p: PortState.LOAD for p in range(N)}
    ps[pi] = PortState.COMPONENT
    cfg = OptimizerConfig(target_frequency_hz=2.45e9, max_components=2, beam_width=10,
                          timeout_seconds=30, bands_mhz=[[2400, 2500]], num_band_points=5)
    opt = MatchingOptimizer(data, lib, cfg)
    sols = opt.optimize_full(port_states=ps, topologies=topos, input_port=pi)
    if sols:
        solutions[pi] = sols[0]
        c = ', '.join(f'{x.connection_type}:{x.component.nominal_value}{x.component.nominal_unit}' for x in sols[0].component_choices)
        print(f"  P{pi+1}: RL={sols[0].s11_db:.1f}dB | {c}")

# ── Sweep (subsampled) ──
print("\nSweeping frequencies...")
sub = slice(None, None, 5)  # every 5th point
fsub = freqs[sub] / 1e9
s11_bef, s11_aft, eff_bef, eff_aft = {}, {}, {}, {}

for pi in range(N):
    raw_m, match_m = [], []
    sol = solutions.get(pi)
    for f_hz in freqs[sub]:
        Sfull = data.get_s_matrix_interpolated(f_hz)
        raw_m.append(abs(Sfull[pi, pi]))
        if sol:
            term = {p: 0.0 for p in range(N) if p != pi}
            St = terminate_ports(Sfull.copy(), term) if term else Sfull.copy()
            S = St.copy()
            try:
                for ch in sol.component_choices:
                    cs = ch.component.get_s_matrix_at_freq(f_hz)
                    if ch.connection_type == 'series': S = _embed_series_on_port(S, cs, ch.port)
                    else: S = _embed_shunt_to_ground(S, cs, ch.port)
                match_m.append(abs(S[0, 0]))
            except: match_m.append(abs(St[0, 0]))
        else: match_m.append(abs(Sfull[pi, pi]))
    raw_m, match_m = np.array(raw_m), np.array(match_m)
    s11_bef[pi] = 20*np.log10(np.clip(raw_m, 1e-15, 1))
    s11_aft[pi] = 20*np.log10(np.clip(match_m, 1e-15, 1))
    eff_bef[pi] = (1 - raw_m**2) * 100
    eff_aft[pi] = (1 - match_m**2) * 100
    print(f"  P{pi+1} done")

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# ═══ PLOT 1: S11 ═══
print("\nPlotting S11...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('S11 Before vs After Matching (6-Port SAR)', fontsize=14, fontweight='bold')
for pi in range(N):
    ax = axes[pi//3][pi%3]
    ax.plot(fsub, s11_bef[pi], '--', color='#ccc', lw=1, label='Before')
    ax.plot(fsub, s11_aft[pi], color=colors[pi], lw=2, label='After')
    ax.axvspan(2.4, 2.5, alpha=0.08, color='blue')
    ax.axhline(-10, color='green', ls=':', alpha=0.5, lw=0.8)
    band = (fsub >= 2.4) & (fsub <= 2.5)
    if np.any(band):
        best = np.min(s11_aft[pi][band])
        ax.annotate(f'{best:.1f}dB', xy=(2.45, best), fontsize=9, fontweight='bold',
                    color=colors[pi], ha='center', va='bottom')
    ax.set_title(f'Port {pi+1}', fontweight='bold')
    ax.set_xlabel('GHz'); ax.set_ylabel('dB'); ax.set_ylim([-40, 0])
    ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(rect=[0,0,1,0.95])
p1 = os.path.join(OUT, "s11_before_after.png")
fig.savefig(p1, bbox_inches='tight'); plt.close(); print(f"  {p1}")

# ═══ PLOT 2: Efficiency ═══
print("Plotting efficiency...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Mismatch Efficiency Before vs After', fontsize=14, fontweight='bold')
for pi in range(N):
    ax = axes[pi//3][pi%3]
    ax.plot(fsub, eff_bef[pi], '--', color='#ccc', lw=1, label='Before')
    ax.plot(fsub, eff_aft[pi], color=colors[pi], lw=2, label='After')
    ax.axvspan(2.4, 2.5, alpha=0.08, color='blue')
    ax.axhline(90, color='green', ls=':', alpha=0.5, lw=0.8)
    band = (fsub >= 2.4) & (fsub <= 2.5)
    if np.any(band):
        avg = np.mean(eff_aft[pi][band])
        ax.annotate(f'Avg {avg:.1f}%', xy=(2.45, avg), fontsize=9, fontweight='bold',
                    color=colors[pi], ha='center', va='bottom')
    ax.set_title(f'Port {pi+1}', fontweight='bold')
    ax.set_xlabel('GHz'); ax.set_ylabel('%'); ax.set_ylim([0, 105])
    ax.legend(fontsize=8, loc='lower left')
plt.tight_layout(rect=[0,0,1,0.95])
p2 = os.path.join(OUT, "efficiency_before_after.png")
fig.savefig(p2, bbox_inches='tight'); plt.close(); print(f"  {p2}")

# ═══ PLOT 3: Smith Chart ═══
print("Plotting Smith chart...")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title('Smith Chart — Before (gray) vs After (colored)', fontsize=14, fontweight='bold')
theta = np.linspace(0, 2*np.pi, 361)
ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1, alpha=0.3)
ax.plot([-1,1],[0,0],'k-', lw=0.5, alpha=0.2); ax.plot([0,0],[-1,1],'k-', lw=0.5, alpha=0.2)
ax.set_aspect('equal'); ax.set_xlim([-1.15,1.15]); ax.set_ylim([-1.15,1.15])
ax.set_xlabel('Re(Gamma)'); ax.set_ylabel('Im(Gamma)')

for pi in range(N):
    gb = [data.get_s_matrix_interpolated(f)[pi, pi] for f in freqs[sub]]
    gb = np.array(gb)
    ax.plot(gb.real, gb.imag, color='#ccc', lw=0.8, alpha=0.5)
    sol = solutions.get(pi)
    if sol:
        ga = []
        for f_hz in freqs[sub]:
            Sfull = data.get_s_matrix_interpolated(f_hz)
            term = {p: 0.0 for p in range(N) if p != pi}
            St = terminate_ports(Sfull.copy(), term) if term else Sfull.copy()
            S = St.copy()
            try:
                for ch in sol.component_choices:
                    cs = ch.component.get_s_matrix_at_freq(f_hz)
                    if ch.connection_type == 'series': S = _embed_series_on_port(S, cs, ch.port)
                    else: S = _embed_shunt_to_ground(S, cs, ch.port)
                ga.append(S[0, 0])
            except: ga.append(St[0, 0])
        ga = np.array(ga)
        ax.plot(ga.real, ga.imag, color=colors[pi], lw=2, label=f'P{pi+1}')
        idx2450 = np.argmin(np.abs(freqs[sub] - 2.45e9))
        ax.plot(ga[idx2450].real, ga[idx2450].imag, 'o', color=colors[pi], ms=8)
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
p3 = os.path.join(OUT, "smith_chart.png")
fig.savefig(p3, bbox_inches='tight'); plt.close(); print(f"  {p3}")

# ═══ PLOT 4: Summary bars ═══
print("Plotting summary...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('6-Port SAR Summary at 2.45 GHz', fontsize=14, fontweight='bold')
ports = [f'P{i+1}' for i in range(N)]
x = np.arange(N); w = 0.35
idx2450 = np.argmin(np.abs(freqs - 2.45e9))

raw_db = [s11_bef[pi][np.argmin(np.abs(fsub - 2.45))] for pi in range(N)]
mat_db = [s11_aft[pi][np.argmin(np.abs(fsub - 2.45))] for pi in range(N)]
ax1.bar(x-w/2, raw_db, w, label='Before', color='#ccc', edgecolor='#999')
ax1.bar(x+w/2, mat_db, w, label='After', color=colors[:N], edgecolor='#333')
ax1.set_xticks(x); ax1.set_xticklabels(ports); ax1.set_ylabel('S11 (dB)')
ax1.set_title('Return Loss'); ax1.legend(); ax1.axhline(-10, color='green', ls=':', alpha=0.5)
for i, m in enumerate(mat_db): ax1.text(i+w/2, m-1.5, f'{m:.1f}', ha='center', va='top', fontsize=8, fontweight='bold')

raw_e = [eff_bef[pi][np.argmin(np.abs(fsub - 2.45))] for pi in range(N)]
mat_e = [eff_aft[pi][np.argmin(np.abs(fsub - 2.45))] for pi in range(N)]
ax2.bar(x-w/2, raw_e, w, label='Before', color='#ccc', edgecolor='#999')
ax2.bar(x+w/2, mat_e, w, label='After', color=colors[:N], edgecolor='#333')
ax2.set_xticks(x); ax2.set_xticklabels(ports); ax2.set_ylabel('%')
ax2.set_title('Mismatch Efficiency'); ax2.legend(); ax2.set_ylim([0, 105])
for i, m in enumerate(mat_e): ax2.text(i+w/2, m+2, f'{m:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.93])
p4 = os.path.join(OUT, "summary_bar.png")
fig.savefig(p4, bbox_inches='tight'); plt.close(); print(f"  {p4}")

print(f"\nDone! Files in {OUT}")
