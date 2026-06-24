"""Generate comparison plots: Independent vs Joint optimization."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.touchstone import load_touchstone_file
from engine.component_lib import scan_murata_directory
from engine.network import terminate_ports, _embed_series_on_port, _embed_shunt_to_ground, s_to_y, y_to_s
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from engine.topology import get_standard_topologies
from engine.joint_optimizer import JointMultiPortOptimizer, embed_all_matching, format_matching_result

OUT = r"E:\RF matching\output"
os.makedirs(OUT, exist_ok=True)

data = load_touchstone_file(r"E:\RF matching\snp\SAR Head Hand and Phone.s6p")
lib = scan_murata_directory(r"E:\RF matching\Murata")
N = data.num_ports
FREQ = 2.45e9

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3})
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# ═══ Independent optimization ═══
print("Independent optimization...")
topos = [t for t in get_standard_topologies() if t.num_components <= 2]
indep_solutions = {}
for pi in range(N):
    ps = {p: PortState.LOAD for p in range(N)}
    ps[pi] = PortState.COMPONENT
    cfg = OptimizerConfig(target_frequency_hz=FREQ, max_components=2, beam_width=10, timeout_seconds=15)
    opt = MatchingOptimizer(data, lib, cfg)
    sols = opt.optimize_full(port_states=ps, topologies=topos, input_port=pi)
    if sols:
        indep_solutions[pi] = sols[0]

# ═══ Joint optimization ═══
print("Joint optimization...")
joint_opt = JointMultiPortOptimizer(data, lib, freq_hz=FREQ, max_components=2, series_l='LQP03HQ', series_c='GJM03')
matching, joint_results, history = joint_opt.optimize_joint(max_iterations=5, target_rl=10.0)

# ═══ Compute S11 curves ═══
print("Computing frequency sweeps...")
freqs = np.array(data.frequencies)
sub = slice(None, None, 5)
fsub = freqs[sub] / 1e9

# Raw, independent, joint S11 for each port
raw_s11 = {}
indep_s11 = {}
joint_s11 = {}

for pi in range(N):
    raw_m, indep_m, joint_m = [], [], []

    for f_hz in freqs[sub]:
        S_full = data.get_s_matrix_interpolated(f_hz)
        raw_m.append(abs(S_full[pi, pi]))

        # Independent: terminate others with 50 ohm, apply matching
        sol = indep_solutions.get(pi)
        if sol:
            term = {p: 0.0 for p in range(N) if p != pi}
            St = terminate_ports(S_full.copy(), term) if term else S_full.copy()
            S = St.copy()
            try:
                for ch in sol.component_choices:
                    cs = ch.component.get_s_matrix_at_freq(f_hz)
                    if ch.connection_type == 'series':
                        S = _embed_series_on_port(S, cs, ch.port)
                    else:
                        S = _embed_shunt_to_ground(S, cs, ch.port)
                indep_m.append(abs(S[0, 0]))
            except:
                indep_m.append(abs(St[0, 0]))
        else:
            indep_m.append(abs(S_full[pi, pi]))

        # Joint: apply ALL matching networks
        try:
            S_j = embed_all_matching(S_full, joint_opt.current_matching_s, f_hz)
            joint_m.append(abs(S_j[pi, pi]))
        except:
            joint_m.append(abs(S_full[pi, pi]))

    raw_s11[pi] = 20 * np.log10(np.clip(np.array(raw_m), 1e-15, 1))
    indep_s11[pi] = 20 * np.log10(np.clip(np.array(indep_m), 1e-15, 1))
    joint_s11[pi] = 20 * np.log10(np.clip(np.array(joint_m), 1e-15, 1))
    print(f"  P{pi+1} done")

# ═══ PLOT 1: S11 comparison (3 lines per port) ═══
print("\nPlotting S11 comparison...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('S11: Raw vs Independent vs Joint Optimization', fontsize=14, fontweight='bold')
for pi in range(N):
    ax = axes[pi//3][pi%3]
    ax.plot(fsub, raw_s11[pi], '--', color='#ccc', lw=1, label='Raw')
    ax.plot(fsub, indep_s11[pi], '-', color=colors[pi], lw=1.5, alpha=0.6, label='Independent')
    ax.plot(fsub, joint_s11[pi], '-', color=colors[pi], lw=2.5, label='Joint')
    ax.axvspan(2.4, 2.5, alpha=0.08, color='blue')
    ax.axhline(-10, color='green', ls=':', alpha=0.5, lw=0.8)

    band = (fsub >= 2.4) & (fsub <= 2.5)
    if np.any(band):
        best_joint = np.min(joint_s11[pi][band])
        best_indep = np.min(indep_s11[pi][band])
        ax.annotate(f'Joint:{best_joint:.0f}dB Indep:{best_indep:.0f}dB',
                    xy=(2.45, best_joint), fontsize=7, color=colors[pi],
                    ha='center', va='bottom')

    ax.set_title(f'Port {pi+1}', fontweight='bold')
    ax.set_xlabel('GHz'); ax.set_ylabel('dB'); ax.set_ylim([-40, 0])
    ax.legend(fontsize=7, loc='lower right')
plt.tight_layout(rect=[0,0,1,0.95])
p1 = os.path.join(OUT, "s11_independent_vs_joint.png")
fig.savefig(p1, bbox_inches='tight'); plt.close()
print(f"  {p1}")

# ═══ PLOT 2: Bar chart comparison ═══
print("Plotting bar chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Independent vs Joint Optimization at 2.45 GHz', fontsize=14, fontweight='bold')

ports = [f'P{i+1}' for i in range(N)]
x = np.arange(N); w = 0.25

# S11 bars
raw_db = [raw_s11[pi][np.argmin(np.abs(fsub - 2.45))] for pi in range(N)]
indep_db = [indep_s11[pi][np.argmin(np.abs(fsub - 2.45))] for pi in range(N)]
joint_db = [joint_s11[pi][np.argmin(np.abs(fsub - 2.45))] for pi in range(N)]

ax1.bar(x - w, raw_db, w, label='Raw', color='#ccc', edgecolor='#999')
ax1.bar(x, indep_db, w, label='Independent', color=[c for c in colors], alpha=0.5, edgecolor='#333')
ax1.bar(x + w, joint_db, w, label='Joint', color=[c for c in colors], edgecolor='#333')
ax1.set_xticks(x); ax1.set_xticklabels(ports); ax1.set_ylabel('S11 (dB)')
ax1.set_title('Return Loss'); ax1.legend(); ax1.axhline(-10, color='green', ls=':', alpha=0.5)

# Efficiency bars
raw_e = [(1 - 10**(raw_db[i]/10)) * 100 for i in range(N)]
indep_e = [(1 - 10**(indep_db[i]/10)) * 100 for i in range(N)]
joint_e = [(1 - 10**(joint_db[i]/10)) * 100 for i in range(N)]

ax2.bar(x - w, raw_e, w, label='Raw', color='#ccc', edgecolor='#999')
ax2.bar(x, indep_e, w, label='Independent', color=[c for c in colors], alpha=0.5, edgecolor='#333')
ax2.bar(x + w, joint_e, w, label='Joint', color=[c for c in colors], edgecolor='#333')
ax2.set_xticks(x); ax2.set_xticklabels(ports); ax2.set_ylabel('%')
ax2.set_title('Mismatch Efficiency'); ax2.legend(); ax2.set_ylim([0, 105])

plt.tight_layout(rect=[0,0,1,0.93])
p2 = os.path.join(OUT, "bar_independent_vs_joint.png")
fig.savefig(p2, bbox_inches='tight'); plt.close()
print(f"  {p2}")

# ═══ PLOT 3: Smith chart comparison ═══
print("Plotting Smith chart...")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title('Smith Chart: Independent (dashed) vs Joint (solid)', fontsize=14, fontweight='bold')
theta = np.linspace(0, 2*np.pi, 361)
ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1, alpha=0.3)
ax.set_aspect('equal'); ax.set_xlim([-1.15, 1.15]); ax.set_ylim([-1.15, 1.15])
ax.set_xlabel('Re(Gamma)'); ax.set_ylabel('Im(Gamma)')

for pi in range(N):
    # Raw gamma
    gamma_raw = [data.get_s_matrix_interpolated(f)[pi, pi] for f in freqs[sub]]
    gamma_raw = np.array(gamma_raw)
    ax.plot(gamma_raw.real, gamma_raw.imag, ':', color='#ccc', lw=0.8)

    # Independent gamma
    sol = indep_solutions.get(pi)
    if sol:
        gamma_ind = []
        for f_hz in freqs[sub]:
            S_full = data.get_s_matrix_interpolated(f_hz)
            term = {p: 0.0 for p in range(N) if p != pi}
            St = terminate_ports(S_full.copy(), term) if term else S_full.copy()
            S = St.copy()
            try:
                for ch in sol.component_choices:
                    cs = ch.component.get_s_matrix_at_freq(f_hz)
                    if ch.connection_type == 'series': S = _embed_series_on_port(S, cs, ch.port)
                    else: S = _embed_shunt_to_ground(S, cs, ch.port)
                gamma_ind.append(S[0, 0])
            except: gamma_ind.append(St[0, 0])
        gamma_ind = np.array(gamma_ind)
        ax.plot(gamma_ind.real, gamma_ind.imag, '--', color=colors[pi], lw=1.5, alpha=0.5)

    # Joint gamma
    gamma_joint = []
    for f_hz in freqs[sub]:
        S_full = data.get_s_matrix_interpolated(f_hz)
        try:
            S_j = embed_all_matching(S_full, joint_opt.current_matching_s, f_hz)
            gamma_joint.append(S_j[pi, pi])
        except: gamma_joint.append(S_full[pi, pi])
    gamma_joint = np.array(gamma_joint)
    ax.plot(gamma_joint.real, gamma_joint.imag, '-', color=colors[pi], lw=2.5, label=f'P{pi+1}')

ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
p3 = os.path.join(OUT, "smith_independent_vs_joint.png")
fig.savefig(p3, bbox_inches='tight'); plt.close()
print(f"  {p3}")

print(f"\nDone! Files: {OUT}")
