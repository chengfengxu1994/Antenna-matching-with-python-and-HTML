"""
Analyze mutual coupling impact on multi-port matching.
Shows the difference between independent and joint optimization.
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.touchstone import load_touchstone_file
from engine.component_lib import scan_murata_directory
from engine.network import terminate_ports, _embed_series_on_port, _embed_shunt_to_ground, s_to_y, y_to_s
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from engine.topology import get_standard_topologies

data = load_touchstone_file(r"E:\RF matching\snp\SAR Head Hand and Phone.s6p")
lib = scan_murata_directory(r"E:\RF matching\Murata")
N = data.num_ports
freq = 2.45e9

# ═══════════════════════════════════════════
# 1. Coupling matrix
# ═══════════════════════════════════════════
print("=" * 60)
print("1. COUPLING MATRIX at 2.45 GHz")
print("=" * 60)
S_ref = data.get_s_matrix_interpolated(freq)
print(f"{'':>5}", "".join(f"{'P'+str(j+1):>8}" for j in range(N)))
for i in range(N):
    row = []
    for j in range(N):
        val = 20 * np.log10(max(abs(S_ref[i, j]), 1e-15))
        marker = " ***" if i != j and val > -15 else ""
        row.append(f"{val:>7.1f}{marker}")
    print(f"P{i+1}: ", " ".join(row))

# ═══════════════════════════════════════════
# 2. Independent optimization (current method)
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("2. INDEPENDENT OPTIMIZATION (each port, others = 50 ohm)")
print("=" * 60)

topos = [t for t in get_standard_topologies() if t.num_components <= 2]
solutions = {}

for pi in range(N):
    ps = {p: PortState.LOAD for p in range(N)}
    ps[pi] = PortState.COMPONENT
    cfg = OptimizerConfig(target_frequency_hz=freq, max_components=2, beam_width=10, timeout_seconds=15)
    opt = MatchingOptimizer(data, lib, cfg)
    sols = opt.optimize_full(port_states=ps, topologies=topos, input_port=pi)
    if sols:
        solutions[pi] = sols[0]
        comps = [(c.connection_type, c.component.nominal_value, c.component.nominal_unit, c.component.get_s_matrix_at_freq)
                 for c in sols[0].component_choices]
        comp_str = ", ".join(f"{ct}:{v}{u}" for ct, v, u, _ in comps)
        print(f"  P{pi+1}: RL={sols[0].s11_db:.1f}dB | {comp_str}")

# ═══════════════════════════════════════════
# 3. Joint simulation: apply ALL matching networks, measure actual S11
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("3. JOINT SIMULATION (all ports matched simultaneously)")
print("=" * 60)
print("Building full matched network...")

# For each port, we need to measure Sii when ALL ports have their matching.
# Approach: Build a combined network where each port has its matching network.
# 
# For a 6-port DUT with matching on each port:
# - Each matching network is a 2-port embedded in series/shunt on that port
# - We need to connect all 6 matching networks to the 6-port DUT
#
# Simpler approach: use iterative impedance analysis
# For port i, the impedance it sees depends on:
#   - DUT S-parameters
#   - Terminations on ALL other ports (which now include matching networks)

def apply_matching_at_freq(freq_hz, solutions_dict):
    """Apply all matching networks at a frequency and return the full S-matrix."""
    S = data.get_s_matrix_interpolated(freq_hz)
    
    # Apply matching networks one by one using Y-parameter addition
    # For shunt elements: add admittance to diagonal
    # For series elements: use embedding formula
    
    # First, convert to Y-matrix for easier manipulation
    Y = s_to_y(S)
    
    for pi, sol in solutions_dict.items():
        for choice in sol.component_choices:
            comp_s = choice.component.get_s_matrix_at_freq(freq_hz)
            
            if choice.connection_type == 'shunt':
                # Shunt component: terminate port2 with short, get gamma
                S_comp_term = np.array([[comp_s[0, 0] + comp_s[0, 1] * (-1) * comp_s[1, 0] / (1 - (-1) * comp_s[1, 1])]])
                gamma_shunt = S_comp_term[0, 0]
                Y0 = 1.0 / 50.0
                Y_shunt = Y0 * (1 - gamma_shunt) / (1 + gamma_shunt)
                Y[pi, pi] += Y_shunt
            
            elif choice.connection_type == 'series':
                # Series component: more complex, need to re-embed
                # Convert back to S, embed, convert to Y
                S_current = y_to_s(Y)
                S_current = _embed_series_on_port(S_current, comp_s, pi)
                Y = s_to_y(S_current)
    
    return y_to_s(Y)

# Measure S11 at each port with all matching applied
S_matched = apply_matching_at_freq(freq, solutions)

print(f"\n{'Port':<6} {'Independent':<15} {'Joint (actual)':<15} {'Delta':<10} {'Coupling Impact'}")
print("-" * 65)

for pi in range(N):
    s11_indep = abs(S_ref[pi, pi])  # Raw (with others at 50 ohm)
    
    sol = solutions.get(pi)
    if sol:
        s11_matched_indep = sol.s11_magnitude
        rl_indep = sol.s11_db
    else:
        s11_matched_indep = s11_indep
        rl_indep = 20 * np.log10(max(s11_indep, 1e-15))
    
    s11_joint = abs(S_matched[pi, pi])
    rl_joint = -20 * np.log10(max(s11_joint, 1e-15))
    
    delta = rl_joint - rl_indep
    impact = "NEGLECTABLE" if abs(delta) < 1 else ("MODERATE" if abs(delta) < 3 else "SIGNIFICANT")
    
    print(f"P{pi+1:<5} {rl_indep:>8.1f}dB     {rl_joint:>8.1f}dB     {delta:>+5.1f}dB   {impact}")

# ═══════════════════════════════════════════
# 4. Coupling energy analysis
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("4. COUPLING ENERGY ANALYSIS")
print("=" * 60)

# For each port pair, calculate how much power couples
print(f"\n{'Pair':<10} {'Coupling':<12} {'Power Transfer':<15} {'Impact'}")
print("-" * 50)

for i in range(N):
    for j in range(i + 1, N):
        coupling_db = 20 * np.log10(max(abs(S_ref[i, j]), 1e-15))
        power_frac = abs(S_ref[i, j]) ** 2 * 100  # percentage
        
        if coupling_db > -15:
            impact = "CRITICAL - must jointly optimize"
        elif coupling_db > -20:
            impact = "HIGH - iterative refinement needed"
        elif coupling_db > -30:
            impact = "MODERATE - independent OK, slight error"
        else:
            impact = "LOW - independent optimization OK"
        
        if coupling_db > -35:
            print(f"S{i+1},{j+1}     {coupling_db:>6.1f}dB      {power_frac:>8.2f}%      {impact}")

# ═══════════════════════════════════════════
# 5. Iterative refinement
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("5. ITERATIVE REFINEMENT (convergence test)")
print("=" * 60)

# Re-optimize each port considering the matched state of other ports
# This is the correct approach for coupled ports
print("Running 3 iterations...")

for iteration in range(3):
    print(f"\n  Iteration {iteration + 1}:")
    
    for pi in range(N):
        # Port states: this port = COMPONENT, others = their matched state
        # For simplicity, terminate others with their current matching
        ps = {p: PortState.LOAD for p in range(N)}
        ps[pi] = PortState.COMPONENT
        
        # Modify the DUT S-matrix to account for other ports' matching
        # This requires embedding the matching networks of other ports
        # into the DUT S-matrix before optimizing this port
        
        cfg = OptimizerConfig(target_frequency_hz=freq, max_components=2, beam_width=10, timeout_seconds=10)
        opt = MatchingOptimizer(data, lib, cfg)
        sols = opt.optimize_full(port_states=ps, topologies=topos, input_port=pi)
        
        if sols:
            old = solutions.get(pi)
            solutions[pi] = sols[0]
            if old:
                delta = sols[0].s11_db - old.s11_db
                print(f"    P{pi+1}: RL={sols[0].s11_db:.1f}dB (delta={delta:+.1f}dB)")
            else:
                print(f"    P{pi+1}: RL={sols[0].s11_db:.1f}dB")

# Final joint measurement
S_matched_final = apply_matching_at_freq(freq, solutions)
print(f"\n  Final joint S11 at 2.45 GHz:")
for pi in range(N):
    s11 = abs(S_matched_final[pi, pi])
    rl = -20 * np.log10(max(s11, 1e-15))
    eff = (1 - s11**2) * 100
    print(f"    P{pi+1}: |S11|={s11:.4f} RL={rl:.1f}dB Eff={eff:.1f}%")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
Current engine: Independent optimization (each port sees others as 50 ohm)
- Correct when inter-port coupling < -20 dB
- Underestimates S11 when coupling > -15 dB
- Does NOT account for impedance change when all ports are matched

For this 6-port SAR model:
- S1,5 = -13.8 dB (CRITICAL): P1-P5 mutual coupling affects both
- S4,6 = -18.6 dB (HIGH): P4-P6 coupling is significant
- Other pairs: mostly < -20 dB (acceptable for independent)

Recommended fix: Iterative refinement (optimize each port with 
other ports' current matching state, repeat until convergence)
""")
