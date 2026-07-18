"""
Joint multi-port matching optimizer.
Accounts for inter-port mutual coupling by iteratively optimizing each port
with the matched state of all other ports embedded into the DUT S-matrix.

Restricted to LQP03HQ (inductors) + GJM03 (capacitors).
"""
import numpy as np, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.touchstone import load_touchstone_file
from engine.component_lib import scan_murata_directory, ComponentInfo
from engine.network import (
    terminate_ports, _embed_series_on_port, _embed_shunt_to_ground,
    s_to_y, y_to_s, s_to_z, z_to_s
)


def build_shunt_s(comp_s_2x2):
    """Convert 2-port component S to 1-port shunt gamma (port2 shorted to ground)."""
    S = comp_s_2x2
    gamma = S[0, 0] + S[0, 1] * (-1) * S[1, 0] / (1 - (-1) * S[1, 1])
    return gamma


def embed_all_matching(S_base, matching_networks, freq_hz):
    """
    Embed matching networks for ALL ports into the DUT S-matrix.
    matching_networks: dict {port_idx: [(conn_type, comp_s_2x2), ...]}
    Returns the full S-matrix with all matching embedded.
    """
    S = S_base.copy()
    n = S.shape[0]
    Z0 = 50.0

    for port_idx, components in matching_networks.items():
        for conn_type, comp_s in components:
            comp_at_freq = comp_s  # Already at target frequency
            if conn_type == 'series':
                S = _embed_series_on_port(S, comp_at_freq, port_idx)
            elif conn_type == 'shunt':
                # Shunt: terminate comp port2 with short, add admittance to port
                gamma_shunt = build_shunt_s(comp_at_freq)
                Y0 = 1.0 / Z0
                Y_shunt = Y0 * (1.0 - gamma_shunt) / (1.0 + gamma_shunt)
                Y = s_to_y(S)
                Y[port_idx, port_idx] += Y_shunt
                S = y_to_s(Y)
    return S


def get_filtered_components(lib, series_l='LQP03HQ', series_c='GJM03', min_val=0.01, max_val_l=100, max_val_c=100):
    """Get components filtered by series, one per unique value, sorted by value."""
    inductors = {}
    for c in lib.inductors:
        if series_l.upper() in c.part_number.upper() and min_val < c.nominal_value <= max_val_l:
            v = c.nominal_value
            if v not in inductors:
                inductors[v] = c

    capacitors = {}
    for c in lib.capacitors:
        if series_c.upper() in c.part_number.upper() and min_val < c.nominal_value <= max_val_c:
            v = c.nominal_value
            if v not in capacitors:
                capacitors[v] = c

    L_sorted = [inductors[v] for v in sorted(inductors.keys())]
    C_sorted = [capacitors[v] for v in sorted(capacitors.keys())]
    return L_sorted, C_sorted


class JointMultiPortOptimizer:
    """
    Joint multi-port matching optimizer.
    Iteratively optimizes each port while accounting for other ports' matching.
    """

    def __init__(self, dut_data, component_lib, freq_hz=2.45e9,
                 max_components=2, series_l='LQP03HQ', series_c='GJM03'):
        self.dut = dut_data
        self.freq_hz = freq_hz
        self.max_components = max_components
        self.N = dut_data.num_ports
        self.Z0 = 50.0

        # Get filtered components
        self.L_parts, self.C_parts = get_filtered_components(component_lib, series_l, series_c)
        print(f"  Library: {len(self.L_parts)}L ({series_l}) + {len(self.C_parts)}C ({series_c})")

        # Pre-cache S-parameters at target frequency
        self.L_s = {}
        for c in self.L_parts:
            self.L_s[c.nominal_value] = c.get_s_matrix_at_freq(freq_hz)
        self.C_s = {}
        for c in self.C_parts:
            self.C_s[c.nominal_value] = c.get_s_matrix_at_freq(freq_hz)

        # Base DUT S-matrix
        self.S_dut = dut_data.get_s_matrix_interpolated(freq_hz)

        # Current matching state: {port_idx: [(conn_type, value, type), ...]}
        self.current_matching = {}
        # Matching S-parameters: {port_idx: [(conn_type, comp_s_2x2), ...]}
        self.current_matching_s = {}

    def _get_comp_s(self, comp_type, value):
        """Get cached S-parameters for a component."""
        if comp_type == 'L':
            return self.L_s.get(value)
        else:
            return self.C_s.get(value)

    def _evaluate_port(self, port_idx, candidate_networks):
        """
        Evaluate S11 at port_idx with ALL ports' matching applied.
        candidate_networks: dict {port_idx: [(conn_type, comp_s), ...]}
        Returns |S11| at port_idx.
        """
        S = embed_all_matching(self.S_dut, candidate_networks, self.freq_hz)
        return abs(S[port_idx, port_idx])

    def optimize_single_port(self, port_idx, other_networks):
        """
        Optimize matching for one port, given the matching state of all other ports.
        Returns: list of (conn_type, value, type) tuples for the best matching.
        """
        L_vals = [c.nominal_value for c in self.L_parts]
        C_vals = [c.nominal_value for c in self.C_parts]

        best_mag = 1.0
        best_network = None

        if self.max_components == 1:
            # Single component: try all L and C as series or shunt
            for conn in ['series', 'shunt']:
                for val in L_vals:
                    comp_s = self.L_s[val]
                    nets = {**other_networks, port_idx: [(conn, comp_s)]}
                    mag = self._evaluate_port(port_idx, nets)
                    if mag < best_mag:
                        best_mag = mag
                        best_network = [(conn, val, 'L')]

                for val in C_vals:
                    comp_s = self.C_s[val]
                    nets = {**other_networks, port_idx: [(conn, comp_s)]}
                    mag = self._evaluate_port(port_idx, nets)
                    if mag < best_mag:
                        best_mag = mag
                        best_network = [(conn, val, 'C')]

        if self.max_components == 2:
            # Two components: try all L/C combinations with series/shunt patterns
            # Coarse scan: pick best single component first, then refine
            patterns = [
                [('series', 'L'), ('shunt', 'C')],
                [('series', 'C'), ('shunt', 'L')],
                [('shunt', 'L'), ('series', 'C')],
                [('shunt', 'C'), ('series', 'L')],
            ]

            # Pre-filter: only keep C values that make sense at this frequency
            # At 2.45 GHz, Xc = 1/(2*pi*f*C). For useful matching: 1 < Xc < 500 ohm
            omega = 2 * np.pi * self.freq_hz
            useful_C_full = sorted([v for v in C_vals if 1.0/(omega*v*1e-12) > 1.0 and 1.0/(omega*v*1e-12) < 500])
            useful_L_full = sorted([v for v in L_vals if omega*v*1e-9 > 1.0 and omega*v*1e-9 < 500])

            # Sample ~20 values evenly for speed
            def sample(vals, n=20):
                if len(vals) <= n: return vals
                step = len(vals) / n
                return [vals[int(i * step)] for i in range(n)]

            useful_L = sample(useful_L_full, 20)
            useful_C = sample(useful_C_full, 20)

            for pattern in patterns:
                (conn1, type1), (conn2, type2) = pattern
                vals1 = useful_L if type1 == 'L' else useful_C
                vals2 = useful_L if type2 == 'L' else useful_C

                for v1 in vals1:
                    s1 = self.L_s[v1] if type1 == 'L' else self.C_s[v1]
                    for v2 in vals2:
                        s2 = self.L_s[v2] if type2 == 'L' else self.C_s[v2]
                        comps = [(conn1, s1), (conn2, s2)]
                        nets = {**other_networks, port_idx: comps}
                        mag = self._evaluate_port(port_idx, nets)
                        if mag < best_mag:
                            best_mag = mag
                            best_network = [(conn1, v1, type1), (conn2, v2, type2)]

        return best_network, best_mag

    def optimize_joint(self, max_iterations=5, target_rl=15.0):
        """
        Joint multi-port optimization via iterative refinement.
        Returns: dict {port_idx: [(conn_type, value, type), ...]}
        """
        print(f"\n  Joint optimization: {self.N} ports, {max_iterations} iterations max")
        print(f"  Target RL: {target_rl} dB")

        # Initialize: no matching on any port
        self.current_matching = {}
        self.current_matching_s = {}

        history = []

        for iteration in range(max_iterations):
            t_start = time.time()
            changed = False

            for pi in range(self.N):
                # Build other_ports networks (exclude current port)
                other_nets = {}
                for pj, comps in self.current_matching_s.items():
                    if pj != pi:
                        other_nets[pj] = comps

                # Optimize this port
                best_net, best_mag = self.optimize_single_port(pi, other_nets)

                if best_net:
                    # Check if this is better than current
                    old_net = self.current_matching.get(pi)
                    if old_net != best_net:
                        changed = True

                    self.current_matching[pi] = best_net
                    self.current_matching_s[pi] = [
                        (conn, self._get_comp_s(t, v))
                        for conn, v, t in best_net
                    ]

            # Measure all ports simultaneously
            all_nets = self.current_matching_s.copy()
            S_matched = embed_all_matching(self.S_dut, all_nets, self.freq_hz)

            results = {}
            for pi in range(self.N):
                s11 = abs(S_matched[pi, pi])
                rl = -20 * np.log10(max(s11, 1e-15))
                eff = (1 - s11**2) * 100
                results[pi] = {'s11': s11, 'rl': rl, 'eff': eff}

            t_elapsed = time.time() - t_start
            avg_rl = np.mean([r['rl'] for r in results.values()])
            min_rl = min(r['rl'] for r in results.values())

            history.append({
                'iteration': iteration + 1,
                'time': t_elapsed,
                'avg_rl': avg_rl,
                'min_rl': min_rl,
                'results': results.copy(),
            })

            rl_str = " ".join(f"P{i+1}:{results[i]['rl']:.1f}" for i in range(self.N))
            print(f"  Iter {iteration+1}: {t_elapsed:.1f}s | avg={avg_rl:.1f} min={min_rl:.1f}dB | {rl_str}")

            # Convergence: all ports above target
            if min_rl >= target_rl and not changed:
                print(f"  Converged at iteration {iteration+1}")
                break

            if not changed and iteration > 0:
                print(f"  No change at iteration {iteration+1}, stopping")
                break

        return self.current_matching, results, history


def format_matching_result(matching, port_idx):
    """Format matching components as string."""
    comps = matching.get(port_idx, [])
    parts = []
    for conn, val, typ in comps:
        unit = 'nH' if typ == 'L' else 'pF'
        parts.append(f"{conn}:{val:.1f}{unit}")
    return " + ".join(parts)


# ═══════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════
if __name__ == "__main__":
    from project_paths import ARTIFACTS_DIR, MURATA_DIR, SNP_DIR

    SNP = str(SNP_DIR / "SAR Head Hand and Phone.s6p")
    MURATA = str(MURATA_DIR)
    FREQ = 2.45e9

    print("=" * 70)
    print("JOINT MULTI-PORT MATCHING OPTIMIZATION")
    print("DUT: SAR Head Hand and Phone (6-port)")
    print(f"Target: {FREQ/1e9:.3f} GHz")
    print("Components: LQP03HQ (inductors) + GJM03 (capacitors)")
    print("=" * 70)

    t_total_start = time.time()

    data = load_touchstone_file(SNP)
    lib = scan_murata_directory(MURATA)

    # Raw coupling
    S_raw = data.get_s_matrix_interpolated(FREQ)
    print(f"\nRaw S11 at {FREQ/1e9:.3f} GHz:")
    for i in range(data.num_ports):
        s = abs(S_raw[i, i])
        rl = -20 * np.log10(max(s, 1e-15))
        print(f"  P{i+1}: |S11|={s:.4f} RL={rl:.1f}dB")

    # Joint optimization
    optimizer = JointMultiPortOptimizer(
        data, lib, freq_hz=FREQ,
        max_components=2,
        series_l='LQP03HQ', series_c='GJM03'
    )

    matching, results, history = optimizer.optimize_joint(
        max_iterations=5, target_rl=10.0
    )

    t_total = time.time() - t_total_start

    # Final results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (Total time: {t_total:.1f}s)")
    print(f"{'='*70}")
    print(f"{'Port':<6} {'Components':<30} {'|S11|':<10} {'RL(dB)':<10} {'Eff(%)':<10}")
    print("-" * 66)
    for pi in range(data.num_ports):
        comp_str = format_matching_result(matching, pi)
        r = results[pi]
        print(f"P{pi+1:<5} {comp_str:<30} {r['s11']:<10.4f} {r['rl']:<10.1f} {r['eff']:<10.1f}")

    # Isolation with matching
    print(f"\nIsolation after matching at {FREQ/1e9:.3f} GHz:")
    S_matched = embed_all_matching(optimizer.S_dut, optimizer.current_matching_s, FREQ)
    for i in range(data.num_ports):
        for j in range(i+1, data.num_ports):
            iso = 20 * np.log10(max(abs(S_matched[i, j]), 1e-15))
            if iso > -30:
                print(f"  S{i+1},{j+1}: {iso:.1f}dB {'***' if iso > -15 else ''}")

    # Save for plotting
    out_dir = str(ARTIFACTS_DIR / "output")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, 'joint_results.npz'),
        S_matched=S_matched,
        S_raw=S_raw,
        freq=FREQ,
        results=str(results),
    )
    print(f"\nResults saved to output/joint_results.npz")
