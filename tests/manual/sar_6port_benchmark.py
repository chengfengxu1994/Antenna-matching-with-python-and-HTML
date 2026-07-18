"""
6-port SAR Head/Hand/Phone matching test.
Each port: 2 components, band 2400-2500 MHz.
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

from engine.touchstone import load_touchstone_file
from engine.component_lib import scan_murata_directory
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from engine.topology import get_standard_topologies
from project_paths import MURATA_DIR, SNP_DIR

SNP = SNP_DIR / "SAR Head Hand and Phone.s6p"
MURATA = MURATA_DIR

print("=" * 60)
print("6-PORT SAR MATCHING TEST")
print("Each port: 2 components, band: 2400-2500 MHz")
print("=" * 60)

data = load_touchstone_file(SNP)
lib = scan_murata_directory(MURATA)
topos = [t for t in get_standard_topologies() if t.num_components <= 2]

N = data.num_ports
print(f"Loaded: {N}-port, {len(data.frequencies)} freqs, {len(lib.inductors)+len(lib.capacitors)} components")
print(f"Topologies: {len(topos)}")

# Show raw S11 at 2.45 GHz
S_raw = data.get_s_matrix_interpolated(2.45e9)
print("\nRaw S11 at 2.45 GHz:")
for i in range(N):
    s = abs(S_raw[i, i])
    print(f"  P{i+1}: |S11|={s:.4f} ({20*np.log10(max(s,1e-15)):.1f} dB)")

all_results = {}
t_total_start = time.time()

for port_idx in range(N):
    t_start = time.time()

    port_states = {p: PortState.LOAD for p in range(N)}
    port_states[port_idx] = PortState.COMPONENT

    config = OptimizerConfig(
        target_frequency_hz=2.45e9,
        max_components=2,
        beam_width=10,
        timeout_seconds=30.0,
        bands_mhz=[[2400, 2500]],
        num_band_points=5,
    )

    opt = MatchingOptimizer(data, lib, config)
    solutions = opt.optimize_full(port_states=port_states, topologies=topos, input_port=port_idx)

    t_elapsed = time.time() - t_start
    raw_db = 20 * np.log10(max(abs(S_raw[port_idx, port_idx]), 1e-15))
    best = solutions[0] if solutions else None
    best_rl = best.s11_db if best else None
    best_topo = best.topology.name if best else "N/A"
    best_comps = ", ".join(
        f"{c.connection_type}:{c.component.nominal_value}{c.component.nominal_unit}"
        for c in best.component_choices
    ) if best else "N/A"

    # Isolation
    isolation = {}
    for other in range(N):
        if other == port_idx:
            continue
        freqs_iso = np.linspace(2400e6, 2500e6, 5)
        vals = []
        for f in freqs_iso:
            S = data.get_s_matrix_interpolated(f)
            vals.append(float(20 * np.log10(max(abs(S[port_idx, other]), 1e-15))))
        isolation[f"S{port_idx+1},{other+1}"] = f"{np.mean(vals):.1f}"

    all_results[port_idx] = {
        "time": t_elapsed, "raw": raw_db, "matched": best_rl,
        "topo": best_topo, "comps": best_comps, "n_sol": len(solutions),
        "iso": isolation,
    }

    print(f"\nPort {port_idx+1}: {t_elapsed:.1f}s | {raw_db:.1f}dB -> {best_rl:.1f}dB | {len(solutions)} sol")
    print(f"  {best_topo} | {best_comps}")
    iso_str = ", ".join(f"{k}:{v}dB" for k, v in list(isolation.items())[:3])
    print(f"  Isolation: {iso_str}")

t_total = time.time() - t_total_start

print(f"\n{'='*60}")
print(f"TOTAL TIME: {t_total:.1f}s ({t_total/60:.1f} min)")
print(f"Per port avg: {t_total/N:.1f}s")

print(f"\n{'Port':<6} {'Raw':<10} {'Matched':<10} {'Delta':<8} {'Time':<8} {'Sols':<6}")
print("-" * 48)
for pi, r in all_results.items():
    delta = (r["matched"] or 0) - r["raw"]
    print(f"P{pi+1:<5} {r['raw']:>6.1f}dB  {r['matched']:>6.1f}dB  {delta:>+5.1f}dB  {r['time']:>5.1f}s  {r['n_sol']:>4}")
