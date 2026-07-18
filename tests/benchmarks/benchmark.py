"""
Benchmark script for RF matching engine.
Tests speed, accuracy, and coverage.
"""
import time
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

from engine.touchstone import load_touchstone_file
from engine.component_lib import scan_murata_directory
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from engine.topology import get_standard_topologies
from project_paths import MURATA_DIR, SNP_DIR


def main():
    # Load data
    print("Loading data...")
    t0 = time.time()
    data = load_touchstone_file(os.path.join(SNP_DIR, "For64MHz.s1p"))
    library = scan_murata_directory(MURATA_DIR)
    t_load = time.time() - t0
    print(f"  Data loaded in {t_load:.1f}s")
    print(f"  Inductors: {len(library.inductors)}, Capacitors: {len(library.capacitors)}")

    # Raw S11
    S_raw = data.get_s_matrix_interpolated(64e6)
    s11_raw_mag = abs(S_raw[0, 0])
    s11_raw_db = 20 * np.log10(s11_raw_mag)
    print(f"  Raw S11 at 64MHz: |S11|={s11_raw_mag:.4f}, dB={s11_raw_db:.1f}")

    # Benchmark 1: L-network (2 components)
    print("\n" + "=" * 60)
    print("Benchmark 1: L-network optimization at 64 MHz (2 components)")
    print("=" * 60)
    config1 = OptimizerConfig(
        target_frequency_hz=64e6,
        max_components=2,
        beam_width=10,
        max_combinations_to_evaluate=50000,
        timeout_seconds=30.0,
    )
    opt1 = MatchingOptimizer(data, library, config1)
    t0 = time.time()
    sols1 = opt1.optimize_l_network({}, input_port=0)
    t1 = time.time() - t0
    print(f"  Time: {t1:.2f}s, Solutions found: {len(sols1)}")
    if sols1:
        best = sols1[0]
        print(f"  Best: |S11|={best.s11_magnitude:.4f}, RL={best.return_loss_db:.1f}dB, VSWR={best.vswr:.2f}")
        comps = ", ".join(
            f"{c.component.part_number}({c.component.nominal_value}{c.component.nominal_unit})"
            for c in best.component_choices
        )
        print(f"  Components: {comps}")
    else:
        print("  WARNING: No solutions found!")

    # Benchmark 2: Full topology (3 components)
    print("\n" + "=" * 60)
    print("Benchmark 2: Full topology optimization (max 3 components)")
    print("=" * 60)
    config2 = OptimizerConfig(
        target_frequency_hz=64e6,
        max_components=3,
        beam_width=10,
        timeout_seconds=30.0,
    )
    opt2 = MatchingOptimizer(data, library, config2)
    t0 = time.time()
    sols2 = opt2.optimize_full(port_states={}, input_port=0)
    t2 = time.time() - t0
    print(f"  Time: {t2:.2f}s, Solutions found: {len(sols2)}")
    if sols2:
        best2 = sols2[0]
        print(f"  Best: |S11|={best2.s11_magnitude:.4f}, RL={best2.return_loss_db:.1f}dB, VSWR={best2.vswr:.2f}")
        print(f"  Topology: {best2.topology.name}")
        comps = ", ".join(
            f"{c.connection_type}:{c.component.nominal_value}{c.component.nominal_unit}"
            for c in best2.component_choices
        )
        print(f"  Components: {comps}")
    else:
        print("  WARNING: No solutions found!")

    # Benchmark 3: Multi-band evaluation
    print("\n" + "=" * 60)
    print("Benchmark 3: Multi-band evaluation (2 bands, 5 points each)")
    print("=" * 60)
    config3 = OptimizerConfig(
        target_frequency_hz=64e6,
        max_components=2,
        beam_width=10,
        timeout_seconds=30.0,
        bands_mhz=[[60, 70], [80, 90]],
        num_band_points=5,
    )
    opt3 = MatchingOptimizer(data, library, config3)
    t0 = time.time()
    sols3 = opt3.optimize_full(port_states={}, input_port=0)
    t3 = time.time() - t0
    print(f"  Time: {t3:.2f}s, Solutions found: {len(sols3)}")
    if sols3:
        best3 = sols3[0]
        print(f"  Best: |S11|={best3.s11_magnitude:.4f}")
        if best3.avg_band_efficiency is not None:
            print(f"  Avg Band Efficiency: {best3.avg_band_efficiency*100:.1f}%")
        if best3.band_efficiency:
            for band in best3.band_efficiency:
                b = band["band_mhz"]
                e = band["avg_efficiency"]
                print(f"    Band {b[0]}-{b[1]} MHz: avg efficiency = {e*100:.1f}%")
    else:
        print("  WARNING: No solutions found!")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    total = len(library.inductors) + len(library.capacitors)
    print(f"Library load time: {t_load:.1f}s ({total} components: {len(library.inductors)}L + {len(library.capacitors)}C)")
    print(f"Unique values: {len(library.get_unique_inductor_values())}L + {len(library.get_unique_capacitor_values())}C")
    print(f"")
    print(f"L-network (2-comp, analytic+search): {t1:.2f}s")
    print(f"Full topology (3-comp, beam search):  {t2:.2f}s")
    print(f"Multi-band (2-comp + 2 bands):        {t3:.2f}s")
    print(f"")
    
    if sols1 and sols2:
        print(f"Accuracy comparison at 64 MHz:")
        print(f"  Raw antenna:     |S11|={s11_raw_mag:.4f} ({s11_raw_db:.1f} dB)")
        print(f"  L-network best:  |S11|={sols1[0].s11_magnitude:.4f} ({sols1[0].return_loss_db:.1f} dB)")
        print(f"  Full topo best:  |S11|={sols2[0].s11_magnitude:.4f} ({sols2[0].return_loss_db:.1f} dB)")
    
    # Topology coverage check
    print(f"\nTopology coverage:")
    topos = get_standard_topologies()
    for n in [2, 3, 4]:
        matching = [t for t in topos if t.num_components == n]
        print(f"  {n}-component: {len(matching)} topologies")
        for t in matching:
            print(f"    - {t.name}")


if __name__ == "__main__":
    main()
