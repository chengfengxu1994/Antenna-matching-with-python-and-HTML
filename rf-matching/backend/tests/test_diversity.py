"""
Regression test: solution diversity in joint optimization.

Checks that:
  1. Joint optimize port1/port2 produces solutions (solutions_count > 0)
  2. Phase1 each port candidate count is sufficient
  3. Final top5 has at least 2 different capacitor values
  4. Final top5 has at least 2 different topologies or connection patterns

This prevents the algorithm from always collapsing to the same boundary
combination (e.g., C=0.1pF + L=150nH for every port).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from engine.touchstone import parse_touchstone
from engine.multiport_optimizer import JointMultiPortOptimizer, PortMatchConfig
from engine.murata_db_adapter import load_murata_db


SNP_PATH = r'E:\RF matching\snp\3_antenna_no_dielectric.s3p'
DB_PATH = r'E:\RF matching\Murata\murata_components.db'




def _get_unique_capacitor_values(joint_solutions, top_n=5):
    """Extract unique capacitor values from top-N joint solutions."""
    cap_values = set()
    for sol in joint_solutions[:top_n]:
        for pi, port_sol in sol.port_solutions.items():
            for c in port_sol.component_choices:
                part = c.component
                part_num = getattr(part, 'part_number', '')
                # Heuristic: capacitors have 'C' or 'GJM' in part number, or value in pF range
                is_cap = (
                    'C' in part_num.upper()[:3]
                    or 'GJM' in part_num.upper()[:3]
                    or getattr(part, 'nominal_unit', '') in ('pF', 'F')
                )
                if is_cap or c.connection_type == 'shunt':
                    val = getattr(part, 'nominal_value', None)
                    if val is not None and val > 0:
                        cap_values.add(round(val, 2))
    return cap_values


def _get_unique_topologies(joint_solutions, top_n=5):
    """Get the set of topology names used in top-N joint solutions."""
    topos = set()
    for sol in joint_solutions[:top_n]:
        for pi, port_sol in sol.port_solutions.items():
            # Topology name encodes the connection pattern
            topos.add(port_sol.topology.name)
    return topos


def _get_unique_connection_patterns(joint_solutions, top_n=5):
    """Get connection type patterns (e.g., 'series+shunt', 'shunt+series')."""
    patterns = set()
    for sol in joint_solutions[:top_n]:
        for pi, port_sol in sol.port_solutions.items():
            pattern = '+'.join(c.connection_type for c in port_sol.component_choices)
            patterns.add(pattern)
    return patterns


def _check_feasible_component_values(joint_solutions, top_n=5,
                                     max_cap_pf=100.0, max_ind_nh=80.0):
    """Assert that top-N solutions use physically plausible component values.

    Capacitor nominal values must be <= max_cap_pf (pF).
    Inductor nominal values must be <= max_ind_nh (nH).
    Returns (max_cap_found, max_ind_found).
    """
    max_cap = 0.0
    max_ind = 0.0
    for sol in joint_solutions[:top_n]:
        for port_sol in sol.port_solutions.values():
            for c in port_sol.component_choices:
                part = c.component
                val = getattr(part, 'nominal_value', None)
                unit = getattr(part, 'nominal_unit', '')
                if val is None:
                    continue
                if 'C' in getattr(part, 'part_number', '').upper()[:3] \
                   or 'GJM' in getattr(part, 'part_number', '').upper()[:3] \
                   or unit in ('pF', 'F'):
                    if unit in ('pF', 'F') or val < 1e-6:
                        cap_pf = val * 1e12 if unit == 'F' else val
                        if cap_pf > max_cap:
                            max_cap = cap_pf
                        assert cap_pf <= max_cap_pf, \
                            "Component %s: capacitor value %.2f pF exceeds max %.1f pF" \
                            % (getattr(part, 'part_number', '?'), cap_pf, max_cap_pf)
                elif 'L' in getattr(part, 'part_number', '').upper()[:3] \
                     or unit in ('nH', 'H'):
                    if unit in ('nH', 'H') or val > 1e-6:
                        ind_nh = val * 1e9 if unit == 'H' else val
                        if ind_nh > max_ind:
                            max_ind = ind_nh
                        assert ind_nh <= max_ind_nh, \
                            "Component %s: inductor value %.2f nH exceeds max %.1f nH" \
                            % (getattr(part, 'part_number', '?'), ind_nh, max_ind_nh)
    return max_cap, max_ind


def test_diversity():
    """Main diversity regression test."""
    print("=" * 60)
    print("DIVERSITY REGRESSION TEST")
    print("=" * 60)

    # 1. Load SNP
    assert os.path.exists(SNP_PATH), "SNP file not found: " + SNP_PATH
    with open(SNP_PATH, 'r', encoding='utf-8', errors='replace') as f:
        dut = parse_touchstone(f.read(), os.path.basename(SNP_PATH))
    print("\nDUT: %d-port, %d freq points" % (dut.num_ports, len(dut.frequencies)))
    print("  Freq range: %.1f - %.1f MHz" % (min(dut.frequencies)/1e6, max(dut.frequencies)/1e6))

    # Check frequency covers 2.4-2.5 GHz
    assert any(2400e6 <= f <= 2500e6 for f in dut.frequencies), \
        "DUT must cover 2.4-2.5 GHz range"

    # 2. Load component library from database
    assert os.path.exists(DB_PATH), "Murata DB not found: " + DB_PATH
    lib = load_murata_db(DB_PATH)
    print("\nLibrary: %d inductors, %d capacitors" % (len(lib.inductors), len(lib.capacitors)))
    assert len(lib.inductors) > 0, "Component library must have inductors"
    assert len(lib.capacitors) > 0, "Component library must have capacitors"

    # 3. Run joint optimization: ports 0 and 1, 2.4-2.5 GHz
    print("\nRunning joint optimization (ports 0 & 1, 2.4-2.5 GHz)...")
    port_configs = [
        PortMatchConfig(
            port_index=0,
            max_components=2,
            target_frequency_hz=2.45e9,
        ),
        PortMatchConfig(
            port_index=1,
            max_components=2,
            target_frequency_hz=2.45e9,
        ),
    ]

    optimizer = JointMultiPortOptimizer(
        dut=dut,
        component_library=lib,
        port_configs=port_configs,
        top_candidates_per_port=8,
        timeout_seconds=120,
        min_avg_balance=0.5,
        debug=True,
        debug_top_n=8,
    )

    joint_solutions = optimizer.optimize()

    # -- Check 1: Solutions exist --
    print("\n== Check 1: Solutions found = %d" % len(joint_solutions))
    assert len(joint_solutions) > 0, \
        "Joint optimization found 0 solutions"
    print("  PASS: %d solutions > 0" % len(joint_solutions))

    # -- Check 2: Phase1 each port has enough candidates --
    print("\n== Check 2: Top candidate counts")
    debug_info = getattr(optimizer, '_debug_info', {})
    if debug_info and 'phase1_candidates' in debug_info:
        for pi_str, cands in debug_info['phase1_candidates'].items():
            print("  Port %s: %d candidates" % (pi_str, len(cands)))
            assert len(cands) >= 3, \
                "Port %s has only %d phase1 candidates (need >= 3)" % (pi_str, len(cands))
        print("  PASS: Each port has sufficient phase1 candidates")
    else:
        print("  (no debug info - check logs for candidate counts)")

    # -- Check 3: Top-5 solutions have at least 2 different capacitor values --
    top_n = min(5, len(joint_solutions))
    print("\n== Check 3: Capacitor value diversity in top %d" % top_n)
    cap_vals = _get_unique_capacitor_values(joint_solutions, top_n=5)
    print("  Unique capacitor values found: %s" % sorted(cap_vals))
    assert len(cap_vals) >= 2, \
        "Top 5 solutions only have %d unique capacitor value(s): %s. " \
        "Expected at least 2 different values." % (len(cap_vals), cap_vals)
    print("  PASS: %d unique capacitor values >= 2" % len(cap_vals))

    # -- Check 4: Top-5 solutions have at least 2 different topology/connection patterns --
    print("\n== Check 4: Topology diversity in top %d" % top_n)
    topos = _get_unique_topologies(joint_solutions, top_n=5)
    patterns = _get_unique_connection_patterns(joint_solutions, top_n=5)
    print("  Unique topologies: %s" % topos)
    print("  Unique connection patterns: %s" % patterns)
    assert len(topos) >= 2 or len(patterns) >= 2, \
        "Top 5 solutions only use %d topology(ies) / %d pattern(s). " \
        "Expected at least 2 different topologies or connection patterns." % (len(topos), len(patterns))
    print("  PASS: at least 2 distinct topologies or patterns")

    # -- Check 5: Feasible component values (regression against implausible endpoints) --
    print("\n== Check 5: Component value feasibility in top %d" % top_n)
    max_cap, max_ind = _check_feasible_component_values(
        joint_solutions, top_n=5, max_cap_pf=100.0, max_ind_nh=80.0)
    print("  Max capacitor: %.2f pF (limit 100 pF)" % max_cap)
    print("  Max inductor:  %.2f nH (limit 80 nH)" % max_ind)
    print("  PASS: All component values within feasible bounds")

    # -- Print best solution summary --
    best = joint_solutions[0]
    print("\n%s" % ('=' * 60))
    print("Best solution: score=%.5f" % best.balanced_score)
    for pi, port_sol in best.port_solutions.items():
        comps = ', '.join(
            "%s:%.4g%s" % (c.connection_type, c.component.nominal_value, c.component.nominal_unit)
            for c in port_sol.component_choices
        )
        pm = best.port_metrics.get(pi, {})
        print("  Port %d: %s [%s]" % (pi, port_sol.topology.name, comps))
        print("    RL=%.2fdB eff=%.4f coupling=%.4f"
              % (pm.get('s11_db', 0), pm.get('total_efficiency', 0), pm.get('coupling_loss', 0)))

    print("\n%s" % ('=' * 60))
    print("ALL DIVERSITY CHECKS PASSED")
    print("%s" % ('=' * 60))


if __name__ == '__main__':
    test_diversity()
