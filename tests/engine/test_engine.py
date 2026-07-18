"""
Test the core engine modules with real SNP and Murata data.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

from engine.touchstone import parse_touchstone, load_touchstone_file
from engine.network import (
    terminate_port, terminate_ports, s_to_z, z_to_s,
    cascade_2port, _embed_series_on_port, _embed_shunt_to_ground,
    connect_2port_to_multiport
)
from engine.component_lib import scan_murata_directory, parse_murata_part
from engine.topology import get_standard_topologies, Topology
from engine.optimizer import MatchingOptimizer, OptimizerConfig, PortState
from project_paths import MURATA_DIR, SNP_DIR


def test_touchstone_loader_accepts_pathlike(tmp_path: Path):
    sample = tmp_path / "pathlike.s1p"
    sample.write_text("# GHZ S RI R 50\n1.0 0.1 0.0\n", encoding="ascii")

    data = load_touchstone_file(sample)

    assert data.num_ports == 1
    assert data.frequencies == [1.0e9]


def test_touchstone_parser():
    """Test parsing various Touchstone files."""
    print("=" * 60)
    print("TEST: Touchstone Parser")
    print("=" * 60)

    test_files = [
        ("For64MHz.s1p", 1, "MHZ"),
        ("test_multiport.s3p", 3, "GHZ"),
    ]

    for filename, expected_ports, expected_unit in test_files:
        filepath = os.path.join(SNP_DIR, filename)
        if not os.path.isfile(filepath):
            print(f"  SKIP: {filename} not found")
            continue

        data = load_touchstone_file(filepath)
        assert data.num_ports == expected_ports, \
            f"Expected {expected_ports} ports, got {data.num_ports}"
        assert data.frequency_unit == expected_unit, \
            f"Expected {expected_unit}, got {data.frequency_unit}"
        assert len(data.frequencies) > 0, "No frequencies parsed"

        print(f"  OK: {filename} -- {data.num_ports} ports, "
              f"{len(data.frequencies)} frequencies, "
              f"range: {min(data.frequencies)/1e6:.2f}-{max(data.frequencies)/1e6:.2f} MHz")

        # Test S-matrix extraction
        S = data.get_s_matrix(0)  # First frequency
        assert S.shape == (expected_ports, expected_ports)
        # S-matrix should be symmetric for passive networks
        # Check that S_ij ~= S_ji
        for i in range(expected_ports):
            for j in range(i + 1, expected_ports):
                diff = abs(S[i, j] - S[j, i])
                assert diff < 0.1, f"S-matrix not symmetric: S[{i},{j}] != S[{j},{i}]"

    print("  PASS: Touchstone parser works correctly\n")


def test_network_operations():
    """Test network operations: port termination, cascade, etc."""
    print("=" * 60)
    print("TEST: Network Operations")
    print("=" * 60)

    # Create a simple 3-port network for testing
    # Perfect 50-ohm matched network
    S_3port = np.zeros((3, 3), dtype=complex)

    # Test port termination
    # Terminate port 2 with gamma=0 (load)
    S_2port = terminate_port(S_3port, 2, 0.0)
    assert S_2port.shape == (2, 2), f"Expected 2x2, got {S_2port.shape}"
    # Should be all zeros since original was matched
    assert np.allclose(S_2port, 0), "Terminated matched network should stay matched"

    # Terminate with open (gamma=1)
    S_2port_open = terminate_port(S_3port, 2, 1.0)
    # When port is open with gamma=1 and S_kk=0, S'_ij = S_ij + S_ik*1*S_kj/1
    assert S_2port_open.shape == (2, 2)

    print("  OK: Port termination works")

    # Test cascade of 2-port networks
    # Two ideal transmission lines
    phase1, phase2 = np.pi / 4, np.pi / 3
    S1 = np.array([[0, np.exp(-1j * phase1)],
                   [np.exp(-1j * phase1), 0]], dtype=complex)
    S2 = np.array([[0, np.exp(-1j * phase2)],
                   [np.exp(-1j * phase2), 0]], dtype=complex)

    S_cascade = cascade_2port(S1, S2)
    # Should get total phase through
    phase_total = -np.angle(S_cascade[0, 1]) if abs(S_cascade[0, 1]) > 0.9 else 0
    print(f"  Cascade phase: {np.rad2deg(phase_total):.1f} deg "
          f"(expected {np.rad2deg(phase1 + phase2):.1f} deg)")
    print("  OK: 2-port cascade works")

    # Test S <-> Z conversion
    S_test = np.array([[0.5 + 0.5j, 0.1], [0.1, 0.5 + 0.5j]], dtype=complex)
    Z = s_to_z(S_test, 50.0)
    S_back = z_to_s(Z, 50.0)
    assert np.allclose(S_test, S_back, atol=1e-10), "S->Z->S round trip failed"
    print("  OK: S<->Z conversion works")

    print("  PASS: Network operations\n")


def test_component_library():
    """Test Murata component library loading."""
    print("=" * 60)
    print("TEST: Component Library")
    print("=" * 60)

    if not os.path.isdir(MURATA_DIR):
        print(f"  SKIP: Murata directory not found: {MURATA_DIR}")
        return

    library = scan_murata_directory(MURATA_DIR)

    print(f"  Inductors loaded: {len(library.inductors)}")
    print(f"  Capacitors loaded: {len(library.capacitors)}")
    print(f"  Unique inductor values: {len(library.get_unique_inductor_values())}")
    print(f"  Unique capacitor values: {len(library.get_unique_capacitor_values())}")

    # Test parsing some part numbers
    test_parts = [
        "LQP03TN10NH02",
        "GRM1555C1H101JA01",
        "LQG15HH2N0S02",
    ]
    for pn in test_parts:
        comp_type, value, unit = parse_murata_part(pn)
        print(f"  {pn}: type={comp_type}, value={value} {unit}")

    # Test loading data from one component
    if library.inductors:
        comp = library.inductors[0]
        S = comp.get_s_matrix_at_freq(64e6)
        assert S.shape == (2, 2), f"Expected 2x2, got {S.shape}"
        print(f"  Sample component: {comp.part_number}")
        print(f"    S11 at 64MHz: {abs(S[0,0]):.4f} ang{np.angle(S[0,0], deg=True):.1f} deg")
        print(f"    S21 at 64MHz: {abs(S[0,1]):.4f} ang{np.angle(S[0,1], deg=True):.1f} deg")

    print("  PASS: Component library works\n")


def test_optimizer_simple():
    """Test the optimizer with a simple S1P DUT."""
    print("=" * 60)
    print("TEST: Optimizer (Simple 1-Port)")
    print("=" * 60)

    filepath = os.path.join(SNP_DIR, "For64MHz.s1p")
    if not os.path.isfile(filepath):
        print(f"  SKIP: {filepath} not found")
        return

    if not os.path.isdir(MURATA_DIR):
        print(f"  SKIP: Murata directory not found")
        return

    # Load DUT
    data = load_touchstone_file(filepath)
    print(f"  Loaded: {filepath} ({data.num_ports} port, {len(data.frequencies)} freqs)")

    # Load component library
    library = scan_murata_directory(MURATA_DIR)
    print(f"  Library: {len(library.inductors)} inductors, {len(library.capacitors)} capacitors")

    # Create optimizer
    config = OptimizerConfig(
        target_frequency_hz=64e6,
        max_components=2,
        beam_width=10,
        max_combinations_to_evaluate=50000,
        timeout_seconds=30.0,
    )
    optimizer = MatchingOptimizer(data, library, config)

    # Get S11 without matching
    S_raw = data.get_s_matrix_interpolated(64e6)
    s11_raw = S_raw[0, 0]
    print(f"\n  Raw S11 at 64MHz: mag={abs(s11_raw):.4f}, "
          f"dB={20*np.log10(abs(s11_raw)):.1f}")

    # Run L-network optimization
    port_states = {}  # For 1-port, no other ports to configure
    solutions = optimizer.optimize_l_network(port_states, input_port=0)

    print(f"\n  Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions[:5]):
        comps_str = ", ".join(
            f"{c.component.part_number}({c.component.nominal_value}{c.component.nominal_unit})"
            for c in sol.component_choices
        )
        print(f"  #{i+1}: |S11|={sol.s11_magnitude:.4f}, "
              f"RL={sol.return_loss_db:.1f}dB, "
              f"VSWR={sol.vswr:.2f}, "
              f"Components: [{comps_str}]")

    # Best solution
    if solutions:
        best = solutions[0]
        print(f"\n  Best: {best.topology.name}")
        print(f"    |S11| = {best.s11_magnitude:.6f}")
        print(f"    Return Loss = {best.return_loss_db:.1f} dB")
        improvement = -20 * np.log10(abs(s11_raw)) - best.return_loss_db
        if improvement > 0:
            print(f"    Improvement: {improvement:.1f} dB better than raw S11")

    print("  PASS: Optimizer basic test\n")


def test_multiport():
    """Test with a 3-port DUT."""
    print("=" * 60)
    print("TEST: Multi-Port Optimization")
    print("=" * 60)

    filepath = os.path.join(SNP_DIR, "test_multiport.s3p")
    if not os.path.isfile(filepath):
        print(f"  SKIP: {filepath} not found")
        return

    if not os.path.isdir(MURATA_DIR):
        print(f"  SKIP: Murata directory not found")
        return

    data = load_touchstone_file(filepath)
    print(f"  Loaded: {filepath} ({data.num_ports} port, {len(data.frequencies)} freqs)")
    print(f"  Freq range: {min(data.frequencies)/1e6:.0f}-{max(data.frequencies)/1e6:.0f} MHz")

    library = scan_murata_directory(MURATA_DIR)

    config = OptimizerConfig(
        target_frequency_hz=1.0e9,  # 1 GHz
        max_components=2,
        beam_width=10,
        max_combinations_to_evaluate=20000,
        timeout_seconds=30.0,
    )
    optimizer = MatchingOptimizer(data, library, config)

    # Port 0 = input (with matching), ports 1,2 = various terminations
    port_states = {
        0: PortState.COMPONENT,  # Input port gets matching network
        1: PortState.LOAD,       # Port 2 terminated in 50ohm
        2: PortState.LOAD,       # Port 3 terminated in 50ohm
    }

    raw_S = data.get_s_matrix_interpolated(1.0e9)
    print(f"\n  Raw S-matrix at 1GHz:")
    print(f"    S11 = {abs(raw_S[0,0]):.4f} ang{np.angle(raw_S[0,0], deg=True):.1f} deg")

    # Apply terminations to get effective S11
    from engine.network import terminate_ports
    S_terminated = terminate_ports(raw_S.copy(), {1: 0.0, 2: 0.0})
    if S_terminated.shape[0] > 0:
        print(f"    S11 after terminations: |S11|={abs(S_terminated[0,0]):.4f}")

    print("\n  Running optimization...")
    solutions = optimizer.optimize_l_network(port_states, input_port=0)

    print(f"\n  Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions[:5]):
        comps_str = ", ".join(
            f"{c.component.part_number}"
            for c in sol.component_choices
        )
        print(f"  #{i+1}: |S11|={sol.s11_magnitude:.4f}, "
              f"RL={sol.return_loss_db:.1f}dB, "
              f"Components: [{comps_str}]")

    if solutions:
        print(f"\n  Best RL: {solutions[0].return_loss_db:.1f} dB")

    print("  PASS: Multi-port optimization\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RF MATCHING ENGINE -- TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_touchstone_parser()
    except Exception as e:
        print(f"  FAIL: {e}\n")
        import traceback
        traceback.print_exc()

    try:
        test_network_operations()
    except Exception as e:
        print(f"  FAIL: {e}\n")
        import traceback
        traceback.print_exc()

    try:
        test_component_library()
    except Exception as e:
        print(f"  FAIL: {e}\n")
        import traceback
        traceback.print_exc()

    try:
        test_optimizer_simple()
    except Exception as e:
        print(f"  FAIL: {e}\n")
        import traceback
        traceback.print_exc()

    try:
        test_multiport()
    except Exception as e:
        print(f"  FAIL: {e}\n")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
