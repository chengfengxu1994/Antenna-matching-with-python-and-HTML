"""
Scoring unit tests.

Verifies:
  - All efficiency values use 0-1 ratio internally (never %)
  - S11 values use magnitude (0-1), not dB
  - Coupling loss uses ratio (0-1)
  - Component loss uses ratio (0-1)
  - Score is in [0, 1] range
  - Scoring is consistent: better match → higher score

Standard formula:
  accepted_efficiency  = 1 - |S11|^2          [0-1]
  coupling_loss        = sum_{j!=i} |Sji|^2   [0-1]
  radiated_efficiency  = accepted - coupling   [0-1]
  total_efficiency     = radiated - comp_loss  [0-1]
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

import numpy as np
from engine.scoring import (
    efficiency_chain,
    compute_accepted_efficiency,
    compute_coupling_loss,
    compute_radiated_efficiency,
    compute_total_efficiency,
    score_single_port,
    score_multi_port,
    estimate_component_loss_power,
    estimate_total_component_loss,
    get_objective_preset,
)


def test_efficiency_chain_units():
    """Verify all efficiency values are 0-1 ratio."""
    result = efficiency_chain(
        s11_mag=0.1,           # |S11| = 0.1 → accepted = 0.99
        coupling_loss=0.05,    # 5% coupled
        radiation_efficiency=0.8,  # 80% antenna efficiency
        component_loss=0.02,   # 2% lost in components
    )
    assert 0 <= result['accepted_efficiency'] <= 1.0, f"accepted not in [0,1]: {result['accepted_efficiency']}"
    assert 0 <= result['coupling_loss'] <= 1.0, f"coupling not in [0,1]: {result['coupling_loss']}"
    assert 0 <= result['radiated_efficiency'] <= 1.0, f"radiated not in [0,1]: {result['radiated_efficiency']}"
    assert 0 <= result['total_efficiency'] <= 1.0, f"total not in [0,1]: {result['total_efficiency']}"
    print(f"  efficiency_chain units OK: {result}")


def test_efficiency_chain_values():
    """Verify efficiency chain math with known values."""
    # |S11| = 0 → perfect match
    e = efficiency_chain(0.0, 0.0, 1.0, 0.0)
    assert e['accepted_efficiency'] == 1.0, f"Perfect match should have 100% accepted"
    assert e['total_efficiency'] == 1.0, f"Perfect match should have 100% total"

    # |S11| = 1.0 → total reflection
    e = efficiency_chain(1.0, 0.0, 1.0, 0.0)
    assert e['accepted_efficiency'] == 0.0, f"Full reflection should have 0% accepted"
    assert e['total_efficiency'] == 0.0, f"Full reflection should have 0% total"

    # Known calculation: |S11|=0.1 → accepted = 0.99, coupling=0.05 → radiated=0.94
    # component_loss=0.02, η_rad=0.8 → total = 0.8*0.94 - 0.02 = 0.732
    e = efficiency_chain(0.1, 0.05, 0.8, 0.02)
    assert abs(e['accepted_efficiency'] - 0.99) < 1e-10
    assert abs(e['radiated_efficiency'] - 0.94) < 1e-10
    assert abs(e['total_efficiency'] - 0.732) < 1e-10
    print(f"  efficiency_chain values OK")


def test_compute_accepted_efficiency():
    """Verify accepted efficiency formula: 1 - |S11|^2."""
    assert compute_accepted_efficiency(0.0) == 1.0
    assert compute_accepted_efficiency(1.0) == 0.0
    assert abs(compute_accepted_efficiency(0.5) - 0.75) < 1e-10
    assert abs(compute_accepted_efficiency(0.316) - 0.9001) < 0.001
    print(f"  accepted_efficiency OK")


def test_compute_coupling_loss():
    """Verify coupling loss sums |Sji|^2 for j != i."""
    # 3-port: S[:,:,0] is port0's column = [S00, S10, S20]
    S = np.array([
        [0.1+0j, 0.2+0j, 0.3+0j],
        [0.4+0j, 0.5+0j, 0.6+0j],
        [0.7+0j, 0.8+0j, 0.9+0j],
    ], dtype=complex)
    # For port 0: coupled = |S10|^2 + |S20|^2 = 0.4^2 + 0.7^2 = 0.16 + 0.49 = 0.65
    cl = compute_coupling_loss(S, 0)
    assert abs(cl - 0.65) < 1e-10, f"coupling_loss port 0: {cl} != 0.65"
    # For port 1: coupled = |S01|^2 + |S21|^2 = 0.2^2 + 0.8^2 = 0.04 + 0.64 = 0.68
    cl = compute_coupling_loss(S, 1)
    assert abs(cl - 0.68) < 1e-10, f"coupling_loss port 1: {cl} != 0.68"
    print(f"  coupling_loss OK")


def test_score_single_port_units():
    """Verify single-port score is in [0, 1]."""
    preset = get_objective_preset('balanced')
    for avg_eff in [0.0, 0.25, 0.5, 0.75, 1.0]:
        score = score_single_port(np.array([avg_eff]), 2, preset)
        assert 0 <= score <= 1.0, f"Score out of range [0,1] for avg_eff={avg_eff}: {score}"
    print(f"  single_port score range OK")


def test_score_single_port_ordering():
    """Verify better efficiency → higher score."""
    preset = get_objective_preset('average_efficiency')
    # Same component count, different efficiencies
    low_score = score_single_port(np.array([0.3, 0.4, 0.5]), 2, preset)
    high_score = score_single_port(np.array([0.8, 0.85, 0.9]), 2, preset)
    assert high_score > low_score, f"Higher efficiency should give higher score: {high_score} <= {low_score}"

    # More components with same efficiency → lower or equal score
    few_comps = score_single_port(np.array([0.8, 0.85, 0.9]), 1, preset)
    many_comps = score_single_port(np.array([0.8, 0.85, 0.9]), 6, preset)
    assert few_comps >= many_comps, f"More components should not increase score: {few_comps} < {many_comps}"
    print(f"  single_port ordering OK")


def test_score_multi_port_units():
    """Verify multi-port score is in [0, 1]."""
    preset = get_objective_preset('balanced')
    for avg in [0.2, 0.5, 0.8]:
        per_port = {0: np.array([avg, avg+0.05]), 1: np.array([avg-0.1, avg])}
        per_coupling = {0: 0.03, 1: 0.08}
        score = score_multi_port(per_port, per_coupling, 3, preset)
        assert 0 <= score <= 1.0, f"Multi-port score out of range: {score}"
    print(f"  multi_port score range OK")


def test_score_multi_port_ordering():
    """Verify better coupling → higher multi-port score."""
    preset = get_objective_preset('low_coupling')
    effs = {0: np.array([0.8, 0.85]), 1: np.array([0.7, 0.75])}

    low_coupling = score_multi_port(effs, {0: 0.01, 1: 0.02}, 4, preset)
    high_coupling = score_multi_port(effs, {0: 0.3, 1: 0.4}, 4, preset)
    assert low_coupling >= high_coupling, f"Lower coupling should give >= score: {low_coupling} < {high_coupling}"

    print(f"  multi_port ordering OK")


def test_component_loss_estimates():
    """Verify component loss estimation."""
    # Perfect component (lossless): S11=0, S21=1
    perfect = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    loss = estimate_component_loss_power(perfect)
    assert loss == 0.0, f"Lossless component should have 0 loss: {loss}"

    # Lossy component: S11=0, S21=0.9 → |S21|^2=0.81, loss=0.19
    lossy = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=complex)
    loss = estimate_component_loss_power(lossy)
    assert abs(loss - 0.19) < 1e-10, f"Lossy component: {loss} != 0.19"

    # Total loss with multiple components
    total = estimate_total_component_loss([
        (perfect, 'series'),    # 0 loss, full power
        (lossy, 'shunt'),       # 0.19 loss, 0.3 ratio
    ])
    expected = 0.19 * 0.3  # 0.057
    assert abs(total - expected) < 1e-10, f"Total loss: {total} != {expected}"
    print(f"  component_loss OK")


def test_score_consistent_with_efficiency():
    """Verify the system score follows efficiency changes monotonically."""
    preset = get_objective_preset('balanced')

    # Scenario A: good efficiency, low coupling
    effs_a = {0: np.array([0.9, 0.92, 0.88]), 1: np.array([0.85, 0.87, 0.83])}
    coup_a = {0: 0.02, 1: 0.03}
    score_a = score_multi_port(effs_a, coup_a, 4, preset)

    # Scenario B: poor efficiency, high coupling
    effs_b = {0: np.array([0.4, 0.35, 0.38]), 1: np.array([0.3, 0.28, 0.32])}
    coup_b = {0: 0.2, 1: 0.25}
    score_b = score_multi_port(effs_b, coup_b, 4, preset)

    assert score_a > score_b, f"Good scenario should score higher: {score_a} <= {score_b}"
    print(f"  cross-scenario consistency OK (good={score_a:.3f}, bad={score_b:.3f})")


if __name__ == "__main__":
    print("Scoring unit tests...")
    test_efficiency_chain_units()
    test_efficiency_chain_values()
    test_compute_accepted_efficiency()
    test_compute_coupling_loss()
    test_score_single_port_units()
    test_score_single_port_ordering()
    test_score_multi_port_units()
    test_score_multi_port_ordering()
    test_component_loss_estimates()
    test_score_consistent_with_efficiency()
    print("\nAll scoring tests PASSED")
