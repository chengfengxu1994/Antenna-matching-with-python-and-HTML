from pathlib import Path

import numpy as np
import pytest

from rfmatch_core import (
    Band,
    LumpedLossModel,
    MatchingOptimizer,
    Objective,
    Problem,
    SearchConfig,
    load_touchstone,
)


def test_loss_aware_synthesis_recalls_real_optenni_pcsl_solution():
    root = Path(__file__).resolve().parents[3]
    path = root / "benchmarks" / "optenni_exports" / "optimization_settings_original.s1p"
    if not path.exists():
        pytest.skip("Optenni synthesis baseline is not available")
    dut = load_touchstone(path)
    active = (dut.frequencies_hz >= 1.7e9) & (dut.frequencies_hz <= 2.5e9)
    problem = Problem(
        dut.frequencies_hz[active],
        dut.s_parameters[active],
        {0: (Band(1.7e9, 2.5e9),)},
        dut.z0,
    )
    # The six canonical two-element forms shown by this Optenni project.
    topologies = (
        (("shunt", "C", 0), ("series", "L", 0)),
        (("shunt", "L", 0), ("series", "C", 0)),
        (("series", "L", 0), ("shunt", "C", 0)),
        (("series", "C", 0), ("series", "L", 0)),
        (("shunt", "C", 0), ("shunt", "L", 0)),
        (("series", "C", 0), ("shunt", "L", 0)),
    )
    result = MatchingOptimizer(
        problem,
        Objective(),
        SearchConfig(restarts=8, iterations=25, keep=50, seed=1),
        LumpedLossModel(
            inductor_q=30.0,
            inductor_q_reference_hz=1e9,
            capacitor_esr=0.4,
        ),
    ).optimize(topologies)
    best = result.best
    assert [(item.connection, item.kind) for item in best.elements] == [
        ("shunt", "C"), ("series", "L")
    ]
    capacitance_pf = best.elements[0].value * 1e12
    inductance_nh = best.elements[1].value * 1e9
    np.testing.assert_allclose(capacitance_pf, 0.48398948153925053, rtol=0.01)
    np.testing.assert_allclose(inductance_nh, 5.915099856061325, rtol=0.01)
    assert best.score_db > -0.98
    assert best.metrics["maximum_power_balance_error"] < 1e-12
