from pathlib import Path

import numpy as np
import pytest

from rfmatch_core import (
    Band,
    Candidate,
    Element,
    LumpedLossModel,
    Objective,
    Problem,
    load_touchstone,
)
from rfmatch_core.evaluator import evaluate_lumped_physical
from rfmatch_core.switch_golden import load_switch_export_csv


def test_real_optenni_four_band_network_reproduces_native_export():
    root = Path(__file__).resolve().parents[3] / "benchmarks" / "optenni_exports"
    export_path = root / "optimization_settings_4bands_default_best_plot.txt"
    if not export_path.exists():
        pytest.skip("Optenni four-band baseline is not available")
    dut = load_touchstone(root / "optimization_settings_original.s1p")
    exported = load_switch_export_csv(
        export_path, default_configuration="4 bands default best"
    )
    assert len(exported) == 531
    problem = Problem(
        dut.frequencies_hz,
        dut.s_parameters,
        {0: (
            Band(1.0e9, 1.1e9),
            Band(1.7e9, 2.0e9),
            Band(2.9e9, 3.1e9),
            Band(4.8e9, 5.0e9),
        )},
        dut.z0,
    )
    # Optenni topology PLSCSLPCSLPC, listed from the DUT outwards. Values are
    # recovered from the 531-point native curve because the UI rounds them.
    candidate = Candidate([
        Element("shunt", "L", 0, 32.0897021954055e-9),
        Element("series", "C", 0, 0.601663035233869e-12),
        Element("series", "L", 0, 6.43920991110147e-9),
        Element("shunt", "C", 0, 0.559175118349776e-12),
        Element("series", "L", 0, 4.63296894532993e-9),
        Element("shunt", "C", 0, 0.803532270134045e-12),
    ])
    evaluated = evaluate_lumped_physical(
        problem,
        candidate,
        Objective(),
        LumpedLossModel(
            inductor_q=30.0,
            inductor_q_reference_hz=1e9,
            capacitor_esr=0.4,
        ),
    )
    predicted_db = 20.0 * np.log10(np.maximum(
        np.abs(evaluated.metrics["s_parameters"][:, 0, 0]), 1e-15
    ))
    exported_db = np.asarray([point.s11_db for point in exported])
    error_db = predicted_db - exported_db
    assert float(np.sqrt(np.mean(error_db ** 2))) < 1e-4
    assert float(np.max(np.abs(error_db))) < 1e-3
    assert evaluated.metrics["maximum_power_balance_error"] < 1e-12

    frequencies = np.asarray([point.frequency_hz for point in exported])
    efficiency_db = 10.0 * np.log10(np.asarray([
        point.total_efficiency for point in exported
    ]))
    expected = ((-3.3, -2.8), (-3.0, -1.4), (-3.2, -2.9), (-3.3, -2.7))
    for band, displayed in zip(problem.bands_by_port[0], expected):
        values = efficiency_db[band.mask(frequencies)]
        assert round(float(np.min(values)), 1) == displayed[0]
        # This explicitly locks Optenni's arithmetic-in-dB average semantics.
        assert round(float(np.mean(values)), 1) == displayed[1]
