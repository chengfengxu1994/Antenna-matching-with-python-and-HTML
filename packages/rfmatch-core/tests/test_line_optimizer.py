import numpy as np
import pytest

from rfmatch_core import (
    Band,
    LineSearchConfig,
    MicrostripDesignRules,
    Objective,
    OptimizationCancelled,
    Problem,
    PCBSubstrate,
    TransmissionLineOptimizer,
    TransmissionLineModel,
    ModelPlacement,
    TransmissionLineStubModel,
)


def _single_frequency_problem(gamma: complex) -> Problem:
    frequency = 1.0e9
    return Problem(
        np.asarray([frequency]),
        np.asarray([[[gamma]]], dtype=complex),
        {0: (Band(frequency, frequency),)},
        50.0,
    )


def test_optimizer_recovers_quarter_wave_transform():
    load_ohm = 100.0
    gamma = (load_ohm - 50.0) / (load_ohm + 50.0)
    optimizer = TransmissionLineOptimizer(
        _single_frequency_problem(gamma), 0, 1.0e9,
        Objective(within_band_average_weight=0.0),
        LineSearchConfig(
            characteristic_impedance_min_ohm=60.0,
            characteristic_impedance_max_ohm=80.0,
            electrical_length_min_deg=70.0,
            electrical_length_max_deg=110.0,
            topologies=("through_line",),
            restarts=8,
            iterations=40,
            seed=11,
        ),
    )
    result = optimizer.optimize()
    best = result.best
    reflection = abs(best.metrics["s_parameters"][0, 0, 0])
    assert reflection < 2e-4
    assert best.characteristic_impedance_ohm == pytest.approx(np.sqrt(5000.0), abs=0.02)
    assert best.line_length_deg == pytest.approx(90.0, abs=0.02)
    assert best.metrics["maximum_power_balance_error"] < 1e-12


def test_optimizer_recovers_open_stub_susceptance_cancellation():
    dut_admittance = 0.02 - 0.02j
    dut_impedance = 1.0 / dut_admittance
    gamma = (dut_impedance - 50.0) / (dut_impedance + 50.0)
    result = TransmissionLineOptimizer(
        _single_frequency_problem(gamma), 0, 1.0e9,
        config=LineSearchConfig(
            characteristic_impedance_min_ohm=49.0,
            characteristic_impedance_max_ohm=51.0,
            electrical_length_min_deg=30.0,
            electrical_length_max_deg=60.0,
            topologies=("open_stub",),
            restarts=6,
            iterations=40,
            seed=7,
        ),
    ).optimize()
    best = result.best
    assert abs(best.metrics["s_parameters"][0, 0, 0]) < 2e-4
    assert best.characteristic_impedance_ohm == pytest.approx(50.0, abs=0.05)
    assert best.line_length_deg == pytest.approx(45.0, abs=0.05)


def test_two_element_names_describe_connector_to_dut_order():
    optimizer = TransmissionLineOptimizer(
        _single_frequency_problem(0j), 0, 1.0e9,
        config=LineSearchConfig(
            topologies=("connector_line_open_stub_dut",),
            restarts=1, iterations=1,
        ),
    )
    candidate = optimizer.optimize().best
    # Stored placement order is DUT-outward, hence stub then through line.
    assert isinstance(candidate.placements[0].model, TransmissionLineStubModel)
    assert candidate.placements[1].connection == "series"


def test_optimizer_honors_cooperative_cancellation():
    optimizer = TransmissionLineOptimizer(
        _single_frequency_problem(0.3 + 0.1j), 0, 1.0e9,
        config=LineSearchConfig(topologies=("through_line",)),
        cancel_check=lambda: True,
    )
    with pytest.raises(OptimizationCancelled):
        optimizer.optimize()


def test_complexity_objective_penalizes_line_plus_stub_topology():
    result = TransmissionLineOptimizer(
        _single_frequency_problem(0j), 0, 1.0e9,
        Objective(complexity_penalty_db=0.5),
        LineSearchConfig(
            characteristic_impedance_min_ohm=49.0,
            characteristic_impedance_max_ohm=51.0,
            electrical_length_min_deg=0.001,
            electrical_length_max_deg=0.002,
            topologies=("through_line", "connector_line_open_stub_dut"),
            restarts=1, iterations=1, keep=5,
        ),
    ).optimize()
    by_topology = {item.topology: item for item in result.candidates}
    assert (
        by_topology["through_line"].score_db
        - by_topology["connector_line_open_stub_dut"].score_db
    ) > 0.49


def test_optimizer_scores_manufacturable_microstrip_with_physical_loss():
    substrate = PCBSubstrate(
        "FR4", 4.5, 1.6e-3, loss_tangent=0.02,
        copper_thickness_m=35e-6,
    )
    load_ohm = 100.0
    gamma = (load_ohm - 50.0) / (load_ohm + 50.0)
    result = TransmissionLineOptimizer(
        _single_frequency_problem(gamma), 0, 1.0e9,
        config=LineSearchConfig(
            characteristic_impedance_min_ohm=60.0,
            characteristic_impedance_max_ohm=80.0,
            electrical_length_min_deg=70.0,
            electrical_length_max_deg=110.0,
            topologies=("through_line",), restarts=8, iterations=40,
            microstrip_rules=MicrostripDesignRules(substrate, 0.1e-3, 10e-3),
        ),
    ).optimize()
    best = result.best
    component = best.components()[0]
    assert component["physical_model"] == "microstrip_hammerstad_kirschning_wheeler"
    assert 0.1e-3 <= component["width_m"] <= 10e-3
    assert component["length_m"] > 0
    assert component["conductor_loss_db"] > 0
    assert component["dielectric_loss_db"] > 0
    assert best.metrics["maximum_power_balance_error"] < 1e-12


def test_optimizer_cascades_fixed_measured_layout_block_in_declared_order():
    frequency = 1.0e9
    layout = TransmissionLineModel(
        "connector_launch.s2p", 50.0, 12.0, frequency, attenuation_db=1.0
    ).as_s2p_model([frequency])
    result = TransmissionLineOptimizer(
        _single_frequency_problem(0j), 0, frequency,
        config=LineSearchConfig(
            characteristic_impedance_min_ohm=49.0,
            characteristic_impedance_max_ohm=51.0,
            electrical_length_min_deg=1.0,
            electrical_length_max_deg=2.0,
            topologies=("through_line",), restarts=1, iterations=2,
            fixed_dut_side=(ModelPlacement("series", 0, layout),),
        ),
    ).optimize()
    best = result.best
    components = best.components()
    assert components[0]["comp_type"] == "layout_s2p"
    assert components[0]["part_number"] == "connector_launch.s2p"
    assert components[1]["comp_type"] == "transmission_line"
    assert best.metrics["total_efficiency"][0, 0] == pytest.approx(10 ** (-0.1), abs=1e-10)
    assert best.metrics["component_loss"][0, 0] == pytest.approx(1 - 10 ** (-0.1), abs=1e-10)
    assert best.metrics["maximum_power_balance_error"] < 1e-12
