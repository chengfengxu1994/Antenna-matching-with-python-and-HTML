import numpy as np
import pytest

from rfmatch_core import (
    Band,
    Branch,
    CircuitTopology,
    MicrostripLineModel,
    MicrostripVariation,
    Objective,
    PCBSubstrate,
    Problem,
    ToleranceModel,
    YieldCriteria,
    microstrip_properties,
    monte_carlo_yield,
)


def _line(**tolerances):
    substrate = PCBSubstrate(
        "FR-4", 4.5, 1.6e-3, loss_tangent=0.02,
        copper_thickness_m=35e-6,
    )
    return MicrostripLineModel.from_electrical_design(
        "MS1", substrate, 50.0, 45.0, 2.45e9, 0.1e-3, 10e-3,
        **tolerances,
    )


def test_scalar_microstrip_tolerance_scales_trace_width_not_length():
    line = _line()
    frequency = 2.45e9
    varied = line.properties_at(frequency, 1.1)
    expected = microstrip_properties(line.substrate, line.width_m * 1.1, frequency)
    assert varied.characteristic_impedance_ohm == pytest.approx(
        expected.characteristic_impedance_ohm, abs=1e-12
    )
    propagation = line.propagation_length(frequency, 1.1)
    assert propagation.imag == pytest.approx(
        expected.phase_constant_rad_per_m * line.length_m, abs=1e-12
    )


def test_structured_microstrip_variation_changes_all_manufacturing_variables():
    line = _line()
    frequency = 2.45e9
    variation = MicrostripVariation(1.1, 0.98, 1.05, 0.96)
    properties = line.properties_at(frequency, variation)
    varied_substrate = PCBSubstrate(
        line.substrate.name,
        line.substrate.relative_permittivity * 0.96,
        line.substrate.height_m * 1.05,
        line.substrate.loss_tangent,
        line.substrate.copper_thickness_m,
        line.substrate.copper_resistivity_ohm_m,
        line.substrate.copper_roughness_rms_m,
    )
    expected = microstrip_properties(
        varied_substrate, line.width_m * 1.1, frequency
    )
    assert properties.characteristic_impedance_ohm == pytest.approx(
        expected.characteristic_impedance_ohm, abs=1e-12
    )
    assert line.propagation_length(frequency, variation).imag == pytest.approx(
        expected.phase_constant_rad_per_m * line.length_m * 0.98, abs=1e-12
    )


def test_microstrip_monte_carlo_is_reproducible_and_reports_explicit_variables():
    frequencies = np.asarray([2.4e9, 2.45e9, 2.5e9])
    problem = Problem(
        frequencies,
        np.zeros((len(frequencies), 1, 1), dtype=complex),
        {0: (Band(2.4e9, 2.5e9),)},
    )
    line = _line(
        width_tolerance=0.10,
        length_tolerance=0.02,
        substrate_height_tolerance=0.05,
        relative_permittivity_tolerance=0.04,
    )
    topology = CircuitTopology(
        ("p1",), (Branch("board_line", "p1", "dut1", line),), ("dut1",)
    )
    kwargs = dict(
        problem=problem,
        topology=topology,
        criteria=YieldCriteria(),
        samples=20,
        seed=17,
        distribution="uniform",
        tolerance_model=ToleranceModel(batch_correlation=0.35),
    )
    first = monte_carlo_yield(**kwargs)
    second = monte_carlo_yield(**kwargs)
    np.testing.assert_array_equal(first.sample_scores_db, second.sample_scores_db)
    assert set(first.worst_sample) == {
        "board_line.trace_width",
        "board_line.physical_length",
        "board_line.substrate_height",
        "board_line.relative_permittivity",
    }
    assert 0.9 <= first.worst_sample["board_line.trace_width"] <= 1.1
    assert 0.98 <= first.worst_sample["board_line.physical_length"] <= 1.02
    assert first.variation_model["microstrip_scope"] == (
        "one_shared_manufactured_board_draw_across_states"
    )
