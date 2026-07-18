from pathlib import Path

import numpy as np
import pytest

from rfmatch_core import (
    Band,
    FrequencyConfiguration,
    OptimizationCancelled,
    SwitchMeasuredComponentOptimizer,
    SwitchSearchConfig,
    SwitchTunableOptimizer,
    SwitchTunableProblem,
    load_mdif,
    load_coilcraft_0402cs_catalog,
    load_murata_gjm15_catalog,
    load_touchstone,
    standard_switch_input_topologies,
)


def _official_problem(root: Path, fixed_states: bool = False) -> tuple[SwitchTunableProblem, object]:
    dut = load_touchstone(root / "Switch_Tuner_Tutorial.s1p")
    mask = np.zeros(len(dut.frequencies_hz), dtype=bool)
    for start, stop in ((704e6, 746e6), (791e6, 862e6), (880e6, 960e6), (1920e6, 2170e6)):
        mask |= (dut.frequencies_hz >= start) & (dut.frequencies_hz <= stop)
    problem = SwitchTunableProblem(
        dut.frequencies_hz[mask],
        dut.s_parameters[mask, 0, 0],
        (
            FrequencyConfiguration("Set 1", {0: (Band(704e6, 746e6), Band(1920e6, 2170e6))}),
            FrequencyConfiguration("Set 2", {0: (Band(791e6, 862e6), Band(1920e6, 2170e6))}),
            FrequencyConfiguration("Set 3", {0: (Band(880e6, 960e6), Band(1920e6, 2170e6))}),
        ),
        state_options_by_configuration={
            "Set 1": ("100",),
            "Set 2": ("010",),
            "Set 3": ("001",),
        } if fixed_states else {},
    )
    return problem, load_mdif(root / "SP3T_ideal.mdif")


def test_standard_switch_input_topologies_cover_ordered_zero_to_two_elements():
    topologies = standard_switch_input_topologies(2)
    assert len(topologies) == 21
    assert () in topologies
    assert (("series", "L"),) in topologies
    assert (("shunt", "C"), ("series", "L")) in topologies
    assert (("series", "L"), ("shunt", "C")) in topologies
    assert standard_switch_input_topologies(0) == ((),)


def test_switch_optimizer_validates_branch_count():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    problem, switch = _official_problem(root)
    optimizer = SwitchTunableOptimizer(problem, switch, config=SwitchSearchConfig(restarts=1, iterations=1))
    with pytest.raises(ValueError, match="each switch throw"):
        optimizer.optimize(branch_topologies=[("C", "C")])


def test_official_sp3t_joint_search_is_deterministic_and_selects_three_states():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    problem, switch = _official_problem(root)
    config = SwitchSearchConfig(restarts=2, iterations=8, keep=4, seed=3)
    first = SwitchTunableOptimizer(problem, switch, config=config).optimize(
        branch_topologies=[("C", "C", "C")],
        input_topologies=[(("series", "L"),)],
    )
    second = SwitchTunableOptimizer(problem, switch, config=config).optimize(
        branch_topologies=[("C", "C", "C")],
        input_topologies=[(("series", "L"),)],
    )
    assert first.evaluations == second.evaluations
    assert first.state_precomputations == 8
    assert first.best.state_by_configuration == second.best.state_by_configuration
    np.testing.assert_allclose(first.best.score_db, second.best.score_db, atol=1e-12)
    np.testing.assert_allclose(
        [item.value for item in first.best.branch_reactances],
        [item.value for item in second.best.branch_reactances],
        rtol=1e-12,
    )
    assert len(set(first.best.state_by_configuration.values())) == 3


def test_official_sp3t_search_respects_tutorial_state_assignments():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    problem, switch = _official_problem(root, fixed_states=True)
    result = SwitchTunableOptimizer(
        problem,
        switch,
        config=SwitchSearchConfig(restarts=2, iterations=8, keep=2, seed=5),
    ).optimize(
        branch_topologies=[("C", "C", "C")],
        input_topologies=[(("series", "L"),)],
    )
    assert result.best.state_by_configuration == {"Set 1": "100", "Set 2": "010", "Set 3": "001"}
    assert np.isfinite(result.best.score_db)


def test_official_sp2t_search_discovers_state_mapping_and_near_tutorial_values():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    problem, _ = _official_problem(root)
    switch = load_mdif(root / "tutorial_SP2T.mdif")
    result = SwitchTunableOptimizer(
        problem, switch, config=SwitchSearchConfig(restarts=1, iterations=8, keep=2)
    ).optimize(
        branch_topologies=[("C", "C")],
        input_topologies=[(("series", "L"),)],
    )
    best = result.best
    assert best.state_by_configuration == {
        "Set 1": "all on", "Set 2": "RFC-RF1", "Set 3": "RFC-RF2"
    }
    capacitances_pf = [item.value * 1e12 for item in best.branch_reactances]
    inductance_nh = best.input_reactances[0].value * 1e9
    assert 1.0 < capacitances_pf[0] < 1.5
    assert 0.7 < capacitances_pf[1] < 1.1
    assert 10.0 < inductance_nh < 14.0


def test_switch_search_reports_progress_and_cancels_cooperatively():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    if not root.exists():
        pytest.skip("Optenni switch tutorial data is not installed")
    problem, switch = _official_problem(root)
    progress = []
    checks = {"count": 0}

    def cancel():
        checks["count"] += 1
        return checks["count"] > 8

    optimizer = SwitchTunableOptimizer(
        problem,
        switch,
        config=SwitchSearchConfig(restarts=2, iterations=20),
        progress_callback=progress.append,
        cancel_check=cancel,
    )
    with pytest.raises(OptimizationCancelled):
        optimizer.optimize(input_topologies=[(("series", "L"),)])
    assert progress[0]["stage"] == "switch_state_precompute"
    assert any(item["stage"] == "switch_topology_search" for item in progress)


def test_official_sp2t_measured_refinement_recalls_tutorial_part_values():
    root = Path(
        r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials\10 - Tunable antennas\10.6 Impedance tuning using a switch"
    )
    component_root = Path(r"C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary")
    if not root.exists() or not component_root.exists():
        pytest.skip("licensed Optenni switch/component inputs are not installed")
    problem, _ = _official_problem(root)
    optimizer = SwitchTunableOptimizer(
        problem,
        load_mdif(root / "tutorial_SP2T.mdif"),
        config=SwitchSearchConfig(restarts=2, iterations=10, keep=8),
    )
    ideal = optimizer.optimize(input_topologies=[(("series", "L"),)])
    measured = SwitchMeasuredComponentOptimizer(
        optimizer,
        load_coilcraft_0402cs_catalog(
            component_root / "Inductors" / "Coilcraft Inductors 0402cs"
        ),
        load_murata_gjm15_catalog(
            component_root / "Capacitors" / "Murata Capacitors gjm15"
        ),
        nearest_parts=3,
        ideal_seed_keep=2,
        result_keep=8,
    ).optimize(ideal.candidates)
    best = measured.best
    assert [item.value_display for item in best.branch_components] == ["1.2 pF", "0.8 pF"]
    assert [item.value_display for _, item in best.input_components] == ["13 nH"]
    assert best.state_by_configuration == {
        "Set 1": "all on", "Set 2": "RFC-RF1", "Set 3": "RFC-RF2"
    }
    assert best.metrics["average_matching_network_loss"] > 0
    assert best.metrics["maximum_power_balance_error"] < 1e-10
    assert measured.physical_evaluations == 54
