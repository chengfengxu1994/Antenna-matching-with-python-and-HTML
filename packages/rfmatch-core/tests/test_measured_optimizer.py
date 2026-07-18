import tempfile
import unittest
from pathlib import Path

import numpy as np

from rfmatch_core import (
    Band,
    ComponentSpec,
    LazyComponentSpec,
    MeasuredComponentOptimizer,
    MeasuredCandidate,
    MeasuredPlacement,
    MeasuredSearchConfig,
    Objective,
    PORT_TOPOLOGY_PATTERNS,
    Problem,
    S2PModel,
    build_circuit_topology,
    evaluate_physical_problem,
    load_coilcraft_0402cs_catalog,
    load_coilcraft_0402hp_catalog,
    load_murata_gjm15_catalog,
    load_murata_gqm18_catalog,
)
from rfmatch_core.models import Candidate, Element
from rfmatch_core.network import apply_elements_sweep


COMPONENT_ROOT = Path(r"C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary")


def _write_through_fixture(path: Path) -> None:
    # A reciprocal 25-ohm series fixture; finite impedance avoids the singular
    # ideal-through conversion while remaining valid in series or grounded use.
    reflection = 0.2
    transmission = 0.8
    path.write_text(
        "# GHZ S RI R 50\n"
        f"1 {reflection} 0 {transmission} 0 {transmission} 0 {reflection} 0\n"
        f"2 {reflection} 0 {transmission} 0 {transmission} 0 {reflection} 0\n",
        encoding="ascii",
    )


def _write_reactive_fixture(path: Path, frequencies: np.ndarray, kind: str, value: float) -> None:
    lines = ["# HZ S RI R 50"]
    for frequency in frequencies:
        omega = 2 * np.pi * frequency
        impedance = 1j * omega * value if kind == "L" else 1 / (1j * omega * value)
        normalized = impedance / 50.0
        reflection = normalized / (2 + normalized)
        transmission = 2 / (2 + normalized)
        values = (reflection, transmission, transmission, reflection)
        lines.append(str(frequency) + " " + " ".join(f"{x.real:.16g} {x.imag:.16g}" for x in values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class MeasuredOptimizerTests(unittest.TestCase):
    def test_topology_catalog_covers_zero_one_and_observed_two_element_codes(self):
        codes = {
            "".join(("S" if connection == "series" else "P") + kind for connection, kind in pattern)
            for pattern in PORT_TOPOLOGY_PATTERNS
        }
        self.assertTrue({"", "SL", "PL", "SC", "PC"}.issubset(codes))
        self.assertTrue({"PLSL", "SLPL", "PLSC", "SLPC", "PCPL", "SCPL", "SCSL", "PCSL"}.issubset(codes))
        self.assertTrue({"PCSLPC", "PLSCPL", "SLPCSL", "SCPLSC"}.issubset(codes))
        self.assertTrue({"SLPCSLPC", "SCPLSCPL"}.issubset(codes))

    def test_no_series_network_aliases_only_its_own_external_port(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "C1.s2p"
            _write_through_fixture(model_path)
            spec = ComponentSpec("C1", "C", 1e-12, 0.05, "fixture", model_path)
            topology = build_circuit_topology(2, (MeasuredPlacement("shunt", 1, spec),))
        self.assertEqual(topology.external_nodes, ("p1", "p2"))
        self.assertEqual(topology.dut_nodes, ("p1", "p2"))
        self.assertEqual(topology.branches[0].node_a, "p2")
        self.assertIsNone(topology.branches[0].node_b)

    def test_lazy_adapter_model_loads_only_when_physical_candidate_is_evaluated(self):
        frequencies = np.array([1e9, 2e9])
        calls = []

        def load_model():
            calls.append("loaded")
            reflection, transmission = 0.2, 0.8
            matrix = np.array([
                [reflection, transmission],
                [transmission, reflection],
            ], dtype=complex)
            return S2PModel("lazy", frequencies, np.stack([matrix, matrix]))

        spec = LazyComponentSpec(
            "L_lazy", "L", 1e-9, 0.05, "database", "db:1", load_model
        )
        placement = MeasuredPlacement("series", 0, spec)
        cache = {}
        self.assertEqual(calls, [])
        first = build_circuit_topology(1, (placement,), cache)
        second = build_circuit_topology(1, (placement,), cache)
        self.assertEqual(calls, ["loaded"])
        self.assertIs(first.branches[0].model, second.branches[0].model)
        self.assertEqual(first.branches[0].model.kind, "L")
        self.assertEqual(first.branches[0].model.nominal_value, 1e-9)

    def test_tiny_search_is_deterministic_and_power_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            l_path, c_path = directory / "L1.s2p", directory / "C1.s2p"
            _write_through_fixture(l_path)
            _write_through_fixture(c_path)
            inductors = [ComponentSpec("L1", "L", 1e-9, 0.05, "fixture", l_path)]
            capacitors = [ComponentSpec("C1", "C", 1e-12, 0.05, "fixture", c_path)]
            frequencies = np.array([1e9, 2e9])
            s = np.array([[[0.2 + 0.1j]], [[0.1 + 0.05j]]])
            problem = Problem(frequencies, s, {0: [Band(1e9, 2e9)]})
            config = MeasuredSearchConfig(ideal_restarts=1, ideal_iterations=1, ideal_keep=1, nearest_parts=1, per_port_keep=2, result_keep=2, seed=9)
            first = MeasuredComponentOptimizer(problem, inductors, capacitors, Objective(), config).optimize()
            second = MeasuredComponentOptimizer(problem, inductors, capacitors, Objective(), config).optimize()
        self.assertAlmostEqual(first.best.score_db, second.best.score_db, places=12)
        self.assertEqual(
            [(x.connection, x.component.name) for x in first.best.placements],
            [(x.connection, x.component.name) for x in second.best.placements],
        )
        self.assertLess(first.best.metrics["maximum_power_balance_error"], 1e-9)

    def test_measured_search_enforces_allowed_topology_codes_and_zero_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            l_path, c_path = directory / "L1.s2p", directory / "C1.s2p"
            _write_reactive_fixture(l_path, np.array([1e9, 1.1e9]), "L", 2e-9)
            _write_reactive_fixture(c_path, np.array([1e9, 1.1e9]), "C", 2e-12)
            result = MeasuredComponentOptimizer(
                Problem(
                    np.array([1e9, 1.1e9]),
                    np.array([[[0.35 + 0.1j]], [[0.3 + 0.08j]]]),
                    {0: [Band(1e9, 1.1e9)]},
                ),
                [ComponentSpec("L1", "L", 2e-9, 0.05, "fixture", l_path)],
                [ComponentSpec("C1", "C", 2e-12, 0.05, "fixture", c_path)],
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1, ideal_iterations=1, ideal_keep=4,
                    nearest_parts=1, per_port_keep=4, result_keep=4,
                    joint_refine_seeds=0, joint_refine_passes=0,
                    max_components_per_port=2,
                    allowed_topology_codes=frozenset({"SL"}),
                    include_zero_component=False,
                ),
            ).optimize()
        self.assertTrue(result.candidates)
        self.assertEqual({candidate.topology_code for candidate in result.candidates}, {"SL"})

    def test_measured_search_enforces_different_topology_codes_per_port(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frequencies = np.array([1e9, 1.1e9])
            l_path, c_path = directory / "L1.s2p", directory / "C1.s2p"
            _write_reactive_fixture(l_path, frequencies, "L", 2e-9)
            _write_reactive_fixture(c_path, frequencies, "C", 2e-12)
            result = MeasuredComponentOptimizer(
                Problem(
                    frequencies,
                    np.array([
                        [[0.35 + 0.1j, 0.05], [0.05, 0.3 + 0.08j]],
                        [[0.32 + 0.08j, 0.04], [0.04, 0.28 + 0.06j]],
                    ]),
                    {0: [Band(1e9, 1.1e9)], 1: [Band(1e9, 1.1e9)]},
                ),
                [ComponentSpec("L1", "L", 2e-9, 0.05, "fixture", l_path)],
                [ComponentSpec("C1", "C", 2e-12, 0.05, "fixture", c_path)],
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1, ideal_iterations=1, ideal_keep=4,
                    nearest_parts=1, per_port_keep=4, result_keep=4,
                    joint_refine_seeds=0, joint_refine_passes=0,
                    max_components_per_port=1,
                    allowed_topology_codes_by_port={
                        0: frozenset({"SL"}),
                        1: frozenset({"PC"}),
                    },
                ),
            ).optimize()

        self.assertTrue(result.candidates)
        expected = {0: "SL", 1: "PC"}
        for port, candidates in result.per_port_candidates.items():
            self.assertTrue(candidates)
            self.assertEqual({candidate.topology_code for candidate in candidates}, {expected[port]})
        for candidate in result.candidates:
            by_port = {
                port: "".join(
                    ("S" if item.connection == "series" else "P") + item.component.kind
                    for item in candidate.placements if item.port == port
                )
                for port in expected
            }
            self.assertEqual(by_port, expected)

    def test_port_block_refine_beam_crosses_a_coupled_score_valley(self):
        frequencies = np.array([1e9, 1.1e9])
        problem = Problem(
            frequencies,
            np.zeros((2, 2, 2), dtype=complex),
            {0: [Band(1e9, 1.1e9)], 1: [Band(1e9, 1.1e9)]},
        )
        dummy = Path("unused.s2p")
        l1 = ComponentSpec("L1", "L", 1e-9, 0.05, "test", dummy)
        l2 = ComponentSpec("L2", "L", 2e-9, 0.05, "test", dummy)
        c1 = ComponentSpec("C1", "C", 1e-12, 0.05, "test", dummy)
        c2 = ComponentSpec("C2", "C", 2e-12, 0.05, "test", dummy)
        seed = MeasuredCandidate((
            MeasuredPlacement("series", 0, l1),
            MeasuredPlacement("shunt", 0, c1),
            MeasuredPlacement("series", 1, l1),
            MeasuredPlacement("shunt", 1, c1),
        ), 0.0)

        def score(placements):
            selected = tuple(item.component.name for item in placements)
            target_count = sum(name in {"L2", "C2"} for name in selected)
            # Changing one complete port is temporarily worse; changing both
            # ports together is the only improving basin.
            value = (
                10.0 if target_count == 4
                else -1.0 if target_count == 2
                else -2.0 if target_count > 0
                else 0.0
            )
            return MeasuredCandidate(tuple(placements), value)

        def run(beam_width):
            optimizer = MeasuredComponentOptimizer(
                problem, [l1, l2], [c1, c2], Objective(),
                MeasuredSearchConfig(
                    joint_refine_passes=2,
                    joint_refine_neighbors=2,
                    joint_refine_port_blocks=True,
                    joint_refine_beam_width=beam_width,
                ),
            )
            optimizer._evaluate = lambda _problem, placements, _objective: score(placements)
            return optimizer._coordinate_refine(seed)

        greedy = run(1)
        beamed = run(4)
        self.assertEqual(greedy[0].score_db, 0.0)
        self.assertEqual(beamed[0].score_db, 10.0)
        self.assertEqual(
            tuple(item.component.name for item in beamed[0].placements),
            ("L2", "C2", "L2", "C2"),
        )

    def test_measured_search_prunes_topologies_to_available_component_kinds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "L1.s2p"
            _write_reactive_fixture(path, np.array([1e9, 1.1e9]), "L", 2e-9)
            result = MeasuredComponentOptimizer(
                Problem(
                    np.array([1e9, 1.1e9]),
                    np.array([[[0.35 + 0.1j]], [[0.3 + 0.08j]]]),
                    {0: [Band(1e9, 1.1e9)]},
                ),
                [ComponentSpec("L1", "L", 2e-9, 0.05, "fixture", path)],
                [],
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1, ideal_iterations=1, ideal_keep=8,
                    nearest_parts=1, per_port_keep=8, result_keep=8,
                    joint_refine_seeds=0, joint_refine_passes=0,
                    max_components_per_port=2,
                ),
            ).optimize()
        self.assertTrue(any(candidate.placements for candidate in result.candidates))
        self.assertTrue(all(
            placement.component.kind == "L"
            for candidate in result.candidates for placement in candidate.placements
        ))

    def test_measured_search_reports_port_level_product_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            l_path, c_path = directory / "L1.s2p", directory / "C1.s2p"
            _write_through_fixture(l_path)
            _write_through_fixture(c_path)
            frequencies = np.array([1e9, 2e9])
            events = []
            MeasuredComponentOptimizer(
                Problem(
                    frequencies,
                    np.array([[[0.2 + 0.1j]], [[0.1 + 0.05j]]]),
                    {0: [Band(1e9, 2e9)]},
                ),
                [ComponentSpec("L1", "L", 1e-9, 0.05, "fixture", l_path)],
                [ComponentSpec("C1", "C", 1e-12, 0.05, "fixture", c_path)],
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1, ideal_iterations=1, ideal_keep=2,
                    nearest_parts=1, per_port_keep=2, result_keep=2,
                    joint_refine_seeds=0, max_components_per_port=1,
                ),
                progress_callback=events.append,
            ).optimize()

        stages = {event["stage"] for event in events}
        self.assertIn("ideal_search", stages)
        self.assertIn("measured_expansion", stages)
        measured = next(event for event in events if event["stage"] == "measured_expansion")
        self.assertEqual(measured["port"], 0)
        self.assertIn("Port 1", measured["message"])
        self.assertGreater(measured["total"], 0)

    def test_measured_search_physically_evaluates_a_six_element_ladder(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            l_path, c_path = directory / "L1.s2p", directory / "C1.s2p"
            _write_reactive_fixture(l_path, np.array([1e9, 1.1e9]), "L", 2e-9)
            _write_reactive_fixture(c_path, np.array([1e9, 1.1e9]), "C", 2e-12)
            problem = Problem(
                np.array([1e9, 1.1e9]),
                np.array([[[0.35 + 0.1j]], [[0.3 + 0.08j]]]),
                {0: [Band(1e9, 1.1e9)]},
            )
            optimizer = MeasuredComponentOptimizer(
                problem,
                [ComponentSpec("L1", "L", 2e-9, 0.05, "fixture", l_path)],
                [ComponentSpec("C1", "C", 2e-12, 0.05, "fixture", c_path)],
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1, ideal_iterations=1, ideal_keep=8,
                    nearest_parts=1, per_port_keep=20, result_keep=20,
                    joint_refine_seeds=1, joint_refine_passes=0,
                    max_components_per_port=6, topology_beam_width=2,
                ),
            )
            result = optimizer.optimize()
        six_element = [candidate for candidate in result.candidates if len(candidate.placements) == 6]
        self.assertTrue(six_element)
        self.assertLess(six_element[0].metrics["maximum_power_balance_error"], 1e-10)
        self.assertEqual({key[0] for key in optimizer.evaluation_cache}, {id(problem)})

    def test_four_component_search_evaluates_bounded_ladder_families(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            l_path, c_path = directory / "L1.s2p", directory / "C1.s2p"
            _write_through_fixture(l_path)
            _write_through_fixture(c_path)
            inductors = [ComponentSpec("L1", "L", 1e-9, 0.05, "fixture", l_path)]
            capacitors = [ComponentSpec("C1", "C", 1e-12, 0.05, "fixture", c_path)]
            frequencies = np.array([1e9, 2e9])
            problem = Problem(
                frequencies,
                np.array([[[0.2 + 0.1j]], [[0.1 + 0.05j]]]),
                {0: [Band(1e9, 2e9)]},
            )
            result = MeasuredComponentOptimizer(
                problem,
                inductors,
                capacitors,
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1,
                    ideal_iterations=1,
                    ideal_keep=32,
                    nearest_parts=1,
                    per_port_keep=32,
                    result_keep=32,
                    joint_refine_seeds=0,
                    joint_refine_passes=0,
                    seed=9,
                    max_components_per_port=4,
                ),
            ).optimize()

        searched_codes = {item.topology_code for item in result.per_port_candidates[0]}
        self.assertTrue({"SLPCSLPC", "SCPLSCPL"}.issubset(searched_codes))
        self.assertLess(result.best.metrics["maximum_power_balance_error"], 1e-9)

    def test_joint_search_enforces_component_limit_for_each_port(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            l_path, c_path = directory / "L1.s2p", directory / "C1.s2p"
            _write_through_fixture(l_path)
            _write_through_fixture(c_path)
            inductors = [ComponentSpec("L1", "L", 1e-9, 0.05, "fixture", l_path)]
            capacitors = [ComponentSpec("C1", "C", 1e-12, 0.05, "fixture", c_path)]
            frequencies = np.array([1e9, 2e9])
            matrices = np.array([
                [[0.25 + 0.05j, 0.15], [0.15, 0.30 - 0.05j]],
                [[0.20 + 0.02j, 0.12], [0.12, 0.25 - 0.02j]],
            ])
            result = MeasuredComponentOptimizer(
                Problem(
                    frequencies,
                    matrices,
                    {0: [Band(1e9, 2e9)], 1: [Band(1e9, 2e9)]},
                ),
                inductors,
                capacitors,
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1,
                    ideal_iterations=1,
                    ideal_keep=4,
                    nearest_parts=1,
                    per_port_keep=5,
                    result_keep=10,
                    joint_refine_seeds=0,
                    joint_refine_passes=0,
                    max_components_per_port=1,
                    max_components_by_port={0: 0, 1: 1},
                ),
            ).optimize()

        self.assertTrue(result.candidates)
        self.assertTrue(all(not item.placements for item in result.per_port_candidates[0]))
        self.assertTrue(all(
            len(item.placements) <= 1
            for item in result.per_port_candidates[1]
        ))
        self.assertTrue(all(
            not any(placement.port == 0 for placement in item.placements)
            for item in result.candidates
        ))

    def test_cancelled_search_returns_power_balanced_baseline_with_provenance(self):
        frequencies = np.array([1e9, 2e9])
        matrices = np.array([
            [[0.25 + 0.05j, 0.15], [0.15, 0.30 - 0.05j]],
            [[0.20 + 0.02j, 0.12], [0.12, 0.25 - 0.02j]],
        ])
        result = MeasuredComponentOptimizer(
            Problem(
                frequencies,
                matrices,
                {0: [Band(1e9, 2e9)], 1: [Band(1e9, 2e9)]},
            ),
            (),
            (),
            Objective(),
            MeasuredSearchConfig(max_components_per_port=0),
            cancel_check=lambda: True,
        ).optimize()

        self.assertTrue(result.truncated)
        self.assertEqual(
            result.termination_reason, "cancelled during per-port search for port 0"
        )
        self.assertEqual(result.physical_evaluations, 1)
        self.assertEqual(result.best.topology_code, "0")
        self.assertLess(result.best.metrics["maximum_power_balance_error"], 1e-12)

    def test_single_port_cancel_keeps_completed_measured_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frequencies = np.array([1e9, 1.1e9])
            l_path, c_path = directory / "L.s2p", directory / "C.s2p"
            _write_reactive_fixture(l_path, frequencies, "L", 3.3e-9)
            _write_reactive_fixture(c_path, frequencies, "C", 1.5e-12)
            optimizer = MeasuredComponentOptimizer(
                Problem(
                    frequencies,
                    np.array([[[0.35 + 0.1j]], [[0.3 + 0.05j]]]),
                    {0: [Band(1e9, 1.1e9)]},
                ),
                (ComponentSpec("L", "L", 3.3e-9, 0.05, "fixture", l_path),),
                (ComponentSpec("C", "C", 1.5e-12, 0.05, "fixture", c_path),),
                Objective(),
                MeasuredSearchConfig(
                    ideal_restarts=1, ideal_iterations=1, ideal_keep=8,
                    nearest_parts=1, per_port_keep=8, result_keep=8,
                    max_components_per_port=2,
                ),
            )
            optimizer.cancel_check = lambda: optimizer.physical_evaluations >= 3
            result = optimizer.optimize()
        self.assertTrue(result.truncated)
        self.assertTrue(any(candidate.placements for candidate in result.candidates))
        self.assertIn("measured expansion", result.termination_reason)
        self.assertLess(result.best.metrics["maximum_power_balance_error"], 1e-10)

    def test_same_optimizer_resumes_without_repeating_exact_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frequencies = np.array([1e9, 1.1e9])
            l_path, c_path = directory / "L.s2p", directory / "C.s2p"
            _write_reactive_fixture(l_path, frequencies, "L", 3.3e-9)
            _write_reactive_fixture(c_path, frequencies, "C", 1.5e-12)
            problem = Problem(
                frequencies,
                np.array([[[0.35 + 0.1j]], [[0.3 + 0.05j]]]),
                {0: [Band(1e9, 1.1e9)]},
            )
            inductors = (ComponentSpec("L", "L", 3.3e-9, 0.05, "fixture", l_path),)
            capacitors = (ComponentSpec("C", "C", 1.5e-12, 0.05, "fixture", c_path),)
            config = MeasuredSearchConfig(
                ideal_restarts=1, ideal_iterations=1, ideal_keep=8,
                nearest_parts=1, per_port_keep=8, result_keep=8,
                joint_refine_seeds=1, joint_refine_passes=1,
                max_components_per_port=2,
            )
            optimizer = MeasuredComponentOptimizer(
                problem, inductors, capacitors, Objective(), config,
            )
            optimizer.cancel_check = lambda: optimizer.physical_evaluations >= 3
            partial = optimizer.optimize()
            prior_physical = optimizer.physical_evaluations
            prior_ideal = optimizer.ideal_evaluations
            prior_cache = len(optimizer.evaluation_cache)

            optimizer.cancel_check = None
            resumed = optimizer.optimize()
            fresh = MeasuredComponentOptimizer(
                problem, inductors, capacitors, Objective(), config,
            ).optimize()

        self.assertTrue(partial.truncated)
        self.assertFalse(resumed.truncated)
        self.assertEqual(resumed.ideal_evaluations, prior_ideal)
        self.assertGreater(resumed.physical_evaluations, prior_physical)
        self.assertGreater(len(optimizer.evaluation_cache), prior_cache)
        self.assertEqual(resumed.physical_evaluations, fresh.physical_evaluations)
        self.assertAlmostEqual(resumed.best.score_db, fresh.best.score_db, places=12)

    def test_multiport_resume_reuses_port_subproblems_and_joint_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frequencies = np.array([1e9, 1.1e9])
            l_path, c_path = directory / "L.s2p", directory / "C.s2p"
            _write_reactive_fixture(l_path, frequencies, "L", 3.3e-9)
            _write_reactive_fixture(c_path, frequencies, "C", 1.5e-12)
            problem = Problem(
                frequencies,
                np.array([
                    [[0.30 + 0.05j, 0.12], [0.12, 0.25 - 0.03j]],
                    [[0.28 + 0.04j, 0.10], [0.10, 0.23 - 0.02j]],
                ]),
                {0: [Band(1e9, 1.1e9)], 1: [Band(1e9, 1.1e9)]},
            )
            inductors = (ComponentSpec("L", "L", 3.3e-9, 0.05, "fixture", l_path),)
            capacitors = (ComponentSpec("C", "C", 1.5e-12, 0.05, "fixture", c_path),)
            config = MeasuredSearchConfig(
                ideal_restarts=1, ideal_iterations=1, ideal_keep=4,
                nearest_parts=1, per_port_keep=5, result_keep=10,
                joint_refine_seeds=0, joint_refine_passes=0,
                max_components_per_port=1,
            )
            optimizer = MeasuredComponentOptimizer(
                problem, inductors, capacitors, Objective(), config,
            )
            optimizer.cancel_check = lambda: optimizer.physical_evaluations >= 4
            partial = optimizer.optimize()
            prior_subproblem_ids = {
                port: id(subproblem)
                for port, subproblem in optimizer.subproblems_by_port.items()
            }
            prior_physical = optimizer.physical_evaluations

            optimizer.cancel_check = None
            resumed = optimizer.optimize()
            fresh = MeasuredComponentOptimizer(
                problem, inductors, capacitors, Objective(), config,
            ).optimize()

        self.assertTrue(partial.truncated)
        self.assertFalse(resumed.truncated)
        self.assertGreater(resumed.physical_evaluations, prior_physical)
        self.assertEqual(
            prior_subproblem_ids,
            {port: id(subproblem) for port, subproblem in optimizer.subproblems_by_port.items()
             if port in prior_subproblem_ids},
        )
        self.assertEqual(resumed.physical_evaluations, fresh.physical_evaluations)
        self.assertAlmostEqual(resumed.best.score_db, fresh.best.score_db, places=12)

    def test_physical_two_element_order_matches_ideal_dut_outward_order(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frequencies = np.array([1e9, 1.2e9])
            l_path, c_path = directory / "L.s2p", directory / "C.s2p"
            _write_reactive_fixture(l_path, frequencies, "L", 4.7e-9)
            _write_reactive_fixture(c_path, frequencies, "C", 2.2e-12)
            l_spec = ComponentSpec("L", "L", 4.7e-9, 0.05, "fixture", l_path)
            c_spec = ComponentSpec("C", "C", 2.2e-12, 0.05, "fixture", c_path)
            placements = (MeasuredPlacement("shunt", 0, l_spec), MeasuredPlacement("series", 0, c_spec))
            s = np.array([[[0.2 + 0.1j]], [[0.15 - 0.05j]]])
            problem = Problem(frequencies, s, {0: [Band(1e9, 1.2e9)]})
            physical = evaluate_physical_problem(problem, build_circuit_topology(1, placements)).s_parameters
            ideal = apply_elements_sweep(
                s,
                [Element("shunt", "L", 0, l_spec.value), Element("series", "C", 0, c_spec.value)],
                frequencies,
            )
        self.assertTrue(np.allclose(physical, ideal, atol=1e-10))

    @unittest.skipUnless(COMPONENT_ROOT.exists(), "Optenni component library is not installed")
    def test_installed_optenni_catalogs_have_expected_standard_values(self):
        inductors = load_coilcraft_0402hp_catalog(COMPONENT_ROOT / "Inductors" / "Coilcraft Inductors 0402hp")
        capacitors = load_murata_gqm18_catalog(COMPONENT_ROOT / "Capacitors" / "Murata Capacitors gqm18")
        self.assertEqual(len(inductors), 41)
        self.assertEqual((inductors[0].value, inductors[-1].value), (1e-9, 51e-9))
        self.assertEqual(len(capacitors), 51)
        self.assertEqual((capacitors[0].value, capacitors[-1].value), (1e-12, 100e-12))
        self.assertEqual(len({item.value for item in capacitors}), len(capacitors))

        tutorial_inductors = load_coilcraft_0402cs_catalog(
            COMPONENT_ROOT / "Inductors" / "Coilcraft Inductors 0402cs"
        )
        tutorial_capacitors = load_murata_gjm15_catalog(
            COMPONENT_ROOT / "Capacitors" / "Murata Capacitors gjm15",
            prefer_loosest_tolerance=True,
        )
        self.assertEqual(len(tutorial_inductors), 43)
        self.assertEqual((tutorial_inductors[0].value, tutorial_inductors[-1].value), (1e-9, 68e-9))
        self.assertEqual(len(tutorial_capacitors), 116)
        selected = next(item for item in tutorial_capacitors if abs(item.value - 5.6e-12) < 1e-18)
        self.assertEqual(selected.name, "GJM1552C1H5R6DB01")
        self.assertAlmostEqual(selected.tolerance, 0.5 / 5.6)


if __name__ == "__main__":
    unittest.main()
