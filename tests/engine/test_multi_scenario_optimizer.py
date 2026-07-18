import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

from engine.efficiency_data import EfficiencyData
from engine.multi_scenario_optimizer import MultiScenarioOptimizer, Scenario, ideal_component_s
from engine.topology import get_standard_topologies
from engine.touchstone import TouchstoneData
from rfmatch_core import OptimizationCancelled


class FakeComponent:
    def __init__(self, part_number, component_type, value):
        self.part_number = part_number
        self.component_type = component_type
        self.nominal_value = value
        self.nominal_unit = 'nH' if component_type == 'inductor' else 'pF'
        self.model_calls = 0

    def get_s_matrix_at_freq(self, freq_hz):
        self.model_calls += 1
        return ideal_component_s(self.component_type, self.nominal_value, freq_hz)


class FakeLibrary:
    def __init__(self):
        self.inductors = [FakeComponent('L1', 'inductor', 1.0), FakeComponent('L10', 'inductor', 10.0)]
        self.capacitors = [FakeComponent('C1', 'capacitor', 1.0), FakeComponent('C10', 'capacitor', 10.0)]


def dut(filename, gamma):
    return TouchstoneData(
        filename=filename, frequency_unit='HZ', parameter_type='S', data_format='RI',
        reference_resistance=50.0, num_ports=1,
        frequencies=[0.9e9, 1.1e9],
        sparameters={(1, 1): [gamma, gamma]},
    )


def test_total_efficiency_is_converted_to_radiation_efficiency():
    total = EfficiencyData(np.array([0.9e9, 1.1e9]), np.array([0.6, 0.6]), 'total.txt')
    scenarios = [
        Scenario('a.s1p', dut('a.s1p', 0.5 + 0j), efficiency=total, efficiency_kind='total'),
        Scenario('b.s1p', dut('b.s1p', 0.5 + 0j), efficiency=total, efficiency_kind='total'),
    ]
    optimizer = MultiScenarioOptimizer(scenarios, FakeLibrary(), [[1000, 1000.1]], num_band_points=2)
    result = optimizer.evaluate([])

    assert abs(result['avg_total_efficiency'] - 0.6) < 1e-9
    point = result['scenarios'][0]['points'][0]
    assert abs(point['radiation_efficiency'] - 0.8) < 1e-9


def test_optimizer_returns_one_shared_part_for_every_scenario():
    scenarios = [
        Scenario('free-space.s1p', dut('free-space.s1p', 0.6 + 0.2j), weight=1.0),
        Scenario('hand.s1p', dut('hand.s1p', 0.4 - 0.3j), weight=2.0),
    ]
    topology = next(t for t in get_standard_topologies() if t.name == '1-Element (Series-L)')
    optimizer = MultiScenarioOptimizer(
        scenarios, FakeLibrary(), [[950, 1050]], num_band_points=3,
        objective='balanced', beam_width=2, timeout_seconds=5,
    )
    results = optimizer.optimize([topology])

    assert results
    assert len(results[0]['components']) == 1
    assert results[0]['components'][0]['part_number'] in {'L1', 'L10'}
    assert {item['filename'] for item in results[0]['scenarios']} == {'free-space.s1p', 'hand.s1p'}
    assert 0 <= results[0]['score'] <= 1
    assert results[0]['score_db'] <= 0
    assert results[0]['maximum_power_balance_error'] < 1e-9


def test_candidate_sampling_spans_log_values_without_linear_bias():
    library = FakeLibrary()
    library.capacitors = [FakeComponent(f'C{value}', 'capacitor', value) for value in (0.1, 0.3, 1, 3, 10, 100, 10000)]
    scenarios = [Scenario('a.s1p', dut('a.s1p', 0.5)), Scenario('b.s1p', dut('b.s1p', 0.4))]
    optimizer = MultiScenarioOptimizer(
        scenarios, library, [[900, 1100]], num_band_points=2,
        max_candidates_per_position=4,
    )
    values = [component.nominal_value for component in optimizer._candidates('capacitor')]

    assert min(values) <= 0.3
    assert max(values) >= 10
    assert any(0.3 <= value <= 100 for value in values)


def test_physical_component_models_are_cached_and_diagnostics_are_auditable():
    library = FakeLibrary()
    scenarios = [Scenario('a.s1p', dut('a.s1p', 0.5)), Scenario('b.s1p', dut('b.s1p', 0.4))]
    optimizer = MultiScenarioOptimizer(
        scenarios, library, [[900, 1100]], num_band_points=3,
    )
    component = library.inductors[0]
    specs = [{
        'position': 0, 'connection_type': 'series',
        'component_type': 'inductor', 'component': component,
    }]
    optimizer.evaluate(specs, 'series-L')
    optimizer.evaluate(specs, 'series-L')

    diagnostics = optimizer.diagnostics()
    assert diagnostics['physical_evaluations'] == 2
    assert diagnostics['component_models_built'] == 1
    assert diagnostics['component_model_cache_entries'] == 1
    assert component.model_calls == 3


def test_all_topologies_receive_a_fair_screen_and_results_are_dense_verified():
    scenarios = [Scenario('a.s1p', dut('a.s1p', 0.5)), Scenario('b.s1p', dut('b.s1p', 0.4))]
    optimizer = MultiScenarioOptimizer(
        scenarios, FakeLibrary(), [[900, 1100]], num_band_points=2,
        beam_width=2, timeout_seconds=20,
    )
    topologies = [item for item in get_standard_topologies() if item.num_components == 2]
    results = optimizer.optimize(topologies)
    diagnostics = optimizer.diagnostics()

    assert results
    assert diagnostics['topologies_requested'] == len(topologies)
    assert diagnostics['topologies_screened'] == len(topologies)
    assert diagnostics['screen_evaluations'] >= len(topologies)

    verified, verification = optimizer.verify_solutions(results[:2], 7)
    assert len(verified) == min(2, len(results))
    assert verification['physical_evaluations'] == len(verified)
    assert verification['frequency_points'] == 7
    assert len(verified[0]['scenarios'][0]['points']) == 7
    assert 'search_estimate_score_db' in verified[0]


def test_optimizer_reports_progress_and_honors_cooperative_cancellation():
    scenarios = [Scenario('a.s1p', dut('a.s1p', 0.5)), Scenario('b.s1p', dut('b.s1p', 0.4))]
    optimizer = MultiScenarioOptimizer(
        scenarios, FakeLibrary(), [[900, 1100]], num_band_points=2,
        beam_width=2, timeout_seconds=20,
    )
    topologies = [item for item in get_standard_topologies() if item.num_components == 2]
    progress = []

    with __import__('pytest').raises(OptimizationCancelled):
        optimizer.optimize(
            topologies,
            progress_callback=lambda item: progress.append(item),
            cancel_check=lambda: len(progress) >= 3,
        )

    assert progress
    assert progress[0]['stage'] == 'ideal_screen'
    assert any(item['stage'] == 'physical_screen' for item in progress)
