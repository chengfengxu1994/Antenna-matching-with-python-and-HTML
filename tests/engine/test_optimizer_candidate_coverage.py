import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

from engine.component_lib import ComponentInfo, ComponentLibrary
from engine.optimizer import MatchingOptimizer, OptimizerConfig
from engine.multiport_optimizer import optenni_compatible_topologies
from rfmatch_core import PORT_TOPOLOGY_PATTERNS
from engine.topology import get_standard_topologies
from engine.touchstone import TouchstoneData


def _component(part, comp_type, value):
    return ComponentInfo(
        part_number=part,
        s2p_filename="unused.s2p",
        zip_path="unused.zip",
        component_type=comp_type,
        nominal_value=value,
        nominal_unit="nH" if comp_type == "inductor" else "pF",
    )


def test_two_component_candidates_cover_all_primary_values():
    library = ComponentLibrary()
    for idx in range(75):
        library.add_component(_component(f"L{idx}", "inductor", idx + 1.0))
    library.add_component(_component("L_DUP_A", "inductor", 10.0))
    library.add_component(_component("L_DUP_B", "inductor", 10.0))
    for idx in range(83):
        library.add_component(_component(f"C{idx}", "capacitor", idx + 0.5))
    library.add_component(_component("C_DUP_A", "capacitor", 1.5))
    library.add_component(_component("C_DUP_B", "capacitor", 1.5))

    dut = TouchstoneData(
        filename="synthetic.s1p",
        frequency_unit="HZ",
        parameter_type="S",
        data_format="RI",
        reference_resistance=50.0,
        num_ports=1,
        frequencies=[1.0e9],
        sparameters={(1, 1): [0.5 + 0.0j]},
    )

    optimizer = MatchingOptimizer(dut, library, OptimizerConfig(target_frequency_hz=1.0e9))
    topology = next(t for t in get_standard_topologies() if t.name == "L-Network (Series-L, Shunt-C)")

    first = optimizer._candidate_components_for_element(topology.elements[0], exhaustive=True)
    second = optimizer._candidate_components_for_element(topology.elements[1], exhaustive=True)

    assert len(first) == 75
    assert len(second) == 83
    assert first[0].nominal_value == 1.0
    assert first[-1].nominal_value == 75.0
    assert second[0].nominal_value == 0.5
    assert second[-1].nominal_value == 82.5
    assert [c.part_number for c in first if c.nominal_value == 10.0] == ["L9"]
    assert [c.part_number for c in second if c.nominal_value == 1.5] == ["C1"]


def test_large_two_component_product_switches_to_progressive_search():
    library = ComponentLibrary()
    for idx in range(75):
        library.add_component(_component(f"L{idx}", "inductor", idx + 1.0))
    for idx in range(83):
        library.add_component(_component(f"C{idx}", "capacitor", idx + 0.5))
    dut = TouchstoneData(
        filename="synthetic.s1p", frequency_unit="HZ", parameter_type="S",
        data_format="RI", reference_resistance=50.0, num_ports=1,
        frequencies=[1.0e9], sparameters={(1, 1): [0.5 + 0.0j]},
    )
    optimizer = MatchingOptimizer(
        dut, library,
        OptimizerConfig(target_frequency_hz=1.0e9, max_combinations_to_evaluate=5000),
    )
    topology = next(t for t in get_standard_topologies() if t.name == "L-Network (Series-L, Shunt-C)")
    called = []
    optimizer.optimize_progressive = lambda *args: called.append(True) or ["progressive"]
    result = optimizer._exhaustive_search(np.array([[0.5 + 0j]]), topology, {}, 0)
    assert called == [True]
    assert result == ["progressive"]


def test_product_and_core_share_the_same_optenni_topology_set():
    for max_components in (2, 4):
        actual = {
            tuple((element.connection_type.value, "L" if element.component_type == "inductor" else "C") for element in topology.elements)
            for topology in optenni_compatible_topologies(max_components)
        }
        expected = {
            pattern for pattern in PORT_TOPOLOGY_PATTERNS
            if pattern and len(pattern) <= max_components
        }
        assert actual == expected


def test_component_cache_supports_database_components_without_zip_path():
    class DatabaseComponent:
        part_number = "DB-C1"
        s2p_filename = ""
        db_id = 17

        def get_s_matrix_at_freq(self, frequency_hz):
            return np.array([[0.1, 0.9], [0.9, 0.1]], dtype=complex)

    library = ComponentLibrary()
    dut = TouchstoneData(
        filename="synthetic.s1p", frequency_unit="HZ", parameter_type="S",
        data_format="RI", reference_resistance=50.0, num_ports=1,
        frequencies=[1.0e9], sparameters={(1, 1): [0.5 + 0.0j]},
    )
    optimizer = MatchingOptimizer(dut, library, OptimizerConfig(target_frequency_hz=1.0e9))
    component = DatabaseComponent()
    first = optimizer._get_component_s(component)
    second = optimizer._get_component_s(component)
    assert np.array_equal(first, second)
    assert len(optimizer._component_cache) == 1


def test_component_cache_can_resolve_raw_database_record_through_library():
    class RawRecord:
        db_id = 9
        part_number = "RAW-C9"
        s2p_filename = ""

    class DatabaseLibrary(ComponentLibrary):
        def get_s_matrix_at_freq(self, db_id, freq_mhz):
            assert db_id == 9
            assert freq_mhz == 1000.0
            return np.array([[0.2, 0.8], [0.8, 0.2]], dtype=complex)

    dut = TouchstoneData(
        filename="synthetic.s1p", frequency_unit="HZ", parameter_type="S",
        data_format="RI", reference_resistance=50.0, num_ports=1,
        frequencies=[1.0e9], sparameters={(1, 1): [0.5 + 0.0j]},
    )
    optimizer = MatchingOptimizer(dut, DatabaseLibrary(), OptimizerConfig(target_frequency_hz=1.0e9))
    matrix = optimizer._get_component_s(RawRecord())
    assert matrix[0, 0] == 0.2
