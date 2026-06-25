import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.component_lib import ComponentInfo, ComponentLibrary
from engine.optimizer import MatchingOptimizer, OptimizerConfig
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
