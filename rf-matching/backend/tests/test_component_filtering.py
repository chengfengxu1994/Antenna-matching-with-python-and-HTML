import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.component_lib import scan_s2p_directory, filter_component_library


OPTENNI_DIR = r"C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary"


def test_optenni_case1_component_filter_finds_requested_series():
    if not os.path.isdir(OPTENNI_DIR):
        return

    library = scan_s2p_directory(OPTENNI_DIR)
    filtered = filter_component_library(
        library,
        inductor_tokens=["COILCRAFT INDUCTORS 0402HP", "0402HP", "04HP"],
        capacitor_tokens=["MURATA CAPACITORS GQM18", "GQM18"],
    )

    assert filtered.inductors
    assert filtered.capacitors
    assert any(c.part_number.upper() == "04HP2N0" for c in filtered.inductors)
    assert any(c.part_number.upper() == "GQM1885C2A1R0BB01" for c in filtered.capacitors)
    assert all(
        "0402HP" in (c.s2p_filename + c.part_number).upper()
        or c.part_number.upper().startswith("04HP")
        for c in filtered.inductors
    )
    assert all("GQM18" in (c.s2p_filename + c.part_number).upper() for c in filtered.capacitors)
