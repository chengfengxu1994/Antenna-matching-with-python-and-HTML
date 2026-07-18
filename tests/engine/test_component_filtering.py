import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api'))

from engine.component_lib import (
    ComponentInfo, ComponentLibrary, component_metadata,
    scan_s2p_directory, filter_component_library,
    component_series_id, filter_component_library_by_series,
    filter_component_library_by_parameters,
)


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

    inductor_ids = {component_series_id(component) for component in library.inductors}
    capacitor_ids = {component_series_id(component) for component in library.capacitors}
    hp_id = next(item for item in inductor_ids if "0402HP" in item.upper())
    gqm_id = next(item for item in capacitor_ids if "GQM18" in item.upper())
    exact = filter_component_library_by_series(library, [hp_id, gqm_id])
    assert exact.inductors
    assert exact.capacitors
    assert {component_series_id(component) for component in exact.inductors} == {hp_id}
    assert {component_series_id(component) for component in exact.capacitors} == {gqm_id}


def _metadata_component(
    part, kind, manufacturer="", package="", tolerance=None,
    voltage_code="", dielectric="",
):
    component = ComponentInfo(
        part, f"/{part}.s2p", "__DIR__", kind, 1.0,
        "nH" if kind == "inductor" else "pF",
        manufacturer=manufacturer,
        series=f"{manufacturer} family" if manufacturer else "",
        size_code=package,
        tolerance_pct=tolerance,
        voltage_code=voltage_code,
        dielectric=dielectric,
        metadata_provenance={
            "manufacturer": "database" if manufacturer else "unknown",
            "package_code": "database" if package else "unknown",
            "tolerance_pct": "database" if tolerance is not None else "unknown",
            "voltage_code": "part_number_inferred" if voltage_code else "unknown",
            "dielectric": "part_number_inferred" if dielectric else "unknown",
        },
    )
    return component


def test_parameter_filter_has_explicit_unknown_metadata_semantics():
    library = ComponentLibrary()
    library.add_component(_metadata_component("L_GOOD", "inductor", "Murata", "0402", 2.0))
    library.add_component(_metadata_component("L_WRONG", "inductor", "TDK", "0402", 2.0))
    library.add_component(_metadata_component("C_GOOD", "capacitor", "Murata", "0402", 2.0, "1H", "C0G"))
    library.add_component(_metadata_component("C_UNKNOWN", "capacitor"))
    filters = {
        "manufacturers": ["Murata"], "package_codes": ["0402"],
        "maximum_tolerance_pct": 5.0, "unknown_metadata_policy": "include",
        "voltage_codes": ["1H"], "dielectrics": ["C0G"],
    }

    compatible, stats = filter_component_library_by_parameters(library, filters)
    assert [item.part_number for item in compatible.inductors] == ["L_GOOD"]
    assert [item.part_number for item in compatible.capacitors] == ["C_GOOD", "C_UNKNOWN"]
    assert stats["included_with_unknown"] == 1
    assert stats["excluded_by_value"] == 1
    assert stats["metadata_sources"]["part_number_inferred"] == 2

    strict, stats = filter_component_library_by_parameters(
        library, {**filters, "unknown_metadata_policy": "exclude"}
    )
    assert [item.part_number for item in strict.inductors] == ["L_GOOD"]
    assert [item.part_number for item in strict.capacitors] == ["C_GOOD"]
    assert stats["excluded_unknown"] == 1


def test_directory_scan_labels_path_metadata_without_claiming_database_authority():
    if not os.path.isdir(OPTENNI_DIR):
        return
    library = scan_s2p_directory(os.path.join(OPTENNI_DIR, "Capacitors", "Murata Capacitors gqm18"))
    metadata = component_metadata(library.capacitors[0])
    assert metadata["manufacturer"] == "Murata"
    assert metadata["provenance"]["manufacturer"] == "catalog_path"
    assert metadata["provenance"]["tolerance_pct"] == "unknown"
