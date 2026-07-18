import json
from types import SimpleNamespace

import pytest

from engine.component_environment import (
    apply_component_environment_catalog,
    load_component_environment_catalog,
)
from engine.component_lib import ComponentInfo, component_metadata
from engine.murata_db_adapter import DBComponentAdapter


def _document():
    return {
        "schema_version": 1,
        "source": {
            "name": "RF lab chamber characterization",
            "document": "LAB-RF-2026-014",
            "evidence_level": "laboratory_measurement",
        },
        "components": [{
            "part_number": "L5N6_TEST",
            "tempco_ppm_per_c": 125.0,
            "systematic_bias_pct": -0.4,
        }],
    }


def test_environment_sidecar_is_exact_part_auditable_and_applied(tmp_path):
    path = tmp_path / "component_environment.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")
    catalog = load_component_environment_catalog(path)
    matching = ComponentInfo(
        "l5n6_test", "model.s2p", "__DIR__", "inductor", 5.6, "nH",
    )
    unrelated = ComponentInfo(
        "C1P0_OTHER", "model.s2p", "__DIR__", "capacitor", 1.0, "pF",
    )

    assert apply_component_environment_catalog([matching, unrelated], catalog) == 1
    metadata = component_metadata(matching)
    assert metadata["tempco_ppm_per_c"] == 125.0
    assert metadata["systematic_bias_pct"] == -0.4
    assert metadata["environment_metadata"]["source_document"] == "LAB-RF-2026-014"
    assert metadata["provenance"]["tempco_ppm_per_c"] == (
        "environment_sidecar:laboratory_measurement"
    )
    assert component_metadata(unrelated)["tempco_ppm_per_c"] is None
    assert catalog.summary()["tempco_count"] == 1
    assert len(catalog.sha256) == 64

    db_library = SimpleNamespace(
        environment_catalog=catalog,
        get_series_summary=lambda: [{"name": "SERIES", "manufacturer": "Vendor"}],
    )
    db_component = DBComponentAdapter(SimpleNamespace(
        part_number="L5N6_TEST", component_type="inductor",
        nominal_value=5.6, nominal_unit="nH", s2p_filename="model.s2p",
        zip_path="archive.zip", series="SERIES", size_code="0402",
        tolerance_code="J", tolerance_pct=5.0, voltage_code="", dielectric="",
    ), db_library)
    assert db_component.tempco_ppm_per_c == 125.0
    assert db_component.systematic_bias_pct == -0.4


@pytest.mark.parametrize("mutation, message", [
    (lambda value: value.update(schema_version=2), "schema_version 1"),
    (lambda value: value["components"].append(dict(value["components"][0])), "duplicate"),
    (lambda value: value["components"][0].update(systematic_bias_pct=-100), "non-positive"),
    (lambda value: value["source"].update(evidence_level="guessed"), "requires one of"),
])
def test_environment_sidecar_rejects_ambiguous_or_nonphysical_data(tmp_path, mutation, message):
    document = _document()
    mutation(document)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_component_environment_catalog(path)
