"""Portable, content-addressed measured-component catalog regression tests."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "apps", "api"
))

from engine.component_lib import (
    COMPONENT_CATALOG_FINGERPRINT_SCHEMA,
    ComponentInfo,
    ComponentLibrary,
    component_library_content_fingerprint,
)


def component(source: str, *, zip_path: str = "__DIR__", series: str = "0402HP"):
    return ComponentInfo(
        part_number="04HP5N6",
        s2p_filename=source,
        zip_path=zip_path,
        component_type="inductor",
        nominal_value=5.6,
        nominal_unit="nH",
        manufacturer="Coilcraft",
        series=series,
        size_code="0402",
        tolerance_pct=5.0,
    )


def library(item: ComponentInfo):
    result = ComponentLibrary()
    result.add_component(item)
    return result


class ComponentCatalogFingerprintTests(unittest.TestCase):
    def test_directory_roots_do_not_affect_fingerprint(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            left_file = Path(left) / "04HP5N6.s2p"
            right_file = Path(right) / "04HP5N6.s2p"
            payload = b"# GHZ S RI R 50\n1 0 0 1 0 1 0 0 0\n"
            left_file.write_bytes(payload)
            right_file.write_bytes(payload)

            first = component_library_content_fingerprint(library(component(str(left_file))))
            second = component_library_content_fingerprint(library(component(str(right_file))))
            self.assertEqual(first["schema"], COMPONENT_CATALOG_FINGERPRINT_SCHEMA)
            self.assertEqual(first["digest"], second["digest"])
            self.assertTrue(first["content_verified"])
            self.assertEqual(first["component_count"], 1)
            self.assertEqual(first["source_count"], 1)

    def test_source_bytes_and_engineering_metadata_are_both_covered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "04HP5N6.s2p"
            path.write_bytes(b"original measured model")
            baseline = component_library_content_fingerprint(library(component(str(path))))

            path.write_bytes(b"changed measured model with a different length")
            changed_source = component_library_content_fingerprint(library(component(str(path))))
            changed_metadata = component_library_content_fingerprint(
                library(component(str(path), series="Different series"))
            )
            self.assertNotEqual(baseline["digest"], changed_source["digest"])
            self.assertNotEqual(changed_source["digest"], changed_metadata["digest"])

    def test_zip_repacking_does_not_change_entry_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            first_zip = Path(directory) / "first.zip"
            second_zip = Path(directory) / "second.zip"
            entry = "Inductors/04HP5N6.s2p"
            payload = b"measured zip entry"
            with zipfile.ZipFile(first_zip, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(entry, payload)
            with zipfile.ZipFile(second_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(entry, payload)

            first = component_library_content_fingerprint(
                library(component(entry, zip_path=str(first_zip)))
            )
            second = component_library_content_fingerprint(
                library(component(entry, zip_path=str(second_zip)))
            )
            self.assertEqual(first["digest"], second["digest"])
            self.assertTrue(first["content_verified"])

    def test_unreadable_source_never_claims_content_verification(self):
        missing = component_library_content_fingerprint(
            library(component("Z:/missing/04HP5N6.s2p"))
        )
        self.assertFalse(missing["content_verified"])
        self.assertEqual(missing["unreadable_source_count"], 1)


if __name__ == "__main__":
    unittest.main()
