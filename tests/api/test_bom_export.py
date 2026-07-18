import csv
from io import StringIO
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

from bom_export import render_project_bom_csv


def test_bom_csv_aggregates_identical_parts_and_keeps_placements():
    component = {
        "part_number": "L_TEST", "type": "inductor", "value": "5.6nH",
        "connection_type": "series", "manufacturer": "Vendor", "series": "RF family",
        "package_code": "0402", "tolerance_pct": 5.0,
        "metadata_provenance": {"manufacturer": "database"},
    }
    document = {
        "results": {
            "selected_index": 0,
            "candidates": [{
                "per_port": {
                    "0": {"components": [component, component]},
                    "1": {"components": [component]},
                },
            }],
        },
    }
    rows = list(csv.DictReader(StringIO(render_project_bom_csv(document).lstrip("\ufeff"))))
    assert len(rows) == 1
    assert rows[0]["Quantity"] == "3"
    assert rows[0]["Placements"] == "P1:1 + P1:2 + P2:1"
    assert rows[0]["Metadata Sources"] == "database"
