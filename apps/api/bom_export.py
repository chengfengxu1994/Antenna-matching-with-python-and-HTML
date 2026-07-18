"""Deterministic procurement BOM export for validated project snapshots."""

from __future__ import annotations

import csv
from io import StringIO


def _selected_solution(document: dict) -> dict:
    results = document.get("results") or {}
    solutions = list(results.get("candidates") or [])
    if not solutions:
        raise ValueError("Project has no saved solutions")
    index = int(results.get("selected_index", 0) or 0)
    if index < 0 or index >= len(solutions):
        raise ValueError("Project selected solution index is invalid")
    return solutions[index]


def render_project_bom_csv(document: dict) -> str:
    """Aggregate identical measured parts while retaining placement traceability."""
    solution = _selected_solution(document)
    aggregated = {}
    per_port = solution.get("per_port") or {}
    for port_key, metrics in sorted(per_port.items(), key=lambda item: int(item[0])):
        for position, component in enumerate(metrics.get("components") or [], start=1):
            part_number = component.get("part_number") or component.get("part") or "ideal"
            key = (
                part_number,
                component.get("type") or component.get("comp_type") or "",
                component.get("value") or "",
                component.get("manufacturer") or "",
                component.get("series") or "",
                component.get("package_code") or "",
                component.get("tolerance_pct"),
                component.get("voltage_code") or "",
                component.get("dielectric") or "",
            )
            row = aggregated.setdefault(key, {
                "quantity": 0,
                "placements": [],
                "connection": set(),
                "metadata_sources": set(),
            })
            row["quantity"] += 1
            row["placements"].append(f"P{int(port_key) + 1}:{position}")
            row["connection"].add(
                component.get("connection_type") or component.get("connection") or ""
            )
            row["metadata_sources"].update(
                value for value in (component.get("metadata_provenance") or {}).values() if value
            )

    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow([
        "Line", "Quantity", "Part Number", "Manufacturer", "Series", "Type", "Value",
        "Package", "Tolerance (%)", "Voltage Code", "Dielectric", "Connection",
        "Placements", "Metadata Sources",
    ])
    for line, (key, row) in enumerate(sorted(aggregated.items(), key=lambda item: item[0]), start=1):
        part, comp_type, value, manufacturer, series, package, tolerance, voltage, dielectric = key
        writer.writerow([
            line, row["quantity"], part, manufacturer, series, comp_type, value, package,
            "" if tolerance is None else tolerance, voltage, dielectric,
            " + ".join(sorted(filter(None, row["connection"]))),
            " + ".join(row["placements"]),
            " + ".join(sorted(row["metadata_sources"])),
        ])
    return "\ufeff" + output.getvalue()
