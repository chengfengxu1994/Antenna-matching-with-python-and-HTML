"""Read-only extraction of reproducible evidence from Optenni 4.x OPR XML.

OPR projects store the input impedance matrix, optimization settings and saved
candidate circuits.  They do not store authoritative calculated result curves,
so this parser deliberately exposes project evidence without claiming to be an
Optenni numerical result exporter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PureWindowsPath
import re
from typing import Any
import xml.etree.ElementTree as ET


_PORT_TOPOLOGY = re.compile(r"P(\d+):([^,]*)")


def _number(value: str | None, default: float = 0.0) -> float:
    return float(value) if value not in (None, "") else default


def _integer(value: str | None, default: int = 0) -> int:
    return int(value) if value not in (None, "") else default


def _optional_number(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _candidate(node: ET.Element, index: int) -> dict[str, Any]:
    name = node.get("name", "")
    topology_by_port = {
        str(int(port) - 1): topology.strip()
        for port, topology in _PORT_TOPOLOGY.findall(name)
    }
    components = []
    for wrapper in node.iter():
        if wrapper.tag not in {"Capacitor_from_series", "Inductor_from_series"}:
            continue
        component = wrapper.find("ComponentFromSeries")
        if component is None:
            continue
        label = wrapper.get("componentLabel", "")
        port_match = re.search(r"(\d+)$", label)
        component_type = "capacitor" if wrapper.tag.startswith("Capacitor") else "inductor"
        value = _number(component.get("value"))
        filename = component.get("filename", "")
        relative_tolerance = _number(component.get("relTol"), -1.0)
        absolute_tolerance = _number(component.get("absTol"), -1.0)
        components.append({
            "port": int(port_match.group(1)) - 1 if port_match else None,
            "label": label,
            "connection": "series" if wrapper.get("orientation") == "hor" else "shunt",
            "component_type": component_type,
            "part_number": PureWindowsPath(filename).stem if filename else component.get("code", ""),
            "manufacturer_code": component.get("code", ""),
            "value": value,
            "nominal_unit": "pF" if component_type == "capacitor" else "nH",
            "value_si": value * (1e-12 if component_type == "capacitor" else 1e-9),
            "manufacturer": component.get("manufacturer", ""),
            "series": component.get("seriesName", ""),
            "library_subdirectory": component.get("subdirectory", ""),
            "model_filename": PureWindowsPath(filename).name if filename else "",
            "relative_tolerance_pct": relative_tolerance if relative_tolerance >= 0 else None,
            "absolute_tolerance": absolute_tolerance if absolute_tolerance >= 0 else None,
        })
    return {
        "index": index,
        "name": name,
        "topology_by_port": topology_by_port,
        "components": components,
    }


def parse_optenni_opr(path: str | Path) -> dict[str, Any]:
    """Extract settings and saved candidates from an Optenni OPR project."""
    source = Path(path)
    raw = source.read_bytes()
    root = ET.fromstring(raw)
    impedance = root.find("./ImpedanceConfigurations/ImpedanceConfigurationItem/ImpedanceData")
    if impedance is None:
        raise ValueError("Optenni project has no impedance configuration data")
    data = impedance.find("Data")
    embedded_text = data.text if data is not None and data.text is not None else ""

    candidate_nodes = root.findall("./ResultTree/RootTreeItem//MultiPortTreeItem")
    candidates = [_candidate(node, index) for index, node in enumerate(candidate_nodes)]
    matched = next((item for item in candidates if item["components"]), None)

    settings_node = None
    if matched is not None:
        settings_node = candidate_nodes[matched["index"]].find("MultiPortDialogSave")
    bands = []
    if settings_node is not None:
        for item in settings_node.findall(
            "./FrequencySetList/MultiPortFrequencySet/FrequencyModelList/FrequencyModel/FrequencyItem"
        ):
            bands.append({
                "port": _integer(item.get("sparI"), 1) - 1,
                "start_hz": _number(item.get("startFreq")),
                "stop_hz": _number(item.get("endFreq")),
                "start_index": _integer(item.get("startFreqInd")),
                "stop_index": _integer(item.get("endFreqInd")),
                "label": item.get("label", ""),
                "s_parameter": item.get("sparLabel", ""),
            })

    tolerance_nodes = [
        node for node in root.iter()
        if node.tag in {"MatchDialogSave", "MultiPortDialogSave", "ComponentParameters"}
        and any(node.get(name) not in (None, "") for name in ("indToler", "capToler", "msWidthToler"))
    ]
    tolerance_profiles = []
    seen_profiles = set()
    for node in tolerance_nodes:
        profile = {
            "source_element": node.tag,
            "inductor_value_tolerance_pct": _optional_number(node.get("indToler")),
            "capacitor_value_tolerance_pct": _optional_number(node.get("capToler")),
            "microstrip_width_tolerance_pct": _optional_number(node.get("msWidthToler")),
            "uses_transmission_line_synthesis": node.get("useTLSynthesis") == "1" if node.get("useTLSynthesis") is not None else None,
            "includes_substrate": node.get("includeSubstrate") == "1" if node.get("includeSubstrate") is not None else None,
        }
        key = tuple(profile.values())
        if key not in seen_profiles:
            seen_profiles.add(key)
            tolerance_profiles.append(profile)
    manufacturing_settings = next((
        profile for profile in tolerance_profiles
        if profile["uses_transmission_line_synthesis"] is True
        and (profile["microstrip_width_tolerance_pct"] or 0.0) > 0.0
    ), next((
        profile for profile in tolerance_profiles
        if (profile["microstrip_width_tolerance_pct"] or 0.0) > 0.0
    ), tolerance_profiles[0] if tolerance_profiles else None))

    return {
        "schema_version": 1,
        "format": "Optenni OPR XML",
        "optenni_version": root.get("Version"),
        "file_version": root.get("FileVersion"),
        "has_results": root.get("hasResults") == "1",
        "project_sha256": hashlib.sha256(raw).hexdigest(),
        "impedance_configuration": {
            "name": root.find("./ImpedanceConfigurations/ImpedanceConfigurationItem").get("name", ""),
            "ports": _integer(impedance.get("np")),
            "frequency_points": _integer(impedance.get("nfreq")),
            "reference_impedance_ohm": _number(impedance.get("refImp")),
            "source_filename": PureWindowsPath(impedance.get("fileName", "")).name,
            "embedded_data_sha256": hashlib.sha256(embedded_text.encode("utf-8")).hexdigest(),
            "embedded_data_rows": len([line for line in embedded_text.splitlines() if line.strip()]),
        },
        "objective": {
            "alpha_in_band": _number(settings_node.get("alphaInBand")) if settings_node is not None else None,
            "alpha_total": _number(settings_node.get("alphaTotal")) if settings_node is not None else None,
            "initial_search_deepness": _number(settings_node.get("initialSearchDeepness")) if settings_node is not None else None,
        },
        "manufacturing_tolerance_settings": manufacturing_settings,
        "manufacturing_tolerance_profiles": tolerance_profiles,
        "bands": bands,
        "candidate_count": len(candidates),
        "matched_candidate_count": sum(bool(item["components"]) for item in candidates),
        "saved_winner": matched,
        "candidate_names": [item["name"] for item in candidates],
    }
