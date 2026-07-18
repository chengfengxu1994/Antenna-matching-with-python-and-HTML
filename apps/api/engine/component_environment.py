"""Auditable per-part environmental metadata overlays for measured components."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable


EVIDENCE_LEVELS = {
    "manufacturer_datasheet",
    "laboratory_measurement",
    "engineering_assumption",
}


@dataclass(frozen=True)
class ComponentEnvironmentRecord:
    part_number: str
    tempco_ppm_per_c: float | None
    systematic_bias_pct: float | None
    evidence_level: str
    source_name: str
    source_document: str
    source_sha256: str | None

    def provenance_label(self) -> str:
        return f"environment_sidecar:{self.evidence_level}"

    def as_dict(self) -> dict:
        return {
            "part_number": self.part_number,
            "tempco_ppm_per_c": self.tempco_ppm_per_c,
            "systematic_bias_pct": self.systematic_bias_pct,
            "evidence_level": self.evidence_level,
            "source_name": self.source_name,
            "source_document": self.source_document,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class ComponentEnvironmentCatalog:
    path: Path
    sha256: str
    source_name: str
    records: dict[str, ComponentEnvironmentRecord]

    def lookup(self, part_number: str) -> ComponentEnvironmentRecord | None:
        return self.records.get(str(part_number).strip().casefold())

    def summary(self) -> dict:
        return {
            "filename": self.path.name,
            "sha256": self.sha256,
            "source_name": self.source_name,
            "component_count": len(self.records),
            "tempco_count": sum(
                record.tempco_ppm_per_c is not None for record in self.records.values()
            ),
            "bias_count": sum(
                record.systematic_bias_pct is not None for record in self.records.values()
            ),
            "evidence_levels": {
                level: sum(record.evidence_level == level for record in self.records.values())
                for level in sorted(EVIDENCE_LEVELS)
            },
        }


def _optional_finite(entry: dict, name: str) -> float | None:
    value = entry.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number or null")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number or null") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def load_component_environment_catalog(path: str | Path) -> ComponentEnvironmentCatalog:
    """Load a strict schema-v1 overlay without inferring missing engineering data."""
    catalog_path = Path(path).expanduser().resolve()
    raw = catalog_path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid component environment JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("component environment sidecar requires schema_version 1")
    source = document.get("source")
    if not isinstance(source, dict) or not str(source.get("name", "")).strip():
        raise ValueError("component environment sidecar requires source.name")
    components = document.get("components")
    if not isinstance(components, list):
        raise ValueError("component environment sidecar requires a components array")

    records: dict[str, ComponentEnvironmentRecord] = {}
    for index, entry in enumerate(components, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"component environment entry #{index} must be an object")
        part_number = str(entry.get("part_number", "")).strip()
        if not part_number:
            raise ValueError(f"component environment entry #{index} requires part_number")
        key = part_number.casefold()
        if key in records:
            raise ValueError(f"duplicate component environment part_number {part_number!r}")
        tempco = _optional_finite(entry, "tempco_ppm_per_c")
        bias = _optional_finite(entry, "systematic_bias_pct")
        if tempco is None and bias is None:
            raise ValueError(f"component environment entry {part_number!r} has no environmental values")
        if bias is not None and bias <= -100.0:
            raise ValueError(f"component environment entry {part_number!r} has non-positive biased value")
        evidence = str(entry.get("evidence_level", source.get("evidence_level", ""))).strip()
        if evidence not in EVIDENCE_LEVELS:
            raise ValueError(
                f"component environment entry {part_number!r} requires one of "
                f"{sorted(EVIDENCE_LEVELS)}"
            )
        source_document = str(
            entry.get("source_document", source.get("document", ""))
        ).strip()
        if evidence != "engineering_assumption" and not source_document:
            raise ValueError(
                f"component environment entry {part_number!r} requires a source document"
            )
        source_sha256 = str(
            entry.get("source_sha256", source.get("sha256", ""))
        ).strip().upper() or None
        if source_sha256 is not None and not re.fullmatch(r"[0-9A-F]{64}", source_sha256):
            raise ValueError(f"component environment entry {part_number!r} has invalid source_sha256")
        records[key] = ComponentEnvironmentRecord(
            part_number=part_number,
            tempco_ppm_per_c=tempco,
            systematic_bias_pct=bias,
            evidence_level=evidence,
            source_name=str(source.get("name")).strip(),
            source_document=source_document,
            source_sha256=source_sha256,
        )
    return ComponentEnvironmentCatalog(
        path=catalog_path,
        sha256=hashlib.sha256(raw).hexdigest().upper(),
        source_name=str(source.get("name")).strip(),
        records=records,
    )


def apply_component_environment_catalog(
    components: Iterable[object], catalog: ComponentEnvironmentCatalog | None
) -> int:
    """Attach exact-part records to mutable component adapters and return coverage."""
    if catalog is None:
        return 0
    matched = 0
    for component in components:
        record = catalog.lookup(getattr(component, "part_number", ""))
        if record is None:
            continue
        component.tempco_ppm_per_c = record.tempco_ppm_per_c
        component.systematic_bias_pct = record.systematic_bias_pct
        component.environment_metadata = record.as_dict()
        provenance = dict(getattr(component, "metadata_provenance", {}) or {})
        label = record.provenance_label()
        if record.tempco_ppm_per_c is not None:
            provenance["tempco_ppm_per_c"] = label
        if record.systematic_bias_pct is not None:
            provenance["systematic_bias_pct"] = label
        component.metadata_provenance = provenance
        matched += 1
    return matched
