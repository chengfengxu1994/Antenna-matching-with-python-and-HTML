"""Versioned, integrity-checked persistence for local RF Matching projects."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_FORMAT = "rfmatch.project"
SCHEMA_VERSION = 2
OLDEST_SUPPORTED_SCHEMA_VERSION = 1
PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ProjectValidationError(ValueError):
    """Raised when a project document is unsafe, corrupt, or unsupported."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(document: dict[str, Any]) -> bytes:
    unsigned = deepcopy(document)
    unsigned.pop("integrity", None)
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON primitives."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def document_digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def sign_document(document: dict[str, Any]) -> dict[str, Any]:
    signed = deepcopy(document)
    signed["integrity"] = {
        "algorithm": "sha256",
        "digest": document_digest(signed),
        "scope": "canonical_document_without_integrity",
    }
    return signed


def _validate_integrity(document: dict[str, Any]) -> None:
    integrity = document.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256":
        raise ProjectValidationError("project integrity metadata is missing or unsupported")
    expected = document_digest(document)
    if integrity.get("digest") != expected:
        raise ProjectValidationError("project integrity check failed")


def _validate_common_fields(document: dict[str, Any]) -> None:
    project_id = document.get("project_id")
    if not isinstance(project_id, str) or not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ProjectValidationError("project_id is invalid")
    if not isinstance(document.get("name"), str) or not document["name"].strip():
        raise ProjectValidationError("project name is required")
    for field in ("created_at", "updated_at", "software", "input", "configuration", "results"):
        if field not in document:
            raise ProjectValidationError(f"project document missing field: {field}")
    for field in ("software", "input", "configuration", "results"):
        if not isinstance(document[field], dict):
            raise ProjectValidationError(f"project field must be an object: {field}")


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (minimum is None or numeric >= minimum)


def _validate_manual_workspace_extension(workspace: Any) -> None:
    if not isinstance(workspace, dict) or workspace.get("schema_version") != 1:
        raise ProjectValidationError("manual_workspace extension schema is invalid")
    active_port = workspace.get("active_input_port")
    if not isinstance(active_port, int) or isinstance(active_port, bool) or not 0 <= active_port <= 63:
        raise ProjectValidationError("manual_workspace active port is invalid")
    if not _finite_number(workspace.get("target_frequency_hz"), minimum=1e-300):
        raise ProjectValidationError("manual_workspace target frequency is invalid")

    networks = workspace.get("working_networks")
    if not isinstance(networks, dict) or len(networks) > 64:
        raise ProjectValidationError("manual_workspace working networks are invalid")
    for port, components in networks.items():
        if not isinstance(port, str) or not port.isdigit() or not 0 <= int(port) <= 63:
            raise ProjectValidationError("manual_workspace working network port is invalid")
        if (
            not isinstance(components, list) or len(components) > 12
            or not all(isinstance(component, dict) for component in components)
        ):
            raise ProjectValidationError("manual_workspace working network components are invalid")

    variants = workspace.get("variants")
    if not isinstance(variants, list) or len(variants) > 12:
        raise ProjectValidationError("manual_workspace variants are invalid")
    variant_ids = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ProjectValidationError("manual_workspace variant must be an object")
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str) or PROJECT_ID_PATTERN.fullmatch(variant_id) is None:
            raise ProjectValidationError("manual_workspace variant_id is invalid")
        variant_ids.append(variant_id)
        if not isinstance(variant.get("name"), str) or not 1 <= len(variant["name"].strip()) <= 100:
            raise ProjectValidationError("manual_workspace variant name is invalid")
        input_port = variant.get("input_port")
        if not isinstance(input_port, int) or isinstance(input_port, bool) or not 0 <= input_port <= 63:
            raise ProjectValidationError("manual_workspace variant port is invalid")
        if not _finite_number(variant.get("target_frequency_hz"), minimum=1e-300):
            raise ProjectValidationError("manual_workspace variant frequency is invalid")
        components = variant.get("components")
        if (
            not isinstance(components, list) or len(components) > 12
            or not all(isinstance(component, dict) for component in components)
        ):
            raise ProjectValidationError("manual_workspace variant components are invalid")
        port_states = variant.get("port_states")
        if (
            not isinstance(port_states, list) or len(port_states) > 64
            or not all(isinstance(state, dict) for state in port_states)
        ):
            raise ProjectValidationError("manual_workspace variant port states are invalid")
        metrics = variant.get("metrics")
        if not isinstance(metrics, dict):
            raise ProjectValidationError("manual_workspace variant metrics are invalid")
        for field in (
            "return_loss_db", "return_loss_improvement_db", "vswr",
            "input_impedance_real", "input_impedance_imag", "maximum_power_balance_error",
        ):
            minimum = 0.0 if field in {"return_loss_db", "maximum_power_balance_error"} else None
            if field == "vswr":
                minimum = 1.0
            if not _finite_number(metrics.get(field), minimum=minimum):
                raise ProjectValidationError(f"manual_workspace metric is invalid: {field}")
        if not isinstance(metrics.get("numeric_core"), str) or not metrics["numeric_core"]:
            raise ProjectValidationError("manual_workspace numeric core is invalid")
        if not isinstance(variant.get("created_at"), str) or not variant["created_at"]:
            raise ProjectValidationError("manual_workspace variant timestamp is invalid")
    if len(variant_ids) != len(set(variant_ids)):
        raise ProjectValidationError("manual_workspace variant_id values must be unique")
    selected = workspace.get("selected_variant_id")
    if selected is not None and selected not in variant_ids:
        raise ProjectValidationError("manual_workspace selected variant does not exist")
    overlays = workspace.get("overlay_variant_ids", [])
    if (
        not isinstance(overlays, list) or len(overlays) > 4
        or not all(isinstance(item, str) for item in overlays)
        or len(overlays) != len(set(overlays))
        or any(item not in variant_ids for item in overlays)
    ):
        raise ProjectValidationError("manual_workspace overlay variants are invalid")


def _validate_version(document: dict[str, Any], schema_version: int) -> None:
    _validate_common_fields(document)
    if schema_version == 1:
        _validate_integrity(document)
        return
    if schema_version == 2:
        if document.get("project_format") != PROJECT_FORMAT:
            raise ProjectValidationError("project_format is missing or unsupported")
        if not isinstance(document.get("extensions"), dict):
            raise ProjectValidationError("project extensions must be an object")
        if "manual_workspace" in document["extensions"]:
            _validate_manual_workspace_extension(document["extensions"]["manual_workspace"])
        history = document.get("migration_history")
        if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
            raise ProjectValidationError("project migration_history must be an array of objects")
        for item in history:
            from_version = item.get("from_version")
            to_version = item.get("to_version")
            source_sha256 = item.get("source_sha256")
            if (
                not isinstance(from_version, int)
                or isinstance(from_version, bool)
                or not isinstance(to_version, int)
                or isinstance(to_version, bool)
                or not (OLDEST_SUPPORTED_SCHEMA_VERSION <= from_version < to_version <= SCHEMA_VERSION)
                or not isinstance(source_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
            ):
                raise ProjectValidationError("project migration_history entry is invalid")
        _validate_integrity(document)
        return
    raise ProjectValidationError(f"unsupported project schema_version: {schema_version!r}")


def migrate_document(document: Any) -> dict[str, Any]:
    """Validate and deterministically migrate a supported snapshot to the current schema."""
    if not isinstance(document, dict):
        raise ProjectValidationError("project document must be a JSON object")
    schema_version = document.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ProjectValidationError(
            f"unsupported project schema_version: {schema_version!r}"
        )
    if schema_version < OLDEST_SUPPORTED_SCHEMA_VERSION or schema_version > SCHEMA_VERSION:
        raise ProjectValidationError(
            f"unsupported project schema_version: {schema_version!r}; "
            f"supported range is {OLDEST_SUPPORTED_SCHEMA_VERSION}–{SCHEMA_VERSION}"
        )

    migrated = deepcopy(document)
    _validate_version(migrated, schema_version)
    if schema_version == 1:
        source_digest = migrated["integrity"]["digest"]
        migrated.pop("integrity", None)
        migrated.update({
            "schema_version": 2,
            "project_format": PROJECT_FORMAT,
            "extensions": {},
            "migration_history": [{
                "from_version": 1,
                "to_version": 2,
                "source_sha256": source_digest,
            }],
        })
        migrated = sign_document(migrated)
        schema_version = 2
    _validate_version(migrated, schema_version)
    return migrated


def validate_document(document: Any) -> dict[str, Any]:
    """Return a current-schema, integrity-checked project document."""
    return migrate_document(document)


def new_project_id() -> str:
    return f"rfm-{uuid.uuid4().hex[:16]}"


class ProjectStore:
    """Atomic JSON project store constrained to one configured directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _path(self, project_id: str) -> Path:
        if not PROJECT_ID_PATTERN.fullmatch(project_id or ""):
            raise ProjectValidationError("project_id is invalid")
        path = (self.root / f"{project_id}.rfmatch.json").resolve()
        if path.parent != self.root:
            raise ProjectValidationError("project path escapes the project store")
        return path

    def _write_document(self, document: dict[str, Any]) -> dict[str, Any]:
        """Validate and atomically persist one already assembled document."""
        document = validate_document(document)
        project_id = document["project_id"]
        path = self._path(project_id)
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{project_id}.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return document

    def save(
        self,
        *,
        name: str,
        input_snapshot: dict[str, Any],
        configuration: dict[str, Any],
        results: dict[str, Any],
        software: dict[str, Any],
        extensions: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ProjectValidationError("project name is required")
        project_id = project_id or new_project_id()
        path = self._path(project_id)
        now = utc_now()
        created_at = now
        migration_history: list[dict[str, Any]] = []
        saved_extensions: dict[str, Any] = {}
        if path.exists():
            existing = self.load(project_id)
            created_at = existing["created_at"]
            migration_history = deepcopy(existing.get("migration_history") or [])
            saved_extensions = deepcopy(existing.get("extensions") or {})
        saved_extensions.update(json_safe(extensions or {}))
        document = sign_document(
            {
                "schema_version": SCHEMA_VERSION,
                "project_format": PROJECT_FORMAT,
                "project_id": project_id,
                "name": name.strip(),
                "created_at": created_at,
                "updated_at": now,
                "migration_history": migration_history,
                "extensions": saved_extensions,
                "software": json_safe(software),
                "input": json_safe(input_snapshot),
                "configuration": json_safe(configuration),
                "results": json_safe(results),
            }
        )
        return self._write_document(document)

    def import_document(
        self,
        document: Any,
        *,
        conflict_policy: str = "copy",
    ) -> dict[str, Any]:
        """Import a signed snapshot without silently replacing local work."""
        if conflict_policy not in {"copy", "replace", "reject"}:
            raise ProjectValidationError("unsupported project import conflict policy")

        validated = validate_document(document)
        source_project_id = validated["project_id"]
        source_digest = validated["integrity"]["digest"]
        destination = self._path(source_project_id)

        if not destination.exists():
            stored = self._write_document(validated)
            return {"status": "imported", "document": stored}

        existing = None
        try:
            existing = self.load(source_project_id)
        except ProjectValidationError:
            # A corrupt local file is still a conflict; only explicit replacement
            # may overwrite it. The safe default creates a new project instead.
            pass
        if existing and existing["integrity"]["digest"] == source_digest:
            return {"status": "unchanged", "document": existing}
        if conflict_policy == "reject":
            raise ProjectValidationError(
                f"project_id already exists with different content: {source_project_id}"
            )
        if conflict_policy == "replace":
            stored = self._write_document(validated)
            return {"status": "replaced", "document": stored}

        copied = deepcopy(validated)
        copied.pop("integrity", None)
        copied["project_id"] = new_project_id()
        copied["name"] = f"{validated['name']} (imported)"
        now = utc_now()
        copied["created_at"] = now
        copied["updated_at"] = now
        extensions = deepcopy(copied.get("extensions") or {})
        extensions["import"] = {
            "source_project_id": source_project_id,
            "source_integrity_sha256": source_digest,
            "imported_at": now,
        }
        copied["extensions"] = extensions
        stored = self._write_document(sign_document(copied))
        return {"status": "copied", "document": stored}

    def load(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        if not path.is_file():
            raise FileNotFoundError(f"project not found: {project_id}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectValidationError(f"project JSON is invalid: {exc}") from exc
        return validate_document(document)

    def replace_document(
        self,
        document: Any,
        *,
        expected_project_id: str,
    ) -> dict[str, Any]:
        """Atomically replace one existing project after full signature validation."""
        validated = validate_document(document)
        if validated["project_id"] != expected_project_id:
            raise ProjectValidationError("replacement project_id does not match target")
        if not self._path(expected_project_id).is_file():
            raise FileNotFoundError(f"project not found: {expected_project_id}")
        return self._write_document(validated)

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        projects: list[dict[str, Any]] = []
        for path in self.root.glob("*.rfmatch.json"):
            project_id = path.name.removesuffix(".rfmatch.json")
            try:
                document = self.load(project_id)
                projects.append(
                    {
                        "project_id": document["project_id"],
                        "name": document["name"],
                        "created_at": document["created_at"],
                        "updated_at": document["updated_at"],
                        "input_filename": document["input"].get("filename", ""),
                        "solutions_count": len(document["results"].get("candidates", [])),
                        "manual_variants_count": len(
                            ((document.get("extensions") or {}).get("manual_workspace") or {}).get("variants", [])
                        ),
                        "schema_version": document["schema_version"],
                        "migrated_from_version": (
                            document["migration_history"][0].get("from_version")
                            if document["migration_history"] else None
                        ),
                        "imported_from_project_id": (
                            (document.get("extensions", {}).get("import") or {}).get(
                                "source_project_id"
                            )
                        ),
                        "status": "valid",
                    }
                )
            except (OSError, ProjectValidationError) as exc:
                projects.append(
                    {
                        "project_id": project_id,
                        "name": path.stem,
                        "updated_at": datetime.fromtimestamp(
                            path.stat().st_mtime, timezone.utc
                        ).isoformat(),
                        "status": "invalid",
                        "error": str(exc),
                    }
                )
        return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)
