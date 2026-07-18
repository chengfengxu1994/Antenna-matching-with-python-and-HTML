"""Durable, content-bound provenance for imported solver/VNA Touchstones."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import uuid


SCHEMA_VERSION = 1
STORE_FILENAME = ".rfmatch-snp-provenance.json"
_DIGEST = re.compile(r"[0-9a-f]{64}")


class SnpProvenanceStore:
    def __init__(self):
        self._lock = threading.RLock()

    @staticmethod
    def _relative_name(root: Path, filename: str) -> str:
        candidate = (root / filename).resolve()
        try:
            relative = candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("Touchstone provenance path escapes the SNP directory") from exc
        name = str(relative)
        if name == STORE_FILENAME or not re.search(r"\.s\d+p$", name, re.IGNORECASE):
            raise ValueError("Touchstone provenance requires a relative *.sNp filename")
        return name

    @staticmethod
    def _read(path: Path) -> dict:
        if not path.is_file():
            return {"schema_version": SCHEMA_VERSION, "entries": {}}
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"SNP provenance store is invalid: {exc}") from exc
        if document.get("schema_version") != SCHEMA_VERSION or not isinstance(document.get("entries"), dict):
            raise ValueError("SNP provenance store has an unsupported schema")
        return document

    def record(self, root: str | os.PathLike[str], filename: str, *, sha256: str,
               source: str, ingestion_method: str, details: dict | None = None) -> dict:
        if not _DIGEST.fullmatch(sha256):
            raise ValueError("Touchstone provenance SHA-256 is invalid")
        root_path = Path(root).resolve()
        name = self._relative_name(root_path, filename)
        store_path = root_path / STORE_FILENAME
        entry = {
            "filename": name,
            "sha256": sha256,
            "source": str(source),
            "ingestion_method": str(ingestion_method),
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        if details:
            entry["details"] = deepcopy(details)
        with self._lock:
            root_path.mkdir(parents=True, exist_ok=True)
            document = self._read(store_path)
            document["entries"][name] = entry
            temporary = store_path.with_name(store_path.name + f".{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, store_path)
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)
        return deepcopy(entry)

    def lookup(self, root: str | os.PathLike[str], filename: str, *, sha256: str) -> dict | None:
        root_path = Path(root).resolve()
        name = self._relative_name(root_path, filename)
        with self._lock:
            document = self._read(root_path / STORE_FILENAME)
            entry = document["entries"].get(name)
            if not isinstance(entry, dict) or entry.get("sha256") != sha256:
                return None
            return deepcopy(entry)

