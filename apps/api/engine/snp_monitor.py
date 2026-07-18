"""Polling monitor for stable Touchstone exports.

CST/HFSS commonly write result files in-place.  Reading a file as soon as it
appears can therefore expose a valid filename with a truncated matrix.  This
monitor records size/mtime signatures per watch session and only parses a
changed export after the signature has remained unchanged for a configured
settling interval.

The monitor is deliberately polling-based: no platform-specific filesystem
service or background thread is required, and every reported path remains
inside the already configured SNP workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import threading
import time
import uuid
from typing import Callable


_SNP_PATTERN = re.compile(r"\.s\d+p$", re.IGNORECASE)


@dataclass(frozen=True)
class FileSignature:
    size: int
    mtime_ns: int


@dataclass
class PendingObservation:
    signature: FileSignature
    stable_since: float


@dataclass
class WatchSession:
    root: Path
    stable_seconds: float
    source: str
    baseline: dict[str, FileSignature]
    pending: dict[str, PendingObservation] = field(default_factory=dict)
    delivered: dict[str, FileSignature] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


class TouchstoneDirectoryMonitor:
    """Manage short-lived, client-owned watches on a configured SNP folder."""

    def __init__(self, parser: Callable[[str, str], object], *, max_sessions: int = 32,
                 clock: Callable[[], float] = time.monotonic):
        self._parser = parser
        self._max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, WatchSession] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _snapshot(root: Path) -> dict[str, FileSignature]:
        snapshot: dict[str, FileSignature] = {}
        if not root.is_dir():
            return snapshot
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                if not _SNP_PATTERN.search(filename):
                    continue
                path = Path(directory, filename)
                try:
                    stat = path.stat()
                except OSError:
                    continue
                relative = str(path.relative_to(root))
                snapshot[relative] = FileSignature(stat.st_size, stat.st_mtime_ns)
        return snapshot

    def start(self, root: str | os.PathLike[str], *, stable_seconds: float = 1.0,
              source: str = "CST") -> dict:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"SNP directory does not exist: {resolved}")
        if not 0.25 <= stable_seconds <= 30.0:
            raise ValueError("stable_seconds must be between 0.25 and 30")
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].started_at)
                self._sessions.pop(oldest, None)
            watch_id = uuid.uuid4().hex
            baseline = self._snapshot(resolved)
            session = WatchSession(resolved, stable_seconds, source, baseline)
            self._sessions[watch_id] = session
        return {
            "watch_id": watch_id,
            "directory": str(resolved),
            "source": source,
            "stable_ms": int(round(stable_seconds * 1000)),
            "baseline_count": len(baseline),
            "started_at": session.started_at,
        }

    def stop(self, watch_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(watch_id, None) is not None

    def status(self, watch_id: str) -> dict:
        now_mono = self._clock()
        with self._lock:
            session = self._sessions.get(watch_id)
            if session is None:
                raise KeyError(watch_id)
            current = self._snapshot(session.root)
            ready: list[dict] = []
            invalid: list[dict] = []
            pending_names: list[str] = []

            deleted = sorted(set(session.baseline) - set(current))
            for name in deleted:
                session.baseline.pop(name, None)
                session.pending.pop(name, None)
                session.delivered.pop(name, None)

            for name, signature in sorted(current.items()):
                if session.baseline.get(name) == signature:
                    session.pending.pop(name, None)
                    continue
                if session.delivered.get(name) == signature:
                    continue
                observation = session.pending.get(name)
                if observation is None or observation.signature != signature:
                    session.pending[name] = PendingObservation(signature, now_mono)
                    pending_names.append(name)
                    continue
                elapsed = now_mono - observation.stable_since
                if elapsed < session.stable_seconds:
                    pending_names.append(name)
                    continue

                path = session.root / name
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    parsed = self._parser(content, name)
                    references = list(getattr(parsed, "port_impedances", []) or [])
                    if not references:
                        references = [getattr(parsed, "reference_resistance", 50.0)] * int(parsed.num_ports)
                    ready.append({
                        "filename": name,
                        "num_ports": int(parsed.num_ports),
                        "freq_count": len(parsed.frequencies),
                        "freq_min_hz": float(min(parsed.frequencies)),
                        "freq_max_hz": float(max(parsed.frequencies)),
                        "reference_impedances_ohm": [float(value.real) for value in references],
                        "size": signature.size,
                        "mtime_ns": signature.mtime_ns,
                        "source": session.source,
                    })
                except (OSError, TypeError, ValueError) as exc:
                    invalid.append({
                        "filename": name,
                        "error": str(exc),
                        "size": signature.size,
                        "mtime_ns": signature.mtime_ns,
                        "source": session.source,
                    })
                session.baseline[name] = signature
                session.delivered[name] = signature
                session.pending.pop(name, None)

            return {
                "watch_id": watch_id,
                "active": True,
                "directory": str(session.root),
                "source": session.source,
                "ready": ready,
                "invalid": invalid,
                "pending": pending_names,
                "deleted": deleted,
                "scanned_files": len(current),
                "server_time": time.time(),
            }
