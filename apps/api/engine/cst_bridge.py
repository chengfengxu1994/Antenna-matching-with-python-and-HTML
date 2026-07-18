"""Version-isolated bridge to the official CST Studio Suite Python API."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable, Iterable


SENTINEL = "RFMATCH_CST_JSON="


class CSTBridgeError(RuntimeError):
    pass


def _registry_installations() -> list[dict]:
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []
    results = []
    roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for hive, key_name in roots:
        try:
            with winreg.OpenKey(hive, key_name) as parent:
                for index in range(winreg.QueryInfoKey(parent)[0]):
                    try:
                        child_name = winreg.EnumKey(parent, index)
                        with winreg.OpenKey(parent, child_name) as child:
                            display_name = str(winreg.QueryValueEx(child, "DisplayName")[0])
                            if not re.search(r"CST Studio Suite", display_name, re.IGNORECASE):
                                continue
                            install = str(winreg.QueryValueEx(child, "InstallLocation")[0])
                            results.append({"display_name": display_name, "home": install})
                    except OSError:
                        continue
        except OSError:
            continue
    return results


class CSTBridge:
    def __init__(self, worker_path: str | os.PathLike[str], *,
                 installation_roots: Iterable[str | os.PathLike[str]] | None = None,
                 runner: Callable = subprocess.run, cache_seconds: float = 3.0):
        self.worker_path = Path(worker_path).resolve()
        self._explicit_roots = list(installation_roots or [])
        self._runner = runner
        self._cache_seconds = cache_seconds
        self._cache: tuple[float, dict] | None = None
        self._lock = threading.RLock()

    def installations(self) -> list[dict]:
        candidates: list[dict] = []
        configured = os.environ.get("RFMATCH_CST_HOME")
        if configured:
            candidates.append({"display_name": "Configured CST", "home": configured})
        candidates.extend({"display_name": "Configured CST", "home": str(path)} for path in self._explicit_roots)
        candidates.extend(_registry_installations())
        unique = {}
        for item in candidates:
            home = Path(item["home"]).expanduser().resolve()
            python = home / "Opera" / "code" / "bin" / "python.exe"
            interface = home / "AMD64" / "python_cst_libraries" / "cst" / "interface" / "__init__.py"
            if python.is_file() and interface.is_file():
                unique[str(home).casefold()] = {
                    "display_name": item["display_name"],
                    "home": str(home),
                    "python": str(python),
                }
        return sorted(unique.values(), key=lambda item: item["display_name"], reverse=True)

    @staticmethod
    def _encode(payload: dict) -> str:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _invoke(self, installation: dict, payload: dict, timeout: float = 12.0) -> dict:
        completed = self._runner(
            [installation["python"], str(self.worker_path), self._encode(payload)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
        response = None
        for line in (completed.stdout or "").splitlines():
            if line.startswith(SENTINEL):
                response = json.loads(line[len(SENTINEL):])
        if response is None:
            detail = (completed.stderr or completed.stdout or "CST worker returned no result").strip()
            raise CSTBridgeError(detail[-1000:])
        if not response.get("ok"):
            raise CSTBridgeError(response.get("error") or "CST bridge operation failed")
        return response["result"]

    def status(self, *, force: bool = False) -> dict:
        now = time.monotonic()
        with self._lock:
            if not force and self._cache and now - self._cache[0] < self._cache_seconds:
                return dict(self._cache[1])
            installations = self.installations()
            if not installations:
                result = {"available": False, "installations": [], "projects": [],
                          "running_design_environments": []}
            else:
                selected = installations[0]
                try:
                    probe = self._invoke(selected, {"operation": "probe"})
                    result = {"available": True, "installation": selected,
                              "installations": installations, **probe, "error": None}
                except (CSTBridgeError, OSError, subprocess.SubprocessError) as exc:
                    result = {"available": True, "installation": selected,
                              "installations": installations, "projects": [],
                              "running_design_environments": [], "error": str(exc)}
            self._cache = (now, result)
            return dict(result)

    def project_tree(self, pid: int, project_path: str) -> dict:
        status = self.status()
        installation = status.get("installation")
        if not installation:
            raise CSTBridgeError("CST Studio Suite Python API is not available")
        resolved = str(Path(project_path).resolve())
        known = {
            (int(item["pid"]), str(Path(item["path"]).resolve()))
            for item in status.get("projects", [])
        }
        if (int(pid), resolved) not in known:
            raise CSTBridgeError("Selected CST project is not open in the requested Design Environment")
        return self._invoke(installation, {
            "operation": "project_tree", "pid": int(pid), "project_path": resolved,
        })

    def export_touchstone(
        self, pid: int, project_path: str, output_base: str | os.PathLike[str], *,
        allowed_root: str | os.PathLike[str],
    ) -> dict:
        """Ask an explicitly selected open CST project to export Touchstone."""
        status = self.status(force=True)
        installation = status.get("installation")
        if not installation:
            raise CSTBridgeError("CST Studio Suite Python API is not available")
        resolved_project = str(Path(project_path).resolve())
        known = {
            (int(item["pid"]), str(Path(item["path"]).resolve()))
            for item in status.get("projects", [])
        }
        if (int(pid), resolved_project) not in known:
            raise CSTBridgeError("Selected CST project is not open in the requested Design Environment")
        root = Path(allowed_root).resolve()
        destination = Path(output_base).resolve()
        if destination.parent != root:
            raise CSTBridgeError("CST export destination must be inside the configured SNP directory")
        result = self._invoke(installation, {
            "operation": "export_touchstone", "pid": int(pid),
            "project_path": resolved_project, "output_base": str(destination),
        }, timeout=30.0)
        exported = Path(result.get("exported_path", "")).resolve()
        if exported.parent != root or not re.search(r"\.s\d+p$", exported.name, re.IGNORECASE):
            raise CSTBridgeError("CST returned an invalid Touchstone export path")
        return result
