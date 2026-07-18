"""Run CST's version-matched Python API and emit one machine-readable result.

This script is launched by the product through the Python interpreter bundled
with the detected CST installation. CST may write diagnostic text to stdout;
the parent therefore reads only the line prefixed with ``RFMATCH_CST_JSON=``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import sys
import time


SENTINEL = "RFMATCH_CST_JSON="


def _decode_payload(value: str) -> dict:
    padding = "=" * (-len(value) % 4)
    return json.loads(base64.urlsafe_b64decode(value + padding).decode("utf-8"))


def _project_filename(project) -> str | None:
    value = getattr(project, "filename", None)
    if callable(value):
        value = value()
    return str(value) if value else None


def _probe(interface) -> dict:
    raw_version = str(interface.DesignEnvironment.version())
    matches = re.findall(r"20\d{2}\.\d+\s+Release[^\r\n]*", raw_version)
    version = matches[-1] if matches else raw_version.splitlines()[0].strip()
    environments = []
    projects = []
    for raw_pid in interface.running_design_environments():
        pid = int(raw_pid)
        item = {"pid": pid, "connected": False, "projects": [], "active_project": None}
        try:
            design_environment = interface.DesignEnvironment.connect(pid)
            item["connected"] = bool(design_environment.is_connected())
            item["projects"] = [str(value) for value in design_environment.list_open_projects()]
            if design_environment.has_active_project():
                item["active_project"] = _project_filename(design_environment.active_project())
            for project_path in item["projects"]:
                projects.append({"pid": pid, "path": project_path, "active": project_path == item["active_project"]})
        except Exception as exc:  # CST exceptions are runtime-version specific.
            item["error"] = str(exc)
        environments.append(item)
    return {
        "runtime_version": version,
        "running_design_environments": environments,
        "projects": projects,
    }


def _project_tree(interface, payload: dict) -> dict:
    pid = int(payload["pid"])
    project_path = str(Path(payload["project_path"]).resolve())
    design_environment = interface.DesignEnvironment.connect(pid)
    project = design_environment.get_open_project(project_path)
    model3d = project.model3d
    if model3d is None:
        return {"pid": pid, "project_path": project_path, "solver_running": False, "tree_items": []}
    items = [str(value) for value in model3d.get_tree_items(timeout=10)]
    result_items = [value for value in items if "S-Parameters" in value or "S-Parameter" in value]
    return {
        "pid": pid,
        "project_path": project_path,
        "solver_running": bool(model3d.is_solver_running(timeout=5)),
        "active_solver": str(model3d.get_active_solver_name(timeout=5)),
        "tree_items": result_items,
        "tree_item_count": len(items),
    }


def _export_touchstone(interface, payload: dict) -> dict:
    """Export the currently available project S-parameters through CST itself."""
    pid = int(payload["pid"])
    project_path = str(Path(payload["project_path"]).resolve())
    output_base = Path(payload["output_base"]).resolve()
    if any(character in str(output_base) for character in ('"', "\r", "\n")):
        raise ValueError("CST export path contains unsupported characters")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    design_environment = interface.DesignEnvironment.connect(pid)
    project = design_environment.get_open_project(project_path)
    model3d = project.model3d
    if model3d is None:
        raise RuntimeError("selected CST project has no 3D model")
    if model3d.is_solver_running(timeout=5):
        raise RuntimeError("CST solver is still running; wait for results before exporting")

    before = {
        str(path.resolve()): path.stat().st_mtime_ns
        for path in output_base.parent.glob(output_base.name + "*") if path.is_file()
    }
    from cst.post_processing.s_parameters import export_touchstone
    export_touchstone(project, str(output_base))

    # CST chooses the .sNp suffix from the number of ports. Give its file
    # writer a short grace period and return only the file created/updated by
    # this operation.
    deadline = time.monotonic() + 5.0
    exported = []
    while time.monotonic() < deadline:
        candidates = [path for path in output_base.parent.glob(output_base.name + "*") if path.is_file()]
        exported = [
            path for path in candidates
            if re.search(r"\.s\d+p$", path.name, re.IGNORECASE)
            and path.stat().st_mtime_ns != before.get(str(path.resolve()))
            and path.stat().st_size > 0
        ]
        if exported:
            break
        time.sleep(0.1)
    if not exported:
        raise RuntimeError("CST completed the export but no *.sNp file was produced")
    newest = max(exported, key=lambda path: path.stat().st_mtime_ns)
    return {
        "pid": pid,
        "project_path": project_path,
        "exported_path": str(newest.resolve()),
        "size_bytes": newest.stat().st_size,
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("expected one encoded request")
    payload = _decode_payload(sys.argv[1])
    import cst.interface as interface

    operation = payload.get("operation", "probe")
    if operation == "probe":
        result = _probe(interface)
    elif operation == "project_tree":
        result = _project_tree(interface, payload)
    elif operation == "export_touchstone":
        result = _export_touchstone(interface, payload)
    else:
        raise ValueError(f"unsupported CST bridge operation: {operation}")
    print(SENTINEL + json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(SENTINEL + json.dumps({
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }, ensure_ascii=False))
        raise SystemExit(1)
