import base64
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from engine.cst_bridge import CSTBridge, CSTBridgeError, SENTINEL


def _fake_install(root: Path) -> Path:
    (root / "Opera/code/bin").mkdir(parents=True)
    (root / "AMD64/python_cst_libraries/cst/interface").mkdir(parents=True)
    (root / "Opera/code/bin/python.exe").write_bytes(b"")
    (root / "AMD64/python_cst_libraries/cst/interface/__init__.py").write_text("", encoding="utf-8")
    return root


def _payload_from_command(command):
    encoded = command[-1]
    encoded += "=" * (-len(encoded) % 4)
    return json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))


def test_bridge_detects_runtime_and_caches_probe(tmp_path: Path):
    home = _fake_install(tmp_path / "CST25")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        payload = _payload_from_command(command)
        assert payload == {"operation": "probe"}
        result = {
            "runtime_version": "2025.0 Release",
            "running_design_environments": [{"pid": 42, "connected": True}],
            "projects": [{"pid": 42, "path": str(tmp_path / "antenna.cst"), "active": True}],
        }
        return CompletedProcess(command, 0, SENTINEL + json.dumps({"ok": True, "result": result}) + "\n", "")

    bridge = CSTBridge(worker, installation_roots=[home], runner=runner, cache_seconds=60)
    first = bridge.status()
    second = bridge.status()

    assert first["available"] is True
    assert first["runtime_version"] == "2025.0 Release"
    assert first["projects"][0]["pid"] == 42
    assert first["installation"]["home"] == str(home.resolve())
    assert second == first
    assert len(calls) == 1


def test_bridge_reads_only_a_known_open_project_tree(tmp_path: Path):
    home = _fake_install(tmp_path / "CST25")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    project = tmp_path / "antenna.cst"

    def runner(command, **kwargs):
        payload = _payload_from_command(command)
        if payload["operation"] == "probe":
            result = {
                "runtime_version": "2025.0",
                "running_design_environments": [{"pid": 7, "connected": True}],
                "projects": [{"pid": 7, "path": str(project), "active": True}],
            }
        else:
            assert payload == {"operation": "project_tree", "pid": 7, "project_path": str(project.resolve())}
            result = {
                "pid": 7, "project_path": str(project.resolve()), "solver_running": False,
                "tree_items": [r"1D Results\S-Parameters\S1,1"], "tree_item_count": 10,
            }
        return CompletedProcess(command, 0, SENTINEL + json.dumps({"ok": True, "result": result}) + "\n", "")

    bridge = CSTBridge(worker, installation_roots=[home], runner=runner)
    tree = bridge.project_tree(7, str(project))
    assert tree["tree_items"] == [r"1D Results\S-Parameters\S1,1"]

    with pytest.raises(CSTBridgeError, match="not open"):
        bridge.project_tree(8, str(project))


def test_bridge_exports_only_inside_configured_snp_root(tmp_path: Path):
    home = _fake_install(tmp_path / "CST25")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    project = tmp_path / "antenna.cst"
    snp_root = tmp_path / "snp"
    snp_root.mkdir()

    def runner(command, **kwargs):
        payload = _payload_from_command(command)
        if payload["operation"] == "probe":
            result = {
                "runtime_version": "2025.0", "running_design_environments": [],
                "projects": [{"pid": 7, "path": str(project), "active": True}],
            }
        else:
            assert payload["operation"] == "export_touchstone"
            assert payload["output_base"] == str((snp_root / "export").resolve())
            result = {"exported_path": str((snp_root / "export.s2p").resolve()), "size_bytes": 100}
        return CompletedProcess(command, 0, SENTINEL + json.dumps({"ok": True, "result": result}) + "\n", "")

    bridge = CSTBridge(worker, installation_roots=[home], runner=runner)
    exported = bridge.export_touchstone(7, project, snp_root / "export", allowed_root=snp_root)
    assert exported["exported_path"].endswith("export.s2p")

    with pytest.raises(CSTBridgeError, match="inside"):
        bridge.export_touchstone(7, project, tmp_path / "outside", allowed_root=snp_root)


def test_bridge_surfaces_worker_error_without_parsing_cst_noise(tmp_path: Path):
    home = _fake_install(tmp_path / "CST25")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")

    def runner(command, **kwargs):
        output = "CST diagnostic noise\n" + SENTINEL + json.dumps({"ok": False, "error": "connection refused"})
        return CompletedProcess(command, 1, output, "")

    bridge = CSTBridge(worker, installation_roots=[home], runner=runner)
    status = bridge.status()
    assert status["available"] is True
    assert status["projects"] == []
    assert status["error"] == "connection refused"
