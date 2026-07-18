import asyncio
from pathlib import Path

from fastapi import HTTPException

from api import server
from engine.cst_bridge import CSTBridgeError


class FakeBridge:
    def __init__(self):
        self.force = None
        self.revision = 0

    def status(self, *, force=False):
        self.force = force
        return {
            "available": True,
            "runtime_version": "2025.0 Release",
            "projects": [{"pid": 99, "path": r"C:\work\antenna.cst", "active": True}],
        }

    def project_tree(self, pid, project_path):
        if pid != 99:
            raise CSTBridgeError("Selected CST project is not open")
        return {"pid": pid, "project_path": project_path, "tree_items": [r"1D Results\S-Parameters"]}

    def export_touchstone(self, pid, project_path, output_base, *, allowed_root):
        self.revision += 1
        output = Path(str(output_base) + ".s1p")
        reflection = self.revision / 10
        output.write_text(
            f"# Hz S RI R 50\n100000000 {reflection} 0\n200000000 0.2 0\n",
            encoding="utf-8",
        )
        return {"exported_path": str(output), "size_bytes": output.stat().st_size}


class FormattingOnlyBridge(FakeBridge):
    def export_touchstone(self, pid, project_path, output_base, *, allowed_root):
        self.revision += 1
        output = Path(str(output_base) + ".s1p")
        output.write_text(
            f"! CST export run {self.revision}\n"
            "# Hz S RI R 50\n100000000 0.1 0\n200000000 0.2 0\n",
            encoding="utf-8",
        )
        return {"exported_path": str(output), "size_bytes": output.stat().st_size}


def test_cst_status_and_project_tree_routes_delegate_to_bridge():
    original = server.cst_bridge
    fake = FakeBridge()
    server.cst_bridge = fake
    try:
        status = asyncio.run(server.cst_status(force=True))
        assert fake.force is True
        assert status["projects"][0]["pid"] == 99

        tree = asyncio.run(server.cst_project_tree(99, r"C:\work\antenna.cst"))
        assert tree["tree_items"] == [r"1D Results\S-Parameters"]

        try:
            asyncio.run(server.cst_project_tree(1, r"C:\work\missing.cst"))
            raise AssertionError("bridge errors must become client errors")
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "not open" in exc.detail
    finally:
        server.cst_bridge = original


def test_cst_export_route_validates_records_and_stores_touchstone(tmp_path):
    original_bridge = server.cst_bridge
    original_snp_dir = server.state.snp_dir
    server.cst_bridge = FakeBridge()
    server.state.snp_dir = str(tmp_path)
    try:
        result = asyncio.run(server.cst_export_touchstone(99, r"C:\work\antenna.cst"))
        assert result["filename"] == "antenna.s1p"
        assert result["num_ports"] == 1
        assert result["freq_count"] == 2
        assert result["provenance"]["ingestion_method"] == "cst_python_bridge"
        assert result["replaced_existing"] is False
        assert (tmp_path / "antenna.s1p").is_file()
    finally:
        server.cst_bridge = original_bridge
        server.state.snp_dir = original_snp_dir


def test_cst_export_replaces_only_the_same_project_revision(tmp_path):
    original_bridge = server.cst_bridge
    original_snp_dir = server.state.snp_dir
    fake = FakeBridge()
    server.cst_bridge = fake
    server.state.snp_dir = str(tmp_path)
    try:
        first = asyncio.run(server.cst_export_touchstone(99, r"C:\work\antenna.cst"))
        second = asyncio.run(server.cst_export_touchstone(99, r"C:\work\antenna.cst"))

        assert first["filename"] == second["filename"] == "antenna.s1p"
        assert second["replaced_existing"] is True
        assert second["previous_sha256"] == first["sha256"]
        assert second["sha256"] != first["sha256"]
        assert second["content_changed"] is True
        assert second["provenance"]["details"]["revision_of_sha256"] == first["sha256"]
        assert sorted(path.name for path in tmp_path.glob("*.s1p")) == ["antenna.s1p"]
    finally:
        server.cst_bridge = original_bridge
        server.state.snp_dir = original_snp_dir


def test_cst_export_preserves_unowned_same_name_file(tmp_path):
    original_bridge = server.cst_bridge
    original_snp_dir = server.state.snp_dir
    server.cst_bridge = FakeBridge()
    server.state.snp_dir = str(tmp_path)
    original_content = "# Hz S RI R 50\n100000000 0.9 0\n200000000 0.8 0\n"
    (tmp_path / "antenna.s1p").write_text(original_content, encoding="utf-8")
    try:
        result = asyncio.run(server.cst_export_touchstone(99, r"C:\work\antenna.cst"))
        assert result["filename"] == "antenna-2.s1p"
        assert result["replaced_existing"] is False
        assert (tmp_path / "antenna.s1p").read_text(encoding="utf-8") == original_content
    finally:
        server.cst_bridge = original_bridge
        server.state.snp_dir = original_snp_dir


def test_cst_export_distinguishes_byte_changes_from_network_changes(tmp_path):
    original_bridge = server.cst_bridge
    original_snp_dir = server.state.snp_dir
    server.cst_bridge = FormattingOnlyBridge()
    server.state.snp_dir = str(tmp_path)
    try:
        first = asyncio.run(server.cst_export_touchstone(99, r"C:\work\antenna.cst"))
        second = asyncio.run(server.cst_export_touchstone(99, r"C:\work\antenna.cst"))
        assert second["sha256"] != first["sha256"]
        assert second["network_sha256"] == first["network_sha256"]
        assert second["content_changed"] is False
    finally:
        server.cst_bridge = original_bridge
        server.state.snp_dir = original_snp_dir
