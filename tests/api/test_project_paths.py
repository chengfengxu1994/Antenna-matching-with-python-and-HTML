from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_storage_directories_can_be_moved_to_writable_locations(tmp_path):
    artifacts = tmp_path / "artifacts"
    projects = tmp_path / "projects"
    snp = tmp_path / "snp"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "apps" / "api"),
        "RFMATCH_ARTIFACTS_DIR": str(artifacts),
        "RFMATCH_PROJECTS_DIR": str(projects),
        "RFMATCH_SNP_DIR": str(snp),
    }
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from project_paths import ARTIFACTS_DIR, PROJECTS_DIR, SNP_DIR; "
                "print(json.dumps([str(ARTIFACTS_DIR), str(PROJECTS_DIR), str(SNP_DIR)]))"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(process.stdout) == [str(artifacts), str(projects), str(snp)]
