"""Repository-relative paths shared by the API and numerical engine."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WEB_DIST_DIR = PROJECT_ROOT / "apps" / "web" / "dist"


def configured_path(environment_name: str, fallback: Path) -> Path:
    value = os.environ.get(environment_name)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


PROJECTS_DIR = configured_path("RFMATCH_PROJECTS_DIR", ARTIFACTS_DIR / "projects")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


SNP_DIR = configured_path("RFMATCH_SNP_DIR", DATA_DIR / "snp")
MURATA_DIR = configured_path("RFMATCH_MURATA_DIR", DATA_DIR / "Murata")
