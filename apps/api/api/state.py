"""Application state and portable data-path defaults for the desktop API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os

from project_paths import MURATA_DIR, SNP_DIR

DEFAULT_SNP_DIR = str(SNP_DIR)
DEFAULT_MURATA_DIR = str(MURATA_DIR)
DEFAULT_OPTENNI_COMPONENT_DIR = os.environ.get(
    "RFMATCH_OPTENNI_COMPONENT_DIR",
    "",
)


class AppState:
    """Mutable state for the current local desktop session.

    Keeping all session data in one object avoids module-level state that can be
    accidentally shadowed by route functions. This remains intentionally
    single-user; a future hosted service should replace it with per-session
    storage and injected services.
    """

    def __init__(self) -> None:
        self.snp_dir = DEFAULT_SNP_DIR
        self.murata_dir = DEFAULT_MURATA_DIR
        self.db_path = ""
        self.loaded_snp: Any = None
        self.loaded_snp_filename = ""
        self.loaded_snp_provenance: dict[str, Any] | None = None
        self.component_library: Any = None
        self.full_component_library: Any = None
        self.optenni_component_dir = DEFAULT_OPTENNI_COMPONENT_DIR
        self.environment_metadata_path = ""
        self.component_environment_catalog: Any = None
        self.tunable_component_library: Any = None
        self.db_library: Any = None
        self.use_db = True
        self.optimizer: Any = None
        self.last_solutions: list[Any] = []
        self.last_joint_results: Any = None
        self.last_multi_scenario_results: Any = None
        self.per_port_efficiency_data: dict[int, Any] = {}
        self.global_efficiency_data: Any = None

    def database_candidates(self) -> list[tuple[str, str]]:
        """Return configured and conventional databases in priority order."""
        candidates: list[tuple[str, str]] = []
        explicit = os.environ.get("RFMATCH_DB_PATH")
        if explicit:
            candidates.append((str(Path(explicit).expanduser()), "configured"))
        root = Path(self.murata_dir)
        candidates.extend(
            [
                (str(root / "full_components.db"), "full"),
                (str(root / "murata_components.db"), "murata"),
            ]
        )
        return candidates

    def clear_efficiency(self, port_index: int = -1) -> None:
        if port_index < 0:
            self.per_port_efficiency_data.clear()
            self.global_efficiency_data = None
        else:
            self.per_port_efficiency_data.pop(port_index, None)
