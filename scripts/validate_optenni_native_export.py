"""Validate a native Optenni plot + complex matching-network export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))

from rfmatch_core import (  # noqa: E402
    compare_optenni_native_plot,
    load_optenni_plot_export,
    load_touchstone,
    replay_one_port_dut_through_network,
)


DEFAULT_MANIFEST = (
    ROOT / "benchmarks" / "optenni_exports"
    / "quick_start_0402cs_gjm15_pcsl_manifest.json"
)
DEFAULT_TUTORIAL_ROOT = Path(
    r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def validate(manifest_path: Path, tutorial_root: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    network_path = base / manifest["files"]["matching_network"]["path"]
    plot_path = base / manifest["files"]["plotted_response"]["path"]
    dut_path = tutorial_root / manifest["source_dut"]["tutorial_relative_path"]
    paths = {
        "dut": (dut_path, manifest["source_dut"]["sha256"]),
        "matching_network": (
            network_path, manifest["files"]["matching_network"]["sha256"]
        ),
        "plotted_response": (
            plot_path, manifest["files"]["plotted_response"]["sha256"]
        ),
    }
    integrity = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "expected_sha256": expected,
        }
        for name, (path, expected) in paths.items()
    }
    for item in integrity.values():
        item["matches"] = bool(
            item["exists"] and item["sha256"] == item["expected_sha256"]
        )
    if not all(item["matches"] for item in integrity.values()):
        return {"status": "invalid", "integrity": integrity}

    source_port = int(manifest["selected_candidate"]["network_source_port"]) - 1
    load_port = int(manifest["selected_candidate"]["network_load_port"]) - 1
    plot = load_optenni_plot_export(plot_path)
    replay = replay_one_port_dut_through_network(
        load_touchstone(dut_path),
        load_touchstone(network_path),
        source_port=source_port,
        load_port=load_port,
    )
    comparison = compare_optenni_native_plot(plot, replay)
    bands = manifest["matching_settings"]["target_bands_mhz"]
    band_mask = np.zeros(len(plot.frequencies_hz), dtype=bool)
    for start_mhz, stop_mhz in bands:
        band_mask |= (
            (plot.frequencies_hz >= float(start_mhz) * 1e6)
            & (plot.frequencies_hz <= float(stop_mhz) * 1e6)
        )
    if not np.any(band_mask):
        raise ValueError("manifest target bands contain no exported frequency points")
    limits = manifest["native_complex_replay"]["acceptance"]
    passed = bool(
        comparison.maximum_s11_error_db
        <= float(limits["maximum_s11_error_db"])
        and comparison.maximum_efficiency_error_db
        <= float(limits["maximum_efficiency_error_db"])
    )
    return {
        "status": "valid" if passed else "mismatch",
        "case": manifest["case"],
        "integrity": integrity,
        "comparison": comparison.__dict__,
        "acceptance": limits,
        "band_summary": {
            "points": int(np.count_nonzero(band_mask)),
            "minimum_total_efficiency_db": float(
                np.min(replay.total_efficiency_db[band_mask])
            ),
            "average_total_efficiency_db": float(
                np.mean(replay.total_efficiency_db[band_mask])
            ),
            "best_s11_db": float(np.min(replay.s11_db[band_mask])),
            "worst_s11_db": float(np.max(replay.s11_db[band_mask])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tutorial-root", type=Path, default=DEFAULT_TUTORIAL_ROOT)
    args = parser.parse_args()
    result = validate(args.manifest, args.tutorial_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
