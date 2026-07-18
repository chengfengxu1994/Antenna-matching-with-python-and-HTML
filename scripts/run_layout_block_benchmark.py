"""Generate the fixed EM/VNA S2P layout-block physical cascade baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rfmatch-core" / "src"))
sys.path.insert(0, str(ROOT / "apps" / "api"))

from rfmatch_core import (  # noqa: E402
    Band,
    LineSearchConfig,
    ModelPlacement,
    Problem,
    TransmissionLineOptimizer,
    flip_s2p_ports,
    renormalize_s_parameters,
    cascade_s2p,
    deembed_s2p,
)
from engine.touchstone import parse_touchstone  # noqa: E402
from engine.tuning_service import core_s2p_layout_from_touchstone  # noqa: E402


DEFAULT_LAYOUT = ROOT / "benchmarks" / "layout_blocks" / "matched_launch_0p7db.s2p"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(layout_path: Path = DEFAULT_LAYOUT) -> dict:
    layout_data = parse_touchstone(
        layout_path.read_text(encoding="utf-8"), layout_path.name
    )
    layout, diagnostics = core_s2p_layout_from_touchstone(layout_data)
    frequencies = np.asarray(layout_data.frequencies, dtype=float)
    problem = Problem(
        frequencies,
        np.zeros((len(frequencies), 1, 1), dtype=complex),
        {0: (Band(float(frequencies[0]), float(frequencies[-1])),)},
        50.0,
    )

    def synthesize(location: str):
        placement = ModelPlacement("series", 0, layout)
        config = LineSearchConfig(
            characteristic_impedance_min_ohm=49.9,
            characteristic_impedance_max_ohm=50.1,
            electrical_length_min_deg=0.1,
            electrical_length_max_deg=0.2,
            topologies=("through_line",), restarts=1, iterations=2,
            fixed_dut_side=(placement,) if location == "dut_side" else (),
            fixed_connector_side=(placement,) if location == "connector_side" else (),
        )
        return TransmissionLineOptimizer(
            problem, 0, 1.0e9, config=config
        ).optimize().best

    expected_delivered = 10 ** (-0.7 / 10.0)
    locations = {}
    for location in ("dut_side", "connector_side"):
        candidate = synthesize(location)
        total = np.asarray(candidate.metrics["total_efficiency"])[:, 0]
        loss = np.asarray(candidate.metrics["component_loss"])[:, 0]
        locations[location] = {
            "component_order": [item["comp_type"] for item in candidate.components()],
            "total_efficiency": total.tolist(),
            "component_loss": loss.tolist(),
            "maximum_total_efficiency_error": float(np.max(np.abs(total - expected_delivered))),
            "maximum_component_loss_error": float(np.max(np.abs(loss - (1.0 - expected_delivered)))),
            "maximum_power_balance_error": float(candidate.metrics["maximum_power_balance_error"]),
        }
    directional = np.asarray([[0.1 + 0.02j, 0.2], [0.7j, -0.3 + 0.1j]])
    flipped = flip_s2p_ports(directional)
    at_75 = renormalize_s_parameters(directional, 50.0, 75.0)
    round_trip = renormalize_s_parameters(at_75, 75.0, 50.0)
    left_fixture = np.asarray([[0.03, 0.9], [0.88, -0.02]], dtype=complex)
    right_fixture = np.asarray([[-0.04, 0.91], [0.89, 0.02]], dtype=complex)
    measured = cascade_s2p(left_fixture, directional, right_fixture)
    recovered, deembedding = deembed_s2p(
        measured, left_fixture=left_fixture, right_fixture=right_fixture,
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "case": "fixed reciprocal 0.7 dB EM/VNA S2P launch cascade",
        "input": {
            "path": str(layout_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(layout_path),
            "frequency_points": len(frequencies),
        },
        "layout_diagnostics": diagnostics,
        "expected": {
            "insertion_loss_db": 0.7,
            "delivered_power": expected_delivered,
            "component_loss": 1.0 - expected_delivered,
        },
        "locations": locations,
        "reference_plane_transforms": {
            "directional_original": [[[value.real, value.imag] for value in row] for row in directional],
            "directional_flipped": [[[value.real, value.imag] for value in row] for row in flipped],
            "maximum_renormalization_round_trip_error": float(np.max(np.abs(round_trip - directional))),
            "maximum_deembedding_recovery_error": float(np.max(np.abs(recovered - directional))),
            "deembedding_diagnostics": deembedding,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "layout-block-physical-baseline.json",
    )
    args = parser.parse_args()
    payload = run(args.layout.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
