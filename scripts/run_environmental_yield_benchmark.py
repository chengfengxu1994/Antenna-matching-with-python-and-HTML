"""Benchmark process-bias and environment-aware yield on an Optenni golden DUT."""

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

from rfmatch_core import (  # noqa: E402
    Band,
    Branch,
    CircuitTopology,
    LumpedModel,
    Problem,
    ToleranceModel,
    YieldCriteria,
    load_touchstone,
    monte_carlo_yield,
    tolerance_summary,
)


BASELINE = ROOT / "benchmarks" / "optenni_exports"
DEFAULT_DUT = BASELINE / "optimization_settings_original.s1p"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _scenario(problem, topology, criteria, samples, seed, label, model):
    result = monte_carlo_yield(
        problem,
        topology,
        criteria,
        samples=samples,
        seed=seed,
        distribution="uniform",
        tolerance_model=model,
    )
    return {"name": label, **tolerance_summary(result)}


def run(dut_path: Path, samples: int, seed: int) -> dict:
    dut = load_touchstone(dut_path)
    mask = (dut.frequencies_hz >= 1.7e9) & (dut.frequencies_hz <= 2.5e9)
    frequencies = dut.frequencies_hz[mask]
    problem = Problem(
        frequencies,
        dut.s_parameters[mask],
        {0: (Band(1.7e9, 2.5e9),)},
        dut.z0,
    )
    # Exact values inferred from the exported Optenni PCSL S2P. Loss settings
    # are the same Q/ESR assumptions stored in the native golden manifest.
    inductance_h = 5.915099856061325e-9
    capacitance_f = 0.48398948153925053e-12
    topology = CircuitTopology(
        external_nodes=("input",),
        dut_nodes=("dut",),
        branches=(
            Branch(
                "series_inductor", "input", "dut",
                LumpedModel(
                    "L", "L", inductance_h, tolerance=0.02,
                    q=30.0, q_reference_hz=1e9,
                ),
            ),
            Branch(
                "shunt_capacitor", "dut", None,
                LumpedModel("C", "C", capacitance_f, tolerance=0.02, esr=0.4),
            ),
        ),
    )
    criteria = YieldCriteria(
        minimum_total_efficiency=10 ** (-1.0 / 10.0),
        minimum_average_total_efficiency=10 ** (-0.7 / 10.0),
    )
    control = _scenario(
        problem,
        topology,
        criteria,
        samples,
        seed,
        "Optenni-aligned independent manufacturing control",
        ToleranceModel(),
    )
    environmental = _scenario(
        problem,
        topology,
        criteria,
        samples,
        seed,
        "Engineering what-if: correlated batch plus temperature",
        ToleranceModel(
            batch_correlation=0.7,
            temperature_min_c=-40.0,
            temperature_max_c=85.0,
            inductor_tempco_ppm_per_c=100.0,
            capacitor_tempco_ppm_per_c=-30.0,
        ),
    )
    biased = _scenario(
        problem,
        topology,
        criteria,
        samples,
        seed,
        "Engineering what-if: systematic L/C process bias",
        ToleranceModel(
            inductor_bias_pct=1.0,
            capacitor_bias_pct=-1.0,
        ),
    )
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "case": "Optenni optimization-settings PCSL environmental yield",
        "evidence_scope": {
            "control": "Optenni-validated independent uniform +/-2% manufacturing model",
            "environmental": "engineering assumption; not an Optenni native export",
            "systematic_bias": "engineering assumption; not an Optenni native export",
        },
        "input": {
            "path": str(dut_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(dut_path),
            "active_frequency_points": len(frequencies),
        },
        "network": {
            "topology": "PCSL",
            "series_inductor_h": inductance_h,
            "shunt_capacitor_f": capacitance_f,
            "inductor_q_at_1ghz": 30.0,
            "capacitor_esr_ohm": 0.4,
            "component_tolerance_fraction": 0.02,
        },
        "criteria": {
            "minimum_total_efficiency_db": -1.0,
            "minimum_average_total_efficiency_db": -0.7,
        },
        "samples": samples,
        "seed": seed,
        "scenarios": [control, environmental, biased],
        "delta": {
            "environmental": {
                "yield_fraction": environmental["yield_fraction"] - control["yield_fraction"],
                "p5_score_margin_db": (
                    environmental["score_percentiles_db"]["5"]
                    - control["score_percentiles_db"]["5"]
                ),
            },
            "systematic_bias": {
                "yield_fraction": biased["yield_fraction"] - control["yield_fraction"],
                "p5_score_margin_db": (
                    biased["score_percentiles_db"]["5"]
                    - control["score_percentiles_db"]["5"]
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dut", type=Path, default=DEFAULT_DUT)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "optenni-environmental-yield.json",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    payload = run(args.dut.resolve(), args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
