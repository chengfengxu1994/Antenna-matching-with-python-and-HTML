from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/benchmarks/optenni-environmental-yield.json"
SCRIPT = ROOT / "scripts/run_environmental_yield_benchmark.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location("environmental_yield_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module.run


def test_environmental_yield_artifact_is_reproducible_and_scoped():
    stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 2
    assert stored["input"]["sha256"] == (
        "317BDF0F9CA5FE3FAC111162C5F2C8E6D87E2D13793174126E162612149BFECD"
    )
    assert "not an Optenni native export" in stored["evidence_scope"]["environmental"]
    assert stored["scenarios"][0]["passed_samples"] == 0
    assert stored["scenarios"][0]["variation_model"]["batch_correlation"] == 0.0
    assert stored["scenarios"][1]["variation_model"]["batch_correlation"] == 0.7
    assert stored["scenarios"][1]["variation_model"]["temperature_min_c"] == -40.0
    assert stored["scenarios"][1]["variation_model"]["temperature_max_c"] == 85.0
    assert "not an Optenni native export" in stored["evidence_scope"]["systematic_bias"]
    assert stored["scenarios"][2]["variation_model"]["inductor_bias_pct"] == 1.0
    assert stored["scenarios"][2]["variation_model"]["capacitor_bias_pct"] == -1.0

    reproduced = _load_runner()(
        ROOT / stored["input"]["path"], stored["samples"], stored["seed"]
    )
    assert reproduced["scenarios"] == stored["scenarios"]
    assert reproduced["delta"] == stored["delta"]
