"""Run deterministic rfmatch-core baselines against licensed Optenni tutorial inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "packages" / "rfmatch-core"
sys.path.insert(0, str(CORE_ROOT / "src"))

from rfmatch_core import __version__
from rfmatch_core.benchmarks import (
    CASES,
    MULTIPORT_CASES,
    MULTI_SCENARIO_CASES,
    TUNABLE_CASES,
    SWITCH_CASES,
    run_case,
    run_multi_scenario_case,
    run_multi_scenario_measured_case,
    run_multiport_case,
    run_multiport_measured_case,
    run_tunable_variable_capacitor_case,
    run_tunable_variable_capacitor_synthesis_case,
    run_switch_tutorial_case,
    run_switch_tutorial_synthesis_case,
    run_switch_tutorial_measured_synthesis_case,
)
from rfmatch_core import IsolationTarget, MeasuredSearchConfig


DEFAULT_TUTORIAL_ROOT = Path(r"E:\ProgramX\OptenniLab\Optenni Lab Tutorials")
DEFAULT_COMPONENT_ROOT = Path(r"C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(result: dict) -> dict:
    if result.get("mode") in {"tunable-mdif-measured-s2p", "tunable-mdif-auto-synthesis", "switch-mdif-tutorial-replay", "switch-mdif-auto-synthesis", "switch-mdif-measured-synthesis", "switch-mdif-full-network-measured-synthesis"}:
        return result
    if result.get("mode") in {"shared-ideal-lc", "shared-measured-s2p"}:
        normalized = dict(result)
        normalized["score_db"] = float(result["score_db"])
        normalized["evaluations"] = int(result["evaluations"])
        if result["mode"] == "shared-ideal-lc":
            normalized["elements"] = [
                {
                    "connection": connection,
                    "kind": kind,
                    "port": int(port),
                    "value_si": float(value),
                }
                for connection, kind, port, value in result["elements"]
            ]
        return normalized
    if result.get("mode") == "measured-s2p":
        return {
            "case": result["case"],
            "mode": result["mode"],
            "score_db": float(result["score_db"]),
            "evaluations": int(result["evaluations"]),
            "ideal_evaluations": int(result["ideal_evaluations"]),
            "physical_evaluations": int(result["physical_evaluations"]),
            "component_models_loaded": int(result["component_models_loaded"]),
            "elements": result["elements"],
            "port_scores_db": result["port_scores_db"],
            "frequency_points": result["frequency_points"],
            "maximum_power_balance_error": float(result["maximum_power_balance_error"]),
            "mean_component_loss": float(result["mean_component_loss"]),
            "directed_isolation_db": result["directed_isolation_db"],
            "isolation_targets": result["isolation_targets"],
        }
    elements = []
    for item in result["elements"]:
        connection, kind, *tail = item
        if len(tail) == 1:
            port, value = 0, tail[0]
        else:
            port, value = tail
        elements.append({"connection": connection, "kind": kind, "port": int(port), "value_si": float(value)})
    return {
        "case": result["case"],
        "score_db": float(result["score_db"]),
        "evaluations": int(result["evaluations"]),
        "elements": elements,
        **({"port_scores_db": result["port_scores_db"]} if "port_scores_db" in result else {}),
        **({"frequency_points": result["frequency_points"]} if "frequency_points" in result else {}),
        **({"directed_isolation_db": result["directed_isolation_db"]} if "directed_isolation_db" in result else {}),
        **({"isolation_targets": result["isolation_targets"]} if "isolation_targets" in result else {}),
    }


def isolation_target(value: str) -> IsolationTarget:
    """Parse 1-based SOURCE:DESTINATION:START_HZ:STOP_HZ:MAX_DB[:WEIGHT[:AVG_WEIGHT]]."""
    fields = value.split(":")
    if len(fields) not in (5, 6, 7):
        raise argparse.ArgumentTypeError("expected SOURCE:DESTINATION:START_HZ:STOP_HZ:MAX_DB[:WEIGHT[:AVG_WEIGHT]]")
    try:
        source, destination = int(fields[0]) - 1, int(fields[1]) - 1
        start, stop, maximum = map(float, fields[2:5])
        weight = float(fields[5]) if len(fields) >= 6 else 1.0
        average_weight = float(fields[6]) if len(fields) == 7 else 0.0
        return IsolationTarget(source, destination, start, stop, maximum, weight, average_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tutorial-root",
        type=Path,
        default=Path(os.environ.get("OPTENNI_TUTORIAL_ROOT", DEFAULT_TUTORIAL_ROOT)),
    )
    parser.add_argument("--measured-components", action="store_true", help="use installed vendor S2P parts for multiport cases")
    parser.add_argument("--auto-synthesize-tunable", action="store_true", help="search the tunable case topology and fixed measured parts instead of replaying them")
    parser.add_argument("--auto-synthesize-switch", action="store_true", help="search switch branch types, values, and configuration states instead of replaying tutorial values")
    parser.add_argument("--switch-full-network", action="store_true", help="include the ordered 0–2 element shared input block in measured switch synthesis")
    parser.add_argument(
        "--isolation-target",
        action="append",
        type=isolation_target,
        default=[],
        metavar="SRC:DST:START:STOP:MAX_DB[:WEIGHT[:AVG_WEIGHT]]",
        help="add a directed, 1-based S(DST)(SRC) isolation constraint to a multiport case",
    )
    parser.add_argument(
        "--component-root",
        type=Path,
        default=Path(os.environ.get("OPTENNI_COMPONENT_ROOT", DEFAULT_COMPONENT_ROOT)),
    )
    all_cases = sorted(set(CASES) | set(MULTIPORT_CASES) | set(MULTI_SCENARIO_CASES) | set(TUNABLE_CASES) | set(SWITCH_CASES))
    parser.add_argument("--case", action="append", choices=all_cases)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "optenni-baseline.json")
    args = parser.parse_args()

    selected = args.case or all_cases
    results = []
    for name in selected:
        relative_inputs = None
        if name in SWITCH_CASES:
            if args.isolation_target or args.auto_synthesize_tunable:
                parser.error("switch-tuning does not combine with the selected optimization flags")
            if args.measured_components and not args.auto_synthesize_switch:
                parser.error("switch measured components require --auto-synthesize-switch")
            if args.switch_full_network and not args.measured_components:
                parser.error("--switch-full-network currently requires --measured-components")
            relative_inputs = SWITCH_CASES[name]
            relative_input = relative_inputs[0]
            runner = (
                (lambda root, case: run_switch_tutorial_measured_synthesis_case(
                    root, case, args.component_root, full_network=args.switch_full_network
                ))
                if args.measured_components
                else run_switch_tutorial_synthesis_case if args.auto_synthesize_switch
                else run_switch_tutorial_case
            )
        elif name in TUNABLE_CASES:
            if args.isolation_target:
                parser.error("--isolation-target is supported only for multiport cases")
            relative_inputs = TUNABLE_CASES[name]
            relative_input = relative_inputs[0]
            runner = (
                (lambda root, case: run_tunable_variable_capacitor_synthesis_case(root, case, args.component_root))
                if args.auto_synthesize_tunable
                else (lambda root, case: run_tunable_variable_capacitor_case(root, case, args.component_root))
            )
        elif name in MULTIPORT_CASES:
            relative_input, _ = MULTIPORT_CASES[name]
            if args.measured_components:
                runner = lambda root, case: run_multiport_measured_case(
                    root,
                    case,
                    args.component_root,
                    MeasuredSearchConfig(),
                    tuple(args.isolation_target),
                )
            else:
                runner = lambda root, case: run_multiport_case(root, case, tuple(args.isolation_target))
        elif name in MULTI_SCENARIO_CASES:
            if args.isolation_target:
                parser.error("--isolation-target is supported only for multiport cases")
            relative_inputs, _ = MULTI_SCENARIO_CASES[name]
            if args.measured_components:
                runner = lambda root, case: run_multi_scenario_measured_case(
                    root, case, args.component_root,
                    MeasuredSearchConfig(joint_refine_seeds=8),
                )
            else:
                runner = run_multi_scenario_case
            relative_input = relative_inputs[0]
        else:
            if args.isolation_target:
                parser.error("--isolation-target is supported only for multiport cases")
            _, relative_input, _ = CASES[name]
            runner = run_case
        input_path = args.tutorial_root / relative_input
        started = time.perf_counter()
        result = normalize(runner(args.tutorial_root, name))
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        if name in SWITCH_CASES and "reference_source" in result:
            reference_path = input_path.parent / result["reference_source"]["document"]
            result["reference_source"] = {
                **result["reference_source"],
                "sha256": sha256(reference_path),
            }
        if relative_inputs is None:
            result["input"] = str(relative_input).replace("\\", "/")
            result["input_sha256"] = sha256(input_path)
        else:
            result["inputs"] = [str(item).replace("\\", "/") for item in relative_inputs]
            result["input_sha256"] = {
                str(item).replace("\\", "/"): sha256(args.tutorial_root / item)
                for item in relative_inputs
            }
        results.append(result)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "core_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tutorial_root": str(args.tutorial_root),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Baseline written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
