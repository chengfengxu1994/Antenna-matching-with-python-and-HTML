"""Run the repeatable quality gate for API, numerical core, engine, and web UI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
CORE_SRC = ROOT / "packages" / "rfmatch-core" / "src"


def run(label: str, command: list[str], env: dict[str, str], cwd: Path = ROOT) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-web", action="store_true")
    parser.add_argument("--with-baseline", action="store_true", help="also run the slower Optenni baseline suite")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(API_ROOT), str(CORE_SRC), env.get("PYTHONPATH", "")])
    try:
        run("Python compile", [sys.executable, "-m", "compileall", "-q", "start.py", "apps/api", "packages/rfmatch-core/src", "scripts", "tests"], env)
        run(
            "Python contract and regression tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "packages/rfmatch-core/tests",
                "tests/api",
                "tests/engine",
                "-q",
            ],
            env,
        )
        run("Optenni input inventory", [sys.executable, "scripts/check_reference_cases.py"], env)
        if not args.skip_web:
            npm = shutil.which("npm")
            if not npm:
                raise RuntimeError("npm is required for the web build; use --skip-web only in Python-only environments")
            run("Web production build", [npm, "run", "build"], env, ROOT / "apps" / "web")
        if args.with_baseline:
            run("Optenni numerical baseline", [sys.executable, "scripts/run_optenni_baseline.py"], env)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nQUALITY GATE FAILED: {exc}", file=sys.stderr)
        return 1
    print("\nQUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
