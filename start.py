"""Cross-platform launcher for the RF Matching desktop web application."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "apps" / "api"
FRONTEND = ROOT / "apps" / "web"
DIST = FRONTEND / "dist"


def command(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise SystemExit(f"Required command not found: {name}")
    return executable


def build_frontend() -> None:
    npm = command("npm")
    if not (FRONTEND / "node_modules").is_dir():
        subprocess.run([npm, "ci"], cwd=FRONTEND, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)


def open_when_ready(url: str, health_url: str) -> None:
    for _ in range(80):
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.25)


def backend_command(host: str, port: int, reload: bool = False) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.server:app",
        "--app-dir",
        str(BACKEND),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        args.append("--reload")
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch RF Matching")
    parser.add_argument("--dev", action="store_true", help="run Vite and API with hot reload")
    parser.add_argument("--build", action="store_true", help="rebuild the frontend before launch")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.build or (not args.dev and not (DIST / "index.html").is_file()):
        print("Building frontend...")
        build_frontend()

    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    processes: list[subprocess.Popen] = []
    try:
        api_process = subprocess.Popen(
            backend_command(args.host, args.port, reload=args.dev),
            cwd=ROOT,
            env=environment,
        )
        processes.append(api_process)

        if args.dev:
            npm = command("npm")
            ui_process = subprocess.Popen(
                [npm, "run", "dev", "--", "--host", args.host],
                cwd=FRONTEND,
                env=environment,
            )
            processes.append(ui_process)
            url = "http://127.0.0.1:3000"
        else:
            url = f"http://127.0.0.1:{args.port}"

        if not args.no_browser:
            threading.Thread(
                target=open_when_ready,
                args=(url, f"http://127.0.0.1:{args.port}/api/health"),
                daemon=True,
            ).start()

        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
        return next((process.returncode for process in processes if process.returncode), 0)
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
