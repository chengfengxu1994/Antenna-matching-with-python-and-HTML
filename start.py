#!/usr/bin/env python3
"""
RF Matching Tool -- One-click launcher
Supports production mode (backend serves built frontend) and dev mode (backend + Vite HMR).

Usage:
    python start.py            # production (builds frontend, backend on :8000)
    python start.py --no-build # production (skip frontend build, use existing dist)
    python start.py --dev      # dev mode (backend on :8000, Vite HMR on :3000)
"""
import subprocess
import sys
import os
import time
import webbrowser
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = ROOT / "rf-matching" / "backend"
FRONTEND_DIR = ROOT / "rf-matching" / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"

def check_deps():
    """Ensure Python backend dependencies are installed."""
    try:
        import fastapi  # noqa
        import uvicorn  # noqa
        print("  Backend deps: OK")
    except ImportError:
        print("  Installing backend dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r",
             str(BACKEND_DIR / "requirements.txt")],
            check=True,
        )
        print("  Backend deps: installed")

def check_node():
    """Ensure Node.js is available (needed for dev mode or build)."""
    result = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  [ERROR] Node.js is required but not found.")
        print("  Download from: https://nodejs.org/")
        sys.exit(1)
    print(f"  Node: {result.stdout.strip()}")

def install_frontend_deps():
    """Install npm dependencies if node_modules is missing."""
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("  Installing frontend dependencies (npm install)...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), check=True)
        print("  Frontend deps: installed")
    else:
        print("  Frontend deps: OK")

def build_frontend():
    """Build the frontend for production."""
    install_frontend_deps()
    print("  Building frontend (npm run build)...")
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(FRONTEND_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("  [ERROR] Frontend build failed:")
        print(result.stderr)
        sys.exit(1)
    for line in result.stdout.strip().split("\n"):
        if "built in" in line:
            print(f"  Frontend build: {line.strip()}")
            break
    else:
        print("  Frontend build: OK")

def main():
    parser = argparse.ArgumentParser(
        description="RF Matching Tool -- Antenna Matching + Murata LC",
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Development mode: starts backend + Vite HMR on :3000",
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Production mode: skip frontend rebuild (use existing dist)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Backend server port (default: 8000)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Backend bind address (default: 0.0.0.0)",
    )
    args = parser.parse_args()
    port = args.port
    host = args.host

    print()
    print("=" * 60)
    print("  RF Matching Tool")
    print("  Antenna Matching + Component Library")
    print("=" * 60)
    print(f"  Root: {ROOT}")
    print(f"  Python: {sys.version.split()[0]}")

    check_deps()

    if args.dev:
        check_node()
        install_frontend_deps()

        if not args.no_build and not (FRONTEND_DIR / "dist").exists():
            build_frontend()

        backend_url = f"http://localhost:{port}"
        frontend_url = "http://localhost:3000"

        print()
        print("  +- Dev Mode --------------------------------+")
        print(f"  | Backend API:  {backend_url:<33} |")
        print(f"  | Frontend HMR: {frontend_url:<33} |")
        print("  | (Vite proxies /api -> backend)           |")
        print("  +------------------------------------------+")
        print("  | Open the FRONTEND URL in your browser     |")
        print("  | Press Ctrl+C to stop                      |")
        print("  +------------------------------------------+")
        print()

        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.server:app",
             "--host", host, "--port", str(port)],
            cwd=str(BACKEND_DIR),
        )
        time.sleep(2)

        try:
            subprocess.run(["npm", "run", "dev"], cwd=str(FRONTEND_DIR))
        except KeyboardInterrupt:
            pass
        finally:
            backend.terminate()
            backend.wait()
    else:
        check_node()

        # Production mode: always rebuild to ensure latest frontend
        # Use --no-build to skip rebuild and use existing dist
        if args.no_build:
            if not DIST_DIR.exists():
                print("  [ERROR] No dist found. Run without --no-build first.")
                sys.exit(1)
            print("  Frontend dist: using existing")
        else:
            build_frontend()

        url = f"http://localhost:{port}"
        print()
        print("  +- Production Mode -------------------------+")
        print(f"  | Open browser: {url:<39} |")
        print("  | (backend serves frontend at /)           |")
        print("  | Press Ctrl+C to stop                      |")
        print("  +------------------------------------------+")
        print()

        def open_browser():
            time.sleep(2)
            webbrowser.open(url)

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

        try:
            subprocess.run(
                [sys.executable, "-m", "uvicorn", "api.server:app",
                 "--host", host, "--port", str(port)],
                cwd=str(BACKEND_DIR),
            )
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()

