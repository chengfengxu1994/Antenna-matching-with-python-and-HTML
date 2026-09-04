"""Start the API with a Windows-safe asyncio event-loop factory."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parent


def windows_selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Create the selector loop explicitly instead of Uvicorn's IOCP loop."""
    return asyncio.SelectorEventLoop()


def uvicorn_loop_factory(platform_name: str = sys.platform):
    """Return a custom Windows factory; preserve Uvicorn defaults elsewhere."""
    return windows_selector_loop_factory if platform_name == "win32" else "auto"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the RF Matching API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(API_ROOT))
    import uvicorn

    uvicorn.run(
        "api.server:app",
        app_dir=str(API_ROOT),
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop=uvicorn_loop_factory(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
