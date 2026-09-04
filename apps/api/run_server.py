"""Start the API with a Windows-safe asyncio event-loop policy."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parent


def configure_event_loop(platform_name: str = sys.platform) -> bool:
    """Use selector sockets on Windows to avoid intermittent IOCP accept faults."""
    if platform_name != "win32":
        return False
    policy_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_factory is None:
        return False
    asyncio.set_event_loop_policy(policy_factory())
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the RF Matching API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    configure_event_loop()
    sys.path.insert(0, str(API_ROOT))
    import uvicorn

    uvicorn.run(
        "api.server:app",
        app_dir=str(API_ROOT),
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
