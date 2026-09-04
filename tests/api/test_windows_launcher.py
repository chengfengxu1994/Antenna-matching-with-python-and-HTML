from __future__ import annotations

import run_server


def test_windows_launcher_supplies_selector_loop_directly_to_uvicorn():
    factory = run_server.uvicorn_loop_factory("win32")
    loop = factory()
    try:
        assert isinstance(loop, run_server.asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_non_windows_launcher_preserves_uvicorn_default_loop_factory():
    assert run_server.uvicorn_loop_factory("linux") == "auto"
