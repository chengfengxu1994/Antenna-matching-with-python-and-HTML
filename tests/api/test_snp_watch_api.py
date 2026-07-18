import asyncio
from pathlib import Path
import tempfile
import time

from fastapi import HTTPException

from api import server


VALID_S1P = "# GHZ S RI R 50\n1.0 0.1 0\n1.1 0.2 0\n"


def test_watch_api_discovers_only_new_stable_exports():
    original = server.state.snp_dir
    try:
        with tempfile.TemporaryDirectory() as directory:
            server.state.snp_dir = directory
            Path(directory, "baseline.s1p").write_text(VALID_S1P, encoding="utf-8")
            watch = asyncio.run(server.start_snp_watch(stable_ms=250, source="CST"))
            watch_id = watch["watch_id"]
            assert watch["baseline_count"] == 1

            Path(directory, "latest.s1p").write_text(VALID_S1P, encoding="utf-8")
            pending = asyncio.run(server.snp_watch_status(watch_id))
            assert pending["pending"] == ["latest.s1p"]
            assert pending["ready"] == []

            time.sleep(0.27)
            ready = asyncio.run(server.snp_watch_status(watch_id))
            assert ready["ready"][0]["filename"] == "latest.s1p"
            assert ready["ready"][0]["source"] == "CST"
            assert ready["invalid"] == []

            stopped = asyncio.run(server.stop_snp_watch(watch_id))
            assert stopped["stopped"] is True
            try:
                asyncio.run(server.snp_watch_status(watch_id))
                raise AssertionError("stopped watch should not remain accessible")
            except HTTPException as exc:
                assert exc.status_code == 404
    finally:
        server.state.snp_dir = original

