from pathlib import Path

from engine.snp_monitor import TouchstoneDirectoryMonitor
from engine.touchstone import parse_touchstone


VALID_S1P = """! stable CST export
# GHZ S RI R 50
1.0 0.1 0.0
2.0 0.2 0.0
"""


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_monitor_ignores_baseline_and_emits_stable_new_export(tmp_path: Path):
    (tmp_path / "existing.s1p").write_text(VALID_S1P, encoding="utf-8")
    clock = FakeClock()
    monitor = TouchstoneDirectoryMonitor(parse_touchstone, clock=clock)
    watch = monitor.start(tmp_path, stable_seconds=0.25, source="CST")

    first = monitor.status(watch["watch_id"])
    assert first["ready"] == []
    assert first["pending"] == []

    (tmp_path / "antenna.s1p").write_text(VALID_S1P, encoding="utf-8")
    pending = monitor.status(watch["watch_id"])
    assert pending["pending"] == ["antenna.s1p"]
    assert pending["ready"] == []

    clock.advance(0.3)
    stable = monitor.status(watch["watch_id"])
    assert stable["pending"] == []
    assert stable["invalid"] == []
    assert stable["ready"][0]["filename"] == "antenna.s1p"
    assert stable["ready"][0]["num_ports"] == 1
    assert stable["ready"][0]["freq_count"] == 2
    assert stable["ready"][0]["source"] == "CST"

    assert monitor.status(watch["watch_id"])["ready"] == []


def test_monitor_never_reports_half_written_file_as_ready(tmp_path: Path):
    clock = FakeClock()
    monitor = TouchstoneDirectoryMonitor(parse_touchstone, clock=clock)
    watch_id = monitor.start(tmp_path, stable_seconds=0.25)["watch_id"]
    export = tmp_path / "sweep.s1p"

    export.write_text("# GHZ S RI R 50\n1.0 0.1\n", encoding="utf-8")
    assert monitor.status(watch_id)["pending"] == ["sweep.s1p"]
    clock.advance(0.3)
    invalid = monitor.status(watch_id)
    assert invalid["ready"] == []
    assert invalid["invalid"][0]["filename"] == "sweep.s1p"

    export.write_text(VALID_S1P, encoding="utf-8")
    # A corrected export with the same path is a new revision.
    assert monitor.status(watch_id)["pending"] == ["sweep.s1p"]
    clock.advance(0.3)
    ready = monitor.status(watch_id)
    assert ready["invalid"] == []
    assert ready["ready"][0]["filename"] == "sweep.s1p"


def test_monitor_reports_overwrite_and_delete(tmp_path: Path):
    export = tmp_path / "antenna.s1p"
    export.write_text(VALID_S1P, encoding="utf-8")
    clock = FakeClock()
    monitor = TouchstoneDirectoryMonitor(parse_touchstone, clock=clock)
    watch_id = monitor.start(tmp_path, stable_seconds=0.25)["watch_id"]

    export.write_text(VALID_S1P + "! rerun\n", encoding="utf-8")
    assert monitor.status(watch_id)["pending"] == ["antenna.s1p"]
    clock.advance(0.3)
    assert monitor.status(watch_id)["ready"][0]["filename"] == "antenna.s1p"

    export.unlink()
    deleted = monitor.status(watch_id)
    assert deleted["deleted"] == ["antenna.s1p"]
