"""bench/slurm/memguard.py: the start check's arithmetic and the watchdog stopping a thrashing run."""

import importlib.util
import json
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location("memguard", Path(__file__).parents[1] / "bench/slurm/memguard.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)


def test_pinned_gb():
    assert mg.pinned_gb(["sched_good_n400", "--dram-gb", "150", "--ssd-gb", "none"]) == 600
    assert mg.pinned_gb(["x", "--dram-gb", "none", "--ssd-gb", "128"]) == 4 * mg.STAGING_GB
    assert mg.pinned_gb(["x", "--dram-gb", "37.5", "--ssd-gb", "128", "--staging-gb", "4"]) == 4 * 41.5
    assert mg.pinned_gb(["one_32k"]) == 0


def test_precheck(monkeypatch):
    monkeypatch.setattr(mg, "mem_available_gb", lambda: 650.0)
    monkeypatch.setattr(mg, "psi_full_avg10", lambda: 0.0)
    assert mg.precheck(["x", "--dram-gb", "150", "--ssd-gb", "none"]) == 3      # 650 < 600 + 100
    assert mg.precheck(["x", "--dram-gb", "100", "--ssd-gb", "none"]) == 0
    monkeypatch.setattr(mg, "psi_full_avg10", lambda: 7.0)
    assert mg.precheck(["x", "--dram-gb", "100", "--ssd-gb", "none"]) == 3


def test_watch_stops_a_thrashing_run(tmp_path, monkeypatch):
    p = subprocess.Popen(["sleep", "60"], start_new_session=True)
    (tmp_path / "t0.json").write_text(json.dumps({"t0_unix_ns": 0}))
    monkeypatch.setattr(mg, "psi_full_avg10", lambda: 50.0)
    monkeypatch.setattr(mg, "PSI_STOP_S", 1)
    monkeypatch.setattr(mg, "KILL_WAIT_S", 0)
    assert mg.watch(p.pid, tmp_path) == 4
    assert "PSI" in (tmp_path / "INVALID.txt").read_text()
    assert p.wait(timeout=5) != 0


def test_watch_ends_with_the_run(tmp_path):
    p = subprocess.Popen(["sleep", "0.5"], start_new_session=True)
    assert mg.watch(p.pid, tmp_path) == 0 or p.poll() is not None
