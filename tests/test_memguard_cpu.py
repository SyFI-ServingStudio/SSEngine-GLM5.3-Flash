"""benchmarks/jobs/memguard.py: the start check's arithmetic and the watchdog stopping a thrashing run."""

import importlib.util
import json
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location("memguard", Path(__file__).parents[1] / "benchmarks/jobs/memguard.py")
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


def _node(tmp_path, n, free_kb, active_file_kb, inactive_file_kb, shmem_kb=0):
    d = tmp_path / f"node{n}"
    d.mkdir()
    (d / "meminfo").write_text(
        f"Node {n} MemTotal:       1000000000 kB\nNode {n} MemFree:        {free_kb} kB\n"
        f"Node {n} Active(file):   {active_file_kb} kB\nNode {n} Inactive(file): {inactive_file_kb} kB\n"
        f"Node {n} Shmem:          {shmem_kb} kB\n")


def test_node_avail_counts_free_and_file_pages_only(tmp_path):
    _node(tmp_path, 0, 100_000_000, 20_000_000, 30_000_000, shmem_kb=400_000_000)
    assert abs(mg.node_avail_gb(0, str(tmp_path)) - 150_000_000 * 1024 / 1e9) < 1e-6


def test_precheck_per_node(monkeypatch):
    """7996: host MemAvailable was ample but node 0 could not hold three 150 GB arenas."""
    monkeypatch.setattr(mg, "mem_available_gb", lambda: 1300.0)
    monkeypatch.setattr(mg, "psi_full_avg10", lambda: 0.0)
    monkeypatch.setattr(mg, "gpu_nodes", lambda: [0, 0, 0, 1])
    avail = {0: 400.0, 1: 900.0}
    monkeypatch.setattr(mg, "node_avail_gb", lambda n: avail[n])
    assert mg.precheck(["x", "--dram-gb", "150", "--ssd-gb", "none"]) == 3      # node 0: 400 < 3 x 150 + 115
    avail[0] = 570.0
    assert mg.precheck(["x", "--dram-gb", "150", "--ssd-gb", "none"]) == 0
    assert mg.precheck(["one_32k"]) == 0                                         # pins nothing: no node check


def test_watch_stops_on_a_short_node_before_t0(tmp_path, monkeypatch):
    p = subprocess.Popen(["sleep", "60"], start_new_session=True)
    monkeypatch.setattr(mg, "node_avail_gb", lambda n: 30.0 if n == 0 else 500.0)
    monkeypatch.setattr(mg, "KILL_WAIT_S", 0)
    run_dir = tmp_path / "run"                                                   # not made yet
    assert mg.watch(p.pid, run_dir, [0, 1]) == 4
    assert "node 0" in (run_dir / "INVALID.txt").read_text()
    assert p.wait(timeout=5) != 0
