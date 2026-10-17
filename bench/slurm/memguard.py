"""Host-memory guards for runs that pin host memory (bench/slurm/m6.sbatch).

    python bench/slurm/memguard.py precheck -- <bench.run args>      # exit 3: do not start (reason on stdout)
    python bench/slurm/memguard.py watch --pid PID --run-dir DIR     # stop the run if the host thrashes

precheck: MemAvailable must exceed the run's pinned host memory (4 GPUs x (DRAM + staging with SSD)) plus 100 GB,
and the memory PSI's `full avg10` must be under 5%. A run started on a thrashing host measures the host.

watch (from the run's t0.json on, i.e. the timed part): memory PSI `full avg10` over 20% for 30 s in a row, or a
process of the run (PID and its descendants) in state D for over 10 s in a row, marks the run invalid
(DIR/INVALID.txt with the reason) and stops it (SIGTERM to PID's process group, SIGKILL 20 s later).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

GPUS = 4
STAGING_GB = 8.0          # TierConfig.staging_gb's default
HEADROOM_GB = 100.0
PSI_START = 5.0           # % full avg10 to start
PSI_STOP, PSI_STOP_S = 20.0, 30
D_STOP_S = 10
KILL_WAIT_S = 20


def psi_full_avg10() -> float:
    for line in open("/proc/pressure/memory"):
        if line.startswith("full"):
            return float(line.split()[1].split("=")[1])
    return 0.0


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    return 0.0


def pinned_gb(args: list[str]) -> float:
    """Host memory the run pins over its GPUs, from bench.run's tier flags (none: 0)."""
    def opt(name):
        return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None

    dram, ssd, staging = opt("--dram-gb"), opt("--ssd-gb"), opt("--staging-gb")
    per = float(dram) if dram not in (None, "none") else 0.0
    if ssd not in (None, "none"):
        per += float(staging) if staging else STAGING_GB
    return GPUS * per


def precheck(args: list[str]) -> int:
    need, avail, psi = pinned_gb(args) + HEADROOM_GB, mem_available_gb(), psi_full_avg10()
    ok = avail > need and psi < PSI_START
    print(f"memguard precheck: MemAvailable {avail:.0f} GB (need > {need:.0f}: pinned {need - HEADROOM_GB:.0f} + "
          f"{HEADROOM_GB:.0f}), memory PSI full avg10 {psi:.1f}% (need < {PSI_START:.0f}): {'start' if ok else 'NOT STARTED'}",
          flush=True)
    return 0 if ok else 3


def descendants(pid: int) -> set[int]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo += kids.get(p, [])
    return out


def state(pid: int) -> str | None:
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def watch(pid: int, run_dir: Path) -> int:
    armed = False
    high_since = None
    d_since: dict[int, float] = {}
    while True:
        if state(pid) in (None, "Z"):          # gone (a zombie until its parent waits)
            return 0
        now = time.monotonic()
        if not armed:
            armed = (run_dir / "t0.json").exists()
            time.sleep(1)
            continue
        reason = None
        psi = psi_full_avg10()
        high_since = (high_since or now) if psi > PSI_STOP else None
        if high_since is not None and now - high_since >= PSI_STOP_S:
            reason = f"memory PSI full avg10 over {PSI_STOP:.0f}% for {PSI_STOP_S} s (now {psi:.1f}%)"
        procs = descendants(pid)
        for p in procs:
            if state(p) == "D":
                d_since.setdefault(p, now)
                if now - d_since[p] >= D_STOP_S and reason is None:
                    reason = f"process {p} in state D for {now - d_since[p]:.0f} s"
            else:
                d_since.pop(p, None)
        if reason:
            msg = f"INVALID {time.strftime('%Y-%m-%dT%H:%M:%S')}: {reason}; MemAvailable {mem_available_gb():.0f} GB"
            print(f"memguard: {msg}; stopping the run", flush=True)
            (run_dir / "INVALID.txt").write_text(msg + "\n")
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(KILL_WAIT_S)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return 4
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precheck")
    p.add_argument("args", nargs=argparse.REMAINDER)
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--run-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "precheck":
        sys.exit(precheck([x for x in a.args if x != "--"]))
    sys.exit(watch(a.pid, a.run_dir))


if __name__ == "__main__":
    main()
