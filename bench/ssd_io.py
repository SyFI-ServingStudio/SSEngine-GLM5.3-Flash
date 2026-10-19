"""SSD tier I/O on this node (lanes/m6 TIERS.md §2.5): the pattern of the KV pool's SSD tier, disk only, no GPU.

Each of `--procs` processes (one per GPU) has its own preallocated file of whole slots (12,857,344 B, GLM-5.3 PP4
balanced) and a host arena, and runs two loops at once:
- reads: whole sessions, a random set of 3-60 file slots each, one read at a time (the member's SSD read thread),
  arriving at `--read-gbs` per process (Poisson; 0: back to back, the maximum; negative: no reads);
- write-back: demotes of 3-60 slots, one at a time, paced to `--write-gbs` per process.
Mechanisms: `threads` (the old SsdTier: one pool of `--threads` threads, os.preadv / os.pwritev of whole slots, reads
and writes in one pool), `uring` (`ppimpl.kvpool.uring.UringFile`: fixed file and buffers, pieces of `--piece-mib`,
`--read-depth` / `--write-depth` pieces in flight, reads first) or `aio` (`ppimpl.kvpool.aio.AioFile`: the same
pieces, depths and queues on a Linux AIO context).

    python -m bench.ssd_io fill  --dir D --procs 4 --file-gb 16
    python -m bench.ssd_io run   --dir D --procs 4 --mode uring --piece-mib 13 --read-depth 8 --write-depth 4 \\
        --read-gbs 0 --write-gbs 3 --secs 25 --node 0 --out results.jsonl --tag name
    python -m bench.ssd_io clean --dir D

Per process it reports read and write GB/s over the timed window, read latency p50 / p99 (service: from the read's
start; and with its queueing behind earlier reads) by session size, write-back latency p50 / p99 per slot (a demote's
time / its slots), and the process's CPU (cores = CPU s / wall s).
Processes are bound to `--node` (CPUs and memory) with numactl.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import random
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SLOT = 12_857_344
ARENA_SLOTS = 128            # 60 read slots + 60 write slots + spare


def _file(d: str, i: int) -> str:
    return os.path.join(d, f"ssd_io_{i}.bin")


def fill(a):
    os.makedirs(a.dir, exist_ok=True)
    n = int(a.file_gb * 1e9 // SLOT)
    chunk = 64 * SLOT
    buf = mmap.mmap(-1, chunk)
    buf.write(np.random.default_rng(0).integers(0, 255, chunk, dtype=np.uint8).tobytes())
    t = time.time()
    for i in range(a.procs):
        fd = os.open(_file(a.dir, i), os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600)
        os.posix_fallocate(fd, 0, n * SLOT)
        for off in range(0, n * SLOT, chunk):
            os.pwritev(fd, [memoryview(buf)[:min(chunk, n * SLOT - off)]], off)
        os.close(fd)
    print(f"filled {a.procs} x {n} slots ({n * SLOT / 1e9:.1f} GB) in {time.time() - t:.1f} s", flush=True)


def clean(a):
    for i in range(a.procs):
        try:
            os.unlink(_file(a.dir, i))
        except FileNotFoundError:
            pass
    try:
        os.rmdir(a.dir)
    except OSError:
        pass


class Threads:
    def __init__(self, fd, n):
        self.fd, self.pool = fd, ThreadPoolExecutor(n)

    def _io(self, kind, slot, ptr):
        buf = (ctypes.c_char * SLOT).from_address(ptr)
        f = os.preadv if kind == "r" else os.pwritev
        if f(self.fd, [buf], slot * SLOT) != SLOT:
            raise OSError("short I/O")

    def submit(self, kind, slot, ptr):
        return self.pool.submit(self._io, kind, slot, ptr)

    def close(self):
        self.pool.shutdown()


class Backend:
    def __init__(self, fd, base, nbytes, a):
        from ppimpl.kvpool.backends import SSD_BACKENDS
        kw = {"submitters": a.aio_submitters} if a.mode == "aio" and a.aio_submitters else {}
        self.u = SSD_BACKENDS[a.mode](fd, [(base, nbytes)], int(a.piece_mib * (1 << 20)), a.read_depth,
                                      a.write_depth, **kw)

    def submit(self, kind, slot, ptr):
        return self.u.submit(kind, slot * SLOT, ptr, SLOT)

    def close(self):
        self.u.close()


def child(a):
    nslots = os.path.getsize(_file(a.dir, a.rank)) // SLOT
    fd = os.open(_file(a.dir, a.rank), os.O_RDWR | os.O_DIRECT)
    arena = mmap.mmap(-1, ARENA_SLOTS * SLOT)
    base = ctypes.addressof(ctypes.c_char.from_buffer(arena))
    ctypes.memset(base, 1, ARENA_SLOTS * SLOT)              # touch (numactl --membind places it)
    io = Threads(fd, a.threads) if a.mode == "threads" else Backend(fd, base, ARENA_SLOTS * SLOT, a)
    rng = random.Random(1000 + a.rank)
    t_start = time.monotonic() + 1.0                          # warm-up second, not counted
    t_end = t_start + a.secs
    reads, writes = [], []
    stop = threading.Event()

    def reader():
        if a.read_gbs < 0:
            return
        t_arr = time.monotonic()
        mean = 31.5 * SLOT
        while not stop.is_set():
            k = rng.randint(3, 60)
            if a.read_gbs > 0:
                t_arr += rng.expovariate(a.read_gbs * 1e9 / mean)
                d = t_arr - time.monotonic()
                if d > 0:
                    time.sleep(d)
            else:
                t_arr = time.monotonic()
            t0 = time.monotonic()
            futs = [io.submit("r", s, base + i * SLOT) for i, s in enumerate(rng.sample(range(nslots), k))]
            for f in futs:
                f.result()
            t1 = time.monotonic()
            reads.append((t0, t1, t_arr, k))

    def writer():
        if a.write_gbs <= 0:
            return
        t_due = time.monotonic()
        while not stop.is_set():
            k = rng.randint(3, 60)
            d = t_due - time.monotonic()
            if d > 0:
                time.sleep(d)
            t0 = time.monotonic()
            futs = [io.submit("w", s, base + (64 + i % 64) * SLOT) for i, s in enumerate(rng.sample(range(nslots), k))]
            for f in futs:
                f.result()
            writes.append((t0, time.monotonic(), k))
            t_due = max(t_due + k * SLOT / (a.write_gbs * 1e9), time.monotonic() - 1.0)

    ts = [threading.Thread(target=reader), threading.Thread(target=writer)]
    for t in ts:
        t.start()
    while time.monotonic() < t_start:
        time.sleep(0.01)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    w0 = time.monotonic()
    while time.monotonic() < t_end:
        time.sleep(0.05)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    w1 = time.monotonic()
    stop.set()
    for t in ts:
        t.join()
    io.close()
    os.close(fd)
    win = [r for r in reads if t_start <= r[0] and r[1] <= t_end]
    rb = sum(k for t0, t1, _, k in reads if t_start <= t1 <= t_end) * SLOT
    wb = sum(k for t0, t1, k in writes if t_start <= t1 <= t_end) * SLOT
    out = dict(rank=a.rank, read_gbs=rb / a.secs / 1e9, write_gbs=wb / a.secs / 1e9, reads=len(win),
               cpu_cores=(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) / (w1 - w0), lat={})
    for name, lo, hi in (("3-10", 3, 10), ("11-30", 11, 30), ("31-60", 31, 60), ("all", 3, 60)):
        sv = [1e3 * (t1 - t0) for t0, t1, _, k in win if lo <= k <= hi]
        qd = [1e3 * (t1 - ta) for t0, t1, ta, k in win if lo <= k <= hi]
        if sv:
            out["lat"][name] = dict(n=len(sv), p50=float(np.percentile(sv, 50)), p99=float(np.percentile(sv, 99)),
                                    q_p50=float(np.percentile(qd, 50)), q_p99=float(np.percentile(qd, 99)))
    ws = [1e3 * (t1 - t0) / k for t0, t1, k in writes if t_start <= t0 and t1 <= t_end]
    if ws:
        out["write_ms_per_slot"] = dict(n=len(ws), p50=float(np.percentile(ws, 50)), p99=float(np.percentile(ws, 99)))
    print(json.dumps(out), flush=True)


def run(a):
    args = [x for x in sys.argv[2:]]
    ps = [subprocess.Popen(["numactl", f"--cpunodebind={a.node}", f"--membind={a.node}", sys.executable, "-m",
                            "bench.ssd_io", "child", *args, "--rank", str(i)], stdout=subprocess.PIPE, text=True)
          for i in range(a.procs)]
    rows = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
    assert all(p.returncode == 0 for p in ps)
    cfg = {k: getattr(a, k) for k in ("mode", "threads", "piece_mib", "read_depth", "write_depth", "read_gbs",
                                      "write_gbs", "secs", "procs")}
    rec = dict(tag=a.tag, cfg=cfg, procs=rows)
    agg = lambda key: [round(r[key], 2) for r in rows]
    lat = lambda b, f: [round(r["lat"].get(b, {}).get(f, float("nan")), 1) for r in rows]
    wl = [tuple(round(r.get("write_ms_per_slot", {}).get(f, float("nan")), 2) for f in ("p50", "p99")) for r in rows]
    print(f"{a.tag}: read GB/s {agg('read_gbs')} write {agg('write_gbs')} cores {agg('cpu_cores')}; write ms per "
          f"slot p50/p99 {wl}")
    for b in ("3-10", "11-30", "31-60"):
        print(f"  {b:>5} slots: p50 {lat(b, 'p50')} p99 {lat(b, 'p99')} ms; with queueing p50 {lat(b, 'q_p50')} "
              f"p99 {lat(b, 'q_p99')}")
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fill", "run", "clean", "child"])
    p.add_argument("--dir", required=True)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--file-gb", type=float, default=16.0)
    p.add_argument("--mode", choices=["threads", "uring", "aio"], default="uring")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--piece-mib", type=float, default=13.0)     # >= a slot: one request per slot
    p.add_argument("--aio-submitters", type=int)                # aio: io_submit threads (default aio.SUBMITTERS)
    p.add_argument("--read-depth", type=int, default=8)
    p.add_argument("--write-depth", type=int, default=4)
    p.add_argument("--read-gbs", type=float, default=0.0)
    p.add_argument("--write-gbs", type=float, default=3.0)
    p.add_argument("--secs", type=float, default=25.0)
    p.add_argument("--node", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--rank", type=int, default=0)
    a = p.parse_args()
    {"fill": fill, "run": run, "clean": clean, "child": child}[a.cmd](a)


if __name__ == "__main__":
    main()
