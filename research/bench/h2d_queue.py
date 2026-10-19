"""A small host -> device copy while tier-like reads keep the H2D copy engine busy (one GPU).

Background ("reads"): `cuMemcpyBatchAsync` batches (`kvpool.mover.Driver`, as the tier mover) of `--pieces` pieces
of `--piece-mb` from pinned host memory into HBM on their own stream, topped up to `--depth` batches in flight;
`memcpy`: the same bytes as one plain `cudaMemcpyAsync` per batch (torch copy_). Foreground, every ~1 ms on another
stream, one of:
  ce        a `--small-mb` pinned tensor .to(device, non_blocking=True) (the Pack.copy of a microbatch's metadata)
  ce_hi     the same on a highest-priority stream
  sm        the same bytes by `ssengine.model.sm_h2d.SmH2D` (a kernel reading mapped pinned memory)
  pageable  7 x torch.tensor(<64 ints>, device=...) + synchronize (FlashInfer KDA make_plan, `_kda_plan`)
Per foreground op: host time from before its launch to its completion (spinning on an event after it; includes
~10 us of launch), and GPU time between events before and after it on its idle stream (an event ahead of a copy
can be processed in the copy engine's queue, so this one can hide the wait); per case the reads' GB/s.
usage: python -m research.bench.h2d_queue [--out JSON]"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ssengine.kvpool.mover import Driver
from ssengine.model.sm_h2d import SmH2D


def run_case(bg: str, op: str, depth: int, a, drv: Driver, st_bg, iters: int, flags: int = 0):
    dev = a.dev
    small = torch.randint(0, 1 << 30, (int(a.small_mb * 2**20) // 8,), dtype=torch.int64).pin_memory()
    sm = SmH2D(dev)
    st_fg = torch.cuda.Stream(dev, priority=-5 if op == "ce_hi" else 0)
    side = torch.cuda.Stream(dev)
    piece = int(a.piece_mb * 2**20)
    nslots = a.host_mb * 2**20 // piece
    rng = np.random.default_rng(0)
    inflight: list = []
    marks = []
    nbatch = 0
    drv._attr[5] = flags
    host_ptr, dev_ptr = a.host.data_ptr(), a.devbuf.data_ptr()

    def top_up():
        nonlocal nbatch
        while inflight and inflight[0].query():
            inflight.pop(0)
        while bg != "none" and len(inflight) < depth:
            with torch.cuda.stream(st_bg):
                if bg == "batch":
                    k = rng.choice(nslots, a.pieces, replace=False).astype(np.int64)
                    drv.batch(st_bg.cuda_stream, host_ptr + k * piece, dev_ptr + np.arange(a.pieces) * piece,
                              np.full(a.pieces, piece, dtype=np.int64))
                else:
                    a.devbuf[: a.pieces * piece].copy_(a.host[: a.pieces * piece], non_blocking=True)
                ev = torch.cuda.Event(enable_timing=True)
                ev.record(st_bg)
            if not marks:
                marks.append(ev)
            inflight.append(ev)
            nbatch += 1

    for _ in range(depth * 3):      # fill the queue first
        top_up()
    first_n = nbatch
    res, host_ms = [], []
    for i in range(iters):
        top_up()
        if op != "none":
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            if op == "pageable":
                with torch.cuda.stream(side):
                    e0.record(side)
                    xs = [torch.tensor(list(range(64)), dtype=torch.int32, device=dev) for _ in range(7)]
                    e1.record(side)
                side.synchronize()
                host_ms.append((time.perf_counter() - t) * 1e3)
            else:
                with torch.cuda.stream(st_fg):
                    e0.record(st_fg)
                    if op == "sm":
                        out = sm.to_device(small)
                    else:
                        out = small.to(dev, non_blocking=True)
                    e1.record(st_fg)
                while not e1.query():          # host time to completion: an event before a copy can itself be
                    pass                        # processed in the copy engine's queue, so GPU events hide the wait
                host_ms.append((time.perf_counter() - t) * 1e3)
            res.append((e0, e1))
        time.sleep(0.001)
    torch.cuda.synchronize(dev)
    last = inflight[-1] if inflight else None
    gbs = None
    if bg != "none" and last is not None:
        ms = marks[0].elapsed_time(last)
        gbs = (nbatch - first_n) * a.pieces * piece / (ms / 1e3) / 1e9 if ms > 0 else None
    lat = np.array([e0.elapsed_time(e1) for e0, e1 in res]) if res else np.zeros(0)
    if op == "sm":   # bytes equal
        assert torch.equal(out.cpu(), small), "sm copy differs"
    out = {"bg": bg, "op": op, "depth": depth, "flags": flags, "reads_GBps": gbs,
           "gpu_ms": {q: float(np.percentile(lat, q)) for q in (10, 50, 90, 99)} if len(lat) else None,
           "gpu_ms_max": float(lat.max()) if len(lat) else None}
    if host_ms:
        out["host_ms"] = {q: float(np.percentile(host_ms, q)) for q in (10, 50, 90, 99)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--small-mb", type=float, default=1.0)
    p.add_argument("--piece-mb", type=float, default=1.5)
    p.add_argument("--pieces", type=int, default=32)
    p.add_argument("--host-mb", type=int, default=2048)
    a = p.parse_args()
    a.dev = torch.device("cuda", 0)
    torch.cuda.set_device(a.dev)
    a.host = torch.empty(a.host_mb * 2**20, dtype=torch.uint8).pin_memory()
    a.devbuf = torch.empty(int(a.pieces * a.piece_mb * 2**20), dtype=torch.uint8, device=a.dev)
    drv = Driver()
    st_bg = torch.cuda.Stream(a.dev)
    cases = [("none", "ce", 0), ("none", "sm", 0), ("none", "pageable", 0),
             ("batch", "none", 4), ("batch", "ce", 4), ("batch", "ce_hi", 4), ("batch", "sm", 4),
             ("batch", "pageable", 4),
             ("batch", "none", 16), ("batch", "ce", 16), ("batch", "sm", 16),
             ("memcpy", "ce", 4), ("memcpy", "sm", 4)]
    # warm-up (Triton compile, allocator)
    run_case("none", "sm", 0, a, drv, st_bg, 20)
    out = []
    for bg, op, depth in cases:
        r = run_case(bg, op, depth, a, drv, st_bg, a.iters)
        out.append(r)
        print(json.dumps(r), flush=True)
    r = run_case("batch", "ce", 4, a, drv, st_bg, a.iters, flags=1)   # CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE
    out.append(r)
    print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
