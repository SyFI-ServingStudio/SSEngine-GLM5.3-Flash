"""One kernel's time against the SM clock and board power while it alone loads the GPU (1 GPU).

Each kernel runs back to back for `--seconds` from wherever the GPU is (cold at job start). Every `--every` calls the
batch's mean call time (CUDA events) is recorded with the SM clock, board power and temperature that NVML reports
right after the batch is queued (so during it). As the board heats toward its power cap the clock falls, giving
time-vs-clock pairs for the same launches and inputs. Per kernel: ms by 25-MHz bin, ms by 50-W bin, the fit
ms = a + b / MHz (least squares), and the fit's ms at `--at` MHz (the serving clocks) and at 1965 MHz.

Kernels (`--kernels`, in order):
  - `moe`: the trtllm NVFP4 fused MoE (FC1 + FC2 + routing + finalize as served), 4,096 tokens, tactic [128,41], the
    first 4,096-row stage-2 microbatch's MoE layer 0 of `--experts` (PPIMPL_LOG_EXPERTS run) as routing;
  - `mla`: TRTLLM sparse MLA, 4,096 queries of one request at prefix 30,104 (2,176 indices each), as
    research.bench.sparse_mla_shapes builds it.

  python -m research.bench.clock_curve --experts <run>/experts_stage2.jsonl --kernels moe,mla --seconds 15 --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def nvml():
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return lambda: (pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, pynvml.nvmlDeviceGetTemperature(h, 0))


def make(kernel: str, experts: Path | None):
    import torch
    if kernel == "moe":
        from research.bench import moe_tactics as M
        line = next(r for r in map(json.loads, open(experts)) if sum(r["counts"][0]) == 4096 * 8)
        M.Forced.install()
        set_routing, call = M.build_call(4096)
        set_routing(line["counts"][0])
        M.Forced.tactic = (128, 41)
        return call
    if kernel == "mla":
        from research.bench import sparse_mla_shapes as S
        idx, cnt, ct = S.build_indices(torch, [{"prefix": 30104, "append": 4096}], pool_blocks=None, seed=0,
                                       device="cuda")
        ops = S.make_operands(torch, idx, cnt, ct, "cuda")
        return S.launcher(ops, torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"))
    raise SystemExit(f"unknown kernel {kernel}")


def fit(points, at):
    """ms = a + b / MHz by least squares; None if the clock never moved."""
    xs = [1.0 / p["mhz"] for p in points]
    ys = [p["ms"] for p in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = (sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n) ** 0.5
    return {"a_ms": round(a, 5), "b_ms_mhz": round(b, 2), "rms_resid_ms": round(res, 5),
            "pred_ms": {str(m): round(a + b / m, 4) for m in [*at, 1965]},
            "a_share_at_1965": round(a / (a + b / 1965), 3)}


def binned(points, key, width):
    out = {}
    for p in points:
        out.setdefault(int(p[key] // width * width), []).append(p["ms"])
    return {k: {"n": len(v), "ms": round(sum(v) / len(v), 4)} for k, v in sorted(out.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="moe,mla")
    ap.add_argument("--experts", type=Path)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--at", default="1576,1637")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    read = nvml()
    at = [int(x) for x in a.at.split(",")]
    res = {"job": os.environ.get("SLURM_JOB_ID"), "kernels": {}}
    for k in a.kernels.split(","):
        run = make(k, a.experts)
        run()
        torch.cuda.synchronize()
        idle = read()
        pts, t0 = [], time.time()
        while time.time() < t0 + a.seconds:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(a.every):
                run()
            e.record()
            mhz, w, temp = read()
            e.synchronize()
            pts.append({"t": round(time.time() - t0, 3), "ms": round(s.elapsed_time(e) / a.every, 5), "mhz": mhz,
                        "w": round(w), "temp": temp})
        r = {"start": {"mhz": idle[0], "w": round(idle[1]), "temp": idle[2]}, "n": len(pts),
             "by_mhz": binned(pts, "mhz", 25), "by_w": binned(pts, "w", 50), "fit": fit(pts, at),
             "first": pts[:3], "last": pts[-3:], "points": pts}
        res["kernels"][k] = r
        print(json.dumps({"kernel": k, **{x: r[x] for x in ("start", "n", "by_mhz", "by_w", "fit", "first", "last")}}),
              flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res))
        del run
        torch.cuda.empty_cache()
        time.sleep(5)       # let the board settle a little before the next kernel


if __name__ == "__main__":
    main()
