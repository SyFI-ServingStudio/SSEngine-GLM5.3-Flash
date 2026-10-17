"""Copies on a stage's 4,096-row microbatch path, timed alone on one GPU (CUDA events, L2 flushed before each rep).

  python -m bench.glue_bench [--rows 4096,8192,16384] [--reps 50] [--out x.json]

- `indexer_float`: the DSA indexer's `hidden_states.float()` (glm5next `Indexer.forward`, [T, 4096] BF16 -> FP32,
  3 per stage-2 microbatch) as torch runs it, vs a one-kernel Triton cast (candidate; BF16 -> FP32 is exact, the
  bench checks the bits).
- `edge_clone`: `StageModel._run_graph`'s copy of the graph output before the send ([T, 16384] BF16 `.clone()`).
- `edge_recv_copy`: its copy of the received rows into the graphs' static input (`e.hidden[:T].copy_`).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_to_f32_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m).to(tl.float32), mask=m)


def triton_float(x, block=4096, warps=8):
    y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _bf16_to_f32_kernel[(triton.cdiv(x.numel(), block),)](x, y, x.numel(), BLOCK=block, num_warps=warps)
    return y


def timed(fn, reps, flush):
    ts = []
    for _ in range(reps):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return round(statistics.median(ts), 2)


def sm_clock():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetClockInfo(pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="4096,8192,16384")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    res = {"cases": []}
    for T in map(int, a.rows.split(",")):
        x = torch.randn(T, 4096, device=dev, generator=g).to(torch.bfloat16)
        eq = torch.equal(x.float(), triton_float(x))
        r = {"rows": T, "indexer_float_torch_us": timed(lambda: x.float(), a.reps, flush), "bitwise": eq}
        for block, warps in ((2048, 4), (4096, 8), (8192, 8)):
            r[f"indexer_float_triton_b{block}_w{warps}_us"] = timed(lambda: triton_float(x, block, warps), a.reps,
                                                                     flush)
        h = torch.randn(T, 16384, device=dev, generator=g).to(torch.bfloat16)
        static = torch.empty(max(T, 8192), 16384, dtype=torch.bfloat16, device=dev)
        r["edge_clone_us"] = timed(lambda: h.clone(), a.reps, flush)
        r["edge_recv_copy_us"] = timed(lambda: static[:T].copy_(h), a.reps, flush)
        r["sm_mhz"] = sm_clock()
        res["cases"].append(r)
        print(f"[glue_bench] {r}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
