"""One stage's forward of a served-like microbatch on one GPU: its GPU time, per-kernel breakdown and launch sites.

  python -m bench.stage_bench [--stage 0] [--spec 45000:3000,10000:1096] [--mode time|kernels|stacks] [--reps 40]
                              [--out x.json]

The microbatch: one chunk per `prefix:append` pair (a request continuing `prefix` cached tokens, on its own KV
blocks; the KV contents are whatever the pool holds), token ids from enwik9 on stage 0 (served-like MoE routing),
seeded random hidden rows on later stages. Modes:
- time: CUDA graphs on, as served; the forwards are enqueued behind a GPU sleep (the host ahead, as stages 1-3 and a
  queued stage 0 run), back to back, one event pair each; median GPU ms and the SM clock (NVML) over the reps.
- kernels: the same under torch.profiler: per forward, each kernel's and memcpy's time and count, and the gaps (span
  minus the union of the GPU activities), by name.
- stacks: eager (no graphs) under torch.profiler with Python stacks: each kernel name's launch sites (the innermost
  frames in ppimpl / vllm / flashinfer), to find the code behind glue kernels.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics

from ppimpl.engine.kv import block_pages
from pathlib import Path
from types import SimpleNamespace as NS

DEFAULT_SPEC = "45000:3000,10000:1096"   # 4,096 rows, 151M pairs: a median stage-2 4,096-row mb of good N=400 (7763)


def build_plan(spec: str, block_tokens: int, first: bool, checkpoint: str):
    pairs = [tuple(int(x) for x in p.split(":")) for p in spec.split(",")]
    total = sum(a for _, a in pairs)
    ids = None
    if first:
        from bench.common import enwik9_tokens
        ids = enwik9_tokens(checkpoint, total)
    chunks, nb, at = [], 1, 0
    for i, (p, a) in enumerate(pairs):
        n_kv = -(-(p + a) // block_tokens)
        chunks.append(NS(req_id=f"r{i}", start=p, num_tokens=a, kv_pages=block_pages(list(range(nb, nb + n_kv))),
                         state_blocks=list(range(nb + n_kv, nb + n_kv + 6)), is_last=True,
                         token_ids=ids[at:at + a] if ids else None))
        nb += n_kv + 6
        at += a
    return NS(mb_id=0, chunks=chunks), nb, total, max(p + a for p, a in pairs)


def sm_clock(dev_index: int) -> int | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(dev_index)
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


def main():
    import torch

    from ppimpl.config import CHECKPOINT
    from ppimpl.model.stage import StageModel, bench_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--mode", choices=["time", "kernels", "stacks"], default="time")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=5, help="time: rounds of --reps forwards")
    ap.add_argument("--mb-tokens", type=int, default=8192, help="the run's largest microbatch (good: 32768)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    B = 8576   # blocks: the plan's, and at least the graph warm-up's (16 requests x 7 blocks + a long prefix)
    plan, nblocks, total, ctx = build_plan(a.spec, B, a.stage == 0, CHECKPOINT)
    graphs = a.mode != "stacks"
    model = StageModel(bench_config(a.mb_tokens, cuda_graphs=graphs), a.stage, dev, num_blocks=max(nblocks + 1, 160),
                       max_model_len=max(ctx, a.mb_tokens) + B)
    g = torch.Generator(device=dev).manual_seed(7)
    hidden = None if model.first else \
        (torch.randn(total, model.hidden_width, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    res = {"stage": a.stage, "spec": a.spec, "mode": a.mode, "tokens": total,
           "graph_bucket": model.prepare(plan).bucket}

    def fwd():
        with torch.inference_mode():
            return model.forward(plan, hidden)

    for _ in range(5):
        fwd()
    torch.cuda.synchronize(dev)
    if a.mode == "time":
        times, clocks = [], []
        for _ in range(a.rounds):
            torch.cuda._sleep(400_000_000)
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(a.reps + 1)]
            ev[0].record()
            for i in range(a.reps):
                fwd()
                ev[i + 1].record()
            for i in range(a.reps):
                ev[i + 1].synchronize()
                if i == a.reps // 2:
                    clocks.append(sm_clock(dev.index))
            times += [ev[i].elapsed_time(ev[i + 1]) for i in range(1, a.reps)]   # skip the first (after the sleep)
        res.update(gpu_ms=round(statistics.median(times), 4), p10=round(sorted(times)[len(times) // 10], 4),
                   p90=round(sorted(times)[len(times) * 9 // 10], 4), n=len(times),
                   sm_mhz=[c for c in clocks if c])
        print(f"[stage_bench] stage {a.stage} {a.spec}: gpu {res['gpu_ms']:.3f} ms (p10 {res['p10']:.3f} p90 "
              f"{res['p90']:.3f}) sm {res['sm_mhz']}", flush=True)
    else:
        from torch.profiler import ProfilerActivity, profile
        reps = a.reps if a.mode == "kernels" else min(a.reps, 3)
        torch.cuda._sleep(400_000_000)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=a.mode == "stacks") as prof:
            for _ in range(reps):
                fwd()
            torch.cuda.synchronize(dev)
        if a.mode == "kernels":
            res.update(kernels(prof, reps))
        else:
            res.update(stacks(prof, reps))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def gpu_events(prof):
    """(start us, end us, name, kind) of every GPU activity (kernels, memcpy, memset)."""
    out = []
    for e in prof.events():
        if e.device_type.name == "CUDA":
            kind = "memcpy" if "Memcpy" in e.name else "memset" if "Memset" in e.name else "kernel"
            out.append((e.time_range.start, e.time_range.end, e.name, kind))
    out.sort()
    return out


def kernels(prof, reps: int) -> dict:
    ev = gpu_events(prof)
    by = collections.defaultdict(lambda: [0, 0.0])
    busy, end = 0.0, None
    for s, e, n, kind in ev:
        by[(kind, n)][0] += 1
        by[(kind, n)][1] += e - s
        if end is None or s >= end:
            busy += e - s
            end = e
        elif e > end:
            busy += e - end
            end = e
    span = ev[-1][1] - ev[0][0]
    rows = sorted(((k, n / reps, t / reps) for k, (n, t) in by.items()), key=lambda r: -r[2])
    print(f"[stage_bench] per forward: span {span / reps / 1e3:.3f} ms, busy {busy / reps / 1e3:.3f} ms, "
          f"gaps {(span - busy) / reps / 1e3:.3f} ms (incl. the span's start)", flush=True)
    for (kind, name), n, t in rows[:60]:
        print(f"  {t:9.1f} us  n {n:6.1f}  {t / max(n, 1e-9):7.1f} us  {kind:6} {name[:120]}")
    return {"span_us": span / reps, "busy_us": busy / reps,
            "rows": [{"kind": k, "name": n, "n": c, "us": t} for (k, n), c, t in rows]}


def stacks(prof, reps: int) -> dict:
    """Kernel name -> launch sites: the CPU op that launched each kernel, its innermost frames in our code."""
    keep = ("ppimpl/", "vllm/", "flashinfer/", "fla/")
    sites = collections.defaultdict(lambda: collections.Counter())
    tot = collections.defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type.name != "CPU" or not e.kernels:
            continue
        p, frames = e, []
        while p is not None and not frames:          # the op's stack, else its nearest parent's
            frames = [f for f in (p.stack or []) if any(k in f for k in keep)][:4]
            p = p.cpu_parent
        if not frames:
            p, chain = e.cpu_parent, []
            while p is not None and len(chain) < 4:
                chain.append(p.name)
                p = p.cpu_parent
            frames = [" < ".join(chain) or "?"]
        for k in e.kernels:
            sites[k.name][(e.name, " < ".join(frames))] += 1
            tot[k.name][0] += 1
            tot[k.name][1] += k.duration
    rows = sorted(tot.items(), key=lambda kv: -kv[1][1])
    out = []
    for name, (n, t) in rows:
        top = sites[name].most_common(4)
        out.append({"name": name, "n": n / reps, "us": t / reps,
                    "sites": [{"op": op, "stack": st, "n": c / reps} for (op, st), c in top]})
        print(f"{t / reps:9.1f} us n {n / reps:6.1f}  {name[:110]}")
        for (op, st), c in top:
            print(f"      {c / reps:5.1f}x {op}  <<  {st[:300]}")
    return {"kernels": out}


if __name__ == "__main__":
    main()
