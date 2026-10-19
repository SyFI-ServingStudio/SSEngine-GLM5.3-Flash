"""Stage forward host cost vs GPU time for one request continuing a cached prefix (a round >= 1 microbatch).

  python -m research.bench.host_front [--stage 0] [--prefix 2048] [--append 256] [--reps 30] [--out x.json] [--profile]

Two timings of the same forward (CUDA graphs on, as the server runs):
- `direct`: the forward enqueued on an idle GPU, as stage 0 runs it: compute = start event (host t = 0) -> end
  event. The GPU waits for the host wherever the host is slower than the kernels it launches.
- `ahead`: the same forward enqueued behind a GPU sleep, so the host is ahead of the GPU, as stages 1-3 run it
  (they enqueue before the upstream hidden state arrives): compute = the GPU time alone.
direct - ahead is what the host costs the stage on the GPU timeline. With --idle-ms X also `direct_sleep` (the
thread sleeps X ms before each forward, as a worker blocks on its plan socket between microbatches) and
`direct_spin` (it busy-waits X ms instead): what an idle, cold host thread adds. Also the host time of `forward` (perf
counter), and with --profile one cProfile of a direct forward (top functions by own and by cumulative time).
With --stamps, low-overhead host timings of direct forwards (ssengine/runtime/host_stamps.py): when each graph
segment and eager break starts after the forward's start, each chosen function's inclusive time, GC time.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import random
import statistics
import time

from ssengine.engine.kv import block_pages
from pathlib import Path
from types import SimpleNamespace as NS

def main():
    import torch

    from ssengine.model.stage import StageModel, bench_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--append", type=int, default=256)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--stamps", action="store_true")
    ap.add_argument("--idle-ms", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    model = StageModel(bench_config(8192, cuda_graphs=True), a.stage, dev, num_blocks=64, max_model_len=8192 + 8576)
    B = model.kv.block_tokens
    n_kv = -(-(a.prefix + a.append) // B)
    assert n_kv + 7 <= model.num_blocks, (n_kv, model.num_blocks, B)
    plan = NS(mb_id=0, chunks=[NS(req_id="r", start=a.prefix, num_tokens=a.append,
                                  kv_pages=block_pages(list(range(1, 1 + n_kv))),
                                  is_last=True, state_blocks=list(range(1 + n_kv, 7 + n_kv)),
                                  token_ids=[random.Random(a.append).randrange(1, 150_000) for _ in range(a.append)])])
    # (varied ids: one repeated id routes every token to the same experts, and the MoE then costs far less than served)
    hidden = None if model.first else torch.zeros(a.append, model.hidden_width, dtype=torch.bfloat16, device=dev)
    res = {"stage": a.stage, "prefix": a.prefix, "append": a.append, "block_tokens": B}

    def one(ahead: bool, idle: str = ""):
        torch.cuda.synchronize(dev)
        if idle == "sleep":
            time.sleep(a.idle_ms / 1e3)
        elif idle == "spin":
            until = time.perf_counter() + a.idle_ms / 1e3
            while time.perf_counter() < until:
                pass
        if ahead:
            torch.cuda._sleep(200_000_000)          # ~100 ms of GPU spin: the host enqueues the forward meanwhile
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        with torch.inference_mode():
            b = model.prepare(plan)
            prep.append(1e3 * (time.perf_counter() - t0))
            model.run(b, hidden)
        e1.record()
        host = 1e3 * (time.perf_counter() - t0)
        e1.synchronize()
        return host, e0.elapsed_time(e1)

    prep = []

    for _ in range(5):
        one(False)
    modes = ("direct", "ahead") + (("direct_sleep", "direct_spin") if a.idle_ms else ())
    for mode in modes:
        h, g = zip(*(one(mode == "ahead", mode.partition("_")[2]) for _ in range(a.reps)))
        res[mode] = {"host_ms": round(statistics.median(h), 3), "gpu_ms": round(statistics.median(g), 3),
                     "prepare_ms": round(statistics.median(prep[-a.reps:]), 3)}
        print(f"[host_front] stage {a.stage} P {a.prefix} A {a.append} {mode}: forward host "
              f"{res[mode]['host_ms']:.3f} ms (prepare {res[mode]['prepare_ms']:.3f}), compute "
              f"{res[mode]['gpu_ms']:.3f} ms", flush=True)
    res["host_cost_on_gpu_ms"] = round(res["direct"]["gpu_ms"] - res["ahead"]["gpu_ms"], 3)
    print(f"[host_front] direct - ahead = {res['host_cost_on_gpu_ms']:.3f} ms", flush=True)
    for mode in modes[2:]:
        print(f"[host_front] {mode} - ahead = {res[mode]['gpu_ms'] - res['ahead']['gpu_ms']:.3f} ms", flush=True)
    if a.profile:
        torch.cuda.synchronize(dev)
        prof = cProfile.Profile()
        prof.enable()
        with torch.inference_mode():
            model.forward(plan, hidden)
        prof.disable()
        torch.cuda.synchronize(dev)
        st = pstats.Stats(prof).stats          # {(file, line, fn): (cc, nc, tottime, cumtime, callers)}
        for key, col in (("own", 2), ("cumulative", 3)):
            rows = sorted(st.items(), key=lambda kv: -kv[1][col])[:40]
            lines = [f"{v[col] * 1e6:9.0f} us {v[1]:5d}x  {k[2]} ({Path(k[0]).name}:{k[1]})" for k, v in rows]
            print(f"[host_front] profile by {key} time (us, calls):\n" + "\n".join(lines), flush=True)
    if a.stamps:
        res["stamps"] = stamps(model, plan, hidden, dev, a.reps)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def stamps(model, plan, hidden, dev, reps: int) -> dict:
    """--stamps: segment start times and function times (ssengine.runtime.host_stamps) over `reps` direct forwards,
    medians per forward."""
    import torch

    from ssengine.runtime.host_stamps import HostStamps

    st = HostStamps(model)
    if st.missing:
        print(f"[host_front] stamps: not found: {st.missing}", flush=True)
    recs = []
    for r in range(reps + 3):
        torch.cuda.synchronize(dev)
        st.begin()
        with torch.inference_mode():
            model.forward(plan, hidden)
        rec = st.end()
        torch.cuda.synchronize(dev)
        if r >= 3:
            recs.append(rec)
    med = statistics.median
    segs: dict = {}
    for rec in recs:
        for i, k, t in rec["seg"]:
            segs.setdefault((i, k), []).append(t)
    fns: dict = {}
    for rec in recs:
        for q, (ms, n) in rec["fn"].items():
            fns.setdefault(q, []).append((ms, n))
    out = {"host_ms": round(med(r["host_ms"] for r in recs), 3), "gc_ms": round(med(r["gc_ms"] for r in recs), 3),
           "segments": [[i, k, round(med(v), 3)] for (i, k), v in sorted(segs.items())],
           "functions": {q: [round(med(m for m, _ in v), 3), v[0][1]]
                         for q, v in sorted(fns.items(), key=lambda kv: -med(m for m, _ in kv[1]))}}
    print(f"[host_front] stamps: forward host {out['host_ms']:.3f} ms (monitored), gc {out['gc_ms']:.3f} ms; "
          "segment starts (ms):", flush=True)
    print("  " + "  ".join(f"{i}{k}@{t:.3f}" for i, k, t in out["segments"]), flush=True)
    for q, (ms, n) in out["functions"].items():
        print(f"  {ms:7.3f} ms  {n:3d}x  {q}", flush=True)
    return out


if __name__ == "__main__":
    main()
