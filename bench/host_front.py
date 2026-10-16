"""Stage forward host cost vs GPU time for one request continuing a cached prefix (a round >= 1 microbatch).

  python -m bench.host_front [--stage 0] [--prefix 2048] [--append 256] [--reps 30] [--out x.json] [--profile]

Two timings of the same forward (CUDA graphs on, as the server runs):
- `direct`: the forward enqueued on an idle GPU, as stage 0 runs it: compute = start event (host t = 0) -> end
  event. The GPU waits for the host wherever the host is slower than the kernels it launches.
- `ahead`: the same forward enqueued behind a GPU sleep, so the host is ahead of the GPU, as stages 1-3 run it
  (they enqueue before the upstream hidden state arrives): compute = the GPU time alone.
direct - ahead is what the host costs the stage on the GPU timeline. With --idle-ms X also `direct_sleep` (the
thread sleeps X ms before each forward, as a worker blocks on its plan socket between microbatches) and
`direct_spin` (it busy-waits X ms instead): what an idle, cold host thread adds. Also the host time of `forward` (perf
counter), and with --profile one cProfile of a direct forward (top functions by own and by cumulative time).
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace as NS


def main():
    import torch

    from ppimpl.model.stage import StageModel, bench_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--append", type=int, default=256)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--idle-ms", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    model = StageModel(bench_config(8192, cuda_graphs=True), a.stage, dev, num_blocks=64, max_model_len=8192 + 8576)
    B = model.kv.block_tokens
    n_kv = -(-(a.prefix + a.append) // B)
    assert n_kv + 7 <= model.num_blocks, (n_kv, model.num_blocks, B)
    plan = NS(mb_id=0, chunks=[NS(req_id="r", start=a.prefix, num_tokens=a.append, kv_blocks=list(range(1, 1 + n_kv)),
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
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
