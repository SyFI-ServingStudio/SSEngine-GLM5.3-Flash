"""One stage on one GPU: build with warm-up, then capture the CUDA graphs of the buckets given (or run one eager
forward of that many tokens). Each call is its own process, so a CUDA error names its bucket.

  python -m bench.graph_probe --stage 0 --buckets 7168            # warm-up + capture of 7168 only
  CUDA_LAUNCH_BLOCKING=1 python -m bench.graph_probe --stage 0 --eager 7168
  python -m bench.graph_probe --stage 0 --full-pool --together --buckets 8192,7168   # as the server captures

Prints "[graph_probe] ... ok" or raises.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace as NS


def main():
    import torch

    from ppimpl.model.stage import StageModel, bench_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--buckets", default="", help="comma-separated token counts to capture")
    ap.add_argument("--eager", type=int, default=0, help="instead: one eager forward of this many tokens")
    ap.add_argument("--together", action="store_true", help="capture all --buckets in one call (as the server does)")
    ap.add_argument("--full-pool", action="store_true", help="the server's KV pool (all free memory), not 64 blocks")
    ap.add_argument("--loop", action="store_true",
                    help="capture --buckets with a copy of StageModel._capture_graphs that prints after each bucket")
    ap.add_argument("--pool-per-bucket", action="store_true", help="with --loop: a fresh graph pool per bucket")
    ap.add_argument("--keep-eager-args", action="store_true",
                    help="with --loop: eager breaks hold strong references to their arguments (no weak refs)")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    t0 = time.perf_counter()
    pool = {} if a.full_pool else {"num_blocks": 64, "max_model_len": 8192 + 8576}
    model = StageModel(bench_config(8192, cuda_graphs=False), a.stage, dev, **pool)
    torch.cuda.synchronize(dev)
    print(f"[graph_probe] stage {a.stage}: built and warmed up in {time.perf_counter() - t0:.0f} s", flush=True)
    if a.eager:
        T = a.eager
        plan = NS(mb_id=0, chunks=[NS(req_id="e", start=0, num_tokens=T, kv_blocks=[1], is_last=True,
                                      state_blocks=list(range(2, 8)), token_ids=[1] * T)])
        hidden = None if model.first else torch.zeros(T, model.hidden_width, dtype=torch.bfloat16, device=dev)
        with torch.inference_mode():
            model.forward(plan, hidden)
        torch.cuda.synchronize(dev)
        print(f"[graph_probe] stage {a.stage}: eager {T} ok", flush=True)
    buckets = [int(x) for x in a.buckets.split(",") if x]
    if a.loop:
        return capture_loop(model, sorted(buckets, reverse=True), a)
    for b in [tuple(buckets)] if a.together else [(b,) for b in buckets]:
        model.capture_graphs(buckets=b)
        torch.cuda.synchronize(dev)
        print(f"[graph_probe] stage {a.stage}: capture {b} ok: {model.graph_info}", flush=True)


def capture_loop(model, buckets, a):
    """StageModel._capture_graphs, step by step: eager pass, sync, capture, sync, print, per bucket."""
    import torch
    from vllm.compilation import breakable_cudagraph as bc

    if a.keep_eager_args:
        bc._weak_ref_capture_arg = lambda arg: arg
    pool = torch.cuda.graph_pool_handle()
    hidden = None if model.first else torch.zeros(buckets[0], model.hidden_width, dtype=torch.bfloat16,
                                                  device=model.device)
    from ppimpl.model.stage import GraphEntry

    with torch.inference_mode():
        for Tb in buckets:
            if a.pool_per_bucket:
                pool = torch.cuda.graph_pool_handle()
            plan = NS(mb_id=-10, chunks=[NS(req_id="g", start=0, num_tokens=Tb, kv_blocks=[1], is_last=True,
                                            state_blocks=list(range(2, 8)), token_ids=[1] * Tb)])
            b = model.meta.build(plan, with_tokens=model.first, bucket=Tb)
            e = GraphEntry(Tb, torch.empty_like(b.graph_rows), None if hidden is None else hidden[:Tb])
            e.rows.copy_(b.graph_rows)
            e.slots = {name: e.rows[1] for name in b.slots}
            with model._graph_context(b, e):
                model._graph_body(e)
            torch.cuda.synchronize(model.device)
            print(f"[graph_probe] {Tb}: eager ok", flush=True)
            e.capture = bc.BreakableCUDAGraphCapture(pool=pool)
            with model._graph_context(b, e), e.capture:
                e.out = model._graph_body(e)
            torch.cuda.synchronize(model.device)
            print(f"[graph_probe] {Tb}: capture ok", flush=True)
            model.graphs[Tb] = e
    print(f"[graph_probe] loop {buckets} ok", flush=True)


if __name__ == "__main__":
    main()
