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
    for b in [tuple(buckets)] if a.together else [(b,) for b in buckets]:
        model.capture_graphs(buckets=b)
        torch.cuda.synchronize(dev)
        print(f"[graph_probe] stage {a.stage}: capture {b} ok: {model.graph_info}", flush=True)


if __name__ == "__main__":
    main()
