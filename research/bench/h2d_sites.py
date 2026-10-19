"""Where a stage's prepare + run copy host memory to the device: every torch-level H2D call (Tensor.to / .cuda /
.copy_ from a CPU tensor, torch.tensor / as_tensor with a CUDA device) with its bytes, whether the source is pinned,
and the first caller frames outside torch, over a few microbatches (new requests, continuations, a long context).

    python -m research.bench.h2d_sites [--stage 0] [--len 1024]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import traceback
from types import SimpleNamespace as NS

from benchmarks import common as lb
from research.bench.overlap import make_plans
from ppimpl.engine.kv import block_pages


def main() -> None:
    import torch
    from torch.overrides import TorchFunctionMode

    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    tokens = lb.enwik9_tokens(CHECKPOINT, 40000)
    plans, used = make_plans(3, 6, a.len, tokens)
    # a long-context continuation: 3 KV blocks of context, then a chunk
    B = 8576
    kv = list(range(used, used + 4))
    st = list(range(used + 4, used + 10))
    plans.append(NS(mb_id=9, chunks=[NS(req_id="ctx", start=3 * B, num_tokens=4096, kv_pages=block_pages(kv),
                                     state_blocks=st,
                                        is_last=True, token_ids=tokens[3 * B:3 * B + 4096])]))
    used += 10
    budget = max(sum(c.num_tokens for c in p.chunks) for p in plans)
    model = StageModel(bench_config(max(budget, 8192)), a.stage, dev, num_blocks=used + 8,
                       max_model_len=4 * B + 8576)
    sites: collections.Counter = collections.Counter()
    nbytes: collections.Counter = collections.Counter()

    def where():
        fr = [f for f in traceback.extract_stack()[:-3] if "/torch/" not in f.filename]
        return " < ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in fr[-1:-4:-1])

    class Log(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            name = getattr(func, "__name__", str(func))
            src = None
            if name in ("to", "cuda") and isinstance(args[0], torch.Tensor) and args[0].device.type == "cpu" \
                    and isinstance(out, torch.Tensor) and out.device.type == "cuda":
                src = args[0]
            elif name == "copy_" and args[0].device.type == "cuda" and isinstance(args[1], torch.Tensor) \
                    and args[1].device.type == "cpu":
                src = args[1]
            elif name in ("tensor", "as_tensor") and isinstance(out, torch.Tensor) and out.device.type == "cuda":
                src = "data"
            if src is not None:
                b = out.numel() * out.element_size() if name != "copy_" else args[1].numel() * args[1].element_size()
                pinned = src.is_pinned() if isinstance(src, torch.Tensor) else False
                key = f"{name} pinned={pinned} {'>=4KB' if b >= 4096 else '<4KB'} @ {where()}"
                sites[key] += 1
                nbytes[key] += b
            return out

    with torch.inference_mode():
        gen = torch.Generator(device=dev).manual_seed(0)
        for i, p in enumerate(plans * 2):
            h = None if model.first else torch.randn(sum(c.num_tokens for c in p.chunks), model.hidden_width,
                                                      device=dev, generator=gen).to(torch.bfloat16)
            if i < len(plans):
                model.forward(p, h)            # warm-up (caches)
                torch.cuda.synchronize()
                continue
            with Log():
                model.forward(p, h)
            torch.cuda.synchronize()
    out = [{"site": k, "calls": v, "bytes": nbytes[k]} for k, v in sites.most_common()]
    for r in out:
        print(json.dumps(r), flush=True)
    path = lb.MEASURED / "host" / f"h2d_sites_s{a.stage}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
