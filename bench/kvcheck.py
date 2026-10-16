"""KV cache and KDA state correctness on GPU: one prompt computed whole, in chunks, and in chunks on scrambled blocks.

    python -m bench.kvcheck [--prompt-len 32768] [--chunk 8000]     # 4 GPUs

Four processes, one `StageModel` each, joined by the serving transport (`open_pipe`: the CE
ring), as in the runtime. Stages 0-2 run `StageModel.forward` with CUDA graphs on (chunks up to 8,192 tokens replay
a graph); stage 3 runs the same layers eagerly and returns the last position's logits (graph replay equals eager,
GPU test f). The same prompt (the first L tokens of enwik9) runs as:
- `whole`: one microbatch [0, L).
- `whole_filler`: one microbatch [0, L) plus a 1,000-token request after it. Only the batch differs, so its distance
  to `whole` is the batch-numerics noise (mHC split-K count, NVFP4) that the others are judged against.
- `chunked`: chunks of `--chunk` tokens (not on block boundaries) in consecutive microbatches; each chunk reads the
  KV its earlier chunks wrote and resumes the KDA and conv state they left.
- `scrambled`: as `chunked`, on KV and state blocks drawn at random from blocks the earlier variants used (stale
  contents), each microbatch also carrying a 1,000-token request on other random blocks.
Per variant: the last position's logits against `whole` (max |diff|, KL, top-1, top-5 overlap) and each stage's
output rows of the prompt's last chunk against the same rows of `whole` (relative L2). A KV or state hand-off bug
moves `chunked`/`scrambled` far past `whole_filler`. Results: `measured/kvcheck/kvcheck_<L>_<job>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from types import SimpleNamespace as NS

from bench.common import MEASURED, enwik9_tokens

OUT = MEASURED / "kvcheck"
JOB = os.environ.get("SLURM_JOB_ID", "local")
FILLER = 1000
NUM_BLOCKS = 96          # blocks 1.. ; a 32k prompt needs 4 KV + 6 state, a filler 1 + 6


def _chunk(rid, start, n, kv, state, last, toks):
    return NS(req_id=rid, start=start, num_tokens=n, kv_blocks=kv, state_blocks=state, is_last=last,
              token_ids=toks[start:start + n])


def variants(tokens: list[int], fillers: list[list[int]], chunk: int, seed: int = 0) -> dict[str, list]:
    """Each variant: its microbatches (plans). Block ids 1.. in the order the variants run, so `scrambled` draws
    from blocks the earlier variants left written."""
    L = len(tokens)
    nkv = -(-L // 8576)
    nxt = iter(range(1, NUM_BLOCKS))

    def take(n):
        return [next(nxt) for _ in range(n)]

    out = {}
    out["whole"] = [NS(mb_id=0, chunks=[_chunk("p", 0, L, take(nkv), take(6), True, tokens)])]
    kv, st = take(nkv), take(6)
    f = fillers[0]
    out["whole_filler"] = [NS(mb_id=0, chunks=[_chunk("p", 0, L, kv, st, True, tokens),
                                               _chunk("f0", 0, FILLER, take(1), take(6), True, f)])]
    kv, st = take(nkv), take(6)
    out["chunked"] = [NS(mb_id=i, chunks=[_chunk("p", s, min(chunk, L - s), kv, st, s + chunk >= L, tokens)])
                      for i, s in enumerate(range(0, L, chunk))]
    used = list(range(1, NUM_BLOCKS))
    rng = random.Random(seed)
    rng.shuffle(used)
    kv, st = used[:nkv], used[nkv:nkv + 6]
    rest = used[nkv + 6:]
    mbs = []
    for i, s in enumerate(range(0, L, chunk)):
        fk, fs = rest[7 * i:7 * i + 1], rest[7 * i + 1:7 * i + 7]
        mbs.append(NS(mb_id=i, chunks=[_chunk("p", s, min(chunk, L - s), kv, st, s + chunk >= L, tokens),
                                       _chunk(f"f{i + 1}", 0, FILLER, fk, fs, True, fillers[(i + 1) % len(fillers)])]))
    out["scrambled"] = mbs
    return out


def _worker(rank: int, L: int, chunk: int, port: int) -> None:
    import torch
    from ppimpl.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig
    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config
    from ppimpl.runtime.transport import open_pipe
    from vllm.forward_context import set_forward_context

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    rc = RunConfig(port=port, scheduler=SchedulerConfig(mb_tokens=L + FILLER))   # ring slot: largest microbatch
    pipe = open_pipe(rc, rank, dev, HIDDEN_WIDTH)
    t0 = time.time()
    m = StageModel(bench_config(L + FILLER, cuda_graphs=True), rank, dev, num_blocks=NUM_BLOCKS,
                   max_model_len=L + 8576, cuda_graphs=rank < 3)   # stage 3 eager: it returns logits
    print(f"[kvcheck] rank {rank} layers {m.layer_range} built in {time.time() - t0:.0f} s, "
          f"graphs {getattr(m, 'graph_info', None)}", flush=True)
    ids = enwik9_tokens(CHECKPOINT, L + 8 * FILLER)
    tokens, fillers = ids[:L], [ids[L + FILLER * i:L + FILLER * (i + 1)] for i in range(8)]
    last_start = (L - 1) // chunk * chunk          # first token of the prompt's last chunk
    res = {}
    with torch.inference_mode():
        for name, plans in variants(tokens, fillers, chunk).items():
            for plan in plans:
                n = sum(c.num_tokens for c in plan.chunks)
                hidden = pipe.recv(n, HIDDEN_WIDTH) if rank > 0 else None
                if rank < 3:
                    out = m.forward(plan, hidden)
                    done = torch.cuda.Event()
                    done.record()
                    if hidden is not None:
                        pipe.release()
                    pipe.send(out, done)
                    p = plan.chunks[0]
                    rows = out[:p.num_tokens].float()
                else:
                    b = m.prepare(plan)
                    with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                        x, r, po, co = m.run_layers(b, hidden)
                        logits = m.head_logits(b, x, r, po, co) if b.last_rows.shape[0] else None
                        rows = None
                    pipe.release()
                    p = plan.chunks[0]
                if rank < 3 and p.start + p.num_tokens == L:   # the prompt's last microbatch: keep its last-chunk rows
                    keep = rows[last_start - p.start:].cpu()
                    res[name] = {"rows": keep}
                if rank == 3 and p.is_last:
                    res[name] = {"logits": logits[0].float().cpu()}
            torch.cuda.synchronize()
    base = res["whole"]
    report = {}
    for name, r in res.items():
        if name == "whole":
            continue
        if rank < 3:
            a, x = base["rows"], r["rows"]
            report[name] = {"rel_l2": float((x - a).norm() / a.norm()), "max_abs": float((x - a).abs().max()),
                            "finite": bool(torch.isfinite(x).all())}
        else:
            a, x = base["logits"], r["logits"]
            pa, px = torch.log_softmax(a, -1), torch.log_softmax(x, -1)
            report[name] = {"max_abs_logit": float((x - a).abs().max()),
                            "kl": float((pa.exp() * (pa - px)).sum()),
                            "top1": [int(a.argmax()), int(x.argmax())],
                            "top5_overlap": len(set(a.topk(5).indices.tolist()) & set(x.topk(5).indices.tolist())),
                            "finite": bool(torch.isfinite(x).all())}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"kvcheck_{L}_rank{rank}_{JOB}.json"
    path.write_text(json.dumps({"stage": rank, "prompt_tokens": L, "chunk": chunk, "variants": report}, indent=1))
    print(f"[kvcheck] rank {rank}: {json.dumps(report)}", flush=True)
    pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-len", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=8000)
    a = ap.parse_args()
    import torch.multiprocessing as mp

    port = 21000 + (int(JOB) % 100) * 100 if JOB.isdigit() else 21000
    mp.spawn(_worker, args=(a.prompt_len, a.chunk, port), nprocs=4, join=True)


if __name__ == "__main__":
    main()
