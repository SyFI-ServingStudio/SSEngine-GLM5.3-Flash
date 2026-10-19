"""KV cache and KDA state correctness on GPU: one prompt computed whole, in chunks, and in chunks on scrambled blocks.

    python -m research.bench.kvcheck [--prompt-len 32768] [--chunk 8000]     # 4 GPUs

Four processes, one `StageModel` each, joined by the serving transport (`open_pipe`: the CE
ring), as in the runtime. Stages 0-2 run `StageModel.forward` with CUDA graphs on (chunks up to 8,192 tokens replay
a graph); stage 3 runs the same layers eagerly and returns the last position's logits (graph replay equals eager,
GPU test f). The same prompt (the first L tokens of enwik9) runs as:
- `whole`: one microbatch [0, L).
- `whole_filler`: one microbatch [0, L) plus a 1,000-token request after it. Only the batch differs, so its distance
  to `whole` is the batch-numerics noise (mHC split-K count, NVFP4) that the others are judged against.
- `chunked`: chunks of `--chunk` tokens (not on block boundaries) in consecutive microbatches; each chunk reads the
  KV its earlier chunks wrote and resumes the KDA and conv state they left.
- `chunked_b`: as `chunked` with chunks of `--chunk-b` tokens: chunked against chunked is the spread of resuming at
  other boundaries.
- `scrambled`: as `chunked`, on KV and state blocks drawn at random from blocks the earlier variants used (stale
  contents), each microbatch also carrying a 1,000-token request on other random blocks.
Baseline for the resume: each chunk's output rows against the same rows of `whole` (`per_chunk`, stages 0-2). The
first chunk starts at position 0 and resumes nothing, so its distance is what the microbatch size alone does (other
GEMM tactics and accumulation orders at T = chunk vs T = L); the later chunks add the KV and state hand-off.
Per variant: the last position's logits against `whole` (max |diff|, KL, top-1, top-5 overlap) and each stage's
output rows of the prompt's last chunk against the same rows of `whole` (relative L2). A KV or state hand-off bug
moves `chunked`/`scrambled` far past `whole_filler`. Results: `research/measured/kvcheck/kvcheck_<L>_<job>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from types import SimpleNamespace as NS

from benchmarks.common import MEASURED, enwik9_tokens
from ssengine.engine.kv import block_pages

OUT = MEASURED / "kvcheck"
JOB = os.environ.get("SLURM_JOB_ID", "local")
FILLER = 1000
NUM_BLOCKS = 96          # blocks 1.. ; a 32k prompt needs 4 KV + 6 state, a filler 1 + 6


def _chunk(rid, start, n, kv, state, last, toks):
    return NS(req_id=rid, start=start, num_tokens=n, kv_pages=block_pages(kv), state_blocks=state, is_last=last,
              token_ids=toks[start:start + n])


def variants(tokens: list[int], fillers: list[list[int]], chunk: int, seed: int = 0,
             chunk_b: int | None = None) -> dict[str, list]:
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
    if chunk_b:
        kv, st = take(nkv), take(6)
        out["chunked_b"] = [NS(mb_id=i, chunks=[_chunk("p", s, min(chunk_b, L - s), kv, st, s + chunk_b >= L,
                                                       tokens)])
                            for i, s in enumerate(range(0, L, chunk_b))]
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


def _rel(x, a) -> float:
    return float((x.float() - a.float()).norm() / a.float().norm())


def _worker(rank: int, L: int, chunk: int, port: int, chunk_b: int | None = None) -> None:
    import torch
    from ssengine.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig
    from ssengine.model.stage import CHECKPOINT, StageModel, bench_config
    from ssengine.runtime.transport import open_pipe
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
    res, per_chunk, whole_rows = {}, {}, None
    with torch.inference_mode():
        for name, plans in variants(tokens, fillers, chunk, chunk_b=chunk_b).items():
            step = chunk_b if name == "chunked_b" else chunk
            last_start = (L - 1) // step * step          # first token of the prompt's last chunk
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
                    if name == "whole":
                        whole_rows = out[:L].clone()
                    elif whole_rows is not None and name != "whole_filler":
                        per_chunk.setdefault(name, []).append(
                            {"start": p.start, "tokens": p.num_tokens,
                             "rel_l2": _rel(rows, whole_rows[p.start:p.start + p.num_tokens])})
                else:
                    b = m.prepare(plan)
                    with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                        x, r, po, co = m.run_layers(b, hidden)
                        logits = m.head_logits(b, x, r, po, co) if b.last_rows.shape[0] else None
                        rows = None
                    pipe.release()
                    p = plan.chunks[0]
                if rank < 3 and p.start + p.num_tokens == L:   # the prompt's last microbatch: keep its last-chunk rows
                    keep = rows[(last_start if name != "whole" else (L - 1) // chunk * chunk) - p.start:].cpu()
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
            if name == "chunked_b":                 # its last chunk covers other rows: compare the common tail
                n = min(len(a), len(x))
                a, x = a[-n:], x[-n:]
            report[name] = {"rel_l2": float((x - a).norm() / a.norm()), "max_abs": float((x - a).abs().max()),
                            "finite": bool(torch.isfinite(x).all()), "per_chunk": per_chunk.get(name)}
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
    path.write_text(json.dumps({"stage": rank, "prompt_tokens": L, "chunk": chunk, "chunk_b": chunk_b,
                                "variants": report}, indent=1))
    print(f"[kvcheck] rank {rank}: {json.dumps(report)}", flush=True)
    pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-len", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--chunk-b", type=int, default=8192, help="chunk size of `chunked_b` (0: skip it)")
    ap.add_argument("--exact", default="",
                    help="comma-separated variants that must equal `whole` bitwise (rows and logits): those whose "
                         "microbatches keep whole's shapes, i.e. chunks on 8,192-token boundaries (one mHC split-K "
                         "count, starts on KDA's 32-token chunk grid)")
    ap.add_argument("--noise-len", type=int,
                    help="judge the other variants against this job's `--noise-variant` run at this prompt length "
                         "(the batch-numerics baseline: 32768 in 8,000-token chunks leaves a 768-token last chunk, "
                         "another split-K count): per stage, max per-chunk rel. L2 <= factor x the baseline's; "
                         "logits: top-1 equal and KL <= factor^2 x the baseline's")
    ap.add_argument("--noise-variant", default="chunked")
    ap.add_argument("--noise-factor", type=float, default=2.0)
    a = ap.parse_args()
    import torch.multiprocessing as mp

    port = 21000 + (int(JOB) % 100) * 100 if JOB.isdigit() else 21000
    mp.spawn(_worker, args=(a.prompt_len, a.chunk, port, a.chunk_b or None), nprocs=4, join=True)
    if a.exact or a.noise_len:
        raise SystemExit(judge(a.prompt_len, [x for x in a.exact.split(",") if x], a.noise_len, a.noise_variant,
                               a.noise_factor))


def judge(L: int, exact: list[str], noise_len: int | None, noise_variant: str, factor: float) -> int:
    """The preflight rule: exactness where the microbatch shapes are whole's, a noise baseline elsewhere."""
    rep = [json.loads((OUT / f"kvcheck_{L}_rank{r}_{JOB}.json").read_text())["variants"] for r in range(4)]
    base = None if noise_len is None else \
        [json.loads((OUT / f"kvcheck_{noise_len}_rank{r}_{JOB}.json").read_text())["variants"][noise_variant]
         for r in range(4)]

    def dist(v, r):
        if r == 3:
            return v["max_abs_logit"]
        return max([v["rel_l2"], *(c["rel_l2"] for c in (v.get("per_chunk") or []))])

    bad, lines = [], []
    for name in rep[0]:
        vs = [rep[r][name] for r in range(4)]
        if not all(v["finite"] for v in vs):
            bad.append(f"{name}: not finite")
        if name in exact:
            d = [dist(vs[r], r) for r in range(4)]
            lines.append(f"{name} exact: {d}")
            if any(d):
                bad.append(f"{name}: not bitwise {d}")
        elif base is not None and name != "whole_filler":
            lim = [factor * dist(base[r], r) for r in range(3)]
            d = [dist(vs[r], r) for r in range(3)]
            kl, kl_lim = vs[3]["kl"], factor ** 2 * base[3]["kl"]
            lines.append(f"{name}: stage rel. L2 {[round(x, 3) for x in d]} (limit {[round(x, 3) for x in lim]}), "
                         f"KL {kl:.3g} (limit {kl_lim:.3g}), top-1 {vs[3]['top1']}")
            if any(x > y for x, y in zip(d, lim)) or kl > kl_lim or vs[3]["top1"][0] != vs[3]["top1"][1]:
                bad.append(f"{name}: outside the noise baseline")
    for x in lines:
        print(f"[kvcheck] judge {L}: {x}", flush=True)
    print(f"[kvcheck] judge {L}: " + ("; ".join(bad) if bad else "pass"), flush=True)
    return 1 if bad else 0

if __name__ == "__main__":
    main()
