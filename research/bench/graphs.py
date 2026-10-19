"""CUDA graphs of one stage: replay vs eager, per bucket (1 GPU).

    python -m research.bench.graphs --stage S [--out JSON]

Builds the stage with CUDA graphs (`StageModel.capture_graphs`) and, for every bucket, two plans: exactly the bucket
(no padding) and a padded one (two requests, a token count not divisible by 16). For each:
- replay vs the same body run eagerly on the same static buffers (same padded T): must be bitwise equal, and the
  eager body must repeat bitwise;
- replay with junk in the padding rows (token ids, positions, received rows): the real rows must not change;
- replay vs the eager forward at the unpadded T: equal without padding; with padding the mHC split-K count changes
  (n_SMs // ceil(T/64)), so the rows differ by the batch numerics of `research/measured/issues.md` (reported, last stage:
  tokens too).
A replay of every bucket runs in between, so each check also covers replay after other buckets.

    python -m research.bench.graphs --stage S --memory [--mb-tokens 32768]

The memory check instead: the stage as the worker builds it (KV pool sized by the reserve, warm-up at the microbatch
budget, then the graphs), then an eager forward at the budget and a replay of the largest bucket. Reports Ready's
block count, the graph pool, and the lowest free HBM seen.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

from benchmarks import common as lb
from ssengine.engine.kv import block_pages


def plans_for(Tb: int, tokens: list[int], prev: int | None = None):
    """The bucket exactly, and a padded plan of two requests between the next smaller bucket `prev` and Tb (not
    divisible by 16)."""
    def chunk(rid, n, kv, base_tok):
        return NS(req_id=rid, start=0, num_tokens=n, kv_pages=block_pages([kv]), is_last=True,
                  state_blocks=list(range(kv + 1, kv + 7)), token_ids=tokens[base_tok:base_tok + n])
    exact = NS(mb_id=0, chunks=[chunk("a", Tb, 1, 0)])
    T = Tb - (Tb - prev) // 8 - 3 if prev else Tb - 21
    n1 = T // 3
    padded = NS(mb_id=1, chunks=[chunk("b", T - n1, 1, 100), chunk("c", n1, 8, 7000)])
    return {"exact": exact, "padded": padded}


def main() -> None:
    import torch
    from vllm.forward_context import set_forward_context

    from ssengine.model import pp_patch
    from ssengine.model.stage import CHECKPOINT, StageModel, bench_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--mb-tokens", type=int, default=32768)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    free0 = torch.cuda.mem_get_info(dev)[0]
    if a.memory:
        return memory_check(a, dev, free0)
    model = StageModel(bench_config(8192, cuda_graphs=True), a.stage, dev, num_blocks=64, max_model_len=8192 + 8576)
    tokens = lb.enwik9_tokens(CHECKPOINT, 16384)
    res = {"stage": a.stage, "graph_info": model.graph_info, "cases": []}
    print(f"[graphs] stage {a.stage}: {model.graph_info}", flush=True)

    def eager_raw(plan, h):
        b = model.meta.build(plan, with_tokens=model.first)
        with set_forward_context(b.attn, model.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x, residual, post, comb = model.run_layers(b, h)
            if not model.last:
                return (pp_patch.edge_send(model.layers[-1], x, residual, post, comb).clone(),), b
            return (x, residual, post, comb), b

    def replay_junk_pads(b, h):
        """Replay with the padding rows filled with junk (token ids, positions, received rows; the slots stay -1):
        the first T rows must not change."""
        e, T = model.graphs[b.bucket], b.num_tokens
        e.rows.copy_(b.graph_rows)
        if T < e.bucket:
            e.rows[0, T:] = torch.randint(0, 8000, (e.bucket - T,), device=dev, generator=gen)
            if model.first:
                e.rows[2, T:] = torch.randint(1, 150000, (e.bucket - T,), device=dev, generator=gen)
        if not model.first:
            e.hidden[:T].copy_(h)
            e.hidden[T:].normal_(generator=gen)
        with model._graph_context(b, e):
            e.capture.replay()
            out = e.out if model.last else (pp_patch.edge_send(model.layers[-1], *e.out),)   # as _run_graph
        return tuple(None if t is None else t[:T].clone() for t in out)

    def head_tokens(b, out):
        return model.output_head(b, *out).tolist()

    def rel(x, y):
        return float((x.float() - y.float()).norm() / x.float().norm().clamp_min(1e-30))

    gen = torch.Generator(device=dev).manual_seed(0)
    ok = True
    with torch.inference_mode():
        for Tb in sorted(model.graphs):
            prev = max((t for t in model.graphs if t < Tb), default=None)
            for kind, plan in plans_for(Tb, tokens, prev).items():
                T = sum(c.num_tokens for c in plan.chunks)
                h = None if model.first else torch.randn(T, model.hidden_width, device=dev, generator=gen).to(
                    torch.bfloat16)
                b = model.prepare(plan)
                assert b.bucket == Tb, (b.bucket, Tb)
                g = model._run_graph(b, h, replay=True, raw=True)
                other = max(t for t in model.graphs if t != Tb) if len(model.graphs) > 1 else Tb
                model.forward(plans_for(other, tokens)["exact"], None if model.first else torch.zeros(
                    other, model.hidden_width, dtype=torch.bfloat16, device=dev))     # another bucket in between
                b2 = model.prepare(plan)
                e = model._run_graph(b2, h, replay=False, raw=True)
                e2 = model._run_graph(model.prepare(plan), h, replay=False, raw=True)    # eager is deterministic
                j = replay_junk_pads(model.prepare(plan), h)
                u, bu = eager_raw(plan, h)
                torch.cuda.synchronize()
                exact = all(torch.equal(x, y) for x, y in zip(g, e) if x is not None)
                r = {"bucket": Tb, "kind": kind, "T": T, "replay_eq_eager_padded": exact,
                     "eager_repeat_eq": all(torch.equal(x, y) for x, y in zip(e, e2) if x is not None),
                     "pad_independent": all(torch.equal(x, y) for x, y in zip(g, j) if x is not None),
                     "replay_vs_eager_unpadded_rel": [rel(y, x) for x, y in zip(g, u) if x is not None],
                     "replay_eq_eager_unpadded": all(torch.equal(x, y) for x, y in zip(g, u) if x is not None),
                     "finite": all(bool(torch.isfinite(x).all()) for x in g if x is not None)}
                if model.last:
                    r["tokens_replay"] = head_tokens(b, g)
                    r["tokens_eager_unpadded"] = head_tokens(bu, u)
                ok &= exact and r["eager_repeat_eq"] and r["pad_independent"] and r["finite"] and (kind != "exact" or r["replay_eq_eager_unpadded"])
                res["cases"].append(r)
                print(f"[graphs] {r}", flush=True)
    res["pass"] = ok
    res["memory"] = {"free_before_model_gib": round(free0 / 2**30, 2),
                     "free_after_gib": round(torch.cuda.mem_get_info(dev)[0] / 2**30, 2),
                     "max_allocated_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)}
    print(f"[graphs] pass {ok}; memory {res['memory']}", flush=True)
    out = Path(a.out or lb.MEASURED / "graphs" / f"s{a.stage}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[graphs] wrote {out}", flush=True)


def memory_check(a, dev, free0) -> None:
    import torch

    from ssengine.model.stage import CHECKPOINT, StageModel, bench_config

    gib = lambda v: round(v / 2**30, 2)  # noqa: E731
    model = StageModel(bench_config(a.mb_tokens, cuda_graphs=True), a.stage, dev)
    tokens = lb.enwik9_tokens(CHECKPOINT, 2 * a.mb_tokens)
    res = {"stage": a.stage, "mb_tokens": a.mb_tokens, "num_blocks": model.num_blocks, "graph_info": model.graph_info,
           "free_before_model_gib": gib(free0), "free_after_init_gib": gib(torch.cuda.mem_get_info(dev)[0]), "runs": []}
    big = max(model.graphs)
    with torch.inference_mode():
        for T in (a.mb_tokens, big, a.mb_tokens, big):
            nb = -(-T // model.kv.block_tokens)
            plan = NS(mb_id=0, chunks=[NS(req_id="m", start=0, num_tokens=T, 
                                          kv_pages=block_pages(list(range(1, 1 + nb))),
                                          is_last=True, state_blocks=list(range(1 + nb, 7 + nb)),
                                          token_ids=tokens[:T])])
            h = None if model.first else torch.zeros(T, model.hidden_width, dtype=torch.bfloat16, device=dev)
            b = model.prepare(plan)
            model.run(b, h)
            torch.cuda.synchronize(dev)
            res["runs"].append({"T": T, "graph": b.bucket is not None,
                                "free_gib": gib(torch.cuda.mem_get_info(dev)[0]),
                                "reserved_gib": gib(torch.cuda.memory_reserved(dev)),
                                "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(dev))})
            print(f"[graphs] memory {res['runs'][-1]}", flush=True)
    res["min_free_gib"] = min(r["free_gib"] for r in res["runs"])
    print(f"[graphs] memory stage {a.stage}: blocks {model.num_blocks}, graph pool "
          f"{model.graph_info.get('graph_memory_gib')} GiB, min free {res['min_free_gib']} GiB", flush=True)
    out = Path(a.out or lb.MEASURED / "graphs" / f"mem_s{a.stage}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
