"""Where a stage's eager body stops repeating bitwise at one graph bucket (1 GPU).

    python -m research.bench.graph_repeat_probe --stage S --buckets 4608,4864,5120 [--repeats 6] [--out JSON]

Builds the stage as research/bench/graphs.py does (CUDA graphs captured), then for each bucket's exact and padded plan
(research.bench.graphs.plans_for) runs the eager body on the graph's static buffers `--repeats` times and the replay as many
times, each against the first. On a difference it reruns the eager body twice with forward hooks on every submodule
and names the first module (in call order) whose output differs, with its relative L2 difference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks import common as lb
from research.bench.graphs import plans_for


def main() -> None:
    import torch

    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    model = StageModel(bench_config(8192, cuda_graphs=True), a.stage, dev, num_blocks=64, max_model_len=8192 + 8576)
    tokens = lb.enwik9_tokens(CHECKPOINT, 16384)
    gen = torch.Generator(device=dev).manual_seed(0)
    res = {"stage": a.stage, "graph_info": model.graph_info, "cases": []}

    def rel(x, y):
        return float((x.float() - y.float()).norm() / x.float().norm().clamp_min(1e-30))

    def same(u, v):
        return all(torch.equal(x, y) for x, y in zip(u, v) if x is not None)

    def first_diff(plan, h):
        """Hooks on every submodule; two eager runs; the first module call whose output differs."""
        names = {m: n for n, m in model.inner.named_modules()}
        rec: list[list] = [[], []]
        cur = [0]

        def hook(mod, _inp, out):
            outs = out if isinstance(out, tuple) else (out,)
            ts = [t.detach().clone() for t in outs if isinstance(t, torch.Tensor)]
            if ts:
                rec[cur[0]].append((names.get(mod, type(mod).__name__), ts))
        mods = [m for layer in model.layers for m in layer.modules()]
        hs = [m.register_forward_hook(hook) for m in mods]
        try:
            for k in (0, 1):
                cur[0] = k
                model._run_graph(model.prepare(plan), h, replay=False, raw=True)
                torch.cuda.synchronize()
        finally:
            for x in hs:
                x.remove()
        diffs = []
        for (n0, t0), (n1, t1) in zip(*rec):
            if n0 != n1:
                return {"call_order_differs": [n0, n1]}
            bad = [(i, rel(x, y)) for i, (x, y) in enumerate(zip(t0, t1)) if x.shape == y.shape and not torch.equal(x, y)]
            if bad:
                diffs.append({"module": n0, "outputs": bad})
                if len(diffs) >= 6:
                    break
        return {"calls": len(rec[0]), "first_differing": diffs}

    with torch.inference_mode():
        buckets = sorted(model.graphs)
        for Tb in (int(x) for x in a.buckets.split(",")):
            prev = max((t for t in buckets if t < Tb), default=None)
            for kind, plan in plans_for(Tb, tokens, prev).items():
                T = sum(c.num_tokens for c in plan.chunks)
                h = None if model.first else torch.randn(T, model.hidden_width, device=dev, generator=gen).to(
                    torch.bfloat16)
                e = [model._run_graph(model.prepare(plan), h, replay=False, raw=True) for _ in range(a.repeats)]
                g = [model._run_graph(model.prepare(plan), h, replay=True, raw=True) for _ in range(a.repeats)]
                torch.cuda.synchronize()
                r = {"bucket": Tb, "kind": kind, "T": T,
                     "eager_differs": [i for i in range(1, a.repeats) if not same(e[0], e[i])],
                     "eager_rel": [[rel(x, y) for x, y in zip(e[0], e[i]) if x is not None] for i in range(1, a.repeats)],
                     "replay_differs": [i for i in range(1, a.repeats) if not same(g[0], g[i])],
                     "replay_eq_eager0": same(g[0], e[0])}
                if r["eager_differs"] or r["replay_differs"]:
                    r["locate"] = first_diff(plan, h)
                res["cases"].append(r)
                print(f"[probe] {json.dumps(r)[:2000]}", flush=True)
    out = Path(a.out or lb.MEASURED / "graphs" / f"repeat_s{a.stage}_{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
