"""One request's output under different batch mates and block histories, one stage per process (1 GPU).

    python bench/batchflip.py --stage S --prompts measured/batchflip/prompts.npz --work DIR --out JSON [--moe-probe]

Prompts: `bench/batchflip_prompts.py` (the token ids req-frontend sent, checked by sha256).

M3 7140/7162: uni_1k_3 (the one_1k_3 prompt) returns 11490 instead of 2669 when it runs in an 8 x 1k microbatch
after other requests have used the pool, and the right token alone in a fresh pool (7134). Each case below runs the
target through this stage and keeps its rows of every layer's attention and MoE output; stage S reads case c's
input from `DIR/c.pt` (written by stage S-1) and writes its output there. Cases:

- fresh:        the target alone, zeroed cache, kv block 1, state blocks 2..7
- reuse_same:   another 1k request first in the same blocks (same roles), then the target
- reuse_mixed:  an 8 x 1k batch first over blocks 1..56, then the target with roles mixed (its kv block was a state
                block, its state blocks were kv / other groups' state blocks)
- garbage:      every cache tensor filled with random values first (NaN possible where dtypes alias), then the target
- b114 / b206:  the target in the 7140 mb 114 / 7162 mb 206 batches (same prompts, same order), zeroed cache
- b114x:        b114's shape with the other 7 prompts replaced (uni_1k_100..106), zeroed cache
- b114r:        b114, but the target's rows of every attention and MoE input (and of the stage input) replaced by
                the fresh case's: fresh vs b114r is then each sublayer's own error from running in an 8 x 1k batch
                instead of alone (same input rows), not the error carried in from earlier layers

With a zeroed cache and the same shapes, only cross-request dependence can change the target's rows between b114
and b114x; fresh vs reuse_* / garbage changes only if a start=0 request reads state it did not write.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers as lb  # noqa: E402

L = 1024
TARGET = "uni_1k_3"
BATCHES = {
    "b114": [3, 15424, 6, 14880, 11, 7908, 29, 10544],
    "b206": [5, 10, 4803, 3, 5328, 8, 7953, 16],
    "b114x": [3, 100, 101, 102, 103, 104, 105, 106],
}


def chunk(name, kv, states, toks):
    return NS(req_id=name, start=0, num_tokens=L, kv_blocks=[kv], state_blocks=list(states), is_last=True,
              token_ids=list(map(int, toks)))


def batch_plan(ords, prompts, base=1):
    """Request i: kv block base + 7i, state blocks base + 7i + 1..6."""
    return NS(mb_id=0, chunks=[chunk(f"uni_1k_{k}", base + 7 * i, range(base + 7 * i + 1, base + 7 * i + 7),
                                     prompts[f"uni_1k_{k}"]) for i, k in enumerate(ords)])


def cases(prompts):
    alone = NS(mb_id=0, chunks=[chunk(TARGET, 1, range(2, 8), prompts[TARGET])])
    mixed = NS(mb_id=0, chunks=[chunk(TARGET, 9, [1, 8, 16, 2, 30, 45], prompts[TARGET])])
    other = NS(mb_id=0, chunks=[chunk("uni_1k_100", 1, range(2, 8), prompts["uni_1k_100"])])
    return {
        "fresh": (None, alone, 0),
        "reuse_same": (other, alone, 0),
        "reuse_mixed": (batch_plan(BATCHES["b114x"], prompts), mixed, 0),
        "garbage": ("garbage", alone, 0),
        **{name: (None, batch_plan(ords, prompts), ords.index(3)) for name, ords in BATCHES.items()},
        "b114r": (None, batch_plan(BATCHES["b114"], prompts), 0),
    }


def cache_tensors(model):
    ctx = model.vc.compilation_config.static_forward_context
    kv = model.meta.kv
    names = kv.mla_names + kv.indexer_names + kv.tail_names + [n for g in kv.kda_groups for n in g]
    out, seen = [], set()

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif hasattr(x, "data_ptr") and x.numel():
            key = (x.data_ptr(), x.dtype, tuple(x.shape))
            if key not in seen:
                seen.add(key)
                out.append(x)

    for n in names:
        walk(ctx[n].kv_cache)
    return out


def main() -> None:
    import numpy as np
    import torch
    from ppimpl.model.stage import StageModel, bench_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--moe-probe", action="store_true",
                    help="also run every MoE of the stage on the fresh case's input rows alone (T=1024) and inside "
                         "the b114 batch (T=8192): output, router logits, expert sets, shared expert")
    a = ap.parse_args()
    prompts = {k: v for k, v in np.load(a.prompts).items()}
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda", 0)
    model = StageModel(bench_config(8192), a.stage, dev, num_blocks=64, max_model_len=L + 8576)
    caches = cache_tensors(model)
    print(f"[batchflip] stage {a.stage} layers {model.layer_range}; {len(caches)} cache tensors, "
          f"{sum(t.numel() * t.element_size() for t in caches) / 2**30:.2f} GiB", flush=True)

    rows = {}           # (layer, part) -> target rows of this case
    fresh_in = {}       # (layer, part) -> the fresh case's input rows
    big_in = {}         # (layer, "mlp") -> b114's whole MoE input (--moe-probe)
    mode = [None]       # "capture" (fresh) / "replay" (b114r)
    hooks = []
    span = [0, L]
    for li, layer in zip(model.layer_range, model.layers):
        for part in ("self_attn", "mlp"):
            def hook(mod, inp, out, key=(li, part)):
                o = out[0] if isinstance(out, tuple) else out
                rows[key] = o[span[0]:span[1]].float().clone()

            def pre(mod, args, kwargs, key=(li, part)):
                x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                if mode[0] == "capture":
                    fresh_in[key] = x[span[0]:span[1]].clone()
                elif mode[0] == "capture_big" and key[1] == "mlp" and a.moe_probe:
                    big_in[key] = x.clone()
                elif mode[0] == "replay":
                    x[span[0]:span[1]] = fresh_in[key]      # in place: the caller's tensor is the module input
                return None
            hooks.append(getattr(layer, part).register_forward_hook(hook))
            hooks.append(getattr(layer, part).register_forward_pre_hook(pre, with_kwargs=True))
    res_kinds = {li: layer.layer_kind for li, layer in zip(model.layer_range, model.layers)}
    print(f"[batchflip] layer kinds {res_kinds}", flush=True)

    def stage_in(name, plan):
        if model.first:
            return None
        if name == "b114r":
            x = torch.load(work / "b114.pt", map_location=dev)
            x[span[0]:span[1]] = torch.load(work / "fresh.pt", map_location=dev)
            return x
        p = work / f"{name}.pt"
        if p.exists():
            return torch.load(p, map_location=dev)
        g = torch.Generator(device=dev).manual_seed(1)
        T = sum(c.num_tokens for c in plan.chunks)
        return torch.randn(T, model.hidden_width, device=dev, generator=g).to(torch.bfloat16)

    def forward(plan, hidden):
        from vllm.forward_context import set_forward_context
        b = model.prepare(plan)
        with set_forward_context(b.attn, model.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x, residual, post, comb = model.run_layers(b, hidden)
            if model.last:
                return model.head_logits(b, x, residual, post, comb, fp32=True)
            from ppimpl.model import pp_patch
            return pp_patch.edge_send(model.layers[-1], x, residual, post, comb)

    res = {"stage": a.stage, "layers": [model.layer_range.start, model.layer_range.stop], "kinds": res_kinds,
           "cases": {}}
    got = {}
    with torch.inference_mode():
        for name, (pre, plan, pos) in cases(prompts).items():
            for t in caches:
                t.zero_()
            if pre == "garbage":
                g = torch.Generator(device=dev).manual_seed(7)
                for t in caches:
                    if t.dtype.is_floating_point:
                        t.copy_(torch.randn(t.shape, device=dev, generator=g, dtype=torch.float32))
                    else:
                        t.copy_(torch.randint(0, 255, t.shape, device=dev, generator=g))
            elif pre is not None:
                span[:] = [0, 0]
                forward(pre, stage_in(f"pre_{name}", pre))   # dirties the blocks; its input need not be real
            span[:] = [pos * L, (pos + 1) * L]
            rows.clear()
            mode[0] = {"fresh": "capture", "b114r": "replay", "b114": "capture_big"}.get(name)
            out = forward(plan, stage_in(name, plan))
            mode[0] = None
            torch.cuda.synchronize()
            got[name] = dict(rows)
            r = {"finite": bool(torch.isfinite(out).all())}
            if model.last:
                lg = out[pos].float()       # one row per is_last chunk, in plan order
                lp = torch.log_softmax(lg, -1)
                top = torch.topk(lp, 5)
                r.update(token=int(top.indices[0]), top5=[[int(i), round(float(v), 4)]
                                                          for i, v in zip(top.indices, top.values)],
                         lp_2669=round(float(lp[2669]), 4), lp_11490=round(float(lp[11490]), 4))
            else:
                torch.save(out.contiguous(), work / f"{name}.pt")
            res["cases"][name] = r
            print(f"[batchflip] {name}: {r}", flush=True)

    def rel(x, y):
        return float((x - y).norm() / x.norm().clamp_min(1e-30))

    cmp = {}
    for ref, other in [("fresh", "reuse_same"), ("fresh", "reuse_mixed"), ("fresh", "garbage"), ("fresh", "b114"),
                       ("fresh", "b206"), ("b114", "b114x"), ("b114", "b206"), ("fresh", "b114r")]:
        cmp[f"{ref}~{other}"] = [[k[0], k[1], rel(got[ref][k], got[other][k]),
                                  float((got[ref][k] - got[other][k]).abs().max())] for k in got[ref]]
    res["compare"] = cmp
    for k, v in cmp.items():
        print(f"[batchflip] {k}: " + " ".join(f"{li}{p[0]}:{r:.1e}" for li, p, r, _ in v), flush=True)
    for h in hooks:
        h.remove()
    if a.moe_probe:
        with torch.inference_mode():
            res["moe_probe"] = moe_probe(model, cases(prompts), fresh_in, big_in, rel)
        for r in res["moe_probe"]:
            print(f"[batchflip] moe probe {r}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


def moe_probe(model, cs, fresh_in, big_in, rel):
    """Per MoE layer, the same 1024 input rows run alone (T=1024) and as rows 0..1023 of b114's input (T=8192)."""
    import torch
    from vllm.forward_context import set_forward_context

    def call(plan, mlp, x):
        b = model.prepare(plan)
        got = {}
        hs = [mlp.gate.register_forward_hook(lambda m, i, o: got.__setitem__("gate", (o[0] if isinstance(o, tuple)
                                                                                    else o).float().clone()))]
        if getattr(mlp, "shared_experts", None) is not None:
            hs.append(mlp.shared_experts.register_forward_hook(
                lambda m, i, o: got.__setitem__("shared", (o[0] if isinstance(o, tuple) else o).float().clone())))
        try:
            with set_forward_context(b.attn, model.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                y = mlp(x.clone(), already_sequence_parallel=False)
        finally:
            for h in hs:
                h.remove()
        torch.cuda.synchronize()
        return (y[0] if isinstance(y, tuple) else y).float(), got

    out = []
    for li, layer in zip(model.layer_range, model.layers):
        key = (li, "mlp")
        if key not in big_in or not hasattr(layer.mlp, "gate"):
            continue
        x1 = fresh_in[key]
        xb = big_in[key].clone()
        xb[:L] = x1
        y1, g1 = call(cs["fresh"][1], layer.mlp, x1)
        y8, g8 = call(cs["b114"][1], layer.mlp, xb)
        y8 = y8[:L]
        r = {"layer": li, "out_rel": rel(y1, y8), "out_equal": bool(torch.equal(y1, y8))}
        if "gate" in g1 and "gate" in g8:
            a1, a8 = g1["gate"][:L], g8["gate"][:L]
            r["router_rel"] = rel(a1, a8)
            r["router_equal"] = bool(torch.equal(a1, a8))
            bias = layer.mlp.gate.e_score_correction_bias
            k = layer.mlp.experts.top_k if hasattr(layer.mlp.experts, "top_k") else 8

            def experts(lg):
                sc = lg.sigmoid() + (bias.float() if bias is not None else 0)
                return torch.topk(sc, k, dim=-1).indices.sort(-1).values
            r["tokens_expert_set_differs"] = int((experts(a1) != experts(a8)).any(-1).sum())
        if "shared" in g1 and "shared" in g8:
            r["shared_rel"] = rel(g1["shared"][:L], g8["shared"][:L])
        out.append(r)
    return out


if __name__ == "__main__":
    main()
