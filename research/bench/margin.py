"""Top-2 margins of chosen prompts on the real model, chunked as the server chunks them, repeated (4 GPUs).

  python -m research.bench.margin <cases.json> [--repeats 3] [--out research/measured/margins/<name>.json]

cases.json: [{"name": str, "file": <.npy int32 token ids, relative to the repo root>,
"schedule": [[start, tokens, is_last], ...], "decode_canon": true, "decode_topk": "prefill"}, ...] (decode_canon false: without pp_patch's decode top-k
canonicalization; decode_topk: pp_patch.DECODE_TOPK_PATH, "prefill", "deep_select" or "kernel"). Each
case runs `--repeats` times on the same KV and state blocks (one request, a fresh start each time), each schedule
entry as one microbatch, as the server runs a request alone. At every is_last entry: the last position's logits,
bf16 as the server's argmax takes them (`StageModel.output_head`) and FP32, their top-5 and the top-1 minus top-2
margin in logprob (nats, the margin rule of benchmarks/check_tokens.py). Repeats show whether the same schedule gives the
same token in one process; research/bench/prefix_check.py shows it across server processes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

from benchmarks.common import MEASURED, ROOT
from ssengine.engine.kv import block_pages

JOB = os.environ.get("SLURM_JOB_ID", "local")
MB = 8192


def case_file(case: dict) -> Path:
    """A case's token file: relative to the repo root (absolute paths are taken as they are)."""
    f = Path(case["file"])
    return f if f.is_absolute() else ROOT / f


def server_schedule(n: int, start: int = 0) -> list[list]:
    """The server's chunks of a request alone: 8,192 tokens each from `start`, the rest last."""
    out, s = [], start
    while s < n:
        k = min(MB, n - s)
        out.append([s, k, s + k >= n])
        s += k
    return out


def _top(logits, k=5):
    import torch

    lp = torch.log_softmax(logits.float(), -1)
    v, i = lp.topk(k)
    raw = logits.float()[i]
    return {"top": [[int(a), round(float(b), 4), round(float(c), 4)] for a, b, c in zip(i, v, raw)],
            "margin": round(float(v[0] - v[1]), 4)}


TRACE = {"on": False, "rows": 0, "calls": [], "decode": []}   # --trace-topk: the indexer top-k rows of each is_last microbatch


def _install_topk_trace() -> None:
    """Wrap the fork's kpool indexer (as an eager break, so CUDA-graph replays call it too): while TRACE is on, keep
    sha1 of each layer's top-k rows of the microbatch's real tokens, as returned (set and order) and sorted (set)."""
    import hashlib

    import vllm.models.glm5next.nvidia.sparse_indexer as si
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    orig = getattr(si.sparse_attn_indexer_kpool, "__wrapped__", si.sparse_attn_indexer_kpool)

    def hooked(*args, **kwargs):
        r = orig(*args, **kwargs)
        if TRACE["on"]:
            rows = r[:TRACE["rows"]].cpu()
            TRACE["calls"].append([hashlib.sha1(rows.numpy().tobytes()).hexdigest()[:16],
                                   hashlib.sha1(rows.sort(dim=-1).values.numpy().tobytes()).hexdigest()[:16]])
        return r

    si.sparse_attn_indexer_kpool = eager_break_during_capture(hooked)


def _install_decode_trace() -> None:
    """After the stage model's patches (so it sees the top-k the model uses): the decode path's pool logits and
    top-k. The indexer looks `get_indexer_topk` up at call time, also on CUDA-graph replays (an eager break). Also
    runs vLLM's kernel 5 more times on the same logits and checks each answer is a valid top-k."""
    import hashlib

    import torch

    import vllm.models.glm5next.nvidia.sparse_indexer as si

    orig_get = si.get_indexer_topk

    def get_topk(backend):
        fn = orig_get(backend)

        def traced(logits, seq_lens, next_n, dst, k, *rest):
            fn(logits, seq_lens, next_n, dst, k, *rest)
            if TRACE["on"]:
                # decode path: pool logits, pool-granular seq_lens, select k pools per row
                d0 = _dispatcher(backend)    # vLLM's dispatcher, also under a wrapper (pp_patch decode canon)
                TRACE["backend"] = d0.resolve_backend(logits, k, logits.shape[0])
                lens = d0._row_ends(seq_lens, next_n, logits.shape[0]).tolist()   # the kernels' causal row ends
                lg, d = logits.float().cpu(), dst.cpu()
                rows = []
                for r, L in enumerate(lens):
                    v = lg[r, :L]
                    kth = float(v.topk(min(k, L)).values[-1]) if L else 0.0
                    rows.append({"keys": L, "ties_at_kth": int((v == kth).sum()),
                                 "above_kth": int((v > kth).sum())})
                # the bare kernel again on these logits: is each answer a valid top-k (wrong set vs reordering)?
                def valid(dd):
                    ok = True
                    for r, L in enumerate(lens):
                        sel = dd[r, :min(k, L)]
                        m = torch.ones(L, dtype=torch.bool)
                        m[sel.clamp(0, L - 1).long()] = False
                        ok &= bool((sel >= 0).all() and (sel < L).all() and sel.unique().numel() == sel.numel()
                                   and (L <= k or lg[r, :L][~m].min() >= lg[r, :L][m].max()))
                    return ok

                raw_valid, raw_sets = [], set()
                for _ in range(5):
                    d1 = torch.full_like(dst, -1)
                    d0(logits, seq_lens, next_n, d1, k, *rest)
                    d1 = d1.cpu()
                    raw_sets.add(hashlib.sha1(d1.sort(dim=-1).values.numpy().tobytes()).hexdigest()[:16])
                    raw_valid.append(valid(d1))
                rows.append({"raw_valid": raw_valid, "raw_sets": len(raw_sets), "used_valid": valid(d)})
                TRACE["decode"].append([hashlib.sha1(b"".join(lg[r, :L].numpy().tobytes()
                                                             for r, L in enumerate(lens))).hexdigest()[:16],
                                        hashlib.sha1(d.numpy().tobytes()).hexdigest()[:16],
                                        hashlib.sha1(d.sort(dim=-1).values.numpy().tobytes()).hexdigest()[:16],
                                        rows])

        return traced

    si.get_indexer_topk = get_topk


def _dispatcher(backend):
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk

    return get_indexer_topk(backend)


def _worker(rank: int, cases: list[dict], repeats: int, port: int, out: str, trace_topk: bool = False,
            nvtx: bool = False) -> None:
    import hashlib

    import numpy as np
    import torch
    from ssengine.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig
    from ssengine.model import pp_patch
    from ssengine.model.stage import StageModel, bench_config
    from ssengine.runtime.transport import open_pipe
    from vllm.forward_context import set_forward_context

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    if trace_topk:
        _install_topk_trace()
    traces = []
    toks = {c["name"]: np.load(case_file(c)).astype(np.int64).tolist() for c in cases}
    L = max(len(t) for t in toks.values())
    nkv = -(-L // 8576)
    rc = RunConfig(port=port, scheduler=SchedulerConfig(mb_tokens=MB))
    pipe = open_pipe(rc, rank, dev, HIDDEN_WIDTH)
    m = StageModel(bench_config(MB, cuda_graphs=True), rank, dev, num_blocks=nkv + 8, max_model_len=L + 8576,
                   cuda_graphs=rank < 3)
    if trace_topk:
        _install_decode_trace()
    kv, st = list(range(1, nkv + 1)), list(range(nkv + 1, nkv + 7))
    res, mbs = [], []
    with torch.inference_mode():
        for c in cases:
            t = toks[c["name"]]
            pp_patch.DECODE_TOPK_CANON = c.get("decode_canon", True)   # a case may run without it (before/after)
            pp_patch.DECODE_TOPK_PATH = c.get("decode_topk", "prefill")
            for rep in range(repeats):
                t0 = time.time()
                for i, (s, n, last) in enumerate(c["schedule"]):
                    plan = NS(mb_id=i, chunks=[NS(req_id="p", start=s, num_tokens=n, kv_pages=block_pages(kv),
                                                  state_blocks=st, is_last=last, token_ids=t[s:s + n])])
                    hidden = pipe.recv(n, HIDDEN_WIDTH) if rank > 0 else None
                    if nvtx:     # benchmarks/nsys_trace.py attributes kernels by this range, as in the server's workers
                        torch.cuda.nvtx.range_push(f"stage {rank} mb {len(mbs)}")
                        mbs.append({"mb": len(mbs), "case": c["name"], "repeat": rep, "chunk": i, "start": s,
                                    "tokens": n, "is_last": last})
                    TRACE.update(on=trace_topk and last, rows=n, calls=[], decode=[])
                    if rank < 3:
                        o = m.forward(plan, hidden)
                        done = torch.cuda.Event()
                        done.record()
                        if hidden is not None:
                            pipe.release()
                        pipe.send(o, done)
                        if nvtx:
                            torch.cuda.nvtx.range_pop()
                        if TRACE["on"]:
                            traces.append({"case": c["name"], "repeat": rep, "position": s + n - 1,
                                           "topk": TRACE["calls"], "decode": TRACE["decode"],
                                           "backend": TRACE.get("backend")})
                        TRACE["on"] = False
                        continue
                    b = m.prepare(plan)
                    with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                        x, r, po, co = m.run_layers(b, hidden)
                        if last:
                            lb = m.head_logits(b, x, r, po, co, fp32=False)[0]
                            lf = m.head_logits(b, x, r, po, co, fp32=True)[0]
                    pipe.release()
                    if nvtx:
                        torch.cuda.nvtx.range_pop()
                    if last:
                        res.append({"case": c["name"], "repeat": rep, "position": s + n - 1,
                                    "token": int(lb.argmax()), "bf16": _top(lb), "fp32": _top(lf),
                                    "logits_sha1": hashlib.sha1(lf.float().cpu().numpy().tobytes()).hexdigest()[:16]})
                        if TRACE["on"]:
                            traces.append({"case": c["name"], "repeat": rep, "position": s + n - 1,
                                           "topk": TRACE["calls"], "decode": TRACE["decode"],
                                           "backend": TRACE.get("backend")})
                        print(f"[margin] {c['name']} repeat {rep} pos {s + n - 1}: token {int(lb.argmax())} "
                              f"margin bf16 {res[-1]['bf16']['margin']} fp32 {res[-1]['fp32']['margin']} "
                              f"({time.time() - t0:.1f} s)", flush=True)
                    TRACE["on"] = False
                torch.cuda.synchronize()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if rank == 3:
        Path(out).write_text(json.dumps(res, indent=1) + "\n")
    if nvtx and rank == 0:
        Path(out).with_suffix(".mbs.json").write_text(json.dumps(mbs) + "\n")
    if trace_topk:
        Path(out).with_suffix(f".topk{rank}.json").write_text(json.dumps(traces) + "\n")
    pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--judge", action="store_true",
                    help="exit 1 unless, for every case with `expect_top2` and in every repeat, the last position's "
                         "top 2 hold those tokens and its top-2 margin is below --min-margin (a near tie)")
    ap.add_argument("--min-margin", type=float, default=3.0)
    ap.add_argument("--trace-topk", action="store_true",
                    help="also hash every indexer layer's top-k rows (as returned and sorted) and the FP32 logits "
                         "of each is_last microbatch; prints how many distinct values each case gives over repeats")
    ap.add_argument("--nvtx", action="store_true",
                    help="NVTX range `stage <s> mb <n>` per microbatch (n counts all microbatches; <out>.mbs.json "
                         "maps n to case, repeat and chunk), for `nsys profile` + benchmarks.nsys_trace.load")
    a = ap.parse_args()
    import torch.multiprocessing as mp

    cases = json.loads(a.cases.read_text())
    out = a.out or MEASURED / "margins" / f"{a.cases.stem}_{JOB}.json"
    port = 21500 + (int(JOB) % 100) * 100 if JOB.isdigit() else 21500
    from benchmarks.common import free_port_block
    port = free_port_block(port)
    mp.spawn(_worker, args=(cases, a.repeats, port, str(out), a.trace_topk, a.nvtx), nprocs=4, join=True)
    if a.trace_topk:
        print(json.dumps(variation(Path(out)), indent=1), flush=True)
    if a.judge:
        raise SystemExit(judge(cases, json.loads(Path(out).read_text()), a.min_margin))


def variation(out: Path) -> dict:
    """Per case and is_last position, over the repeats: distinct FP32 logits, tokens, and per stage the layers whose
    top-k rows differ as returned (set or order) and as sorted (set)."""
    res = json.loads(out.read_text())
    rep: dict = {}
    for r in res:
        k = f"{r['case']}@{r['position']}"
        e = rep.setdefault(k, {"repeats": 0, "logits": set(), "tokens": set(), "margins": set()})
        e["repeats"] += 1
        e["logits"].add(r["logits_sha1"])
        e["tokens"].add(r["token"])
        e["margins"].add(r["bf16"]["margin"])
    for st in range(4):
        f = out.with_suffix(f".topk{st}.json")
        if not f.exists():
            continue
        by: dict = {}
        for t in json.loads(f.read_text()):
            by.setdefault(f"{t['case']}@{t['position']}", []).append(t["topk"])
        for k, runs in by.items():
            layers = len(runs[0])
            order = [i for i in range(layers) if len({r[i][0] for r in runs}) > 1]
            sets = [i for i in range(layers) if len({r[i][1] for r in runs}) > 1]
            dec = [t for t in json.loads(f.read_text()) if f"{t['case']}@{t['position']}" == k]
            dv = {}
            if dec and dec[0]["decode"]:
                nl = len(dec[0]["decode"])
                dv = {"decode_layers": nl,
                      "decode_logits_vary": [i for i in range(nl) if len({d["decode"][i][0] for d in dec}) > 1],
                      "decode_topk_order_or_set_vary": [i for i in range(nl)
                                                        if len({d["decode"][i][1] for d in dec}) > 1],
                      "decode_topk_set_vary": [i for i in range(nl) if len({d["decode"][i][2] for d in dec}) > 1],
                      "backend": dec[0].get("backend"),
                      "max_ties_at_kth": max(r["ties_at_kth"] for d in dec for x in d["decode"] for r in x[3]
                                             if "ties_at_kth" in r),
                      "kernel_invalid_layers": sorted({i for d in dec for i, x in enumerate(d["decode"])
                                                      for r in x[3] if "raw_valid" in r and not all(r["raw_valid"])}),
                      "used_invalid_layers": sorted({i for d in dec for i, x in enumerate(d["decode"])
                                                    for r in x[3] if r.get("used_valid") is False}),
                      "kernel_set_varies_layers": sorted({i for d in dec for i, x in enumerate(d["decode"])
                                                         for r in x[3] if r.get("raw_sets", 1) > 1})}
            rep.setdefault(k, {})[f"stage{st}"] = {"layers": layers, "order_or_set_varies": order,
                                                  "set_varies": sets, **dv}
    for e in rep.values():
        for f in ("logits", "tokens", "margins"):
            if f in e:
                e[f] = sorted(e[f]) if f != "logits" else len(e[f])
    return rep


def judge(cases: list[dict], res: list[dict], min_margin: float) -> int:
    bad = []
    for c in cases:
        if not c.get("expect_top2"):
            continue
        last = max(r["position"] for r in res if r["case"] == c["name"])
        for r in (r for r in res if r["case"] == c["name"] and r["position"] == last):
            top2 = [t[0] for t in r["bf16"]["top"][:2]]
            ok = set(c["expect_top2"]) <= set(top2) and r["bf16"]["margin"] < min_margin
            print(f"[margin] judge {c['name']} repeat {r['repeat']}: top 2 {top2}, margin {r['bf16']['margin']}, "
                  f"server tokens {c['expect_top2']}: {'near tie' if ok else 'NOT a near tie'}", flush=True)
            if not ok:
                bad.append(c["name"])
    print(f"[margin] judge: {'pass' if not bad else 'fail: ' + ', '.join(sorted(set(bad)))}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    main()
