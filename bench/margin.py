"""Top-2 margins of chosen prompts on the real model, chunked as the server chunks them, repeated (4 GPUs).

  python -m bench.margin <cases.json> [--repeats 3] [--out measured/margins/<name>.json]

cases.json: [{"name": str, "file": <.npy int32 token ids>, "schedule": [[start, tokens, is_last], ...]}, ...]. Each
case runs `--repeats` times on the same KV and state blocks (one request, a fresh start each time), each schedule
entry as one microbatch, as the server runs a request alone. At every is_last entry: the last position's logits,
bf16 as the server's argmax takes them (`StageModel.output_head`) and FP32, their top-5 and the top-1 minus top-2
margin in logprob (nats, the margin rule of bench/check_tokens.py). Repeats show whether the same schedule gives the
same token in one process; bench/prefix_check.py shows it across server processes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

from bench.common import MEASURED

JOB = os.environ.get("SLURM_JOB_ID", "local")
MB = 8192


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


def _worker(rank: int, cases: list[dict], repeats: int, port: int, out: str) -> None:
    import numpy as np
    import torch
    from ppimpl.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig
    from ppimpl.model.stage import StageModel, bench_config
    from ppimpl.runtime.transport import open_pipe
    from vllm.forward_context import set_forward_context

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    toks = {c["name"]: np.load(c["file"]).astype(np.int64).tolist() for c in cases}
    L = max(len(t) for t in toks.values())
    nkv = -(-L // 8576)
    rc = RunConfig(port=port, scheduler=SchedulerConfig(mb_tokens=MB))
    pipe = open_pipe(rc, rank, dev, HIDDEN_WIDTH)
    m = StageModel(bench_config(MB, cuda_graphs=True), rank, dev, num_blocks=nkv + 8, max_model_len=L + 8576,
                   cuda_graphs=rank < 3)
    kv, st = list(range(1, nkv + 1)), list(range(nkv + 1, nkv + 7))
    res = []
    with torch.inference_mode():
        for c in cases:
            t = toks[c["name"]]
            for rep in range(repeats):
                t0 = time.time()
                for i, (s, n, last) in enumerate(c["schedule"]):
                    plan = NS(mb_id=i, chunks=[NS(req_id="p", start=s, num_tokens=n, kv_blocks=kv,
                                                  state_blocks=st, is_last=last, token_ids=t[s:s + n])])
                    hidden = pipe.recv(n, HIDDEN_WIDTH) if rank > 0 else None
                    if rank < 3:
                        o = m.forward(plan, hidden)
                        done = torch.cuda.Event()
                        done.record()
                        if hidden is not None:
                            pipe.release()
                        pipe.send(o, done)
                        continue
                    b = m.prepare(plan)
                    with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                        x, r, po, co = m.run_layers(b, hidden)
                        if last:
                            lb = m.head_logits(b, x, r, po, co, fp32=False)[0]
                            lf = m.head_logits(b, x, r, po, co, fp32=True)[0]
                    pipe.release()
                    if last:
                        res.append({"case": c["name"], "repeat": rep, "position": s + n - 1,
                                    "token": int(lb.argmax()), "bf16": _top(lb), "fp32": _top(lf)})
                        print(f"[margin] {c['name']} repeat {rep} pos {s + n - 1}: token {int(lb.argmax())} "
                              f"margin bf16 {res[-1]['bf16']['margin']} fp32 {res[-1]['fp32']['margin']} "
                              f"({time.time() - t0:.1f} s)", flush=True)
                torch.cuda.synchronize()
    if rank == 3:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1) + "\n")
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
    a = ap.parse_args()
    import torch.multiprocessing as mp

    cases = json.loads(a.cases.read_text())
    out = a.out or MEASURED / "margins" / f"{a.cases.stem}_{JOB}.json"
    port = 21500 + (int(JOB) % 100) * 100 if JOB.isdigit() else 21500
    from bench.common import free_port_block
    port = free_port_block(port)
    mp.spawn(_worker, args=(cases, a.repeats, port, str(out)), nprocs=4, join=True)
    if a.judge:
        raise SystemExit(judge(cases, json.loads(Path(out).read_text()), a.min_margin))


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
