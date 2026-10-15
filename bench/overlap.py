"""Back-to-back forwards without a sync vs the same forwards run one at a time: must be bitwise equal.

    python bench/overlap.py [--stage 1] [--plans 5] [--reqs 8] [--len 1024] [--sleep-ms 2000]

The runtime builds a stage's metadata (host) up to several microbatches ahead of its GPU. Any host buffer that a
non-blocking H2D copy reads, and that a later `prepare` rewrites before the GPU has run that copy, corrupts the
earlier microbatch. This bench enqueues `--plans` distinct microbatches back to back behind a GPU sleep (the host
is then that many forwards ahead), and compares every output with a synchronous run of the same plans (the same
order, so KV and state evolve identically). Plans: `--reqs` requests of `--len` tokens each from position 0, plus,
from the second plan on, a continuation chunk of a request started in the previous plan (initial KDA/conv state,
KV prefix).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers as lb  # noqa: E402


def make_plans(n_plans: int, reqs: int, L: int, tokens: list[int]):
    nxt = [1]

    def blocks(n):
        b = list(range(nxt[0], nxt[0] + n))
        nxt[0] += n
        return b

    plans, carry = [], None
    for p in range(n_plans):
        chunks = []
        for r in range(reqs):
            kv, st = blocks(1), blocks(6)
            chunks.append(NS(req_id=f"p{p}r{r}", start=0, num_tokens=L, kv_blocks=kv, state_blocks=st,
                             is_last=True, token_ids=tokens[(p * reqs + r) * 7 % 4000:][:L]))
        if carry is not None:     # second chunk of the previous plan's long request
            kv, st, start = carry
            chunks.append(NS(req_id=f"c{p}", start=start, num_tokens=L, kv_blocks=kv, state_blocks=st,
                             is_last=True, token_ids=tokens[start:start + L]))
        kv, st = blocks(1), blocks(6)
        chunks.append(NS(req_id=f"long{p}", start=0, num_tokens=L, kv_blocks=kv, state_blocks=st, is_last=False,
                         token_ids=tokens[:L]))
        carry = (kv, st, L)
        plans.append(NS(mb_id=p, chunks=chunks))
    return plans, nxt[0]


def blocked_at(model, plans, hidden, sleep_cycles: int) -> dict:
    """Where the host waits for the GPU while it enqueues the forwards behind a GPU sleep: the main thread's Python
    stack sampled every 2 ms (a C call that blocks shows as its Python caller), the most common stacks; and the
    synchronizing torch calls (`torch.cuda.set_sync_debug_mode`) with their callers."""
    import collections
    import threading
    import traceback
    import warnings

    import torch

    main = threading.get_ident()
    samples: collections.Counter = collections.Counter()
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            f = sys._current_frames().get(main)
            if f is not None:
                st = traceback.extract_stack(f)[-8:]
                samples[" <- ".join(f"{Path(x.filename).name}:{x.lineno}:{x.name}" for x in reversed(st))] += 1
            time.sleep(0.002)

    syncs: collections.Counter = collections.Counter()
    old_show = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        st = traceback.extract_stack()[-9:-1]
        syncs[f"{message}".splitlines()[0][:80] + " | " + " <- ".join(
            f"{Path(x.filename).name}:{x.lineno}:{x.name}" for x in reversed(st))] += 1

    torch.cuda.synchronize()
    torch.cuda._sleep(sleep_cycles)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    warnings.showwarning = show
    torch.cuda.set_sync_debug_mode("warn")
    t = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show
            for p, h in zip(plans, hidden):
                model.forward(p, h)
    finally:
        torch.cuda.set_sync_debug_mode(0)
        warnings.showwarning = old_show
        stop.set()
        th.join()
    host_ms = (time.perf_counter() - t) * 1e3
    torch.cuda.synchronize()
    n = sum(samples.values())
    return {"host_ms": round(host_ms, 1), "samples": n,
            "top_stacks": [[c, s] for s, c in samples.most_common(6)],
            "sync_calls": [[c, s] for s, c in syncs.most_common(20)]}


def main() -> None:
    import torch
    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--plans", type=int, default=5)
    ap.add_argument("--reqs", type=int, default=7)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--sleep-ms", type=float, default=2000)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    tokens = lb.enwik9_tokens(CHECKPOINT, 8192)
    plans, used = make_plans(a.plans, a.reqs, a.len, tokens)
    budget = max(sum(c.num_tokens for c in p.chunks) for p in plans)
    model = StageModel(bench_config(max(budget, 8192)), a.stage, dev, num_blocks=used + 8,
                       max_model_len=4 * a.len + 8576)
    print(f"[overlap] stage {a.stage} layers {model.layer_range}; {len(plans)} plans of {budget} tokens; "
          f"tactics {model.tactics.get('source')}", flush=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    hidden = [None if model.first else
              torch.randn(sum(c.num_tokens for c in p.chunks), model.hidden_width, device=dev, generator=gen,
                          dtype=torch.float32).to(torch.bfloat16) for p in plans]
    cpm = lb.sleep_cycles_per_ms()
    from ppimpl.model import metadata as md_mod
    from ppimpl.model.pp_patch import TOPK_SEGMENTS

    packs: list = []          # (host snapshot right after the copy was issued, host buffer, device buffer)
    orig_copy = md_mod.Pack.copy

    def traced_copy(self, device, pin=True):
        orig_copy(self, device, pin)
        packs.append((self.host_buf.clone(), self.host_buf, self.dev_buf))

    md_mod.Pack.copy = traced_copy

    def pack_check():
        """Per Pack of the pass: was its pinned host buffer changed after the copy was issued, and does the device
        buffer hold what the host had at that moment."""
        torch.cuda.synchronize()
        out = [{"host_changed": not torch.equal(snap, hb), "dev_equals_snapshot": torch.equal(db.cpu(), snap)}
               for snap, hb, db in packs]
        packs.clear()
        return out

    def compare(outs):
        rows = []
        for i, (x, y) in enumerate(zip(ref, outs)):
            d = (x.float() - y.float()).abs()
            rows.append({"plan": i, "equal": bool(torch.equal(x, y)), "max_abs": float(d.max()),
                         "rows_differing": int((d.reshape(d.shape[0], -1).amax(1) > 0).sum())})
        return rows

    res = {}
    with torch.inference_mode():
        ref = []
        for p, h in zip(plans, hidden):          # synchronous: one microbatch at a time
            ref.append(model.forward(p, h).clone())
            torch.cuda.synchronize()
        res["sync_packs"] = pack_check()
        for rep in range(2):                     # forward() back to back behind a GPU sleep
            torch.cuda.synchronize()
            torch.cuda._sleep(int(a.sleep_ms * cpm))
            t = time.perf_counter()
            outs, each = [], []
            for p, h in zip(plans, hidden):
                t1 = time.perf_counter()
                outs.append(model.forward(p, h))
                each.append(round((time.perf_counter() - t1) * 1e3, 1))
            host_ms = (time.perf_counter() - t) * 1e3
            rows = compare(outs)
            res[f"unsynced_{rep}"] = {"host_ms": round(host_ms, 1), "host_ms_each": each, "plans": rows,
                                      "packs": pack_check()}
            print(f"[overlap] host ms per forward {each}", flush=True)
            print(f"[overlap] unsynced pass {rep}: host {host_ms:.0f} ms (GPU sleep {a.sleep_ms:.0f} ms); "
                  f"{rows}; packs {res[f'unsynced_{rep}']['packs']}", flush=True)
        # every prepare() done (and its copy complete) before any run(): only run-time effects remain
        batches, segs = [], []
        for p in plans:
            batches.append(model.prepare(p))
            segs.append(dict(TOPK_SEGMENTS))
            torch.cuda.synchronize()
        torch.cuda._sleep(int(a.sleep_ms * cpm))
        outs = []
        for b, sg, h in zip(batches, segs, hidden):
            TOPK_SEGMENTS.clear()
            TOPK_SEGMENTS.update(sg)
            outs.append(model.run(b, h))
        rows = compare(outs)
        res["prepared_then_unsynced"] = {"plans": rows, "packs": pack_check()}
        print(f"[overlap] prepared, then runs back to back: {rows}", flush=True)
        # prepare() back to back behind the sleep (copies queued), each run() only after a sync
        torch.cuda._sleep(int(a.sleep_ms * cpm))
        batches, segs = [], []
        for p in plans:
            batches.append(model.prepare(p))
            segs.append(dict(TOPK_SEGMENTS))
        torch.cuda.synchronize()
        outs = []
        for b, sg, h in zip(batches, segs, hidden):
            TOPK_SEGMENTS.clear()
            TOPK_SEGMENTS.update(sg)
            outs.append(model.run(b, h))
            torch.cuda.synchronize()
        rows = compare(outs)
        res["queued_prepares_synced_runs"] = {"plans": rows, "packs": pack_check()}
        print(f"[overlap] prepares queued, runs synced: {rows}", flush=True)
        res["host_blocked_at"] = blocked_at(model, plans, hidden, int(a.sleep_ms * cpm))
        print(f"[overlap] host blocked at {json.dumps(res['host_blocked_at'], indent=1)}", flush=True)
        again = []
        for p, h in zip(plans, hidden):          # synchronous again: the run-to-run baseline
            again.append(model.forward(p, h).clone())
            torch.cuda.synchronize()
        res["sync_repeat_equal"] = [bool(torch.equal(x, y)) for x, y in zip(ref, again)]
        pack_check()
    res["pass"] = all(r["equal"] for k, v in res.items() if isinstance(v, dict) and "plans" in v
                      for r in v["plans"])
    print(f"[overlap] sync repeat equal {res['sync_repeat_equal']}; pass {res['pass']}", flush=True)
    out = lb.MEASURED / "host" / f"overlap_s{a.stage}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[overlap] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
