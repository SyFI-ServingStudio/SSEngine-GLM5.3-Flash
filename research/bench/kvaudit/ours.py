"""Our side of the KV-cache audit (research/bench/kvaudit/common.py): pp-impl PP4, one `StageModel` per GPU.

    python -m research.bench.kvaudit.ours --length 8192 [--out DIR]      # 4 GPUs

As research/bench/kvcheck.py: four processes joined by the serving transport (`open_pipe`, the CE ring), each holding its
stage's layers [12, 11, 11, 11]. Every case of `cases(L).ours` runs its rounds' chunks as consecutive microbatches on
fresh blocks (each case its own KV and 6 state blocks), then every stage dumps its layers' caches in the canonical
form; stage 3 also the first output token's log-probabilities. Eager (no CUDA graphs: the top-k hooks must see the
indexer; graph replay is bitwise eager, GPU test f).

Before the first dump each stage checks the indexer gather table against the one vLLM's indexer metadata builder
derives for the same blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

from research.bench.kvaudit import common as C
from ssengine.engine.kv import block_pages

JOB = os.environ.get("SLURM_JOB_ID", "local")
B = 8576


def _plans(case: C.OursCase, toks: list[int], kv: list[int], st: list[int]) -> list:
    plans, s = [], 0
    for rnd in case.rounds:
        for j, n in enumerate(rnd):
            last = j == len(rnd) - 1
            plans.append(NS(mb_id=len(plans), chunks=[NS(req_id=case.name, start=s, num_tokens=n,
                                                         kv_pages=block_pages(kv),
                                                         state_blocks=st, is_last=last, token_ids=toks[s:s + n])]))
            s += n
    return plans


def _check_indexer_table(m, kv: list[int], T: int, toks: list[int]) -> dict:
    """Our gather table == the builder's `indexer_block_table` for a prefill chunk ending at T."""
    import torch

    n = min(1000, T - 1)
    plan = NS(mb_id=-1, chunks=[NS(req_id="chk", start=T - n, num_tokens=n, kv_pages=block_pages(kv),
                                   state_blocks=[0] * 6,
                                   is_last=True, token_ids=toks[T - n:T])])
    b = m.meta.build(plan, with_tokens=m.first)
    i = next(i for i in C.DSA_LAYERS if i in m.layer_range)
    name = f"{C.PREFIX}.{i}.self_attn.indexer.k_cache"
    md = b.attn[name]
    ref = md.prefill.chunks[0].block_table[0].tolist()
    kvt = C.kv_of(m.vc.compilation_config.static_forward_context[name])
    mine = C.indexer_table(kv, T // C.KPOOL, B, kvt.shape[1])
    ok = ref[:len(mine)] == mine
    info = {"ok": ok, "cache_shape": list(kvt.shape), "mine": mine[:8], "builder": ref[:8]}
    torch.cuda.synchronize()
    if not ok:
        raise SystemExit(f"[kvaudit-ours] indexer gather table differs from the builder's: {info}")
    return info


def _worker(rank: int, L: int, port: int, out_root: str, only: list[str] | None) -> None:
    import torch
    from vllm.forward_context import set_forward_context

    from ssengine.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig
    from ssengine.model.stage import StageModel, bench_config
    from ssengine.runtime.transport import open_pipe

    cs = C.cases(L)
    todo = [c for c in cs.ours if not only or c.name in only]
    max_mb = max(n for c in todo for r in c.rounds for n in r)
    maxN = max(c.tokens for c in todo)
    nblocks = max(24, 1 + sum(-(-c.tokens // B) + 6 for c in todo))   # >= what StageModel.warmup's plans use
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    pipe = open_pipe(RunConfig(port=port, scheduler=SchedulerConfig(mb_tokens=max_mb)), rank, dev, HIDDEN_WIDTH)
    t0 = time.time()
    m = StageModel(bench_config(max_mb), rank, dev, num_blocks=nblocks, max_model_len=maxN + B, cuda_graphs=False)
    hooks = C.install_topk_hooks(m.model)
    ctx = m.vc.compilation_config.static_forward_context
    kda_index = m.kv.kda_state_index()
    print(f"[kvaudit-ours] rank {rank} layers {m.layer_range} built in {time.time() - t0:.0f} s, {nblocks} blocks, "
          f"{hooks} indexer hooks", flush=True)
    ids = C.prompt(maxN)
    nxt = iter(range(1, nblocks))
    checked = None
    with torch.inference_mode():
        for case in todo:
            T = case.tokens
            kv = [next(nxt) for _ in range(-(-T // B))]
            st = [next(nxt) for _ in range(6)]
            plans = _plans(case, ids, kv, st)
            C.topk_start(T)
            logits = None
            t1 = time.time()
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
                else:
                    b = m.prepare(plan)
                    with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                        x, r, po, co = m.run_layers(b, hidden)
                        if b.last_rows.shape[0]:
                            logits = m.head_logits(b, x, r, po, co)[0]
                    pipe.release()
            torch.cuda.synchronize()
            dt = time.time() - t1
            if checked is None:
                checked = _check_indexer_table(m, kv, T, ids)
            d = Path(out_root) / case.name
            d.mkdir(parents=True, exist_ok=True)
            for i in m.layer_range:
                if i in C.DSA_LAYERS:
                    layer = C.dsa_layer(ctx, i, kv, st[5], T, B)
                else:
                    layer = C.kda_layer(ctx, i, st[kda_index[f"{C.PREFIX}.{i}.self_attn"]])
                torch.save(layer, d / f"layer{i:02d}.pt")
            torch.save(C.topk_take(), d / f"topk_rank{rank}.pt")
            if rank == 3:
                torch.save(torch.log_softmax(logits.float(), -1).cpu(), d / "logprobs.pt")
                C.write_meta(d, {"side": "ours", "case": case.name, "tokens": T, "rounds": case.rounds, "job": JOB,
                                 "prefill_s": round(dt, 2), "token": int(logits.argmax()), "block_size": B,
                                 "kv_blocks": kv, "state_blocks": st, "indexer_table_check": checked})
            print(f"[kvaudit-ours] rank {rank} {case.name}: {len(plans)} microbatches {dt:.1f} s", flush=True)
    pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    import torch.multiprocessing as mp

    from benchmarks.common import free_port_block

    # the pipe's torch.distributed rendezvous binds base_port + 1 + stages: probe a free block (other users hold 20000s ports)
    port = free_port_block(23000 + (int(JOB) % 100) * 100 if JOB.isdigit() else 23000)
    print(f"[kvaudit-ours] port block {port}", flush=True)
    out = a.out or C.DUMPS / str(a.length)
    mp.spawn(_worker, args=(a.length, port, str(out), a.only), nprocs=4, join=True)


if __name__ == "__main__":
    main()
