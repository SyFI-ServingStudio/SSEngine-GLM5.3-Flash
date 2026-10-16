"""vLLM side of the KV-cache audit (bench/kvaudit/common.py): the reference engine in process, TP4 + EP.

    python -m bench.kvaudit.vllm_side --length 8192 [--out DIR]      # 4 GPUs

Flags as the reference server (measured/reference.md, job 7029): fp8_e4m3 KV, prefix caching off, text model only,
max_num_batched_tokens 16384 (the server's chunked-prefill budget), greedy. Differences, for this audit: in process
(`vllm.LLM`), max_model_len sized to the largest case, `enforce_eager` (no torch.compile/CUDA graphs; forward hooks
see the indexer), full-vocabulary logprobs (max_logprobs -1).

Each case is one `generate` call (max_tokens 1); after it, `collective_rpc` runs `_dump` on every TP rank: the
request's block ids (recorded from the scheduler output each step by `_setup`'s wrappers of the model runner's
`add_requests` / `update_requests`) -> canonical tensors. With prefix caching off nothing reuses a finished
request's blocks before the next request, so they still hold its cache.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")    # collective_rpc of the functions below

from bench.kvaudit import common as C  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "local")


# ------------------------------------------------------------------------------------------------- worker side


def _setup(self) -> dict:
    """Record every request's block ids per KV cache group; topk hooks on rank 0."""
    import torch.distributed as dist

    runner = self.model_runner
    rec: dict = {}
    add, upd = runner.add_requests, runner.update_requests

    def add_requests(so):
        for r in so.scheduled_new_reqs:
            rec[r.req_id] = {"prompt_len": int(r.prompt_len), "blocks": [list(g) for g in r.block_ids],
                             "seq": len(rec)}
        return add(so)

    def update_requests(so):
        updates = so.block_table_updates or {}
        for rid, bids in updates.items():
            if rid in rec:
                rec[rid]["blocks"] = [list(g) for g in bids]
        reqs = so.scheduled_cached_reqs
        for rid, nb in zip(reqs.req_ids, reqs.new_block_ids):
            if nb is not None and rid not in updates and rid in rec:
                for g, ids in enumerate(nb):
                    rec[rid]["blocks"][g].extend(ids)
        return upd(so)

    runner.add_requests, runner.update_requests = add_requests, update_requests
    runner._kvaudit = rec
    rank = dist.get_rank()
    hooks = C.install_topk_hooks(runner.model) if rank == 0 else 0
    cfg = runner.kv_cache_config
    return {"rank": rank, "block_size": int(self.vllm_config.cache_config.block_size),
            "num_blocks": int(cfg.num_blocks), "topk_hooks": hooks,
            "groups": [[type(g.kv_cache_spec).__name__, len(g.layer_names), g.layer_names[0]]
                       for g in cfg.kv_cache_groups],
            "runner": type(runner).__name__}


def _topk_start(self, T: int) -> None:
    C.topk_start(T)


def _dump(self, T: int, out: str) -> dict:
    """The newest request of T prompt tokens -> rank 0 writes the replicated DSA tensors and every rank its KDA
    shards; returns the replicated tensors' fingerprints and the block ids used."""
    import torch
    import torch.distributed as dist

    runner = self.model_runner
    rank = dist.get_rank()
    rid, r = max(((k, v) for k, v in runner._kvaudit.items() if v["prompt_len"] == T), key=lambda kv: kv[1]["seq"])
    cfg = runner.kv_cache_config
    gid = {name: g for g, grp in enumerate(cfg.kv_cache_groups) for name in grp.layer_names}
    bt = int(self.vllm_config.cache_config.block_size)
    ctx = self.vllm_config.compilation_config.static_forward_context
    out = Path(out)
    fps, kda, used = {}, {}, {}
    torch.cuda.synchronize()
    with torch.inference_mode():
        for i in C.DSA_LAYERS:
            mla_name = f"{C.PREFIX}.{i}.self_attn.attn"
            g0 = r["blocks"][gid[mla_name]][: -(-T // bt)]
            assert gid[f"{C.PREFIX}.{i}.self_attn.indexer.k_cache"] == gid[mla_name]
            tail = r["blocks"][gid[f"{C.PREFIX}.{i}.self_attn.indexer.tail_cache"]][0]
            d = C.dsa_layer(ctx, i, g0, tail, T, bt)
            fps[i] = C.dsa_fingerprint(d)
            used[i] = {"g0": g0, "tail": tail}
            if rank == 0:
                torch.save(d, out / f"layer{i:02d}.pt")
        for i in C.KDA_LAYERS:
            name = f"{C.PREFIX}.{i}.self_attn"
            blocks = r["blocks"][gid[name]]
            kda[i] = C.kda_layer(ctx, i, blocks[0])
            used[i] = {"state": blocks}
        torch.save(kda, out / f"kda_rank{rank}.pt")
        if rank == 0:
            torch.save(C.topk_take(), out / "topk.pt")
    return {"rank": rank, "req_id": rid, "fingerprints": fps, "blocks": used, "block_size": bt}


# ------------------------------------------------------------------------------------------------- driver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="case names to run (default: all of the length)")
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams

    from ppimpl.config import CHECKPOINT

    cs = C.cases(a.length)
    todo = [c for c in cs.vllm if not a.only or c.name in a.only]
    out_root = a.out or C.DUMPS / str(a.length)
    maxN = max(c.tokens for c in todo)
    t0 = time.time()
    llm = LLM(model=CHECKPOINT, tensor_parallel_size=4, enable_expert_parallel=True, kv_cache_dtype="fp8_e4m3",
              language_model_only=True, max_model_len=maxN + 64, enable_prefix_caching=False, enforce_eager=True,
              max_num_batched_tokens=16384, max_logprobs=-1, seed=0)
    info = llm.collective_rpc(_setup)
    print(f"[kvaudit-vllm] engine up in {time.time() - t0:.0f} s: {json.dumps(info[0])}", flush=True)
    vocab = llm.llm_engine.model_config.get_vocab_size()
    ids = C.prompt(maxN)
    fill = C.filler(maxN)
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, logprobs=-1, detokenize=False)
    for case in todo:
        d = out_root / case.name
        d.mkdir(parents=True, exist_ok=True)
        tokens = ids[:case.tokens]
        prompts = ([{"prompt_token_ids": fill}] if case.filler_first else []) + [{"prompt_token_ids": tokens}]
        llm.collective_rpc(_topk_start, args=(case.tokens,))
        t1 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.time() - t1
        res = llm.collective_rpc(_dump, args=(case.tokens, str(d)))
        o = outs[-1].outputs[0]
        lp = o.logprobs[0]
        full = torch.full((vocab,), float("-inf"))
        for tok, v in lp.items():
            full[tok] = v.logprob
        torch.save(full, d / "logprobs.pt")
        # KDA: merge the rank shards into our layout, one file per layer
        shards = [torch.load(d / f"kda_rank{r}.pt") for r in range(4)]
        for i in C.KDA_LAYERS:
            torch.save(C.merge_kda_shards([s[i] for s in shards]), d / f"layer{i:02d}.pt")
        for r in range(4):
            (d / f"kda_rank{r}.pt").unlink()
        fp0 = res[0]["fingerprints"]
        replicas = {str(i): {k: all(x["fingerprints"][i][k] == fp0[i][k] for x in res) for k in fp0[i]}
                    for i in fp0}
        bad = [f"{i}.{k}" for i, v in replicas.items() for k, ok in v.items() if not ok]
        C.write_meta(d, {"side": "vllm", "case": case.name, "tokens": case.tokens, "filler_first": case.filler_first,
                         "job": JOB, "prefill_s": round(dt, 2), "token": int(o.token_ids[0]),
                         "block_size": res[0]["block_size"], "blocks": res[0]["blocks"],
                         "replicas_equal": not bad, "replicas_differ": bad, "engine": info[0]})
        print(f"[kvaudit-vllm] {case.name}: {dt:.1f} s, token {o.token_ids[0]}, replicas "
              f"{'equal' if not bad else 'DIFFER ' + ','.join(bad)}", flush=True)
    print(f"[kvaudit-vllm] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
