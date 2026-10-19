"""TRTLLM sparse MLA (config 25) at the simulator's row shape vs served microbatch shapes (1 GPU).

Served stage-2 microbatches run the sparse-MLA kernel at ~1.0 ms per 4,096-token call; the simulator prices the same
calls at the profile.db row it maps them to (4,096 queries, 5,254 cache tokens, causal ramp capped at 2,176:
0.649 ms). This times one callable, `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` with the rope-free
GLM-5.3 layout (64 heads, latent 512, top-k 2,176, FP8 query and cache, page 64), on operands built like the
ServingStudioSim profiler runner's (`dsa_sparse_mla_attention`, PROFILER), case by case:

  - `row`: the runner's own row operands (valid counts c:1159..5254@2176 over a 5,254-token cache,
    unique_scattered_pages), as the simulator prices;
  - for each served microbatch (`--run`'s microbatches.jsonl, stage `--stage`, exactly `--tokens` rows; its
    requests' prefix and append give every query row's valid count min(2,176, prefix + i + 1)):
      `mb<i>.ctx`: indices spread over each request's own context, laid out contiguously from slot 0 (a small cache);
      `mb<i>.pool`: the same context positions, each request's KV in blocks of 8,576 tokens placed at random block
        ids of the stage's pool (`--pool-blocks` x 8,576 slots: the served layout, ssengine/model/kv*.py);
  - `u2176.pool`: every row 2,176 indices over the pool (the fully past-prefix case).
Indices within a context are stratified uniform (one per equal slice of the row's context), unique and ascending.

Timing per case, the same operands: `cupti` = the profiler's Timer.cupti (64 MiB L2 flush before each launch, idle
between launches), `sustained` = `--calls` back-to-back launches between two CUDA events after a warm-up (as served,
no flush), with the SM clock sampled (NVML) at the end.

  python -m research.bench.sparse_mla_shapes --run research/measured/runs_m5/nsys_good_n400_7787 --stage 2 --mbs 6 \\
      --out research/measured/sparse_mla_shapes/<job>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from benchmarks.common import PROFILER

HEADS, LATENT, TOPK, PAGE, BLOCK = 64, 512, 2176, 64, 8576
SCALE = 0.0625


def row_counts(requests: list[dict], k: int = TOPK) -> list[int]:
    return [min(k, r["prefix"] + i + 1) for r in requests for i in range(r["append"])]


def served_mbs(run: Path, stage: int, tokens: int, n: int) -> list[dict]:
    mbs = [m for m in map(json.loads, open(run / "microbatches.jsonl"))
           if m["stage"] == stage and sum(r["append"] for r in m["requests"]) == tokens]
    if len(mbs) <= n:
        return mbs
    # spread over the run, and over context: sort by mean context, take evenly spaced
    mbs.sort(key=lambda m: sum(r["append"] * (r["prefix"] + r["append"] / 2) for r in m["requests"]))
    return [mbs[round(i * (len(mbs) - 1) / (n - 1))] for i in range(n)]


def build_indices(torch, requests, *, pool_blocks: int | None, seed: int, device):
    """[rows, TOPK] int32 slot indices and [rows] counts; pool_blocks None = each request's context contiguous from
    its own offset in a compact cache (requests back to back)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    blocks_used = 0
    out_idx, out_cnt = [], []
    perm = torch.randperm(pool_blocks, generator=g) if pool_blocks else None
    base = 0
    for r in requests:
        ctx = r["prefix"] + r["append"]
        nblk = math.ceil(ctx / BLOCK)
        if pool_blocks:
            blk = perm[blocks_used:blocks_used + nblk].to(torch.int64)
            blocks_used += nblk
        rows = torch.arange(r["append"]) + r["prefix"] + 1          # visible context per row
        cnt = rows.clamp(max=TOPK)
        j = torch.arange(TOPK)
        u = torch.rand(r["append"], TOPK, generator=g)
        pos = ((j + u) * rows.unsqueeze(1) / cnt.unsqueeze(1)).floor().to(torch.int64)   # stratified, ascending
        pos = torch.minimum(pos, rows.unsqueeze(1) - 1)
        if pool_blocks:
            slot = blk[pos // BLOCK] * BLOCK + pos % BLOCK
        else:
            slot = pos + base
        slot = torch.where(j.unsqueeze(0) < cnt.unsqueeze(1), slot, torch.zeros_like(slot))
        out_idx.append(slot.to(torch.int32))
        out_cnt.append(cnt.to(torch.int32))
        base += ctx
    idx = torch.cat(out_idx).to(device)
    cnt = torch.cat(out_cnt).to(device)
    cache_tokens = pool_blocks * BLOCK if pool_blocks else base
    return idx, cnt, cache_tokens


def make_operands(torch, idx, cnt, cache_tokens, device, cache=None):
    fp8 = torch.float8_e4m3fn
    q = torch.empty((idx.shape[0], 1, HEADS, LATENT), dtype=fp8, device=device)
    q.copy_(torch.linspace(-0.75, 0.75, LATENT, device=device).view(1, 1, 1, LATENT).expand_as(q))
    pages = math.ceil(cache_tokens / PAGE)
    if cache is None or cache.shape[0] < pages:
        cache = torch.empty((pages, 1, PAGE, LATENT), dtype=fp8, device=device)
        cache.copy_(torch.linspace(-0.5, 0.5, LATENT, device=device).view(1, 1, 1, LATENT).expand(pages, 1, PAGE,
                                                                                                    LATENT))
    return dict(query=q, kv_cache=cache[:pages], block_tables=idx.view(-1, 1, TOPK), seq_lens=cnt,
                top_k_lens=cnt.clamp(min=1))


def launcher(ops, workspace):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    def run():
        return trtllm_batch_decode_with_kv_cache_mla(
            query=ops["query"], kv_cache=ops["kv_cache"], workspace_buffer=workspace, qk_nope_head_dim=LATENT,
            kv_lora_rank=LATENT, qk_rope_head_dim=0, block_tables=ops["block_tables"], seq_lens=ops["seq_lens"],
            max_seq_len=TOPK, bmm1_scale=SCALE, bmm2_scale=1.0, sparse_mla_top_k=TOPK,
            sparse_mla_top_k_lens=ops["top_k_lens"])
    return run


def sm_clock():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetClockInfo(pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


def sustained(torch, run, calls: int, warm: int = 20) -> float:
    for _ in range(warm):
        run()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(calls):
        run()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--mbs", type=int, default=6)
    ap.add_argument("--pool-blocks", type=int, default=7143)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--no-cupti", action="store_true")
    ap.add_argument("--dry", action="store_true", help="CPU: build the cases' counts and indices only")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch

    device = "cpu" if a.dry else "cuda"
    if str(PROFILER) not in sys.path:
        sys.path.append(str(PROFILER))
    mbs = served_mbs(a.run, a.stage, a.tokens, a.mbs)
    cases = []
    for i, m in enumerate(mbs):
        reqs = m["requests"]
        meta = {"mb": m["index"], "requests": len(reqs), "prefix_max": max(r["prefix"] for r in reqs),
                "valid_mean": round(sum(row_counts(reqs)) / a.tokens, 1)}
        cases.append((f"mb{i}.ctx", reqs, None, meta))
        cases.append((f"mb{i}.pool", reqs, a.pool_blocks, meta))
    cases.append(("u2176.pool", [{"prefix": 10 ** 6, "append": a.tokens}], a.pool_blocks, {}))
    report = {"job": os.environ.get("SLURM_JOB_ID"), "run": str(a.run), "stage": a.stage, "cases": []}
    if a.dry:
        for name, reqs, pool, meta in cases:
            r = [dict(x, prefix=min(x["prefix"], (pool or 10 ** 9) * BLOCK - x["append"] - 1)) for x in reqs]
            idx, cnt, ct = build_indices(torch, r, pool_blocks=pool, seed=0, device="cpu")
            assert idx.shape == (a.tokens, TOPK) and int(idx.max()) < ct and bool((idx >= 0).all())
            print(name, meta, "cache", ct, "valid mean", round(float(cnt.float().mean()), 1), flush=True)
        return
    from profiling.profilers.timer import Timer
    from profiling.runners.attention import dsa_sparse_mla_attention as R

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    # the row, through the runner itself (its Timer.cupti) and through our sustained timing of its operands
    row_args = dict(num_queries=a.tokens, num_cache_tokens=5254, num_heads=HEADS, num_kv_heads=1, selected_k=TOPK,
                    latent_dim=LATENT, rope_dim=0, value_dim=LATENT, softmax_scale=SCALE, q_dtype="fp8_e4m3",
                    cache_dtype="fp8_e4m3", index_dtype="int32", output_dtype="bf16",
                    valid_counts=f"c:{5254 - a.tokens + 1}..5254@{TOPK}", index_distribution="unique_scattered_pages",
                    cache_layout="hnd_paged_mqa_fp8_latent")
    t0 = time.perf_counter()
    m = R.profile_dsa_sparse_mla_attention_flashinfer_trtllm_fp8(**row_args)
    v = R._validate_args(**row_args, expected_q_dtype=R.DType.FP8_E4M3, expected_cache_dtype=R.DType.FP8_E4M3,
                         expected_cache_layout=R._TRTLLM_FP8_NOPE_CACHE_LAYOUT, expected_rope_dim=0)
    ro = R._build_trtllm_fp8_operands(torch, v, num_heads=HEADS, rope_dim=0, selected_k=TOPK, device=device)
    run = launcher(dict(query=ro.query, kv_cache=ro.cache, block_tables=ro.block_tables, seq_lens=ro.seq_lens,
                        top_k_lens=ro.top_k_lens), workspace)
    case = {"case": "row", "cupti_ms": round(m.time_ms, 4), "sustained_ms": round(sustained(torch, run, a.calls), 4),
            "sm_mhz": sm_clock(), "cache_tokens": 5254, "valid_mean": round(sum(v.valid_counts) / a.tokens, 1),
            "wall_s": round(time.perf_counter() - t0, 1)}
    report["cases"].append(case)
    print(json.dumps(case), flush=True)
    cache = None
    for name, reqs, pool, meta in cases:
        t0 = time.perf_counter()
        idx, cnt, ct = build_indices(torch, reqs if pool is None or name != "u2176.pool" else
                                     [{"prefix": pool * BLOCK - a.tokens - 1, "append": a.tokens}],
                                     pool_blocks=pool, seed=len(report["cases"]), device=device)
        ops = make_operands(torch, idx, cnt, ct, device, cache)
        cache = ops["kv_cache"] if pool else cache
        run = launcher(ops, workspace)
        run()
        torch.cuda.synchronize()
        case = {"case": name, **meta, "cache_tokens": ct, "valid_mean": round(float(cnt.float().mean()), 1),
                "cupti_ms": None if a.no_cupti else round(Timer.cupti(run, kernel_name="fmhaSm100"), 4),
                "sustained_ms": round(sustained(torch, run, a.calls), 4), "sm_mhz": sm_clock(),
                "wall_s": round(time.perf_counter() - t0, 1)}
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
