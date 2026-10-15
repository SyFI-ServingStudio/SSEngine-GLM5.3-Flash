"""Build the re-measure work queue: every row of the selected configurations, in chunks.

Configuration numbers follow lanes/model/measured/kernels_identity.md (the 35 of
2_kernels.md §2). Each chunk is one profiler CLI call (one kind/backend, <= --chunk specs),
written to <queue>/todo/<nn>__<kind>__<backend>__<k>.jsonl with a sidecar .rows.jsonl
holding the handoff rows (db_time_ms, config) in the same order.

usage: python make_queue.py <kernel_specs.jsonl> <queue_dir> --configs 1,5,7,... [--chunk 100]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# (kind, backend, fixed fields) of each configuration number.
CONFIGS = {
    1: ("mhc_pre_rms_norm", "vllm_tilelang", {}),
    2: ("single_gemm", "torch_linear_vllm", {"n": 24896, "k": 4096}),
    3: ("single_gemm", "torch_linear_vllm", {"n": 8192, "k": 128}),
    4: ("gdn_causal_conv_prefill", "dao_channellast", {}),
    5: ("kda_chunk_prefill", "flashinfer_cute_persistent", {}),
    6: ("single_gemm", "torch_linear_vllm", {"n": 4096, "k": 8192}),
    7: ("mhc_fused_post_pre_rms_norm", "deepgemm_mega_nonshifted", {}),
    8: ("nvfp4_quant", "vllm_cuda", {"hidden_size": 4096, "scale_format": "swizzled_e4m3"}),
    9: ("single_gemm", "flashinfer_cutedsl", {"n": 24576, "k": 4096}),
    10: ("nvfp4_quant", "vllm_cuda", {"hidden_size": 12288, "scale_format": "swizzled_e4m3"}),
    11: ("single_gemm", "flashinfer_cutedsl", {"n": 4096, "k": 12288}),
    12: ("single_gemm", "torch_linear_vllm", {"n": 2048, "k": 4096}),
    13: ("q_kv_rms_norm", "vllm_triton", {}),
    14: ("single_gemm", "torch_linear_vllm", {"n": 16384, "k": 1536}),
    15: ("single_gemm", "torch_linear_vllm", {"n": 4096, "k": 1536}),
    16: ("single_gemm", "torch_linear_vllm", {"n": 160, "k": 4096}),
    17: ("gemm_fp32_output", "torch_cublas", {"n": 32, "k": 4096}),
    18: ("single_gemm", "torch_linear_vllm", {"n": 128, "k": 4096}),
    19: ("dsa_mqa_logits_prefill", "deepgemm_fp8", {}),
    20: ("dsa_topk_prefill", "deep_select", {}),
    21: ("dsa_topk_prefill", "vllm_cuda", {}),
    22: ("batched_gemm", "torch_mla_q_absorb_no_rope", {}),
    23: ("mla_cache_append", "vllm_cuda", {}),
    24: ("dsa_sparse_index_remap", "vllm_triton", {}),
    25: ("dsa_sparse_mla_attention", "flashinfer_trtllm_fp8", {}),
    26: ("batched_gemm", "torch_mla_v_up_unpadded", {}),
    27: ("single_gemm", "torch_linear_vllm", {"n": 4096, "k": 16384}),
    28: ("gemm_fp32_output", "torch_cublas", {"n": 288, "k": 4096}),
    29: ("nvfp4_quant", "vllm_cuda", {"hidden_size": 4096, "scale_format": "linear_e4m3"}),
    30: ("nvfp4_fused_moe", "flashinfer_trtllm_sm100", {}),
    31: ("single_gemm", "torch_linear_vllm", {"n": 4096, "k": 4096}),
    32: ("single_gemm", "torch_linear_vllm", {"n": 4096, "k": 2048}),
    33: ("mhc_fused_post_pre_rms_norm", "vllm_tilelang", {}),
    34: ("rms_norm", "vllm_cuda", {}),
    35: ("single_gemm", "torch_linear_vllm", {"n": 154880, "k": 4096}),
}


def config_of(row):
    hits = [c for c, (kind, backend, fixed) in CONFIGS.items()
            if row["kind"] == kind and row["backend"] == backend
            and all(row["spec"].get(f) == v for f, v in fixed.items())]
    # A (kind, backend) with several configurations: the one with fixed fields wins.
    hits.sort(key=lambda c: -len(CONFIGS[c][2]))
    return hits[0]


def cli_spec(spec):
    out = dict(spec)
    if isinstance(out.get("per_expert_batches"), str):
        out["per_expert_batches"] = json.loads(out["per_expert_batches"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("queue")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args()
    wanted = {int(c) for c in a.configs.split(",")}
    groups = defaultdict(list)
    counts = defaultdict(int)
    for line in open(a.specs):
        row = json.loads(line)
        c = config_of(row)
        counts[c] += 1
        if c in wanted:
            groups[(row["kind"], row["backend"])].append({**row, "config": c})
    assert set(counts) == set(CONFIGS), sorted(set(CONFIGS) - set(counts))
    todo = Path(a.queue) / "todo"
    todo.mkdir(parents=True, exist_ok=True)
    for d in ("claimed", "done", "failed"):
        (Path(a.queue) / d).mkdir(exist_ok=True)
    n_chunks = 0
    # Interleave kinds so concurrent lanes do not all start on one library's JIT.
    chunks = []
    for (kind, backend), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["db_time_ms"])
        k = -(-len(rows) // a.chunk)
        for i in range(k):
            chunks.append((i, kind, backend, rows[i::k]))  # strided: each chunk spans all sizes
    chunks.sort(key=lambda c: (c[0], -len(c[3])))
    for order, (i, kind, backend, rows) in enumerate(chunks):
        stem = f"{order:03d}__{kind}__{backend}__{i}"
        with open(todo / f"{stem}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(cli_spec(r["spec"])) + "\n")
        with open(todo / f"{stem}.rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_chunks += 1
    total = sum(len(v) for v in groups.values())
    print(f"configs {sorted(wanted)}: {total} rows in {n_chunks} chunks; rows per config:",
          {c: counts[c] for c in sorted(wanted)})


if __name__ == "__main__":
    main()
