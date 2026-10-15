# M1 call-path identity: profiler rows vs the vLLM fork's glm5next path

For each of the 35 configurations of `2_kernels.md` §2: the callable the ServingStudioSim profiler row timed, the
callable the fork (`third_party/vllm` @ `892da0822f`, `vllm/models/glm5next/`) runs at that position, and the row's
library versions next to ours. **Covered** means same callable, same arguments and layout where they affect the
launch, and an equivalent library version. Only uncovered configurations need re-measuring, through the profiler's
own runner.

Method: read-only code tracing; no GPU runs. Row provenance comes from `_profile_run` joined on `run_key` in the
frozen `profiling/profile.db` (`7bb224a6`), matched to each spec's `db_time_ms`. `profiler_git_hash` gives the runner
source and the `vllm_env` image: `f8bd625a4d`, `f554be2fe8` and `90553f37a1` used `vibesim-profiler-vllm:cu130`;
`82c85c3203`, `e3c5fcf4ef`, `0219dfc274`, `151a70c7ec`, `92bf581bad` and later used `cu130-3f667d7e`. Per-config
evidence with file:line is in `identity_notes/`:
- `gemm_quant_norm.md`: configs 2, 3, 6, 8–18, 22, 26–29, 31, 32, 34, 35.
- `dsa.md`: configs 19–21 and 23–25.
- `kda_mhc_moe.md`: configs 1, 4, 5, 7, 30, 33.

| Environment | Versions |
|---|---|
| row image `cu130` | vLLM 0.23.0+vibesim.0b8edfb · torch 2.11.0 · cuBLAS 13.1.0.3 · FlashInfer 0.6.12 · Triton 3.6.0 · TileLang 0.1.9 · cutlass-dsl 4.5.2 |
| row image `cu130-3f667d7e` | vLLM 0.28.1rc0+vibesim.3f667d7 · torch 2.13.0 · cuBLAS 13.1.1.3 · FlashInfer 0.6.18 · Triton 3.7.1 · TileLang 0.1.12 · cutlass-dsl 4.6.2 |
| `vllm_upstream_fork_env` (rows 7, 20) | the fork at `892da0822f` |
| **ours** (`pp-impl/.venv`, `ENV.md`) | fork 0.1.dev21920+g892da0822 · torch 2.13.0+cu130 · cuBLAS 13.1.1.3 · FlashInfer 0.7.0 · Triton 3.7.1 · TileLang 0.1.12 · cutlass-dsl 4.7.1 · DeepGEMM (vendored, 2.8.0) · DeepSelect · FlashKDA b59532f1 (`_flashkda_C`) |

## Table

Verdicts:
- **SAME**: same function, same arguments and layout.
- **ARGS**: same function, but an argument or layout the launch depends on differs.
- **DIFF**: the fork calls another function.

| # | Kind / backend · position | Profiler row calls | Fork calls | Verdict | Row version → ours | Covered |
|---|---|---|---|---|---|---|
| 1 | mhc_pre_rms_norm / vllm_tilelang · layer 0 `attn_mhc_pre` | `mhc_pre_tilelang`: DeepGEMM `tf32_hc_prenorm_gemm` + TileLang big-fuse with RMSNorm | same | SAME (only the eps scalar differs, 1e-6 vs 1e-5) | 0b8edfb, TileLang 0.1.9, DeepGEMM 891d57b4 → 892da08, TileLang 0.1.12, DeepGEMM e1f418c2. TileLang and DeepGEMM kernel sources changed. | **no** |
| 2 | single_gemm n24896 k4096 · KDA in_proj | `F.linear` | `F.linear` (`UnquantizedLinearMethod`, `linear_backend=auto`) | SAME | torch 2.13 / cuBLAS 13.1.1.3 = ours | yes |
| 3 | single_gemm n8192 k128 · KDA f_b/g_b | `F.linear`, contiguous A | `F.linear` on a column view of the in_proj output (lda 24896) | ARGS | = ours | **no** |
| 4 | gdn_causal_conv_prefill / dao_channellast · KDA short conv | Dao causal-conv1d 1.7.0 `causal_conv1d_fn`, one launch per sequence, BF16 weight | vLLM Triton `causal_conv1d_fn` (`kda.py:570-582`), one launch per varlen batch, **FP32** weight, conv states | DIFF | causal-conv1d 1.7.0 (CUDA 12.8 build) | **no** |
| 5 | kda_chunk_prefill / flashinfer_cute_persistent · KDA prefill | FlashInfer `recurrent_kda(backend="cute-dsl-persistent")` | FlashKDA `torch.ops._flashkda_C.fwd` (`kda.py:127-150, 380`; `kda_prefill_backend=auto` → flashkda on SM100) | DIFF | FlashInfer 0.7.2.dev20261008 (73b63f88) + cutlass-dsl 4.8.0. Ours is FlashInfer 0.7.0, which has no persistent KDA. | **no** |
| 6 | single_gemm n4096 k8192 · KDA o_proj | `F.linear` | same | SAME | = ours | yes |
| 7 | mhc_fused_post_pre_rms_norm / deepgemm_mega_nonshifted · layer boundaries | DeepGEMM `mega_mhc` (non-shifted, one persistent launch) | `mhc_fused_post_pre_tilelang`: TileLang post + DeepGEMM prenorm GEMM + TileLang big-fuse, 3 launches above 32 tokens. No glm5next path reaches `mega_mhc`. | DIFF | fork 892da08 = ours | **no** |
| 8 | nvfp4_quant swizzled h4096 · dense FFN gate_up input | `scaled_fp4_quant(swizzled, flashinfer-cutedsl)` | same | SAME | 0.28.1rc0 (3f667d7). csrc fp4 quant unchanged to 892da08. | yes |
| 9 | single_gemm flashinfer_cutedsl n24576 k4096 · dense gate_up | `flashinfer_scaled_fp4_mm(backend="cute-dsl")`, heuristic tactic | same | SAME | FlashInfer 0.6.18 / cutlass-dsl 4.6.2 → 0.7.0 / 4.7.1. Same tactic on all 68 m (CPU replay), new DSL compiler. | **no** |
| 10 | nvfp4_quant swizzled h12288 · dense down input | as 8 | same | SAME | as 8 | yes |
| 11 | single_gemm flashinfer_cutedsl n4096 k12288 · dense down | as 9 | same | SAME | as 9 | **no** |
| 12 | single_gemm n2048 k4096 · DSA fused_qkv_a | `F.linear` | `F.linear` (the min-latency fused kernel needs a 2112×7168 weight, so it is off) | SAME | torch 2.11 / cuBLAS 13.1.0.3 → 2.13 / 13.1.1.3 | **no** |
| 13 | q_kv_rms_norm / vllm_triton · DSA q/kv A-norm | `fused_q_kv_rmsnorm` | same | SAME | fork 3f667d7 venv, Triton 3.7.1; kernel body and launch identical at 892da08 | yes |
| 14 | single_gemm n16384 k1536 · q_b_proj | `F.linear` | same | SAME | = ours | yes |
| 15 | single_gemm n4096 k1536 · indexer wq_b | `F.linear` | same | SAME | cuBLAS 13.1.0.3 → 13.1.1.3 | **no** |
| 16 | single_gemm n160 k4096 · indexer wk_weights | `F.linear` | same | SAME | cuBLAS 13.1.0.3 → 13.1.1.3 | **no** |
| 17 | gemm_fp32_output fp32 n32 k4096 · indexer head weights | `torch.mm(x.float(), W)` at precision highest | same | SAME | torch 2.13 / cuBLAS 13.1.1.3 (old `torch_cublas_vllm_fork`) = ours | yes |
| 18 | single_gemm n128 k4096 · indexer kpool gate | `F.linear` | same | SAME | cuBLAS 13.1.0.3 → 13.1.1.3 | **no** |
| 19 | dsa_mqa_logits_prefill / deepgemm_fp8 · indexer logits | `fp8_fp4_mqa_logits`, one key per query, one sequence | same wrapper; pool-granular ends `(pos+1)//4`, packed requests (ks≠0), chunked to ≤512 MiB with a query-split fallback (`indexer.py:1210-1258`) | ARGS | DeepGEMM 2.5.0 (`sm100_fp8_mqa_logits`) → 2.8.0 (`sm100_mqa_logits`, rewritten) | **no** |
| 20 | dsa_topk_prefill / deep_select · indexer top-k | `torch.ops.deep_select.topk` | not called: kpool prefill always runs `_C.top_k_per_row_prefill` (`sparse_indexer.py:341-350`) | DIFF | fork 892da08 = ours | **no** |
| 21 | dsa_topk_prefill / vllm_cuda · indexer top-k | `_C.top_k_per_row_prefill`, k=512 | same, k=512 (pool-granular ends, nonzero starts for packed requests, real logits) | SAME | 0.28.1rc0; prefill CUDA logic unchanged | yes |
| 22 | batched_gemm / q_absorb_no_rope | `torch.bmm` into contiguous [N,T,512] | `torch.bmm` into a token-major buffer (ldc 32768) | ARGS | = ours | **no** |
| 23 | mla_cache_append / vllm_cuda | `concat_and_cache_mla` (bf16 [T,512], rope 0, fp8, block 64) | same | SAME | 0.23.0; kernel byte-identical at 892da08 | yes |
| 24 | dsa_sparse_index_remap / vllm_triton | `triton_convert_req_index_to_global_index`; 275 of 325 rows with `HAS_PREFILL_WORKSPACE=True` | same function, `HAS_PREFILL_WORKSPACE=False`, side stream | ARGS | 0.28.1rc0 → 892da08: a369a7becc changed width 2176 from 17×128 tiles to one 4096 tile and dropped the init launches | **no** |
| 25 | dsa_sparse_mla_attention / flashinfer_trtllm_fp8 | `trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=2176)` | same, one launch for all rows. `qk_nope` 512 vs 256 changes only the autotune key. | SAME | FlashInfer 0.6.18 → 0.7.0, same trtllm-gen cubin hash | yes |
| 26 | batched_gemm / v_up_unpadded | `torch.bmm`, 64 heads, unpadded | same | SAME | = ours | yes |
| 27 | single_gemm n4096 k16384 · DSA o_proj | `F.linear` | same | SAME | = ours | yes |
| 28 | gemm_fp32_output bf16 n288 k4096 · router gate ×2 | `torch.mm(x, W.T, out_dtype=fp32)` | same for m>16; CuTe-DSL `ll_bf16_gemm` for m≤16 (`GateLinear` tier 1). Runs **once** per MoE layer: `gate_recompute` was removed in 1768273c13. | ARGS | cuBLAS 13.1.0.3 → 13.1.1.3 | **no** |
| 29 | nvfp4_quant linear h4096 · routed-expert input | `scaled_fp4_quant(linear)` | same, a separate launch before the MoE call | SAME | 0.28.1rc0; source unchanged | yes |
| 30 | nvfp4_fused_moe / flashinfer_trtllm_sm100 | `trtllm_fp4_block_scale_moe`, DeepSeekV3 routing, fp32 logits, `gemm1_clamp_limit=None` | same call; `gemm1_clamp_limit` = 10/g1_alphas per expert; `do_finalize` may be deferred (`should_defer_moe_finalize`); tactic tuned per process | ARGS | FlashInfer 0.6.18 → 0.7.0: TRTLLM-gen batched-GEMM cubins changed (fa419f4 → 09795a1) | **no** |
| 31 | single_gemm n4096 k4096 · shared gate_up | `F.linear` | same | SAME | cuBLAS 13.1.0.3 → 13.1.1.3 | **no** |
| 32 | single_gemm n4096 k2048 · shared down | `F.linear` | same | SAME | cuBLAS 13.1.0.3 → 13.1.1.3 | **no** |
| 33 | mhc_fused_post_pre_rms_norm / vllm_tilelang · final_mhc_post | `mhc_fused_post_pre_tilelang` (post + prenorm GEMM + big-fuse) | `mhc_post_tilelang` (1 launch) + `x.mean(dim=1)` (`model.py:551-554`), then the final norm (config 34) | DIFF | 0.23.0 | **no** |
| 34 | rms_norm / vllm_cuda · final norm | `_C.rms_norm` | same (torch.compile is off for this model) | SAME | 0.23.0 → 892da08: `layernorm_kernels.cu` changed (fp32 weight multiply, new stride argument). Rows start at m = 32. | **no** |
| 35 | single_gemm n154880 k4096 · lm_head | `F.linear` | same | SAME | = ours | yes |

**Covered: 14 of 35.** Configurations 2, 6, 8, 10, 13, 14, 17, 21, 23, 25, 26, 27, 29 and 35.

## Configurations to re-measure (21)

Each goes through the profiler's existing runner in our environment (the measure lane's node-check path), with the
callable and arguments the fork uses.

| Why | Configs | Re-measure with |
|---|---|---|
| Library version only | 1 (TileLang/DeepGEMM), 9, 11 (FlashInfer/cutlass-dsl), 12, 15, 16, 18, 31, 32 (cuBLAS), 34 (layernorm kernel) | the same backend in our env |
| Arguments or layout | 3 (lda 24896), 22 (token-major out), 19 (pool-granular spans, packed requests, DeepGEMM 2.8.0), 24 (`HAS_PREFILL_WORKSPACE=False`, new tiling), 28 (cuBLAS; plus the m≤16 `ll_bf16_gemm` path), 30 (clamp tensor, FlashInfer 0.7.0) | the same backend with the fork's arguments |
| Different callable | 4 → `gdn_causal_conv_prefill/vllm_triton` with FP32 weights · 5 → `kda_chunk_prefill/flashkda` (the runner's standalone `flash_kda` is 17a037d; the fork's is b59532f1) · 7 → `mhc_fused_post_pre_rms_norm/vllm_tilelang` (config 33's backend) · 33 → `mhc_post_tilelang` + mean (no kind yet) · 20 → only if pp-impl patches the prefill top-k to DeepSelect (see below) | the fork's backend |

## Consequences for the targets and the build

- **Router GEMM count.** The targets price `gate_recompute`, a second router GEMM per MoE layer. The fork at
  `892da0822f` runs the router once (1768273c13), so the targets count one router GEMM per MoE layer too many.
- **Indexer top-k per shape.** `2_kernels.md` §3 picks DeepSelect or `top_k_per_row_prefill` per shape. The fork's
  kpool prefill always calls `top_k_per_row_prefill`; the top-k backend setting affects decode only (`sparse_indexer.py:578`). To follow the
  design, `ppimpl/kernels.py` holds the per-shape choice and `pp_patch.py` routes the prefill top-k through it.
- **Short prefills.** With the longest prefill row ≤ 2,048 tokens, the fork skips indexer logits and top-k (configs
  19–21) and fills causal indices (`sparse_indexer.py:253-272`). It also runs dense MHA for prefill when the max
  context+query ≤ 2,048, which skips the q-absorb and V-up GEMMs (22, 26) and adds a `kv_b_proj` decompression that
  has no row.
- **Kpool expansion.** The fork expands 512 pools to 2,048 tokens plus up to 3 tail tokens, giving [rows, 2051], in one
  Triton launch (`expand_pools_and_append_tail`, `kpool_compress.py:843-916`) plus an int32→int64 cast and a copy. It
  is a glue op (`2_kernels.md` §6).
- **Sparse MLA row limit.** No split above 32,768 rows exists in the fork or in FlashInfer's launcher (gridDim.z =
  rows). The profiler's 32,768-row split (`7bb224a6`) is not fork behavior.
- **mHC stage edge.** The fork drops `post`/`comb` at the PP edge. See `issues.md`.
- **"-dirty" provenance.** Rows of configs 4 and 7 carry `92bf581bad…-dirty`. The runner source was reconstructed from
  the next commits (`2de7cdcc`, `9741a317`), which are byte-identical to HEAD. The exact dirty tree is unverified.
