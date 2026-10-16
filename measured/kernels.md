# Kernel rows re-measured in pp-impl/.venv (2_kernels.md §8)

Every row of the configurations below, measured by the ServingStudioSim profiler's own runner
(`launcher kernel-profile run --fresh`, cold L2 as the rows, nothing written to any DB) with the worker
executing in `pp-impl/.venv` (`bench/kernels/ppimpl_profile.py`). Ratio = real ÷ `db_time_ms`.
Per row: `kernels.jsonl` (with the CUPTI kernel names and the tactic note where the runner gives one).
Smallest / median / largest: the rows of the configuration ordered by `db_time_ms`, as DB ms → ratio.

| # | Kind / backend | Fixed configuration | Rows | Measured | Smallest | Median row | Largest | Median ratio | Min | Max | > 1.10 | < 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `mhc_pre_rms_norm` / `vllm_tilelang` | — | 68 | 68 | 0.00829 → 1.109 | 0.05763 → 0.997 | 0.8979 → 0.989 | 0.989 | 0.951 | 1.109 | 1 | 0 |
| 5 | `kda_chunk_prefill` / `flashinfer_cute_persistent` | — | 512 | 512 | 0.02302 → 0.997 | 0.3737 → 1.000 | 3.176 → 1.002 | 1.001 | 0.970 | 1.022 | 0 | 0 |
| 7 | `mhc_fused_post_pre_rms_norm` / `deepgemm_mega_nonshifted` | — | 88 | 88 | 0.01129 → 1.054 | 0.06306 → 0.996 | 1.387 → 1.001 | 1.000 | 0.972 | 1.054 | 0 | 0 |
| 9 | `single_gemm` / `flashinfer_cutedsl` | n 24576, k 4096 | 68 | 68 | 0.01158 → 1.038 | 0.1419 → 0.946 | 3.507 → 0.978 | 0.964 | 0.843 | 1.053 | 0 | 2 |
| 11 | `single_gemm` / `flashinfer_cutedsl` | n 4096, k 12288 | 68 | 68 | 0.008729 → 1.077 | 0.06667 → 1.008 | 1.719 → 1.002 | 0.995 | 0.914 | 1.107 | 2 | 0 |
| 12 | `single_gemm` / `torch_linear_vllm` | n 2048, k 4096 | 68 | 68 | 0.007627 → 1.093 | 0.04441 → 0.987 | 0.7652 → 0.985 | 0.989 | 0.877 | 1.094 | 0 | 1 |
| 15 | `single_gemm` / `torch_linear_vllm` | n 4096, k 1536 | 68 | 68 | 0.004363 → 1.225 | 0.02911 → 0.987 | 0.5704 → 0.996 | 1.009 | 0.807 | 1.237 | 5 | 2 |
| 16 | `single_gemm` / `torch_linear_vllm` | n 160, k 4096 | 68 | 68 | 0.004927 → 1.264 | 0.008903 → 1.119 | 0.1009 → 0.955 | 1.040 | 0.926 | 1.264 | 16 | 0 |
| 18 | `single_gemm` / `torch_linear_vllm` | n 128, k 4096 | 68 | 68 | 0.004581 → 1.373 | 0.008487 → 1.064 | 0.09231 → 1.013 | 1.075 | 0.873 | 1.373 | 19 | 2 |
| 20 | `dsa_topk_prefill` / `deep_select` | — | 412 | 412 | 0.001678 → 0.994 | 0.01097 → 1.011 | 4.679 → 1.001 | 1.001 | 0.856 | 1.187 | 8 | 4 |
| 25 | `dsa_sparse_mla_attention` / `flashinfer_trtllm_fp8` | — | 1324 | 1324 | 0.005606 → 1.083 | 0.0182 → 1.033 | 7.285 → 0.979 | 1.030 | 0.916 | 1.197 | 68 | 0 |
| 30 | `nvfp4_fused_moe` / `flashinfer_trtllm_sm100` | — | 73 | 73 | 0.02977 → 0.988 | 0.8017 → 1.038 | 6.464 → 0.997 | 1.003 | 0.953 | 1.057 | 0 | 0 |
| 31 | `single_gemm` / `torch_linear_vllm` | n 4096, k 4096 | 68 | 68 | 0.007638 → 1.301 | 0.08587 → 0.943 | 1.499 → 1.047 | 0.973 | 0.907 | 1.301 | 5 | 0 |
| 32 | `single_gemm` / `torch_linear_vllm` | n 4096, k 2048 | 68 | 68 | 0.004994 → 1.240 | 0.04094 → 0.905 | 0.7614 → 0.981 | 0.982 | 0.738 | 1.240 | 5 | 2 |
| 34 | `rms_norm` / `vllm_cuda` | — | 63 | 63 | 0.002622 → 1.201 | 0.01694 → 0.909 | 0.2284 → 0.961 | 0.948 | 0.897 | 1.285 | 10 | 2 |

## Rows above 1.10

**1. `mhc_pre_rms_norm` / `vllm_tilelang`**: 1 rows. Largest ratios:
- 1.109: DB 0.00829 ms, real 0.009192 ms, `{"num_tokens": 1, "hidden_size": 4096, "hc_mult": 4, "hidden_dtype": "bf16"}`

**11. `single_gemm` / `flashinfer_cutedsl`**: 2 rows. Largest ratios:
- 1.107: DB 0.009347 ms, real 0.01035 ms, `{"m": 32, "n": 4096, "k": 12288, "dtype": "nvfp4_e2m1"}`
- 1.100: DB 0.008885 ms, real 0.009775 ms, `{"m": 16, "n": 4096, "k": 12288, "dtype": "nvfp4_e2m1"}`

**15. `single_gemm` / `torch_linear_vllm`**: 5 rows. Largest ratios:
- 1.237: DB 0.00439 ms, real 0.00543 ms, `{"m": 32, "n": 4096, "k": 1536, "dtype": "bf16"}`
- 1.225: DB 0.004363 ms, real 0.005346 ms, `{"m": 16, "n": 4096, "k": 1536, "dtype": "bf16"}`
- 1.223: DB 0.004544 ms, real 0.005555 ms, `{"m": 64, "n": 4096, "k": 1536, "dtype": "bf16"}`
- 1.202: DB 0.004822 ms, real 0.005797 ms, `{"m": 128, "n": 4096, "k": 1536, "dtype": "bf16"}`
- 1.113: DB 0.005594 ms, real 0.006229 ms, `{"m": 4, "n": 4096, "k": 1536, "dtype": "bf16"}`

**16. `single_gemm` / `torch_linear_vllm`**: 16 rows. Largest ratios:
- 1.264: DB 0.004927 ms, real 0.006226 ms, `{"m": 2, "n": 160, "k": 4096, "dtype": "bf16"}`
- 1.191: DB 0.008601 ms, real 0.01025 ms, `{"m": 3584, "n": 160, "k": 4096, "dtype": "bf16"}`
- 1.157: DB 0.008409 ms, real 0.009733 ms, `{"m": 3072, "n": 160, "k": 4096, "dtype": "bf16"}`
- 1.147: DB 0.004961 ms, real 0.005688 ms, `{"m": 8, "n": 160, "k": 4096, "dtype": "bf16"}`
- 1.143: DB 0.004955 ms, real 0.005661 ms, `{"m": 1, "n": 160, "k": 4096, "dtype": "bf16"}`
- 1.132: DB 0.008488 ms, real 0.009612 ms, `{"m": 3456, "n": 160, "k": 4096, "dtype": "bf16"}`
- 1.132: DB 0.008618 ms, real 0.009759 ms, `{"m": 3712, "n": 160, "k": 4096, "dtype": "bf16"}`
- 1.132: DB 0.008816 ms, real 0.009978 ms, `{"m": 3968, "n": 160, "k": 4096, "dtype": "bf16"}`

**18. `single_gemm` / `torch_linear_vllm`**: 19 rows. Largest ratios:
- 1.373: DB 0.004581 ms, real 0.006292 ms, `{"m": 2, "n": 128, "k": 4096, "dtype": "bf16"}`
- 1.212: DB 0.008207 ms, real 0.009949 ms, `{"m": 3584, "n": 128, "k": 4096, "dtype": "bf16"}`
- 1.154: DB 0.004955 ms, real 0.005717 ms, `{"m": 8, "n": 128, "k": 4096, "dtype": "bf16"}`
- 1.136: DB 0.008061 ms, real 0.009158 ms, `{"m": 3200, "n": 128, "k": 4096, "dtype": "bf16"}`
- 1.135: DB 0.005117 ms, real 0.005808 ms, `{"m": 16, "n": 128, "k": 4096, "dtype": "bf16"}`
- 1.135: DB 0.008153 ms, real 0.009253 ms, `{"m": 3456, "n": 128, "k": 4096, "dtype": "bf16"}`
- 1.125: DB 0.008607 ms, real 0.009683 ms, `{"m": 4096, "n": 128, "k": 4096, "dtype": "bf16"}`
- 1.119: DB 0.005033 ms, real 0.005633 ms, `{"m": 1, "n": 128, "k": 4096, "dtype": "bf16"}`

**20. `dsa_topk_prefill` / `deep_select`**: 8 rows. Largest ratios:
- 1.187: DB 0.001707 ms, real 0.002026 ms, `{"num_queries": 1, "num_keys": 8, "num_sequences": 1, "top_k": 512, "logits_row_stride": 512, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`
- 1.183: DB 0.001712 ms, real 0.002025 ms, `{"num_queries": 1, "num_keys": 128, "num_sequences": 1, "top_k": 512, "logits_row_stride": 512, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`
- 1.179: DB 0.001705 ms, real 0.00201 ms, `{"num_queries": 1, "num_keys": 2, "num_sequences": 1, "top_k": 512, "logits_row_stride": 512, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`
- 1.168: DB 0.001704 ms, real 0.001991 ms, `{"num_queries": 1, "num_keys": 256, "num_sequences": 1, "top_k": 512, "logits_row_stride": 512, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`
- 1.165: DB 0.001707 ms, real 0.001989 ms, `{"num_queries": 1, "num_keys": 32, "num_sequences": 1, "top_k": 512, "logits_row_stride": 512, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`
- 1.164: DB 0.001701 ms, real 0.001979 ms, `{"num_queries": 1, "num_keys": 512, "num_sequences": 1, "top_k": 512, "logits_row_stride": 768, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`
- 1.141: DB 0.005177 ms, real 0.005907 ms, `{"num_queries": 4, "num_keys": 2047, "num_sequences": 1, "top_k": 512, "logits_row_stride": 2304, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`
- 1.118: DB 0.005052 ms, real 0.005649 ms, `{"num_queries": 2, "num_keys": 2049, "num_sequences": 1, "top_k": 512, "logits_row_stride": 2560, "logits_dtype": "fp32", "index_dtype": "int32", "span_mode": "single_causal_tail"}`

**25. `dsa_sparse_mla_attention` / `flashinfer_trtllm_fp8`**: 68 rows. Largest ratios:
- 1.197: DB 0.01066 ms, real 0.01276 ms, `{"num_queries": 11, "num_cache_tokens": 524288, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "u:2176x11", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`
- 1.188: DB 0.0107 ms, real 0.01271 ms, `{"num_queries": 11, "num_cache_tokens": 262144, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "u:2176x11", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`
- 1.175: DB 0.006205 ms, real 0.007293 ms, `{"num_queries": 2, "num_cache_tokens": 2, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "u:2x2", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`
- 1.169: DB 0.01074 ms, real 0.01256 ms, `{"num_queries": 11, "num_cache_tokens": 131072, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "u:2176x11", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`
- 1.165: DB 0.006077 ms, real 0.007081 ms, `{"num_queries": 2, "num_cache_tokens": 6, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "r:5..6", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`
- 1.162: DB 0.01076 ms, real 0.01251 ms, `{"num_queries": 11, "num_cache_tokens": 1048576, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "u:2176x11", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`
- 1.161: DB 0.006317 ms, real 0.007335 ms, `{"num_queries": 2, "num_cache_tokens": 2, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "r:1..2", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`
- 1.154: DB 0.01033 ms, real 0.01192 ms, `{"num_queries": 129, "num_cache_tokens": 16, "num_heads": 64, "num_kv_heads": 1, "selected_k": 2176, "latent_dim": 512, "rope_dim": 0, "value_dim": 512, "softmax_scale": 0.0625, "q_dtype": "fp8_e4m3", "cache_dtype": "fp8_e4m3", "index_dtype": "int32", "output_dtype": "bf16", "valid_counts": "u:16x129", "index_distribution": "unique_scattered_pages", "cache_layout": "hnd_paged_mqa_fp8_latent"}`

**31. `single_gemm` / `torch_linear_vllm`**: 5 rows. Largest ratios:
- 1.301: DB 0.007638 ms, real 0.009935 ms, `{"m": 8, "n": 4096, "k": 4096, "dtype": "bf16"}`
- 1.270: DB 0.00794 ms, real 0.01008 ms, `{"m": 4, "n": 4096, "k": 4096, "dtype": "bf16"}`
- 1.221: DB 0.008411 ms, real 0.01027 ms, `{"m": 2, "n": 4096, "k": 4096, "dtype": "bf16"}`
- 1.183: DB 0.00787 ms, real 0.009308 ms, `{"m": 64, "n": 4096, "k": 4096, "dtype": "bf16"}`
- 1.137: DB 0.008541 ms, real 0.009712 ms, `{"m": 128, "n": 4096, "k": 4096, "dtype": "bf16"}`

**32. `single_gemm` / `torch_linear_vllm`**: 5 rows. Largest ratios:
- 1.240: DB 0.004994 ms, real 0.006192 ms, `{"m": 32, "n": 4096, "k": 2048, "dtype": "bf16"}`
- 1.218: DB 0.005172 ms, real 0.0063 ms, `{"m": 64, "n": 4096, "k": 2048, "dtype": "bf16"}`
- 1.209: DB 0.005007 ms, real 0.006055 ms, `{"m": 16, "n": 4096, "k": 2048, "dtype": "bf16"}`
- 1.207: DB 0.005435 ms, real 0.00656 ms, `{"m": 128, "n": 4096, "k": 2048, "dtype": "bf16"}`
- 1.112: DB 0.006663 ms, real 0.007412 ms, `{"m": 8, "n": 4096, "k": 2048, "dtype": "bf16"}`

**34. `rms_norm` / `vllm_cuda`**: 10 rows. Largest ratios:
- 1.285: DB 0.003123 ms, real 0.004011 ms, `{"m": 256, "hidden": 4096, "dtype": "bf16"}`
- 1.231: DB 0.002641 ms, real 0.003251 ms, `{"m": 64, "hidden": 4096, "dtype": "bf16"}`
- 1.201: DB 0.002622 ms, real 0.00315 ms, `{"m": 32, "hidden": 4096, "dtype": "bf16"}`
- 1.199: DB 0.003457 ms, real 0.004145 ms, `{"m": 384, "hidden": 4096, "dtype": "bf16"}`
- 1.197: DB 0.002725 ms, real 0.003262 ms, `{"m": 128, "hidden": 4096, "dtype": "bf16"}`
- 1.188: DB 0.003654 ms, real 0.004341 ms, `{"m": 512, "hidden": 4096, "dtype": "bf16"}`
- 1.157: DB 0.006634 ms, real 0.007677 ms, `{"m": 1280, "hidden": 4096, "dtype": "bf16"}`
- 1.140: DB 0.004022 ms, real 0.004585 ms, `{"m": 640, "hidden": 4096, "dtype": "bf16"}`

## Reading

All 3,084 rows were measured, and none failed. 154 rows fall outside ±10%:
- 149 of them have DB times below 20 µs and differ by −2.2 to +2.3 µs: 68 sparse-MLA rows, 56 GEMM rows, 12 top-k
  rows, 12 `rms_norm` rows and 1 mHC row. These are launch-bound sizes, where a µs is a large fraction.
- 5 rows of 21–101 µs run 10–16% *faster* than the DB: config 9 at m 2432 and 2560 (0.84, 0.85), config 15 at
  m 5888 (0.88), config 18 at m 12288 (0.87) and config 32 at m 4352 (0.90).
- Per configuration, Σ real ÷ Σ DB lies between 0.967 and 1.021, with one exception: `rms_norm` (config 34) is 0.951 (median ratio
  0.948). It is about 5% faster across the board in this env.

Effect on the M2/M3 stage targets, if these rows replaced the DB rows (`bench/nodecheck/stage_bound.py
data measured/kernels.jsonl`, nearest re-measured row in time per kernel call of `data/layer_targets.jsonl`; only
the 15 configurations, every other kernel left at its DB time):

| Prompt | Stage | Stage ms | Estimate ms (%) | Uncovered ms (%) 
|---|---|---|---|---
| 1k | 0 | 12.33 | +0.016 (+0.13%) | 3.734 (30.3%) 
| 1k | 1 | 13.02 | +0.024 (+0.18%) | 3.390 (26.0%) 
| 1k | 2 | 13.08 | +0.025 (+0.19%) | 3.447 (26.4%) 
| 1k | 3 | 13.31 | +0.025 (+0.19%) | 3.669 (27.6%) 
| 8k | 0 | 57.09 | -0.198 (-0.35%) | 25.907 (45.4%) 
| 8k | 1 | 54.13 | -0.149 (-0.28%) | 23.264 (43.0%) 
| 8k | 2 | 55.28 | -0.168 (-0.30%) | 23.708 (42.9%) 
| 8k | 3 | 55.66 | -0.170 (-0.31%) | 24.050 (43.2%) 
| 128k | 0 | 871.48 | -1.024 (-0.12%) | 422.132 (48.4%) 
| 128k | 1 | 792.60 | +0.135 (+0.02%) | 372.897 (47.0%) 
| 128k | 2 | 821.19 | -0.425 (-0.05%) | 388.189 (47.3%) 
| 128k | 3 | 823.91 | -0.443 (-0.05%) | 390.451 (47.4%) 

The change is at most 0.35% of a stage at any length. The targets need no replacement rows. For per-kernel checks of
the engine (`2_kernels.md` §8), use `real_time_ms` from `kernels.jsonl` as the expectation in this env, in particular
for `rms_norm` and the 5 faster mid-size GEMM rows above.

## How it was run

Slurm jobs on the `priority` partition (`bench/slurm/kernels_queue.sbatch`), 2 GPUs each, one lane per GPU
taking chunks of one (kind, backend) from `kernels_runs/queue` (`bench/kernels/make_queue.py`: <= 50 rows,
<= 12 for the NVFP4 MoE, strided over sizes). Each chunk is one call of
`uv run --no-sync python bench/kernels/ppimpl_profile.py run <kind> --backend <b> --specs <chunk> --fresh
--gpu-name "NVIDIA B200" --json` in `wt-pp-main`, under `bench/kernels/run_call.sh` (own session, killed
only by session on timeout). The wrapper re-registers every profiling env on `pp-impl/.venv`'s Python and
pins FlashInfer's JIT to `/usr/bin/ninja`; FlashInfer autotune tactics persist per pass in
`kernels_runs/queue/autotune`. Logs per job: `kernels_runs/logs/<job>/` (`env.txt` with `uptime`).

