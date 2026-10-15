# Kernel node check (2_kernels.md §4)

The ServingStudioSim profiler (`launcher kernel-profile run --fresh`, nothing written to any DB) re-measured on
this node the smallest, median and largest row (by `db_time_ms`) of each of the 35 configurations in
`data/kernel_specs.jsonl`. Ratio = fresh ÷ `db_time_ms`. Rows outside ±10% are marked **bold**.
Per-spec data: `kernels_nodecheck.jsonl`; how it was run: the last section.

| Kind / backend | Fixed configuration | Rows | Smallest: DB ms / ratio | Median: DB ms / ratio | Largest: DB ms / ratio | Outside ±10% |
|---|---|---|---|---|---|---|
| `batched_gemm` / `torch_mla_q_absorb_no_rope` | — | 68 | 0.003649 / **1.466** | 0.06919 / **0.889** | 1.37 / 0.989 | 2 |
| `batched_gemm` / `torch_mla_v_up_unpadded` | — | 68 | 0.003587 / **1.490** | 0.06231 / 0.957 | 1.095 / 1.091 | 1 |
| `dsa_mqa_logits_prefill` / `deepgemm_fp8` | — | 336 | 0.003228 / **1.224** | 0.02032 / 0.908 | 17.74 / **0.886** | 2 |
| `dsa_sparse_index_remap` / `vllm_triton` | — | 325 | 0.005264 / 1.034 | 0.03733 / 1.016 | 0.2734 / 0.999 | 0 |
| `dsa_sparse_mla_attention` / `flashinfer_trtllm_fp8` | — | 1324 | 0.005606 / 1.084 | 0.0182 / 1.028 | 7.285 / 0.982 | 0 |
| `dsa_topk_prefill` / `deep_select` | — | 412 | 0.001678 / 1.003 | 0.01097 / 0.993 | 4.679 / 1.000 | 0 |
| `dsa_topk_prefill` / `vllm_cuda` | — | 412 | 0.001702 / **1.134** | 0.009213 / 0.991 | 6.922 / 0.999 | 1 |
| `gdn_causal_conv_prefill` / `dao_channellast` | — | 144 | 0.003779 / 1.001 | 0.2139 / 0.995 | 6.886 / 1.002 | 0 |
| `gemm_fp32_output` / `torch_cublas` | n 288, k 4096, input_dtype bf16 | 68 | 0.006096 / **1.293** | 0.01196 / 1.039 | 0.1387 / 0.954 | 1 |
| `gemm_fp32_output` / `torch_cublas` | n 32, k 4096, input_dtype fp32 | 68 | 0.005146 / 1.057 | 0.05305 / 0.996 | 0.4027 / 0.998 | 0 |
| `kda_chunk_prefill` / `flashinfer_cute_persistent` | — | 512 | 0.02302 / 1.000 | 0.3737 / 1.000 | 3.176 / 0.999 | 0 |
| `mhc_fused_post_pre_rms_norm` / `deepgemm_mega_nonshifted` | — | 88 | 0.01129 / 1.089 | 0.06306 / 1.000 | 1.387 / 1.004 | 0 |
| `mhc_fused_post_pre_rms_norm` / `vllm_tilelang` | — | 69 | 0.007995 / **1.215** | 0.09423 / 0.986 | 1.572 / 0.994 | 1 |
| `mhc_pre_rms_norm` / `vllm_tilelang` | — | 68 | 0.00829 / **1.113** | 0.05763 / 0.993 | 0.8979 / 0.990 | 1 |
| `mla_cache_append` / `vllm_cuda` | — | 70 | 0.001594 / **1.730** | 0.008189 / **1.367** | 0.1696 / 0.997 | 2 |
| `nvfp4_fused_moe` / `flashinfer_trtllm_sm100` | — | 73 | 0.02977 / 0.987 | 0.8017 / 0.999 | 6.464 / 1.069 | 0 |
| `nvfp4_quant` / `vllm_cuda` | hidden_size 12288, scale_format swizzled_e4m3 | 68 | 0.001976 / **1.222** | 0.01832 / 0.972 | 0.318 / 0.985 | 1 |
| `nvfp4_quant` / `vllm_cuda` | hidden_size 4096, scale_format linear_e4m3 | 68 | 0.001552 / **1.374** | 0.007247 / 1.076 | 0.1136 / 0.993 | 1 |
| `nvfp4_quant` / `vllm_cuda` | hidden_size 4096, scale_format swizzled_e4m3 | 68 | 0.001699 / **1.313** | 0.007493 / 1.079 | 0.1154 / 0.995 | 1 |
| `q_kv_rms_norm` / `vllm_triton` | — | 68 | 0.001633 / **1.326** | 0.01071 / 1.046 | 0.1719 / 0.995 | 1 |
| `rms_norm` / `vllm_cuda` | — | 63 | 0.002622 / **1.202** | 0.01694 / 0.908 | 0.2284 / 0.961 | 1 |
| `single_gemm` / `flashinfer_cutedsl` | n 24576, k 4096, dtype nvfp4_e2m1 | 68 | 0.01158 / 1.042 | 0.1419 / 0.945 | 3.507 / 0.981 | 0 |
| `single_gemm` / `flashinfer_cutedsl` | n 4096, k 12288, dtype nvfp4_e2m1 | 68 | 0.008729 / **1.102** | 0.06667 / 1.000 | 1.719 / 0.990 | 1 |
| `single_gemm` / `torch_linear_vllm` | n 128, k 4096, dtype bf16 | 68 | 0.004581 / **1.254** | 0.008487 / **1.104** | 0.09231 / 1.024 | 2 |
| `single_gemm` / `torch_linear_vllm` | n 154880, k 4096, dtype bf16 | 68 | 0.1994 / 0.992 | 3.289 / 0.942 | 57.04 / 0.986 | 0 |
| `single_gemm` / `torch_linear_vllm` | n 160, k 4096, dtype bf16 | 68 | 0.004927 / **1.195** | 0.008903 / **1.197** | 0.1009 / 0.949 | 2 |
| `single_gemm` / `torch_linear_vllm` | n 16384, k 1536, dtype bf16 | 68 | 0.01002 / **1.111** | 0.1311 / 0.927 | 2.353 / 0.986 | 1 |
| `single_gemm` / `torch_linear_vllm` | n 2048, k 4096, dtype bf16 | 68 | 0.007627 / **1.102** | 0.04441 / 0.998 | 0.7652 / 1.001 | 1 |
| `single_gemm` / `torch_linear_vllm` | n 24896, k 4096, dtype bf16 | 68 | 0.03541 / 1.016 | 0.5048 / 1.037 | 9.389 / 0.968 | 0 |
| `single_gemm` / `torch_linear_vllm` | n 4096, k 1536, dtype bf16 | 68 | 0.004363 / **1.236** | 0.02911 / 0.996 | 0.5704 / 0.986 | 1 |
| `single_gemm` / `torch_linear_vllm` | n 4096, k 16384, dtype bf16 | 68 | 0.02492 / 0.988 | 0.3795 / 0.976 | 6.173 / 0.979 | 0 |
| `single_gemm` / `torch_linear_vllm` | n 4096, k 2048, dtype bf16 | 68 | 0.004994 / **1.635** | 0.04094 / 0.924 | 0.7614 / 0.978 | 1 |
| `single_gemm` / `torch_linear_vllm` | n 4096, k 4096, dtype bf16 | 68 | 0.007638 / **1.293** | 0.08587 / 0.935 | 1.499 / 0.992 | 1 |
| `single_gemm` / `torch_linear_vllm` | n 4096, k 8192, dtype bf16 | 68 | 0.01492 / 1.039 | 0.1856 / 0.936 | 3.196 / 0.969 | 0 |
| `single_gemm` / `torch_linear_vllm` | n 8192, k 128, dtype bf16 | 68 | 0.002415 / **1.256** | 0.0147 / 0.997 | 0.2381 / 0.952 | 1 |

105 of 105 specs measured; ratio median 0.999, min 0.886, max 1.730; 27 outside ±10%.

Outside ±10% by rank: smallest 22 (fresh − DB +0.23 to +3.17 µs); median 4 (fresh − DB -7.70 to +3.01 µs); largest 1 (fresh − DB -2016.84 to -2016.84 µs).

## Outside ±10%

- `batched_gemm` / `torch_mla_q_absorb_no_rope` (smallest): DB 0.003649 ms, fresh 0.005351 ms, ratio 1.466. Spec `{"num_batches":64,"m":1,"n":512,"k":256,"dtype":"bf16"}`
- `batched_gemm` / `torch_mla_q_absorb_no_rope` (median): DB 0.06919 ms, fresh 0.06149 ms, ratio 0.889. Spec `{"num_batches":64,"m":3456,"n":512,"k":256,"dtype":"bf16"}`
- `batched_gemm` / `torch_mla_v_up_unpadded` (smallest): DB 0.003587 ms, fresh 0.005346 ms, ratio 1.490. Spec `{"num_batches":64,"m":8,"n":256,"k":512,"dtype":"bf16"}`
- `dsa_mqa_logits_prefill` / `deepgemm_fp8` (smallest): DB 0.003228 ms, fresh 0.003949 ms, ratio 1.224. Spec `{"num_queries":3,"num_keys":8,"num_sequences":1,"num_heads":32,"head_dim":128,"q_dtype":"fp8_e4m3","k_dtype":"fp8_e4m3","k_scale_dtype":"fp32","weight_dtype":"fp32","output_dtype":"fp32","span_mode":"single_causal_tail","clean_logits":"0"}`
- `dsa_mqa_logits_prefill` / `deepgemm_fp8` (largest): DB 17.74 ms, fresh 15.73 ms, ratio 0.886. Spec `{"num_queries":4096,"num_keys":1048576,"num_sequences":1,"num_heads":32,"head_dim":128,"q_dtype":"fp8_e4m3","k_dtype":"fp8_e4m3","k_scale_dtype":"fp32","weight_dtype":"fp32","output_dtype":"fp32","span_mode":"single_causal_tail","clean_logits":"0"}`
- `dsa_topk_prefill` / `vllm_cuda` (smallest): DB 0.001702 ms, fresh 0.001931 ms, ratio 1.134. Spec `{"num_queries":2,"num_keys":255,"num_sequences":1,"top_k":512,"logits_row_stride":512,"logits_dtype":"fp32","index_dtype":"int32","span_mode":"single_causal_tail"}`
- `gemm_fp32_output` / `torch_cublas` (smallest): DB 0.006096 ms, fresh 0.00788 ms, ratio 1.293. Spec `{"m":8,"n":288,"k":4096,"input_dtype":"bf16"}`
- `mhc_fused_post_pre_rms_norm` / `vllm_tilelang` (smallest): DB 0.007995 ms, fresh 0.009716 ms, ratio 1.215. Spec `{"num_tokens":2,"hidden_size":4096,"hc_mult":4,"hidden_dtype":"bf16"}`
- `mhc_pre_rms_norm` / `vllm_tilelang` (smallest): DB 0.00829 ms, fresh 0.009229 ms, ratio 1.113. Spec `{"num_tokens":1,"hidden_size":4096,"hc_mult":4,"hidden_dtype":"bf16"}`
- `mla_cache_append` / `vllm_cuda` (smallest): DB 0.001594 ms, fresh 0.002757 ms, ratio 1.730. Spec `{"num_tokens":2,"kv_lora_rank":512,"rope_dim":0,"block_size":64,"input_dtype":"bf16","kv_dtype":"fp8_e4m3","cache_format":"plain"}`
- `mla_cache_append` / `vllm_cuda` (median): DB 0.008189 ms, fresh 0.0112 ms, ratio 1.367. Spec `{"num_tokens":3456,"kv_lora_rank":512,"rope_dim":0,"block_size":64,"input_dtype":"bf16","kv_dtype":"fp8_e4m3","cache_format":"plain"}`
- `nvfp4_quant` / `vllm_cuda` (smallest): DB 0.001976 ms, fresh 0.002414 ms, ratio 1.222. Spec `{"num_tokens":1,"hidden_size":12288,"group_size":16,"input_dtype":"bf16","scale_format":"swizzled_e4m3"}`
- `nvfp4_quant` / `vllm_cuda` (smallest): DB 0.001552 ms, fresh 0.002133 ms, ratio 1.374. Spec `{"num_tokens":1,"hidden_size":4096,"group_size":16,"input_dtype":"bf16","scale_format":"linear_e4m3"}`
- `nvfp4_quant` / `vllm_cuda` (smallest): DB 0.001699 ms, fresh 0.002231 ms, ratio 1.313. Spec `{"num_tokens":1,"hidden_size":4096,"group_size":16,"input_dtype":"bf16","scale_format":"swizzled_e4m3"}`
- `q_kv_rms_norm` / `vllm_triton` (smallest): DB 0.001633 ms, fresh 0.002165 ms, ratio 1.326. Spec `{"num_tokens":2,"q_dim":1536,"kv_dim":512,"rms_eps":1e-05,"dtype":"bf16"}`
- `rms_norm` / `vllm_cuda` (smallest): DB 0.002622 ms, fresh 0.003151 ms, ratio 1.202. Spec `{"m":32,"hidden":4096,"dtype":"bf16"}`
- `single_gemm` / `flashinfer_cutedsl` (smallest): DB 0.008729 ms, fresh 0.009619 ms, ratio 1.102. Spec `{"m":1,"n":4096,"k":12288,"dtype":"nvfp4_e2m1"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.004581 ms, fresh 0.005744 ms, ratio 1.254. Spec `{"m":2,"n":128,"k":4096,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (median): DB 0.008487 ms, fresh 0.009368 ms, ratio 1.104. Spec `{"m":2944,"n":128,"k":4096,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.004927 ms, fresh 0.005889 ms, ratio 1.195. Spec `{"m":2,"n":160,"k":4096,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (median): DB 0.008903 ms, fresh 0.01066 ms, ratio 1.197. Spec `{"m":4096,"n":160,"k":4096,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.01002 ms, fresh 0.01113 ms, ratio 1.111. Spec `{"m":2,"n":16384,"k":1536,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.007627 ms, fresh 0.008403 ms, ratio 1.102. Spec `{"m":64,"n":2048,"k":4096,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.004363 ms, fresh 0.005391 ms, ratio 1.236. Spec `{"m":16,"n":4096,"k":1536,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.004994 ms, fresh 0.008164 ms, ratio 1.635. Spec `{"m":32,"n":4096,"k":2048,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.007638 ms, fresh 0.009874 ms, ratio 1.293. Spec `{"m":8,"n":4096,"k":4096,"dtype":"bf16"}`
- `single_gemm` / `torch_linear_vllm` (smallest): DB 0.002415 ms, fresh 0.003033 ms, ratio 1.256. Spec `{"m":8,"n":8192,"k":128,"dtype":"bf16"}`

## How it was run

Slurm jobs 7020, 7074, 7079 (`bench/nodecheck/run_nodecheck.sbatch`; later jobs re-ran single calls after a
spec fix), one profiler CLI call per (kind, backend), one GPU per lane:
`uv run --no-sync python -m launcher kernel-profile run <kind> --backend <b> --specs <file> --fresh
--gpu-name "NVIDIA B200" --db <scratch copy of profile.db> --json` in `wt-pp-main` (ServingStudioSim at the
handoff commit), each backend in its registered profiler environment (most in the `vllm_env` container
`vibesim-profiler-vllm:cu130-3f667d7e`), CUPTI timing with the profiler's cold-L2 clear. The samples come
from `bench/nodecheck/sample_specs.py` (`measured/nodecheck_specs/`); DB text flags `clean_logits` and
`return_valid_counts` are passed as booleans. `uptime`, GPUs and versions per job: `nodecheck_runs/<job>/env.txt`.
