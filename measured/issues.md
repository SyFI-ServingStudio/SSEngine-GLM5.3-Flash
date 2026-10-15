# Gap log (handoff `issues.md`)

Two kinds (`1_contract.md` §6): **setup difference** (names the targets it affects) and **implementation gap**
(measured, target, suspected cause).

## Node check: 27 of 105 sampled kernel rows outside ±10% — closed

Kind: setup difference, closed (the per-stage effect is below 1%). Source: `kernels_nodecheck.md` (jobs 7020, 7074 and 7079;
ServingStudioSim profiler, `--fresh`, on this node). Across all 105 rows the median ratio is 0.999.

- **22 smallest rows, launch-bound**: the shapes have m ≤ 64 or 1–2 tokens, and the DB times are 1.5–12 µs.
  Fresh runs are 0.2 to 3.2 µs slower in absolute terms.
- **4 median rows**: `mla_cache_append` +3.0 µs (1.367), `single_gemm` n 160 +1.8 µs (1.197), n 128 +0.9 µs (1.104), and
  `batched_gemm` q_absorb −7.7 µs (0.889).
- **1 largest row**: `dsa_mqa_logits_prefill` at 4,096 queries × 1,048,576 keys is 2.0 ms faster than the DB
  (15.73 vs 17.74 ms, 0.886). Shapes this large occur only in contexts near 1M tokens.

Effect on one microbatch per stage, from `bench/nodecheck/stage_bound.py`. It applies each node-check ratio to every
kernel call of `data/layer_targets.jsonl`: each call gets the ratio of the sampled row of its configuration nearest
in time. "From misses" counts only the calls whose nearest row is outside ±10%. "Worst-ratio bound" instead applies
each configuration's worst |ratio − 1| to all of its calls. It is a loose bound, because it puts the launch-bound
ratios of 1.3–1.7 on large calls. "Uncovered" is time in kinds outside the 35 configurations, mostly elementwise
Triton glue.

| Prompt | Stage | Stage ms | Estimate ms (%) | From misses ms (%) | Worst-ratio bound ms (%) | Uncovered ms (%) | Largest estimate term |
|---|---|---|---|---|---|---|---|
| 8k | 0 | 57.09 | -0.336 (-0.59%) | -0.022 (-0.04%) | 4.511 (7.90%) | 2.659 (4.7%) | `single_gemm/torch_linear_vllm` -0.123 ms |
| 8k | 1 | 54.13 | -0.223 (-0.41%) | -0.014 (-0.03%) | 4.343 (8.02%) | 2.034 (3.8%) | `single_gemm/torch_linear_vllm` -0.124 ms |
| 8k | 2 | 55.28 | -0.334 (-0.60%) | -0.022 (-0.04%) | 4.680 (8.47%) | 2.370 (4.3%) | `single_gemm/torch_linear_vllm` -0.179 ms |
| 8k | 3 | 55.66 | -0.339 (-0.61%) | -0.022 (-0.04%) | 4.718 (8.48%) | 2.421 (4.3%) | `single_gemm/torch_linear_vllm` -0.181 ms |
| 128k | 0 | 871.48 | -5.583 (-0.64%) | -3.001 (-0.34%) | 73.667 (8.45%) | 40.326 (4.6%) | `single_gemm/torch_linear_vllm` -8.931 ms |
| 128k | 1 | 792.60 | -1.843 (-0.23%) | -2.001 (-0.25%) | 67.571 (8.53%) | 30.772 (3.9%) | `nvfp4_fused_moe/flashinfer_trtllm_sm100` +9.543 ms |
| 128k | 2 | 821.19 | -2.619 (-0.32%) | -3.001 (-0.37%) | 74.871 (9.12%) | 35.873 (4.4%) | `nvfp4_fused_moe/flashinfer_trtllm_sm100` +9.543 ms |
| 128k | 3 | 823.91 | -2.646 (-0.32%) | -3.001 (-0.36%) | 75.255 (9.13%) | 36.629 (4.4%) | `nvfp4_fused_moe/flashinfer_trtllm_sm100` +9.543 ms |

The estimated stage error is −0.2% to −0.6% at 8k and 128k. The misses contribute −0.04% at 8k and −0.25% to −0.37% at
128k, both below 1%. The 128k share of the misses comes from the 1M-key mqa row. It is the sampled row nearest in time
to a 128k layer's mqa call (8.8 ms), so this 0.886 is applied conservatively at 128k. The handoff has no layer targets
past 128k, and a stage at 1M would be the first place where it counts.
