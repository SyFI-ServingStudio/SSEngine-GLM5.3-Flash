# Gap log (handoff `issues.md`)

Two kinds (`1_contract.md` §6): **setup difference** (names the targets it affects) and **implementation gap**
(measured, target, suspected cause).

## Design difference: mHC boundary at the stage edge (M2/M3)

- The targets price the first layer of stages 1–3 with the fused `attn_mhc_post_pre` (DeepGEMM mega), while the send
  carries only the 4-stream residual (32 KiB per token). A fused post+pre on the receiver needs the previous layer's
  `x`, `post` and `comb`, which that send does not carry.
- pp-impl applies the deferred `hc_post` (TileLang) on the sender and runs a standalone `hc_pre` (TileLang) on the
  receiver. From the targets' own rows: +0.092 − 0.062 = +0.030 ms per edge at 8k, +1.307 − 0.986 = +0.32 ms at 128k
  (< 0.06% of a stage). Sending `x`/`post`/`comb` too (~40 KiB per token) would add ~25% to every send instead.

## Target over-count: router `gate_recompute` (M2/M3)

- The targets price two router GEMMs per MoE layer (`gate` + `gate_recompute`, `gemm_fp32_output` n=288). The fork at
  `892da0822f` runs the router once (commit 1768273c13, "no duplicate router GEMM"), and pp-impl keeps one.
- Effect: real is faster than the target by one router GEMM per MoE layer (42 MoE layers). Per-stage size at M2.

## Design difference: output head on the returned rows only (M2/M3)

- The targets price `final_mhc_post` and `hc_contract_mean` on all T tokens (and `final_norm` at m = T in the cost
  tree). Both are per token and only the `is_last` rows reach `lm_head`, so pp-impl selects those rows first and runs
  the final `hc_post`, the stream mean, the norm and `lm_head` on them (layer 44 defers its final post to the head).
- Effect: `output_head` is faster than the target by ~0.14 ms at 8k and ~2.4 ms at 128k (stage 3 only).

## Implementation choice: KDA metadata without vLLM's GDN builder (M2/M3)

- `Glm5NextLinearAttention._forward` is replaced (Dao conv + FlashInfer persistent KDA, configs 4/5), so the GDN
  metadata builder that only that method reads is not run. Our `KdaPrefillMeta` carries the sequences, state blocks
  and the FlashInfer work plan (built once per distinct sequence offsets on a side stream: FlashInfer's
  `recurrent_kda` wrapper calls `cu_seqlens.tolist()`, a host sync, so pp-impl calls the persistent `fwd` directly).
- Glue the targets do not price: q/k/v are made contiguous for FlashInfer (3 copies of [T, 8192] BF16) and beta
  (one small copy). Measured at M2 under position `null` in `measured/layers.jsonl`.

## Target rules the real engine cannot follow (found on the CPU replay, engine lane 5696d19; affects M4–M6)

- **Rewind rounds.** 4,933 rounds of `sessions_2ps.csv` keep only part of the previous context (`prefix_len` < previous
  prompt + 1); 1,146 more drop it. The targets count the partial ones as hits (hit = min(kept, declared)). With one
  KDA state per kept context, the real engine cannot resume inside a context and recomputes them. Hit share 100% →
  98.9%; at N=500 full memory the replay goes 118.3k → 103.6k new tok/s, TTFT p99 6.5 → 60.8 s. Resuming from the
  nearest earlier round end would recompute a median 216 tokens, but needs KDA states kept a median 28 rounds back.
  This is a model fact (the KDA state exists only where it was saved), so it is a simulator fidelity gap, not an
  implementation gap. Decision needed: model it in the simulator (and re-derive targets), or add round-end state
  checkpoints to the design (with their memory cost).
- **Session slot.** The simulator frees a session's slot at its last round's token; req-frontend (the trusted
  client) holds it through the last tool wait (mean 14 s). About −5% rounds/s at N=400 against every session target.
- Block-rounded vs token accounting: < 1% on the design runs.

## M2 results (bench/layers.py; measured/layers.md, jobs 7068-7070, 7080)

No layer type is slower than its target by more than 10% at any prompt length except the two heads at sizes where
they are tens of microseconds:
- `input_head` 1k ×1.13 (0.016 vs 0.014 ms), 8k ×1.10 (0.099 vs 0.090 ms): the embedding gather plus the 4-stream
  concatenation; immaterial to a stage (< 0.2%).
- `output_head` 1k ×1.10 (0.250 vs 0.227 ms): `lm_head` alone is 0.200 ms (= target); the rest is the four row
  selects, the final post/mean/norm on one row and argmax (fixed ~0.05 ms; the target's T-proportional final post
  and mean make it cheaper only below ~2k).

Faster than the target by more than 10% (recorded, not failures):
- KDA layers (`first_kda_dense`, `kda_dense`: ×0.77–0.90; `kda_moe` ×0.84–0.94): FlashInfer persistent KDA
  (0.7.2.dev20261008) runs `chunk_prefill` at about half the row's time (8k: 0.40 vs 0.84 ms), and the BF16 GEMMs
  (`in_proj` 0.92 vs 1.20 ms, `o_proj` 0.31 vs 0.41 ms at 8k) beat the cold-L2 rows. Version-only configs are
  re-measured by the measure lane.
- `output_head` from 4k on (row selection first: see above).
Stage sums are ×0.87–0.99 of the README one-request stage times (stage 0 measured directly matches its sum).

Fixed during M2 (each found by the per-position comparison):
- **Sparse MLA past 32,768 rows.** A 65,536-row TRTLLM sparse MLA launch returned in 0.72 ms instead of ~11.7 ms
  (no error, no work done). `pp_patch.install_sparse_mla_split` splits launches at 32,768 rows into one output, as
  the target does; `dsa_moe` 64k went 34.2 → 48.9 ms (×1.08), 128k → 104.7 ms (×1.10).
- **Short prefills.** The fork runs prompts within the top-k (≤ 2,048 pools) through dense MHA prefill (FlashAttention
  plus `kv_b_proj` up-projection), ×1.11 at 2k. `sparse_mla_force_mqa` keeps them on TRTLLM sparse MLA as priced
  (2k ×1.00; also faster at 1k).
- KDA q/k/v copies (3 × 0.3 ms at 16k): conv now writes q, k, v into contiguous buffers.
- `hc_expand`: the fork's broadcast copy ran at ~1 TB/s (×3.3); vectorized concatenation, < 2^31 elements per launch.
- vLLM IR op priorities were unset outside the engine (native RMSNorm); set as vLLM's worker does.

Open:
- **Host launch time.** Eager launching of a stage-0 forward costs 25–30 ms of host time at ≤ 8k (54–61 ms at 64k).
  With launches queued ahead the layers run back to back (≤ 0.03 ms of gaps per layer), but a stage forward whose GPU
  time is below its host time leaves gaps: 1k 13.8 ms, 2k 9.1 ms, 4k 3.2 ms; none from 8k on. The runtime hides it
  when it launches a plan ahead of its input (recv on the stream); stage 0 of a lone short request cannot. CUDA graphs
  would remove it for fixed shapes; not done.
- **128k host block.** A 128k forward blocks the host for ~1.1 s inside the forward (no allocator retry, not seen by
  torch's sync debug mode); GPU gaps stay < 2 ms because the GPU is the bottleneck there. Located in job 7080.

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

## Kernel library versions in pp-impl/.venv: rows re-measured — closed

Kind: setup difference, now closed. The engine's env differs from the environments the rows were measured in: FlashInfer
0.7.2.dev20261008 and CuTe DSL 4.8.0 against the fork's 0.7.0 / 4.7.1, and most rows came from the
`cu130-3f667d7e` container (`setup.md`). Every row of the 15 configurations selected for the re-measure (1, 5, 7, 9, 11, 12, 15, 16, 18, 20, 25, 30, 31, 32, 34)
(3,084 rows) was re-measured in `pp-impl/.venv` with the profiler's own runners. See `kernels.md` and `kernels.jsonl`.

- 154 rows fall outside ±10%. 149 of them are below 20 µs and differ by at most 2.3 µs.
- 5 GEMM rows of 21–101 µs are 10–16% faster.
- `rms_norm` is about 5% faster overall.
- Swapping these rows into `data/layer_targets.jsonl`'s calls would move a stage target by at most 0.35% at 1k,
  8k or 128k, so the targets stay as shipped.
- For per-kernel expectations in this env, use `real_time_ms`.

## Local SSD for prefix KV: free space short of 1 TB — open (M6)

Kind: setup difference, open. `1_contract.md` §1 gives the node 1 TB of local SSD (256 GB per GPU) for prefix KV.
`/raid` (md1, RAID 0 of 8 NVMe, shared with other users) had 469 GB free at 01:48 and 313 GB at 04:15 on 2026-10-15
(`setup.md`). Bandwidth is not the problem: 40 GB/s with deep queues, 103 GB/s at most. Affects `7_offloading.md`:
the SSD tier runs (`tiers`, `ssd_heavy`) cannot hold the contract's 1 TB. Before M6, free or reserve the space, or
record the smaller SSD tier each run actually had.
