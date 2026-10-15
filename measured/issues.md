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
- **Host launch time** (job 7091, `bench/host.py`, `measured/host/7091.json`). Plain host time of a stage-0
  forward (no profiler) is 18.2 ms at 1k and 20.8 ms at 8k. M2's 25–30 ms included its per-module record_function
  scopes. GPU time is 11.8 / 16.8 / 28 / 50 ms at 1k–8k, so host > GPU only below ~2.5k.
  - Per layer at 1k: dsa_moe 2.15 ms / 52 launches, kda_moe 1.08 ms / 25, kda_dense 1.0 ms / 19, prepare 0.95 ms;
    390 kernels per forward, ~47 µs of host per kernel.
  - CUDA runtime calls are 7–8% of it. Torch ops, including the Python inside vLLM custom ops, are 55–62%; Python
    outside ops is 30–45%. In py-spy --native the leaf frames are 37% Python, 29% torch C++, 13% CUDA/cuBLAS.
  - No single wrapper dominates. Option (A), done in M3: trim per-call wrapper Python (vLLM env cache, FlashInfer
    tactic memo, mm_fp4 skip_check, plain parameters; `pp_patch.install_host_trims`).
  - **Follow-up (B), deferred:** piecewise CUDA graphs per token bucket (512/1k/1.5k/2k/3k/4k, eager above), using
    vLLM's compile path with the DSA attention + indexer op and our KDA prefill as split points. They would capture
    MoE, mHC, the linears, norms and quant, ~50–55% of host time (est. 1k: ~9 ms).
    Trigger: M4/M5 session runs showing host-bound stage-0 microbatches, i.e. appends of < 2.5k tokens alone in a
    microbatch.
- **128k host block: launch-queue back-pressure, not a sync** (7091). A 128k forward issues 1,245 kernels. The first
  launch over 1 ms is #1223, with 1,028 launched kernels still unfinished; every later block has 1,025–1,030 pending.
  So the queue is capped at ~1,024 pending kernels. M2's 1.6 s block equals its 1.69 s lead sleep.
  - StageMetadata copies: all pinned + non_blocking. `prepare` takes the same time with a 347 ms GPU sleep queued
    ahead (945 vs 957 µs at 1k), so no copy waits for the stream.

- **M3 host results** (bench/host.py, clean timing before any profiler session).
  - Option (A) trims plus StageMetadata packing: stage-0 forward 16.6 / 16.5 / 17.0 / 17.4 ms at 1k / 2k / 4k / 8k
    (7113). The untrimmed 1k is 18.15 ms (7091).
  - Exactness, 7106: bitwise equal at 1k, 2k and a two-request plan; same kernels at every length.
  - A torch.profiler session leaves CUPTI attached, which adds about 4 µs per later launch, so time before
    profiling.
- **Forward not bitwise reproducible from 4k** (7106). The same untrimmed forward, run twice in one process,
  differs by up to 0.23 max-abs on the hidden state at 4k and 8k; it is bitwise equal at 1k and 2k. Likely cause:
  above 2,048 keys the indexer's top-k index order varies, which changes the sparse-MLA summation order. Not a bug
  in our code. Token checks at ≤ 2k are exact.
- **FlashInfer tactics persisted** (`ppimpl/data/flashinfer_tactics_b200.json`, written by bench/tune_tactics.py,
  jobs 7114 and 7127). Warm-up takes 6 s instead of ~180 s, with 0 autotuner misses over 1k–128k. Every stage and
  run uses the same tactics. `StageModel.tactics` records the file and its sha256 for run.json.
- **Cold first microbatch** (bench/cold.py; 7126 before, 7132 after).
  - Before: the first 1k forward after warm-up took 138 ms vs 28 ms.
    - 87 ms was the first layer's standalone mHC pre, which loads a DeepGEMM + TileLang kernel per split-K count
      (n_SMs // ceil(T/64) below ~5k tokens). It hits every stage.
    - A length not divisible by 16 compiled two Triton kernels (120 ms).
  - Fix: warm-up now also runs every power-of-2 length from 512, a length of 1000 and the mHC pre at every split
    count, taking 9 s in all. 1k, 2k, 4k, 8k and 1000 are now warm on first use.
  - Remaining: a length never seen before (1100, 3000, 6000) pays 12–18 ms once on stage 0, in per-M first calls
    of the GEMMs (BF16 projections, router gate, o_proj; 1.3–2.7 ms each). Open; matters only for a lone short
    request of a new length.

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

## Implementation gap: NCCL p2p sends slower than the targets' send (M3) — deferred, with trigger

- Measured (job 7128, `bench/p2p.py`, 2 B200s, NCCL 2.29.7, 32 KiB per token, p50):

  | tokens | NCCL send/recv | peer copy (copy engine) | target |
  | ---: | ---: | ---: | ---: |
  | 1k | 0.134 ms | 0.061 ms | 0.068 ms |
  | 2k | 0.187 ms | 0.104 ms | |
  | 4k | 0.31 ms | 0.19 ms | |
  | 8k | 0.543 ms | 0.361 ms | 0.482 ms |
  | 16k | 0.848 ms | 0.705 ms | |

- NCCL settings change nothing (32 P2P channels, 2 MB NVLink chunk, Simple protocol). A peer copy meets the targets.
- Effect on one-request runs: 3 sends × ~0.07 ms excess, < 0.5% of TTFT at 1k and 0.1% at 8k.
- Deferred: a copy-engine transport (CUDA IPC buffers and events, a host-side handshake per microbatch). Revisit if
  the saturated runs show sends, or receives posted early, costing stage time.

## Setup difference: one-request runs of 64k and 128k cap the KV pool (M3)

- One request is one microbatch (`4_iteration.md` §3), so a 64k or 128k prompt runs as a single forward. Its
  activations outgrow the contract's 13.6 GB reserve beside the full 61.3M-token pool. In job 7117 (full pool),
  one_64k ran once, then ran out of memory in requests 2–5 (174.6 of 178.4 GiB in use, ~37 GB above weights and KV).
  one_128k ran out of memory in warm-up.
- `bench/run.py` caps the pool at 2,097,152 tokens for these two runs. Stage times do not depend on the pool size.
- Affects: the one_64k and one_128k rows of `iteration_targets.json` only. Saturated runs use 8k microbatches and the
  full pool.

## Saturated 1k: one wrong token is batch numerics, not a race or stale state — decided (M3 token check)

Kind: numerics of the model as served, decided. In 7140 (saturated_1k, no per-microbatch sync) and 7162 (same, sync after
every microbatch), uni_1k_3 (the one_1k_3 prompt) returns 11490 where the one-request reference gives 2669. The
reference has 2669 at −0.19 and 11490 at −2.19, a margin of 2.0. The cause is that the model's output depends on the
microbatch's total token count:

- No race. `bench/overlap.py` (7164/7165) enqueues 5 microbatches back to back behind a GPU sleep. The outputs are
  bitwise equal to a synced run. So are three variants: all prepares then all runs, queued prepares with synced runs,
  and the synced repeat. Every pinned staging buffer is unchanged after its copy, and the device copy equals the host
  snapshot. The host gets at most about 1.7 forwards ahead: the CUDA launch queue (~1,024 kernels) blocks it.
- No stale state. `bench/batchflip.py` (7170) runs the target in fresh blocks, in blocks a previous request used in
  the same roles, in blocks a previous 8 × 1k batch used in mixed roles, and with every cache tensor filled with
  random values. All are bitwise equal in all 45 layers.
- No cross-request leak. The target's rows are bitwise equal in three 8 × 1k batches: 7140 mb 114 (target at
  position 0), 7162 mb 206 (position 3), and mb 114's shape with the 7 partners replaced.
- Attention and MoE do not depend on T. 7172 replaces the target rows of every attention and MoE input in the
  8 × 1k batch with the alone run's rows, and every KDA, MLA + indexer and MoE output is bitwise equal to the alone
  run (token 2669, same logprobs). 7174 runs each stage-0 MoE on the same rows at T = 1024 and T = 8192, and the
  router logits, expert sets, shared expert and output are bitwise equal.
- All the T dependence is in the mHC glue. Standalone `hc_pre`, and the post + split-K pre-norm GEMM for T > 32
  (`vllm/model_executor/kernels/mhc/tilelang_kernels.py`), split K into `compute_num_split` = 148 // ceil(T/64)
  pieces. That is 9 at T = 1024 and 1 at T = 8192, a different TF32 summation order. The ~1e-3 difference is
  amplified by NVFP4 activation quantization: relative 8e-2 after layer 0, 0.2 by the end of stage 0, 0.4–0.5 in
  stage 3. Alone gives 2669 (−0.13; 11490 −2.26). Every 8 × 1k batch gives 11490 (−0.47; 2669 −1.35).
- vLLM does the same (7173, TP4+EP, the 8 prompts scheduled as one 8,192-token step). Alone gives 2669 (−0.19).
  b114 gives 2669 (−0.17). b206 gives 11490 (−0.47; 2669 −1.22), nearly our batched numbers. vLLM runs mHC
  sequence-parallel (T/4 per rank) under the same split rule, so its result also depends on the batch and the
  request's position.

Decision (coordinator, 2026-10-15): no batch-invariant mode. Every run's token check judges only requests whose
reference top-1 margin is at least 3 nats. Below that, it reports the request as "near tie, not judged" with both
logprobs. One-request runs keep judging nearly every request. The rejected option was an mHC split count independent
of T: it gives the alone result in any batch, but mHC is slower at small T, and the TP-vs-PP difference to vLLM would
still need the margin rule.
Data: `measured/batchflip/{7170,7172}_s*.json`, `7174_s0.json`, `vllm_7173.json`, `measured/host/overlap_s1_716{4,5}.json`.
