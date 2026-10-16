# KV-cache audit: pp-impl PP4 vs vLLM TP4+EP (GLM-5.3-Flash NVFP4)

Branch `lane-kvaudit` (worktree `lanes/kvaudit`, from `lane-longctx` 7a59497). Code: `bench/kvaudit/`. The tables
below come from `python -m bench.kvaudit.report --header measured/kvaudit-notes.md`; this text is that header.

## Method

The same enwik9 prompt goes through both engines. After prefill, each side dumps every layer's cache in one canonical
form: the MLA latent (fp8 [T, 512], no-rope MLA), the DSA indexer kpool (fp8 K [pools, 128] plus fp32 scales), the
kpool tail ring (bf16 [2, 4, 128], valid slots < T%4), the KDA recurrent state (fp32 [64, 128, 128]; vLLM's 4 TP
shards merged), and the KDA conv state (bf16 [3, 3x8192]). Each side also records the indexer top-k for the last 8
rows and the final full-vocab logprobs. `compare.py` reports per layer and tensor the relative L2, max abs, cosine
and fraction of equal bytes, with per-token bins (16 position bins, chunk-start / resume / end means). `report.py`
flags anything at or above 3x noise (and above 1e-3).

- vLLM: `LLM(tp=4, enable_expert_parallel, kv_cache_dtype=fp8_e4m3, enforce_eager, max_num_batched_tokens=16384,
  no prefix caching)`. Block ids come from the V2 runner's add/update_requests. Dumps go through `collective_rpc`.
  All 4 TP replicas of the replicated tensors were checked equal.
- Ours: 4 stage processes (split [12,11,11,11]) with the CE pipe, eager, a fresh request per case. Before the first
  dump, our indexer gather table is checked against the builder's `block_table` (passed).
- Variants (ours): whole prompt; chunked at 8192; chunked at an odd size (3001 or 8001: unaligned chunk starts); and
  prefix hits. A prefix hit runs round 0 = P tokens and round 1 = P+A from an unaligned start (P = L+3, A = 1021;
  decode-class A = 256 and A = 3 at P = 3000 and P = 100003). Each is compared with vLLM computing the full P+A
  prompt.
- Noise baselines:
  - vLLM vs itself, with a 1000-token filler request ahead of it (`noise_vllm`).
  - Ours chunked vs ours whole (`noise_ours`; at 512k, 8001 vs 8192).
  - Tail noise: the same two at T%4 != 0 lengths (3003, 100006, 100259).
  
  A layer's noise is the running max over the layers up to it (an envelope), because one noise sample per layer is
  itself noisy.

## Findings

1. **Layer 0 KDA: a kernel difference, not batch noise, and FlashKDA's error makes up most of it.** Layer 0's
   conv state is byte-exact against vLLM. Its recurrent state differs by 0.008 / 0.0097 / 0.011 / 0.0099 at 8k /
   32k / 128k / 512k, about 3x noise. Ours vs ours is 0.003, and vLLM vs itself is exact through layer 3.
   - The difference is the kernel, by design. Ours runs FlashInfer persistent `recurrent_kda` (the design row,
     `kernels.py` configs 4/5; Dao conv with a BF16 weight). The fork runs FlashKDA (Triton conv with an FP32
     weight).
   - It is a constant offset; it does not grow with length.
   - Finding 6 measures both kernels against float64. Most of the offset is FlashKDA's error: its layer-0 state is
     0.011 from float64, against 0.0023 for ours.
   - With vLLM on its Triton prefill (finding 7), layer 0 drops to 0.0033, which is ours-vs-ours noise.
2. **Up to 32k, the caches agree within noise.** No pool, block or layer offset is systematically wrong.
   - At 8k and 32k every tensor sits within the noise envelope. The worst case is the layer-43 MLA at 0.24, against
     noise of 0.18 (vLLM) and 0.24 (ours).
   - Ours is chunk-invariant: c8192 is bitwise equal to whole at 32k and 128k.
   - Unaligned chunk starts (c3001/c8001) are at noise.
   - Prefix hits (P+3, A=1021) match our own full prefill to 0.004–0.07 (MLA), and match vLLM at the same level as
     our full prefill does.
3. **Decode-class prefix hits (A=256, A=3 at P=3000 and P=100003) are at noise.** Round 1's kpool, tail ring, MLA and
   KDA state match vLLM's full prompt as closely as our own full prefill does. For A=3, the cache equals our full
   prefill bitwise except the 3 new rows.
   - The tail ring at 3003/3256 measures 0.09–0.10, against tail noise of 0.10–0.11 (vLLM 0.113, ours 0.102).
   - One single-token outlier: at 128k, o_prefix_100003_256 layer 19, tail slot 1 (token 100257) has a relative
     error of about 1.0 on K, while slots 0 and 2 are fine.
   - Round 1's pools (25000–25063) are at noise in every layer. Single pools reach about 1.0 even in the noise pairs
     (per-pool p99.9 is about 1.0). This is read as per-token fragility, not a decode-path bug.
4. **From 128k on, a few long-memory KDA heads drift with length.** They are above noise, and the error grows with
   position.

   Trend, relative L2 of ours (o_whole; o_c8192 at 512k) vs v_full. vLLM vs itself is exactly 0 through layer 3 at
   every length.

   | layer, tensor | 8k | 32k | 128k | 512k | noise at 512k (ours) |
   |---|---|---|---|---|---|
   | L0 KDA state | 0.008 | 0.0097 | 0.011 | 0.010 | 0.003 |
   | L1 KDA state | 0.018 | 0.032 | 0.057 | 0.208 | 0.015 |
   | L2 KDA state | 0.033 | 0.047 | 0.139 | 0.301 | 0.019 |
   | L3 MLA latent | 0.069 | 0.069 | 0.123 | 0.278 | 0.061 |
   | L3 indexer K | 0.039 | 0.043 | 0.120 | 0.299 | 0.035 |
   | L3 top-k overlap, last 8 rows | 0.97 | 0.93 | 0.70 | 0.56 (0.35 at 525,312) | 0.92 |
   | layers 31-44, worst (state / MLA) | at noise | at noise | ~1.1x noise | state 0.40 (L42) vs 0.16, MLA 0.26 (L43) vs 0.21 | |

   - In a handful of heads, ours vs vLLM grows with length:

     | head | 8k | 128k | 512k |
     |---|---|---|---|
     | layer 1, head 45 | 0.03 | 0.28 | 0.90 |
     | layer 2, head 56 | 0.09 | 0.42 | 0.82 |
     | layer 5, head 35 | 0.08 | 0.30 | 0.52 |

     The median head stays at 0.02–0.09, and ours vs ours stays at noise in the same heads (at most 0.035–0.096).
   - Whole-layer states: layer 1 is 0.21 (14x noise) and layer 2 is 0.30 (16x) at 512k.
   - Downstream, the layer-3 MLA latent (a per-token function of the layer-3 input, no rope) is off by a per-token
     error that grows with position:
     - 0.068 flat for the first ~24k tokens at every length, rising to 0.18 at the end of 128k and 0.39 at the end
       of 512k.
     - Ours vs ours stays flat at 0.06, and vLLM vs itself is 0.
     - The layer-3 indexer follows: 0.04 → 0.19 at 128k and 0.30 at 512k (8.5x noise).
   - The end-of-prompt conv states (the last 3 tokens' layer inputs) stay at 8k levels: layer 1 is 0.015 at 512k vs
     0.018 at 8k. So the inputs agree, and the recurrence diverges in long-memory heads.
   - Layers 0–2 have no other position-dependent op (KDA, MoE, mHC, no rope).
5. **Indexer top-k overlap for the last 8 rows at 512k** (mean over rows, ours c8192 vs vLLM, against vLLM-self and
   ours-self noise):

   | layer | ours vs vLLM | noise (vLLM / ours) |
   |---|---|---|
   | 3 | 0.56 | 1.00 / 0.92 |
   | 7 | 0.73 | 0.93 / 0.91 |
   | 11 | 0.64 | 0.90 / 0.90 |
   | 23 | 0.51 | 0.68 / 0.66 |
   | 39 | 0.48 | 0.69 / 0.68 |
   | 43 | 0.49 | 0.82 / 0.83 |

   - For o_full_525312 vs vLLM, layer 3 is 0.35.
   - At 128k, layer 3 is 0.70 (noise 1.00 / 0.93) and the other layers are within about 0.05 of noise.
   - At 8k and 32k the overlap is at noise (0.88–0.99).
   - Despite this, the final logits agree at every length: top-1 is equal in every pair.
     - At 512k, KL is at most 6e-4 (noise up to 2e-4).
     - At 128k, KL is up to 0.11 on the 100006-token case, which is at noise there (0.13 for vLLM vs itself).
6. **Which side is right: vLLM's FlashKDA drifts; ours (FlashInfer persistent) tracks float64.**
   `bench/kvaudit/kda_probe.py` captures our FlashInfer call's exact inputs for KDA layers 0-2 (q, k, v after the
   conv, the raw gate, the raw beta) and gives each kernel the same inputs, so this compares the kernels and not
   their inputs. The reference is the model's recurrence in float64, token by token, over all 64 heads:
   - gate: `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, with lower bound -5 (`config.linear_lower_bound`,
     which is part of the model's bounded gate, not a kernel clamp; both kernels apply it identically);
   - beta: sigmoid;
   - q and k: l2-normalized; scale: D^-0.5.

   A selftest on random inputs (T = 1024) agrees on conventions: FlashInfer is 0.0043 from float64, FlashKDA 0.0058,
   and the fork's Triton chunk kernel 0.003. The GPU float64 path matches a CPU per-head loop to 4e-16.

   Output relative error against float64 (all heads; worst head in brackets), first → last of 16 position bins:

   | length (job) | layer | FlashInfer persistent (ours) | FlashKDA (vLLM, pin b59532f) | fork Triton chunk |
   |---|---|---|---|---|
   | 8k (7523) | 0 | 0.0038 flat | 0.0048 flat | 0.0038 |
   | 8k (7523) | 1 | 0.0029 flat | 0.0042 → 0.0066 | 0.0038 |
   | 8k (7523) | 2 | 0.0030 flat | 0.0037 → 0.039 | 0.0029 |
   | 128k (7524) | 0 | 0.0038 flat (max head 0.0045) | 0.0049 flat (head 10: 0.009 → 0.027) | 0.0038 |
   | 128k (7524) | 1 | 0.0029 flat (head 45: 0.004) | 0.0044 → 0.023 (head 45: 0.009 → 0.43) | 0.0038 |
   | 128k (7524) | 2 | 0.0025 flat (head 56: 0.002) | 0.022 → 0.24 (head 56: 0.03 → 0.27) | 0.0028 |
   | 512k (7539) | 0 | 0.0038 flat | 0.0049 flat (head 10: 0.013 → 0.024) | 0.0038 |
   | 512k (7539) | 1 | 0.0029 → 0.0034 (head 45: 0.003 → 0.016) | 0.0059 → 0.083 (head 45: 0.045 → 0.72) | 0.0038 |
   | 512k (7539) | 2 | 0.0028 → 0.0026 (head 56: 0.002) | 0.061 → 0.41 (head 56: 0.08 → 0.46) | 0.0029 |

   At 128k and 512k, FlashKDA and the Triton chunk kernel are called in 16,384-token pieces with the state carried
   between them, as vLLM's prefill calls them. At 128k this gives the same numbers as one whole-sequence call
   (7524). Ours runs as captured, in 16,384-token microbatches; ours is chunk-invariant bitwise.

   Final state at 128k (vs float64):

   | kernel | layer 0 | layer 1 | layer 2 |
   |---|---|---|---|
   | FlashInfer | 0.0023 | 0.0020 | 0.0017 |
   | FlashKDA | 0.011 | 0.054 (head 45: 0.26) | 0.128 (head 56: 0.38) |
   | FlashInfer, 512k | 0.0024 | 0.0048 (head 45: 0.015) | 0.0017 |
   | FlashKDA, 512k | 0.0097 | 0.209 (head 45: 0.63) | 0.226 (head 56: 0.60) |

   Ours also drifts slightly at 512k in the slowest head: layer 1 head 45 goes from 0.003 to 0.016 (state 0.015).
   That is about 45x below FlashKDA in the same head, but above the Triton chunk kernel, which stays at 0.003.

   The drifting heads are the slowest-decaying ones. Ranked by mean per-token log decay, head 45 is the slowest in
   layer 1 (-0.0048) and head 56 is the slowest in layer 2 (-0.013). These are exactly the heads the cache audit
   flagged.

   - **Layer 0 per token.** Ours is uniform along the sequence at 0.0038, while FlashKDA is 0.0049. The constant
     0.008–0.011 layer-0 state difference in the audit is FlashKDA's error (0.011 vs float64) plus ours (0.002). It
     does not grow at layer 0 because layer 0's heads forget faster (head 10, FlashKDA's worst head there, decays
     at -0.036 per token; layer 1's head 45 decays at -0.0048).
   - **Cause.** The upstream FlashKDA commit 17a037d ("[Bug][GLM-Flash-5.3] Keep the recurrent state in fp32 between
     tiles", 2026-09-26) explains it. Kernel 2 held the [K, V] state as bf16 and rounded it after every 16-token
     tile, and the error concentrates in slow-decay heads and compounds across chunked-prefill boundaries.
     - vLLM's `cmake/external_projects/flashkda.cmake` pins b59532f, the commit before that fix. The installed
       `vllm/_flashkda_C.abi3.so` was built 2026-10-11 from that pin.
     - Even after the fix, FlashKDA narrows the state to bf16 at each chunk boundary (vLLM chunks at 16,384 tokens):
       a 0.0016 floor per boundary, per the commit message.
     - Ours keeps the state in FP32 throughout ("exact FP32 state handoffs" between persistent CTAs), so no change
       is needed on our side.
   - **Fix options, all on the vLLM (reference) side:**
     - (a) `--kda-prefill-backend triton`: a config switch, slower than FlashKDA. It is accurate here (0.003 vs
       float64), and with it the audit is at noise at 512k (finding 7). Keep `max_num_batched_tokens` under 262,144:
       its l2norm faults above that (finding 8).
     - (b) Rebuild vLLM's FlashKDA at ≥ 17a037d: an fp32 state within a chunk, still bf16 at chunk boundaries.
     - (c) Leave it, and treat ours-vs-vLLM KDA differences past ~24k tokens as vLLM error. Then any alignment that
       uses vLLM as ground truth at long context inherits it.
7. **Confirmation: with vLLM on its Triton KDA prefill, the long-context difference disappears.** Job 7530 reruns
   vLLM at 512k with `--kda-prefill-backend triton`; the log confirms the resolved backend is `["triton"]`. Our
   7451 dumps are compared against it unchanged (`measured/kvaudit/524288-triton/compare.json`).

   | ours c8192 vs ... at 512k | L0 state | L1 state (worst head) | L2 state (worst head) | L3 MLA (bins first → last) | L3 indexer | L3 top-k | mean top-k | layers 31-44 state / MLA |
   |---|---|---|---|---|---|---|---|---|
   | vLLM, FlashKDA (default) | 0.0099 | 0.208 (h45 0.90) | 0.301 (h56 0.82) | 0.278 (0.069 → 0.393) | 0.299 | 0.56 | 0.59 | 0.396 / 0.261 |
   | vLLM, Triton prefill | 0.0033 | 0.016 (h7 0.035) | 0.021 (h10 0.055) | 0.062 (0.067 → 0.059) | 0.036 | 0.92 | 0.82 | 0.153 / 0.216 |
   | noise: vLLM vs itself (Triton) | 0 | 0 | 0 | 0 | 0 | 1.00 | 0.84 | 0.144 / 0.209 |
   | noise: ours c8001 vs c8192 | 0.003 | 0.015 | 0.019 | 0.061 | 0.035 | 0.92 | 0.83 | 0.149 / 0.213 |

   Against Triton-prefill vLLM, ours is at noise at every layer at 512k. The position-growing error, the
   long-memory-head drift, the lower top-k overlap and the 2x deep-layer gap all come from vLLM's FlashKDA, not
   from pp-impl.
8. **Fault bisect (7525 crash).** The probe's 512k run (7525) crashed with an illegal memory access. `kda_fault.py`
   (7534) runs each kernel alone on random inputs, one subprocess each, with `CUDA_LAUNCH_BLOCKING=1`.
   - The fork's Triton `l2norm_fwd`, the first kernel of `chunk_kda_with_fused_gate`, faults once
     T x 64 x 128 > 2^31: it fails at 262,208 tokens and passes at 262,080. This looks like int32 offsets.
   - FlashInfer persistent (ours) and FlashKDA both run 524,288 tokens in one call with finite output and no zero
     rows.
   - It is not a serving path here. vLLM on SM100 uses FlashKDA and calls it with at most `max_num_batched_tokens`
     (16,384 in the audit). The fault matters only for the Triton prefill with a single call over 262,143 tokens.
   - The probe now calls FlashKDA and the Triton kernel in 16,384-token pieces, carrying the state, as vLLM does.

## Decisions (user, 2026-10-16)

1. **vLLM accuracy references use `--kda-prefill-backend triton`.** These are the token and KV checks at long
   context.
   - The default is now `triton` in `bench/kvaudit/vllm_side.py`, `bench/slurm/reference.sbatch` and
     `bench/chain.py vllm`.
   - `max_num_batched_tokens` must stay below 262,144 (finding 8). The default of 16,384 is fine.
   - Timing references keep FlashKDA as served.
   - Our kernels do not change.
2. **Our 512k drift is recorded, not pursued now.** Layer 1 head 45 goes from 0.003 to 0.016 against float64. That
   is about 45x better than FlashKDA, and top-1 and KL are unaffected. See `measured/issues.md`.

## Rerun

From the worktree root, with `/raid/kanzhu/ServingStudio/.env` sourced:

```
sbatch -p priority --time=00:30:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch vllm <L>   # 8192 32768 131072 524288
sbatch -p priority --time=00:30:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch ours <L>
../../.venv/bin/python -m bench.kvaudit.compare --length <L> [--delete]
../../.venv/bin/python -m bench.kvaudit.report --header measured/kvaudit-notes.md
# KDA kernels against float64 (1 GPU), the fault bisect, and vLLM on its Triton prefill:
sbatch -p priority --time=00:25:00 --gres=gpu:1 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch probe 524288
sbatch -p priority --time=00:20:00 --gres=gpu:1 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch fault 4096 262080 262208 524288
sbatch -p priority --time=00:10:00 --gres=gpu:4 -o measured/logs/%j.out bench/slurm/kvaudit.sbatch vllm 524288 \
    --out tmp/kvaudit/524288-triton --only v_full_524288 v_filler_524288 --kda-prefill-backend triton
../../.venv/bin/python -m bench.kvaudit.compare --length 524288 --dumps tmp/kvaudit/524288-triton --tag=-triton
```

Before the `-triton` compare, symlink our 512k case directories into `tmp/kvaudit/524288-triton`.

Jobs:

| purpose | jobs |
|---|---|
| vLLM | 7444, 7446, 7448, 7450, 7468 |
| ours | 7445, 7447, 7449, 7451, 7483 (7469 failed: pool too small) |
| probe | 7523/7537 (8k), 7524/7538 (128k), 7539 (512k) |
| probe runs that failed or were superseded | 7496, 7500, 7504, 7509, 7519, 7525 |
| fault bisect | 7534 |
| vLLM with Triton prefill | 7530 |

Probe outputs are `measured/kvaudit/kda_probe_<job>.json` and `kda_fault_7534.json`.

## 8,192 tokens

| pair (x vs reference) | kind | MLA rel L2 (layer) | MLA min exact bytes | indexer rel L2 (layer) | tail rel L2, valid slots | KDA state rel L2 (layer) | conv rel L2 (layer) | KL | top-1 | top-5 | top-k min overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v_filler_8192 vs v_full_8192 | noise_vllm | 0.184 (43) | 0.36 | 0.0871 (23) | —, 0 | 0.161 (44) | 0.198 (44) | 0.00552 | = | 4/5 | 0.895 |
| o_whole_8192 vs v_full_8192 | audit | 0.242 (43) | 0.137 | 0.123 (23) | —, 0 | 0.165 (38) | 0.171 (44) | 0.00086 | = | 4/5 | 0.883 |
| o_c3001_8192 vs v_full_8192 | audit | 0.244 (43) | 0.137 | 0.127 (23) | —, 0 | 0.171 (44) | 0.19 (44) | 0.000192 | = | 4/5 | 0.895 |
| o_c3001_8192 vs o_whole_8192 | noise_ours | 0.242 (43) | 0.137 | 0.121 (23) | —, 0 | 0.164 (44) | 0.206 (44) | 0.00111 | = | 4/5 | 0.901 |
| v_filler_3003 vs v_full_3003 | noise_vllm | 0.297 (27) | 0.124 | 0.17 (23) | 0.113 (43), 3 | 0.2 (42) | 0.193 (44) | 0.0235 | = | 4/5 | 0.945 |
| o_c1501_3003 vs o_full_3003 | noise_ours | 0.297 (27) | 0.122 | 0.162 (23) | 0.102 (43), 3 | 0.2 (44) | 0.189 (44) | 0.0447 | = | 5/5 | 0.94 |
| o_full_9216 vs v_full_9216 | audit | 0.241 (43) | 0.137 | 0.115 (23) | —, 0 | 0.153 (44) | 0.152 (24) | 0.00834 | = | 5/5 | 0.903 |
| o_prefix_8195_1021 vs v_full_9216 | audit_prefix | 0.241 (43) | 0.137 | 0.116 (23) | —, 0 | 0.158 (42) | 0.178 (24) | 0.00653 | = | 5/5 | 0.854 |
| o_prefix_8195_1021 vs o_full_9216 | prefix_vs_ours_full | 0.0714 (43) | 0.905 | 0.0322 (43) | —, 0 | 0.141 (42) | 0.182 (24) | 0.00648 | = | 5/5 | 0.901 |
| o_full_3256 vs v_full_3256 | audit | 0.296 (27) | 0.125 | 0.186 (23) | —, 0 | 0.182 (42) | 0.166 (40) | 5.66e-06 | = | 4/5 | 0.901 |
| o_prefix_3000_256 vs v_full_3256 | audit_prefix | 0.294 (27) | 0.126 | 0.188 (23) | —, 0 | 0.188 (42) | 0.188 (37) | 2.6e-06 | = | 4/5 | 0.885 |
| o_prefix_3000_256 vs o_full_3256 | prefix_vs_ours_full | 0.289 (27) | 0.124 | 0.166 (23) | —, 0 | 0.18 (42) | 0.181 (37) | 3.83e-06 | = | 4/5 | 0.901 |
| o_full_3003 vs v_full_3003 | audit | 0.295 (27) | 0.124 | 0.169 (23) | 0.0949 (43), 3 | 0.198 (42) | 0.176 (41) | 0.0507 | = | 5/5 | 0.942 |
| o_prefix_3000_3 vs v_full_3003 | audit_prefix | 0.295 (27) | 0.124 | 0.169 (23) | 0.0947 (43), 3 | 0.196 (42) | 0.164 (41) | 0.0238 | = | 5/5 | 0.943 |
| o_prefix_3000_3 vs o_full_3003 | prefix_vs_ours_full | 0.00406 (23) | 0.999 | 0 (3) | 0.0615 (43), 3 | 0.0547 (41) | 0.126 (40) | 0.00728 | = | 5/5 | 0.982 |

<details><summary>o_whole_8192 vs v_full_8192: 0 above noise, 0 patterns</summary>

Per layer, `o_whole_8192` vs `v_full_8192`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0688 | 0.0667 | 0.0675 | 1 | 0.997631 | 0.385 |  |
| 7 | mla | 0.122 | 0.119 | 0.12 | 1 | 0.99261 | 0.248 |  |
| 11 | mla | 0.0955 | 0.119 | 0.12 | 0.79 | 0.995439 | 0.302 |  |
| 15 | mla | 0.119 | 0.135 | 0.135 | 0.88 | 0.99291 | 0.257 |  |
| 19 | mla | 0.186 | 0.234 | 0.234 | 0.8 | 0.982639 | 0.192 |  |
| 23 | mla | 0.232 | 0.292 | 0.29 | 0.79 | 0.973117 | 0.16 |  |
| 27 | mla | 0.228 | 0.297 | 0.297 | 0.77 | 0.974052 | 0.176 |  |
| 31 | mla | 0.146 | 0.297 | 0.297 | 0.49 | 0.989273 | 0.236 |  |
| 35 | mla | 0.154 | 0.297 | 0.297 | 0.52 | 0.988123 | 0.24 |  |
| 39 | mla | 0.207 | 0.297 | 0.297 | 0.7 | 0.978613 | 0.171 |  |
| 43 | mla | 0.242 | 0.297 | 0.297 | 0.82 | 0.970606 | 0.137 |  |
| 3 | indexer | 0.0391 | 0.0373 | 0.0385 | 1 | 0.999234 | 0.58 |  |
| 7 | indexer | 0.0467 | 0.0457 | 0.0465 | 1 | 0.998908 | 0.531 |  |
| 11 | indexer | 0.0545 | 0.0537 | 0.0542 | 1 | 0.998513 | 0.457 |  |
| 15 | indexer | 0.0619 | 0.0652 | 0.0644 | 0.95 | 0.998084 | 0.405 |  |
| 19 | indexer | 0.0998 | 0.143 | 0.136 | 0.7 | 0.995018 | 0.377 |  |
| 23 | indexer | 0.123 | 0.17 | 0.162 | 0.73 | 0.992379 | 0.345 |  |
| 27 | indexer | 0.117 | 0.17 | 0.162 | 0.69 | 0.993176 | 0.376 |  |
| 31 | indexer | 0.0764 | 0.17 | 0.162 | 0.45 | 0.997082 | 0.39 |  |
| 35 | indexer | 0.0897 | 0.17 | 0.162 | 0.53 | 0.995977 | 0.432 |  |
| 39 | indexer | 0.1 | 0.17 | 0.162 | 0.59 | 0.994984 | 0.398 |  |
| 43 | indexer | 0.108 | 0.17 | 0.162 | 0.63 | 0.994222 | 0.259 |  |
| 0 | kda_rec | 0.00799 | 0.00478 | 0.00304 | 1.7 | 0.999968 | 9.54e-06 |  |
| 1 | kda_rec | 0.0177 | 0.0187 | 0.0205 | 0.87 | 0.999843 | 0 |  |
| 2 | kda_rec | 0.0327 | 0.0284 | 0.0304 | 1.1 | 0.999477 | 0 |  |
| 4 | kda_rec | 0.0396 | 0.0401 | 0.0408 | 0.97 | 0.999214 | 1.91e-06 |  |
| 5 | kda_rec | 0.028 | 0.0401 | 0.0408 | 0.68 | 0.99961 | 9.54e-07 |  |
| 6 | kda_rec | 0.048 | 0.0497 | 0.0482 | 0.97 | 0.998849 | 0 |  |
| 8 | kda_rec | 0.0328 | 0.0497 | 0.0482 | 0.66 | 0.999518 | 0 |  |
| 9 | kda_rec | 0.0658 | 0.0732 | 0.0722 | 0.9 | 0.997837 | 0 |  |
| 10 | kda_rec | 0.0512 | 0.0732 | 0.0722 | 0.7 | 0.998735 | 9.54e-07 |  |
| 12 | kda_rec | 0.0735 | 0.0901 | 0.0927 | 0.79 | 0.9973 | 0 |  |
| 13 | kda_rec | 0.0411 | 0.0901 | 0.0927 | 0.44 | 0.999159 | 0 |  |
| 14 | kda_rec | 0.0592 | 0.0901 | 0.0927 | 0.64 | 0.99825 | 0.000185 |  |
| 16 | kda_rec | 0.053 | 0.0901 | 0.0927 | 0.57 | 0.998601 | 0.000117 |  |
| 17 | kda_rec | 0.0568 | 0.0901 | 0.0927 | 0.61 | 0.998388 | 1.91e-06 |  |
| 18 | kda_rec | 0.0812 | 0.121 | 0.123 | 0.66 | 0.996723 | 0 |  |
| 20 | kda_rec | 0.0815 | 0.121 | 0.123 | 0.66 | 0.996675 | 0 |  |
| 21 | kda_rec | 0.121 | 0.157 | 0.162 | 0.75 | 0.992786 | 0 |  |
| 22 | kda_rec | 0.0998 | 0.157 | 0.162 | 0.62 | 0.995491 | 0 |  |
| 24 | kda_rec | 0.0926 | 0.157 | 0.162 | 0.57 | 0.995919 | 0 |  |
| 25 | kda_rec | 0.0648 | 0.157 | 0.162 | 0.4 | 0.997917 | 9.54e-07 |  |
| 26 | kda_rec | 0.0981 | 0.157 | 0.162 | 0.61 | 0.995195 | 0 |  |
| 28 | kda_rec | 0.146 | 0.181 | 0.162 | 0.8 | 0.989354 | 1.91e-06 |  |
| 29 | kda_rec | 0.122 | 0.181 | 0.162 | 0.67 | 0.992513 | 0 |  |
| 30 | kda_rec | 0.0818 | 0.181 | 0.162 | 0.45 | 0.998892 | 0 |  |
| 32 | kda_rec | 0.138 | 0.181 | 0.162 | 0.76 | 0.990542 | 0 |  |
| 33 | kda_rec | 0.15 | 0.181 | 0.162 | 0.82 | 0.988898 | 9.54e-07 |  |
| 34 | kda_rec | 0.0452 | 0.181 | 0.162 | 0.25 | 0.999078 | 0 |  |
| 36 | kda_rec | 0.0728 | 0.181 | 0.162 | 0.4 | 0.998625 | 0 |  |
| 37 | kda_rec | 0.151 | 0.181 | 0.162 | 0.83 | 0.988589 | 0 |  |
| 38 | kda_rec | 0.165 | 0.192 | 0.18 | 0.86 | 0.986451 | 7.53e-05 |  |
| 40 | kda_rec | 0.159 | 0.192 | 0.18 | 0.83 | 0.987475 | 0.000122 |  |
| 41 | kda_rec | 0.151 | 0.192 | 0.18 | 0.79 | 0.988646 | 0 |  |
| 42 | kda_rec | 0.159 | 0.2 | 0.189 | 0.8 | 0.987386 | 0 |  |
| 44 | kda_rec | 0.164 | 0.2 | 0.2 | 0.82 | 0.986627 | 0 |  |
| 0 | kda_conv | 0 | 0.000437 | 0.000807 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0181 | 0.025 | 0.0329 | 0.55 | 0.999837 | 0.0918 |  |
| 2 | kda_conv | 0.0359 | 0.0638 | 0.0674 | 0.53 | 0.999357 | 0.0485 |  |
| 4 | kda_conv | 0.0353 | 0.0638 | 0.0674 | 0.52 | 0.999382 | 0.0491 |  |
| 5 | kda_conv | 0.0518 | 0.0638 | 0.0674 | 0.77 | 0.998664 | 0.0318 |  |
| 6 | kda_conv | 0.0502 | 0.0638 | 0.0674 | 0.74 | 0.998741 | 0.0317 |  |
| 8 | kda_conv | 0.0732 | 0.0789 | 0.0716 | 0.93 | 0.997319 | 0.0243 |  |
| 9 | kda_conv | 0.0711 | 0.0812 | 0.0718 | 0.88 | 0.997466 | 0.0234 |  |
| 10 | kda_conv | 0.0575 | 0.0812 | 0.0718 | 0.71 | 0.998348 | 0.0304 |  |
| 12 | kda_conv | 0.0571 | 0.0812 | 0.0718 | 0.7 | 0.998367 | 0.0304 |  |
| 13 | kda_conv | 0.0905 | 0.106 | 0.094 | 0.86 | 0.995905 | 0.0201 |  |
| 14 | kda_conv | 0.0794 | 0.106 | 0.094 | 0.75 | 0.996841 | 0.0222 |  |
| 16 | kda_conv | 0.0641 | 0.106 | 0.094 | 0.61 | 0.997945 | 0.0259 |  |
| 17 | kda_conv | 0.0733 | 0.107 | 0.106 | 0.69 | 0.997312 | 0.0228 |  |
| 18 | kda_conv | 0.0716 | 0.107 | 0.119 | 0.6 | 0.997436 | 0.0215 |  |
| 20 | kda_conv | 0.0762 | 0.107 | 0.122 | 0.63 | 0.997093 | 0.0213 |  |
| 21 | kda_conv | 0.101 | 0.126 | 0.144 | 0.7 | 0.994929 | 0.0154 |  |
| 22 | kda_conv | 0.0819 | 0.126 | 0.144 | 0.57 | 0.996676 | 0.02 |  |
| 24 | kda_conv | 0.0799 | 0.126 | 0.144 | 0.55 | 0.996935 | 0.0193 |  |
| 25 | kda_conv | 0.0515 | 0.126 | 0.144 | 0.36 | 0.9987 | 0.0299 |  |
| 26 | kda_conv | 0.0755 | 0.126 | 0.144 | 0.52 | 0.99717 | 0.0212 |  |
| 28 | kda_conv | 0.118 | 0.132 | 0.144 | 0.82 | 0.993177 | 0.0137 |  |
| 29 | kda_conv | 0.107 | 0.132 | 0.144 | 0.74 | 0.994309 | 0.0154 |  |
| 30 | kda_conv | 0.072 | 0.132 | 0.144 | 0.5 | 0.997405 | 0.0221 |  |
| 32 | kda_conv | 0.125 | 0.132 | 0.144 | 0.86 | 0.992217 | 0.0133 |  |
| 33 | kda_conv | 0.12 | 0.132 | 0.144 | 0.83 | 0.992842 | 0.0131 |  |
| 34 | kda_conv | 0.054 | 0.132 | 0.144 | 0.37 | 0.998542 | 0.0246 |  |
| 36 | kda_conv | 0.0617 | 0.132 | 0.144 | 0.43 | 0.998105 | 0.0231 |  |
| 37 | kda_conv | 0.144 | 0.141 | 0.144 | 1 | 0.98962 | 0.0109 |  |
| 38 | kda_conv | 0.162 | 0.155 | 0.151 | 1 | 0.986871 | 0.0107 |  |
| 40 | kda_conv | 0.171 | 0.175 | 0.165 | 0.98 | 0.985402 | 0.00985 |  |
| 41 | kda_conv | 0.16 | 0.178 | 0.166 | 0.9 | 0.987299 | 0.00975 |  |
| 42 | kda_conv | 0.153 | 0.178 | 0.166 | 0.86 | 0.988435 | 0.0102 |  |
| 44 | kda_conv | 0.171 | 0.198 | 0.206 | 0.83 | 0.985782 | 0.00882 |  |


</details>

<details><summary>o_c3001_8192 vs v_full_8192: 0 above noise, 0 patterns</summary>

Per layer, `o_c3001_8192` vs `v_full_8192`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.069 | 0.0667 | 0.0675 | 1 | 0.997622 | 0.384 |  |
| 7 | mla | 0.122 | 0.119 | 0.12 | 1 | 0.992572 | 0.247 |  |
| 11 | mla | 0.096 | 0.119 | 0.12 | 0.8 | 0.995391 | 0.3 |  |
| 15 | mla | 0.121 | 0.135 | 0.135 | 0.9 | 0.992648 | 0.252 |  |
| 19 | mla | 0.189 | 0.234 | 0.234 | 0.81 | 0.982066 | 0.19 |  |
| 23 | mla | 0.234 | 0.292 | 0.29 | 0.8 | 0.972679 | 0.158 |  |
| 27 | mla | 0.229 | 0.297 | 0.297 | 0.77 | 0.973693 | 0.175 |  |
| 31 | mla | 0.148 | 0.297 | 0.297 | 0.5 | 0.989119 | 0.235 |  |
| 35 | mla | 0.156 | 0.297 | 0.297 | 0.52 | 0.98787 | 0.24 |  |
| 39 | mla | 0.209 | 0.297 | 0.297 | 0.7 | 0.978102 | 0.17 |  |
| 43 | mla | 0.244 | 0.297 | 0.297 | 0.82 | 0.970241 | 0.137 |  |
| 3 | indexer | 0.0395 | 0.0373 | 0.0385 | 1 | 0.999221 | 0.578 |  |
| 7 | indexer | 0.0465 | 0.0457 | 0.0465 | 1 | 0.99892 | 0.533 |  |
| 11 | indexer | 0.055 | 0.0537 | 0.0542 | 1 | 0.998484 | 0.454 |  |
| 15 | indexer | 0.064 | 0.0652 | 0.0644 | 0.98 | 0.99795 | 0.392 |  |
| 19 | indexer | 0.105 | 0.143 | 0.136 | 0.73 | 0.994498 | 0.37 |  |
| 23 | indexer | 0.127 | 0.17 | 0.162 | 0.74 | 0.991971 | 0.341 |  |
| 27 | indexer | 0.119 | 0.17 | 0.162 | 0.7 | 0.992981 | 0.372 |  |
| 31 | indexer | 0.0778 | 0.17 | 0.162 | 0.46 | 0.996978 | 0.388 |  |
| 35 | indexer | 0.0944 | 0.17 | 0.162 | 0.55 | 0.995541 | 0.434 |  |
| 39 | indexer | 0.109 | 0.17 | 0.162 | 0.64 | 0.994077 | 0.397 |  |
| 43 | indexer | 0.107 | 0.17 | 0.162 | 0.63 | 0.99426 | 0.256 |  |
| 0 | kda_rec | 0.00798 | 0.00478 | 0.00304 | 1.7 | 0.999968 | 5.72e-06 |  |
| 1 | kda_rec | 0.018 | 0.0187 | 0.0205 | 0.88 | 0.999838 | 9.54e-07 |  |
| 2 | kda_rec | 0.033 | 0.0284 | 0.0304 | 1.1 | 0.999461 | 0 |  |
| 4 | kda_rec | 0.0408 | 0.0401 | 0.0408 | 1 | 0.999174 | 0 |  |
| 5 | kda_rec | 0.03 | 0.0401 | 0.0408 | 0.74 | 0.999549 | 0 |  |
| 6 | kda_rec | 0.0482 | 0.0497 | 0.0482 | 0.97 | 0.998837 | 0 |  |
| 8 | kda_rec | 0.0327 | 0.0497 | 0.0482 | 0.66 | 0.999505 | 0 |  |
| 9 | kda_rec | 0.0649 | 0.0732 | 0.0722 | 0.89 | 0.997889 | 9.54e-07 |  |
| 10 | kda_rec | 0.0491 | 0.0732 | 0.0722 | 0.67 | 0.99882 | 0 |  |
| 12 | kda_rec | 0.0763 | 0.0901 | 0.0927 | 0.82 | 0.997095 | 0 |  |
| 13 | kda_rec | 0.0412 | 0.0901 | 0.0927 | 0.44 | 0.999153 | 0 |  |
| 14 | kda_rec | 0.0579 | 0.0901 | 0.0927 | 0.62 | 0.998341 | 0.000187 |  |
| 16 | kda_rec | 0.0525 | 0.0901 | 0.0927 | 0.57 | 0.998633 | 0.000117 |  |
| 17 | kda_rec | 0.0555 | 0.0901 | 0.0927 | 0.6 | 0.998482 | 9.54e-07 |  |
| 18 | kda_rec | 0.0787 | 0.121 | 0.123 | 0.64 | 0.996915 | 0 |  |
| 20 | kda_rec | 0.0808 | 0.121 | 0.123 | 0.66 | 0.996728 | 0 |  |
| 21 | kda_rec | 0.121 | 0.157 | 0.162 | 0.75 | 0.99279 | 0 |  |
| 22 | kda_rec | 0.0998 | 0.157 | 0.162 | 0.62 | 0.995575 | 0 |  |
| 24 | kda_rec | 0.1 | 0.157 | 0.162 | 0.62 | 0.995201 | 0 |  |
| 25 | kda_rec | 0.0683 | 0.157 | 0.162 | 0.42 | 0.997686 | 3.81e-06 |  |
| 26 | kda_rec | 0.0997 | 0.157 | 0.162 | 0.62 | 0.995096 | 0 |  |
| 28 | kda_rec | 0.154 | 0.181 | 0.162 | 0.85 | 0.988213 | 1.91e-06 |  |
| 29 | kda_rec | 0.123 | 0.181 | 0.162 | 0.68 | 0.992455 | 0 |  |
| 30 | kda_rec | 0.081 | 0.181 | 0.162 | 0.45 | 0.998907 | 0 |  |
| 32 | kda_rec | 0.144 | 0.181 | 0.162 | 0.8 | 0.989739 | 0 |  |
| 33 | kda_rec | 0.153 | 0.181 | 0.162 | 0.84 | 0.988323 | 0 |  |
| 34 | kda_rec | 0.0444 | 0.181 | 0.162 | 0.24 | 0.999134 | 0 |  |
| 36 | kda_rec | 0.0752 | 0.181 | 0.162 | 0.41 | 0.998579 | 0 |  |
| 37 | kda_rec | 0.154 | 0.181 | 0.162 | 0.85 | 0.98828 | 9.54e-07 |  |
| 38 | kda_rec | 0.163 | 0.192 | 0.18 | 0.85 | 0.986799 | 7.53e-05 |  |
| 40 | kda_rec | 0.159 | 0.192 | 0.18 | 0.83 | 0.987442 | 0.000122 |  |
| 41 | kda_rec | 0.153 | 0.192 | 0.18 | 0.79 | 0.988392 | 9.54e-07 |  |
| 42 | kda_rec | 0.16 | 0.2 | 0.189 | 0.8 | 0.987298 | 0 |  |
| 44 | kda_rec | 0.171 | 0.2 | 0.2 | 0.86 | 0.985458 | 0 |  |
| 0 | kda_conv | 0.000409 | 0.000437 | 0.000807 | 0.51 | 1 | 0.958 |  |
| 1 | kda_conv | 0.0177 | 0.025 | 0.0329 | 0.54 | 0.999843 | 0.0954 |  |
| 2 | kda_conv | 0.0363 | 0.0638 | 0.0674 | 0.54 | 0.99934 | 0.0466 |  |
| 4 | kda_conv | 0.0374 | 0.0638 | 0.0674 | 0.55 | 0.999341 | 0.0455 |  |
| 5 | kda_conv | 0.0517 | 0.0638 | 0.0674 | 0.77 | 0.998669 | 0.0324 |  |
| 6 | kda_conv | 0.049 | 0.0638 | 0.0674 | 0.73 | 0.998801 | 0.0342 |  |
| 8 | kda_conv | 0.0704 | 0.0789 | 0.0716 | 0.89 | 0.997522 | 0.0261 |  |
| 9 | kda_conv | 0.0638 | 0.0812 | 0.0718 | 0.79 | 0.997964 | 0.0266 |  |
| 10 | kda_conv | 0.0499 | 0.0812 | 0.0718 | 0.61 | 0.998756 | 0.0328 |  |
| 12 | kda_conv | 0.0568 | 0.0812 | 0.0718 | 0.7 | 0.998387 | 0.0273 |  |
| 13 | kda_conv | 0.0926 | 0.106 | 0.094 | 0.88 | 0.995743 | 0.0178 |  |
| 14 | kda_conv | 0.08 | 0.106 | 0.094 | 0.76 | 0.996801 | 0.0205 |  |
| 16 | kda_conv | 0.0629 | 0.106 | 0.094 | 0.6 | 0.998024 | 0.0262 |  |
| 17 | kda_conv | 0.0739 | 0.107 | 0.106 | 0.69 | 0.9973 | 0.0215 |  |
| 18 | kda_conv | 0.0715 | 0.107 | 0.119 | 0.6 | 0.997465 | 0.0213 |  |
| 20 | kda_conv | 0.0885 | 0.107 | 0.122 | 0.73 | 0.996167 | 0.0183 |  |
| 21 | kda_conv | 0.0985 | 0.126 | 0.144 | 0.68 | 0.995154 | 0.0167 |  |
| 22 | kda_conv | 0.082 | 0.126 | 0.144 | 0.57 | 0.99667 | 0.0207 |  |
| 24 | kda_conv | 0.07 | 0.126 | 0.144 | 0.48 | 0.99761 | 0.0221 |  |
| 25 | kda_conv | 0.0561 | 0.126 | 0.144 | 0.39 | 0.998436 | 0.027 |  |
| 26 | kda_conv | 0.0706 | 0.126 | 0.144 | 0.49 | 0.997526 | 0.0226 |  |
| 28 | kda_conv | 0.123 | 0.132 | 0.144 | 0.85 | 0.992464 | 0.0139 |  |
| 29 | kda_conv | 0.105 | 0.132 | 0.144 | 0.73 | 0.994528 | 0.0161 |  |
| 30 | kda_conv | 0.0669 | 0.132 | 0.144 | 0.46 | 0.997766 | 0.0232 |  |
| 32 | kda_conv | 0.127 | 0.132 | 0.144 | 0.88 | 0.992023 | 0.0125 |  |
| 33 | kda_conv | 0.121 | 0.132 | 0.144 | 0.84 | 0.992742 | 0.0126 |  |
| 34 | kda_conv | 0.0571 | 0.132 | 0.144 | 0.4 | 0.998371 | 0.0247 |  |
| 36 | kda_conv | 0.0662 | 0.132 | 0.144 | 0.46 | 0.997806 | 0.0231 |  |
| 37 | kda_conv | 0.142 | 0.141 | 0.144 | 0.98 | 0.989922 | 0.0109 |  |
| 38 | kda_conv | 0.144 | 0.155 | 0.151 | 0.93 | 0.989627 | 0.012 |  |
| 40 | kda_conv | 0.163 | 0.175 | 0.165 | 0.93 | 0.98673 | 0.0094 |  |
| 41 | kda_conv | 0.161 | 0.178 | 0.166 | 0.9 | 0.987112 | 0.00941 |  |
| 42 | kda_conv | 0.155 | 0.178 | 0.166 | 0.87 | 0.988167 | 0.0104 |  |
| 44 | kda_conv | 0.19 | 0.198 | 0.206 | 0.92 | 0.982056 | 0.00825 |  |


</details>

<details><summary>o_full_9216 vs v_full_9216: 0 above noise, 0 patterns</summary>

Per layer, `o_full_9216` vs `v_full_9216`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0687 | 0.0667 | 0.0675 | 1 | 0.997641 | 0.385 |  |
| 7 | mla | 0.122 | 0.119 | 0.12 | 1 | 0.992569 | 0.247 |  |
| 11 | mla | 0.0954 | 0.119 | 0.12 | 0.79 | 0.995451 | 0.302 |  |
| 15 | mla | 0.117 | 0.135 | 0.135 | 0.87 | 0.993108 | 0.259 |  |
| 19 | mla | 0.182 | 0.234 | 0.234 | 0.78 | 0.983489 | 0.195 |  |
| 23 | mla | 0.224 | 0.292 | 0.29 | 0.77 | 0.974833 | 0.162 |  |
| 27 | mla | 0.22 | 0.297 | 0.297 | 0.74 | 0.975737 | 0.18 |  |
| 31 | mla | 0.142 | 0.297 | 0.297 | 0.48 | 0.989971 | 0.241 |  |
| 35 | mla | 0.148 | 0.297 | 0.297 | 0.5 | 0.989018 | 0.244 |  |
| 39 | mla | 0.201 | 0.297 | 0.297 | 0.68 | 0.979774 | 0.173 |  |
| 43 | mla | 0.241 | 0.297 | 0.297 | 0.81 | 0.970929 | 0.137 |  |
| 3 | indexer | 0.0395 | 0.0373 | 0.0385 | 1 | 0.99922 | 0.58 |  |
| 7 | indexer | 0.0468 | 0.0457 | 0.0465 | 1 | 0.998906 | 0.531 |  |
| 11 | indexer | 0.0545 | 0.0537 | 0.0542 | 1 | 0.998515 | 0.456 |  |
| 15 | indexer | 0.0615 | 0.0652 | 0.0644 | 0.94 | 0.99811 | 0.409 |  |
| 19 | indexer | 0.0989 | 0.143 | 0.136 | 0.69 | 0.995105 | 0.383 |  |
| 23 | indexer | 0.115 | 0.17 | 0.162 | 0.68 | 0.993357 | 0.35 |  |
| 27 | indexer | 0.106 | 0.17 | 0.162 | 0.62 | 0.994342 | 0.382 |  |
| 31 | indexer | 0.074 | 0.17 | 0.162 | 0.43 | 0.997264 | 0.395 |  |
| 35 | indexer | 0.0854 | 0.17 | 0.162 | 0.5 | 0.996352 | 0.436 |  |
| 39 | indexer | 0.097 | 0.17 | 0.162 | 0.57 | 0.995302 | 0.401 |  |
| 43 | indexer | 0.108 | 0.17 | 0.162 | 0.64 | 0.994142 | 0.255 |  |
| 0 | kda_rec | 0.00822 | 0.00478 | 0.00304 | 1.7 | 0.999966 | 5.72e-06 |  |
| 1 | kda_rec | 0.0265 | 0.0187 | 0.0205 | 1.3 | 0.999651 | 0 |  |
| 2 | kda_rec | 0.0336 | 0.0284 | 0.0304 | 1.1 | 0.999445 | 0 |  |
| 4 | kda_rec | 0.0376 | 0.0401 | 0.0408 | 0.92 | 0.999292 | 0 |  |
| 5 | kda_rec | 0.0353 | 0.0401 | 0.0408 | 0.87 | 0.999415 | 0 |  |
| 6 | kda_rec | 0.0501 | 0.0497 | 0.0482 | 1 | 0.998747 | 0 |  |
| 8 | kda_rec | 0.0269 | 0.0497 | 0.0482 | 0.54 | 0.999654 | 0 |  |
| 9 | kda_rec | 0.0654 | 0.0732 | 0.0722 | 0.89 | 0.997864 | 0 |  |
| 10 | kda_rec | 0.0559 | 0.0732 | 0.0722 | 0.76 | 0.998472 | 0 |  |
| 12 | kda_rec | 0.0634 | 0.0901 | 0.0927 | 0.68 | 0.997992 | 0 |  |
| 13 | kda_rec | 0.0387 | 0.0901 | 0.0927 | 0.42 | 0.999283 | 0 |  |
| 14 | kda_rec | 0.0693 | 0.0901 | 0.0927 | 0.75 | 0.99766 | 0.000201 |  |
| 16 | kda_rec | 0.055 | 0.0901 | 0.0927 | 0.59 | 0.998489 | 0.000122 |  |
| 17 | kda_rec | 0.0612 | 0.0901 | 0.0927 | 0.66 | 0.998135 | 0 |  |
| 18 | kda_rec | 0.0836 | 0.121 | 0.123 | 0.68 | 0.99652 | 0 |  |
| 20 | kda_rec | 0.089 | 0.121 | 0.123 | 0.72 | 0.996029 | 9.54e-07 |  |
| 21 | kda_rec | 0.135 | 0.157 | 0.162 | 0.84 | 0.990864 | 0 |  |
| 22 | kda_rec | 0.0974 | 0.157 | 0.162 | 0.6 | 0.995513 | 0 |  |
| 24 | kda_rec | 0.102 | 0.157 | 0.162 | 0.63 | 0.994977 | 9.54e-07 |  |
| 25 | kda_rec | 0.0696 | 0.157 | 0.162 | 0.43 | 0.997579 | 7.63e-06 |  |
| 26 | kda_rec | 0.116 | 0.157 | 0.162 | 0.72 | 0.993314 | 0 |  |
| 28 | kda_rec | 0.131 | 0.181 | 0.162 | 0.72 | 0.991352 | 1.91e-06 |  |
| 29 | kda_rec | 0.116 | 0.181 | 0.162 | 0.64 | 0.993307 | 0 |  |
| 30 | kda_rec | 0.076 | 0.181 | 0.162 | 0.42 | 0.998887 | 0 |  |
| 32 | kda_rec | 0.108 | 0.181 | 0.162 | 0.59 | 0.994201 | 0 |  |
| 33 | kda_rec | 0.113 | 0.181 | 0.162 | 0.62 | 0.993597 | 0 |  |
| 34 | kda_rec | 0.0554 | 0.181 | 0.162 | 0.31 | 0.998636 | 9.54e-07 |  |
| 36 | kda_rec | 0.0784 | 0.181 | 0.162 | 0.43 | 0.998312 | 0 |  |
| 37 | kda_rec | 0.116 | 0.181 | 0.162 | 0.64 | 0.993297 | 0 |  |
| 38 | kda_rec | 0.147 | 0.192 | 0.18 | 0.77 | 0.989153 | 0.000106 |  |
| 40 | kda_rec | 0.138 | 0.192 | 0.18 | 0.72 | 0.990485 | 0.000122 |  |
| 41 | kda_rec | 0.14 | 0.192 | 0.18 | 0.73 | 0.990204 | 0 |  |
| 42 | kda_rec | 0.15 | 0.2 | 0.189 | 0.75 | 0.988724 | 0 |  |
| 44 | kda_rec | 0.153 | 0.2 | 0.2 | 0.76 | 0.988371 | 0 |  |
| 0 | kda_conv | 0.000458 | 0.000437 | 0.000807 | 0.57 | 1 | 0.955 |  |
| 1 | kda_conv | 0.0501 | 0.025 | 0.0329 | 1.5 | 0.998751 | 0.0388 |  |
| 2 | kda_conv | 0.0442 | 0.0638 | 0.0674 | 0.66 | 0.999025 | 0.0385 |  |
| 4 | kda_conv | 0.0345 | 0.0638 | 0.0674 | 0.51 | 0.999406 | 0.0473 |  |
| 5 | kda_conv | 0.068 | 0.0638 | 0.0674 | 1 | 0.997688 | 0.0245 |  |
| 6 | kda_conv | 0.0616 | 0.0638 | 0.0674 | 0.91 | 0.998102 | 0.0271 |  |
| 8 | kda_conv | 0.0888 | 0.0789 | 0.0716 | 1.1 | 0.996051 | 0.0203 |  |
| 9 | kda_conv | 0.0925 | 0.0812 | 0.0718 | 1.1 | 0.995748 | 0.021 |  |
| 10 | kda_conv | 0.0713 | 0.0812 | 0.0718 | 0.88 | 0.997469 | 0.0253 |  |
| 12 | kda_conv | 0.0765 | 0.0812 | 0.0718 | 0.94 | 0.997068 | 0.021 |  |
| 13 | kda_conv | 0.0984 | 0.106 | 0.094 | 0.93 | 0.995199 | 0.0161 |  |
| 14 | kda_conv | 0.105 | 0.106 | 0.094 | 0.99 | 0.994536 | 0.0153 |  |
| 16 | kda_conv | 0.0822 | 0.106 | 0.094 | 0.78 | 0.996614 | 0.0195 |  |
| 17 | kda_conv | 0.115 | 0.107 | 0.106 | 1.1 | 0.993462 | 0.0146 |  |
| 18 | kda_conv | 0.125 | 0.107 | 0.119 | 1.1 | 0.992229 | 0.0123 |  |
| 20 | kda_conv | 0.132 | 0.107 | 0.122 | 1.1 | 0.991217 | 0.0124 |  |
| 21 | kda_conv | 0.131 | 0.126 | 0.144 | 0.91 | 0.9914 | 0.0123 |  |
| 22 | kda_conv | 0.13 | 0.126 | 0.144 | 0.9 | 0.991506 | 0.0127 |  |
| 24 | kda_conv | 0.152 | 0.126 | 0.144 | 1.1 | 0.988521 | 0.0111 |  |
| 25 | kda_conv | 0.134 | 0.126 | 0.144 | 0.93 | 0.99117 | 0.0122 |  |
| 26 | kda_conv | 0.145 | 0.126 | 0.144 | 1 | 0.989558 | 0.011 |  |
| 28 | kda_conv | 0.122 | 0.132 | 0.144 | 0.84 | 0.992568 | 0.0137 |  |
| 29 | kda_conv | 0.122 | 0.132 | 0.144 | 0.84 | 0.992746 | 0.0134 |  |
| 30 | kda_conv | 0.116 | 0.132 | 0.144 | 0.8 | 0.993348 | 0.0139 |  |
| 32 | kda_conv | 0.101 | 0.132 | 0.144 | 0.7 | 0.994897 | 0.016 |  |
| 33 | kda_conv | 0.0976 | 0.132 | 0.144 | 0.68 | 0.995298 | 0.0167 |  |
| 34 | kda_conv | 0.0725 | 0.132 | 0.144 | 0.5 | 0.997372 | 0.018 |  |
| 36 | kda_conv | 0.086 | 0.132 | 0.144 | 0.6 | 0.996298 | 0.0172 |  |
| 37 | kda_conv | 0.103 | 0.141 | 0.144 | 0.71 | 0.994713 | 0.0142 |  |
| 38 | kda_conv | 0.104 | 0.155 | 0.151 | 0.67 | 0.994559 | 0.015 |  |
| 40 | kda_conv | 0.125 | 0.175 | 0.165 | 0.72 | 0.992137 | 0.0126 |  |
| 41 | kda_conv | 0.145 | 0.178 | 0.166 | 0.82 | 0.989403 | 0.0116 |  |
| 42 | kda_conv | 0.124 | 0.178 | 0.166 | 0.69 | 0.992317 | 0.0133 |  |
| 44 | kda_conv | 0.14 | 0.198 | 0.206 | 0.68 | 0.990253 | 0.0113 |  |


</details>

<details><summary>o_prefix_8195_1021 vs v_full_9216: 0 above noise, 0 patterns</summary>

Per layer, `o_prefix_8195_1021` vs `v_full_9216`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0687 | 0.0667 | 0.0675 | 1 | 0.997641 | 0.385 |  |
| 7 | mla | 0.122 | 0.119 | 0.12 | 1 | 0.992564 | 0.247 |  |
| 11 | mla | 0.0954 | 0.119 | 0.12 | 0.79 | 0.995448 | 0.302 |  |
| 15 | mla | 0.117 | 0.135 | 0.135 | 0.87 | 0.993104 | 0.258 |  |
| 19 | mla | 0.182 | 0.234 | 0.234 | 0.78 | 0.983437 | 0.195 |  |
| 23 | mla | 0.225 | 0.292 | 0.29 | 0.77 | 0.974787 | 0.162 |  |
| 27 | mla | 0.22 | 0.297 | 0.297 | 0.74 | 0.975705 | 0.18 |  |
| 31 | mla | 0.142 | 0.297 | 0.297 | 0.48 | 0.989947 | 0.241 |  |
| 35 | mla | 0.149 | 0.297 | 0.297 | 0.5 | 0.988973 | 0.244 |  |
| 39 | mla | 0.202 | 0.297 | 0.297 | 0.68 | 0.979693 | 0.173 |  |
| 43 | mla | 0.241 | 0.297 | 0.297 | 0.81 | 0.970866 | 0.137 |  |
| 3 | indexer | 0.0395 | 0.0373 | 0.0385 | 1 | 0.999222 | 0.58 |  |
| 7 | indexer | 0.0468 | 0.0457 | 0.0465 | 1 | 0.998904 | 0.531 |  |
| 11 | indexer | 0.0545 | 0.0537 | 0.0542 | 1 | 0.998516 | 0.456 |  |
| 15 | indexer | 0.0615 | 0.0652 | 0.0644 | 0.94 | 0.998105 | 0.409 |  |
| 19 | indexer | 0.0995 | 0.143 | 0.136 | 0.69 | 0.995049 | 0.382 |  |
| 23 | indexer | 0.116 | 0.17 | 0.162 | 0.68 | 0.993276 | 0.35 |  |
| 27 | indexer | 0.107 | 0.17 | 0.162 | 0.63 | 0.994256 | 0.382 |  |
| 31 | indexer | 0.074 | 0.17 | 0.162 | 0.43 | 0.997266 | 0.397 |  |
| 35 | indexer | 0.0856 | 0.17 | 0.162 | 0.5 | 0.996332 | 0.436 |  |
| 39 | indexer | 0.0978 | 0.17 | 0.162 | 0.57 | 0.99522 | 0.402 |  |
| 43 | indexer | 0.108 | 0.17 | 0.162 | 0.64 | 0.994141 | 0.255 |  |
| 0 | kda_rec | 0.00828 | 0.00478 | 0.00304 | 1.7 | 0.999966 | 6.68e-06 |  |
| 1 | kda_rec | 0.0263 | 0.0187 | 0.0205 | 1.3 | 0.999655 | 0 |  |
| 2 | kda_rec | 0.0333 | 0.0284 | 0.0304 | 1.1 | 0.999449 | 0 |  |
| 4 | kda_rec | 0.0387 | 0.0401 | 0.0408 | 0.95 | 0.999249 | 0 |  |
| 5 | kda_rec | 0.0356 | 0.0401 | 0.0408 | 0.87 | 0.999411 | 0 |  |
| 6 | kda_rec | 0.048 | 0.0497 | 0.0482 | 0.97 | 0.998845 | 0 |  |
| 8 | kda_rec | 0.0267 | 0.0497 | 0.0482 | 0.54 | 0.999663 | 0 |  |
| 9 | kda_rec | 0.0642 | 0.0732 | 0.0722 | 0.88 | 0.997947 | 0 |  |
| 10 | kda_rec | 0.055 | 0.0732 | 0.0722 | 0.75 | 0.998566 | 0 |  |
| 12 | kda_rec | 0.0648 | 0.0901 | 0.0927 | 0.7 | 0.997904 | 0 |  |
| 13 | kda_rec | 0.0385 | 0.0901 | 0.0927 | 0.41 | 0.999277 | 9.54e-07 |  |
| 14 | kda_rec | 0.0694 | 0.0901 | 0.0927 | 0.75 | 0.997632 | 0.0002 |  |
| 16 | kda_rec | 0.0553 | 0.0901 | 0.0927 | 0.6 | 0.998473 | 0.000123 |  |
| 17 | kda_rec | 0.0614 | 0.0901 | 0.0927 | 0.66 | 0.998139 | 0 |  |
| 18 | kda_rec | 0.0861 | 0.121 | 0.123 | 0.7 | 0.996308 | 0 |  |
| 20 | kda_rec | 0.0925 | 0.121 | 0.123 | 0.75 | 0.995716 | 0 |  |
| 21 | kda_rec | 0.137 | 0.157 | 0.162 | 0.85 | 0.990667 | 0 |  |
| 22 | kda_rec | 0.106 | 0.157 | 0.162 | 0.66 | 0.99488 | 0 |  |
| 24 | kda_rec | 0.107 | 0.157 | 0.162 | 0.66 | 0.994474 | 9.54e-07 |  |
| 25 | kda_rec | 0.0721 | 0.157 | 0.162 | 0.45 | 0.997398 | 7.63e-06 |  |
| 26 | kda_rec | 0.127 | 0.157 | 0.162 | 0.78 | 0.991979 | 0 |  |
| 28 | kda_rec | 0.151 | 0.181 | 0.162 | 0.83 | 0.988562 | 1.91e-06 |  |
| 29 | kda_rec | 0.122 | 0.181 | 0.162 | 0.67 | 0.992532 | 0 |  |
| 30 | kda_rec | 0.0773 | 0.181 | 0.162 | 0.43 | 0.998742 | 0 |  |
| 32 | kda_rec | 0.122 | 0.181 | 0.162 | 0.67 | 0.992578 | 0 |  |
| 33 | kda_rec | 0.117 | 0.181 | 0.162 | 0.65 | 0.99312 | 0 |  |
| 34 | kda_rec | 0.0558 | 0.181 | 0.162 | 0.31 | 0.998597 | 0 |  |
| 36 | kda_rec | 0.079 | 0.181 | 0.162 | 0.44 | 0.998268 | 0 |  |
| 37 | kda_rec | 0.121 | 0.181 | 0.162 | 0.67 | 0.99261 | 0 |  |
| 38 | kda_rec | 0.153 | 0.192 | 0.18 | 0.8 | 0.988325 | 0.000106 |  |
| 40 | kda_rec | 0.143 | 0.192 | 0.18 | 0.74 | 0.989816 | 0.000122 |  |
| 41 | kda_rec | 0.145 | 0.192 | 0.18 | 0.76 | 0.989413 | 0 |  |
| 42 | kda_rec | 0.158 | 0.2 | 0.189 | 0.79 | 0.987555 | 9.54e-07 |  |
| 44 | kda_rec | 0.155 | 0.2 | 0.2 | 0.78 | 0.988016 | 0 |  |
| 0 | kda_conv | 0.000458 | 0.000437 | 0.000807 | 0.57 | 1 | 0.955 |  |
| 1 | kda_conv | 0.047 | 0.025 | 0.0329 | 1.4 | 0.998893 | 0.0407 |  |
| 2 | kda_conv | 0.0434 | 0.0638 | 0.0674 | 0.64 | 0.999058 | 0.0388 |  |
| 4 | kda_conv | 0.0339 | 0.0638 | 0.0674 | 0.5 | 0.999425 | 0.0464 |  |
| 5 | kda_conv | 0.0744 | 0.0638 | 0.0674 | 1.1 | 0.997233 | 0.0228 |  |
| 6 | kda_conv | 0.0672 | 0.0638 | 0.0674 | 1 | 0.997747 | 0.025 |  |
| 8 | kda_conv | 0.0876 | 0.0789 | 0.0716 | 1.1 | 0.996155 | 0.0203 |  |
| 9 | kda_conv | 0.0764 | 0.0812 | 0.0718 | 0.94 | 0.997081 | 0.0223 |  |
| 10 | kda_conv | 0.0694 | 0.0812 | 0.0718 | 0.85 | 0.997599 | 0.0248 |  |
| 12 | kda_conv | 0.0782 | 0.0812 | 0.0718 | 0.96 | 0.996942 | 0.0221 |  |
| 13 | kda_conv | 0.0952 | 0.106 | 0.094 | 0.9 | 0.995473 | 0.0164 |  |
| 14 | kda_conv | 0.1 | 0.106 | 0.094 | 0.95 | 0.99499 | 0.0176 |  |
| 16 | kda_conv | 0.0858 | 0.106 | 0.094 | 0.81 | 0.996354 | 0.0197 |  |
| 17 | kda_conv | 0.116 | 0.107 | 0.106 | 1.1 | 0.993278 | 0.0145 |  |
| 18 | kda_conv | 0.131 | 0.107 | 0.119 | 1.1 | 0.991394 | 0.012 |  |
| 20 | kda_conv | 0.137 | 0.107 | 0.122 | 1.1 | 0.990712 | 0.0126 |  |
| 21 | kda_conv | 0.147 | 0.126 | 0.144 | 1 | 0.989149 | 0.012 |  |
| 22 | kda_conv | 0.16 | 0.126 | 0.144 | 1.1 | 0.987149 | 0.0107 |  |
| 24 | kda_conv | 0.178 | 0.126 | 0.144 | 1.2 | 0.984134 | 0.00979 |  |
| 25 | kda_conv | 0.151 | 0.126 | 0.144 | 1 | 0.988799 | 0.0101 |  |
| 26 | kda_conv | 0.168 | 0.126 | 0.144 | 1.2 | 0.985888 | 0.00993 |  |
| 28 | kda_conv | 0.147 | 0.132 | 0.144 | 1 | 0.989196 | 0.0105 |  |
| 29 | kda_conv | 0.139 | 0.132 | 0.144 | 0.96 | 0.990347 | 0.0117 |  |
| 30 | kda_conv | 0.137 | 0.132 | 0.144 | 0.95 | 0.990633 | 0.0119 |  |
| 32 | kda_conv | 0.115 | 0.132 | 0.144 | 0.8 | 0.993322 | 0.0142 |  |
| 33 | kda_conv | 0.11 | 0.132 | 0.144 | 0.76 | 0.993944 | 0.0136 |  |
| 34 | kda_conv | 0.0838 | 0.132 | 0.144 | 0.58 | 0.996484 | 0.0162 |  |
| 36 | kda_conv | 0.0985 | 0.132 | 0.144 | 0.68 | 0.995137 | 0.0152 |  |
| 37 | kda_conv | 0.121 | 0.141 | 0.144 | 0.84 | 0.992598 | 0.0119 |  |
| 38 | kda_conv | 0.126 | 0.155 | 0.151 | 0.81 | 0.992121 | 0.0122 |  |
| 40 | kda_conv | 0.14 | 0.175 | 0.165 | 0.8 | 0.990161 | 0.0114 |  |
| 41 | kda_conv | 0.159 | 0.178 | 0.166 | 0.89 | 0.987375 | 0.0104 |  |
| 42 | kda_conv | 0.139 | 0.178 | 0.166 | 0.78 | 0.99036 | 0.0123 |  |
| 44 | kda_conv | 0.147 | 0.198 | 0.206 | 0.71 | 0.989273 | 0.0107 |  |


</details>

<details><summary>o_full_3256 vs v_full_3256: 0 above noise, 0 patterns</summary>

Per layer, `o_full_3256` vs `v_full_3256`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0693 | 0.0667 | 0.0675 | 1 | 0.997597 | 0.382 |  |
| 7 | mla | 0.121 | 0.119 | 0.12 | 1 | 0.99265 | 0.245 |  |
| 11 | mla | 0.0979 | 0.119 | 0.12 | 0.81 | 0.995207 | 0.295 |  |
| 15 | mla | 0.135 | 0.135 | 0.135 | 1 | 0.99092 | 0.234 |  |
| 19 | mla | 0.234 | 0.234 | 0.234 | 1 | 0.972722 | 0.162 |  |
| 23 | mla | 0.292 | 0.292 | 0.29 | 1 | 0.957334 | 0.133 |  |
| 27 | mla | 0.296 | 0.297 | 0.297 | 1 | 0.956108 | 0.145 |  |
| 31 | mla | 0.187 | 0.297 | 0.297 | 0.63 | 0.982484 | 0.198 |  |
| 35 | mla | 0.199 | 0.297 | 0.297 | 0.67 | 0.980197 | 0.203 |  |
| 39 | mla | 0.256 | 0.297 | 0.297 | 0.86 | 0.9672 | 0.152 |  |
| 43 | mla | 0.277 | 0.297 | 0.297 | 0.93 | 0.961539 | 0.125 |  |
| 3 | indexer | 0.039 | 0.0373 | 0.0385 | 1 | 0.999239 | 0.579 |  |
| 7 | indexer | 0.0464 | 0.0457 | 0.0465 | 1 | 0.998924 | 0.53 |  |
| 11 | indexer | 0.0542 | 0.0537 | 0.0542 | 1 | 0.998528 | 0.465 |  |
| 15 | indexer | 0.0649 | 0.0652 | 0.0644 | 1 | 0.997893 | 0.401 |  |
| 19 | indexer | 0.136 | 0.143 | 0.136 | 0.95 | 0.990763 | 0.313 |  |
| 23 | indexer | 0.186 | 0.17 | 0.162 | 1.1 | 0.982665 | 0.285 |  |
| 27 | indexer | 0.177 | 0.17 | 0.162 | 1 | 0.98437 | 0.313 |  |
| 31 | indexer | 0.0986 | 0.17 | 0.162 | 0.58 | 0.995146 | 0.333 |  |
| 35 | indexer | 0.115 | 0.17 | 0.162 | 0.67 | 0.993458 | 0.375 |  |
| 39 | indexer | 0.124 | 0.17 | 0.162 | 0.73 | 0.992273 | 0.355 |  |
| 43 | indexer | 0.121 | 0.17 | 0.162 | 0.71 | 0.992622 | 0.239 |  |
| 0 | kda_rec | 0.00735 | 0.00478 | 0.00304 | 1.5 | 0.999973 | 4.77e-06 |  |
| 1 | kda_rec | 0.0232 | 0.0187 | 0.0205 | 1.1 | 0.999731 | 9.54e-07 |  |
| 2 | kda_rec | 0.034 | 0.0284 | 0.0304 | 1.1 | 0.999422 | 9.54e-07 |  |
| 4 | kda_rec | 0.0396 | 0.0401 | 0.0408 | 0.97 | 0.999216 | 0 |  |
| 5 | kda_rec | 0.0293 | 0.0401 | 0.0408 | 0.72 | 0.999582 | 0 |  |
| 6 | kda_rec | 0.0444 | 0.0497 | 0.0482 | 0.89 | 0.999016 | 1.91e-06 |  |
| 8 | kda_rec | 0.026 | 0.0497 | 0.0482 | 0.52 | 0.999663 | 0 |  |
| 9 | kda_rec | 0.0657 | 0.0732 | 0.0722 | 0.9 | 0.997847 | 0 |  |
| 10 | kda_rec | 0.0443 | 0.0732 | 0.0722 | 0.6 | 0.999021 | 0 |  |
| 12 | kda_rec | 0.0658 | 0.0901 | 0.0927 | 0.71 | 0.997836 | 9.54e-07 |  |
| 13 | kda_rec | 0.0426 | 0.0901 | 0.0927 | 0.46 | 0.999092 | 0 |  |
| 14 | kda_rec | 0.0721 | 0.0901 | 0.0927 | 0.78 | 0.997402 | 0.000116 |  |
| 16 | kda_rec | 0.0608 | 0.0901 | 0.0927 | 0.66 | 0.998154 | 0.000123 |  |
| 17 | kda_rec | 0.064 | 0.0901 | 0.0927 | 0.69 | 0.997951 | 0 |  |
| 18 | kda_rec | 0.116 | 0.121 | 0.123 | 0.95 | 0.993213 | 0 |  |
| 20 | kda_rec | 0.109 | 0.121 | 0.123 | 0.89 | 0.994004 | 0 |  |
| 21 | kda_rec | 0.148 | 0.157 | 0.162 | 0.91 | 0.989046 | 0 |  |
| 22 | kda_rec | 0.112 | 0.157 | 0.162 | 0.69 | 0.993791 | 9.54e-07 |  |
| 24 | kda_rec | 0.142 | 0.157 | 0.162 | 0.88 | 0.990134 | 0 |  |
| 25 | kda_rec | 0.0791 | 0.157 | 0.162 | 0.49 | 0.996875 | 1.62e-05 |  |
| 26 | kda_rec | 0.153 | 0.157 | 0.162 | 0.95 | 0.98828 | 0 |  |
| 28 | kda_rec | 0.174 | 0.181 | 0.162 | 0.96 | 0.984839 | 1.91e-06 |  |
| 29 | kda_rec | 0.126 | 0.181 | 0.162 | 0.7 | 0.992032 | 0 |  |
| 30 | kda_rec | 0.0974 | 0.181 | 0.162 | 0.54 | 0.998032 | 9.54e-07 |  |
| 32 | kda_rec | 0.142 | 0.181 | 0.162 | 0.78 | 0.989921 | 0 |  |
| 33 | kda_rec | 0.139 | 0.181 | 0.162 | 0.77 | 0.990362 | 0 |  |
| 34 | kda_rec | 0.0261 | 0.181 | 0.162 | 0.14 | 0.999662 | 9.54e-07 |  |
| 36 | kda_rec | 0.0421 | 0.181 | 0.162 | 0.23 | 0.999114 | 9.54e-07 |  |
| 37 | kda_rec | 0.149 | 0.181 | 0.162 | 0.82 | 0.988897 | 0 |  |
| 38 | kda_rec | 0.174 | 0.192 | 0.18 | 0.9 | 0.98486 | 0.000129 |  |
| 40 | kda_rec | 0.163 | 0.192 | 0.18 | 0.85 | 0.986706 | 0.000122 |  |
| 41 | kda_rec | 0.171 | 0.192 | 0.18 | 0.89 | 0.98538 | 0 |  |
| 42 | kda_rec | 0.182 | 0.2 | 0.189 | 0.91 | 0.983352 | 0 |  |
| 44 | kda_rec | 0.176 | 0.2 | 0.2 | 0.88 | 0.984485 | 0 |  |
| 0 | kda_conv | 0 | 0.000437 | 0.000807 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.033 | 0.025 | 0.0329 | 1 | 0.999459 | 0.0563 |  |
| 2 | kda_conv | 0.0698 | 0.0638 | 0.0674 | 1 | 0.997565 | 0.0257 |  |
| 4 | kda_conv | 0.0251 | 0.0638 | 0.0674 | 0.37 | 0.999685 | 0.0595 |  |
| 5 | kda_conv | 0.0428 | 0.0638 | 0.0674 | 0.63 | 0.999089 | 0.0392 |  |
| 6 | kda_conv | 0.0399 | 0.0638 | 0.0674 | 0.59 | 0.999214 | 0.0384 |  |
| 8 | kda_conv | 0.066 | 0.0789 | 0.0716 | 0.84 | 0.997826 | 0.0245 |  |
| 9 | kda_conv | 0.0553 | 0.0812 | 0.0718 | 0.68 | 0.99847 | 0.0294 |  |
| 10 | kda_conv | 0.0435 | 0.0812 | 0.0718 | 0.54 | 0.999054 | 0.0366 |  |
| 12 | kda_conv | 0.0568 | 0.0812 | 0.0718 | 0.7 | 0.998387 | 0.0284 |  |
| 13 | kda_conv | 0.0731 | 0.106 | 0.094 | 0.69 | 0.997334 | 0.022 |  |
| 14 | kda_conv | 0.072 | 0.106 | 0.094 | 0.68 | 0.997436 | 0.0222 |  |
| 16 | kda_conv | 0.0669 | 0.106 | 0.094 | 0.63 | 0.997779 | 0.023 |  |
| 17 | kda_conv | 0.0861 | 0.107 | 0.106 | 0.81 | 0.996318 | 0.0188 |  |
| 18 | kda_conv | 0.12 | 0.107 | 0.119 | 1 | 0.992991 | 0.0134 |  |
| 20 | kda_conv | 0.129 | 0.107 | 0.122 | 1.1 | 0.991695 | 0.0134 |  |
| 21 | kda_conv | 0.129 | 0.126 | 0.144 | 0.89 | 0.991729 | 0.0126 |  |
| 22 | kda_conv | 0.113 | 0.126 | 0.144 | 0.78 | 0.99362 | 0.0146 |  |
| 24 | kda_conv | 0.147 | 0.126 | 0.144 | 1 | 0.989418 | 0.0108 |  |
| 25 | kda_conv | 0.0868 | 0.126 | 0.144 | 0.6 | 0.99624 | 0.0171 |  |
| 26 | kda_conv | 0.144 | 0.126 | 0.144 | 1 | 0.989635 | 0.0118 |  |
| 28 | kda_conv | 0.123 | 0.132 | 0.144 | 0.85 | 0.992504 | 0.0133 |  |
| 29 | kda_conv | 0.111 | 0.132 | 0.144 | 0.77 | 0.993838 | 0.0139 |  |
| 30 | kda_conv | 0.125 | 0.132 | 0.144 | 0.87 | 0.992155 | 0.0129 |  |
| 32 | kda_conv | 0.125 | 0.132 | 0.144 | 0.87 | 0.992249 | 0.0125 |  |
| 33 | kda_conv | 0.13 | 0.132 | 0.144 | 0.9 | 0.991485 | 0.0108 |  |
| 34 | kda_conv | 0.0891 | 0.132 | 0.144 | 0.62 | 0.996052 | 0.0152 |  |
| 36 | kda_conv | 0.0942 | 0.132 | 0.144 | 0.65 | 0.995562 | 0.016 |  |
| 37 | kda_conv | 0.166 | 0.141 | 0.144 | 1.1 | 0.986235 | 0.00916 |  |
| 38 | kda_conv | 0.141 | 0.155 | 0.151 | 0.91 | 0.990089 | 0.0117 |  |
| 40 | kda_conv | 0.166 | 0.175 | 0.165 | 0.95 | 0.986117 | 0.00909 |  |
| 41 | kda_conv | 0.147 | 0.178 | 0.166 | 0.83 | 0.989116 | 0.0111 |  |
| 42 | kda_conv | 0.105 | 0.178 | 0.166 | 0.59 | 0.994487 | 0.0143 |  |
| 44 | kda_conv | 0.118 | 0.198 | 0.206 | 0.57 | 0.99306 | 0.0125 |  |


</details>

<details><summary>o_prefix_3000_256 vs v_full_3256: 0 above noise, 0 patterns</summary>

Per layer, `o_prefix_3000_256` vs `v_full_3256`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0694 | 0.0667 | 0.0675 | 1 | 0.997592 | 0.381 |  |
| 7 | mla | 0.122 | 0.119 | 0.12 | 1 | 0.992587 | 0.245 |  |
| 11 | mla | 0.0982 | 0.119 | 0.12 | 0.82 | 0.995182 | 0.294 |  |
| 15 | mla | 0.134 | 0.135 | 0.135 | 0.99 | 0.991024 | 0.235 |  |
| 19 | mla | 0.232 | 0.234 | 0.234 | 0.99 | 0.973148 | 0.162 |  |
| 23 | mla | 0.29 | 0.292 | 0.29 | 0.99 | 0.958021 | 0.133 |  |
| 27 | mla | 0.294 | 0.297 | 0.297 | 0.99 | 0.956842 | 0.145 |  |
| 31 | mla | 0.182 | 0.297 | 0.297 | 0.61 | 0.983381 | 0.199 |  |
| 35 | mla | 0.195 | 0.297 | 0.297 | 0.66 | 0.981029 | 0.204 |  |
| 39 | mla | 0.254 | 0.297 | 0.297 | 0.85 | 0.967782 | 0.152 |  |
| 43 | mla | 0.274 | 0.297 | 0.297 | 0.92 | 0.962452 | 0.126 |  |
| 3 | indexer | 0.039 | 0.0373 | 0.0385 | 1 | 0.999241 | 0.581 |  |
| 7 | indexer | 0.0466 | 0.0457 | 0.0465 | 1 | 0.998915 | 0.527 |  |
| 11 | indexer | 0.0543 | 0.0537 | 0.0542 | 1 | 0.998527 | 0.465 |  |
| 15 | indexer | 0.0657 | 0.0652 | 0.0644 | 1 | 0.997843 | 0.4 |  |
| 19 | indexer | 0.141 | 0.143 | 0.136 | 0.99 | 0.990053 | 0.31 |  |
| 23 | indexer | 0.188 | 0.17 | 0.162 | 1.1 | 0.982259 | 0.284 |  |
| 27 | indexer | 0.18 | 0.17 | 0.162 | 1.1 | 0.983822 | 0.316 |  |
| 31 | indexer | 0.0943 | 0.17 | 0.162 | 0.55 | 0.995553 | 0.339 |  |
| 35 | indexer | 0.118 | 0.17 | 0.162 | 0.69 | 0.993028 | 0.376 |  |
| 39 | indexer | 0.124 | 0.17 | 0.162 | 0.73 | 0.992306 | 0.359 |  |
| 43 | indexer | 0.121 | 0.17 | 0.162 | 0.71 | 0.992618 | 0.238 |  |
| 0 | kda_rec | 0.00737 | 0.00478 | 0.00304 | 1.5 | 0.999973 | 5.72e-06 |  |
| 1 | kda_rec | 0.0231 | 0.0187 | 0.0205 | 1.1 | 0.999733 | 0 |  |
| 2 | kda_rec | 0.035 | 0.0284 | 0.0304 | 1.2 | 0.999392 | 0 |  |
| 4 | kda_rec | 0.0395 | 0.0401 | 0.0408 | 0.97 | 0.999218 | 3.81e-06 |  |
| 5 | kda_rec | 0.0297 | 0.0401 | 0.0408 | 0.73 | 0.999577 | 9.54e-07 |  |
| 6 | kda_rec | 0.0455 | 0.0497 | 0.0482 | 0.92 | 0.998965 | 0 |  |
| 8 | kda_rec | 0.0263 | 0.0497 | 0.0482 | 0.53 | 0.999654 | 0 |  |
| 9 | kda_rec | 0.0667 | 0.0732 | 0.0722 | 0.91 | 0.997777 | 0 |  |
| 10 | kda_rec | 0.0492 | 0.0732 | 0.0722 | 0.67 | 0.998787 | 0 |  |
| 12 | kda_rec | 0.0654 | 0.0901 | 0.0927 | 0.71 | 0.997858 | 0 |  |
| 13 | kda_rec | 0.0406 | 0.0901 | 0.0927 | 0.44 | 0.999177 | 0 |  |
| 14 | kda_rec | 0.0744 | 0.0901 | 0.0927 | 0.8 | 0.997228 | 0.000115 |  |
| 16 | kda_rec | 0.0608 | 0.0901 | 0.0927 | 0.66 | 0.99815 | 0.000122 |  |
| 17 | kda_rec | 0.064 | 0.0901 | 0.0927 | 0.69 | 0.997949 | 0 |  |
| 18 | kda_rec | 0.112 | 0.121 | 0.123 | 0.91 | 0.993661 | 0 |  |
| 20 | kda_rec | 0.108 | 0.121 | 0.123 | 0.88 | 0.994187 | 0 |  |
| 21 | kda_rec | 0.149 | 0.157 | 0.162 | 0.92 | 0.988932 | 0 |  |
| 22 | kda_rec | 0.108 | 0.157 | 0.162 | 0.67 | 0.994193 | 0 |  |
| 24 | kda_rec | 0.134 | 0.157 | 0.162 | 0.83 | 0.991168 | 0 |  |
| 25 | kda_rec | 0.0795 | 0.157 | 0.162 | 0.49 | 0.996858 | 1.62e-05 |  |
| 26 | kda_rec | 0.161 | 0.157 | 0.162 | 1 | 0.987098 | 0 |  |
| 28 | kda_rec | 0.187 | 0.181 | 0.162 | 1 | 0.982653 | 1.91e-06 |  |
| 29 | kda_rec | 0.123 | 0.181 | 0.162 | 0.68 | 0.992433 | 0 |  |
| 30 | kda_rec | 0.0636 | 0.181 | 0.162 | 0.35 | 0.998687 | 0 |  |
| 32 | kda_rec | 0.153 | 0.181 | 0.162 | 0.84 | 0.988289 | 0 |  |
| 33 | kda_rec | 0.141 | 0.181 | 0.162 | 0.78 | 0.990058 | 0 |  |
| 34 | kda_rec | 0.0248 | 0.181 | 0.162 | 0.14 | 0.999694 | 9.54e-07 |  |
| 36 | kda_rec | 0.0439 | 0.181 | 0.162 | 0.24 | 0.999035 | 9.54e-07 |  |
| 37 | kda_rec | 0.153 | 0.181 | 0.162 | 0.85 | 0.988287 | 0 |  |
| 38 | kda_rec | 0.18 | 0.192 | 0.18 | 0.94 | 0.983783 | 0.000129 |  |
| 40 | kda_rec | 0.171 | 0.192 | 0.18 | 0.89 | 0.985359 | 0.000122 |  |
| 41 | kda_rec | 0.179 | 0.192 | 0.18 | 0.93 | 0.983938 | 0 |  |
| 42 | kda_rec | 0.188 | 0.2 | 0.189 | 0.94 | 0.982211 | 0 |  |
| 44 | kda_rec | 0.174 | 0.2 | 0.2 | 0.87 | 0.984913 | 0 |  |
| 0 | kda_conv | 0.000617 | 0.000437 | 0.000807 | 0.76 | 1 | 0.919 |  |
| 1 | kda_conv | 0.032 | 0.025 | 0.0329 | 0.98 | 0.999487 | 0.0545 |  |
| 2 | kda_conv | 0.0672 | 0.0638 | 0.0674 | 1 | 0.997741 | 0.0267 |  |
| 4 | kda_conv | 0.0242 | 0.0638 | 0.0674 | 0.36 | 0.999708 | 0.0638 |  |
| 5 | kda_conv | 0.0452 | 0.0638 | 0.0674 | 0.67 | 0.99898 | 0.0366 |  |
| 6 | kda_conv | 0.0425 | 0.0638 | 0.0674 | 0.63 | 0.999098 | 0.0377 |  |
| 8 | kda_conv | 0.0636 | 0.0789 | 0.0716 | 0.81 | 0.997983 | 0.0256 |  |
| 9 | kda_conv | 0.0626 | 0.0812 | 0.0718 | 0.77 | 0.998039 | 0.0256 |  |
| 10 | kda_conv | 0.0468 | 0.0812 | 0.0718 | 0.58 | 0.998904 | 0.0336 |  |
| 12 | kda_conv | 0.0548 | 0.0812 | 0.0718 | 0.67 | 0.998504 | 0.0288 |  |
| 13 | kda_conv | 0.0646 | 0.106 | 0.094 | 0.61 | 0.997915 | 0.0241 |  |
| 14 | kda_conv | 0.0868 | 0.106 | 0.094 | 0.82 | 0.996371 | 0.0188 |  |
| 16 | kda_conv | 0.0779 | 0.106 | 0.094 | 0.74 | 0.996965 | 0.0192 |  |
| 17 | kda_conv | 0.0951 | 0.107 | 0.106 | 0.89 | 0.995534 | 0.0164 |  |
| 18 | kda_conv | 0.108 | 0.107 | 0.119 | 0.91 | 0.994111 | 0.0142 |  |
| 20 | kda_conv | 0.113 | 0.107 | 0.122 | 0.93 | 0.993636 | 0.0143 |  |
| 21 | kda_conv | 0.116 | 0.126 | 0.144 | 0.8 | 0.993295 | 0.013 |  |
| 22 | kda_conv | 0.104 | 0.126 | 0.144 | 0.72 | 0.994561 | 0.0151 |  |
| 24 | kda_conv | 0.132 | 0.126 | 0.144 | 0.92 | 0.991359 | 0.0125 |  |
| 25 | kda_conv | 0.0846 | 0.126 | 0.144 | 0.59 | 0.996414 | 0.0192 |  |
| 26 | kda_conv | 0.134 | 0.126 | 0.144 | 0.93 | 0.990967 | 0.0127 |  |
| 28 | kda_conv | 0.133 | 0.132 | 0.144 | 0.92 | 0.991136 | 0.0129 |  |
| 29 | kda_conv | 0.137 | 0.132 | 0.144 | 0.95 | 0.990566 | 0.0127 |  |
| 30 | kda_conv | 0.123 | 0.132 | 0.144 | 0.86 | 0.992404 | 0.0148 |  |
| 32 | kda_conv | 0.145 | 0.132 | 0.144 | 1 | 0.989411 | 0.0118 |  |
| 33 | kda_conv | 0.16 | 0.132 | 0.144 | 1.1 | 0.987179 | 0.00918 |  |
| 34 | kda_conv | 0.0921 | 0.132 | 0.144 | 0.64 | 0.995814 | 0.0164 |  |
| 36 | kda_conv | 0.0966 | 0.132 | 0.144 | 0.67 | 0.995353 | 0.0167 |  |
| 37 | kda_conv | 0.188 | 0.141 | 0.144 | 1.3 | 0.982228 | 0.00761 |  |
| 38 | kda_conv | 0.162 | 0.155 | 0.151 | 1 | 0.986766 | 0.00996 |  |
| 40 | kda_conv | 0.188 | 0.175 | 0.165 | 1.1 | 0.982346 | 0.00833 |  |
| 41 | kda_conv | 0.169 | 0.178 | 0.166 | 0.95 | 0.985646 | 0.00884 |  |
| 42 | kda_conv | 0.112 | 0.178 | 0.166 | 0.63 | 0.993744 | 0.0134 |  |
| 44 | kda_conv | 0.117 | 0.198 | 0.206 | 0.57 | 0.993136 | 0.0129 |  |


</details>

<details><summary>o_full_3003 vs v_full_3003: 0 above noise, 0 patterns</summary>

Per layer, `o_full_3003` vs `v_full_3003`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0694 | 0.0667 | 0.0675 | 1 | 0.997592 | 0.381 |  |
| 7 | mla | 0.122 | 0.119 | 0.12 | 1 | 0.992618 | 0.245 |  |
| 11 | mla | 0.099 | 0.119 | 0.12 | 0.82 | 0.9951 | 0.292 |  |
| 15 | mla | 0.138 | 0.135 | 0.135 | 1 | 0.990475 | 0.231 |  |
| 19 | mla | 0.235 | 0.234 | 0.234 | 1 | 0.972307 | 0.159 |  |
| 23 | mla | 0.288 | 0.292 | 0.29 | 0.99 | 0.958343 | 0.132 |  |
| 27 | mla | 0.295 | 0.297 | 0.297 | 0.99 | 0.956491 | 0.143 |  |
| 31 | mla | 0.186 | 0.297 | 0.297 | 0.63 | 0.982636 | 0.196 |  |
| 35 | mla | 0.197 | 0.297 | 0.297 | 0.66 | 0.980521 | 0.2 |  |
| 39 | mla | 0.259 | 0.297 | 0.297 | 0.87 | 0.966493 | 0.149 |  |
| 43 | mla | 0.283 | 0.297 | 0.297 | 0.95 | 0.959967 | 0.124 |  |
| 3 | indexer | 0.0389 | 0.0373 | 0.0385 | 1 | 0.999243 | 0.581 |  |
| 7 | indexer | 0.0466 | 0.0457 | 0.0465 | 1 | 0.998915 | 0.527 |  |
| 11 | indexer | 0.0546 | 0.0537 | 0.0542 | 1 | 0.998508 | 0.462 |  |
| 15 | indexer | 0.0667 | 0.0652 | 0.0644 | 1 | 0.997776 | 0.394 |  |
| 19 | indexer | 0.141 | 0.143 | 0.136 | 0.98 | 0.990114 | 0.313 |  |
| 23 | indexer | 0.169 | 0.17 | 0.162 | 0.99 | 0.985739 | 0.285 |  |
| 27 | indexer | 0.164 | 0.17 | 0.162 | 0.96 | 0.986604 | 0.312 |  |
| 31 | indexer | 0.0941 | 0.17 | 0.162 | 0.55 | 0.995571 | 0.334 |  |
| 35 | indexer | 0.114 | 0.17 | 0.162 | 0.67 | 0.993532 | 0.372 |  |
| 39 | indexer | 0.126 | 0.17 | 0.162 | 0.74 | 0.992085 | 0.347 |  |
| 43 | indexer | 0.122 | 0.17 | 0.162 | 0.72 | 0.992508 | 0.238 |  |
| 3 | tail | 0.0499 | 0.0455 | 0.0503 | 0.99 | 0.998753 | 0.0391 |  |
| 7 | tail | 0.0455 | 0.0465 | 0.0503 | 0.9 | 0.998965 | 0.0378 |  |
| 11 | tail | 0.0565 | 0.0491 | 0.0537 | 1.1 | 0.998402 | 0.0286 |  |
| 15 | tail | 0.054 | 0.0509 | 0.055 | 0.98 | 0.998541 | 0.026 |  |
| 19 | tail | 0.0652 | 0.0604 | 0.0654 | 1 | 0.997911 | 0.0286 |  |
| 23 | tail | 0.0848 | 0.0757 | 0.0784 | 1.1 | 0.996397 | 0.026 |  |
| 27 | tail | 0.0656 | 0.0757 | 0.0784 | 0.84 | 0.997848 | 0.0312 |  |
| 31 | tail | 0.0589 | 0.0757 | 0.0784 | 0.75 | 0.998276 | 0.0312 |  |
| 35 | tail | 0.053 | 0.0757 | 0.0784 | 0.68 | 0.998593 | 0.0404 |  |
| 39 | tail | 0.0655 | 0.0757 | 0.0784 | 0.84 | 0.997853 | 0.0208 |  |
| 43 | tail | 0.0949 | 0.113 | 0.102 | 0.84 | 0.995488 | 0.0169 |  |
| 0 | kda_rec | 0.00767 | 0.00478 | 0.00304 | 1.6 | 0.999971 | 4.77e-06 |  |
| 1 | kda_rec | 0.023 | 0.0187 | 0.0205 | 1.1 | 0.999738 | 0 |  |
| 2 | kda_rec | 0.0326 | 0.0284 | 0.0304 | 1.1 | 0.999468 | 0 |  |
| 4 | kda_rec | 0.0413 | 0.0401 | 0.0408 | 1 | 0.999151 | 9.54e-07 |  |
| 5 | kda_rec | 0.0306 | 0.0401 | 0.0408 | 0.75 | 0.999534 | 0 |  |
| 6 | kda_rec | 0.0498 | 0.0497 | 0.0482 | 1 | 0.998765 | 0 |  |
| 8 | kda_rec | 0.0276 | 0.0497 | 0.0482 | 0.56 | 0.999619 | 0 |  |
| 9 | kda_rec | 0.0765 | 0.0732 | 0.0722 | 1 | 0.997109 | 0 |  |
| 10 | kda_rec | 0.0513 | 0.0732 | 0.0722 | 0.7 | 0.998697 | 9.54e-07 |  |
| 12 | kda_rec | 0.0875 | 0.0901 | 0.0927 | 0.94 | 0.99618 | 0 |  |
| 13 | kda_rec | 0.0488 | 0.0901 | 0.0927 | 0.53 | 0.998815 | 0 |  |
| 14 | kda_rec | 0.0842 | 0.0901 | 0.0927 | 0.91 | 0.996464 | 0.000137 |  |
| 16 | kda_rec | 0.0672 | 0.0901 | 0.0927 | 0.72 | 0.997744 | 0.000117 |  |
| 17 | kda_rec | 0.0844 | 0.0901 | 0.0927 | 0.91 | 0.996435 | 0 |  |
| 18 | kda_rec | 0.127 | 0.121 | 0.123 | 1 | 0.991935 | 9.54e-07 |  |
| 20 | kda_rec | 0.119 | 0.121 | 0.123 | 0.97 | 0.992903 | 0 |  |
| 21 | kda_rec | 0.166 | 0.157 | 0.162 | 1 | 0.986169 | 0 |  |
| 22 | kda_rec | 0.125 | 0.157 | 0.162 | 0.77 | 0.992137 | 9.54e-07 |  |
| 24 | kda_rec | 0.125 | 0.157 | 0.162 | 0.77 | 0.992257 | 0 |  |
| 25 | kda_rec | 0.0773 | 0.157 | 0.162 | 0.48 | 0.997029 | 7.63e-06 |  |
| 26 | kda_rec | 0.137 | 0.157 | 0.162 | 0.85 | 0.990598 | 0 |  |
| 28 | kda_rec | 0.182 | 0.181 | 0.162 | 1 | 0.983614 | 0 |  |
| 29 | kda_rec | 0.131 | 0.181 | 0.162 | 0.72 | 0.991434 | 0 |  |
| 30 | kda_rec | 0.0672 | 0.181 | 0.162 | 0.37 | 0.998602 | 0 |  |
| 32 | kda_rec | 0.146 | 0.181 | 0.162 | 0.81 | 0.989398 | 0 |  |
| 33 | kda_rec | 0.152 | 0.181 | 0.162 | 0.84 | 0.988457 | 0 |  |
| 34 | kda_rec | 0.0261 | 0.181 | 0.162 | 0.14 | 0.999662 | 0 |  |
| 36 | kda_rec | 0.0472 | 0.181 | 0.162 | 0.26 | 0.99889 | 0 |  |
| 37 | kda_rec | 0.158 | 0.181 | 0.162 | 0.87 | 0.987518 | 0 |  |
| 38 | kda_rec | 0.191 | 0.192 | 0.18 | 1 | 0.981644 | 9.16e-05 |  |
| 40 | kda_rec | 0.171 | 0.192 | 0.18 | 0.89 | 0.985266 | 0.000122 |  |
| 41 | kda_rec | 0.177 | 0.192 | 0.18 | 0.92 | 0.984193 | 0 |  |
| 42 | kda_rec | 0.198 | 0.2 | 0.189 | 0.99 | 0.980382 | 0 |  |
| 44 | kda_rec | 0.188 | 0.2 | 0.2 | 0.94 | 0.982386 | 0 |  |
| 0 | kda_conv | 0 | 0.000437 | 0.000807 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0338 | 0.025 | 0.0329 | 1 | 0.99943 | 0.0533 |  |
| 2 | kda_conv | 0.072 | 0.0638 | 0.0674 | 1.1 | 0.997405 | 0.0264 |  |
| 4 | kda_conv | 0.0408 | 0.0638 | 0.0674 | 0.6 | 0.999247 | 0.0382 |  |
| 5 | kda_conv | 0.0609 | 0.0638 | 0.0674 | 0.9 | 0.998153 | 0.0278 |  |
| 6 | kda_conv | 0.057 | 0.0638 | 0.0674 | 0.84 | 0.9984 | 0.0291 |  |
| 8 | kda_conv | 0.0757 | 0.0789 | 0.0716 | 0.96 | 0.99714 | 0.0215 |  |
| 9 | kda_conv | 0.0802 | 0.0812 | 0.0718 | 0.99 | 0.996794 | 0.022 |  |
| 10 | kda_conv | 0.0648 | 0.0812 | 0.0718 | 0.8 | 0.997907 | 0.0275 |  |
| 12 | kda_conv | 0.0709 | 0.0812 | 0.0718 | 0.87 | 0.997486 | 0.0245 |  |
| 13 | kda_conv | 0.108 | 0.106 | 0.094 | 1 | 0.994123 | 0.0171 |  |
| 14 | kda_conv | 0.0971 | 0.106 | 0.094 | 0.92 | 0.995306 | 0.018 |  |
| 16 | kda_conv | 0.0822 | 0.106 | 0.094 | 0.78 | 0.996667 | 0.0187 |  |
| 17 | kda_conv | 0.106 | 0.107 | 0.106 | 0.99 | 0.994387 | 0.0154 |  |
| 18 | kda_conv | 0.112 | 0.107 | 0.119 | 0.94 | 0.993701 | 0.0142 |  |
| 20 | kda_conv | 0.132 | 0.107 | 0.122 | 1.1 | 0.991243 | 0.0126 |  |
| 21 | kda_conv | 0.159 | 0.126 | 0.144 | 1.1 | 0.987444 | 0.0113 |  |
| 22 | kda_conv | 0.136 | 0.126 | 0.144 | 0.94 | 0.990746 | 0.0127 |  |
| 24 | kda_conv | 0.131 | 0.126 | 0.144 | 0.91 | 0.991333 | 0.0118 |  |
| 25 | kda_conv | 0.112 | 0.126 | 0.144 | 0.78 | 0.993758 | 0.015 |  |
| 26 | kda_conv | 0.131 | 0.126 | 0.144 | 0.91 | 0.991449 | 0.0126 |  |
| 28 | kda_conv | 0.134 | 0.132 | 0.144 | 0.93 | 0.991056 | 0.0119 |  |
| 29 | kda_conv | 0.132 | 0.132 | 0.144 | 0.92 | 0.991212 | 0.0121 |  |
| 30 | kda_conv | 0.118 | 0.132 | 0.144 | 0.82 | 0.993043 | 0.0133 |  |
| 32 | kda_conv | 0.131 | 0.132 | 0.144 | 0.91 | 0.991463 | 0.0135 |  |
| 33 | kda_conv | 0.132 | 0.132 | 0.144 | 0.91 | 0.991293 | 0.0131 |  |
| 34 | kda_conv | 0.0749 | 0.132 | 0.144 | 0.52 | 0.99719 | 0.018 |  |
| 36 | kda_conv | 0.0841 | 0.132 | 0.144 | 0.58 | 0.996477 | 0.0189 |  |
| 37 | kda_conv | 0.144 | 0.141 | 0.144 | 1 | 0.989587 | 0.0112 |  |
| 38 | kda_conv | 0.134 | 0.155 | 0.151 | 0.87 | 0.991003 | 0.0122 |  |
| 40 | kda_conv | 0.163 | 0.175 | 0.165 | 0.93 | 0.986726 | 0.00971 |  |
| 41 | kda_conv | 0.176 | 0.178 | 0.166 | 0.99 | 0.984654 | 0.00959 |  |
| 42 | kda_conv | 0.153 | 0.178 | 0.166 | 0.85 | 0.988294 | 0.00983 |  |
| 44 | kda_conv | 0.153 | 0.198 | 0.206 | 0.75 | 0.988172 | 0.01 |  |


</details>

<details><summary>o_prefix_3000_3 vs v_full_3003: 0 above noise, 0 patterns</summary>

Per layer, `o_prefix_3000_3` vs `v_full_3003`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0694 | 0.0667 | 0.0675 | 1 | 0.997592 | 0.381 |  |
| 7 | mla | 0.122 | 0.119 | 0.12 | 1 | 0.992618 | 0.245 |  |
| 11 | mla | 0.099 | 0.119 | 0.12 | 0.82 | 0.9951 | 0.292 |  |
| 15 | mla | 0.138 | 0.135 | 0.135 | 1 | 0.990475 | 0.231 |  |
| 19 | mla | 0.235 | 0.234 | 0.234 | 1 | 0.972308 | 0.159 |  |
| 23 | mla | 0.288 | 0.292 | 0.29 | 0.99 | 0.958343 | 0.132 |  |
| 27 | mla | 0.295 | 0.297 | 0.297 | 0.99 | 0.956492 | 0.143 |  |
| 31 | mla | 0.186 | 0.297 | 0.297 | 0.63 | 0.982636 | 0.196 |  |
| 35 | mla | 0.197 | 0.297 | 0.297 | 0.66 | 0.980521 | 0.2 |  |
| 39 | mla | 0.259 | 0.297 | 0.297 | 0.87 | 0.966493 | 0.149 |  |
| 43 | mla | 0.283 | 0.297 | 0.297 | 0.95 | 0.959967 | 0.124 |  |
| 3 | indexer | 0.0389 | 0.0373 | 0.0385 | 1 | 0.999243 | 0.581 |  |
| 7 | indexer | 0.0466 | 0.0457 | 0.0465 | 1 | 0.998915 | 0.527 |  |
| 11 | indexer | 0.0546 | 0.0537 | 0.0542 | 1 | 0.998508 | 0.462 |  |
| 15 | indexer | 0.0667 | 0.0652 | 0.0644 | 1 | 0.997776 | 0.394 |  |
| 19 | indexer | 0.141 | 0.143 | 0.136 | 0.98 | 0.990114 | 0.313 |  |
| 23 | indexer | 0.169 | 0.17 | 0.162 | 0.99 | 0.985739 | 0.285 |  |
| 27 | indexer | 0.164 | 0.17 | 0.162 | 0.96 | 0.986604 | 0.312 |  |
| 31 | indexer | 0.0941 | 0.17 | 0.162 | 0.55 | 0.995571 | 0.334 |  |
| 35 | indexer | 0.114 | 0.17 | 0.162 | 0.67 | 0.993532 | 0.372 |  |
| 39 | indexer | 0.126 | 0.17 | 0.162 | 0.74 | 0.992085 | 0.347 |  |
| 43 | indexer | 0.122 | 0.17 | 0.162 | 0.72 | 0.992508 | 0.238 |  |
| 3 | tail | 0.0548 | 0.0455 | 0.0503 | 1.1 | 0.9985 | 0.0312 |  |
| 7 | tail | 0.044 | 0.0465 | 0.0503 | 0.88 | 0.999031 | 0.0352 |  |
| 11 | tail | 0.0522 | 0.0491 | 0.0537 | 0.97 | 0.998638 | 0.0312 |  |
| 15 | tail | 0.0546 | 0.0509 | 0.055 | 0.99 | 0.998509 | 0.0286 |  |
| 19 | tail | 0.0591 | 0.0604 | 0.0654 | 0.9 | 0.998294 | 0.0286 |  |
| 23 | tail | 0.0873 | 0.0757 | 0.0784 | 1.1 | 0.996187 | 0.0247 |  |
| 27 | tail | 0.0683 | 0.0757 | 0.0784 | 0.87 | 0.997669 | 0.0365 |  |
| 31 | tail | 0.0592 | 0.0757 | 0.0784 | 0.76 | 0.998261 | 0.0378 |  |
| 35 | tail | 0.0584 | 0.0757 | 0.0784 | 0.75 | 0.998291 | 0.0365 |  |
| 39 | tail | 0.0681 | 0.0757 | 0.0784 | 0.87 | 0.997683 | 0.0299 |  |
| 43 | tail | 0.0947 | 0.113 | 0.102 | 0.84 | 0.99551 | 0.00911 |  |
| 0 | kda_rec | 0.00757 | 0.00478 | 0.00304 | 1.6 | 0.999972 | 4.77e-06 |  |
| 1 | kda_rec | 0.0227 | 0.0187 | 0.0205 | 1.1 | 0.999744 | 3.81e-06 |  |
| 2 | kda_rec | 0.0322 | 0.0284 | 0.0304 | 1.1 | 0.999481 | 0 |  |
| 4 | kda_rec | 0.0404 | 0.0401 | 0.0408 | 0.99 | 0.999184 | 1.91e-06 |  |
| 5 | kda_rec | 0.0308 | 0.0401 | 0.0408 | 0.76 | 0.999526 | 0 |  |
| 6 | kda_rec | 0.0495 | 0.0497 | 0.0482 | 1 | 0.998781 | 9.54e-07 |  |
| 8 | kda_rec | 0.0277 | 0.0497 | 0.0482 | 0.56 | 0.999617 | 0 |  |
| 9 | kda_rec | 0.0761 | 0.0732 | 0.0722 | 1 | 0.997134 | 0 |  |
| 10 | kda_rec | 0.0518 | 0.0732 | 0.0722 | 0.71 | 0.998669 | 0 |  |
| 12 | kda_rec | 0.088 | 0.0901 | 0.0927 | 0.95 | 0.996143 | 9.54e-07 |  |
| 13 | kda_rec | 0.0483 | 0.0901 | 0.0927 | 0.52 | 0.998841 | 0 |  |
| 14 | kda_rec | 0.0841 | 0.0901 | 0.0927 | 0.91 | 0.996469 | 0.000137 |  |
| 16 | kda_rec | 0.0673 | 0.0901 | 0.0927 | 0.73 | 0.997738 | 0.000118 |  |
| 17 | kda_rec | 0.084 | 0.0901 | 0.0927 | 0.91 | 0.996462 | 0 |  |
| 18 | kda_rec | 0.126 | 0.121 | 0.123 | 1 | 0.992036 | 9.54e-07 |  |
| 20 | kda_rec | 0.119 | 0.121 | 0.123 | 0.97 | 0.992925 | 0 |  |
| 21 | kda_rec | 0.166 | 0.157 | 0.162 | 1 | 0.98624 | 0 |  |
| 22 | kda_rec | 0.125 | 0.157 | 0.162 | 0.77 | 0.992215 | 9.54e-07 |  |
| 24 | kda_rec | 0.124 | 0.157 | 0.162 | 0.77 | 0.99229 | 0 |  |
| 25 | kda_rec | 0.077 | 0.157 | 0.162 | 0.48 | 0.997044 | 5.72e-06 |  |
| 26 | kda_rec | 0.137 | 0.157 | 0.162 | 0.84 | 0.990711 | 0 |  |
| 28 | kda_rec | 0.181 | 0.181 | 0.162 | 1 | 0.983646 | 0 |  |
| 29 | kda_rec | 0.13 | 0.181 | 0.162 | 0.72 | 0.991495 | 0 |  |
| 30 | kda_rec | 0.0671 | 0.181 | 0.162 | 0.37 | 0.998609 | 0 |  |
| 32 | kda_rec | 0.146 | 0.181 | 0.162 | 0.8 | 0.98943 | 0 |  |
| 33 | kda_rec | 0.151 | 0.181 | 0.162 | 0.83 | 0.988539 | 0 |  |
| 34 | kda_rec | 0.0261 | 0.181 | 0.162 | 0.14 | 0.999663 | 0 |  |
| 36 | kda_rec | 0.0471 | 0.181 | 0.162 | 0.26 | 0.998891 | 0 |  |
| 37 | kda_rec | 0.158 | 0.181 | 0.162 | 0.87 | 0.987527 | 0 |  |
| 38 | kda_rec | 0.191 | 0.192 | 0.18 | 0.99 | 0.981744 | 9.25e-05 |  |
| 40 | kda_rec | 0.172 | 0.192 | 0.18 | 0.89 | 0.985243 | 0.000122 |  |
| 41 | kda_rec | 0.175 | 0.192 | 0.18 | 0.91 | 0.984594 | 0 |  |
| 42 | kda_rec | 0.196 | 0.2 | 0.189 | 0.98 | 0.980691 | 0 |  |
| 44 | kda_rec | 0.186 | 0.2 | 0.2 | 0.93 | 0.982685 | 0 |  |
| 0 | kda_conv | 0.000829 | 0.000437 | 0.000807 | 1 | 1 | 0.908 |  |
| 1 | kda_conv | 0.0338 | 0.025 | 0.0329 | 1 | 0.999428 | 0.0536 |  |
| 2 | kda_conv | 0.0707 | 0.0638 | 0.0674 | 1 | 0.997498 | 0.0279 |  |
| 4 | kda_conv | 0.0374 | 0.0638 | 0.0674 | 0.55 | 0.999311 | 0.0427 |  |
| 5 | kda_conv | 0.0617 | 0.0638 | 0.0674 | 0.92 | 0.998092 | 0.0279 |  |
| 6 | kda_conv | 0.0561 | 0.0638 | 0.0674 | 0.83 | 0.998432 | 0.0307 |  |
| 8 | kda_conv | 0.0777 | 0.0789 | 0.0716 | 0.98 | 0.996991 | 0.0218 |  |
| 9 | kda_conv | 0.0802 | 0.0812 | 0.0718 | 0.99 | 0.996782 | 0.0222 |  |
| 10 | kda_conv | 0.0666 | 0.0812 | 0.0718 | 0.82 | 0.997782 | 0.0255 |  |
| 12 | kda_conv | 0.0736 | 0.0812 | 0.0718 | 0.91 | 0.99729 | 0.0233 |  |
| 13 | kda_conv | 0.102 | 0.106 | 0.094 | 0.97 | 0.994753 | 0.0172 |  |
| 14 | kda_conv | 0.0966 | 0.106 | 0.094 | 0.92 | 0.995328 | 0.0186 |  |
| 16 | kda_conv | 0.0842 | 0.106 | 0.094 | 0.8 | 0.996465 | 0.0187 |  |
| 17 | kda_conv | 0.103 | 0.107 | 0.106 | 0.96 | 0.994705 | 0.0155 |  |
| 18 | kda_conv | 0.102 | 0.107 | 0.119 | 0.85 | 0.994832 | 0.0154 |  |
| 20 | kda_conv | 0.127 | 0.107 | 0.122 | 1 | 0.991988 | 0.0125 |  |
| 21 | kda_conv | 0.15 | 0.126 | 0.144 | 1 | 0.988746 | 0.0114 |  |
| 22 | kda_conv | 0.131 | 0.126 | 0.144 | 0.91 | 0.991405 | 0.0127 |  |
| 24 | kda_conv | 0.123 | 0.126 | 0.144 | 0.85 | 0.992353 | 0.0125 |  |
| 25 | kda_conv | 0.109 | 0.126 | 0.144 | 0.75 | 0.994107 | 0.0155 |  |
| 26 | kda_conv | 0.125 | 0.126 | 0.144 | 0.87 | 0.992151 | 0.0121 |  |
| 28 | kda_conv | 0.13 | 0.132 | 0.144 | 0.9 | 0.991574 | 0.0114 |  |
| 29 | kda_conv | 0.13 | 0.132 | 0.144 | 0.9 | 0.991575 | 0.012 |  |
| 30 | kda_conv | 0.118 | 0.132 | 0.144 | 0.82 | 0.993074 | 0.014 |  |
| 32 | kda_conv | 0.131 | 0.132 | 0.144 | 0.91 | 0.991502 | 0.0127 |  |
| 33 | kda_conv | 0.13 | 0.132 | 0.144 | 0.9 | 0.991551 | 0.0127 |  |
| 34 | kda_conv | 0.0757 | 0.132 | 0.144 | 0.52 | 0.997129 | 0.0178 |  |
| 36 | kda_conv | 0.0838 | 0.132 | 0.144 | 0.58 | 0.996515 | 0.0193 |  |
| 37 | kda_conv | 0.146 | 0.141 | 0.144 | 1 | 0.989475 | 0.0115 |  |
| 38 | kda_conv | 0.133 | 0.155 | 0.151 | 0.86 | 0.991119 | 0.0125 |  |
| 40 | kda_conv | 0.164 | 0.175 | 0.165 | 0.94 | 0.98665 | 0.0102 |  |
| 41 | kda_conv | 0.164 | 0.178 | 0.166 | 0.92 | 0.986711 | 0.00954 |  |
| 42 | kda_conv | 0.148 | 0.178 | 0.166 | 0.83 | 0.989053 | 0.011 |  |
| 44 | kda_conv | 0.151 | 0.198 | 0.206 | 0.73 | 0.98851 | 0.0104 |  |


</details>

## 32,768 tokens

| pair (x vs reference) | kind | MLA rel L2 (layer) | MLA min exact bytes | indexer rel L2 (layer) | tail rel L2, valid slots | KDA state rel L2 (layer) | conv rel L2 (layer) | KL | top-1 | top-5 | top-k min overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v_filler_32768 vs v_full_32768 | noise_vllm | 0.218 (43) | 0.2 | 0.115 (23) | —, 0 | 0.174 (44) | 0.21 (44) | 0.000135 | = | 4/5 | 0.842 |
| o_whole_32768 vs v_full_32768 | audit | 0.239 (43) | 0.141 | 0.132 (23) | —, 0 | 0.21 (22) | 0.174 (44) | 0.000156 | = | 3/5 | 0.817 |
| o_c8192_32768 vs v_full_32768 | audit | 0.239 (43) | 0.141 | 0.132 (23) | —, 0 | 0.21 (22) | 0.174 (44) | 0.000156 | = | 3/5 | 0.817 |
| o_c8192_32768 vs o_whole_32768 | noise_ours | 0 (3) | 1 | 0 (3) | —, 0 | 0 (0) | 0 (0) | 0 | = | 5/5 | 1 |
| o_c8001_32768 vs v_full_32768 | audit | 0.239 (43) | 0.141 | 0.129 (23) | —, 0 | 0.207 (22) | 0.185 (44) | 0.000885 | = | 4/5 | 0.823 |
| o_c8001_32768 vs o_whole_32768 | noise_ours | 0.201 (43) | 0.354 | 0.116 (23) | —, 0 | 0.165 (44) | 0.206 (44) | 0.000918 | = | 3/5 | 0.856 |
| o_full_33792 vs v_full_33792 | audit | 0.237 (43) | 0.141 | 0.133 (23) | —, 0 | 0.219 (22) | 0.187 (44) | 0.105 | = | 4/5 | 0.786 |
| o_prefix_32771_1021 vs v_full_33792 | audit_prefix | 0.237 (43) | 0.141 | 0.132 (23) | —, 0 | 0.219 (22) | 0.187 (44) | 0.128 | = | 4/5 | 0.768 |
| o_prefix_32771_1021 vs o_full_33792 | prefix_vs_ours_full | 0.0364 (43) | 0.974 | 0.0179 (19) | —, 0 | 0.152 (44) | 0.178 (44) | 0.0268 | = | 4/5 | 0.912 |

<details><summary>o_whole_32768 vs v_full_32768: 1 above noise, 0 patterns</summary>

Per layer, `o_whole_32768` vs `v_full_32768`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0693 | 0 | 0.0579 | 1.2 | 0.997599 | 0.382 |  |
| 7 | mla | 0.125 | 0.0856 | 0.109 | 1.2 | 0.992182 | 0.242 |  |
| 11 | mla | 0.0992 | 0.0856 | 0.109 | 0.91 | 0.995075 | 0.291 |  |
| 15 | mla | 0.126 | 0.107 | 0.109 | 1.2 | 0.992096 | 0.243 |  |
| 19 | mla | 0.187 | 0.163 | 0.159 | 1.2 | 0.982425 | 0.194 |  |
| 23 | mla | 0.231 | 0.204 | 0.196 | 1.1 | 0.973208 | 0.163 |  |
| 27 | mla | 0.229 | 0.204 | 0.196 | 1.1 | 0.973837 | 0.181 |  |
| 31 | mla | 0.146 | 0.204 | 0.196 | 0.72 | 0.989296 | 0.242 |  |
| 35 | mla | 0.155 | 0.204 | 0.196 | 0.76 | 0.988061 | 0.246 |  |
| 39 | mla | 0.201 | 0.204 | 0.196 | 0.99 | 0.9798 | 0.177 |  |
| 43 | mla | 0.239 | 0.218 | 0.201 | 1.1 | 0.971432 | 0.141 |  |
| 3 | indexer | 0.0428 | 0 | 0.0334 | 1.3 | 0.999085 | 0.548 |  |
| 7 | indexer | 0.049 | 0.0362 | 0.0417 | 1.2 | 0.9988 | 0.511 |  |
| 11 | indexer | 0.0606 | 0.0472 | 0.0487 | 1.2 | 0.998162 | 0.415 |  |
| 15 | indexer | 0.0714 | 0.0552 | 0.0541 | 1.3 | 0.997453 | 0.36 |  |
| 19 | indexer | 0.108 | 0.0892 | 0.0921 | 1.2 | 0.994134 | 0.373 |  |
| 23 | indexer | 0.132 | 0.115 | 0.116 | 1.1 | 0.991348 | 0.336 |  |
| 27 | indexer | 0.123 | 0.115 | 0.116 | 1.1 | 0.992418 | 0.372 |  |
| 31 | indexer | 0.0796 | 0.115 | 0.116 | 0.69 | 0.996828 | 0.385 |  |
| 35 | indexer | 0.0901 | 0.115 | 0.116 | 0.78 | 0.995937 | 0.428 |  |
| 39 | indexer | 0.106 | 0.115 | 0.116 | 0.91 | 0.994434 | 0.399 |  |
| 43 | indexer | 0.111 | 0.115 | 0.116 | 0.96 | 0.99379 | 0.256 |  |
| 0 | kda_rec | 0.00972 | 0 | 0.00305 | 3.2 | 0.999953 | 6.68e-06 | **above noise** |
| 1 | kda_rec | 0.0316 | 0 | 0.0156 | 2 | 0.999526 | 0 |  |
| 2 | kda_rec | 0.0467 | 0 | 0.0232 | 2 | 0.998917 | 0 |  |
| 4 | kda_rec | 0.0409 | 0.0151 | 0.0378 | 1.1 | 0.999164 | 9.54e-07 |  |
| 5 | kda_rec | 0.0655 | 0.0298 | 0.0378 | 1.7 | 0.998411 | 0 |  |
| 6 | kda_rec | 0.0604 | 0.0298 | 0.0426 | 1.4 | 0.998186 | 0 |  |
| 8 | kda_rec | 0.0307 | 0.0298 | 0.0426 | 0.72 | 0.99953 | 0 |  |
| 9 | kda_rec | 0.0716 | 0.0553 | 0.0699 | 1 | 0.997434 | 0 |  |
| 10 | kda_rec | 0.107 | 0.0553 | 0.0699 | 1.5 | 0.99452 | 0 |  |
| 12 | kda_rec | 0.0763 | 0.0615 | 0.0707 | 1.1 | 0.997103 | 0 |  |
| 13 | kda_rec | 0.0432 | 0.0615 | 0.0707 | 0.61 | 0.999069 | 0 |  |
| 14 | kda_rec | 0.0686 | 0.0615 | 0.0707 | 0.97 | 0.997643 | 0.000112 |  |
| 16 | kda_rec | 0.0869 | 0.0615 | 0.0707 | 1.2 | 0.996543 | 0.000118 |  |
| 17 | kda_rec | 0.069 | 0.0615 | 0.0707 | 0.98 | 0.997619 | 0 |  |
| 18 | kda_rec | 0.118 | 0.101 | 0.104 | 1.1 | 0.993103 | 0 |  |
| 20 | kda_rec | 0.102 | 0.101 | 0.104 | 0.99 | 0.994741 | 0 |  |
| 21 | kda_rec | 0.145 | 0.134 | 0.136 | 1.1 | 0.989673 | 0 |  |
| 22 | kda_rec | 0.21 | 0.134 | 0.136 | 1.5 | 0.985298 | 9.54e-07 |  |
| 24 | kda_rec | 0.131 | 0.134 | 0.136 | 0.97 | 0.991687 | 0 |  |
| 25 | kda_rec | 0.0873 | 0.134 | 0.136 | 0.64 | 0.996185 | 3.81e-06 |  |
| 26 | kda_rec | 0.127 | 0.134 | 0.136 | 0.94 | 0.992036 | 0 |  |
| 28 | kda_rec | 0.128 | 0.134 | 0.136 | 0.94 | 0.99179 | 3.81e-06 |  |
| 29 | kda_rec | 0.114 | 0.134 | 0.136 | 0.84 | 0.993597 | 0 |  |
| 30 | kda_rec | 0.0471 | 0.134 | 0.136 | 0.35 | 0.999019 | 9.54e-07 |  |
| 32 | kda_rec | 0.11 | 0.134 | 0.136 | 0.81 | 0.993925 | 0 |  |
| 33 | kda_rec | 0.142 | 0.134 | 0.136 | 1 | 0.990027 | 0 |  |
| 34 | kda_rec | 0.0822 | 0.134 | 0.136 | 0.61 | 0.997072 | 0 |  |
| 36 | kda_rec | 0.0434 | 0.134 | 0.136 | 0.32 | 0.999232 | 0 |  |
| 37 | kda_rec | 0.147 | 0.141 | 0.143 | 1 | 0.989152 | 0 |  |
| 38 | kda_rec | 0.159 | 0.155 | 0.155 | 1 | 0.987437 | 8.2e-05 |  |
| 40 | kda_rec | 0.163 | 0.155 | 0.155 | 1.1 | 0.986815 | 0.000122 |  |
| 41 | kda_rec | 0.152 | 0.155 | 0.155 | 0.99 | 0.988364 | 0 |  |
| 42 | kda_rec | 0.183 | 0.155 | 0.155 | 1.2 | 0.983565 | 0 |  |
| 44 | kda_rec | 0.178 | 0.174 | 0.165 | 1 | 0.98455 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.00079 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0215 | 0 | 0.0205 | 1 | 0.999768 | 0.0799 |  |
| 2 | kda_conv | 0.0373 | 0 | 0.0367 | 1 | 0.999303 | 0.0464 |  |
| 4 | kda_conv | 0.03 | 0.0147 | 0.0367 | 0.82 | 0.999552 | 0.0541 |  |
| 5 | kda_conv | 0.0543 | 0.0332 | 0.0548 | 0.99 | 0.998528 | 0.0297 |  |
| 6 | kda_conv | 0.0506 | 0.0332 | 0.0548 | 0.92 | 0.998718 | 0.0323 |  |
| 8 | kda_conv | 0.0835 | 0.0575 | 0.0807 | 1 | 0.996518 | 0.0201 |  |
| 9 | kda_conv | 0.0757 | 0.0575 | 0.0807 | 0.94 | 0.997144 | 0.0222 |  |
| 10 | kda_conv | 0.0565 | 0.0575 | 0.0807 | 0.7 | 0.998406 | 0.0292 |  |
| 12 | kda_conv | 0.0688 | 0.0575 | 0.0807 | 0.85 | 0.997655 | 0.0235 |  |
| 13 | kda_conv | 0.0896 | 0.0718 | 0.0823 | 1.1 | 0.995996 | 0.0191 |  |
| 14 | kda_conv | 0.0837 | 0.0738 | 0.0831 | 1 | 0.996503 | 0.0192 |  |
| 16 | kda_conv | 0.0665 | 0.0738 | 0.0831 | 0.8 | 0.997789 | 0.0232 |  |
| 17 | kda_conv | 0.0835 | 0.0985 | 0.0994 | 0.84 | 0.996508 | 0.0185 |  |
| 18 | kda_conv | 0.113 | 0.123 | 0.104 | 0.92 | 0.993665 | 0.0141 |  |
| 20 | kda_conv | 0.103 | 0.123 | 0.114 | 0.84 | 0.994728 | 0.0161 |  |
| 21 | kda_conv | 0.117 | 0.123 | 0.12 | 0.95 | 0.993144 | 0.0137 |  |
| 22 | kda_conv | 0.132 | 0.136 | 0.126 | 0.97 | 0.991316 | 0.0134 |  |
| 24 | kda_conv | 0.134 | 0.136 | 0.126 | 0.98 | 0.991048 | 0.0138 |  |
| 25 | kda_conv | 0.0785 | 0.136 | 0.126 | 0.58 | 0.996925 | 0.0211 |  |
| 26 | kda_conv | 0.114 | 0.136 | 0.126 | 0.84 | 0.993505 | 0.0144 |  |
| 28 | kda_conv | 0.1 | 0.136 | 0.126 | 0.74 | 0.994985 | 0.0166 |  |
| 29 | kda_conv | 0.0987 | 0.136 | 0.126 | 0.72 | 0.995118 | 0.0171 |  |
| 30 | kda_conv | 0.0931 | 0.136 | 0.126 | 0.68 | 0.995668 | 0.0176 |  |
| 32 | kda_conv | 0.11 | 0.136 | 0.126 | 0.81 | 0.993898 | 0.0144 |  |
| 33 | kda_conv | 0.12 | 0.136 | 0.126 | 0.88 | 0.992762 | 0.0137 |  |
| 34 | kda_conv | 0.0696 | 0.136 | 0.126 | 0.51 | 0.997576 | 0.0208 |  |
| 36 | kda_conv | 0.0709 | 0.136 | 0.126 | 0.52 | 0.997488 | 0.0204 |  |
| 37 | kda_conv | 0.15 | 0.141 | 0.144 | 1 | 0.988763 | 0.0108 |  |
| 38 | kda_conv | 0.149 | 0.143 | 0.159 | 0.93 | 0.989067 | 0.0105 |  |
| 40 | kda_conv | 0.169 | 0.163 | 0.168 | 1 | 0.986113 | 0.00954 |  |
| 41 | kda_conv | 0.16 | 0.163 | 0.168 | 0.95 | 0.987335 | 0.00989 |  |
| 42 | kda_conv | 0.152 | 0.163 | 0.168 | 0.91 | 0.988388 | 0.0108 |  |
| 44 | kda_conv | 0.174 | 0.21 | 0.206 | 0.83 | 0.984872 | 0.00891 |  |


</details>

<details><summary>o_c8192_32768 vs v_full_32768: 1 above noise, 0 patterns</summary>

Per layer, `o_c8192_32768` vs `v_full_32768`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0693 | 0 | 0.0579 | 1.2 | 0.997599 | 0.382 |  |
| 7 | mla | 0.125 | 0.0856 | 0.109 | 1.2 | 0.992182 | 0.242 |  |
| 11 | mla | 0.0992 | 0.0856 | 0.109 | 0.91 | 0.995075 | 0.291 |  |
| 15 | mla | 0.126 | 0.107 | 0.109 | 1.2 | 0.992096 | 0.243 |  |
| 19 | mla | 0.187 | 0.163 | 0.159 | 1.2 | 0.982425 | 0.194 |  |
| 23 | mla | 0.231 | 0.204 | 0.196 | 1.1 | 0.973208 | 0.163 |  |
| 27 | mla | 0.229 | 0.204 | 0.196 | 1.1 | 0.973837 | 0.181 |  |
| 31 | mla | 0.146 | 0.204 | 0.196 | 0.72 | 0.989296 | 0.242 |  |
| 35 | mla | 0.155 | 0.204 | 0.196 | 0.76 | 0.988061 | 0.246 |  |
| 39 | mla | 0.201 | 0.204 | 0.196 | 0.99 | 0.9798 | 0.177 |  |
| 43 | mla | 0.239 | 0.218 | 0.201 | 1.1 | 0.971432 | 0.141 |  |
| 3 | indexer | 0.0428 | 0 | 0.0334 | 1.3 | 0.999085 | 0.548 |  |
| 7 | indexer | 0.049 | 0.0362 | 0.0417 | 1.2 | 0.9988 | 0.511 |  |
| 11 | indexer | 0.0606 | 0.0472 | 0.0487 | 1.2 | 0.998162 | 0.415 |  |
| 15 | indexer | 0.0714 | 0.0552 | 0.0541 | 1.3 | 0.997453 | 0.36 |  |
| 19 | indexer | 0.108 | 0.0892 | 0.0921 | 1.2 | 0.994134 | 0.373 |  |
| 23 | indexer | 0.132 | 0.115 | 0.116 | 1.1 | 0.991348 | 0.336 |  |
| 27 | indexer | 0.123 | 0.115 | 0.116 | 1.1 | 0.992418 | 0.372 |  |
| 31 | indexer | 0.0796 | 0.115 | 0.116 | 0.69 | 0.996828 | 0.385 |  |
| 35 | indexer | 0.0901 | 0.115 | 0.116 | 0.78 | 0.995937 | 0.428 |  |
| 39 | indexer | 0.106 | 0.115 | 0.116 | 0.91 | 0.994434 | 0.399 |  |
| 43 | indexer | 0.111 | 0.115 | 0.116 | 0.96 | 0.99379 | 0.256 |  |
| 0 | kda_rec | 0.00972 | 0 | 0.00305 | 3.2 | 0.999953 | 6.68e-06 | **above noise** |
| 1 | kda_rec | 0.0316 | 0 | 0.0156 | 2 | 0.999526 | 0 |  |
| 2 | kda_rec | 0.0467 | 0 | 0.0232 | 2 | 0.998917 | 0 |  |
| 4 | kda_rec | 0.0409 | 0.0151 | 0.0378 | 1.1 | 0.999164 | 9.54e-07 |  |
| 5 | kda_rec | 0.0655 | 0.0298 | 0.0378 | 1.7 | 0.998411 | 0 |  |
| 6 | kda_rec | 0.0604 | 0.0298 | 0.0426 | 1.4 | 0.998186 | 0 |  |
| 8 | kda_rec | 0.0307 | 0.0298 | 0.0426 | 0.72 | 0.99953 | 0 |  |
| 9 | kda_rec | 0.0716 | 0.0553 | 0.0699 | 1 | 0.997434 | 0 |  |
| 10 | kda_rec | 0.107 | 0.0553 | 0.0699 | 1.5 | 0.99452 | 0 |  |
| 12 | kda_rec | 0.0763 | 0.0615 | 0.0707 | 1.1 | 0.997103 | 0 |  |
| 13 | kda_rec | 0.0432 | 0.0615 | 0.0707 | 0.61 | 0.999069 | 0 |  |
| 14 | kda_rec | 0.0686 | 0.0615 | 0.0707 | 0.97 | 0.997643 | 0.000112 |  |
| 16 | kda_rec | 0.0869 | 0.0615 | 0.0707 | 1.2 | 0.996543 | 0.000118 |  |
| 17 | kda_rec | 0.069 | 0.0615 | 0.0707 | 0.98 | 0.997619 | 0 |  |
| 18 | kda_rec | 0.118 | 0.101 | 0.104 | 1.1 | 0.993103 | 0 |  |
| 20 | kda_rec | 0.102 | 0.101 | 0.104 | 0.99 | 0.994741 | 0 |  |
| 21 | kda_rec | 0.145 | 0.134 | 0.136 | 1.1 | 0.989673 | 0 |  |
| 22 | kda_rec | 0.21 | 0.134 | 0.136 | 1.5 | 0.985298 | 9.54e-07 |  |
| 24 | kda_rec | 0.131 | 0.134 | 0.136 | 0.97 | 0.991687 | 0 |  |
| 25 | kda_rec | 0.0873 | 0.134 | 0.136 | 0.64 | 0.996185 | 3.81e-06 |  |
| 26 | kda_rec | 0.127 | 0.134 | 0.136 | 0.94 | 0.992036 | 0 |  |
| 28 | kda_rec | 0.128 | 0.134 | 0.136 | 0.94 | 0.99179 | 3.81e-06 |  |
| 29 | kda_rec | 0.114 | 0.134 | 0.136 | 0.84 | 0.993597 | 0 |  |
| 30 | kda_rec | 0.0471 | 0.134 | 0.136 | 0.35 | 0.999019 | 9.54e-07 |  |
| 32 | kda_rec | 0.11 | 0.134 | 0.136 | 0.81 | 0.993925 | 0 |  |
| 33 | kda_rec | 0.142 | 0.134 | 0.136 | 1 | 0.990027 | 0 |  |
| 34 | kda_rec | 0.0822 | 0.134 | 0.136 | 0.61 | 0.997072 | 0 |  |
| 36 | kda_rec | 0.0434 | 0.134 | 0.136 | 0.32 | 0.999232 | 0 |  |
| 37 | kda_rec | 0.147 | 0.141 | 0.143 | 1 | 0.989152 | 0 |  |
| 38 | kda_rec | 0.159 | 0.155 | 0.155 | 1 | 0.987437 | 8.2e-05 |  |
| 40 | kda_rec | 0.163 | 0.155 | 0.155 | 1.1 | 0.986815 | 0.000122 |  |
| 41 | kda_rec | 0.152 | 0.155 | 0.155 | 0.99 | 0.988364 | 0 |  |
| 42 | kda_rec | 0.183 | 0.155 | 0.155 | 1.2 | 0.983565 | 0 |  |
| 44 | kda_rec | 0.178 | 0.174 | 0.165 | 1 | 0.98455 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.00079 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0215 | 0 | 0.0205 | 1 | 0.999768 | 0.0799 |  |
| 2 | kda_conv | 0.0373 | 0 | 0.0367 | 1 | 0.999303 | 0.0464 |  |
| 4 | kda_conv | 0.03 | 0.0147 | 0.0367 | 0.82 | 0.999552 | 0.0541 |  |
| 5 | kda_conv | 0.0543 | 0.0332 | 0.0548 | 0.99 | 0.998528 | 0.0297 |  |
| 6 | kda_conv | 0.0506 | 0.0332 | 0.0548 | 0.92 | 0.998718 | 0.0323 |  |
| 8 | kda_conv | 0.0835 | 0.0575 | 0.0807 | 1 | 0.996518 | 0.0201 |  |
| 9 | kda_conv | 0.0757 | 0.0575 | 0.0807 | 0.94 | 0.997144 | 0.0222 |  |
| 10 | kda_conv | 0.0565 | 0.0575 | 0.0807 | 0.7 | 0.998406 | 0.0292 |  |
| 12 | kda_conv | 0.0688 | 0.0575 | 0.0807 | 0.85 | 0.997655 | 0.0235 |  |
| 13 | kda_conv | 0.0896 | 0.0718 | 0.0823 | 1.1 | 0.995996 | 0.0191 |  |
| 14 | kda_conv | 0.0837 | 0.0738 | 0.0831 | 1 | 0.996503 | 0.0192 |  |
| 16 | kda_conv | 0.0665 | 0.0738 | 0.0831 | 0.8 | 0.997789 | 0.0232 |  |
| 17 | kda_conv | 0.0835 | 0.0985 | 0.0994 | 0.84 | 0.996508 | 0.0185 |  |
| 18 | kda_conv | 0.113 | 0.123 | 0.104 | 0.92 | 0.993665 | 0.0141 |  |
| 20 | kda_conv | 0.103 | 0.123 | 0.114 | 0.84 | 0.994728 | 0.0161 |  |
| 21 | kda_conv | 0.117 | 0.123 | 0.12 | 0.95 | 0.993144 | 0.0137 |  |
| 22 | kda_conv | 0.132 | 0.136 | 0.126 | 0.97 | 0.991316 | 0.0134 |  |
| 24 | kda_conv | 0.134 | 0.136 | 0.126 | 0.98 | 0.991048 | 0.0138 |  |
| 25 | kda_conv | 0.0785 | 0.136 | 0.126 | 0.58 | 0.996925 | 0.0211 |  |
| 26 | kda_conv | 0.114 | 0.136 | 0.126 | 0.84 | 0.993505 | 0.0144 |  |
| 28 | kda_conv | 0.1 | 0.136 | 0.126 | 0.74 | 0.994985 | 0.0166 |  |
| 29 | kda_conv | 0.0987 | 0.136 | 0.126 | 0.72 | 0.995118 | 0.0171 |  |
| 30 | kda_conv | 0.0931 | 0.136 | 0.126 | 0.68 | 0.995668 | 0.0176 |  |
| 32 | kda_conv | 0.11 | 0.136 | 0.126 | 0.81 | 0.993898 | 0.0144 |  |
| 33 | kda_conv | 0.12 | 0.136 | 0.126 | 0.88 | 0.992762 | 0.0137 |  |
| 34 | kda_conv | 0.0696 | 0.136 | 0.126 | 0.51 | 0.997576 | 0.0208 |  |
| 36 | kda_conv | 0.0709 | 0.136 | 0.126 | 0.52 | 0.997488 | 0.0204 |  |
| 37 | kda_conv | 0.15 | 0.141 | 0.144 | 1 | 0.988763 | 0.0108 |  |
| 38 | kda_conv | 0.149 | 0.143 | 0.159 | 0.93 | 0.989067 | 0.0105 |  |
| 40 | kda_conv | 0.169 | 0.163 | 0.168 | 1 | 0.986113 | 0.00954 |  |
| 41 | kda_conv | 0.16 | 0.163 | 0.168 | 0.95 | 0.987335 | 0.00989 |  |
| 42 | kda_conv | 0.152 | 0.163 | 0.168 | 0.91 | 0.988388 | 0.0108 |  |
| 44 | kda_conv | 0.174 | 0.21 | 0.206 | 0.83 | 0.984872 | 0.00891 |  |


</details>

<details><summary>o_c8001_32768 vs v_full_32768: 1 above noise, 0 patterns</summary>

Per layer, `o_c8001_32768` vs `v_full_32768`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0694 | 0 | 0.0579 | 1.2 | 0.99759 | 0.381 |  |
| 7 | mla | 0.125 | 0.0856 | 0.109 | 1.2 | 0.992164 | 0.242 |  |
| 11 | mla | 0.0991 | 0.0856 | 0.109 | 0.91 | 0.995088 | 0.292 |  |
| 15 | mla | 0.126 | 0.107 | 0.109 | 1.2 | 0.99211 | 0.244 |  |
| 19 | mla | 0.186 | 0.163 | 0.159 | 1.1 | 0.982728 | 0.194 |  |
| 23 | mla | 0.231 | 0.204 | 0.196 | 1.1 | 0.973356 | 0.163 |  |
| 27 | mla | 0.228 | 0.204 | 0.196 | 1.1 | 0.97401 | 0.181 |  |
| 31 | mla | 0.147 | 0.204 | 0.196 | 0.72 | 0.989206 | 0.242 |  |
| 35 | mla | 0.156 | 0.204 | 0.196 | 0.76 | 0.987905 | 0.245 |  |
| 39 | mla | 0.202 | 0.204 | 0.196 | 0.99 | 0.979665 | 0.177 |  |
| 43 | mla | 0.239 | 0.218 | 0.201 | 1.1 | 0.971397 | 0.141 |  |
| 3 | indexer | 0.043 | 0 | 0.0334 | 1.3 | 0.999079 | 0.548 |  |
| 7 | indexer | 0.049 | 0.0362 | 0.0417 | 1.2 | 0.998799 | 0.511 |  |
| 11 | indexer | 0.0601 | 0.0472 | 0.0487 | 1.2 | 0.998191 | 0.419 |  |
| 15 | indexer | 0.0711 | 0.0552 | 0.0541 | 1.3 | 0.997473 | 0.363 |  |
| 19 | indexer | 0.107 | 0.0892 | 0.0921 | 1.2 | 0.994277 | 0.373 |  |
| 23 | indexer | 0.129 | 0.115 | 0.116 | 1.1 | 0.991641 | 0.336 |  |
| 27 | indexer | 0.12 | 0.115 | 0.116 | 1 | 0.992803 | 0.374 |  |
| 31 | indexer | 0.0791 | 0.115 | 0.116 | 0.68 | 0.996873 | 0.385 |  |
| 35 | indexer | 0.0904 | 0.115 | 0.116 | 0.78 | 0.995914 | 0.427 |  |
| 39 | indexer | 0.107 | 0.115 | 0.116 | 0.93 | 0.994259 | 0.397 |  |
| 43 | indexer | 0.111 | 0.115 | 0.116 | 0.96 | 0.993873 | 0.257 |  |
| 0 | kda_rec | 0.00977 | 0 | 0.00305 | 3.2 | 0.999952 | 2.86e-06 | **above noise** |
| 1 | kda_rec | 0.0312 | 0 | 0.0156 | 2 | 0.999539 | 9.54e-07 |  |
| 2 | kda_rec | 0.0452 | 0 | 0.0232 | 1.9 | 0.998988 | 0 |  |
| 4 | kda_rec | 0.0417 | 0.0151 | 0.0378 | 1.1 | 0.999136 | 0 |  |
| 5 | kda_rec | 0.0641 | 0.0298 | 0.0378 | 1.7 | 0.998376 | 9.54e-07 |  |
| 6 | kda_rec | 0.0609 | 0.0298 | 0.0426 | 1.4 | 0.998161 | 0 |  |
| 8 | kda_rec | 0.0298 | 0.0298 | 0.0426 | 0.7 | 0.999555 | 9.54e-07 |  |
| 9 | kda_rec | 0.0731 | 0.0553 | 0.0699 | 1 | 0.997325 | 0 |  |
| 10 | kda_rec | 0.107 | 0.0553 | 0.0699 | 1.5 | 0.994567 | 1.91e-06 |  |
| 12 | kda_rec | 0.0748 | 0.0615 | 0.0707 | 1.1 | 0.997216 | 0 |  |
| 13 | kda_rec | 0.045 | 0.0615 | 0.0707 | 0.64 | 0.998985 | 0 |  |
| 14 | kda_rec | 0.0663 | 0.0615 | 0.0707 | 0.94 | 0.997797 | 0.000111 |  |
| 16 | kda_rec | 0.0855 | 0.0615 | 0.0707 | 1.2 | 0.996597 | 0.00012 |  |
| 17 | kda_rec | 0.0667 | 0.0615 | 0.0707 | 0.94 | 0.997776 | 0 |  |
| 18 | kda_rec | 0.117 | 0.101 | 0.104 | 1.1 | 0.993213 | 0 |  |
| 20 | kda_rec | 0.104 | 0.101 | 0.104 | 1 | 0.994588 | 0 |  |
| 21 | kda_rec | 0.153 | 0.134 | 0.136 | 1.1 | 0.988561 | 0 |  |
| 22 | kda_rec | 0.207 | 0.134 | 0.136 | 1.5 | 0.986396 | 0 |  |
| 24 | kda_rec | 0.132 | 0.134 | 0.136 | 0.97 | 0.991675 | 0 |  |
| 25 | kda_rec | 0.0876 | 0.134 | 0.136 | 0.65 | 0.996157 | 3.81e-06 |  |
| 26 | kda_rec | 0.132 | 0.134 | 0.136 | 0.97 | 0.991448 | 0 |  |
| 28 | kda_rec | 0.13 | 0.134 | 0.136 | 0.96 | 0.991566 | 3.81e-06 |  |
| 29 | kda_rec | 0.113 | 0.134 | 0.136 | 0.83 | 0.993679 | 0 |  |
| 30 | kda_rec | 0.0471 | 0.134 | 0.136 | 0.35 | 0.999017 | 9.54e-07 |  |
| 32 | kda_rec | 0.111 | 0.134 | 0.136 | 0.82 | 0.993881 | 9.54e-07 |  |
| 33 | kda_rec | 0.139 | 0.134 | 0.136 | 1 | 0.990433 | 9.54e-07 |  |
| 34 | kda_rec | 0.0799 | 0.134 | 0.136 | 0.59 | 0.997232 | 0 |  |
| 36 | kda_rec | 0.0434 | 0.134 | 0.136 | 0.32 | 0.999211 | 0 |  |
| 37 | kda_rec | 0.144 | 0.141 | 0.143 | 1 | 0.98965 | 0 |  |
| 38 | kda_rec | 0.159 | 0.155 | 0.155 | 1 | 0.987358 | 8.2e-05 |  |
| 40 | kda_rec | 0.161 | 0.155 | 0.155 | 1 | 0.987153 | 0.000122 |  |
| 41 | kda_rec | 0.15 | 0.155 | 0.155 | 0.97 | 0.988739 | 0 |  |
| 42 | kda_rec | 0.184 | 0.155 | 0.155 | 1.2 | 0.983315 | 0 |  |
| 44 | kda_rec | 0.182 | 0.174 | 0.165 | 1 | 0.983633 | 0 |  |
| 0 | kda_conv | 0.00079 | 0 | 0.00079 | 1 | 1 | 0.9 |  |
| 1 | kda_conv | 0.0209 | 0 | 0.0205 | 1 | 0.999781 | 0.0833 |  |
| 2 | kda_conv | 0.0369 | 0 | 0.0367 | 1 | 0.999318 | 0.0479 |  |
| 4 | kda_conv | 0.0283 | 0.0147 | 0.0367 | 0.77 | 0.999601 | 0.061 |  |
| 5 | kda_conv | 0.0563 | 0.0332 | 0.0548 | 1 | 0.998415 | 0.031 |  |
| 6 | kda_conv | 0.0508 | 0.0332 | 0.0548 | 0.93 | 0.998711 | 0.0325 |  |
| 8 | kda_conv | 0.0815 | 0.0575 | 0.0807 | 1 | 0.996688 | 0.0204 |  |
| 9 | kda_conv | 0.0775 | 0.0575 | 0.0807 | 0.96 | 0.997003 | 0.0213 |  |
| 10 | kda_conv | 0.0582 | 0.0575 | 0.0807 | 0.72 | 0.998308 | 0.029 |  |
| 12 | kda_conv | 0.0618 | 0.0575 | 0.0807 | 0.77 | 0.998093 | 0.0269 |  |
| 13 | kda_conv | 0.0768 | 0.0718 | 0.0823 | 0.93 | 0.997059 | 0.0209 |  |
| 14 | kda_conv | 0.0756 | 0.0738 | 0.0831 | 0.91 | 0.997147 | 0.0208 |  |
| 16 | kda_conv | 0.0638 | 0.0738 | 0.0831 | 0.77 | 0.997962 | 0.0243 |  |
| 17 | kda_conv | 0.102 | 0.0985 | 0.0994 | 1 | 0.994804 | 0.0167 |  |
| 18 | kda_conv | 0.116 | 0.123 | 0.104 | 0.94 | 0.993302 | 0.0143 |  |
| 20 | kda_conv | 0.117 | 0.123 | 0.114 | 0.95 | 0.99309 | 0.0141 |  |
| 21 | kda_conv | 0.136 | 0.123 | 0.12 | 1.1 | 0.990718 | 0.012 |  |
| 22 | kda_conv | 0.134 | 0.136 | 0.126 | 0.98 | 0.990992 | 0.0126 |  |
| 24 | kda_conv | 0.139 | 0.136 | 0.126 | 1 | 0.990356 | 0.0119 |  |
| 25 | kda_conv | 0.0829 | 0.136 | 0.126 | 0.61 | 0.996606 | 0.0198 |  |
| 26 | kda_conv | 0.118 | 0.136 | 0.126 | 0.86 | 0.993043 | 0.0145 |  |
| 28 | kda_conv | 0.1 | 0.136 | 0.126 | 0.73 | 0.99499 | 0.016 |  |
| 29 | kda_conv | 0.0952 | 0.136 | 0.126 | 0.7 | 0.995463 | 0.0175 |  |
| 30 | kda_conv | 0.097 | 0.136 | 0.126 | 0.71 | 0.99529 | 0.0164 |  |
| 32 | kda_conv | 0.106 | 0.136 | 0.126 | 0.78 | 0.994372 | 0.0154 |  |
| 33 | kda_conv | 0.112 | 0.136 | 0.126 | 0.83 | 0.993659 | 0.0149 |  |
| 34 | kda_conv | 0.0694 | 0.136 | 0.126 | 0.51 | 0.997589 | 0.0203 |  |
| 36 | kda_conv | 0.07 | 0.136 | 0.126 | 0.51 | 0.997545 | 0.0205 |  |
| 37 | kda_conv | 0.143 | 0.141 | 0.144 | 0.99 | 0.989776 | 0.0109 |  |
| 38 | kda_conv | 0.161 | 0.143 | 0.159 | 1 | 0.987163 | 0.00979 |  |
| 40 | kda_conv | 0.167 | 0.163 | 0.168 | 0.99 | 0.986161 | 0.00878 |  |
| 41 | kda_conv | 0.164 | 0.163 | 0.168 | 0.98 | 0.986535 | 0.0102 |  |
| 42 | kda_conv | 0.157 | 0.163 | 0.168 | 0.93 | 0.987806 | 0.00898 |  |
| 44 | kda_conv | 0.185 | 0.21 | 0.206 | 0.88 | 0.983141 | 0.00803 |  |


</details>

<details><summary>o_full_33792 vs v_full_33792: 0 above noise, 0 patterns</summary>

Per layer, `o_full_33792` vs `v_full_33792`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0695 | 0 | 0.0579 | 1.2 | 0.997587 | 0.381 |  |
| 7 | mla | 0.125 | 0.0856 | 0.109 | 1.2 | 0.992162 | 0.242 |  |
| 11 | mla | 0.099 | 0.0856 | 0.109 | 0.91 | 0.995095 | 0.292 |  |
| 15 | mla | 0.125 | 0.107 | 0.109 | 1.2 | 0.992153 | 0.244 |  |
| 19 | mla | 0.187 | 0.163 | 0.159 | 1.1 | 0.982534 | 0.194 |  |
| 23 | mla | 0.23 | 0.204 | 0.196 | 1.1 | 0.973459 | 0.163 |  |
| 27 | mla | 0.227 | 0.204 | 0.196 | 1.1 | 0.974203 | 0.181 |  |
| 31 | mla | 0.145 | 0.204 | 0.196 | 0.71 | 0.989495 | 0.243 |  |
| 35 | mla | 0.153 | 0.204 | 0.196 | 0.75 | 0.988368 | 0.247 |  |
| 39 | mla | 0.199 | 0.204 | 0.196 | 0.98 | 0.980196 | 0.177 |  |
| 43 | mla | 0.237 | 0.218 | 0.201 | 1.1 | 0.97182 | 0.141 |  |
| 3 | indexer | 0.0433 | 0 | 0.0334 | 1.3 | 0.999066 | 0.545 |  |
| 7 | indexer | 0.0492 | 0.0362 | 0.0417 | 1.2 | 0.998791 | 0.51 |  |
| 11 | indexer | 0.0605 | 0.0472 | 0.0487 | 1.2 | 0.998167 | 0.415 |  |
| 15 | indexer | 0.0714 | 0.0552 | 0.0541 | 1.3 | 0.99745 | 0.359 |  |
| 19 | indexer | 0.109 | 0.0892 | 0.0921 | 1.2 | 0.994074 | 0.374 |  |
| 23 | indexer | 0.133 | 0.115 | 0.116 | 1.1 | 0.99121 | 0.34 |  |
| 27 | indexer | 0.125 | 0.115 | 0.116 | 1.1 | 0.992221 | 0.377 |  |
| 31 | indexer | 0.0775 | 0.115 | 0.116 | 0.67 | 0.997 | 0.389 |  |
| 35 | indexer | 0.0879 | 0.115 | 0.116 | 0.76 | 0.996138 | 0.432 |  |
| 39 | indexer | 0.104 | 0.115 | 0.116 | 0.89 | 0.99463 | 0.401 |  |
| 43 | indexer | 0.11 | 0.115 | 0.116 | 0.95 | 0.993911 | 0.258 |  |
| 0 | kda_rec | 0.00902 | 0 | 0.00305 | 3 | 0.999959 | 3.81e-06 |  |
| 1 | kda_rec | 0.0273 | 0 | 0.0156 | 1.7 | 0.999647 | 9.54e-07 |  |
| 2 | kda_rec | 0.0442 | 0 | 0.0232 | 1.9 | 0.999054 | 0 |  |
| 4 | kda_rec | 0.0413 | 0.0151 | 0.0378 | 1.1 | 0.999149 | 9.54e-07 |  |
| 5 | kda_rec | 0.0486 | 0.0298 | 0.0378 | 1.3 | 0.998963 | 9.54e-07 |  |
| 6 | kda_rec | 0.0602 | 0.0298 | 0.0426 | 1.4 | 0.998198 | 0 |  |
| 8 | kda_rec | 0.0295 | 0.0298 | 0.0426 | 0.69 | 0.999567 | 0 |  |
| 9 | kda_rec | 0.073 | 0.0553 | 0.0699 | 1 | 0.997334 | 0 |  |
| 10 | kda_rec | 0.113 | 0.0553 | 0.0699 | 1.6 | 0.99394 | 0 |  |
| 12 | kda_rec | 0.0732 | 0.0615 | 0.0707 | 1 | 0.997333 | 0 |  |
| 13 | kda_rec | 0.0432 | 0.0615 | 0.0707 | 0.61 | 0.999078 | 0 |  |
| 14 | kda_rec | 0.0665 | 0.0615 | 0.0707 | 0.94 | 0.997806 | 0.000211 |  |
| 16 | kda_rec | 0.0903 | 0.0615 | 0.0707 | 1.3 | 0.996312 | 0.000115 |  |
| 17 | kda_rec | 0.0855 | 0.0615 | 0.0707 | 1.2 | 0.996337 | 0 |  |
| 18 | kda_rec | 0.124 | 0.101 | 0.104 | 1.2 | 0.992444 | 0 |  |
| 20 | kda_rec | 0.116 | 0.101 | 0.104 | 1.1 | 0.993225 | 0 |  |
| 21 | kda_rec | 0.165 | 0.134 | 0.136 | 1.2 | 0.986688 | 0 |  |
| 22 | kda_rec | 0.219 | 0.134 | 0.136 | 1.6 | 0.984497 | 0 |  |
| 24 | kda_rec | 0.139 | 0.134 | 0.136 | 1 | 0.99075 | 5.72e-06 |  |
| 25 | kda_rec | 0.0936 | 0.134 | 0.136 | 0.69 | 0.995638 | 6.68e-06 |  |
| 26 | kda_rec | 0.137 | 0.134 | 0.136 | 1 | 0.99089 | 0 |  |
| 28 | kda_rec | 0.133 | 0.134 | 0.136 | 0.98 | 0.991243 | 1.91e-06 |  |
| 29 | kda_rec | 0.122 | 0.134 | 0.136 | 0.9 | 0.992695 | 0 |  |
| 30 | kda_rec | 0.047 | 0.134 | 0.136 | 0.35 | 0.998996 | 0 |  |
| 32 | kda_rec | 0.105 | 0.134 | 0.136 | 0.77 | 0.994536 | 0 |  |
| 33 | kda_rec | 0.122 | 0.134 | 0.136 | 0.9 | 0.992657 | 0 |  |
| 34 | kda_rec | 0.0776 | 0.134 | 0.136 | 0.57 | 0.997337 | 0 |  |
| 36 | kda_rec | 0.0456 | 0.134 | 0.136 | 0.34 | 0.999189 | 0 |  |
| 37 | kda_rec | 0.126 | 0.141 | 0.143 | 0.88 | 0.992021 | 0 |  |
| 38 | kda_rec | 0.158 | 0.155 | 0.155 | 1 | 0.987555 | 9.35e-05 |  |
| 40 | kda_rec | 0.162 | 0.155 | 0.155 | 1 | 0.98709 | 0.000122 |  |
| 41 | kda_rec | 0.157 | 0.155 | 0.155 | 1 | 0.987572 | 0 |  |
| 42 | kda_rec | 0.18 | 0.155 | 0.155 | 1.2 | 0.98404 | 0 |  |
| 44 | kda_rec | 0.178 | 0.174 | 0.165 | 1 | 0.984383 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.00079 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0195 | 0 | 0.0205 | 0.95 | 0.999812 | 0.0855 |  |
| 2 | kda_conv | 0.0311 | 0 | 0.0367 | 0.85 | 0.999516 | 0.0544 |  |
| 4 | kda_conv | 0.0251 | 0.0147 | 0.0367 | 0.68 | 0.999687 | 0.0662 |  |
| 5 | kda_conv | 0.0476 | 0.0332 | 0.0548 | 0.87 | 0.998865 | 0.0355 |  |
| 6 | kda_conv | 0.0446 | 0.0332 | 0.0548 | 0.81 | 0.999005 | 0.0364 |  |
| 8 | kda_conv | 0.0729 | 0.0575 | 0.0807 | 0.9 | 0.997344 | 0.0217 |  |
| 9 | kda_conv | 0.076 | 0.0575 | 0.0807 | 0.94 | 0.997105 | 0.0213 |  |
| 10 | kda_conv | 0.0709 | 0.0575 | 0.0807 | 0.88 | 0.997491 | 0.0258 |  |
| 12 | kda_conv | 0.0688 | 0.0575 | 0.0807 | 0.85 | 0.997642 | 0.0251 |  |
| 13 | kda_conv | 0.0934 | 0.0718 | 0.0823 | 1.1 | 0.995644 | 0.0188 |  |
| 14 | kda_conv | 0.0916 | 0.0738 | 0.0831 | 1.1 | 0.995857 | 0.0181 |  |
| 16 | kda_conv | 0.0829 | 0.0738 | 0.0831 | 1 | 0.99657 | 0.0186 |  |
| 17 | kda_conv | 0.107 | 0.0985 | 0.0994 | 1.1 | 0.994291 | 0.0154 |  |
| 18 | kda_conv | 0.12 | 0.123 | 0.104 | 0.97 | 0.992895 | 0.0151 |  |
| 20 | kda_conv | 0.153 | 0.123 | 0.114 | 1.2 | 0.988196 | 0.0107 |  |
| 21 | kda_conv | 0.154 | 0.123 | 0.12 | 1.3 | 0.98817 | 0.0105 |  |
| 22 | kda_conv | 0.154 | 0.136 | 0.126 | 1.1 | 0.988182 | 0.0113 |  |
| 24 | kda_conv | 0.167 | 0.136 | 0.126 | 1.2 | 0.986116 | 0.0114 |  |
| 25 | kda_conv | 0.108 | 0.136 | 0.126 | 0.79 | 0.994165 | 0.0168 |  |
| 26 | kda_conv | 0.145 | 0.136 | 0.126 | 1.1 | 0.98959 | 0.013 |  |
| 28 | kda_conv | 0.11 | 0.136 | 0.126 | 0.81 | 0.993965 | 0.0146 |  |
| 29 | kda_conv | 0.109 | 0.136 | 0.126 | 0.8 | 0.994007 | 0.0153 |  |
| 30 | kda_conv | 0.102 | 0.136 | 0.126 | 0.75 | 0.994762 | 0.0157 |  |
| 32 | kda_conv | 0.0931 | 0.136 | 0.126 | 0.68 | 0.995668 | 0.0178 |  |
| 33 | kda_conv | 0.0965 | 0.136 | 0.126 | 0.71 | 0.995331 | 0.0176 |  |
| 34 | kda_conv | 0.0683 | 0.136 | 0.126 | 0.5 | 0.997666 | 0.0219 |  |
| 36 | kda_conv | 0.0652 | 0.136 | 0.126 | 0.48 | 0.99787 | 0.0217 |  |
| 37 | kda_conv | 0.111 | 0.141 | 0.144 | 0.77 | 0.993845 | 0.0149 |  |
| 38 | kda_conv | 0.116 | 0.143 | 0.159 | 0.73 | 0.993258 | 0.0134 |  |
| 40 | kda_conv | 0.142 | 0.163 | 0.168 | 0.84 | 0.989947 | 0.0106 |  |
| 41 | kda_conv | 0.166 | 0.163 | 0.168 | 0.99 | 0.986112 | 0.00963 |  |
| 42 | kda_conv | 0.152 | 0.163 | 0.168 | 0.9 | 0.988458 | 0.0108 |  |
| 44 | kda_conv | 0.187 | 0.21 | 0.206 | 0.89 | 0.982366 | 0.00788 |  |


</details>

<details><summary>o_prefix_32771_1021 vs v_full_33792: 0 above noise, 0 patterns</summary>

Per layer, `o_prefix_32771_1021` vs `v_full_33792`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.0695 | 0 | 0.0579 | 1.2 | 0.997586 | 0.381 |  |
| 7 | mla | 0.125 | 0.0856 | 0.109 | 1.2 | 0.992161 | 0.242 |  |
| 11 | mla | 0.099 | 0.0856 | 0.109 | 0.91 | 0.995095 | 0.292 |  |
| 15 | mla | 0.125 | 0.107 | 0.109 | 1.2 | 0.992151 | 0.244 |  |
| 19 | mla | 0.187 | 0.163 | 0.159 | 1.1 | 0.982539 | 0.194 |  |
| 23 | mla | 0.23 | 0.204 | 0.196 | 1.1 | 0.973517 | 0.163 |  |
| 27 | mla | 0.227 | 0.204 | 0.196 | 1.1 | 0.974242 | 0.181 |  |
| 31 | mla | 0.145 | 0.204 | 0.196 | 0.71 | 0.989512 | 0.243 |  |
| 35 | mla | 0.152 | 0.204 | 0.196 | 0.75 | 0.988385 | 0.247 |  |
| 39 | mla | 0.199 | 0.204 | 0.196 | 0.98 | 0.980207 | 0.177 |  |
| 43 | mla | 0.237 | 0.218 | 0.201 | 1.1 | 0.971819 | 0.141 |  |
| 3 | indexer | 0.0433 | 0 | 0.0334 | 1.3 | 0.999066 | 0.545 |  |
| 7 | indexer | 0.0492 | 0.0362 | 0.0417 | 1.2 | 0.998791 | 0.51 |  |
| 11 | indexer | 0.0605 | 0.0472 | 0.0487 | 1.2 | 0.998167 | 0.415 |  |
| 15 | indexer | 0.0714 | 0.0552 | 0.0541 | 1.3 | 0.997449 | 0.36 |  |
| 19 | indexer | 0.109 | 0.0892 | 0.0921 | 1.2 | 0.994118 | 0.374 |  |
| 23 | indexer | 0.132 | 0.115 | 0.116 | 1.1 | 0.991269 | 0.34 |  |
| 27 | indexer | 0.125 | 0.115 | 0.116 | 1.1 | 0.992245 | 0.377 |  |
| 31 | indexer | 0.0774 | 0.115 | 0.116 | 0.67 | 0.997003 | 0.389 |  |
| 35 | indexer | 0.0878 | 0.115 | 0.116 | 0.76 | 0.996141 | 0.433 |  |
| 39 | indexer | 0.104 | 0.115 | 0.116 | 0.9 | 0.994618 | 0.401 |  |
| 43 | indexer | 0.11 | 0.115 | 0.116 | 0.95 | 0.993911 | 0.258 |  |
| 0 | kda_rec | 0.00903 | 0 | 0.00305 | 3 | 0.999959 | 4.77e-06 |  |
| 1 | kda_rec | 0.0274 | 0 | 0.0156 | 1.8 | 0.999641 | 0 |  |
| 2 | kda_rec | 0.0449 | 0 | 0.0232 | 1.9 | 0.999017 | 1.91e-06 |  |
| 4 | kda_rec | 0.0417 | 0.0151 | 0.0378 | 1.1 | 0.999129 | 9.54e-07 |  |
| 5 | kda_rec | 0.0485 | 0.0298 | 0.0378 | 1.3 | 0.998991 | 0 |  |
| 6 | kda_rec | 0.0602 | 0.0298 | 0.0426 | 1.4 | 0.998199 | 0 |  |
| 8 | kda_rec | 0.0297 | 0.0298 | 0.0426 | 0.7 | 0.999563 | 9.54e-07 |  |
| 9 | kda_rec | 0.0713 | 0.0553 | 0.0699 | 1 | 0.99746 | 0 |  |
| 10 | kda_rec | 0.113 | 0.0553 | 0.0699 | 1.6 | 0.993983 | 0 |  |
| 12 | kda_rec | 0.0728 | 0.0615 | 0.0707 | 1 | 0.997364 | 0 |  |
| 13 | kda_rec | 0.045 | 0.0615 | 0.0707 | 0.64 | 0.998994 | 0 |  |
| 14 | kda_rec | 0.0679 | 0.0615 | 0.0707 | 0.96 | 0.997693 | 0.000211 |  |
| 16 | kda_rec | 0.0907 | 0.0615 | 0.0707 | 1.3 | 0.996211 | 0.000117 |  |
| 17 | kda_rec | 0.0839 | 0.0615 | 0.0707 | 1.2 | 0.996479 | 0 |  |
| 18 | kda_rec | 0.125 | 0.101 | 0.104 | 1.2 | 0.992258 | 0 |  |
| 20 | kda_rec | 0.114 | 0.101 | 0.104 | 1.1 | 0.993471 | 0 |  |
| 21 | kda_rec | 0.162 | 0.134 | 0.136 | 1.2 | 0.987154 | 0 |  |
| 22 | kda_rec | 0.219 | 0.134 | 0.136 | 1.6 | 0.983839 | 0 |  |
| 24 | kda_rec | 0.138 | 0.134 | 0.136 | 1 | 0.991006 | 5.72e-06 |  |
| 25 | kda_rec | 0.0905 | 0.134 | 0.136 | 0.67 | 0.995908 | 6.68e-06 |  |
| 26 | kda_rec | 0.133 | 0.134 | 0.136 | 0.98 | 0.991456 | 0 |  |
| 28 | kda_rec | 0.131 | 0.134 | 0.136 | 0.97 | 0.991425 | 1.91e-06 |  |
| 29 | kda_rec | 0.12 | 0.134 | 0.136 | 0.89 | 0.992826 | 0 |  |
| 30 | kda_rec | 0.0466 | 0.134 | 0.136 | 0.34 | 0.999013 | 0 |  |
| 32 | kda_rec | 0.1 | 0.134 | 0.136 | 0.74 | 0.994984 | 0 |  |
| 33 | kda_rec | 0.122 | 0.134 | 0.136 | 0.9 | 0.99266 | 0 |  |
| 34 | kda_rec | 0.0776 | 0.134 | 0.136 | 0.57 | 0.997323 | 0 |  |
| 36 | kda_rec | 0.0456 | 0.134 | 0.136 | 0.34 | 0.999213 | 0 |  |
| 37 | kda_rec | 0.125 | 0.141 | 0.143 | 0.87 | 0.992209 | 0 |  |
| 38 | kda_rec | 0.157 | 0.155 | 0.155 | 1 | 0.98762 | 9.35e-05 |  |
| 40 | kda_rec | 0.161 | 0.155 | 0.155 | 1 | 0.987239 | 0.000122 |  |
| 41 | kda_rec | 0.157 | 0.155 | 0.155 | 1 | 0.987598 | 0 |  |
| 42 | kda_rec | 0.182 | 0.155 | 0.155 | 1.2 | 0.983739 | 0 |  |
| 44 | kda_rec | 0.182 | 0.174 | 0.165 | 1 | 0.983798 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.00079 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0207 | 0 | 0.0205 | 1 | 0.999786 | 0.0798 |  |
| 2 | kda_conv | 0.0314 | 0 | 0.0367 | 0.86 | 0.999507 | 0.0549 |  |
| 4 | kda_conv | 0.0252 | 0.0147 | 0.0367 | 0.69 | 0.999682 | 0.0655 |  |
| 5 | kda_conv | 0.0492 | 0.0332 | 0.0548 | 0.9 | 0.998791 | 0.0338 |  |
| 6 | kda_conv | 0.0452 | 0.0332 | 0.0548 | 0.82 | 0.99898 | 0.0369 |  |
| 8 | kda_conv | 0.075 | 0.0575 | 0.0807 | 0.93 | 0.997189 | 0.0212 |  |
| 9 | kda_conv | 0.0713 | 0.0575 | 0.0807 | 0.88 | 0.997464 | 0.0225 |  |
| 10 | kda_conv | 0.0621 | 0.0575 | 0.0807 | 0.77 | 0.998077 | 0.0264 |  |
| 12 | kda_conv | 0.0615 | 0.0575 | 0.0807 | 0.76 | 0.998121 | 0.0275 |  |
| 13 | kda_conv | 0.0838 | 0.0718 | 0.0823 | 1 | 0.996507 | 0.0193 |  |
| 14 | kda_conv | 0.105 | 0.0738 | 0.0831 | 1.3 | 0.994501 | 0.0163 |  |
| 16 | kda_conv | 0.083 | 0.0738 | 0.0831 | 1 | 0.996557 | 0.019 |  |
| 17 | kda_conv | 0.0949 | 0.0985 | 0.0994 | 0.95 | 0.995535 | 0.0164 |  |
| 18 | kda_conv | 0.108 | 0.123 | 0.104 | 0.88 | 0.994191 | 0.0154 |  |
| 20 | kda_conv | 0.14 | 0.123 | 0.114 | 1.1 | 0.99019 | 0.0117 |  |
| 21 | kda_conv | 0.141 | 0.123 | 0.12 | 1.1 | 0.990033 | 0.0118 |  |
| 22 | kda_conv | 0.137 | 0.136 | 0.126 | 1 | 0.990658 | 0.0128 |  |
| 24 | kda_conv | 0.137 | 0.136 | 0.126 | 1 | 0.990717 | 0.0128 |  |
| 25 | kda_conv | 0.0908 | 0.136 | 0.126 | 0.67 | 0.995874 | 0.0187 |  |
| 26 | kda_conv | 0.119 | 0.136 | 0.126 | 0.87 | 0.99306 | 0.0144 |  |
| 28 | kda_conv | 0.0972 | 0.136 | 0.126 | 0.71 | 0.995274 | 0.0167 |  |
| 29 | kda_conv | 0.101 | 0.136 | 0.126 | 0.74 | 0.994864 | 0.0174 |  |
| 30 | kda_conv | 0.0945 | 0.136 | 0.126 | 0.69 | 0.995529 | 0.0171 |  |
| 32 | kda_conv | 0.0879 | 0.136 | 0.126 | 0.65 | 0.99613 | 0.0184 |  |
| 33 | kda_conv | 0.0928 | 0.136 | 0.126 | 0.68 | 0.99569 | 0.0184 |  |
| 34 | kda_conv | 0.0667 | 0.136 | 0.126 | 0.49 | 0.997774 | 0.0233 |  |
| 36 | kda_conv | 0.0638 | 0.136 | 0.126 | 0.47 | 0.997968 | 0.023 |  |
| 37 | kda_conv | 0.11 | 0.141 | 0.144 | 0.77 | 0.993889 | 0.0149 |  |
| 38 | kda_conv | 0.116 | 0.143 | 0.159 | 0.73 | 0.993254 | 0.0135 |  |
| 40 | kda_conv | 0.136 | 0.163 | 0.168 | 0.81 | 0.990772 | 0.0121 |  |
| 41 | kda_conv | 0.16 | 0.163 | 0.168 | 0.95 | 0.987269 | 0.0104 |  |
| 42 | kda_conv | 0.156 | 0.163 | 0.168 | 0.92 | 0.987881 | 0.00971 |  |
| 44 | kda_conv | 0.187 | 0.21 | 0.206 | 0.89 | 0.982391 | 0.00849 |  |


</details>

## 131,072 tokens

| pair (x vs reference) | kind | MLA rel L2 (layer) | MLA min exact bytes | indexer rel L2 (layer) | tail rel L2, valid slots | KDA state rel L2 (layer) | conv rel L2 (layer) | KL | top-1 | top-5 | top-k min overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v_filler_131072 vs v_full_131072 | noise_vllm | 0.231 (27) | 0.167 | 0.119 (23) | —, 0 | 0.236 (38) | 0.616 (40) | 0.00261 | = | 5/5 | 0.439 |
| o_whole_131072 vs v_full_131072 | audit | 0.255 (27) | 0.145 | 0.137 (23) | —, 0 | 0.311 (42) | 0.63 (40) | 0.00282 | = | 5/5 | 0.428 |
| o_c8192_131072 vs v_full_131072 | audit | 0.255 (27) | 0.145 | 0.137 (23) | —, 0 | 0.311 (42) | 0.63 (40) | 0.00282 | = | 5/5 | 0.428 |
| o_c8192_131072 vs o_whole_131072 | noise_ours | 0 (3) | 1 | 0 (3) | —, 0 | 0 (0) | 0 (0) | 0 | = | 5/5 | 1 |
| o_c8001_131072 vs v_full_131072 | audit | 0.256 (27) | 0.145 | 0.138 (23) | —, 0 | 0.286 (24) | 0.299 (29) | 0.000745 | = | 4/5 | 0.671 |
| o_c8001_131072 vs o_whole_131072 | noise_ours | 0.236 (27) | 0.205 | 0.122 (23) | —, 0 | 0.236 (38) | 0.669 (40) | 0.00191 | = | 4/5 | 0.451 |
| v_filler_100259 vs v_full_100259 | noise_vllm | 0.23 (27) | 0.17 | 0.119 (23) | 0.0894 (19), 3 | 0.21 (28) | 0.195 (22) | 5.66e-05 | = | 4/5 | 0.551 |
| o_c8001_100259 vs o_full_100259 | noise_ours | 0.235 (27) | 0.219 | 0.124 (23) | 0.114 (19), 3 | 0.212 (33) | 0.265 (22) | 0.000129 | = | 3/5 | 0.342 |
| v_filler_100006 vs v_full_100006 | noise_vllm | 0.229 (27) | 0.17 | 0.118 (23) | 0.137 (43), 2 | 0.177 (40) | 0.291 (40) | 0.128 | = | 5/5 | 0.702 |
| o_c8001_100006 vs o_full_100006 | noise_ours | 0.235 (27) | 0.219 | 0.124 (23) | 0.0977 (43), 2 | 0.165 (38) | 0.183 (40) | 0.0486 | = | 5/5 | 0.735 |
| o_full_132096 vs v_full_132096 | audit | 0.255 (27) | 0.145 | 0.136 (23) | —, 0 | 0.285 (24) | 0.241 (37) | 0.00781 | = | 5/5 | 0.717 |
| o_prefix_131075_1021 vs v_full_132096 | audit_prefix | 0.255 (27) | 0.145 | 0.136 (23) | —, 0 | 0.284 (24) | 0.231 (37) | 0.0132 | = | 4/5 | 0.684 |
| o_prefix_131075_1021 vs o_full_132096 | prefix_vs_ours_full | 0.0199 (27) | 0.993 | 0.00953 (39) | —, 0 | 0.15 (21) | 0.217 (21) | 0.0129 | = | 4/5 | 0.844 |
| o_full_100259 vs v_full_100259 | audit | 0.252 (23) | 0.145 | 0.136 (23) | 0.151 (3), 3 | 0.298 (33) | 0.252 (22) | 0.00012 | = | 3/5 | 0.432 |
| o_prefix_100003_256 vs v_full_100259 | audit_prefix | 0.252 (23) | 0.145 | 0.136 (23) | 0.686 (19), 3 | 0.287 (33) | 0.254 (18) | 8.53e-05 | = | 5/5 | 0.439 |
| o_prefix_100003_256 vs o_full_100259 | prefix_vs_ours_full | 0.028 (43) | 0.984 | 0.0137 (23) | 0.661 (19), 3 | 0.195 (29) | 0.293 (21) | 0.000137 | = | 3/5 | 0.631 |
| o_full_100006 vs v_full_100006 | audit | 0.253 (23) | 0.145 | 0.137 (23) | 0.154 (3), 2 | 0.271 (24) | 0.225 (40) | 0.109 | = | 4/5 | 0.709 |
| o_prefix_100003_3 vs v_full_100006 | audit_prefix | 0.253 (23) | 0.145 | 0.137 (23) | 0.155 (3), 2 | 0.271 (24) | 0.236 (40) | 0.115 | = | 5/5 | 0.699 |
| o_prefix_100003_3 vs o_full_100006 | prefix_vs_ours_full | 0.000904 (43) | 1 | 0.000331 (7) | 0.0881 (43), 2 | 0.11 (37) | 0.176 (40) | 0.0588 | = | 4/5 | 0.883 |

<details><summary>o_whole_131072 vs v_full_131072: 7 above noise, 0 patterns</summary>

Per layer, `o_whole_131072` vs `v_full_131072`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.123 | 0 | 0.0634 | 1.9 | 0.992475 | 0.263 |  |
| 7 | mla | 0.139 | 0.0865 | 0.121 | 1.1 | 0.990286 | 0.216 |  |
| 11 | mla | 0.12 | 0.0865 | 0.121 | 0.99 | 0.992763 | 0.243 |  |
| 15 | mla | 0.149 | 0.115 | 0.121 | 1.2 | 0.988838 | 0.206 |  |
| 19 | mla | 0.208 | 0.184 | 0.189 | 1.1 | 0.978322 | 0.179 |  |
| 23 | mla | 0.255 | 0.23 | 0.234 | 1.1 | 0.967589 | 0.154 |  |
| 27 | mla | 0.255 | 0.231 | 0.236 | 1.1 | 0.967371 | 0.17 |  |
| 31 | mla | 0.164 | 0.231 | 0.236 | 0.69 | 0.986589 | 0.225 |  |
| 35 | mla | 0.173 | 0.231 | 0.236 | 0.73 | 0.985007 | 0.229 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.977481 | 0.174 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.97121 | 0.145 |  |
| 3 | indexer | 0.12 | 0 | 0.0364 | 3.3 | 0.993034 | 0.307 | **above noise** |
| 7 | indexer | 0.0688 | 0.0369 | 0.047 | 1.5 | 0.997629 | 0.403 |  |
| 11 | indexer | 0.0955 | 0.0488 | 0.055 | 1.7 | 0.995435 | 0.296 |  |
| 15 | indexer | 0.105 | 0.0588 | 0.0623 | 1.7 | 0.994469 | 0.254 |  |
| 19 | indexer | 0.122 | 0.101 | 0.106 | 1.1 | 0.992618 | 0.319 |  |
| 23 | indexer | 0.137 | 0.119 | 0.124 | 1.1 | 0.990586 | 0.303 |  |
| 27 | indexer | 0.132 | 0.119 | 0.124 | 1.1 | 0.991227 | 0.335 |  |
| 31 | indexer | 0.0904 | 0.119 | 0.124 | 0.73 | 0.995914 | 0.348 |  |
| 35 | indexer | 0.0965 | 0.119 | 0.124 | 0.78 | 0.995335 | 0.384 |  |
| 39 | indexer | 0.111 | 0.119 | 0.124 | 0.89 | 0.993872 | 0.363 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.993099 | 0.247 |  |
| 0 | kda_rec | 0.0112 | 0 | 0.00307 | 3.6 | 0.999938 | 3.81e-06 | **above noise** |
| 1 | kda_rec | 0.0565 | 0 | 0.0178 | 3.2 | 0.998402 | 0 | **above noise** |
| 2 | kda_rec | 0.139 | 0 | 0.0238 | 5.8 | 0.99069 | 0 | **above noise** |
| 4 | kda_rec | 0.077 | 0.0129 | 0.0388 | 2 | 0.997032 | 0 |  |
| 5 | kda_rec | 0.148 | 0.0241 | 0.0388 | 3.8 | 0.989465 | 0 | **above noise** |
| 6 | kda_rec | 0.161 | 0.0363 | 0.0482 | 3.3 | 0.987058 | 0 | **above noise** |
| 8 | kda_rec | 0.0559 | 0.0363 | 0.0482 | 1.2 | 0.99844 | 0 |  |
| 9 | kda_rec | 0.114 | 0.0728 | 0.0834 | 1.4 | 0.993476 | 0 |  |
| 10 | kda_rec | 0.256 | 0.0728 | 0.0834 | 3.1 | 0.969908 | 0 | **above noise** |
| 12 | kda_rec | 0.133 | 0.0841 | 0.0935 | 1.4 | 0.991377 | 0 |  |
| 13 | kda_rec | 0.0709 | 0.0841 | 0.0935 | 0.76 | 0.997496 | 0 |  |
| 14 | kda_rec | 0.126 | 0.0841 | 0.0935 | 1.3 | 0.992076 | 0.000217 |  |
| 16 | kda_rec | 0.12 | 0.0841 | 0.0935 | 1.3 | 0.992783 | 0.000103 |  |
| 17 | kda_rec | 0.135 | 0.0841 | 0.0935 | 1.4 | 0.990854 | 0 |  |
| 18 | kda_rec | 0.157 | 0.104 | 0.111 | 1.4 | 0.987709 | 0 |  |
| 20 | kda_rec | 0.15 | 0.117 | 0.128 | 1.2 | 0.988646 | 0 |  |
| 21 | kda_rec | 0.236 | 0.176 | 0.206 | 1.1 | 0.972667 | 0 |  |
| 22 | kda_rec | 0.249 | 0.176 | 0.206 | 1.2 | 0.981209 | 9.54e-07 |  |
| 24 | kda_rec | 0.285 | 0.176 | 0.206 | 1.4 | 0.963566 | 9.54e-07 |  |
| 25 | kda_rec | 0.15 | 0.176 | 0.206 | 0.73 | 0.988725 | 2.86e-06 |  |
| 26 | kda_rec | 0.208 | 0.193 | 0.206 | 1 | 0.978717 | 0 |  |
| 28 | kda_rec | 0.238 | 0.21 | 0.206 | 1.1 | 0.971727 | 1.91e-06 |  |
| 29 | kda_rec | 0.22 | 0.21 | 0.211 | 1 | 0.976388 | 0 |  |
| 30 | kda_rec | 0.0753 | 0.21 | 0.211 | 0.36 | 0.997407 | 9.54e-07 |  |
| 32 | kda_rec | 0.221 | 0.21 | 0.211 | 1 | 0.975427 | 0 |  |
| 33 | kda_rec | 0.269 | 0.221 | 0.214 | 1.2 | 0.964679 | 0 |  |
| 34 | kda_rec | 0.0796 | 0.221 | 0.214 | 0.36 | 0.997135 | 0 |  |
| 36 | kda_rec | 0.0895 | 0.221 | 0.214 | 0.4 | 0.997759 | 0 |  |
| 37 | kda_rec | 0.209 | 0.221 | 0.214 | 0.94 | 0.978027 | 9.54e-07 |  |
| 38 | kda_rec | 0.254 | 0.236 | 0.236 | 1.1 | 0.967733 | 0.00012 |  |
| 40 | kda_rec | 0.229 | 0.236 | 0.236 | 0.97 | 0.974139 | 0.000123 |  |
| 41 | kda_rec | 0.202 | 0.236 | 0.236 | 0.86 | 0.979489 | 0 |  |
| 42 | kda_rec | 0.311 | 0.236 | 0.236 | 1.3 | 0.953673 | 0 |  |
| 44 | kda_rec | 0.173 | 0.236 | 0.236 | 0.73 | 0.985042 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000714 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0286 | 0 | 0.0292 | 0.98 | 0.99959 | 0.055 |  |
| 2 | kda_conv | 0.0547 | 0 | 0.0498 | 1.1 | 0.998507 | 0.0363 |  |
| 4 | kda_conv | 0.0427 | 0.0071 | 0.0498 | 0.86 | 0.999104 | 0.0371 |  |
| 5 | kda_conv | 0.0777 | 0.0324 | 0.0621 | 1.3 | 0.996975 | 0.0222 |  |
| 6 | kda_conv | 0.079 | 0.0441 | 0.0621 | 1.3 | 0.996876 | 0.0217 |  |
| 8 | kda_conv | 0.122 | 0.109 | 0.123 | 0.99 | 0.992647 | 0.0133 |  |
| 9 | kda_conv | 0.0992 | 0.109 | 0.123 | 0.81 | 0.995073 | 0.0173 |  |
| 10 | kda_conv | 0.0851 | 0.109 | 0.123 | 0.69 | 0.996398 | 0.0217 |  |
| 12 | kda_conv | 0.114 | 0.12 | 0.131 | 0.87 | 0.99363 | 0.0144 |  |
| 13 | kda_conv | 0.132 | 0.133 | 0.149 | 0.88 | 0.991397 | 0.0128 |  |
| 14 | kda_conv | 0.107 | 0.133 | 0.149 | 0.72 | 0.994297 | 0.0135 |  |
| 16 | kda_conv | 0.0995 | 0.133 | 0.149 | 0.67 | 0.995043 | 0.0152 |  |
| 17 | kda_conv | 0.121 | 0.16 | 0.188 | 0.64 | 0.99272 | 0.0135 |  |
| 18 | kda_conv | 0.169 | 0.179 | 0.355 | 0.48 | 0.985576 | 0.0109 |  |
| 20 | kda_conv | 0.228 | 0.2 | 0.381 | 0.6 | 0.973691 | 0.00844 |  |
| 21 | kda_conv | 0.338 | 0.268 | 0.474 | 0.71 | 0.941753 | 0.00723 |  |
| 22 | kda_conv | 0.368 | 0.281 | 0.474 | 0.78 | 0.932228 | 0.00766 |  |
| 24 | kda_conv | 0.311 | 0.281 | 0.474 | 0.66 | 0.951539 | 0.00755 |  |
| 25 | kda_conv | 0.308 | 0.281 | 0.474 | 0.65 | 0.952352 | 0.00907 |  |
| 26 | kda_conv | 0.343 | 0.29 | 0.474 | 0.72 | 0.941215 | 0.0077 |  |
| 28 | kda_conv | 0.379 | 0.35 | 0.474 | 0.8 | 0.927143 | 0.00798 |  |
| 29 | kda_conv | 0.459 | 0.439 | 0.474 | 0.97 | 0.892021 | 0.00726 |  |
| 30 | kda_conv | 0.365 | 0.439 | 0.474 | 0.77 | 0.933254 | 0.00916 |  |
| 32 | kda_conv | 0.397 | 0.439 | 0.474 | 0.84 | 0.921705 | 0.00907 |  |
| 33 | kda_conv | 0.552 | 0.541 | 0.647 | 0.85 | 0.837736 | 0.0089 |  |
| 34 | kda_conv | 0.278 | 0.541 | 0.647 | 0.43 | 0.96325 | 0.0116 |  |
| 36 | kda_conv | 0.335 | 0.541 | 0.647 | 0.52 | 0.944636 | 0.0122 |  |
| 37 | kda_conv | 0.536 | 0.541 | 0.647 | 0.83 | 0.853488 | 0.00769 |  |
| 38 | kda_conv | 0.574 | 0.565 | 0.647 | 0.89 | 0.834883 | 0.00804 |  |
| 40 | kda_conv | 0.63 | 0.616 | 0.669 | 0.94 | 0.79268 | 0.00726 |  |
| 41 | kda_conv | 0.541 | 0.616 | 0.669 | 0.81 | 0.849934 | 0.00802 |  |
| 42 | kda_conv | 0.521 | 0.616 | 0.669 | 0.78 | 0.872585 | 0.00865 |  |
| 44 | kda_conv | 0.236 | 0.616 | 0.669 | 0.35 | 0.972596 | 0.0109 |  |


</details>

<details><summary>o_c8192_131072 vs v_full_131072: 7 above noise, 0 patterns</summary>

Per layer, `o_c8192_131072` vs `v_full_131072`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.123 | 0 | 0.0634 | 1.9 | 0.992475 | 0.263 |  |
| 7 | mla | 0.139 | 0.0865 | 0.121 | 1.1 | 0.990286 | 0.216 |  |
| 11 | mla | 0.12 | 0.0865 | 0.121 | 0.99 | 0.992763 | 0.243 |  |
| 15 | mla | 0.149 | 0.115 | 0.121 | 1.2 | 0.988838 | 0.206 |  |
| 19 | mla | 0.208 | 0.184 | 0.189 | 1.1 | 0.978322 | 0.179 |  |
| 23 | mla | 0.255 | 0.23 | 0.234 | 1.1 | 0.967589 | 0.154 |  |
| 27 | mla | 0.255 | 0.231 | 0.236 | 1.1 | 0.967371 | 0.17 |  |
| 31 | mla | 0.164 | 0.231 | 0.236 | 0.69 | 0.986589 | 0.225 |  |
| 35 | mla | 0.173 | 0.231 | 0.236 | 0.73 | 0.985007 | 0.229 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.977481 | 0.174 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.97121 | 0.145 |  |
| 3 | indexer | 0.12 | 0 | 0.0364 | 3.3 | 0.993034 | 0.307 | **above noise** |
| 7 | indexer | 0.0688 | 0.0369 | 0.047 | 1.5 | 0.997629 | 0.403 |  |
| 11 | indexer | 0.0955 | 0.0488 | 0.055 | 1.7 | 0.995435 | 0.296 |  |
| 15 | indexer | 0.105 | 0.0588 | 0.0623 | 1.7 | 0.994469 | 0.254 |  |
| 19 | indexer | 0.122 | 0.101 | 0.106 | 1.1 | 0.992618 | 0.319 |  |
| 23 | indexer | 0.137 | 0.119 | 0.124 | 1.1 | 0.990586 | 0.303 |  |
| 27 | indexer | 0.132 | 0.119 | 0.124 | 1.1 | 0.991227 | 0.335 |  |
| 31 | indexer | 0.0904 | 0.119 | 0.124 | 0.73 | 0.995914 | 0.348 |  |
| 35 | indexer | 0.0965 | 0.119 | 0.124 | 0.78 | 0.995335 | 0.384 |  |
| 39 | indexer | 0.111 | 0.119 | 0.124 | 0.89 | 0.993872 | 0.363 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.993099 | 0.247 |  |
| 0 | kda_rec | 0.0112 | 0 | 0.00307 | 3.6 | 0.999938 | 3.81e-06 | **above noise** |
| 1 | kda_rec | 0.0565 | 0 | 0.0178 | 3.2 | 0.998402 | 0 | **above noise** |
| 2 | kda_rec | 0.139 | 0 | 0.0238 | 5.8 | 0.99069 | 0 | **above noise** |
| 4 | kda_rec | 0.077 | 0.0129 | 0.0388 | 2 | 0.997032 | 0 |  |
| 5 | kda_rec | 0.148 | 0.0241 | 0.0388 | 3.8 | 0.989465 | 0 | **above noise** |
| 6 | kda_rec | 0.161 | 0.0363 | 0.0482 | 3.3 | 0.987058 | 0 | **above noise** |
| 8 | kda_rec | 0.0559 | 0.0363 | 0.0482 | 1.2 | 0.99844 | 0 |  |
| 9 | kda_rec | 0.114 | 0.0728 | 0.0834 | 1.4 | 0.993476 | 0 |  |
| 10 | kda_rec | 0.256 | 0.0728 | 0.0834 | 3.1 | 0.969908 | 0 | **above noise** |
| 12 | kda_rec | 0.133 | 0.0841 | 0.0935 | 1.4 | 0.991377 | 0 |  |
| 13 | kda_rec | 0.0709 | 0.0841 | 0.0935 | 0.76 | 0.997496 | 0 |  |
| 14 | kda_rec | 0.126 | 0.0841 | 0.0935 | 1.3 | 0.992076 | 0.000217 |  |
| 16 | kda_rec | 0.12 | 0.0841 | 0.0935 | 1.3 | 0.992783 | 0.000103 |  |
| 17 | kda_rec | 0.135 | 0.0841 | 0.0935 | 1.4 | 0.990854 | 0 |  |
| 18 | kda_rec | 0.157 | 0.104 | 0.111 | 1.4 | 0.987709 | 0 |  |
| 20 | kda_rec | 0.15 | 0.117 | 0.128 | 1.2 | 0.988646 | 0 |  |
| 21 | kda_rec | 0.236 | 0.176 | 0.206 | 1.1 | 0.972667 | 0 |  |
| 22 | kda_rec | 0.249 | 0.176 | 0.206 | 1.2 | 0.981209 | 9.54e-07 |  |
| 24 | kda_rec | 0.285 | 0.176 | 0.206 | 1.4 | 0.963566 | 9.54e-07 |  |
| 25 | kda_rec | 0.15 | 0.176 | 0.206 | 0.73 | 0.988725 | 2.86e-06 |  |
| 26 | kda_rec | 0.208 | 0.193 | 0.206 | 1 | 0.978717 | 0 |  |
| 28 | kda_rec | 0.238 | 0.21 | 0.206 | 1.1 | 0.971727 | 1.91e-06 |  |
| 29 | kda_rec | 0.22 | 0.21 | 0.211 | 1 | 0.976388 | 0 |  |
| 30 | kda_rec | 0.0753 | 0.21 | 0.211 | 0.36 | 0.997407 | 9.54e-07 |  |
| 32 | kda_rec | 0.221 | 0.21 | 0.211 | 1 | 0.975427 | 0 |  |
| 33 | kda_rec | 0.269 | 0.221 | 0.214 | 1.2 | 0.964679 | 0 |  |
| 34 | kda_rec | 0.0796 | 0.221 | 0.214 | 0.36 | 0.997135 | 0 |  |
| 36 | kda_rec | 0.0895 | 0.221 | 0.214 | 0.4 | 0.997759 | 0 |  |
| 37 | kda_rec | 0.209 | 0.221 | 0.214 | 0.94 | 0.978027 | 9.54e-07 |  |
| 38 | kda_rec | 0.254 | 0.236 | 0.236 | 1.1 | 0.967733 | 0.00012 |  |
| 40 | kda_rec | 0.229 | 0.236 | 0.236 | 0.97 | 0.974139 | 0.000123 |  |
| 41 | kda_rec | 0.202 | 0.236 | 0.236 | 0.86 | 0.979489 | 0 |  |
| 42 | kda_rec | 0.311 | 0.236 | 0.236 | 1.3 | 0.953673 | 0 |  |
| 44 | kda_rec | 0.173 | 0.236 | 0.236 | 0.73 | 0.985042 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000714 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0286 | 0 | 0.0292 | 0.98 | 0.99959 | 0.055 |  |
| 2 | kda_conv | 0.0547 | 0 | 0.0498 | 1.1 | 0.998507 | 0.0363 |  |
| 4 | kda_conv | 0.0427 | 0.0071 | 0.0498 | 0.86 | 0.999104 | 0.0371 |  |
| 5 | kda_conv | 0.0777 | 0.0324 | 0.0621 | 1.3 | 0.996975 | 0.0222 |  |
| 6 | kda_conv | 0.079 | 0.0441 | 0.0621 | 1.3 | 0.996876 | 0.0217 |  |
| 8 | kda_conv | 0.122 | 0.109 | 0.123 | 0.99 | 0.992647 | 0.0133 |  |
| 9 | kda_conv | 0.0992 | 0.109 | 0.123 | 0.81 | 0.995073 | 0.0173 |  |
| 10 | kda_conv | 0.0851 | 0.109 | 0.123 | 0.69 | 0.996398 | 0.0217 |  |
| 12 | kda_conv | 0.114 | 0.12 | 0.131 | 0.87 | 0.99363 | 0.0144 |  |
| 13 | kda_conv | 0.132 | 0.133 | 0.149 | 0.88 | 0.991397 | 0.0128 |  |
| 14 | kda_conv | 0.107 | 0.133 | 0.149 | 0.72 | 0.994297 | 0.0135 |  |
| 16 | kda_conv | 0.0995 | 0.133 | 0.149 | 0.67 | 0.995043 | 0.0152 |  |
| 17 | kda_conv | 0.121 | 0.16 | 0.188 | 0.64 | 0.99272 | 0.0135 |  |
| 18 | kda_conv | 0.169 | 0.179 | 0.355 | 0.48 | 0.985576 | 0.0109 |  |
| 20 | kda_conv | 0.228 | 0.2 | 0.381 | 0.6 | 0.973691 | 0.00844 |  |
| 21 | kda_conv | 0.338 | 0.268 | 0.474 | 0.71 | 0.941753 | 0.00723 |  |
| 22 | kda_conv | 0.368 | 0.281 | 0.474 | 0.78 | 0.932228 | 0.00766 |  |
| 24 | kda_conv | 0.311 | 0.281 | 0.474 | 0.66 | 0.951539 | 0.00755 |  |
| 25 | kda_conv | 0.308 | 0.281 | 0.474 | 0.65 | 0.952352 | 0.00907 |  |
| 26 | kda_conv | 0.343 | 0.29 | 0.474 | 0.72 | 0.941215 | 0.0077 |  |
| 28 | kda_conv | 0.379 | 0.35 | 0.474 | 0.8 | 0.927143 | 0.00798 |  |
| 29 | kda_conv | 0.459 | 0.439 | 0.474 | 0.97 | 0.892021 | 0.00726 |  |
| 30 | kda_conv | 0.365 | 0.439 | 0.474 | 0.77 | 0.933254 | 0.00916 |  |
| 32 | kda_conv | 0.397 | 0.439 | 0.474 | 0.84 | 0.921705 | 0.00907 |  |
| 33 | kda_conv | 0.552 | 0.541 | 0.647 | 0.85 | 0.837736 | 0.0089 |  |
| 34 | kda_conv | 0.278 | 0.541 | 0.647 | 0.43 | 0.96325 | 0.0116 |  |
| 36 | kda_conv | 0.335 | 0.541 | 0.647 | 0.52 | 0.944636 | 0.0122 |  |
| 37 | kda_conv | 0.536 | 0.541 | 0.647 | 0.83 | 0.853488 | 0.00769 |  |
| 38 | kda_conv | 0.574 | 0.565 | 0.647 | 0.89 | 0.834883 | 0.00804 |  |
| 40 | kda_conv | 0.63 | 0.616 | 0.669 | 0.94 | 0.79268 | 0.00726 |  |
| 41 | kda_conv | 0.541 | 0.616 | 0.669 | 0.81 | 0.849934 | 0.00802 |  |
| 42 | kda_conv | 0.521 | 0.616 | 0.669 | 0.78 | 0.872585 | 0.00865 |  |
| 44 | kda_conv | 0.236 | 0.616 | 0.669 | 0.35 | 0.972596 | 0.0109 |  |


</details>

<details><summary>o_c8001_131072 vs v_full_131072: 7 above noise, 0 patterns</summary>

Per layer, `o_c8001_131072` vs `v_full_131072`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.123 | 0 | 0.0634 | 1.9 | 0.992473 | 0.263 |  |
| 7 | mla | 0.139 | 0.0865 | 0.121 | 1.2 | 0.990279 | 0.216 |  |
| 11 | mla | 0.12 | 0.0865 | 0.121 | 0.99 | 0.992789 | 0.244 |  |
| 15 | mla | 0.149 | 0.115 | 0.121 | 1.2 | 0.988841 | 0.206 |  |
| 19 | mla | 0.209 | 0.184 | 0.189 | 1.1 | 0.978221 | 0.179 |  |
| 23 | mla | 0.256 | 0.23 | 0.234 | 1.1 | 0.967304 | 0.153 |  |
| 27 | mla | 0.256 | 0.231 | 0.236 | 1.1 | 0.967226 | 0.17 |  |
| 31 | mla | 0.165 | 0.231 | 0.236 | 0.7 | 0.986384 | 0.223 |  |
| 35 | mla | 0.175 | 0.231 | 0.236 | 0.74 | 0.984763 | 0.227 |  |
| 39 | mla | 0.214 | 0.231 | 0.236 | 0.91 | 0.977179 | 0.173 |  |
| 43 | mla | 0.241 | 0.231 | 0.236 | 1 | 0.970979 | 0.145 |  |
| 3 | indexer | 0.12 | 0 | 0.0364 | 3.3 | 0.993039 | 0.307 | **above noise** |
| 7 | indexer | 0.0687 | 0.0369 | 0.047 | 1.5 | 0.997635 | 0.404 |  |
| 11 | indexer | 0.0951 | 0.0488 | 0.055 | 1.7 | 0.995472 | 0.298 |  |
| 15 | indexer | 0.105 | 0.0588 | 0.0623 | 1.7 | 0.994496 | 0.255 |  |
| 19 | indexer | 0.123 | 0.101 | 0.106 | 1.2 | 0.992492 | 0.316 |  |
| 23 | indexer | 0.138 | 0.119 | 0.124 | 1.1 | 0.990495 | 0.299 |  |
| 27 | indexer | 0.132 | 0.119 | 0.124 | 1.1 | 0.991257 | 0.33 |  |
| 31 | indexer | 0.0923 | 0.119 | 0.124 | 0.74 | 0.99574 | 0.342 |  |
| 35 | indexer | 0.0983 | 0.119 | 0.124 | 0.79 | 0.995162 | 0.379 |  |
| 39 | indexer | 0.113 | 0.119 | 0.124 | 0.91 | 0.993638 | 0.358 |  |
| 43 | indexer | 0.118 | 0.119 | 0.124 | 0.95 | 0.992971 | 0.245 |  |
| 0 | kda_rec | 0.0112 | 0 | 0.00307 | 3.6 | 0.999938 | 4.77e-06 | **above noise** |
| 1 | kda_rec | 0.0563 | 0 | 0.0178 | 3.2 | 0.998417 | 0 | **above noise** |
| 2 | kda_rec | 0.139 | 0 | 0.0238 | 5.9 | 0.99066 | 0 | **above noise** |
| 4 | kda_rec | 0.0787 | 0.0129 | 0.0388 | 2 | 0.996918 | 0 |  |
| 5 | kda_rec | 0.148 | 0.0241 | 0.0388 | 3.8 | 0.989499 | 9.54e-07 | **above noise** |
| 6 | kda_rec | 0.161 | 0.0363 | 0.0482 | 3.3 | 0.987024 | 9.54e-07 | **above noise** |
| 8 | kda_rec | 0.0557 | 0.0363 | 0.0482 | 1.2 | 0.998449 | 0 |  |
| 9 | kda_rec | 0.113 | 0.0728 | 0.0834 | 1.3 | 0.993645 | 0 |  |
| 10 | kda_rec | 0.255 | 0.0728 | 0.0834 | 3.1 | 0.970055 | 9.54e-07 | **above noise** |
| 12 | kda_rec | 0.13 | 0.0841 | 0.0935 | 1.4 | 0.991763 | 0 |  |
| 13 | kda_rec | 0.0702 | 0.0841 | 0.0935 | 0.75 | 0.997569 | 9.54e-07 |  |
| 14 | kda_rec | 0.13 | 0.0841 | 0.0935 | 1.4 | 0.991519 | 0.000218 |  |
| 16 | kda_rec | 0.119 | 0.0841 | 0.0935 | 1.3 | 0.992871 | 0.000103 |  |
| 17 | kda_rec | 0.137 | 0.0841 | 0.0935 | 1.5 | 0.990678 | 0 |  |
| 18 | kda_rec | 0.159 | 0.104 | 0.111 | 1.4 | 0.987502 | 0 |  |
| 20 | kda_rec | 0.151 | 0.117 | 0.128 | 1.2 | 0.988484 | 0 |  |
| 21 | kda_rec | 0.232 | 0.176 | 0.206 | 1.1 | 0.97367 | 0 |  |
| 22 | kda_rec | 0.25 | 0.176 | 0.206 | 1.2 | 0.981326 | 0 |  |
| 24 | kda_rec | 0.286 | 0.176 | 0.206 | 1.4 | 0.963472 | 9.54e-07 |  |
| 25 | kda_rec | 0.144 | 0.176 | 0.206 | 0.7 | 0.989657 | 2.86e-06 |  |
| 26 | kda_rec | 0.201 | 0.193 | 0.206 | 0.98 | 0.980183 | 0 |  |
| 28 | kda_rec | 0.201 | 0.21 | 0.206 | 0.96 | 0.979909 | 1.91e-06 |  |
| 29 | kda_rec | 0.2 | 0.21 | 0.211 | 0.95 | 0.980534 | 9.54e-07 |  |
| 30 | kda_rec | 0.0686 | 0.21 | 0.211 | 0.33 | 0.997906 | 0 |  |
| 32 | kda_rec | 0.16 | 0.21 | 0.211 | 0.76 | 0.987203 | 0 |  |
| 33 | kda_rec | 0.219 | 0.221 | 0.214 | 0.99 | 0.976585 | 0 |  |
| 34 | kda_rec | 0.0768 | 0.221 | 0.214 | 0.35 | 0.997388 | 9.54e-07 |  |
| 36 | kda_rec | 0.0845 | 0.221 | 0.214 | 0.38 | 0.998166 | 9.54e-07 |  |
| 37 | kda_rec | 0.153 | 0.221 | 0.214 | 0.69 | 0.988217 | 0 |  |
| 38 | kda_rec | 0.182 | 0.236 | 0.236 | 0.77 | 0.983526 | 0.00012 |  |
| 40 | kda_rec | 0.177 | 0.236 | 0.236 | 0.75 | 0.984635 | 0.000123 |  |
| 41 | kda_rec | 0.154 | 0.236 | 0.236 | 0.65 | 0.988225 | 0 |  |
| 42 | kda_rec | 0.278 | 0.236 | 0.236 | 1.2 | 0.963421 | 0 |  |
| 44 | kda_rec | 0.153 | 0.236 | 0.236 | 0.65 | 0.988332 | 0 |  |
| 0 | kda_conv | 0.000553 | 0 | 0.000714 | 0.78 | 1 | 0.935 |  |
| 1 | kda_conv | 0.0268 | 0 | 0.0292 | 0.92 | 0.999647 | 0.059 |  |
| 2 | kda_conv | 0.053 | 0 | 0.0498 | 1.1 | 0.998596 | 0.0373 |  |
| 4 | kda_conv | 0.0459 | 0.0071 | 0.0498 | 0.92 | 0.998947 | 0.0378 |  |
| 5 | kda_conv | 0.088 | 0.0324 | 0.0621 | 1.4 | 0.996144 | 0.021 |  |
| 6 | kda_conv | 0.0856 | 0.0441 | 0.0621 | 1.4 | 0.996372 | 0.0201 |  |
| 8 | kda_conv | 0.13 | 0.109 | 0.123 | 1.1 | 0.991572 | 0.0138 |  |
| 9 | kda_conv | 0.105 | 0.109 | 0.123 | 0.85 | 0.994508 | 0.0161 |  |
| 10 | kda_conv | 0.0888 | 0.109 | 0.123 | 0.72 | 0.996104 | 0.021 |  |
| 12 | kda_conv | 0.113 | 0.12 | 0.131 | 0.87 | 0.993697 | 0.015 |  |
| 13 | kda_conv | 0.13 | 0.133 | 0.149 | 0.87 | 0.991624 | 0.0126 |  |
| 14 | kda_conv | 0.113 | 0.133 | 0.149 | 0.76 | 0.993615 | 0.0145 |  |
| 16 | kda_conv | 0.0964 | 0.133 | 0.149 | 0.65 | 0.995358 | 0.0167 |  |
| 17 | kda_conv | 0.183 | 0.16 | 0.188 | 0.98 | 0.983405 | 0.0114 |  |
| 18 | kda_conv | 0.285 | 0.179 | 0.355 | 0.8 | 0.959309 | 0.0101 |  |
| 20 | kda_conv | 0.276 | 0.2 | 0.381 | 0.72 | 0.961458 | 0.00773 |  |
| 21 | kda_conv | 0.299 | 0.268 | 0.474 | 0.63 | 0.955357 | 0.00699 |  |
| 22 | kda_conv | 0.274 | 0.281 | 0.474 | 0.58 | 0.963306 | 0.00693 |  |
| 24 | kda_conv | 0.29 | 0.281 | 0.474 | 0.61 | 0.958178 | 0.00736 |  |
| 25 | kda_conv | 0.263 | 0.281 | 0.474 | 0.55 | 0.965145 | 0.00871 |  |
| 26 | kda_conv | 0.293 | 0.29 | 0.474 | 0.62 | 0.957254 | 0.00713 |  |
| 28 | kda_conv | 0.248 | 0.35 | 0.474 | 0.52 | 0.968952 | 0.00831 |  |
| 29 | kda_conv | 0.299 | 0.439 | 0.474 | 0.63 | 0.954662 | 0.00766 |  |
| 30 | kda_conv | 0.238 | 0.439 | 0.474 | 0.5 | 0.971464 | 0.00916 |  |
| 32 | kda_conv | 0.2 | 0.439 | 0.474 | 0.42 | 0.980022 | 0.0101 |  |
| 33 | kda_conv | 0.253 | 0.541 | 0.647 | 0.39 | 0.968716 | 0.00883 |  |
| 34 | kda_conv | 0.127 | 0.541 | 0.647 | 0.2 | 0.991961 | 0.0137 |  |
| 36 | kda_conv | 0.149 | 0.541 | 0.647 | 0.23 | 0.98894 | 0.0125 |  |
| 37 | kda_conv | 0.196 | 0.541 | 0.647 | 0.3 | 0.980899 | 0.00944 |  |
| 38 | kda_conv | 0.194 | 0.565 | 0.647 | 0.3 | 0.981487 | 0.00941 |  |
| 40 | kda_conv | 0.199 | 0.616 | 0.669 | 0.3 | 0.98048 | 0.00812 |  |
| 41 | kda_conv | 0.191 | 0.616 | 0.669 | 0.28 | 0.982137 | 0.00899 |  |
| 42 | kda_conv | 0.19 | 0.616 | 0.669 | 0.28 | 0.982305 | 0.00903 |  |
| 44 | kda_conv | 0.13 | 0.616 | 0.669 | 0.19 | 0.991751 | 0.0127 |  |


</details>

<details><summary>o_full_132096 vs v_full_132096: 7 above noise, 0 patterns</summary>

Per layer, `o_full_132096` vs `v_full_132096`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.124 | 0 | 0.0634 | 1.9 | 0.992393 | 0.262 |  |
| 7 | mla | 0.14 | 0.0865 | 0.121 | 1.2 | 0.99026 | 0.215 |  |
| 11 | mla | 0.121 | 0.0865 | 0.121 | 1 | 0.992704 | 0.242 |  |
| 15 | mla | 0.15 | 0.115 | 0.121 | 1.2 | 0.98875 | 0.205 |  |
| 19 | mla | 0.209 | 0.184 | 0.189 | 1.1 | 0.97823 | 0.179 |  |
| 23 | mla | 0.254 | 0.23 | 0.234 | 1.1 | 0.96763 | 0.154 |  |
| 27 | mla | 0.255 | 0.231 | 0.236 | 1.1 | 0.967427 | 0.17 |  |
| 31 | mla | 0.164 | 0.231 | 0.236 | 0.69 | 0.986624 | 0.226 |  |
| 35 | mla | 0.173 | 0.231 | 0.236 | 0.73 | 0.98504 | 0.229 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.977577 | 0.175 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.971196 | 0.145 |  |
| 3 | indexer | 0.121 | 0 | 0.0364 | 3.3 | 0.992938 | 0.306 | **above noise** |
| 7 | indexer | 0.0691 | 0.0369 | 0.047 | 1.5 | 0.997611 | 0.402 |  |
| 11 | indexer | 0.0963 | 0.0488 | 0.055 | 1.8 | 0.995357 | 0.295 |  |
| 15 | indexer | 0.106 | 0.0588 | 0.0623 | 1.7 | 0.994341 | 0.251 |  |
| 19 | indexer | 0.122 | 0.101 | 0.106 | 1.1 | 0.992596 | 0.317 |  |
| 23 | indexer | 0.136 | 0.119 | 0.124 | 1.1 | 0.990767 | 0.304 |  |
| 27 | indexer | 0.131 | 0.119 | 0.124 | 1.1 | 0.99139 | 0.335 |  |
| 31 | indexer | 0.0902 | 0.119 | 0.124 | 0.73 | 0.995929 | 0.35 |  |
| 35 | indexer | 0.0979 | 0.119 | 0.124 | 0.79 | 0.9952 | 0.384 |  |
| 39 | indexer | 0.112 | 0.119 | 0.124 | 0.9 | 0.993694 | 0.364 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.993095 | 0.247 |  |
| 0 | kda_rec | 0.011 | 0 | 0.00307 | 3.6 | 0.999939 | 1.91e-06 | **above noise** |
| 1 | kda_rec | 0.0561 | 0 | 0.0178 | 3.2 | 0.998425 | 9.54e-07 | **above noise** |
| 2 | kda_rec | 0.149 | 0 | 0.0238 | 6.3 | 0.989372 | 0 | **above noise** |
| 4 | kda_rec | 0.0821 | 0.0129 | 0.0388 | 2.1 | 0.996638 | 0 |  |
| 5 | kda_rec | 0.162 | 0.0241 | 0.0388 | 4.2 | 0.987375 | 9.54e-07 | **above noise** |
| 6 | kda_rec | 0.158 | 0.0363 | 0.0482 | 3.3 | 0.987557 | 0 | **above noise** |
| 8 | kda_rec | 0.0554 | 0.0363 | 0.0482 | 1.1 | 0.998466 | 0 |  |
| 9 | kda_rec | 0.118 | 0.0728 | 0.0834 | 1.4 | 0.992959 | 0 |  |
| 10 | kda_rec | 0.252 | 0.0728 | 0.0834 | 3 | 0.970662 | 0 | **above noise** |
| 12 | kda_rec | 0.143 | 0.0841 | 0.0935 | 1.5 | 0.990043 | 0 |  |
| 13 | kda_rec | 0.0697 | 0.0841 | 0.0935 | 0.75 | 0.997579 | 9.54e-07 |  |
| 14 | kda_rec | 0.131 | 0.0841 | 0.0935 | 1.4 | 0.991406 | 0.000235 |  |
| 16 | kda_rec | 0.122 | 0.0841 | 0.0935 | 1.3 | 0.99259 | 0.000122 |  |
| 17 | kda_rec | 0.136 | 0.0841 | 0.0935 | 1.5 | 0.990848 | 0 |  |
| 18 | kda_rec | 0.159 | 0.104 | 0.111 | 1.4 | 0.987471 | 0 |  |
| 20 | kda_rec | 0.149 | 0.117 | 0.128 | 1.2 | 0.988862 | 0 |  |
| 21 | kda_rec | 0.231 | 0.176 | 0.206 | 1.1 | 0.973865 | 0 |  |
| 22 | kda_rec | 0.268 | 0.176 | 0.206 | 1.3 | 0.978605 | 0 |  |
| 24 | kda_rec | 0.285 | 0.176 | 0.206 | 1.4 | 0.963474 | 1.91e-06 |  |
| 25 | kda_rec | 0.142 | 0.176 | 0.206 | 0.69 | 0.989938 | 2.86e-06 |  |
| 26 | kda_rec | 0.19 | 0.193 | 0.206 | 0.92 | 0.982387 | 0 |  |
| 28 | kda_rec | 0.186 | 0.21 | 0.206 | 0.89 | 0.982731 | 9.54e-07 |  |
| 29 | kda_rec | 0.193 | 0.21 | 0.211 | 0.91 | 0.982079 | 0 |  |
| 30 | kda_rec | 0.065 | 0.21 | 0.211 | 0.31 | 0.998015 | 0 |  |
| 32 | kda_rec | 0.161 | 0.21 | 0.211 | 0.76 | 0.987095 | 0 |  |
| 33 | kda_rec | 0.215 | 0.221 | 0.214 | 0.97 | 0.977638 | 0 |  |
| 34 | kda_rec | 0.081 | 0.221 | 0.214 | 0.37 | 0.997129 | 0 |  |
| 36 | kda_rec | 0.0912 | 0.221 | 0.214 | 0.41 | 0.997772 | 0 |  |
| 37 | kda_rec | 0.152 | 0.221 | 0.214 | 0.69 | 0.988402 | 0 |  |
| 38 | kda_rec | 0.181 | 0.236 | 0.236 | 0.76 | 0.983685 | 0.000126 |  |
| 40 | kda_rec | 0.17 | 0.236 | 0.236 | 0.72 | 0.985631 | 0.000122 |  |
| 41 | kda_rec | 0.142 | 0.236 | 0.236 | 0.6 | 0.98993 | 0 |  |
| 42 | kda_rec | 0.264 | 0.236 | 0.236 | 1.1 | 0.966813 | 0 |  |
| 44 | kda_rec | 0.151 | 0.236 | 0.236 | 0.64 | 0.988787 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000714 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0263 | 0 | 0.0292 | 0.9 | 0.999655 | 0.0647 |  |
| 2 | kda_conv | 0.0642 | 0 | 0.0498 | 1.3 | 0.99794 | 0.0308 |  |
| 4 | kda_conv | 0.0536 | 0.0071 | 0.0498 | 1.1 | 0.998711 | 0.0338 |  |
| 5 | kda_conv | 0.0948 | 0.0324 | 0.0621 | 1.5 | 0.995508 | 0.0192 |  |
| 6 | kda_conv | 0.0803 | 0.0441 | 0.0621 | 1.3 | 0.996769 | 0.0216 |  |
| 8 | kda_conv | 0.121 | 0.109 | 0.123 | 0.98 | 0.992718 | 0.0144 |  |
| 9 | kda_conv | 0.113 | 0.109 | 0.123 | 0.92 | 0.993586 | 0.0155 |  |
| 10 | kda_conv | 0.0996 | 0.109 | 0.123 | 0.81 | 0.995026 | 0.0174 |  |
| 12 | kda_conv | 0.126 | 0.12 | 0.131 | 0.96 | 0.992208 | 0.015 |  |
| 13 | kda_conv | 0.148 | 0.133 | 0.149 | 0.99 | 0.989143 | 0.0122 |  |
| 14 | kda_conv | 0.141 | 0.133 | 0.149 | 0.95 | 0.989985 | 0.012 |  |
| 16 | kda_conv | 0.123 | 0.133 | 0.149 | 0.82 | 0.992458 | 0.0137 |  |
| 17 | kda_conv | 0.135 | 0.16 | 0.188 | 0.72 | 0.990814 | 0.0138 |  |
| 18 | kda_conv | 0.136 | 0.179 | 0.355 | 0.38 | 0.990722 | 0.0122 |  |
| 20 | kda_conv | 0.181 | 0.2 | 0.381 | 0.47 | 0.983618 | 0.00848 |  |
| 21 | kda_conv | 0.192 | 0.268 | 0.474 | 0.41 | 0.98153 | 0.00829 |  |
| 22 | kda_conv | 0.178 | 0.281 | 0.474 | 0.38 | 0.984141 | 0.0089 |  |
| 24 | kda_conv | 0.186 | 0.281 | 0.474 | 0.39 | 0.982771 | 0.00886 |  |
| 25 | kda_conv | 0.188 | 0.281 | 0.474 | 0.4 | 0.982694 | 0.00907 |  |
| 26 | kda_conv | 0.175 | 0.29 | 0.474 | 0.37 | 0.984836 | 0.00884 |  |
| 28 | kda_conv | 0.181 | 0.35 | 0.474 | 0.38 | 0.983784 | 0.00884 |  |
| 29 | kda_conv | 0.204 | 0.439 | 0.474 | 0.43 | 0.979465 | 0.00806 |  |
| 30 | kda_conv | 0.165 | 0.439 | 0.474 | 0.35 | 0.986369 | 0.0104 |  |
| 32 | kda_conv | 0.179 | 0.439 | 0.474 | 0.38 | 0.984111 | 0.01 |  |
| 33 | kda_conv | 0.227 | 0.541 | 0.647 | 0.35 | 0.974902 | 0.00792 |  |
| 34 | kda_conv | 0.123 | 0.541 | 0.647 | 0.19 | 0.992582 | 0.0122 |  |
| 36 | kda_conv | 0.151 | 0.541 | 0.647 | 0.23 | 0.988653 | 0.0111 |  |
| 37 | kda_conv | 0.241 | 0.541 | 0.647 | 0.37 | 0.970979 | 0.00804 |  |
| 38 | kda_conv | 0.193 | 0.565 | 0.647 | 0.3 | 0.981572 | 0.00906 |  |
| 40 | kda_conv | 0.196 | 0.616 | 0.669 | 0.29 | 0.981104 | 0.00894 |  |
| 41 | kda_conv | 0.176 | 0.616 | 0.669 | 0.26 | 0.984765 | 0.00902 |  |
| 42 | kda_conv | 0.158 | 0.616 | 0.669 | 0.24 | 0.987654 | 0.0106 |  |
| 44 | kda_conv | 0.161 | 0.616 | 0.669 | 0.24 | 0.987198 | 0.0106 |  |


</details>

<details><summary>o_prefix_131075_1021 vs v_full_132096: 7 above noise, 0 patterns</summary>

Per layer, `o_prefix_131075_1021` vs `v_full_132096`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.124 | 0 | 0.0634 | 1.9 | 0.992393 | 0.262 |  |
| 7 | mla | 0.14 | 0.0865 | 0.121 | 1.2 | 0.990259 | 0.215 |  |
| 11 | mla | 0.121 | 0.0865 | 0.121 | 1 | 0.992705 | 0.242 |  |
| 15 | mla | 0.15 | 0.115 | 0.121 | 1.2 | 0.98875 | 0.205 |  |
| 19 | mla | 0.209 | 0.184 | 0.189 | 1.1 | 0.978231 | 0.179 |  |
| 23 | mla | 0.254 | 0.23 | 0.234 | 1.1 | 0.96764 | 0.154 |  |
| 27 | mla | 0.255 | 0.231 | 0.236 | 1.1 | 0.967433 | 0.17 |  |
| 31 | mla | 0.164 | 0.231 | 0.236 | 0.69 | 0.986624 | 0.226 |  |
| 35 | mla | 0.173 | 0.231 | 0.236 | 0.73 | 0.985032 | 0.229 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.977568 | 0.175 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.971191 | 0.145 |  |
| 3 | indexer | 0.121 | 0 | 0.0364 | 3.3 | 0.992937 | 0.306 | **above noise** |
| 7 | indexer | 0.0691 | 0.0369 | 0.047 | 1.5 | 0.997611 | 0.402 |  |
| 11 | indexer | 0.0963 | 0.0488 | 0.055 | 1.8 | 0.995359 | 0.295 |  |
| 15 | indexer | 0.106 | 0.0588 | 0.0623 | 1.7 | 0.99434 | 0.251 |  |
| 19 | indexer | 0.122 | 0.101 | 0.106 | 1.1 | 0.992599 | 0.317 |  |
| 23 | indexer | 0.136 | 0.119 | 0.124 | 1.1 | 0.990766 | 0.304 |  |
| 27 | indexer | 0.131 | 0.119 | 0.124 | 1.1 | 0.991393 | 0.336 |  |
| 31 | indexer | 0.0902 | 0.119 | 0.124 | 0.72 | 0.995931 | 0.35 |  |
| 35 | indexer | 0.098 | 0.119 | 0.124 | 0.79 | 0.995195 | 0.384 |  |
| 39 | indexer | 0.112 | 0.119 | 0.124 | 0.9 | 0.993664 | 0.364 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.993096 | 0.247 |  |
| 0 | kda_rec | 0.011 | 0 | 0.00307 | 3.6 | 0.999939 | 4.77e-06 | **above noise** |
| 1 | kda_rec | 0.0561 | 0 | 0.0178 | 3.2 | 0.998423 | 9.54e-07 | **above noise** |
| 2 | kda_rec | 0.149 | 0 | 0.0238 | 6.3 | 0.98927 | 9.54e-07 | **above noise** |
| 4 | kda_rec | 0.0831 | 0.0129 | 0.0388 | 2.1 | 0.99654 | 0 |  |
| 5 | kda_rec | 0.164 | 0.0241 | 0.0388 | 4.2 | 0.987209 | 0 | **above noise** |
| 6 | kda_rec | 0.159 | 0.0363 | 0.0482 | 3.3 | 0.987381 | 0 | **above noise** |
| 8 | kda_rec | 0.0553 | 0.0363 | 0.0482 | 1.1 | 0.998471 | 0 |  |
| 9 | kda_rec | 0.128 | 0.0728 | 0.0834 | 1.5 | 0.991736 | 0 |  |
| 10 | kda_rec | 0.252 | 0.0728 | 0.0834 | 3 | 0.970604 | 0 | **above noise** |
| 12 | kda_rec | 0.134 | 0.0841 | 0.0935 | 1.4 | 0.991198 | 0 |  |
| 13 | kda_rec | 0.0701 | 0.0841 | 0.0935 | 0.75 | 0.997563 | 0 |  |
| 14 | kda_rec | 0.131 | 0.0841 | 0.0935 | 1.4 | 0.991472 | 0.000235 |  |
| 16 | kda_rec | 0.124 | 0.0841 | 0.0935 | 1.3 | 0.992303 | 0.000121 |  |
| 17 | kda_rec | 0.136 | 0.0841 | 0.0935 | 1.5 | 0.990848 | 0 |  |
| 18 | kda_rec | 0.16 | 0.104 | 0.111 | 1.4 | 0.987188 | 0 |  |
| 20 | kda_rec | 0.154 | 0.117 | 0.128 | 1.2 | 0.988108 | 0 |  |
| 21 | kda_rec | 0.236 | 0.176 | 0.206 | 1.1 | 0.97249 | 0 |  |
| 22 | kda_rec | 0.269 | 0.176 | 0.206 | 1.3 | 0.978428 | 0 |  |
| 24 | kda_rec | 0.284 | 0.176 | 0.206 | 1.4 | 0.963662 | 1.91e-06 |  |
| 25 | kda_rec | 0.143 | 0.176 | 0.206 | 0.69 | 0.989792 | 2.86e-06 |  |
| 26 | kda_rec | 0.192 | 0.193 | 0.206 | 0.93 | 0.982146 | 0 |  |
| 28 | kda_rec | 0.19 | 0.21 | 0.206 | 0.91 | 0.98199 | 9.54e-07 |  |
| 29 | kda_rec | 0.189 | 0.21 | 0.211 | 0.89 | 0.982788 | 0 |  |
| 30 | kda_rec | 0.0647 | 0.21 | 0.211 | 0.31 | 0.99802 | 0 |  |
| 32 | kda_rec | 0.159 | 0.21 | 0.211 | 0.75 | 0.987486 | 0 |  |
| 33 | kda_rec | 0.214 | 0.221 | 0.214 | 0.97 | 0.977721 | 9.54e-07 |  |
| 34 | kda_rec | 0.0807 | 0.221 | 0.214 | 0.36 | 0.997219 | 0 |  |
| 36 | kda_rec | 0.0908 | 0.221 | 0.214 | 0.41 | 0.99782 | 0 |  |
| 37 | kda_rec | 0.151 | 0.221 | 0.214 | 0.68 | 0.988491 | 0 |  |
| 38 | kda_rec | 0.177 | 0.236 | 0.236 | 0.75 | 0.984379 | 0.000126 |  |
| 40 | kda_rec | 0.172 | 0.236 | 0.236 | 0.73 | 0.985338 | 0.000122 |  |
| 41 | kda_rec | 0.144 | 0.236 | 0.236 | 0.61 | 0.989646 | 0 |  |
| 42 | kda_rec | 0.263 | 0.236 | 0.236 | 1.1 | 0.966968 | 9.54e-07 |  |
| 44 | kda_rec | 0.148 | 0.236 | 0.236 | 0.63 | 0.989124 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000714 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0263 | 0 | 0.0292 | 0.9 | 0.999654 | 0.0654 |  |
| 2 | kda_conv | 0.0623 | 0 | 0.0498 | 1.2 | 0.998059 | 0.0314 |  |
| 4 | kda_conv | 0.0575 | 0.0071 | 0.0498 | 1.2 | 0.998507 | 0.0323 |  |
| 5 | kda_conv | 0.102 | 0.0324 | 0.0621 | 1.6 | 0.994798 | 0.0191 |  |
| 6 | kda_conv | 0.0851 | 0.0441 | 0.0621 | 1.4 | 0.99638 | 0.0219 |  |
| 8 | kda_conv | 0.125 | 0.109 | 0.123 | 1 | 0.992116 | 0.0137 |  |
| 9 | kda_conv | 0.125 | 0.109 | 0.123 | 1 | 0.992219 | 0.0144 |  |
| 10 | kda_conv | 0.1 | 0.109 | 0.123 | 0.81 | 0.994998 | 0.0179 |  |
| 12 | kda_conv | 0.116 | 0.12 | 0.131 | 0.89 | 0.993312 | 0.0147 |  |
| 13 | kda_conv | 0.152 | 0.133 | 0.149 | 1 | 0.988436 | 0.0114 |  |
| 14 | kda_conv | 0.136 | 0.133 | 0.149 | 0.91 | 0.990706 | 0.0121 |  |
| 16 | kda_conv | 0.122 | 0.133 | 0.149 | 0.81 | 0.9926 | 0.0142 |  |
| 17 | kda_conv | 0.156 | 0.16 | 0.188 | 0.83 | 0.987744 | 0.0114 |  |
| 18 | kda_conv | 0.154 | 0.179 | 0.355 | 0.43 | 0.988131 | 0.011 |  |
| 20 | kda_conv | 0.209 | 0.2 | 0.381 | 0.55 | 0.977997 | 0.00772 |  |
| 21 | kda_conv | 0.217 | 0.268 | 0.474 | 0.46 | 0.976364 | 0.0077 |  |
| 22 | kda_conv | 0.226 | 0.281 | 0.474 | 0.48 | 0.974555 | 0.00718 |  |
| 24 | kda_conv | 0.204 | 0.281 | 0.474 | 0.43 | 0.97903 | 0.0081 |  |
| 25 | kda_conv | 0.198 | 0.281 | 0.474 | 0.42 | 0.980758 | 0.00781 |  |
| 26 | kda_conv | 0.185 | 0.29 | 0.474 | 0.39 | 0.983173 | 0.0088 |  |
| 28 | kda_conv | 0.172 | 0.35 | 0.474 | 0.36 | 0.985232 | 0.0097 |  |
| 29 | kda_conv | 0.197 | 0.439 | 0.474 | 0.42 | 0.980742 | 0.00849 |  |
| 30 | kda_conv | 0.164 | 0.439 | 0.474 | 0.35 | 0.98641 | 0.0106 |  |
| 32 | kda_conv | 0.176 | 0.439 | 0.474 | 0.37 | 0.98458 | 0.00935 |  |
| 33 | kda_conv | 0.225 | 0.541 | 0.647 | 0.35 | 0.975262 | 0.00892 |  |
| 34 | kda_conv | 0.118 | 0.541 | 0.647 | 0.18 | 0.993039 | 0.0127 |  |
| 36 | kda_conv | 0.144 | 0.541 | 0.647 | 0.22 | 0.989617 | 0.0125 |  |
| 37 | kda_conv | 0.231 | 0.541 | 0.647 | 0.36 | 0.973286 | 0.00856 |  |
| 38 | kda_conv | 0.188 | 0.565 | 0.647 | 0.29 | 0.982458 | 0.00928 |  |
| 40 | kda_conv | 0.202 | 0.616 | 0.669 | 0.3 | 0.979914 | 0.00901 |  |
| 41 | kda_conv | 0.175 | 0.616 | 0.669 | 0.26 | 0.984895 | 0.0094 |  |
| 42 | kda_conv | 0.15 | 0.616 | 0.669 | 0.22 | 0.988715 | 0.0115 |  |
| 44 | kda_conv | 0.138 | 0.616 | 0.669 | 0.21 | 0.990571 | 0.012 |  |


</details>

<details><summary>o_full_100259 vs v_full_100259: 5 above noise, 0 patterns</summary>

Per layer, `o_full_100259` vs `v_full_100259`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.104 | 0 | 0.0634 | 1.6 | 0.994597 | 0.292 |  |
| 7 | mla | 0.135 | 0.0865 | 0.121 | 1.1 | 0.990922 | 0.223 |  |
| 11 | mla | 0.114 | 0.0865 | 0.121 | 0.94 | 0.993531 | 0.255 |  |
| 15 | mla | 0.145 | 0.115 | 0.121 | 1.2 | 0.989488 | 0.213 |  |
| 19 | mla | 0.206 | 0.184 | 0.189 | 1.1 | 0.978722 | 0.182 |  |
| 23 | mla | 0.252 | 0.23 | 0.234 | 1.1 | 0.96819 | 0.156 |  |
| 27 | mla | 0.252 | 0.231 | 0.236 | 1.1 | 0.96823 | 0.174 |  |
| 31 | mla | 0.161 | 0.231 | 0.236 | 0.68 | 0.986961 | 0.23 |  |
| 35 | mla | 0.171 | 0.231 | 0.236 | 0.73 | 0.985363 | 0.234 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.977535 | 0.175 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.971097 | 0.145 |  |
| 3 | indexer | 0.0968 | 0 | 0.0364 | 2.7 | 0.995458 | 0.353 |  |
| 7 | indexer | 0.0619 | 0.0369 | 0.047 | 1.3 | 0.998082 | 0.433 |  |
| 11 | indexer | 0.0853 | 0.0488 | 0.055 | 1.6 | 0.996359 | 0.323 |  |
| 15 | indexer | 0.0975 | 0.0588 | 0.0623 | 1.6 | 0.995241 | 0.273 |  |
| 19 | indexer | 0.121 | 0.101 | 0.106 | 1.1 | 0.992709 | 0.332 |  |
| 23 | indexer | 0.136 | 0.119 | 0.124 | 1.1 | 0.990764 | 0.313 |  |
| 27 | indexer | 0.131 | 0.119 | 0.124 | 1.1 | 0.991426 | 0.346 |  |
| 31 | indexer | 0.0882 | 0.119 | 0.124 | 0.71 | 0.996109 | 0.362 |  |
| 35 | indexer | 0.0957 | 0.119 | 0.124 | 0.77 | 0.995413 | 0.398 |  |
| 39 | indexer | 0.108 | 0.119 | 0.124 | 0.87 | 0.994194 | 0.375 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.993199 | 0.252 |  |
| 3 | tail | 0.151 | 0 | 0.0368 | 4.1 | 0.988853 | 0.00911 | **above noise** |
| 7 | tail | 0.0778 | 0.0551 | 0.0732 | 1.1 | 0.996973 | 0.0169 |  |
| 11 | tail | 0.136 | 0.0883 | 0.0945 | 1.4 | 0.990677 | 0.0104 |  |
| 15 | tail | 0.104 | 0.0883 | 0.0945 | 1.1 | 0.994618 | 0.0221 |  |
| 19 | tail | 0.115 | 0.0894 | 0.114 | 1 | 0.993456 | 0.0104 |  |
| 23 | tail | 0.12 | 0.0894 | 0.114 | 1.1 | 0.992958 | 0.0169 |  |
| 27 | tail | 0.0825 | 0.0894 | 0.114 | 0.72 | 0.996604 | 0.0273 |  |
| 31 | tail | 0.0541 | 0.0894 | 0.114 | 0.47 | 0.9986 | 0.0352 |  |
| 35 | tail | 0.0441 | 0.0894 | 0.114 | 0.39 | 0.999032 | 0.0586 |  |
| 39 | tail | 0.0585 | 0.0894 | 0.114 | 0.51 | 0.998291 | 0.0247 |  |
| 43 | tail | 0.0747 | 0.137 | 0.114 | 0.54 | 0.997261 | 0.0312 |  |
| 0 | kda_rec | 0.0103 | 0 | 0.00307 | 3.4 | 0.999947 | 2.86e-06 | **above noise** |
| 1 | kda_rec | 0.0552 | 0 | 0.0178 | 3.1 | 0.998499 | 9.54e-07 | **above noise** |
| 2 | kda_rec | 0.107 | 0 | 0.0238 | 4.5 | 0.994451 | 9.54e-07 | **above noise** |
| 4 | kda_rec | 0.0602 | 0.0129 | 0.0388 | 1.6 | 0.99819 | 9.54e-07 |  |
| 5 | kda_rec | 0.132 | 0.0241 | 0.0388 | 3.4 | 0.992487 | 0 | **above noise** |
| 6 | kda_rec | 0.124 | 0.0363 | 0.0482 | 2.6 | 0.99231 | 0 |  |
| 8 | kda_rec | 0.0528 | 0.0363 | 0.0482 | 1.1 | 0.998631 | 0 |  |
| 9 | kda_rec | 0.106 | 0.0728 | 0.0834 | 1.3 | 0.994394 | 0 |  |
| 10 | kda_rec | 0.219 | 0.0728 | 0.0834 | 2.6 | 0.977808 | 9.54e-07 |  |
| 12 | kda_rec | 0.123 | 0.0841 | 0.0935 | 1.3 | 0.992616 | 9.54e-07 |  |
| 13 | kda_rec | 0.0646 | 0.0841 | 0.0935 | 0.69 | 0.997962 | 0 |  |
| 14 | kda_rec | 0.112 | 0.0841 | 0.0935 | 1.2 | 0.993709 | 0.00016 |  |
| 16 | kda_rec | 0.114 | 0.0841 | 0.0935 | 1.2 | 0.993456 | 0.000118 |  |
| 17 | kda_rec | 0.12 | 0.0841 | 0.0935 | 1.3 | 0.992761 | 0 |  |
| 18 | kda_rec | 0.143 | 0.104 | 0.111 | 1.3 | 0.989968 | 0 |  |
| 20 | kda_rec | 0.147 | 0.117 | 0.128 | 1.2 | 0.989085 | 0 |  |
| 21 | kda_rec | 0.237 | 0.176 | 0.206 | 1.2 | 0.972839 | 0 |  |
| 22 | kda_rec | 0.251 | 0.176 | 0.206 | 1.2 | 0.981713 | 0 |  |
| 24 | kda_rec | 0.284 | 0.176 | 0.206 | 1.4 | 0.963166 | 2.86e-06 |  |
| 25 | kda_rec | 0.192 | 0.176 | 0.206 | 0.93 | 0.981691 | 2.29e-05 |  |
| 26 | kda_rec | 0.233 | 0.193 | 0.206 | 1.1 | 0.973337 | 0 |  |
| 28 | kda_rec | 0.25 | 0.21 | 0.206 | 1.2 | 0.969197 | 3.81e-06 |  |
| 29 | kda_rec | 0.28 | 0.21 | 0.211 | 1.3 | 0.962231 | 0 |  |
| 30 | kda_rec | 0.0745 | 0.21 | 0.211 | 0.35 | 0.997586 | 0 |  |
| 32 | kda_rec | 0.218 | 0.21 | 0.211 | 1 | 0.976246 | 0 |  |
| 33 | kda_rec | 0.298 | 0.221 | 0.214 | 1.3 | 0.957437 | 0 |  |
| 34 | kda_rec | 0.0801 | 0.221 | 0.214 | 0.36 | 0.997012 | 9.54e-07 |  |
| 36 | kda_rec | 0.0611 | 0.221 | 0.214 | 0.28 | 0.998454 | 0 |  |
| 37 | kda_rec | 0.201 | 0.221 | 0.214 | 0.91 | 0.979649 | 0 |  |
| 38 | kda_rec | 0.2 | 0.236 | 0.236 | 0.85 | 0.979952 | 0.000129 |  |
| 40 | kda_rec | 0.208 | 0.236 | 0.236 | 0.88 | 0.979041 | 0.000122 |  |
| 41 | kda_rec | 0.155 | 0.236 | 0.236 | 0.66 | 0.987992 | 0 |  |
| 42 | kda_rec | 0.243 | 0.236 | 0.236 | 1 | 0.971701 | 0 |  |
| 44 | kda_rec | 0.133 | 0.236 | 0.236 | 0.56 | 0.991337 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000714 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0182 | 0 | 0.0292 | 0.62 | 0.999835 | 0.0983 |  |
| 2 | kda_conv | 0.0404 | 0 | 0.0498 | 0.81 | 0.999183 | 0.0463 |  |
| 4 | kda_conv | 0.031 | 0.0071 | 0.0498 | 0.62 | 0.999523 | 0.0597 |  |
| 5 | kda_conv | 0.0684 | 0.0324 | 0.0621 | 1.1 | 0.997678 | 0.0246 |  |
| 6 | kda_conv | 0.0706 | 0.0441 | 0.0621 | 1.1 | 0.997507 | 0.0238 |  |
| 8 | kda_conv | 0.13 | 0.109 | 0.123 | 1.1 | 0.991568 | 0.0135 |  |
| 9 | kda_conv | 0.12 | 0.109 | 0.123 | 0.98 | 0.992798 | 0.0138 |  |
| 10 | kda_conv | 0.105 | 0.109 | 0.123 | 0.85 | 0.994483 | 0.0171 |  |
| 12 | kda_conv | 0.135 | 0.12 | 0.131 | 1 | 0.990946 | 0.0131 |  |
| 13 | kda_conv | 0.142 | 0.133 | 0.149 | 0.95 | 0.990033 | 0.0115 |  |
| 14 | kda_conv | 0.119 | 0.133 | 0.149 | 0.8 | 0.992903 | 0.0144 |  |
| 16 | kda_conv | 0.118 | 0.133 | 0.149 | 0.79 | 0.993104 | 0.0142 |  |
| 17 | kda_conv | 0.159 | 0.16 | 0.188 | 0.85 | 0.987345 | 0.01 |  |
| 18 | kda_conv | 0.171 | 0.179 | 0.355 | 0.48 | 0.985307 | 0.00967 |  |
| 20 | kda_conv | 0.198 | 0.2 | 0.381 | 0.52 | 0.980389 | 0.00922 |  |
| 21 | kda_conv | 0.234 | 0.268 | 0.474 | 0.49 | 0.972412 | 0.00765 |  |
| 22 | kda_conv | 0.252 | 0.281 | 0.474 | 0.53 | 0.967961 | 0.0068 |  |
| 24 | kda_conv | 0.237 | 0.281 | 0.474 | 0.5 | 0.971791 | 0.00707 |  |
| 25 | kda_conv | 0.241 | 0.281 | 0.474 | 0.51 | 0.970507 | 0.00544 |  |
| 26 | kda_conv | 0.238 | 0.29 | 0.474 | 0.5 | 0.971824 | 0.00685 |  |
| 28 | kda_conv | 0.202 | 0.35 | 0.474 | 0.43 | 0.979435 | 0.00755 |  |
| 29 | kda_conv | 0.215 | 0.439 | 0.474 | 0.45 | 0.976684 | 0.00711 |  |
| 30 | kda_conv | 0.185 | 0.439 | 0.474 | 0.39 | 0.98288 | 0.00822 |  |
| 32 | kda_conv | 0.172 | 0.439 | 0.474 | 0.36 | 0.985186 | 0.0091 |  |
| 33 | kda_conv | 0.215 | 0.541 | 0.647 | 0.33 | 0.977951 | 0.00631 |  |
| 34 | kda_conv | 0.11 | 0.541 | 0.647 | 0.17 | 0.993963 | 0.0124 |  |
| 36 | kda_conv | 0.132 | 0.541 | 0.647 | 0.2 | 0.991197 | 0.0112 |  |
| 37 | kda_conv | 0.147 | 0.541 | 0.647 | 0.23 | 0.98934 | 0.0108 |  |
| 38 | kda_conv | 0.135 | 0.565 | 0.647 | 0.21 | 0.990908 | 0.0111 |  |
| 40 | kda_conv | 0.137 | 0.616 | 0.669 | 0.2 | 0.990949 | 0.0107 |  |
| 41 | kda_conv | 0.122 | 0.616 | 0.669 | 0.18 | 0.992673 | 0.0121 |  |
| 42 | kda_conv | 0.107 | 0.616 | 0.669 | 0.16 | 0.994302 | 0.0125 |  |
| 44 | kda_conv | 0.115 | 0.616 | 0.669 | 0.17 | 0.993407 | 0.0132 |  |


</details>

<details><summary>o_prefix_100003_256 vs v_full_100259: 6 above noise, 0 patterns</summary>

Per layer, `o_prefix_100003_256` vs `v_full_100259`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.104 | 0 | 0.0634 | 1.6 | 0.994597 | 0.292 |  |
| 7 | mla | 0.135 | 0.0865 | 0.121 | 1.1 | 0.99092 | 0.223 |  |
| 11 | mla | 0.114 | 0.0865 | 0.121 | 0.94 | 0.993532 | 0.255 |  |
| 15 | mla | 0.145 | 0.115 | 0.121 | 1.2 | 0.98949 | 0.213 |  |
| 19 | mla | 0.206 | 0.184 | 0.189 | 1.1 | 0.978725 | 0.182 |  |
| 23 | mla | 0.252 | 0.23 | 0.234 | 1.1 | 0.968199 | 0.156 |  |
| 27 | mla | 0.252 | 0.231 | 0.236 | 1.1 | 0.968242 | 0.174 |  |
| 31 | mla | 0.161 | 0.231 | 0.236 | 0.68 | 0.986962 | 0.23 |  |
| 35 | mla | 0.171 | 0.231 | 0.236 | 0.73 | 0.985359 | 0.234 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.97753 | 0.175 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.971089 | 0.145 |  |
| 3 | indexer | 0.0968 | 0 | 0.0364 | 2.7 | 0.995458 | 0.353 |  |
| 7 | indexer | 0.0619 | 0.0369 | 0.047 | 1.3 | 0.998082 | 0.432 |  |
| 11 | indexer | 0.0853 | 0.0488 | 0.055 | 1.6 | 0.996359 | 0.323 |  |
| 15 | indexer | 0.0975 | 0.0588 | 0.0623 | 1.6 | 0.995243 | 0.273 |  |
| 19 | indexer | 0.121 | 0.101 | 0.106 | 1.1 | 0.992712 | 0.332 |  |
| 23 | indexer | 0.136 | 0.119 | 0.124 | 1.1 | 0.990756 | 0.313 |  |
| 27 | indexer | 0.131 | 0.119 | 0.124 | 1.1 | 0.991424 | 0.346 |  |
| 31 | indexer | 0.0882 | 0.119 | 0.124 | 0.71 | 0.996108 | 0.362 |  |
| 35 | indexer | 0.0958 | 0.119 | 0.124 | 0.77 | 0.995405 | 0.398 |  |
| 39 | indexer | 0.108 | 0.119 | 0.124 | 0.87 | 0.994178 | 0.375 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.993198 | 0.252 |  |
| 3 | tail | 0.157 | 0 | 0.0368 | 4.3 | 0.987963 | 0.013 | **above noise** |
| 7 | tail | 0.0779 | 0.0551 | 0.0732 | 1.1 | 0.996966 | 0.0182 |  |
| 11 | tail | 0.139 | 0.0883 | 0.0945 | 1.5 | 0.990352 | 0.00781 |  |
| 15 | tail | 0.117 | 0.0883 | 0.0945 | 1.2 | 0.993145 | 0.00911 |  |
| 19 | tail | 0.686 | 0.0894 | 0.114 | 6 | 0.755069 | 0.013 | **above noise** |
| 23 | tail | 0.121 | 0.0894 | 0.114 | 1.1 | 0.992995 | 0.0104 |  |
| 27 | tail | 0.092 | 0.0894 | 0.114 | 0.81 | 0.99609 | 0.0156 |  |
| 31 | tail | 0.0676 | 0.0894 | 0.114 | 0.59 | 0.997953 | 0.0273 |  |
| 35 | tail | 0.0462 | 0.0894 | 0.114 | 0.41 | 0.998984 | 0.0547 |  |
| 39 | tail | 0.0555 | 0.0894 | 0.114 | 0.49 | 0.99847 | 0.0247 |  |
| 43 | tail | 0.0795 | 0.137 | 0.114 | 0.58 | 0.996879 | 0.026 |  |
| 0 | kda_rec | 0.0104 | 0 | 0.00307 | 3.4 | 0.999946 | 2.86e-06 | **above noise** |
| 1 | kda_rec | 0.0551 | 0 | 0.0178 | 3.1 | 0.998498 | 9.54e-07 | **above noise** |
| 2 | kda_rec | 0.107 | 0 | 0.0238 | 4.5 | 0.994459 | 0 | **above noise** |
| 4 | kda_rec | 0.0602 | 0.0129 | 0.0388 | 1.6 | 0.998188 | 1.91e-06 |  |
| 5 | kda_rec | 0.131 | 0.0241 | 0.0388 | 3.4 | 0.992662 | 0 | **above noise** |
| 6 | kda_rec | 0.123 | 0.0363 | 0.0482 | 2.5 | 0.992438 | 0 |  |
| 8 | kda_rec | 0.0518 | 0.0363 | 0.0482 | 1.1 | 0.998676 | 0 |  |
| 9 | kda_rec | 0.105 | 0.0728 | 0.0834 | 1.3 | 0.994478 | 0 |  |
| 10 | kda_rec | 0.219 | 0.0728 | 0.0834 | 2.6 | 0.977869 | 0 |  |
| 12 | kda_rec | 0.123 | 0.0841 | 0.0935 | 1.3 | 0.992598 | 0 |  |
| 13 | kda_rec | 0.0644 | 0.0841 | 0.0935 | 0.69 | 0.997945 | 0 |  |
| 14 | kda_rec | 0.118 | 0.0841 | 0.0935 | 1.3 | 0.992982 | 0.000161 |  |
| 16 | kda_rec | 0.117 | 0.0841 | 0.0935 | 1.3 | 0.993123 | 0.000118 |  |
| 17 | kda_rec | 0.121 | 0.0841 | 0.0935 | 1.3 | 0.992631 | 0 |  |
| 18 | kda_rec | 0.144 | 0.104 | 0.111 | 1.3 | 0.989719 | 0 |  |
| 20 | kda_rec | 0.147 | 0.117 | 0.128 | 1.2 | 0.98907 | 0 |  |
| 21 | kda_rec | 0.238 | 0.176 | 0.206 | 1.2 | 0.97255 | 0 |  |
| 22 | kda_rec | 0.252 | 0.176 | 0.206 | 1.2 | 0.98198 | 0 |  |
| 24 | kda_rec | 0.285 | 0.176 | 0.206 | 1.4 | 0.962847 | 2.86e-06 |  |
| 25 | kda_rec | 0.187 | 0.176 | 0.206 | 0.91 | 0.982813 | 2.29e-05 |  |
| 26 | kda_rec | 0.235 | 0.193 | 0.206 | 1.1 | 0.973265 | 0 |  |
| 28 | kda_rec | 0.244 | 0.21 | 0.206 | 1.2 | 0.970917 | 3.81e-06 |  |
| 29 | kda_rec | 0.266 | 0.21 | 0.211 | 1.3 | 0.965989 | 0 |  |
| 30 | kda_rec | 0.0748 | 0.21 | 0.211 | 0.35 | 0.997601 | 9.54e-07 |  |
| 32 | kda_rec | 0.207 | 0.21 | 0.211 | 0.98 | 0.978715 | 0 |  |
| 33 | kda_rec | 0.287 | 0.221 | 0.214 | 1.3 | 0.960983 | 0 |  |
| 34 | kda_rec | 0.08 | 0.221 | 0.214 | 0.36 | 0.997116 | 9.54e-07 |  |
| 36 | kda_rec | 0.0606 | 0.221 | 0.214 | 0.27 | 0.998543 | 0 |  |
| 37 | kda_rec | 0.187 | 0.221 | 0.214 | 0.84 | 0.982611 | 0 |  |
| 38 | kda_rec | 0.196 | 0.236 | 0.236 | 0.83 | 0.980884 | 0.000129 |  |
| 40 | kda_rec | 0.201 | 0.236 | 0.236 | 0.85 | 0.980839 | 0.000122 |  |
| 41 | kda_rec | 0.154 | 0.236 | 0.236 | 0.65 | 0.988163 | 0 |  |
| 42 | kda_rec | 0.245 | 0.236 | 0.236 | 1 | 0.971534 | 0 |  |
| 44 | kda_rec | 0.135 | 0.236 | 0.236 | 0.57 | 0.991045 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000714 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0183 | 0 | 0.0292 | 0.62 | 0.999834 | 0.101 |  |
| 2 | kda_conv | 0.0399 | 0 | 0.0498 | 0.8 | 0.999205 | 0.0473 |  |
| 4 | kda_conv | 0.03 | 0.0071 | 0.0498 | 0.6 | 0.999576 | 0.0584 |  |
| 5 | kda_conv | 0.0599 | 0.0324 | 0.0621 | 0.96 | 0.998204 | 0.0272 |  |
| 6 | kda_conv | 0.0658 | 0.0441 | 0.0621 | 1.1 | 0.997833 | 0.0249 |  |
| 8 | kda_conv | 0.124 | 0.109 | 0.123 | 1 | 0.99223 | 0.0124 |  |
| 9 | kda_conv | 0.113 | 0.109 | 0.123 | 0.92 | 0.993558 | 0.0146 |  |
| 10 | kda_conv | 0.102 | 0.109 | 0.123 | 0.83 | 0.994806 | 0.0165 |  |
| 12 | kda_conv | 0.136 | 0.12 | 0.131 | 1 | 0.990809 | 0.0131 |  |
| 13 | kda_conv | 0.147 | 0.133 | 0.149 | 0.99 | 0.989215 | 0.0122 |  |
| 14 | kda_conv | 0.141 | 0.133 | 0.149 | 0.94 | 0.990088 | 0.0117 |  |
| 16 | kda_conv | 0.144 | 0.133 | 0.149 | 0.97 | 0.989633 | 0.0128 |  |
| 17 | kda_conv | 0.18 | 0.16 | 0.188 | 0.96 | 0.983723 | 0.0112 |  |
| 18 | kda_conv | 0.254 | 0.179 | 0.355 | 0.71 | 0.967315 | 0.00875 |  |
| 20 | kda_conv | 0.241 | 0.2 | 0.381 | 0.63 | 0.970561 | 0.00853 |  |
| 21 | kda_conv | 0.248 | 0.268 | 0.474 | 0.52 | 0.968857 | 0.0062 |  |
| 22 | kda_conv | 0.248 | 0.281 | 0.474 | 0.52 | 0.96888 | 0.00711 |  |
| 24 | kda_conv | 0.23 | 0.281 | 0.474 | 0.49 | 0.973323 | 0.00635 |  |
| 25 | kda_conv | 0.224 | 0.281 | 0.474 | 0.47 | 0.974696 | 0.00673 |  |
| 26 | kda_conv | 0.229 | 0.29 | 0.474 | 0.48 | 0.974239 | 0.00629 |  |
| 28 | kda_conv | 0.211 | 0.35 | 0.474 | 0.44 | 0.977896 | 0.00686 |  |
| 29 | kda_conv | 0.236 | 0.439 | 0.474 | 0.5 | 0.971935 | 0.0067 |  |
| 30 | kda_conv | 0.193 | 0.439 | 0.474 | 0.41 | 0.981666 | 0.00814 |  |
| 32 | kda_conv | 0.176 | 0.439 | 0.474 | 0.37 | 0.984371 | 0.00895 |  |
| 33 | kda_conv | 0.233 | 0.541 | 0.647 | 0.36 | 0.974717 | 0.00538 |  |
| 34 | kda_conv | 0.109 | 0.541 | 0.647 | 0.17 | 0.994073 | 0.0121 |  |
| 36 | kda_conv | 0.138 | 0.541 | 0.647 | 0.21 | 0.990389 | 0.0112 |  |
| 37 | kda_conv | 0.158 | 0.541 | 0.647 | 0.24 | 0.987717 | 0.00906 |  |
| 38 | kda_conv | 0.137 | 0.565 | 0.647 | 0.21 | 0.990583 | 0.0106 |  |
| 40 | kda_conv | 0.143 | 0.616 | 0.669 | 0.21 | 0.990189 | 0.00958 |  |
| 41 | kda_conv | 0.123 | 0.616 | 0.669 | 0.18 | 0.992476 | 0.0108 |  |
| 42 | kda_conv | 0.104 | 0.616 | 0.669 | 0.15 | 0.994645 | 0.0123 |  |
| 44 | kda_conv | 0.121 | 0.616 | 0.669 | 0.18 | 0.99273 | 0.0125 |  |


</details>

<details><summary>o_full_100006 vs v_full_100006: 5 above noise, 0 patterns</summary>

Per layer, `o_full_100006` vs `v_full_100006`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.104 | 0 | 0.0634 | 1.6 | 0.994613 | 0.292 |  |
| 7 | mla | 0.135 | 0.0865 | 0.121 | 1.1 | 0.990922 | 0.223 |  |
| 11 | mla | 0.113 | 0.0865 | 0.121 | 0.94 | 0.993562 | 0.256 |  |
| 15 | mla | 0.145 | 0.115 | 0.121 | 1.2 | 0.989539 | 0.213 |  |
| 19 | mla | 0.206 | 0.184 | 0.189 | 1.1 | 0.978743 | 0.182 |  |
| 23 | mla | 0.253 | 0.23 | 0.234 | 1.1 | 0.967979 | 0.156 |  |
| 27 | mla | 0.253 | 0.231 | 0.236 | 1.1 | 0.968014 | 0.173 |  |
| 31 | mla | 0.162 | 0.231 | 0.236 | 0.68 | 0.986943 | 0.229 |  |
| 35 | mla | 0.171 | 0.231 | 0.236 | 0.72 | 0.985371 | 0.233 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.977583 | 0.175 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.971121 | 0.145 |  |
| 3 | indexer | 0.0966 | 0 | 0.0364 | 2.7 | 0.995476 | 0.353 |  |
| 7 | indexer | 0.062 | 0.0369 | 0.047 | 1.3 | 0.998076 | 0.432 |  |
| 11 | indexer | 0.0847 | 0.0488 | 0.055 | 1.5 | 0.996411 | 0.324 |  |
| 15 | indexer | 0.0964 | 0.0588 | 0.0623 | 1.5 | 0.995348 | 0.276 |  |
| 19 | indexer | 0.121 | 0.101 | 0.106 | 1.1 | 0.99274 | 0.329 |  |
| 23 | indexer | 0.137 | 0.119 | 0.124 | 1.1 | 0.990587 | 0.309 |  |
| 27 | indexer | 0.131 | 0.119 | 0.124 | 1.1 | 0.991389 | 0.343 |  |
| 31 | indexer | 0.0888 | 0.119 | 0.124 | 0.71 | 0.996055 | 0.358 |  |
| 35 | indexer | 0.0977 | 0.119 | 0.124 | 0.79 | 0.99522 | 0.394 |  |
| 39 | indexer | 0.109 | 0.119 | 0.124 | 0.88 | 0.994011 | 0.372 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.99318 | 0.251 |  |
| 3 | tail | 0.154 | 0 | 0.0368 | 4.2 | 0.988437 | 0.0156 | **above noise** |
| 7 | tail | 0.103 | 0.0551 | 0.0732 | 1.4 | 0.994686 | 0.0156 |  |
| 11 | tail | 0.115 | 0.0883 | 0.0945 | 1.2 | 0.993341 | 0.0176 |  |
| 15 | tail | 0.112 | 0.0883 | 0.0945 | 1.2 | 0.993707 | 0.0176 |  |
| 19 | tail | 0.0768 | 0.0894 | 0.114 | 0.67 | 0.997049 | 0.0234 |  |
| 23 | tail | 0.0738 | 0.0894 | 0.114 | 0.65 | 0.997274 | 0.0137 |  |
| 27 | tail | 0.056 | 0.0894 | 0.114 | 0.49 | 0.998477 | 0.0449 |  |
| 31 | tail | 0.0548 | 0.0894 | 0.114 | 0.48 | 0.998509 | 0.0449 |  |
| 35 | tail | 0.04 | 0.0894 | 0.114 | 0.35 | 0.999253 | 0.0508 |  |
| 39 | tail | 0.0731 | 0.0894 | 0.114 | 0.64 | 0.99734 | 0.0332 |  |
| 43 | tail | 0.13 | 0.137 | 0.114 | 0.95 | 0.99148 | 0.00781 |  |
| 0 | kda_rec | 0.0107 | 0 | 0.00307 | 3.5 | 0.999942 | 4.77e-06 | **above noise** |
| 1 | kda_rec | 0.0567 | 0 | 0.0178 | 3.2 | 0.99842 | 9.54e-07 | **above noise** |
| 2 | kda_rec | 0.107 | 0 | 0.0238 | 4.5 | 0.994348 | 0 | **above noise** |
| 4 | kda_rec | 0.0638 | 0.0129 | 0.0388 | 1.6 | 0.997973 | 9.54e-07 |  |
| 5 | kda_rec | 0.129 | 0.0241 | 0.0388 | 3.3 | 0.992876 | 1.91e-06 | **above noise** |
| 6 | kda_rec | 0.13 | 0.0363 | 0.0482 | 2.7 | 0.991523 | 0 |  |
| 8 | kda_rec | 0.0538 | 0.0363 | 0.0482 | 1.1 | 0.998568 | 0 |  |
| 9 | kda_rec | 0.107 | 0.0728 | 0.0834 | 1.3 | 0.994306 | 9.54e-07 |  |
| 10 | kda_rec | 0.221 | 0.0728 | 0.0834 | 2.6 | 0.977336 | 0 |  |
| 12 | kda_rec | 0.117 | 0.0841 | 0.0935 | 1.3 | 0.993308 | 9.54e-07 |  |
| 13 | kda_rec | 0.0639 | 0.0841 | 0.0935 | 0.68 | 0.997959 | 0 |  |
| 14 | kda_rec | 0.114 | 0.0841 | 0.0935 | 1.2 | 0.993502 | 4.86e-05 |  |
| 16 | kda_rec | 0.116 | 0.0841 | 0.0935 | 1.2 | 0.993254 | 0.000113 |  |
| 17 | kda_rec | 0.114 | 0.0841 | 0.0935 | 1.2 | 0.993532 | 0 |  |
| 18 | kda_rec | 0.139 | 0.104 | 0.111 | 1.2 | 0.990395 | 9.54e-07 |  |
| 20 | kda_rec | 0.118 | 0.117 | 0.128 | 0.92 | 0.993043 | 0 |  |
| 21 | kda_rec | 0.185 | 0.176 | 0.206 | 0.9 | 0.983333 | 0 |  |
| 22 | kda_rec | 0.262 | 0.176 | 0.206 | 1.3 | 0.980829 | 0 |  |
| 24 | kda_rec | 0.271 | 0.176 | 0.206 | 1.3 | 0.966587 | 9.54e-07 |  |
| 25 | kda_rec | 0.117 | 0.176 | 0.206 | 0.57 | 0.993364 | 4.77e-06 |  |
| 26 | kda_rec | 0.171 | 0.193 | 0.206 | 0.83 | 0.986188 | 0 |  |
| 28 | kda_rec | 0.171 | 0.21 | 0.206 | 0.81 | 0.986021 | 0 |  |
| 29 | kda_rec | 0.181 | 0.21 | 0.211 | 0.86 | 0.985137 | 0 |  |
| 30 | kda_rec | 0.0628 | 0.21 | 0.211 | 0.3 | 0.998396 | 0 |  |
| 32 | kda_rec | 0.135 | 0.21 | 0.211 | 0.64 | 0.991046 | 0 |  |
| 33 | kda_rec | 0.215 | 0.221 | 0.214 | 0.97 | 0.978076 | 0 |  |
| 34 | kda_rec | 0.0791 | 0.221 | 0.214 | 0.36 | 0.997315 | 9.54e-07 |  |
| 36 | kda_rec | 0.0589 | 0.221 | 0.214 | 0.27 | 0.998781 | 0 |  |
| 37 | kda_rec | 0.172 | 0.221 | 0.214 | 0.78 | 0.985269 | 0 |  |
| 38 | kda_rec | 0.193 | 0.236 | 0.236 | 0.82 | 0.981463 | 8.2e-05 |  |
| 40 | kda_rec | 0.194 | 0.236 | 0.236 | 0.82 | 0.982201 | 0.000122 |  |
| 41 | kda_rec | 0.151 | 0.236 | 0.236 | 0.64 | 0.98865 | 9.54e-07 |  |
| 42 | kda_rec | 0.249 | 0.236 | 0.236 | 1.1 | 0.97083 | 0 |  |
| 44 | kda_rec | 0.153 | 0.236 | 0.236 | 0.65 | 0.988571 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000714 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0311 | 0 | 0.0292 | 1.1 | 0.999517 | 0.0634 |  |
| 2 | kda_conv | 0.0442 | 0 | 0.0498 | 0.89 | 0.999021 | 0.0383 |  |
| 4 | kda_conv | 0.0409 | 0.0071 | 0.0498 | 0.82 | 0.999162 | 0.0406 |  |
| 5 | kda_conv | 0.0665 | 0.0324 | 0.0621 | 1.1 | 0.997789 | 0.027 |  |
| 6 | kda_conv | 0.0628 | 0.0441 | 0.0621 | 1 | 0.998032 | 0.0259 |  |
| 8 | kda_conv | 0.113 | 0.109 | 0.123 | 0.92 | 0.993637 | 0.0168 |  |
| 9 | kda_conv | 0.105 | 0.109 | 0.123 | 0.86 | 0.994439 | 0.0168 |  |
| 10 | kda_conv | 0.0804 | 0.109 | 0.123 | 0.65 | 0.996771 | 0.0211 |  |
| 12 | kda_conv | 0.0871 | 0.12 | 0.131 | 0.67 | 0.996261 | 0.02 |  |
| 13 | kda_conv | 0.108 | 0.133 | 0.149 | 0.72 | 0.994211 | 0.0166 |  |
| 14 | kda_conv | 0.0931 | 0.133 | 0.149 | 0.62 | 0.995658 | 0.0167 |  |
| 16 | kda_conv | 0.087 | 0.133 | 0.149 | 0.58 | 0.996219 | 0.0183 |  |
| 17 | kda_conv | 0.103 | 0.16 | 0.188 | 0.55 | 0.994746 | 0.0155 |  |
| 18 | kda_conv | 0.108 | 0.179 | 0.355 | 0.3 | 0.99418 | 0.015 |  |
| 20 | kda_conv | 0.118 | 0.2 | 0.381 | 0.31 | 0.993055 | 0.0137 |  |
| 21 | kda_conv | 0.126 | 0.268 | 0.474 | 0.27 | 0.992034 | 0.0128 |  |
| 22 | kda_conv | 0.131 | 0.281 | 0.474 | 0.28 | 0.991444 | 0.0123 |  |
| 24 | kda_conv | 0.119 | 0.281 | 0.474 | 0.25 | 0.992909 | 0.0137 |  |
| 25 | kda_conv | 0.0874 | 0.281 | 0.474 | 0.18 | 0.996176 | 0.0186 |  |
| 26 | kda_conv | 0.113 | 0.29 | 0.474 | 0.24 | 0.993615 | 0.015 |  |
| 28 | kda_conv | 0.11 | 0.35 | 0.474 | 0.23 | 0.993995 | 0.0141 |  |
| 29 | kda_conv | 0.121 | 0.439 | 0.474 | 0.26 | 0.992771 | 0.0145 |  |
| 30 | kda_conv | 0.0992 | 0.439 | 0.474 | 0.21 | 0.995069 | 0.0148 |  |
| 32 | kda_conv | 0.123 | 0.439 | 0.474 | 0.26 | 0.992411 | 0.0126 |  |
| 33 | kda_conv | 0.136 | 0.541 | 0.647 | 0.21 | 0.990744 | 0.0122 |  |
| 34 | kda_conv | 0.0736 | 0.541 | 0.647 | 0.11 | 0.997287 | 0.0185 |  |
| 36 | kda_conv | 0.0863 | 0.541 | 0.647 | 0.13 | 0.996272 | 0.0173 |  |
| 37 | kda_conv | 0.161 | 0.541 | 0.647 | 0.25 | 0.986912 | 0.0099 |  |
| 38 | kda_conv | 0.184 | 0.565 | 0.647 | 0.29 | 0.983001 | 0.00876 |  |
| 40 | kda_conv | 0.225 | 0.616 | 0.669 | 0.34 | 0.974712 | 0.00715 |  |
| 41 | kda_conv | 0.191 | 0.616 | 0.669 | 0.28 | 0.981791 | 0.00827 |  |
| 42 | kda_conv | 0.188 | 0.616 | 0.669 | 0.28 | 0.982227 | 0.00808 |  |
| 44 | kda_conv | 0.189 | 0.616 | 0.669 | 0.28 | 0.982009 | 0.0089 |  |


</details>

<details><summary>o_prefix_100003_3 vs v_full_100006: 5 above noise, 0 patterns</summary>

Per layer, `o_prefix_100003_3` vs `v_full_100006`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.104 | 0 | 0.0634 | 1.6 | 0.994612 | 0.292 |  |
| 7 | mla | 0.135 | 0.0865 | 0.121 | 1.1 | 0.990922 | 0.223 |  |
| 11 | mla | 0.113 | 0.0865 | 0.121 | 0.94 | 0.993562 | 0.256 |  |
| 15 | mla | 0.145 | 0.115 | 0.121 | 1.2 | 0.989539 | 0.213 |  |
| 19 | mla | 0.206 | 0.184 | 0.189 | 1.1 | 0.978743 | 0.182 |  |
| 23 | mla | 0.253 | 0.23 | 0.234 | 1.1 | 0.967979 | 0.156 |  |
| 27 | mla | 0.253 | 0.231 | 0.236 | 1.1 | 0.968014 | 0.173 |  |
| 31 | mla | 0.162 | 0.231 | 0.236 | 0.68 | 0.986943 | 0.229 |  |
| 35 | mla | 0.171 | 0.231 | 0.236 | 0.72 | 0.985371 | 0.233 |  |
| 39 | mla | 0.212 | 0.231 | 0.236 | 0.9 | 0.977583 | 0.175 |  |
| 43 | mla | 0.24 | 0.231 | 0.236 | 1 | 0.971121 | 0.145 |  |
| 3 | indexer | 0.0966 | 0 | 0.0364 | 2.7 | 0.995476 | 0.353 |  |
| 7 | indexer | 0.062 | 0.0369 | 0.047 | 1.3 | 0.998076 | 0.432 |  |
| 11 | indexer | 0.0847 | 0.0488 | 0.055 | 1.5 | 0.996411 | 0.324 |  |
| 15 | indexer | 0.0964 | 0.0588 | 0.0623 | 1.5 | 0.995348 | 0.276 |  |
| 19 | indexer | 0.121 | 0.101 | 0.106 | 1.1 | 0.99274 | 0.329 |  |
| 23 | indexer | 0.137 | 0.119 | 0.124 | 1.1 | 0.990587 | 0.309 |  |
| 27 | indexer | 0.131 | 0.119 | 0.124 | 1.1 | 0.991389 | 0.343 |  |
| 31 | indexer | 0.0888 | 0.119 | 0.124 | 0.71 | 0.996055 | 0.358 |  |
| 35 | indexer | 0.0977 | 0.119 | 0.124 | 0.79 | 0.99522 | 0.394 |  |
| 39 | indexer | 0.109 | 0.119 | 0.124 | 0.88 | 0.994011 | 0.372 |  |
| 43 | indexer | 0.117 | 0.119 | 0.124 | 0.94 | 0.99318 | 0.251 |  |
| 3 | tail | 0.155 | 0 | 0.0368 | 4.2 | 0.988185 | 0.00781 | **above noise** |
| 7 | tail | 0.109 | 0.0551 | 0.0732 | 1.5 | 0.99411 | 0.0215 |  |
| 11 | tail | 0.118 | 0.0883 | 0.0945 | 1.3 | 0.992962 | 0.0156 |  |
| 15 | tail | 0.111 | 0.0883 | 0.0945 | 1.2 | 0.993868 | 0.0195 |  |
| 19 | tail | 0.0795 | 0.0894 | 0.114 | 0.7 | 0.996839 | 0.0273 |  |
| 23 | tail | 0.071 | 0.0894 | 0.114 | 0.62 | 0.997477 | 0.0312 |  |
| 27 | tail | 0.0526 | 0.0894 | 0.114 | 0.46 | 0.998632 | 0.0625 |  |
| 31 | tail | 0.0556 | 0.0894 | 0.114 | 0.49 | 0.998466 | 0.0371 |  |
| 35 | tail | 0.0416 | 0.0894 | 0.114 | 0.36 | 0.999239 | 0.0391 |  |
| 39 | tail | 0.073 | 0.0894 | 0.114 | 0.64 | 0.997363 | 0.0273 |  |
| 43 | tail | 0.137 | 0.137 | 0.114 | 1 | 0.99052 | 0.00977 |  |
| 0 | kda_rec | 0.0108 | 0 | 0.00307 | 3.5 | 0.999942 | 1.91e-06 | **above noise** |
| 1 | kda_rec | 0.0567 | 0 | 0.0178 | 3.2 | 0.998422 | 9.54e-07 | **above noise** |
| 2 | kda_rec | 0.107 | 0 | 0.0238 | 4.5 | 0.994348 | 0 | **above noise** |
| 4 | kda_rec | 0.0642 | 0.0129 | 0.0388 | 1.7 | 0.997944 | 0 |  |
| 5 | kda_rec | 0.129 | 0.0241 | 0.0388 | 3.3 | 0.992821 | 0 | **above noise** |
| 6 | kda_rec | 0.13 | 0.0363 | 0.0482 | 2.7 | 0.991503 | 0 |  |
| 8 | kda_rec | 0.054 | 0.0363 | 0.0482 | 1.1 | 0.998561 | 0 |  |
| 9 | kda_rec | 0.106 | 0.0728 | 0.0834 | 1.3 | 0.994327 | 0 |  |
| 10 | kda_rec | 0.221 | 0.0728 | 0.0834 | 2.7 | 0.977306 | 0 |  |
| 12 | kda_rec | 0.117 | 0.0841 | 0.0935 | 1.3 | 0.993242 | 0 |  |
| 13 | kda_rec | 0.0639 | 0.0841 | 0.0935 | 0.68 | 0.997957 | 0 |  |
| 14 | kda_rec | 0.114 | 0.0841 | 0.0935 | 1.2 | 0.993474 | 4.86e-05 |  |
| 16 | kda_rec | 0.117 | 0.0841 | 0.0935 | 1.2 | 0.993248 | 0.000113 |  |
| 17 | kda_rec | 0.114 | 0.0841 | 0.0935 | 1.2 | 0.993518 | 0 |  |
| 18 | kda_rec | 0.139 | 0.104 | 0.111 | 1.3 | 0.990306 | 0 |  |
| 20 | kda_rec | 0.119 | 0.117 | 0.128 | 0.93 | 0.992912 | 0 |  |
| 21 | kda_rec | 0.186 | 0.176 | 0.206 | 0.9 | 0.983199 | 9.54e-07 |  |
| 22 | kda_rec | 0.262 | 0.176 | 0.206 | 1.3 | 0.980883 | 0 |  |
| 24 | kda_rec | 0.271 | 0.176 | 0.206 | 1.3 | 0.966609 | 1.91e-06 |  |
| 25 | kda_rec | 0.117 | 0.176 | 0.206 | 0.57 | 0.993309 | 3.81e-06 |  |
| 26 | kda_rec | 0.171 | 0.193 | 0.206 | 0.83 | 0.986176 | 0 |  |
| 28 | kda_rec | 0.17 | 0.21 | 0.206 | 0.81 | 0.986082 | 0 |  |
| 29 | kda_rec | 0.18 | 0.21 | 0.211 | 0.85 | 0.985173 | 0 |  |
| 30 | kda_rec | 0.0628 | 0.21 | 0.211 | 0.3 | 0.998397 | 0 |  |
| 32 | kda_rec | 0.135 | 0.21 | 0.211 | 0.64 | 0.990944 | 0 |  |
| 33 | kda_rec | 0.214 | 0.221 | 0.214 | 0.97 | 0.978159 | 0 |  |
| 34 | kda_rec | 0.0791 | 0.221 | 0.214 | 0.36 | 0.997315 | 9.54e-07 |  |
| 36 | kda_rec | 0.0588 | 0.221 | 0.214 | 0.27 | 0.998777 | 0 |  |
| 37 | kda_rec | 0.174 | 0.221 | 0.214 | 0.79 | 0.984875 | 0 |  |
| 38 | kda_rec | 0.2 | 0.236 | 0.236 | 0.85 | 0.980245 | 8.2e-05 |  |
| 40 | kda_rec | 0.195 | 0.236 | 0.236 | 0.82 | 0.982005 | 0.000122 |  |
| 41 | kda_rec | 0.153 | 0.236 | 0.236 | 0.65 | 0.988403 | 0 |  |
| 42 | kda_rec | 0.251 | 0.236 | 0.236 | 1.1 | 0.970478 | 0 |  |
| 44 | kda_rec | 0.16 | 0.236 | 0.236 | 0.68 | 0.98751 | 0 |  |
| 0 | kda_conv | 0.000585 | 0 | 0.000714 | 0.82 | 1 | 0.952 |  |
| 1 | kda_conv | 0.031 | 0 | 0.0292 | 1.1 | 0.999519 | 0.0663 |  |
| 2 | kda_conv | 0.0433 | 0 | 0.0498 | 0.87 | 0.999061 | 0.0401 |  |
| 4 | kda_conv | 0.0434 | 0.0071 | 0.0498 | 0.87 | 0.999057 | 0.0382 |  |
| 5 | kda_conv | 0.0733 | 0.0324 | 0.0621 | 1.2 | 0.997317 | 0.024 |  |
| 6 | kda_conv | 0.0654 | 0.0441 | 0.0621 | 1.1 | 0.99788 | 0.026 |  |
| 8 | kda_conv | 0.116 | 0.109 | 0.123 | 0.95 | 0.993232 | 0.0156 |  |
| 9 | kda_conv | 0.106 | 0.109 | 0.123 | 0.86 | 0.994362 | 0.0157 |  |
| 10 | kda_conv | 0.085 | 0.109 | 0.123 | 0.69 | 0.996382 | 0.0207 |  |
| 12 | kda_conv | 0.0885 | 0.12 | 0.131 | 0.68 | 0.996113 | 0.0196 |  |
| 13 | kda_conv | 0.108 | 0.133 | 0.149 | 0.73 | 0.994156 | 0.0157 |  |
| 14 | kda_conv | 0.0943 | 0.133 | 0.149 | 0.63 | 0.995551 | 0.0176 |  |
| 16 | kda_conv | 0.0866 | 0.133 | 0.149 | 0.58 | 0.99625 | 0.0174 |  |
| 17 | kda_conv | 0.105 | 0.16 | 0.188 | 0.56 | 0.994506 | 0.0137 |  |
| 18 | kda_conv | 0.119 | 0.179 | 0.355 | 0.34 | 0.992868 | 0.0137 |  |
| 20 | kda_conv | 0.123 | 0.2 | 0.381 | 0.32 | 0.992493 | 0.0127 |  |
| 21 | kda_conv | 0.128 | 0.268 | 0.474 | 0.27 | 0.991787 | 0.0119 |  |
| 22 | kda_conv | 0.128 | 0.281 | 0.474 | 0.27 | 0.99175 | 0.0131 |  |
| 24 | kda_conv | 0.12 | 0.281 | 0.474 | 0.25 | 0.99286 | 0.013 |  |
| 25 | kda_conv | 0.0847 | 0.281 | 0.474 | 0.18 | 0.996437 | 0.0206 |  |
| 26 | kda_conv | 0.11 | 0.29 | 0.474 | 0.23 | 0.993964 | 0.0153 |  |
| 28 | kda_conv | 0.105 | 0.35 | 0.474 | 0.22 | 0.994513 | 0.015 |  |
| 29 | kda_conv | 0.117 | 0.439 | 0.474 | 0.25 | 0.993207 | 0.0144 |  |
| 30 | kda_conv | 0.097 | 0.439 | 0.474 | 0.2 | 0.995295 | 0.0155 |  |
| 32 | kda_conv | 0.121 | 0.439 | 0.474 | 0.26 | 0.992664 | 0.0132 |  |
| 33 | kda_conv | 0.135 | 0.541 | 0.647 | 0.21 | 0.990854 | 0.012 |  |
| 34 | kda_conv | 0.0727 | 0.541 | 0.647 | 0.11 | 0.997354 | 0.0179 |  |
| 36 | kda_conv | 0.0859 | 0.541 | 0.647 | 0.13 | 0.996304 | 0.0168 |  |
| 37 | kda_conv | 0.158 | 0.541 | 0.647 | 0.24 | 0.987471 | 0.00998 |  |
| 38 | kda_conv | 0.189 | 0.565 | 0.647 | 0.29 | 0.982387 | 0.00902 |  |
| 40 | kda_conv | 0.236 | 0.616 | 0.669 | 0.35 | 0.972301 | 0.00776 |  |
| 41 | kda_conv | 0.209 | 0.616 | 0.669 | 0.31 | 0.978124 | 0.00754 |  |
| 42 | kda_conv | 0.226 | 0.616 | 0.669 | 0.34 | 0.974349 | 0.00728 |  |
| 44 | kda_conv | 0.231 | 0.616 | 0.669 | 0.35 | 0.973222 | 0.00742 |  |


</details>

## 524,288 tokens

| pair (x vs reference) | kind | MLA rel L2 (layer) | MLA min exact bytes | indexer rel L2 (layer) | tail rel L2, valid slots | KDA state rel L2 (layer) | conv rel L2 (layer) | KL | top-1 | top-5 | top-k min overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v_filler_524288 vs v_full_524288 | noise_vllm | 0.217 (23) | 0.165 | 0.111 (23) | —, 0 | 0.158 (40) | 0.211 (21) | 0.000171 | = | 5/5 | 0.596 |
| o_c8192_524288 vs v_full_524288 | audit | 0.282 (27) | 0.128 | 0.299 (3) | —, 0 | 0.494 (21) | 0.328 (21) | 0.000123 | = | 4/5 | 0.407 |
| o_c8001_524288 vs v_full_524288 | audit | 0.28 (23) | 0.129 | 0.298 (3) | —, 0 | 0.487 (21) | 0.334 (21) | 0.000104 | = | 4/5 | 0.405 |
| o_c8001_524288 vs o_c8192_524288 | noise_ours | 0.222 (27) | 0.173 | 0.115 (23) | —, 0 | 0.149 (40) | 0.151 (40) | 8.35e-05 | = | 4/5 | 0.577 |
| o_full_525312 vs v_full_525312 | audit | 0.281 (23) | 0.127 | 0.299 (3) | —, 0 | 0.456 (24) | 0.321 (24) | 0.000586 | = | 4/5 | 0.295 |
| o_prefix_524291_1021 vs v_full_525312 | audit_prefix | 0.281 (23) | 0.127 | 0.299 (3) | —, 0 | 0.456 (24) | 0.348 (21) | 0.000105 | = | 4/5 | 0.285 |
| o_prefix_524291_1021 vs o_full_525312 | prefix_vs_ours_full | 0.00822 (43) | 0.998 | 0.00406 (23) | —, 0 | 0.124 (44) | 0.203 (21) | 0.000685 | = | 5/5 | 0.639 |

<details><summary>o_c8192_524288 vs v_full_524288: 13 above noise, 0 patterns</summary>

Per layer, `o_c8192_524288` vs `v_full_524288`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.278 | 0 | 0.061 | 4.6 | 0.96164 | 0.138 | **above noise** |
| 7 | mla | 0.184 | 0.0877 | 0.127 | 1.5 | 0.983055 | 0.168 |  |
| 11 | mla | 0.195 | 0.0877 | 0.127 | 1.5 | 0.980984 | 0.164 |  |
| 15 | mla | 0.225 | 0.112 | 0.127 | 1.8 | 0.9746 | 0.143 |  |
| 19 | mla | 0.256 | 0.177 | 0.184 | 1.4 | 0.96724 | 0.141 |  |
| 23 | mla | 0.282 | 0.217 | 0.222 | 1.3 | 0.960286 | 0.133 |  |
| 27 | mla | 0.282 | 0.217 | 0.222 | 1.3 | 0.960231 | 0.143 |  |
| 31 | mla | 0.199 | 0.217 | 0.222 | 0.9 | 0.980241 | 0.176 |  |
| 35 | mla | 0.206 | 0.217 | 0.222 | 0.93 | 0.978842 | 0.179 |  |
| 39 | mla | 0.239 | 0.217 | 0.222 | 1.1 | 0.971339 | 0.147 |  |
| 43 | mla | 0.261 | 0.217 | 0.222 | 1.2 | 0.966079 | 0.128 |  |
| 3 | indexer | 0.299 | 0 | 0.035 | 8.5 | 0.958789 | 0.139 | **above noise** |
| 7 | indexer | 0.114 | 0.0377 | 0.0493 | 2.3 | 0.993465 | 0.273 |  |
| 11 | indexer | 0.186 | 0.05 | 0.0576 | 3.2 | 0.982624 | 0.166 | **above noise** |
| 15 | indexer | 0.21 | 0.0591 | 0.0632 | 3.3 | 0.977929 | 0.136 | **above noise** |
| 19 | indexer | 0.195 | 0.0971 | 0.102 | 1.9 | 0.981122 | 0.188 |  |
| 23 | indexer | 0.172 | 0.111 | 0.115 | 1.5 | 0.985338 | 0.218 |  |
| 27 | indexer | 0.173 | 0.111 | 0.115 | 1.5 | 0.984994 | 0.23 |  |
| 31 | indexer | 0.137 | 0.111 | 0.115 | 1.2 | 0.990575 | 0.238 |  |
| 35 | indexer | 0.135 | 0.111 | 0.115 | 1.2 | 0.990893 | 0.269 |  |
| 39 | indexer | 0.147 | 0.111 | 0.115 | 1.3 | 0.989071 | 0.248 |  |
| 43 | indexer | 0.168 | 0.111 | 0.115 | 1.5 | 0.985798 | 0.178 |  |
| 0 | kda_rec | 0.00991 | 0 | 0.00296 | 3.3 | 0.999951 | 2.86e-06 | **above noise** |
| 1 | kda_rec | 0.208 | 0 | 0.0149 | 14 | 0.978649 | 1.91e-06 | **above noise** |
| 2 | kda_rec | 0.301 | 0 | 0.0188 | 16 | 0.959137 | 9.54e-07 | **above noise** |
| 4 | kda_rec | 0.125 | 0.0127 | 0.0365 | 3.4 | 0.992149 | 0 | **above noise** |
| 5 | kda_rec | 0.194 | 0.0329 | 0.0365 | 5.3 | 0.981125 | 1.91e-06 | **above noise** |
| 6 | kda_rec | 0.308 | 0.0397 | 0.0469 | 6.6 | 0.952003 | 0 | **above noise** |
| 8 | kda_rec | 0.0935 | 0.0397 | 0.0469 | 2 | 0.995717 | 0 |  |
| 9 | kda_rec | 0.164 | 0.0613 | 0.0744 | 2.2 | 0.986556 | 0 |  |
| 10 | kda_rec | 0.372 | 0.0613 | 0.0744 | 5 | 0.932678 | 0 | **above noise** |
| 12 | kda_rec | 0.192 | 0.0774 | 0.0833 | 2.3 | 0.982154 | 9.54e-07 |  |
| 13 | kda_rec | 0.126 | 0.0774 | 0.0833 | 1.5 | 0.992334 | 9.54e-07 |  |
| 14 | kda_rec | 0.212 | 0.0774 | 0.0833 | 2.5 | 0.977629 | 0.000242 |  |
| 16 | kda_rec | 0.195 | 0.0774 | 0.0833 | 2.3 | 0.980997 | 0.000121 |  |
| 17 | kda_rec | 0.198 | 0.0774 | 0.0833 | 2.4 | 0.980455 | 0 |  |
| 18 | kda_rec | 0.243 | 0.0824 | 0.0833 | 2.9 | 0.970121 | 0 |  |
| 20 | kda_rec | 0.27 | 0.0932 | 0.0924 | 2.9 | 0.962801 | 9.54e-07 |  |
| 21 | kda_rec | 0.494 | 0.134 | 0.112 | 3.7 | 0.891206 | 0 | **above noise** |
| 22 | kda_rec | 0.328 | 0.134 | 0.112 | 2.4 | 0.964437 | 0 |  |
| 24 | kda_rec | 0.473 | 0.134 | 0.112 | 3.5 | 0.889963 | 3.81e-06 | **above noise** |
| 25 | kda_rec | 0.237 | 0.134 | 0.112 | 1.8 | 0.97165 | 1.91e-06 |  |
| 26 | kda_rec | 0.35 | 0.134 | 0.112 | 2.6 | 0.936816 | 0 |  |
| 28 | kda_rec | 0.361 | 0.151 | 0.13 | 2.4 | 0.932492 | 1.91e-06 |  |
| 29 | kda_rec | 0.281 | 0.151 | 0.13 | 1.9 | 0.959922 | 0 |  |
| 30 | kda_rec | 0.139 | 0.151 | 0.13 | 0.92 | 0.990262 | 0 |  |
| 32 | kda_rec | 0.284 | 0.151 | 0.13 | 1.9 | 0.95876 | 0 |  |
| 33 | kda_rec | 0.303 | 0.151 | 0.137 | 2 | 0.953666 | 0 |  |
| 34 | kda_rec | 0.154 | 0.151 | 0.137 | 1 | 0.988131 | 9.54e-07 |  |
| 36 | kda_rec | 0.157 | 0.151 | 0.137 | 1 | 0.988002 | 0 |  |
| 37 | kda_rec | 0.256 | 0.151 | 0.145 | 1.7 | 0.968832 | 0 |  |
| 38 | kda_rec | 0.326 | 0.151 | 0.145 | 2.2 | 0.945437 | 0.000111 |  |
| 40 | kda_rec | 0.255 | 0.158 | 0.149 | 1.6 | 0.96686 | 0.000122 |  |
| 41 | kda_rec | 0.234 | 0.158 | 0.149 | 1.5 | 0.972616 | 0 |  |
| 42 | kda_rec | 0.396 | 0.158 | 0.149 | 2.5 | 0.920806 | 9.54e-07 |  |
| 44 | kda_rec | 0.264 | 0.158 | 0.149 | 1.7 | 0.964486 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000414 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0148 | 0 | 0.0137 | 1.1 | 0.999891 | 0.119 |  |
| 2 | kda_conv | 0.0441 | 0 | 0.0309 | 1.4 | 0.999029 | 0.0392 |  |
| 4 | kda_conv | 0.0423 | 0.00588 | 0.0309 | 1.4 | 0.999133 | 0.0368 |  |
| 5 | kda_conv | 0.0697 | 0.0335 | 0.0523 | 1.3 | 0.997601 | 0.0253 |  |
| 6 | kda_conv | 0.0717 | 0.0388 | 0.0601 | 1.2 | 0.997448 | 0.0227 |  |
| 8 | kda_conv | 0.124 | 0.0786 | 0.0983 | 1.3 | 0.992306 | 0.0142 |  |
| 9 | kda_conv | 0.111 | 0.0786 | 0.0983 | 1.1 | 0.993872 | 0.0154 |  |
| 10 | kda_conv | 0.0961 | 0.0786 | 0.0983 | 0.98 | 0.995428 | 0.0186 |  |
| 12 | kda_conv | 0.129 | 0.0786 | 0.0983 | 1.3 | 0.991926 | 0.0132 |  |
| 13 | kda_conv | 0.114 | 0.088 | 0.0983 | 1.2 | 0.993623 | 0.0145 |  |
| 14 | kda_conv | 0.115 | 0.089 | 0.0983 | 1.2 | 0.993435 | 0.0135 |  |
| 16 | kda_conv | 0.119 | 0.089 | 0.0983 | 1.2 | 0.993037 | 0.0137 |  |
| 17 | kda_conv | 0.142 | 0.108 | 0.106 | 1.3 | 0.989915 | 0.0115 |  |
| 18 | kda_conv | 0.199 | 0.122 | 0.121 | 1.6 | 0.980265 | 0.00803 |  |
| 20 | kda_conv | 0.3 | 0.162 | 0.132 | 1.8 | 0.957336 | 0.00663 |  |
| 21 | kda_conv | 0.328 | 0.211 | 0.142 | 1.6 | 0.94852 | 0.00597 |  |
| 22 | kda_conv | 0.289 | 0.211 | 0.142 | 1.4 | 0.960122 | 0.00601 |  |
| 24 | kda_conv | 0.291 | 0.211 | 0.142 | 1.4 | 0.960578 | 0.00555 |  |
| 25 | kda_conv | 0.213 | 0.211 | 0.142 | 1 | 0.97941 | 0.00764 |  |
| 26 | kda_conv | 0.264 | 0.211 | 0.142 | 1.2 | 0.965901 | 0.00658 |  |
| 28 | kda_conv | 0.215 | 0.211 | 0.142 | 1 | 0.977697 | 0.00689 |  |
| 29 | kda_conv | 0.183 | 0.211 | 0.142 | 0.87 | 0.985131 | 0.00917 |  |
| 30 | kda_conv | 0.221 | 0.211 | 0.142 | 1 | 0.975438 | 0.00701 |  |
| 32 | kda_conv | 0.186 | 0.211 | 0.142 | 0.88 | 0.983755 | 0.00799 |  |
| 33 | kda_conv | 0.208 | 0.211 | 0.142 | 0.99 | 0.979967 | 0.0077 |  |
| 34 | kda_conv | 0.156 | 0.211 | 0.142 | 0.74 | 0.987925 | 0.00787 |  |
| 36 | kda_conv | 0.174 | 0.211 | 0.142 | 0.83 | 0.984775 | 0.00795 |  |
| 37 | kda_conv | 0.195 | 0.211 | 0.142 | 0.92 | 0.980896 | 0.00781 |  |
| 38 | kda_conv | 0.228 | 0.211 | 0.144 | 1.1 | 0.973685 | 0.00647 |  |
| 40 | kda_conv | 0.237 | 0.211 | 0.151 | 1.1 | 0.972546 | 0.00688 |  |
| 41 | kda_conv | 0.222 | 0.211 | 0.151 | 1.1 | 0.975762 | 0.00674 |  |
| 42 | kda_conv | 0.223 | 0.211 | 0.151 | 1.1 | 0.974894 | 0.007 |  |
| 44 | kda_conv | 0.214 | 0.211 | 0.151 | 1 | 0.977293 | 0.0073 |  |


</details>

<details><summary>o_c8001_524288 vs v_full_524288: 13 above noise, 0 patterns</summary>

Per layer, `o_c8001_524288` vs `v_full_524288`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.278 | 0 | 0.061 | 4.6 | 0.961647 | 0.138 | **above noise** |
| 7 | mla | 0.184 | 0.0877 | 0.127 | 1.5 | 0.983026 | 0.168 |  |
| 11 | mla | 0.194 | 0.0877 | 0.127 | 1.5 | 0.981255 | 0.165 |  |
| 15 | mla | 0.224 | 0.112 | 0.127 | 1.8 | 0.974819 | 0.144 |  |
| 19 | mla | 0.255 | 0.177 | 0.184 | 1.4 | 0.967534 | 0.141 |  |
| 23 | mla | 0.28 | 0.217 | 0.222 | 1.3 | 0.960745 | 0.135 |  |
| 27 | mla | 0.28 | 0.217 | 0.222 | 1.3 | 0.960748 | 0.144 |  |
| 31 | mla | 0.196 | 0.217 | 0.222 | 0.89 | 0.980731 | 0.179 |  |
| 35 | mla | 0.203 | 0.217 | 0.222 | 0.92 | 0.979338 | 0.181 |  |
| 39 | mla | 0.237 | 0.217 | 0.222 | 1.1 | 0.971939 | 0.148 |  |
| 43 | mla | 0.258 | 0.217 | 0.222 | 1.2 | 0.966671 | 0.129 |  |
| 3 | indexer | 0.298 | 0 | 0.035 | 8.5 | 0.958803 | 0.139 | **above noise** |
| 7 | indexer | 0.114 | 0.0377 | 0.0493 | 2.3 | 0.993448 | 0.272 |  |
| 11 | indexer | 0.184 | 0.05 | 0.0576 | 3.2 | 0.982973 | 0.168 | **above noise** |
| 15 | indexer | 0.208 | 0.0591 | 0.0632 | 3.3 | 0.97825 | 0.138 | **above noise** |
| 19 | indexer | 0.193 | 0.0971 | 0.102 | 1.9 | 0.981432 | 0.191 |  |
| 23 | indexer | 0.169 | 0.111 | 0.115 | 1.5 | 0.985725 | 0.224 |  |
| 27 | indexer | 0.17 | 0.111 | 0.115 | 1.5 | 0.98548 | 0.235 |  |
| 31 | indexer | 0.134 | 0.111 | 0.115 | 1.2 | 0.99103 | 0.244 |  |
| 35 | indexer | 0.132 | 0.111 | 0.115 | 1.1 | 0.991298 | 0.276 |  |
| 39 | indexer | 0.144 | 0.111 | 0.115 | 1.3 | 0.989567 | 0.255 |  |
| 43 | indexer | 0.164 | 0.111 | 0.115 | 1.4 | 0.986569 | 0.183 |  |
| 0 | kda_rec | 0.00999 | 0 | 0.00296 | 3.4 | 0.99995 | 3.81e-06 | **above noise** |
| 1 | kda_rec | 0.208 | 0 | 0.0149 | 14 | 0.97865 | 2.86e-06 | **above noise** |
| 2 | kda_rec | 0.3 | 0 | 0.0188 | 16 | 0.959299 | 0 | **above noise** |
| 4 | kda_rec | 0.125 | 0.0127 | 0.0365 | 3.4 | 0.992151 | 9.54e-07 | **above noise** |
| 5 | kda_rec | 0.194 | 0.0329 | 0.0365 | 5.3 | 0.981293 | 0 | **above noise** |
| 6 | kda_rec | 0.307 | 0.0397 | 0.0469 | 6.5 | 0.952135 | 0 | **above noise** |
| 8 | kda_rec | 0.0921 | 0.0397 | 0.0469 | 2 | 0.995857 | 0 |  |
| 9 | kda_rec | 0.165 | 0.0613 | 0.0744 | 2.2 | 0.986316 | 0 |  |
| 10 | kda_rec | 0.372 | 0.0613 | 0.0744 | 5 | 0.932724 | 0 | **above noise** |
| 12 | kda_rec | 0.191 | 0.0774 | 0.0833 | 2.3 | 0.982405 | 0 |  |
| 13 | kda_rec | 0.127 | 0.0774 | 0.0833 | 1.5 | 0.992225 | 0 |  |
| 14 | kda_rec | 0.214 | 0.0774 | 0.0833 | 2.6 | 0.977127 | 0.000242 |  |
| 16 | kda_rec | 0.196 | 0.0774 | 0.0833 | 2.4 | 0.980887 | 0.000121 |  |
| 17 | kda_rec | 0.197 | 0.0774 | 0.0833 | 2.4 | 0.980603 | 0 |  |
| 18 | kda_rec | 0.239 | 0.0824 | 0.0833 | 2.9 | 0.971143 | 0 |  |
| 20 | kda_rec | 0.266 | 0.0932 | 0.0924 | 2.9 | 0.96393 | 0 |  |
| 21 | kda_rec | 0.487 | 0.134 | 0.112 | 3.6 | 0.894016 | 0 | **above noise** |
| 22 | kda_rec | 0.314 | 0.134 | 0.112 | 2.3 | 0.966811 | 0 |  |
| 24 | kda_rec | 0.46 | 0.134 | 0.112 | 3.4 | 0.895461 | 3.81e-06 | **above noise** |
| 25 | kda_rec | 0.233 | 0.134 | 0.112 | 1.7 | 0.972751 | 1.91e-06 |  |
| 26 | kda_rec | 0.342 | 0.134 | 0.112 | 2.6 | 0.93967 | 0 |  |
| 28 | kda_rec | 0.357 | 0.151 | 0.13 | 2.4 | 0.933954 | 1.91e-06 |  |
| 29 | kda_rec | 0.277 | 0.151 | 0.13 | 1.8 | 0.961048 | 0 |  |
| 30 | kda_rec | 0.135 | 0.151 | 0.13 | 0.9 | 0.990862 | 0 |  |
| 32 | kda_rec | 0.278 | 0.151 | 0.13 | 1.8 | 0.960637 | 9.54e-07 |  |
| 33 | kda_rec | 0.301 | 0.151 | 0.137 | 2 | 0.954446 | 0 |  |
| 34 | kda_rec | 0.151 | 0.151 | 0.137 | 1 | 0.988655 | 9.54e-07 |  |
| 36 | kda_rec | 0.153 | 0.151 | 0.137 | 1 | 0.988709 | 0 |  |
| 37 | kda_rec | 0.251 | 0.151 | 0.145 | 1.7 | 0.970194 | 0 |  |
| 38 | kda_rec | 0.318 | 0.151 | 0.145 | 2.1 | 0.94817 | 0.000111 |  |
| 40 | kda_rec | 0.247 | 0.158 | 0.149 | 1.6 | 0.968917 | 0.000122 |  |
| 41 | kda_rec | 0.228 | 0.158 | 0.149 | 1.4 | 0.973923 | 0 |  |
| 42 | kda_rec | 0.391 | 0.158 | 0.149 | 2.5 | 0.92301 | 0 |  |
| 44 | kda_rec | 0.258 | 0.158 | 0.149 | 1.6 | 0.966352 | 0 |  |
| 0 | kda_conv | 0.000414 | 0 | 0.000414 | 1 | 1 | 0.955 |  |
| 1 | kda_conv | 0.0148 | 0 | 0.0137 | 1.1 | 0.999892 | 0.122 |  |
| 2 | kda_conv | 0.0442 | 0 | 0.0309 | 1.4 | 0.999025 | 0.0379 |  |
| 4 | kda_conv | 0.0433 | 0.00588 | 0.0309 | 1.4 | 0.999101 | 0.0366 |  |
| 5 | kda_conv | 0.0722 | 0.0335 | 0.0523 | 1.4 | 0.997447 | 0.0228 |  |
| 6 | kda_conv | 0.0676 | 0.0388 | 0.0601 | 1.1 | 0.997716 | 0.024 |  |
| 8 | kda_conv | 0.122 | 0.0786 | 0.0983 | 1.2 | 0.992585 | 0.0157 |  |
| 9 | kda_conv | 0.118 | 0.0786 | 0.0983 | 1.2 | 0.993172 | 0.0135 |  |
| 10 | kda_conv | 0.0974 | 0.0786 | 0.0983 | 0.99 | 0.995293 | 0.0172 |  |
| 12 | kda_conv | 0.126 | 0.0786 | 0.0983 | 1.3 | 0.992345 | 0.0134 |  |
| 13 | kda_conv | 0.117 | 0.088 | 0.0983 | 1.2 | 0.993175 | 0.0135 |  |
| 14 | kda_conv | 0.12 | 0.089 | 0.0983 | 1.2 | 0.992744 | 0.0136 |  |
| 16 | kda_conv | 0.124 | 0.089 | 0.0983 | 1.3 | 0.992397 | 0.0129 |  |
| 17 | kda_conv | 0.147 | 0.108 | 0.106 | 1.4 | 0.989169 | 0.0118 |  |
| 18 | kda_conv | 0.208 | 0.122 | 0.121 | 1.7 | 0.9786 | 0.00935 |  |
| 20 | kda_conv | 0.318 | 0.162 | 0.132 | 2 | 0.952998 | 0.00633 |  |
| 21 | kda_conv | 0.334 | 0.211 | 0.142 | 1.6 | 0.946429 | 0.00693 |  |
| 22 | kda_conv | 0.3 | 0.211 | 0.142 | 1.4 | 0.956293 | 0.00582 |  |
| 24 | kda_conv | 0.292 | 0.211 | 0.142 | 1.4 | 0.959387 | 0.00633 |  |
| 25 | kda_conv | 0.204 | 0.211 | 0.142 | 0.97 | 0.981073 | 0.0084 |  |
| 26 | kda_conv | 0.265 | 0.211 | 0.142 | 1.3 | 0.965924 | 0.00678 |  |
| 28 | kda_conv | 0.214 | 0.211 | 0.142 | 1 | 0.977622 | 0.00769 |  |
| 29 | kda_conv | 0.18 | 0.211 | 0.142 | 0.85 | 0.985439 | 0.0092 |  |
| 30 | kda_conv | 0.224 | 0.211 | 0.142 | 1.1 | 0.974858 | 0.00777 |  |
| 32 | kda_conv | 0.182 | 0.211 | 0.142 | 0.86 | 0.984374 | 0.00846 |  |
| 33 | kda_conv | 0.208 | 0.211 | 0.142 | 0.99 | 0.980232 | 0.00784 |  |
| 34 | kda_conv | 0.157 | 0.211 | 0.142 | 0.74 | 0.987714 | 0.00846 |  |
| 36 | kda_conv | 0.172 | 0.211 | 0.142 | 0.82 | 0.985287 | 0.00807 |  |
| 37 | kda_conv | 0.197 | 0.211 | 0.142 | 0.93 | 0.980636 | 0.00781 |  |
| 38 | kda_conv | 0.229 | 0.211 | 0.144 | 1.1 | 0.973761 | 0.00673 |  |
| 40 | kda_conv | 0.24 | 0.211 | 0.151 | 1.1 | 0.972021 | 0.00636 |  |
| 41 | kda_conv | 0.225 | 0.211 | 0.151 | 1.1 | 0.975447 | 0.00689 |  |
| 42 | kda_conv | 0.224 | 0.211 | 0.151 | 1.1 | 0.974546 | 0.00697 |  |
| 44 | kda_conv | 0.2 | 0.211 | 0.151 | 0.95 | 0.980059 | 0.00739 |  |


</details>

<details><summary>o_full_525312 vs v_full_525312: 13 above noise, 0 patterns</summary>

Per layer, `o_full_525312` vs `v_full_525312`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.279 | 0 | 0.061 | 4.6 | 0.96156 | 0.138 | **above noise** |
| 7 | mla | 0.184 | 0.0877 | 0.127 | 1.5 | 0.983061 | 0.168 |  |
| 11 | mla | 0.194 | 0.0877 | 0.127 | 1.5 | 0.981146 | 0.165 |  |
| 15 | mla | 0.223 | 0.112 | 0.127 | 1.8 | 0.975253 | 0.145 |  |
| 19 | mla | 0.256 | 0.177 | 0.184 | 1.4 | 0.967306 | 0.141 |  |
| 23 | mla | 0.281 | 0.217 | 0.222 | 1.3 | 0.960542 | 0.133 |  |
| 27 | mla | 0.281 | 0.217 | 0.222 | 1.3 | 0.960545 | 0.143 |  |
| 31 | mla | 0.199 | 0.217 | 0.222 | 0.9 | 0.980146 | 0.175 |  |
| 35 | mla | 0.206 | 0.217 | 0.222 | 0.93 | 0.978721 | 0.177 |  |
| 39 | mla | 0.24 | 0.217 | 0.222 | 1.1 | 0.971166 | 0.146 |  |
| 43 | mla | 0.261 | 0.217 | 0.222 | 1.2 | 0.965893 | 0.127 |  |
| 3 | indexer | 0.299 | 0 | 0.035 | 8.5 | 0.958699 | 0.139 | **above noise** |
| 7 | indexer | 0.114 | 0.0377 | 0.0493 | 2.3 | 0.993487 | 0.273 |  |
| 11 | indexer | 0.185 | 0.05 | 0.0576 | 3.2 | 0.982862 | 0.168 | **above noise** |
| 15 | indexer | 0.206 | 0.0591 | 0.0632 | 3.3 | 0.978807 | 0.139 | **above noise** |
| 19 | indexer | 0.195 | 0.0971 | 0.102 | 1.9 | 0.981106 | 0.19 |  |
| 23 | indexer | 0.171 | 0.111 | 0.115 | 1.5 | 0.985373 | 0.218 |  |
| 27 | indexer | 0.173 | 0.111 | 0.115 | 1.5 | 0.985015 | 0.228 |  |
| 31 | indexer | 0.139 | 0.111 | 0.115 | 1.2 | 0.990311 | 0.232 |  |
| 35 | indexer | 0.136 | 0.111 | 0.115 | 1.2 | 0.990718 | 0.262 |  |
| 39 | indexer | 0.149 | 0.111 | 0.115 | 1.3 | 0.98883 | 0.241 |  |
| 43 | indexer | 0.171 | 0.111 | 0.115 | 1.5 | 0.985377 | 0.174 |  |
| 0 | kda_rec | 0.00945 | 0 | 0.00296 | 3.2 | 0.999955 | 5.72e-06 | **above noise** |
| 1 | kda_rec | 0.198 | 0 | 0.0149 | 13 | 0.980559 | 9.54e-07 | **above noise** |
| 2 | kda_rec | 0.296 | 0 | 0.0188 | 16 | 0.960392 | 0 | **above noise** |
| 4 | kda_rec | 0.138 | 0.0127 | 0.0365 | 3.8 | 0.990578 | 9.54e-07 | **above noise** |
| 5 | kda_rec | 0.201 | 0.0329 | 0.0365 | 5.5 | 0.979723 | 9.54e-07 | **above noise** |
| 6 | kda_rec | 0.314 | 0.0397 | 0.0469 | 6.7 | 0.949826 | 0 | **above noise** |
| 8 | kda_rec | 0.0929 | 0.0397 | 0.0469 | 2 | 0.995739 | 0 |  |
| 9 | kda_rec | 0.168 | 0.0613 | 0.0744 | 2.3 | 0.985886 | 0 |  |
| 10 | kda_rec | 0.366 | 0.0613 | 0.0744 | 4.9 | 0.934806 | 0 | **above noise** |
| 12 | kda_rec | 0.199 | 0.0774 | 0.0833 | 2.4 | 0.980869 | 0 |  |
| 13 | kda_rec | 0.129 | 0.0774 | 0.0833 | 1.6 | 0.991981 | 9.54e-07 |  |
| 14 | kda_rec | 0.208 | 0.0774 | 0.0833 | 2.5 | 0.97836 | 0.000241 |  |
| 16 | kda_rec | 0.194 | 0.0774 | 0.0833 | 2.3 | 0.981062 | 0.000122 |  |
| 17 | kda_rec | 0.189 | 0.0774 | 0.0833 | 2.3 | 0.982275 | 9.54e-07 |  |
| 18 | kda_rec | 0.231 | 0.0824 | 0.0833 | 2.8 | 0.973054 | 9.54e-07 |  |
| 20 | kda_rec | 0.248 | 0.0932 | 0.0924 | 2.7 | 0.968753 | 0 |  |
| 21 | kda_rec | 0.447 | 0.134 | 0.112 | 3.3 | 0.91048 | 0 | **above noise** |
| 22 | kda_rec | 0.312 | 0.134 | 0.112 | 2.3 | 0.968076 | 0 |  |
| 24 | kda_rec | 0.456 | 0.134 | 0.112 | 3.4 | 0.898756 | 4.77e-06 | **above noise** |
| 25 | kda_rec | 0.213 | 0.134 | 0.112 | 1.6 | 0.97723 | 5.72e-06 |  |
| 26 | kda_rec | 0.329 | 0.134 | 0.112 | 2.5 | 0.944441 | 0 |  |
| 28 | kda_rec | 0.327 | 0.151 | 0.13 | 2.2 | 0.945002 | 1.91e-06 |  |
| 29 | kda_rec | 0.252 | 0.151 | 0.13 | 1.7 | 0.968072 | 0 |  |
| 30 | kda_rec | 0.137 | 0.151 | 0.13 | 0.91 | 0.990624 | 0 |  |
| 32 | kda_rec | 0.255 | 0.151 | 0.13 | 1.7 | 0.966833 | 0 |  |
| 33 | kda_rec | 0.262 | 0.151 | 0.137 | 1.7 | 0.965513 | 0 |  |
| 34 | kda_rec | 0.145 | 0.151 | 0.137 | 0.96 | 0.989412 | 0 |  |
| 36 | kda_rec | 0.15 | 0.151 | 0.137 | 0.99 | 0.989309 | 9.54e-07 |  |
| 37 | kda_rec | 0.215 | 0.151 | 0.145 | 1.4 | 0.978169 | 0 |  |
| 38 | kda_rec | 0.298 | 0.151 | 0.145 | 2 | 0.954665 | 0.000104 |  |
| 40 | kda_rec | 0.232 | 0.158 | 0.149 | 1.5 | 0.972727 | 0.000122 |  |
| 41 | kda_rec | 0.216 | 0.158 | 0.149 | 1.4 | 0.976558 | 0 |  |
| 42 | kda_rec | 0.374 | 0.158 | 0.149 | 2.4 | 0.929867 | 0 |  |
| 44 | kda_rec | 0.264 | 0.158 | 0.149 | 1.7 | 0.964484 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000414 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0286 | 0 | 0.0137 | 2.1 | 0.999595 | 0.0566 |  |
| 2 | kda_conv | 0.0924 | 0 | 0.0309 | 3 | 0.995719 | 0.0233 |  |
| 4 | kda_conv | 0.0694 | 0.00588 | 0.0309 | 2.2 | 0.997589 | 0.0236 |  |
| 5 | kda_conv | 0.115 | 0.0335 | 0.0523 | 2.2 | 0.993331 | 0.0141 |  |
| 6 | kda_conv | 0.0923 | 0.0388 | 0.0601 | 1.5 | 0.995783 | 0.0177 |  |
| 8 | kda_conv | 0.114 | 0.0786 | 0.0983 | 1.2 | 0.993719 | 0.0141 |  |
| 9 | kda_conv | 0.0958 | 0.0786 | 0.0983 | 0.97 | 0.99543 | 0.0176 |  |
| 10 | kda_conv | 0.0942 | 0.0786 | 0.0983 | 0.96 | 0.995599 | 0.0181 |  |
| 12 | kda_conv | 0.116 | 0.0786 | 0.0983 | 1.2 | 0.993255 | 0.0143 |  |
| 13 | kda_conv | 0.142 | 0.088 | 0.0983 | 1.4 | 0.990059 | 0.0113 |  |
| 14 | kda_conv | 0.158 | 0.089 | 0.0983 | 1.6 | 0.987474 | 0.0105 |  |
| 16 | kda_conv | 0.133 | 0.089 | 0.0983 | 1.3 | 0.991339 | 0.0117 |  |
| 17 | kda_conv | 0.169 | 0.108 | 0.106 | 1.6 | 0.985773 | 0.0105 |  |
| 18 | kda_conv | 0.196 | 0.122 | 0.121 | 1.6 | 0.980578 | 0.00848 |  |
| 20 | kda_conv | 0.266 | 0.162 | 0.132 | 1.6 | 0.964797 | 0.00548 |  |
| 21 | kda_conv | 0.298 | 0.211 | 0.142 | 1.4 | 0.95531 | 0.00514 |  |
| 22 | kda_conv | 0.275 | 0.211 | 0.142 | 1.3 | 0.962537 | 0.00621 |  |
| 24 | kda_conv | 0.321 | 0.211 | 0.142 | 1.5 | 0.950021 | 0.00487 |  |
| 25 | kda_conv | 0.195 | 0.211 | 0.142 | 0.92 | 0.983903 | 0.00783 |  |
| 26 | kda_conv | 0.272 | 0.211 | 0.142 | 1.3 | 0.963124 | 0.00624 |  |
| 28 | kda_conv | 0.177 | 0.211 | 0.142 | 0.84 | 0.984637 | 0.00861 |  |
| 29 | kda_conv | 0.123 | 0.211 | 0.142 | 0.58 | 0.992879 | 0.0124 |  |
| 30 | kda_conv | 0.21 | 0.211 | 0.142 | 0.99 | 0.977754 | 0.00741 |  |
| 32 | kda_conv | 0.147 | 0.211 | 0.142 | 0.69 | 0.989616 | 0.0103 |  |
| 33 | kda_conv | 0.14 | 0.211 | 0.142 | 0.66 | 0.990864 | 0.0115 |  |
| 34 | kda_conv | 0.13 | 0.211 | 0.142 | 0.62 | 0.991492 | 0.00922 |  |
| 36 | kda_conv | 0.13 | 0.211 | 0.142 | 0.61 | 0.991672 | 0.0113 |  |
| 37 | kda_conv | 0.176 | 0.211 | 0.142 | 0.83 | 0.984457 | 0.00875 |  |
| 38 | kda_conv | 0.162 | 0.211 | 0.144 | 0.77 | 0.986807 | 0.0084 |  |
| 40 | kda_conv | 0.197 | 0.211 | 0.151 | 0.93 | 0.980651 | 0.00788 |  |
| 41 | kda_conv | 0.219 | 0.211 | 0.151 | 1 | 0.976101 | 0.00684 |  |
| 42 | kda_conv | 0.207 | 0.211 | 0.151 | 0.98 | 0.978393 | 0.00616 |  |
| 44 | kda_conv | 0.237 | 0.211 | 0.151 | 1.1 | 0.972367 | 0.00609 |  |


</details>

<details><summary>o_prefix_524291_1021 vs v_full_525312: 13 above noise, 0 patterns</summary>

Per layer, `o_prefix_524291_1021` vs `v_full_525312`; noise: vLLM (filler first) / ours (chunked vs whole); ratio = audit / max(noise).

| layer | tensor | audit rel L2 | vLLM noise | ours noise | ratio | cos | exact bytes | flag |
|---|---|---|---|---|---|---|---|---|
| 3 | mla | 0.279 | 0 | 0.061 | 4.6 | 0.96156 | 0.138 | **above noise** |
| 7 | mla | 0.184 | 0.0877 | 0.127 | 1.5 | 0.983061 | 0.168 |  |
| 11 | mla | 0.194 | 0.0877 | 0.127 | 1.5 | 0.981146 | 0.165 |  |
| 15 | mla | 0.223 | 0.112 | 0.127 | 1.8 | 0.975253 | 0.145 |  |
| 19 | mla | 0.256 | 0.177 | 0.184 | 1.4 | 0.967306 | 0.141 |  |
| 23 | mla | 0.281 | 0.217 | 0.222 | 1.3 | 0.960542 | 0.133 |  |
| 27 | mla | 0.281 | 0.217 | 0.222 | 1.3 | 0.960546 | 0.143 |  |
| 31 | mla | 0.199 | 0.217 | 0.222 | 0.9 | 0.980146 | 0.175 |  |
| 35 | mla | 0.206 | 0.217 | 0.222 | 0.93 | 0.978722 | 0.177 |  |
| 39 | mla | 0.24 | 0.217 | 0.222 | 1.1 | 0.971167 | 0.146 |  |
| 43 | mla | 0.261 | 0.217 | 0.222 | 1.2 | 0.965892 | 0.127 |  |
| 3 | indexer | 0.299 | 0 | 0.035 | 8.5 | 0.958699 | 0.139 | **above noise** |
| 7 | indexer | 0.114 | 0.0377 | 0.0493 | 2.3 | 0.993486 | 0.273 |  |
| 11 | indexer | 0.185 | 0.05 | 0.0576 | 3.2 | 0.982862 | 0.168 | **above noise** |
| 15 | indexer | 0.206 | 0.0591 | 0.0632 | 3.3 | 0.978807 | 0.139 | **above noise** |
| 19 | indexer | 0.195 | 0.0971 | 0.102 | 1.9 | 0.981107 | 0.19 |  |
| 23 | indexer | 0.171 | 0.111 | 0.115 | 1.5 | 0.985376 | 0.218 |  |
| 27 | indexer | 0.173 | 0.111 | 0.115 | 1.5 | 0.98502 | 0.228 |  |
| 31 | indexer | 0.139 | 0.111 | 0.115 | 1.2 | 0.990311 | 0.232 |  |
| 35 | indexer | 0.136 | 0.111 | 0.115 | 1.2 | 0.990718 | 0.262 |  |
| 39 | indexer | 0.149 | 0.111 | 0.115 | 1.3 | 0.98883 | 0.241 |  |
| 43 | indexer | 0.171 | 0.111 | 0.115 | 1.5 | 0.985378 | 0.174 |  |
| 0 | kda_rec | 0.00949 | 0 | 0.00296 | 3.2 | 0.999955 | 3.81e-06 | **above noise** |
| 1 | kda_rec | 0.198 | 0 | 0.0149 | 13 | 0.980551 | 0 | **above noise** |
| 2 | kda_rec | 0.296 | 0 | 0.0188 | 16 | 0.9603 | 0 | **above noise** |
| 4 | kda_rec | 0.138 | 0.0127 | 0.0365 | 3.8 | 0.990536 | 0 | **above noise** |
| 5 | kda_rec | 0.2 | 0.0329 | 0.0365 | 5.5 | 0.979825 | 0 | **above noise** |
| 6 | kda_rec | 0.314 | 0.0397 | 0.0469 | 6.7 | 0.949869 | 0 | **above noise** |
| 8 | kda_rec | 0.0939 | 0.0397 | 0.0469 | 2 | 0.995652 | 0 |  |
| 9 | kda_rec | 0.17 | 0.0613 | 0.0744 | 2.3 | 0.985502 | 0 |  |
| 10 | kda_rec | 0.366 | 0.0613 | 0.0744 | 4.9 | 0.934811 | 0 | **above noise** |
| 12 | kda_rec | 0.2 | 0.0774 | 0.0833 | 2.4 | 0.980631 | 0 |  |
| 13 | kda_rec | 0.129 | 0.0774 | 0.0833 | 1.5 | 0.992053 | 0 |  |
| 14 | kda_rec | 0.207 | 0.0774 | 0.0833 | 2.5 | 0.978762 | 0.000241 |  |
| 16 | kda_rec | 0.194 | 0.0774 | 0.0833 | 2.3 | 0.981124 | 0.000123 |  |
| 17 | kda_rec | 0.189 | 0.0774 | 0.0833 | 2.3 | 0.982336 | 0 |  |
| 18 | kda_rec | 0.231 | 0.0824 | 0.0833 | 2.8 | 0.973017 | 0 |  |
| 20 | kda_rec | 0.25 | 0.0932 | 0.0924 | 2.7 | 0.968247 | 0 |  |
| 21 | kda_rec | 0.447 | 0.134 | 0.112 | 3.3 | 0.909709 | 0 | **above noise** |
| 22 | kda_rec | 0.311 | 0.134 | 0.112 | 2.3 | 0.968008 | 0 |  |
| 24 | kda_rec | 0.456 | 0.134 | 0.112 | 3.4 | 0.89868 | 4.77e-06 | **above noise** |
| 25 | kda_rec | 0.211 | 0.134 | 0.112 | 1.6 | 0.977572 | 5.72e-06 |  |
| 26 | kda_rec | 0.33 | 0.134 | 0.112 | 2.5 | 0.94428 | 0 |  |
| 28 | kda_rec | 0.328 | 0.151 | 0.13 | 2.2 | 0.944842 | 1.91e-06 |  |
| 29 | kda_rec | 0.252 | 0.151 | 0.13 | 1.7 | 0.967992 | 0 |  |
| 30 | kda_rec | 0.137 | 0.151 | 0.13 | 0.91 | 0.99063 | 0 |  |
| 32 | kda_rec | 0.258 | 0.151 | 0.13 | 1.7 | 0.96626 | 0 |  |
| 33 | kda_rec | 0.264 | 0.151 | 0.137 | 1.8 | 0.965089 | 9.54e-07 |  |
| 34 | kda_rec | 0.145 | 0.151 | 0.137 | 0.97 | 0.989391 | 0 |  |
| 36 | kda_rec | 0.15 | 0.151 | 0.137 | 1 | 0.989255 | 0 |  |
| 37 | kda_rec | 0.217 | 0.151 | 0.145 | 1.4 | 0.977761 | 0 |  |
| 38 | kda_rec | 0.3 | 0.151 | 0.145 | 2 | 0.953922 | 0.000104 |  |
| 40 | kda_rec | 0.232 | 0.158 | 0.149 | 1.5 | 0.972741 | 0.000122 |  |
| 41 | kda_rec | 0.216 | 0.158 | 0.149 | 1.4 | 0.976635 | 0 |  |
| 42 | kda_rec | 0.374 | 0.158 | 0.149 | 2.4 | 0.929554 | 0 |  |
| 44 | kda_rec | 0.262 | 0.158 | 0.149 | 1.7 | 0.965015 | 0 |  |
| 0 | kda_conv | 0 | 0 | 0.000414 | 0 | 1 | 1 |  |
| 1 | kda_conv | 0.0297 | 0 | 0.0137 | 2.2 | 0.999559 | 0.0537 |  |
| 2 | kda_conv | 0.0926 | 0 | 0.0309 | 3 | 0.995704 | 0.0218 |  |
| 4 | kda_conv | 0.0694 | 0.00588 | 0.0309 | 2.2 | 0.997601 | 0.0235 |  |
| 5 | kda_conv | 0.115 | 0.0335 | 0.0523 | 2.2 | 0.993345 | 0.015 |  |
| 6 | kda_conv | 0.092 | 0.0388 | 0.0601 | 1.5 | 0.995788 | 0.0173 |  |
| 8 | kda_conv | 0.116 | 0.0786 | 0.0983 | 1.2 | 0.993479 | 0.0137 |  |
| 9 | kda_conv | 0.1 | 0.0786 | 0.0983 | 1 | 0.995053 | 0.0166 |  |
| 10 | kda_conv | 0.0925 | 0.0786 | 0.0983 | 0.94 | 0.995786 | 0.0178 |  |
| 12 | kda_conv | 0.115 | 0.0786 | 0.0983 | 1.2 | 0.993427 | 0.0143 |  |
| 13 | kda_conv | 0.141 | 0.088 | 0.0983 | 1.4 | 0.990373 | 0.0118 |  |
| 14 | kda_conv | 0.171 | 0.089 | 0.0983 | 1.7 | 0.985308 | 0.00954 |  |
| 16 | kda_conv | 0.14 | 0.089 | 0.0983 | 1.4 | 0.990278 | 0.0116 |  |
| 17 | kda_conv | 0.176 | 0.108 | 0.106 | 1.6 | 0.984639 | 0.00987 |  |
| 18 | kda_conv | 0.229 | 0.122 | 0.121 | 1.9 | 0.973661 | 0.00766 |  |
| 20 | kda_conv | 0.294 | 0.162 | 0.132 | 1.8 | 0.957093 | 0.00553 |  |
| 21 | kda_conv | 0.348 | 0.211 | 0.142 | 1.6 | 0.939521 | 0.00467 |  |
| 22 | kda_conv | 0.31 | 0.211 | 0.142 | 1.5 | 0.952262 | 0.00589 |  |
| 24 | kda_conv | 0.339 | 0.211 | 0.142 | 1.6 | 0.94335 | 0.00439 |  |
| 25 | kda_conv | 0.205 | 0.211 | 0.142 | 0.97 | 0.981699 | 0.00707 |  |
| 26 | kda_conv | 0.292 | 0.211 | 0.142 | 1.4 | 0.957417 | 0.00538 |  |
| 28 | kda_conv | 0.182 | 0.211 | 0.142 | 0.86 | 0.983617 | 0.00844 |  |
| 29 | kda_conv | 0.13 | 0.211 | 0.142 | 0.61 | 0.992226 | 0.0119 |  |
| 30 | kda_conv | 0.218 | 0.211 | 0.142 | 1 | 0.975947 | 0.00697 |  |
| 32 | kda_conv | 0.152 | 0.211 | 0.142 | 0.72 | 0.988927 | 0.0097 |  |
| 33 | kda_conv | 0.154 | 0.211 | 0.142 | 0.73 | 0.988952 | 0.00968 |  |
| 34 | kda_conv | 0.133 | 0.211 | 0.142 | 0.63 | 0.991127 | 0.00945 |  |
| 36 | kda_conv | 0.132 | 0.211 | 0.142 | 0.62 | 0.991482 | 0.0108 |  |
| 37 | kda_conv | 0.182 | 0.211 | 0.142 | 0.86 | 0.9833 | 0.00845 |  |
| 38 | kda_conv | 0.174 | 0.211 | 0.144 | 0.82 | 0.984885 | 0.00842 |  |
| 40 | kda_conv | 0.194 | 0.211 | 0.151 | 0.92 | 0.981538 | 0.00776 |  |
| 41 | kda_conv | 0.208 | 0.211 | 0.151 | 0.98 | 0.978526 | 0.00736 |  |
| 42 | kda_conv | 0.194 | 0.211 | 0.151 | 0.92 | 0.981184 | 0.00703 |  |
| 44 | kda_conv | 0.205 | 0.211 | 0.151 | 0.97 | 0.979081 | 0.00735 |  |


</details>

## Flags

- 32768: o_whole_32768 layer 0 kda_rec: rel 0.00972 = 3.2x noise (0.00305)
- 32768: o_c8192_32768 layer 0 kda_rec: rel 0.00972 = 3.2x noise (0.00305)
- 32768: o_c8001_32768 layer 0 kda_rec: rel 0.00977 = 3.2x noise (0.00305)
- 131072: o_whole_131072 layer 3 indexer: rel 0.12 = 3.3x noise (0.0364)
- 131072: o_whole_131072 layer 0 kda_rec: rel 0.0112 = 3.6x noise (0.00307)
- 131072: o_whole_131072 layer 1 kda_rec: rel 0.0565 = 3.2x noise (0.0178)
- 131072: o_whole_131072 layer 2 kda_rec: rel 0.139 = 5.8x noise (0.0238)
- 131072: o_whole_131072 layer 5 kda_rec: rel 0.148 = 3.8x noise (0.0388)
- 131072: o_whole_131072 layer 6 kda_rec: rel 0.161 = 3.3x noise (0.0482)
- 131072: o_whole_131072 layer 10 kda_rec: rel 0.256 = 3.1x noise (0.0834)
- 131072: o_c8192_131072 layer 3 indexer: rel 0.12 = 3.3x noise (0.0364)
- 131072: o_c8192_131072 layer 0 kda_rec: rel 0.0112 = 3.6x noise (0.00307)
- 131072: o_c8192_131072 layer 1 kda_rec: rel 0.0565 = 3.2x noise (0.0178)
- 131072: o_c8192_131072 layer 2 kda_rec: rel 0.139 = 5.8x noise (0.0238)
- 131072: o_c8192_131072 layer 5 kda_rec: rel 0.148 = 3.8x noise (0.0388)
- 131072: o_c8192_131072 layer 6 kda_rec: rel 0.161 = 3.3x noise (0.0482)
- 131072: o_c8192_131072 layer 10 kda_rec: rel 0.256 = 3.1x noise (0.0834)
- 131072: o_c8001_131072 layer 3 indexer: rel 0.12 = 3.3x noise (0.0364)
- 131072: o_c8001_131072 layer 0 kda_rec: rel 0.0112 = 3.6x noise (0.00307)
- 131072: o_c8001_131072 layer 1 kda_rec: rel 0.0563 = 3.2x noise (0.0178)
- 131072: o_c8001_131072 layer 2 kda_rec: rel 0.139 = 5.9x noise (0.0238)
- 131072: o_c8001_131072 layer 5 kda_rec: rel 0.148 = 3.8x noise (0.0388)
- 131072: o_c8001_131072 layer 6 kda_rec: rel 0.161 = 3.3x noise (0.0482)
- 131072: o_c8001_131072 layer 10 kda_rec: rel 0.255 = 3.1x noise (0.0834)
- 131072: o_full_132096 layer 3 indexer: rel 0.121 = 3.3x noise (0.0364)
- 131072: o_full_132096 layer 0 kda_rec: rel 0.011 = 3.6x noise (0.00307)
- 131072: o_full_132096 layer 1 kda_rec: rel 0.0561 = 3.2x noise (0.0178)
- 131072: o_full_132096 layer 2 kda_rec: rel 0.149 = 6.3x noise (0.0238)
- 131072: o_full_132096 layer 5 kda_rec: rel 0.162 = 4.2x noise (0.0388)
- 131072: o_full_132096 layer 6 kda_rec: rel 0.158 = 3.3x noise (0.0482)
- 131072: o_full_132096 layer 10 kda_rec: rel 0.252 = 3.0x noise (0.0834)
- 131072: o_prefix_131075_1021 layer 3 indexer: rel 0.121 = 3.3x noise (0.0364)
- 131072: o_prefix_131075_1021 layer 0 kda_rec: rel 0.011 = 3.6x noise (0.00307)
- 131072: o_prefix_131075_1021 layer 1 kda_rec: rel 0.0561 = 3.2x noise (0.0178)
- 131072: o_prefix_131075_1021 layer 2 kda_rec: rel 0.149 = 6.3x noise (0.0238)
- 131072: o_prefix_131075_1021 layer 5 kda_rec: rel 0.164 = 4.2x noise (0.0388)
- 131072: o_prefix_131075_1021 layer 6 kda_rec: rel 0.159 = 3.3x noise (0.0482)
- 131072: o_prefix_131075_1021 layer 10 kda_rec: rel 0.252 = 3.0x noise (0.0834)
- 131072: o_full_100259 layer 3 tail: rel 0.151 = 4.1x noise (0.0368)
- 131072: o_full_100259 layer 0 kda_rec: rel 0.0103 = 3.4x noise (0.00307)
- 131072: o_full_100259 layer 1 kda_rec: rel 0.0552 = 3.1x noise (0.0178)
- 131072: o_full_100259 layer 2 kda_rec: rel 0.107 = 4.5x noise (0.0238)
- 131072: o_full_100259 layer 5 kda_rec: rel 0.132 = 3.4x noise (0.0388)
- 131072: o_prefix_100003_256 layer 3 tail: rel 0.157 = 4.3x noise (0.0368)
- 131072: o_prefix_100003_256 layer 19 tail: rel 0.686 = 6.0x noise (0.114)
- 131072: o_prefix_100003_256 layer 0 kda_rec: rel 0.0104 = 3.4x noise (0.00307)
- 131072: o_prefix_100003_256 layer 1 kda_rec: rel 0.0551 = 3.1x noise (0.0178)
- 131072: o_prefix_100003_256 layer 2 kda_rec: rel 0.107 = 4.5x noise (0.0238)
- 131072: o_prefix_100003_256 layer 5 kda_rec: rel 0.131 = 3.4x noise (0.0388)
- 131072: o_full_100006 layer 3 tail: rel 0.154 = 4.2x noise (0.0368)
- 131072: o_full_100006 layer 0 kda_rec: rel 0.0107 = 3.5x noise (0.00307)
- 131072: o_full_100006 layer 1 kda_rec: rel 0.0567 = 3.2x noise (0.0178)
- 131072: o_full_100006 layer 2 kda_rec: rel 0.107 = 4.5x noise (0.0238)
- 131072: o_full_100006 layer 5 kda_rec: rel 0.129 = 3.3x noise (0.0388)
- 131072: o_prefix_100003_3 layer 3 tail: rel 0.155 = 4.2x noise (0.0368)
- 131072: o_prefix_100003_3 layer 0 kda_rec: rel 0.0108 = 3.5x noise (0.00307)
- 131072: o_prefix_100003_3 layer 1 kda_rec: rel 0.0567 = 3.2x noise (0.0178)
- 131072: o_prefix_100003_3 layer 2 kda_rec: rel 0.107 = 4.5x noise (0.0238)
- 131072: o_prefix_100003_3 layer 5 kda_rec: rel 0.129 = 3.3x noise (0.0388)
- 524288: o_c8192_524288 layer 3 mla: rel 0.278 = 4.6x noise (0.061)
- 524288: o_c8192_524288 layer 3 indexer: rel 0.299 = 8.5x noise (0.035)
- 524288: o_c8192_524288 layer 11 indexer: rel 0.186 = 3.2x noise (0.0576)
- 524288: o_c8192_524288 layer 15 indexer: rel 0.21 = 3.3x noise (0.0632)
- 524288: o_c8192_524288 layer 0 kda_rec: rel 0.00991 = 3.3x noise (0.00296)
- 524288: o_c8192_524288 layer 1 kda_rec: rel 0.208 = 14.0x noise (0.0149)
- 524288: o_c8192_524288 layer 2 kda_rec: rel 0.301 = 16.0x noise (0.0188)
- 524288: o_c8192_524288 layer 4 kda_rec: rel 0.125 = 3.4x noise (0.0365)
- 524288: o_c8192_524288 layer 5 kda_rec: rel 0.194 = 5.3x noise (0.0365)
- 524288: o_c8192_524288 layer 6 kda_rec: rel 0.308 = 6.6x noise (0.0469)
- 524288: o_c8192_524288 layer 10 kda_rec: rel 0.372 = 5.0x noise (0.0744)
- 524288: o_c8192_524288 layer 21 kda_rec: rel 0.494 = 3.7x noise (0.134)
- 524288: o_c8192_524288 layer 24 kda_rec: rel 0.473 = 3.5x noise (0.134)
- 524288: o_c8001_524288 layer 3 mla: rel 0.278 = 4.6x noise (0.061)
- 524288: o_c8001_524288 layer 3 indexer: rel 0.298 = 8.5x noise (0.035)
- 524288: o_c8001_524288 layer 11 indexer: rel 0.184 = 3.2x noise (0.0576)
- 524288: o_c8001_524288 layer 15 indexer: rel 0.208 = 3.3x noise (0.0632)
- 524288: o_c8001_524288 layer 0 kda_rec: rel 0.00999 = 3.4x noise (0.00296)
- 524288: o_c8001_524288 layer 1 kda_rec: rel 0.208 = 14.0x noise (0.0149)
- 524288: o_c8001_524288 layer 2 kda_rec: rel 0.3 = 15.9x noise (0.0188)
- 524288: o_c8001_524288 layer 4 kda_rec: rel 0.125 = 3.4x noise (0.0365)
- 524288: o_c8001_524288 layer 5 kda_rec: rel 0.194 = 5.3x noise (0.0365)
- 524288: o_c8001_524288 layer 6 kda_rec: rel 0.307 = 6.5x noise (0.0469)
- 524288: o_c8001_524288 layer 10 kda_rec: rel 0.372 = 5.0x noise (0.0744)
- 524288: o_c8001_524288 layer 21 kda_rec: rel 0.487 = 3.6x noise (0.134)
- 524288: o_c8001_524288 layer 24 kda_rec: rel 0.46 = 3.4x noise (0.134)
- 524288: o_full_525312 layer 3 mla: rel 0.279 = 4.6x noise (0.061)
- 524288: o_full_525312 layer 3 indexer: rel 0.299 = 8.5x noise (0.035)
- 524288: o_full_525312 layer 11 indexer: rel 0.185 = 3.2x noise (0.0576)
- 524288: o_full_525312 layer 15 indexer: rel 0.206 = 3.3x noise (0.0632)
- 524288: o_full_525312 layer 0 kda_rec: rel 0.00945 = 3.2x noise (0.00296)
- 524288: o_full_525312 layer 1 kda_rec: rel 0.198 = 13.3x noise (0.0149)
- 524288: o_full_525312 layer 2 kda_rec: rel 0.296 = 15.7x noise (0.0188)
- 524288: o_full_525312 layer 4 kda_rec: rel 0.138 = 3.8x noise (0.0365)
- 524288: o_full_525312 layer 5 kda_rec: rel 0.201 = 5.5x noise (0.0365)
- 524288: o_full_525312 layer 6 kda_rec: rel 0.314 = 6.7x noise (0.0469)
- 524288: o_full_525312 layer 10 kda_rec: rel 0.366 = 4.9x noise (0.0744)
- 524288: o_full_525312 layer 21 kda_rec: rel 0.447 = 3.3x noise (0.134)
- 524288: o_full_525312 layer 24 kda_rec: rel 0.456 = 3.4x noise (0.134)
- 524288: o_prefix_524291_1021 layer 3 mla: rel 0.279 = 4.6x noise (0.061)
- 524288: o_prefix_524291_1021 layer 3 indexer: rel 0.299 = 8.5x noise (0.035)
- 524288: o_prefix_524291_1021 layer 11 indexer: rel 0.185 = 3.2x noise (0.0576)
- 524288: o_prefix_524291_1021 layer 15 indexer: rel 0.206 = 3.3x noise (0.0632)
- 524288: o_prefix_524291_1021 layer 0 kda_rec: rel 0.00949 = 3.2x noise (0.00296)
- 524288: o_prefix_524291_1021 layer 1 kda_rec: rel 0.198 = 13.3x noise (0.0149)
- 524288: o_prefix_524291_1021 layer 2 kda_rec: rel 0.296 = 15.8x noise (0.0188)
- 524288: o_prefix_524291_1021 layer 4 kda_rec: rel 0.138 = 3.8x noise (0.0365)
- 524288: o_prefix_524291_1021 layer 5 kda_rec: rel 0.2 = 5.5x noise (0.0365)
- 524288: o_prefix_524291_1021 layer 6 kda_rec: rel 0.314 = 6.7x noise (0.0469)
- 524288: o_prefix_524291_1021 layer 10 kda_rec: rel 0.366 = 4.9x noise (0.0744)
- 524288: o_prefix_524291_1021 layer 21 kda_rec: rel 0.447 = 3.3x noise (0.134)
- 524288: o_prefix_524291_1021 layer 24 kda_rec: rel 0.456 = 3.4x noise (0.134)
