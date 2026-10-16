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
