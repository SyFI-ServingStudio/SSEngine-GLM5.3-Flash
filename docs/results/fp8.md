# FP8 checkpoint: zai-org/GLM-5.3-Flash

SSEngine serves the official FP8 checkpoint as well as the NVFP4 one. The quantization is read from the checkpoint's
`config.json`; there is no flag. Everything else (pipeline split, scheduler, prefix cache, tiers, API) is the same.

## Run it

```bash
# optional: fetch the weights ahead of the first start (~306 GB)
.venv/bin/hf download zai-org/GLM-5.3-Flash --revision 03eb5366286afd40d2221b1d9c63a6dd1ba4832e

.venv/bin/ssengine serve zai-org/GLM-5.3-Flash                               # HBM only
.venv/bin/ssengine serve zai-org/GLM-5.3-Flash --dram-gb 50 --ssd-gb 128 --ssd-dir /nvme/ssengine   # full setting
```

The benchmark runners take the FP8 weights from `SSENGINE_CHECKPOINT=<snapshot directory>`; the token checks then
compare against the FP8 vLLM reference (`benchmarks/data/reference_fp8.jsonl`).

What changes with FP8:

| | NVFP4 | FP8 |
|---|---|---|
| Routed experts | NVFP4 (FlashInfer) | 128 × 128 block FP8 (vLLM's `DEEPGEMM` MoE) |
| Dense FFN (layers 0–2) | NVFP4 | 128 × 128 block FP8 (DeepGEMM `fp8_gemm_nt`) |
| Shared expert, DSA projections q_a/kv_a, q_b, o | BF16 | 128 × 128 block FP8 (DeepGEMM `fp8_gemm_nt`) |
| Router, lm_head, the rest of attention | BF16 | BF16 |
| KV cache | FP8 | FP8 |
| Checkpoint | ~191 GB | ~306 GB |
| KV pool per stage (blocks of 8,576 tokens) | 7,144 on every stage | 6,069 / 7,144 / 4,812 / 4,737 (stages 0–3, tiers on) |

The KV pool is sized at start from a measured warm-up and graph-capture peak, plus the indexer's 2 GiB prefill
logits cap and the tiers' ring; each stage prints it as `KV profile {...}`. With FP8, stage 0's allocator peak over a
20-minute M6 run was 174.2 of 178.35 GiB.

## Results

All on the same 4×B200 node. NVFP4 is the same code's numbers.

### Single request (GPU idle, median of 5, ms)

| Prompt | FP8 GPU span | NVFP4 | FP8 ÷ NVFP4 |
|---|---|---|---|
| 1k | 83.7 | 49.1 | 1.70 |
| 2k | 108.6 | 70.2 | 1.55 |
| 4k | 158.7 | 115.5 | 1.37 |
| 8k | 266.0 | 203.4 | 1.31 |
| 16k | 490.2 | 396.2 | 1.24 |
| 32k | 955.3 | 780.2 | 1.22 |
| 64k | 1,915 | 1,581 | 1.21 |
| 128k | 3,905 | 3,334 | 1.17 |

### Saturated: 2^24 tokens arriving at t = 0 (tok/s)

| Prompt | FP8 | NVFP4 | NVFP4 ÷ FP8 |
|---|---|---|---|
| 1k | 115.6k | 149.2k | 1.29 |
| 2k | 114.7k | 146.5k | 1.28 |
| 4k | 112.1k | 142.8k | 1.27 |
| 8k | 110.0k | 138.4k | 1.26 |
| 16k | 109.6k | 136.8k | 1.25 |
| 32k | 108.8k | 136.4k | 1.25 |
| 64k | 108.0k | 134.2k | 1.24 |
| 128k | 105.7k | 129.4k | 1.22 |

Output tokens match the FP8 vLLM TP4 reference (1k–128k, 40 requests) in both. Stage busy 93–100%.

The gap is the MoE: one MoE block takes 2.6× NVFP4's time at 256 tokens and 1.7× at 32k (FP8 DeepGEMM 90–1,566
TFLOPS, NVFP4 232–2,669). MoE is about a third of an NVFP4 8k microbatch, which predicts 1.39× there; measured 1.33×.
Attention and the KDA layers cost the same in both, so longer prompts dilute the gap.

### Prefix cache: two-round sessions (median of 5, ms, FP8 / NVFP4)

Round 0 computes the prefix P; round 1 hits P and computes the append A.

| P | Round 0 TTFT | ratio | Round 1 span, A = 1k | ratio | Round 1 span, A = 8k | ratio |
|---|---|---|---|---|---|---|
| 8k | 268.4 / 204.0 | 1.32 | 87.5 / 49.3 | 1.78 | 267.4 / 202.5 | 1.32 |
| 32k | 504.9 / 391.5 | 1.29 | 88.4 / 50.0 | 1.77 | 272.2 / 206.3 | 1.32 |
| 128k | 1,443.3 / 1,126.5 | 1.28 | 89.5 / 51.8 | 1.73 | 286.0 / 219.9 | 1.30 |
| 512k | 5,668.4 / 4,523.1 | 1.25 | 96.2 / 58.2 | 1.65 | 344.2 / 280.5 | 1.23 |

The ratio follows the tokens computed (A), not the prefix length.

### M6 sessions: v3 trace, N = 300, full setting (minutes 10–20)

| | r/s | New tok/s | TTFT p50 / p90 / p99 (s) | > 60 s | Hits HBM / DRAM / SSD % |
|---|---|---|---|---|---|
| FP8 | 29.23 | 97.1k | 0.48 / 1.54 / 59.24 | 175 | 91.7 / 7.9 / 0.3 |
| NVFP4 | 33.78 | 113.3k | 0.18 / 0.35 / 2.65 | 13 | 98.0 / 1.8 / 0.2 |

FP8's smaller HBM pool moves hits from HBM to DRAM.

## Jobs

Single request and saturated 8685, 8687, 8688, 8691; MoE block 8684 / 8685; prefix cache 8696 (FP8) / 8697 (NVFP4);
M6 N300 8713 (FP8) and 8541 (NVFP4, docs/results/n_sweep.md); NVFP4 check on the final sizing 8715; FP8 vLLM token
reference 8683 and earlier (1k–32k).
