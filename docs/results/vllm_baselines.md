# vLLM DP4/EP4 baseline: GLM-5.3-Flash NVFP4, 4×B200, M6 v3 sessions

Setup: vLLM fork 0.1.dev21920+g892da0822 (pp-impl/third_party/vllm, unpatched).
- Topology: DP4 × TP1 + expert parallel; FP4 checkpoint; fp8_e4m3 KV.
- Prefix caching on, chunked prefill, `--mamba-cache-mode align`.
- Trace: data/tracelab-v3/m6/sessions_2ps_v3.csv, trace-timed, 22 min client; metrics at 20 min.
- Routing: vllm-router 0.1.15 `cache_aware` (radix tree on token ids; `--intra-node-data-parallel-size 4` → X-data-parallel-rank). 8199 instead uses vLLM's built-in DP balancer, which is load-only, not cache-aware.
- B runs: native OffloadingConnector, `--kv-offloading-size 46.5661` GiB = 50 GB per engine. The size is per engine, because world_size = TP×PP = 1. One /dev/shm region per engine.

Scripts are on lane `lane-vllm-dpep` (pp-impl/lanes/vllm-dpep):
- [benchmarks/jobs/vllm_dpep.sbatch](../../benchmarks/jobs/vllm_dpep.sbatch)
- [benchmarks/vllm_dpep_summary.py](../../benchmarks/vllm_dpep_summary.py)
- [benchmarks/vllm_dpep_table.py](../../benchmarks/vllm_dpep_table.py)
- [benchmarks/vllm_dpep_simwin.py](../../benchmarks/vllm_dpep_simwin.py)

Each run directory holds server_command.txt, router_command.txt, client_command.txt, prom/ (one /metrics snapshot per minute) and dpep_summary.json.

Windows:
- 0-20: first token in [0, 20) min.
- 10-20: first token in [10, 20) min, the sim rows' window.

Other definitions:
- recompute/new: (computed − new) / new.
- full-miss: rounds > 0 with 0 cached tokens.
- HBM/CPU share: vLLM prompt_tokens_by_source, local_cache_hit vs external_kv_transfer, as a share of prompt tokens.

| run | window | r/s | new tok/s | computed tok/s | TTFT p50/p90/p99 s | >60 s | hit share | HBM/CPU share | recompute/new | req max/mean | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hbm_n400_8199 | 0-20 | 2.62 | 25.6k | 103.7k | 130.61/183.73/205.97 | 2570 | 0.315 | 0.257/0.000 | 3.05 | 1.04 | full-miss 0.644; failed 0/3305 |
| hbm_n400_8199 | 10-20 | 2.17 | 13.5k | 104.9k | 172.57/194.05/211.62 | 1299 | 0.276 | 0.248/0.000 | 6.78 | 1.06 | 403/407/358/366; inside 1 rounds, 0 hit |
| hbm_cache_aware_n400_8250 | 0-20 | 6.58 | 41.8k | 113.8k | 36.23/67.53/201.68 | 910 | 0.842 | 0.761/0.000 | 1.73 | 1.16 | full-miss 0.100; failed 0/8086 |
| hbm_cache_aware_n400_8250 | 10-20 | 4.64 | 17.4k | 113.6k | 48.31/148.23/212.89 | 910 | 0.772 | 0.729/0.000 | 5.51 | 1.43 | 1142/794/725/538; inside 1 rounds, 0 hit |
| off50_cache_aware_n400_8251 | 0-20 | 6.41 | 41.1k | 112.8k | 36.25/74.09/211.36 | 1095 | 0.836 | 0.754/0.000 | 1.74 | 1.09 | full-miss 0.105; failed 0/7867 |
| off50_cache_aware_n400_8251 | 10-20 | 4.37 | 16.3k | 112.6k | 53.12/144.37/237.29 | 1089 | 0.756 | 0.715/0.000 | 5.89 | 1.24 | 548/738/940/811; inside 1 rounds, 0 hit |
| hbm_cache_aware_mnbt16384_n300_8280 | 0-20 | 9.70 | 48.4k | 98.3k | 18.86/30.68/43.02 | 0 | 0.943 | 0.897/0.000 | 1.03 | 1.02 | full-miss 0.012; failed 0/12625 |
| hbm_cache_aware_mnbt16384_n300_8280 | 10-20 | 10.10 | 37.2k | 96.8k | 19.37/29.47/34.94 | 0 | 0.947 | 0.918/0.000 | 1.60 | 1.08 | 1740/1757/1882/1986; inside 1 rounds, 0 hit |
| off50_cache_aware_mnbt16384_n300_8281 | 0-20 | 9.76 | 48.5k | 98.8k | 18.53/32.33/44.11 | 0 | 0.943 | 0.897/0.000 | 1.04 | 1.04 | full-miss 0.012; failed 0/12712 |
| off50_cache_aware_mnbt16384_n300_8281 | 10-20 | 10.12 | 37.1k | 97.4k | 18.40/32.84/41.85 | 0 | 0.947 | 0.919/0.000 | 1.62 | 1.07 | 1761/1725/1978/1953; inside 2 rounds, 0 hit |
| tp4_hbm_n400_8419 | 0-20 | 1.78 | 20.4k | 64.1k | 147.10/333.85/346.84 | 1374 | 0.323 | 0.233/0.000 | 2.15 | 1.00 | TP4, no router; full-miss 0.646; failed 0/2226 |
| tp4_hbm_n400_8419 | 10-20 | 1.17 | 7.1k | 64.0k | 319.95/344.78/349.48 | 702 | 0.000 | 0.000/0.000 | 8.01 | 1.00 | 826; inside 0 rounds, 0 hit |
| tp4_hbm_n300_8420 | 0-20 | 1.81 | 17.9k | 63.7k | 68.74/289.89/303.05 | 1400 | 0.386 | 0.296/0.000 | 2.56 | 1.00 | TP4, no router; full-miss 0.561; failed 0/2249 |
| tp4_hbm_n300_8420 | 10-20 | 0.99 | 6.5k | 63.4k | 275.57/299.96/306.23 | 594 | 0.000 | 0.000/0.000 | 8.74 | 1.00 | 700; inside 0 rounds, 0 hit |
| sim 20261018_9_v3_m6/st10/full/c400 | 0-20 | 37.34 | 141.4k | 142.1k | 0.61/1.88/60.90 | 710 | 1.000 | - | 0.00 | - |  |
| sim 20261018_9_v3_m6/st10/full/c400 | 10-20 | 42.26 | 139.6k | 140.0k | 0.48/1.30/33.28 | 194 | 1.000 | - | 0.00 | - |  |
| sim 20261018_9_v3_m6/notier/hbmonly/c400 | 0-20 | 24.34 | 99.9k | 143.2k | 1.01/15.23/62.82 | 1908 | 0.985 | - | 0.43 | - |  |
| sim 20261018_9_v3_m6/notier/hbmonly/c400 | 10-20 | 19.07 | 68.8k | 142.0k | 2.12/60.66/64.58 | 1222 | 0.971 | - | 1.06 | - |  |
| sim 20261018_11_v3_m6_c300/st10/full/c300 | 0-20 | 32.72 | 125.0k | 125.8k | 0.17/0.89/13.61 | 164 | 1.000 | - | 0.01 | - |  |
| sim 20261018_11_v3_m6_c300/st10/full/c300 | 10-20 | 33.97 | 115.3k | 115.7k | 0.15/0.25/1.65 | 4 | 1.000 | - | 0.00 | - |  |
| sim 20261018_11_v3_m6_c300/notier/hbmonly/c300 | 0-20 | 30.45 | 115.2k | 132.5k | 0.26/1.61/44.22 | 315 | 0.996 | - | 0.15 | - |  |
| sim 20261018_11_v3_m6_c300/notier/hbmonly/c300 | 10-20 | 30.03 | 97.3k | 128.1k | 0.20/1.99/20.47 | 134 | 0.993 | - | 0.32 | - |  |

Runs:
- 8199 = out-of-box: vLLM DP balancer, MNBT default 16384.
- 8250/8251 = A/B at N400 with the router, MNBT 32768.
- 8280/8281 = A/B at N300 with the router, MNBT 16384.
  - MNBT 32768 crashed at N300 (job 8252): DP+EP all-gathers up to 4×MNBT tokens into each MoE, and startup profiling does not reserve for that. A 7.91 GiB alloc failed and the engine died.
- 8282 (A400, MNBT 16384) is queued in background as the matching N400 point.
- 8419/8420 = pure tensor parallel: one engine `--tensor-parallel-size 4 --data-parallel-size 1`, no expert parallel, no router (the client talks to vLLM directly). MNBT 32768, HBM only, N400 and N300; otherwise the same flags, client and windows (sbatch `TP=4`).
  - MNBT 32768 started and ran at N300 without OOM, so no 16384 retry was needed. The DP crash in 8252 came from the EP all-gather, which TP4 does not have.
  - 8455 (TP4 N400 + native offload, `OFFLOAD_GB=50`) is queued in background. Under TP, `--kv-offloading-size` is the engine total across its 4 TP workers: CPUOffloadingSpec stores one copy per worker, because the single-copy `replicated_layout` covers pure-MLA models only. The script therefore passes 4 × 50 GB = 186.26 GiB, which is 50 GB per GPU, the same as 8251.

## Findings
- **Compute-bound on recompute.** Every run saturates prefill at about 97–114k computed tok/s total (4 GPUs). The queue holds 250–370 waiting requests, KV usage stays low, and there are no preemptions.
- **Session affinity is the first-order effect.** At N400, the cache_aware router lifts r/s 2.6 → 6.6 over the load-only DP balancer, and hit share goes 0.32 → 0.84. Its load override (max − min in-flight > 64) never fired: the peak spread was 21.
- **Hit granularity is 8576 tokens.** The hybrid allocator sets the attention block to 8576 tokens so the attention page covers the KDA state page. Align mode keeps a KDA state only at block ends, so every round recomputes its tail: mean ≈ 4.4k tokens per round (recompute/new ≈ 1.0–1.6 at N300 with 1.2% full misses).
- **N400 adds HBM eviction.**
  - Active context of about 36–40M tokens meets 4 × 11.85M HBM KV tokens. The router's skew (late requests 1142/794/725/538) makes the heavy engines evict.
  - A tail eviction drops the only KDA state for that context, so the next round of that session is a full miss: 23% of rounds in minutes 10–20, recompute/new 5.5–5.9.
- **The CPU offload tier adds nothing at 50 GB per engine (B ≈ A at both N).**
  - The native connector writes through: every computed chunk is stored as MLA + indexer + KDA state, about 43–51 MB per 8576-token chunk.
  - Capacity is 972 chunks = 8.3M tokens per engine, smaller than HBM (11.85M tokens).
  - The CPU tier cycled 8–13× during each run (B400: 1.94 TB stored, 1 load of 250 MB; B300: 1.69 TB stored, 0 loads). It only ever holds copies of recent HBM content.
- **Inside rounds:** 1–2 in minutes 10–20 on the v3 trace, all 0-hit.
- **Versus targets (10-20 window):**

| Point | vLLM | Sim | Ratio |
|---|---|---|---|
| A400 vs sim hbmonly c400 | 4.6 r/s | 19.1 | 4.1× |
| Best vLLM N400 vs sim st10 full c400 | 4.6 r/s | 42.3 | 9× |
| N300 vs sim hbmonly c300 | 10.1 r/s | 30.0 | 3.0× |
| N300 vs sim st10 full c300 | 10.1 r/s | 34.0 | 3.4× |

The sims compute ≈ new tokens (hit ≈ 1.0 at token granularity). vLLM computes 2.6–7.8× the new tokens.

## TP4 vs DP4/EP4
- **TP4 is far below DP4/EP4.**
  - N400: 1.78 vs 6.58 r/s over 0-20, 1.17 vs 4.64 over 10-20.
  - N300: 1.81 vs 9.70 r/s, though the N300 DP point used MNBT 16384.
  - In minutes 10-20 the prefix hit share is 0.000 at both N.
- **Per-token prefill speed is lower.**
  - TP4 computes about 64k tok/s over the 4 GPUs (16k per GPU), at both N.
  - DP4/EP4 computes 113.6k tok/s at the same MNBT 32768, so TP4 runs at 0.56× the speed.
  - TP4's 10-20 rounds are also shorter: mean prompt 55k tokens vs 95k in 8250.
  - Both setups are pure-prefill bound: about 2 running requests and 280–390 waiting, with the 32768-token step budget used up by one or two long chunked prefills.
  - Not profiled. The likely causes are per-layer all-reduces on 32k-token steps and narrow TP-sharded GEMMs (MoE intermediate/4, attention heads/4) in place of whole-expert EP GEMMs.
- **KV capacity is 2.8× smaller in tokens.**
  - TP4 has one pool of 17.06M tokens (97.08 GiB free per GPU after 47.76 GiB of weights; block 2176 tokens).
  - DP4 has 4 × 11.85M = 47.4M (71.45 GiB per GPU after 59.35 GiB; block 8576).
  - The MLA latent and indexer KV are replicated on all 4 TP ranks, and only the KDA state is sharded. Each token therefore costs about 6.1 KB on every GPU, compared with 6.5 KB on one GPU under DP.
  - The finer hit granularity (2176 tokens, from the smaller per-rank KDA page) does not help, because nothing survives to be hit.
- **Recompute comes from LRU turnover, not from routing.**
  - The pool turns over every ≈ 17.06M / 64k = 270 s. Queueing alone keeps a session's next round out longer than that: TTFT p50 is 275–320 s in minutes 10-20.
  - By the time a round is scheduled, its context has been evicted, and the KDA state with it, so every round is a full recompute.
  - The prefix-hit counter stops growing after about minute 5 (25.1M hit tokens at N400, 34.5M at N300) and stays flat to the end.
  - recompute/new is 8.0 at N400 and 8.7 at N300 in minutes 10-20, versus 5.5 for DP4 (8250).
  - There are no preemptions, and KV usage stays under 4%: the pool fills only with evictable cached blocks.
