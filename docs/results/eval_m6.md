# M6 retention eval: real engine (PP4 GLM-5.3-Flash NVFP4, 4x B200) vs sim vs vLLM, v3 20-min trace

Real code: pp-impl eval tree lanes/m6-eval-run (a78dc6f = e283243 minus the 32k CUDA graphs; 9496233 adds the excl
read-gone fix, used by 8453, 8462, 8466). Sim: wt-pp-staged logs 20261018_11_v3_m6_c300 (N300), 20261018_9_v3_m6 (N400),
20261018_13_v3_m6_c100 (N100). Trace data/tracelab-v3/m6/sessions_2ps_v3.csv, closed loop of N sessions.
Metrics: eval_table.py (windows by first token; TTFT = client first token - submit, server-side differs by 1-5 ms;
hit H/D/S = rounds with a previous context served from HBM/DRAM/SSD, (lost) = recomputed).

Settings (`benchmarks/run.py` sched_good_nN): full = full HBM pool + DRAM 50 GB + SSD 128 GB; tiers = HBM 12.2M tokens +
DRAM 20 GB + SSD 128 GB; ssdheavy = HBM 7.6M + DRAM 12.5 GB + SSD 128 GB; hbmonly = HBM only (61.26M tokens).
st10 = exclusive DRAM writes + 10% HBM loading area (the design); excl = exclusive, no loading area (ablation).

## Headline, minutes 10-20 (r/s; TTFT p50/p90/p99 s)

| N | setting | real r/s | sim r/s | real/sim | real TTFT | sim TTFT | real H/D/S % | sim H/D/S % |
|---|---|---|---|---|---|---|---|---|
| 100 | st10 tiers | 8.62 | 8.62 | 1.00 | 0.09/0.17/0.36 | 0.08/0.16/0.36 | 91/9/1 | 91/8/0 |
| 300 | st10 full | 33.70 | 33.97 | 0.99 | 0.19/0.34/2.87 | 0.15/0.25/1.65 | 98/2/0 | 98/2/0 |
| 300 | st10 tiers | 33.45 | 33.77 | 0.99 | 0.22/0.42/4.36 | 0.18/0.29/1.89 | 44/34/23 | 46/33/21 |
| 300 | st10 ssdheavy | 33.76 | 33.80 | 1.00 | 0.20/0.39/3.08 | 0.19/0.31/2.02 | 26/32/42 | 28/31/41 |
| 300 | excl tiers | 32.88 | 33.83 | 0.97 | 0.27/0.80/5.40 | 0.18/0.29/1.73 | 44/36/20 | 49/33/17 |
| 300 | hbmonly | 27.24 | 30.03 | 0.91 | 0.36/3.01/60.90 | 0.20/1.99/20.47 | lost 0.9 | lost 0.7 |
| 400 | st10 full | 40.51 | 42.26 | 0.96 | 0.61/1.52/60.43 | 0.48/1.30/33.28 | 94/6/1 | 95/5/0 |
| 400 | st10 tiers | 37.66 | 39.77 | 0.95 | 0.84/3.81/60.66 | 0.65/2.53/44.99 | 13/41/47 | 18/42/39 |
| 400 | st10 ssdheavy | 36.71 | 38.36 | 0.96 | 0.84/3.40/60.59 | 0.63/2.83/48.38 | 2/21/77 | 4/34/62 |
| 400 | excl tiers | 33.10 | 39.79 | 0.83 | 1.80/5.28/61.18 | 0.74/2.38/56.04 | 6/48/46 | 23/43/34 |
| 400 | hbmonly | 12.97 | 19.07 | 0.68 | 1.86/60.95/163.5 | 2.12/60.66/64.58 | lost 5.3 | lost 2.9 |

## vLLM baselines (same trace and windows; r/s; TTFT p50/p90/p99 s; rounds over 60 s)

Setup in [vllm_baselines.md](vllm_baselines.md). Computed from each run's requests.jsonl with this file's window rule
(t = 0 at the first submit; a round counts when its first token is inside the window; TTFT = first token - submit);
equal to each run's dpep_summary.json (metrics_0_20, metrics_first_token_10_20). DP4/EP4 = 4 engines + vllm-router
cache_aware, HBM only; N300 uses max_num_batched_tokens 16384, N100/N400 32768. TP4 = one TP4 engine, no router,
HBM only, MNBT 32768. Ours: the eval rows above (N100 st10 tiers 8456; N300/N400 st10 full and st10 tiers).

| N | window | system | run | r/s | TTFT p50/p90/p99 (s) | >60 s |
|---|---|---|---|---|---|---|
| 100 | 0-20 | ours st10 tiers | 8456 | 9.39 | 0.11/0.23/0.94 | 1 |
| 100 | 0-20 | vLLM DP4/EP4 | hbm_cache_aware_n100_8457 | 8.05 | 1.77/4.22/7.92 | 0 |
| 100 | 0-20 | vLLM TP4 | tp4_hbm_n100_8458 | 7.46 | 2.29/7.73/13.94 | 0 |
| 100 | 10-20 | ours st10 tiers | 8456 | 8.62 | 0.09/0.17/0.36 | 0 |
| 100 | 10-20 | vLLM DP4/EP4 | hbm_cache_aware_n100_8457 | 7.46 | 1.31/2.24/3.72 | 0 |
| 100 | 10-20 | vLLM TP4 | tp4_hbm_n100_8458 | 7.26 | 0.76/3.81/7.92 | 0 |
| 300 | 0-20 | ours st10 full | 8447 | 32.03 | 0.23/0.99/19.83 | 205 |
| 300 | 0-20 | ours st10 tiers | 8462 | 31.54 | 0.26/1.14/21.91 | 217 |
| 300 | 0-20 | vLLM DP4/EP4 | hbm_cache_aware_mnbt16384_n300_8280 | 9.70 | 18.86/30.68/43.02 | 0 |
| 300 | 0-20 | vLLM TP4 | tp4_hbm_n300_8420 | 1.81 | 68.74/289.89/303.05 | 1400 |
| 300 | 10-20 | ours st10 full | 8447 | 33.70 | 0.19/0.34/2.87 | 15 |
| 300 | 10-20 | ours st10 tiers | 8462 | 33.45 | 0.22/0.42/4.36 | 22 |
| 300 | 10-20 | vLLM DP4/EP4 | hbm_cache_aware_mnbt16384_n300_8280 | 10.10 | 19.37/29.47/34.94 | 0 |
| 300 | 10-20 | vLLM TP4 | tp4_hbm_n300_8420 | 0.99 | 275.57/299.96/306.23 | 594 |
| 400 | 0-20 | ours st10 full | 8451 | 36.29 | 0.72/2.04/60.93 | 727 |
| 400 | 0-20 | ours st10 tiers | 8431 | 34.07 | 0.99/4.70/61.00 | 707 |
| 400 | 0-20 | vLLM DP4/EP4 | hbm_cache_aware_n400_8250 | 6.58 | 36.23/67.53/201.68 | 910 |
| 400 | 0-20 | vLLM TP4 | tp4_hbm_n400_8419 | 1.78 | 147.10/333.85/346.84 | 1374 |
| 400 | 10-20 | ours st10 full | 8451 | 40.51 | 0.61/1.52/60.43 | 249 |
| 400 | 10-20 | ours st10 tiers | 8431 | 37.66 | 0.84/3.81/60.66 | 260 |
| 400 | 10-20 | vLLM DP4/EP4 | hbm_cache_aware_n400_8250 | 4.64 | 48.31/148.23/212.89 | 910 |
| 400 | 10-20 | vLLM TP4 | tp4_hbm_n400_8419 | 1.17 | 319.95/344.78/349.48 | 702 |

vLLM DP4/EP4 with its CPU offload (50 GB per engine), r/s vs HBM only: N300 +0.6% (0-20) / +0.2% (10-20), N400
-2.6% (0-20) / -5.8% (10-20, 4.37 vs 4.64). N300 8281 9.76 / 10.12 r/s, TTFT
18.53/32.33/44.11 and 18.40/32.84/41.85 s; N400 8251 6.41 / 4.37 r/s, 36.25/74.09/211.36 and 53.12/144.37/237.29 s
(0-20 / 10-20). recompute/new in 10-20: DP4/EP4 1.39 (N100), 1.60 (N300), 5.51 (N400); TP4 0.97, 8.74, 8.01.

## Findings

1. Retention: with tiers, ours sustains 33.7 r/s at N300 and 37-41 r/s at N400 (10-20). HBM only falls to 27.2
   (N300) and 13.0 (N400) as lost contexts are recomputed (recompute/new 0.40 and 1.76). Shrinking HBM to 7.6M
   tokens (ssdheavy, 77% of hits from SSD) costs 0% at N300 and 3% at N400 against the full pool. At N100 nothing is
   saturated and every setting gives the same 8.6 r/s.
2. Baselines: at N400, ours with tiers is 5.2x vLLM DP4/EP4 (0-20: 34.1 vs 6.6 r/s) and 19x TP4 (1.78); at N300
   3.3x DP4/EP4 with st10 full (32.0 vs 9.7). At N100 ours serves every arrival (8.62 r/s, vLLM DP4/EP4 7.46, TP4 7.26
   in 10-20); vLLM's TTFT p50/p90/p99 is 1.31/2.24/3.72 s (DP4/EP4) and 0.76/3.81/7.92 s (TP4) vs 0.09/0.17/0.36 s.
   At N300/N400 (0-20) vLLM DP4/EP4 TTFT p50 is 18.9-36.2 s and TP4 68.7-147.1 s, against 0.23-0.99 s; DP4/EP4 N400
   has 910 rounds over 60 s in 0-20 against 707-727 for ours, all of them in minutes 10-20 (p99 213 s vs 60.4-60.7).
3. Real vs sim, st10: within 1% at N100/N300, 4-5% at N400. TTFT p50 is 20-30% higher in real.
   - Stage 2/3 GPUs run 10-14 C hotter in either 4-GPU set and, at the 1000 W cap, clock 5-7% lower; the bottleneck
     stage is ~7% slower than the sim in those runs (per-stage ms/1k tokens, gpu_samples.csv).
   - A uniform 3% sim slowdown (20261018_14_v3_m6_slow, gpu_time_multiplier 1.025-1.035) reproduces ~2/3 of the N400
     gap (r/s 41.1, TTFT 0.56/1.44/54) but little of N300's: there the loss is concentrated on s2/s3 near saturation.
   - The N400 p99 sits on the 60 s force-admission bound: >60 s rounds are 1.02% (real) vs 0.77% (sim).
4. HBM-only gap is block rounding: 8576-token KV blocks waste ~2.4% of HBM (4.3k tokens per context). The sim with
   -2.4% capacity gives 11.2-12.7 r/s and recompute/new ~1.75 at N400 (real 12.97 / 1.76); the N400 hbmonly window is
   bistable (sim 13.9-19.1 r/s over +-0.5% speed perturbations). Fixed by 128-token pages (below).
5. excl (no loading area) is 12% below st10 in real at N400 but equal in the sim (at N300: -1.7% r/s, TTFT p90
   0.80 vs 0.42 s, recompute/new 0.094 vs 0.025): without a loading area, tier reads wait on
   the eviction writes that free their blocks and those writes wait on DRAM slots of in-flight reads (DRAM read p50
   217 ms vs 55 ms st10; s0 busy 0.875), and the sim charges no write cost. The loading area is required.

## Engine changes measured on the eval settings (minutes 10-20)

| change | run | vs baseline | r/s | TTFT p50/p90/p99 (s) | recompute/new | s0 busy |
|---|---|---|---|---|---|---|
| HBM KV in 128-token pages (lane-m6-pack 8d50066) | hbmonly N400 8475 (set B) | 8450 (set A) | 16.16 vs 12.97 | 1.90/60.84/69.3 vs 1.86/60.95/163.5 | 1.21 vs 1.76 | 0.989 vs 0.983 |
| host time excludes the in-flight gate wait (lane-m6-hostgate 7e391df) | st10 full N300 8473 (set B) | 8447 (set A) | 33.72 vs 33.70 | 0.19/0.34/2.81 vs 0.19/0.34/2.87 | 0.005 vs 0.005 | 0.932 vs 0.909 |
| 128-token pages | st10 full N300 8476 (set A) | 8447 (set A) | 33.67 vs 33.70 | 0.19/0.35/2.50 vs 0.19/0.34/2.87 | 0.005 vs 0.005 | 0.907 vs 0.909 |
| hostgate | st10 full N400 8474 (set A) | 8451 (set B) | 40.45 vs 40.51 | 0.62/1.55/56.1 vs 0.61/1.52/60.4 | 0.003 vs 0.007 | 0.964 vs 0.981 |

- Pages leave the tier path unchanged (N300 st10 full, same GPU set) and remove the block-rounding loss (4.3k -> 63 tokens per context): hbmonly N400 moves to the sim's full-capacity
  ensemble (median ~16 r/s, recompute/new ~1.21). GPU sets differ, but recompute is set by HBM capacity, not clocks.
- hostgate: -4 ms TTFT p50 at N300 (small requests 179 vs 183 ms), neutral at N400; the ~30 ms small-request excess at
  N300 is the hot stage-2/3 GPUs, not the plan lead. GPU sets: A = bus 1B/43/52/61, B = 9D/C3/D1/DF.

## Commit ablation (N400 st10 tiers, workload minutes 2-7; 8-min runs end at ~7.5 min)

| code | r/s | s0 busy | plan lead p50 | tier record p99 |
|---|---|---|---|---|
| A = eval code (32k CUDA graphs removed: graph memory 26 -> 10 GiB, no loss) | 27.87 | 0.975 | 39 ms | 15 us |
| A minus 9780a2f (plan timing) | 27.31 | 0.946 | 19 ms | 16 us |
| A minus ab0099a | 27.81 | 0.974 | 39 ms | 15 us |
| A minus 9780a2f, 14bf48b, ab0099a | 27.13 | 0.948 | 19 ms | 291 us |

9780a2f carries the gain (+3 points s0 busy); 14bf48b removes the record tail; ab0099a is neutral and needs 14bf48b.
All are kept.

## All rows
| run | window | r/s | new tok/s | computed tok/s | recompute/new | TTFT p50/p90/p99 (s) | >60 s | hit H/D/S (lost) % | s0 busy |
|---|---|---|---|---|---|---|---|---|---|
| real st10_full_n300_8447 | 0-20 | 32.03 | 122.1k | 122.9k | 0.006 | 0.23/0.99/19.83 | 205 | 98.6/1.3/0.1 (0.0) | 0.928 |
| real st10_full_n300_8447 | 10-20 | 33.70 | 113.0k | 113.6k | 0.005 | 0.19/0.34/2.87 | 15 | 97.8/1.9/0.2 (0.0) | 0.909 |
| real st10_tiers_n300_8462 | 0-20 | 31.54 | 120.0k | 121.8k | 0.015 | 0.26/1.14/21.91 | 217 | 44.2/37.5/18.3 (0.1) | 0.947 |
| real st10_tiers_n300_8462 | 10-20 | 33.45 | 110.8k | 113.5k | 0.025 | 0.22/0.42/4.36 | 22 | 43.8/33.6/22.5 (0.1) | 0.933 |
| real st10_ssdheavy_n300_8448 | 0-20 | 31.74 | 121.0k | 123.0k | 0.016 | 0.24/1.34/19.18 | 195 | 25.6/33.3/41.1 (0.1) | 0.973 |
| real st10_ssdheavy_n300_8448 | 10-20 | 33.76 | 112.5k | 115.5k | 0.026 | 0.20/0.39/3.08 | 17 | 26.2/31.6/42.2 (0.1) | 0.974 |
| real excl_tiers_n300_8466 | 0-20 | 30.92 | 118.2k | 124.6k | 0.054 | 0.37/1.55/29.40 | 245 | 48.1/36.5/15.2 (0.2) | 0.951 |
| real excl_tiers_n300_8466 | 10-20 | 32.88 | 109.2k | 119.5k | 0.094 | 0.27/0.80/5.40 | 39 | 44.2/35.9/19.6 (0.2) | 0.962 |
| real hbmonly_n300_8446 | 0-20 | 28.50 | 108.6k | 127.9k | 0.179 | 0.43/2.09/60.71 | 400 | 99.5/0.0/0.0 (0.5) | 0.962 |
| real hbmonly_n300_8446 | 10-20 | 27.24 | 88.1k | 123.1k | 0.397 | 0.36/3.01/60.90 | 188 | 99.1/0.0/0.0 (0.9) | 0.958 |
| real st10_full_n400_8451 | 0-20 | 36.29 | 138.1k | 139.2k | 0.008 | 0.72/2.04/60.93 | 727 | 95.4/4.3/0.3 (0.0) | 0.982 |
| real st10_full_n400_8451 | 10-20 | 40.51 | 135.0k | 136.0k | 0.007 | 0.61/1.52/60.43 | 249 | 93.9/5.6/0.5 (0.0) | 0.981 |
| real st10_tiers_n400_8431 | 0-20 | 34.07 | 131.7k | 138.0k | 0.048 | 0.99/4.70/61.00 | 707 | 16.9/34.4/48.6 (0.1) | 0.967 |
| real st10_tiers_n400_8431 | 10-20 | 37.66 | 126.3k | 137.8k | 0.091 | 0.84/3.81/60.66 | 260 | 12.5/40.5/46.7 (0.2) | 0.967 |
| real st10_ssdheavy_n400_8452 | 0-20 | 33.37 | 128.7k | 135.3k | 0.052 | 1.00/4.10/60.91 | 771 | 8.2/17.2/74.3 (0.2) | 0.960 |
| real st10_ssdheavy_n400_8452 | 10-20 | 36.71 | 120.8k | 133.1k | 0.101 | 0.84/3.40/60.59 | 282 | 2.2/20.6/76.9 (0.3) | 0.971 |
| real excl_tiers_n400_8453 | 0-20 | 30.77 | 119.6k | 129.5k | 0.082 | 1.78/5.82/61.42 | 926 | 14.8/48.6/36.3 (0.2) | 0.898 |
| real excl_tiers_n400_8453 | 10-20 | 33.10 | 110.2k | 125.8k | 0.142 | 1.80/5.28/61.18 | 418 | 5.5/48.4/45.8 (0.4) | 0.875 |
| real hbmonly_n400_8450 | 0-20 | 20.70 | 87.9k | 138.3k | 0.573 | 1.03/18.39/67.64 | 1828 | 97.9/0.0/0.0 (2.1) | 0.979 |
| real hbmonly_n400_8450 | 10-20 | 12.97 | 48.1k | 132.5k | 1.757 | 1.86/60.95/163.48 | 1133 | 94.7/0.0/0.0 (5.3) | 0.983 |
| real st10_tiers_n100_8456 | 0-20 | 9.39 | 37.7k | 38.1k | 0.012 | 0.11/0.23/0.94 | 1 | 92.7/6.9/0.4 (0.0) | 0.387 |
| real st10_tiers_n100_8456 | 10-20 | 8.62 | 27.4k | 27.4k | 0.000 | 0.09/0.17/0.36 | 0 | 90.7/8.7/0.6 (0.0) | 0.320 |

| run | window | r/s | new tok/s | computed tok/s | recompute/new | TTFT p50/p90/p99 (s) | >60 s | hit H/D/S (lost) % | s0 busy |
|---|---|---|---|---|---|---|---|---|---|
| sim st10_full_c300 | 0-20 | 32.72 | 125.0k | 125.8k | 0.006 | 0.17/0.89/13.61 | 164 | 98.8/1.1/0.1 (0.0) | 0.961 |
| sim st10_full_c300 | 10-20 | 33.97 | 115.3k | 115.7k | 0.004 | 0.15/0.25/1.65 | 4 | 98.2/1.6/0.1 (0.0) | 0.938 |
| sim st10_tiers_c300 | 0-20 | 32.54 | 124.4k | 126.4k | 0.016 | 0.20/0.97/13.07 | 162 | 48.7/34.5/16.7 (0.1) | 0.967 |
| sim st10_tiers_c300 | 10-20 | 33.77 | 114.5k | 117.5k | 0.026 | 0.18/0.29/1.89 | 6 | 45.7/32.9/21.4 (0.1) | 0.952 |
| sim st10_ssdheavy_c300 | 0-20 | 32.47 | 124.1k | 126.2k | 0.017 | 0.21/1.11/13.53 | 138 | 29.7/32.8/37.5 (0.1) | 0.967 |
| sim st10_ssdheavy_c300 | 10-20 | 33.80 | 114.0k | 117.1k | 0.028 | 0.19/0.31/2.02 | 5 | 28.4/30.6/40.9 (0.1) | 0.950 |
| sim excl_tiers_c300 | 0-20 | 32.55 | 124.5k | 126.5k | 0.016 | 0.20/1.00/13.54 | 159 | 53.0/33.7/13.3 (0.1) | 0.966 |
| sim excl_tiers_c300 | 10-20 | 33.83 | 114.6k | 117.6k | 0.026 | 0.18/0.29/1.73 | 6 | 49.3/33.3/17.3 (0.1) | 0.952 |
| sim hbmonly_c300 | 0-20 | 30.45 | 115.2k | 132.5k | 0.150 | 0.26/1.61/44.22 | 315 | 99.6/0.0/0.0 (0.4) | 0.986 |
| sim hbmonly_c300 | 10-20 | 30.03 | 97.3k | 128.1k | 0.317 | 0.20/1.99/20.47 | 134 | 99.3/0.0/0.0 (0.7) | 0.985 |
| sim st10_full_c400 | 0-20 | 37.34 | 141.4k | 142.1k | 0.005 | 0.61/1.88/60.90 | 710 | 96.0/3.8/0.2 (0.0) | 0.992 |
| sim st10_full_c400 | 10-20 | 42.26 | 139.6k | 140.0k | 0.003 | 0.48/1.30/33.28 | 194 | 94.5/5.1/0.4 (0.0) | 0.992 |
| sim st10_tiers_c400 | 0-20 | 35.90 | 137.1k | 141.8k | 0.034 | 0.79/3.71/60.74 | 594 | 21.1/41.7/37.2 (0.1) | 0.991 |
| sim st10_tiers_c400 | 10-20 | 39.77 | 131.5k | 139.9k | 0.064 | 0.65/2.53/44.99 | 218 | 18.3/42.3/39.2 (0.2) | 0.991 |
| sim st10_ssdheavy_c400 | 0-20 | 35.06 | 134.3k | 140.6k | 0.047 | 0.77/3.54/60.74 | 592 | 9.8/32.1/58.0 (0.1) | 0.990 |
| sim st10_ssdheavy_c400 | 10-20 | 38.36 | 126.6k | 138.1k | 0.090 | 0.63/2.83/48.38 | 176 | 3.8/33.6/62.4 (0.2) | 0.990 |
| sim excl_tiers_c400 | 0-20 | 35.86 | 137.0k | 141.3k | 0.031 | 0.87/3.49/60.92 | 660 | 26.0/45.3/28.6 (0.1) | 0.985 |
| sim excl_tiers_c400 | 10-20 | 39.79 | 131.7k | 139.3k | 0.057 | 0.74/2.38/56.04 | 230 | 22.8/43.4/33.6 (0.1) | 0.985 |
| sim hbmonly_c400 | 0-20 | 24.34 | 99.9k | 143.2k | 0.433 | 1.01/15.23/62.82 | 1908 | 98.6/0.0/0.0 (1.4) | 0.993 |
| sim hbmonly_c400 | 10-20 | 19.07 | 68.8k | 142.0k | 1.062 | 2.12/60.66/64.58 | 1222 | 97.1/0.0/0.0 (2.9) | 0.994 |
| sim st10_full_c100 | 0-20 | 9.42 | 37.7k | 38.2k | 0.012 | 0.10/0.20/0.84 | 1 | 100.0/0.0/0.0 (0.0) | 0.385 |
| sim st10_full_c100 | 10-20 | 8.63 | 27.5k | 27.5k | 0.000 | 0.08/0.16/0.35 | 0 | 100.0/0.0/0.0 (0.0) | 0.317 |
| sim st10_tiers_c100 | 0-20 | 9.42 | 37.7k | 38.2k | 0.012 | 0.10/0.20/0.83 | 0 | 93.2/6.6/0.2 (0.0) | 0.387 |
| sim st10_tiers_c100 | 10-20 | 8.62 | 27.4k | 27.4k | 0.000 | 0.08/0.16/0.36 | 0 | 91.4/8.2/0.4 (0.0) | 0.317 |
| sim st10_ssdheavy_c100 | 0-20 | 9.42 | 37.7k | 38.2k | 0.012 | 0.10/0.21/0.81 | 0 | 81.5/15.9/2.6 (0.0) | 0.389 |
| sim st10_ssdheavy_c100 | 10-20 | 8.63 | 27.4k | 27.4k | 0.000 | 0.08/0.17/0.36 | 0 | 73.8/23.4/2.8 (0.0) | 0.322 |
| sim excl_tiers_c100 | 0-20 | 9.42 | 37.7k | 38.2k | 0.012 | 0.10/0.20/0.82 | 1 | 94.7/5.1/0.1 (0.0) | 0.387 |
| sim excl_tiers_c100 | 10-20 | 8.63 | 27.4k | 27.5k | 0.000 | 0.08/0.16/0.36 | 0 | 93.8/6.0/0.3 (0.0) | 0.317 |
| sim hbmonly_c100 | 0-20 | 9.42 | 37.7k | 38.2k | 0.012 | 0.10/0.20/0.84 | 1 | 100.0/0.0/0.0 (0.0) | 0.385 |
| sim hbmonly_c100 | 10-20 | 8.63 | 27.5k | 27.5k | 0.000 | 0.08/0.16/0.35 | 0 | 100.0/0.0/0.0 (0.0) | 0.317 |

## Status
All eval runs are in. Next eval code: lane-m6-next (c5d4e09 = 128-token pages + hostgate), CPU 310 pass.
