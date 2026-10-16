# The scheduler (6_scheduler.md)

Real runs (bench/run.py, each on a freshly started server) against the handoff v2 targets. Written by `python -m bench.sched summarize`; the part from "## Notes" on is kept.

## Scheduling check (§3)

| Step | Run | Microbatches | Chunk for chunk | First difference | TTFT 128k / 32k / 8k alone / 1k alone (ms) | Worst TTFT deviation | In-run JIT |
|---|---|---|---|---|---|---|---|
| 0 plain | sched_check_plain_7708 | 28 / 28 | yes | – | 1142 vs 1118 / 1376 vs 1344 / 212 vs 224 / 52 vs 52 | 5.3% | 0 |
| 1 shortest first | sched_check_srpt8k_7708 | 28 / 28 | yes | – | 1739 vs 1679 / 758 vs 729 / 210 vs 224 / 53 vs 52 | 10.3% | 0 |
| 2 load budget | sched_check_budget_7708 | 55 / 55 | yes | – | 1856 vs 1742 / 759 vs 711 / 153 vs 155 / 53 vs 52 | 22.3% | 0 |
| 3 even split | sched_check_even_7708 | 58 / 58 | yes | – | 1852 vs 1742 / 757 vs 711 / 133 vs 132 / 51 vs 51 | 23.8% | 0 |
| 4 force-schedule | sched_check_good_7709 | 58 / 58 | yes | – | 1839 vs 1742 / 743 vs 711 / 133 vs 132 / 51 vs 51 | 5.6% | 0 |

## Force check (§4)

| Step | Run | 32k TTFT (s) | 1k TTFT p50 / p90 / p99 / max (s) | Last token (s) | Microbatches | 1k submit rate | Pass | Stage time / fit by microbatch tokens 0–255 / 256–1023 / 1024–8191 / 8192–8192 / 8193–32768 | In-run JIT |
|---|---|---|---|---|---|---|---|---|---|
| 1 shortest first | force_check_srpt8k_7708 | 108.714 vs 110.075 (x0.99) | 14.196 / 25.677 / 28.27 / 28.576 vs 15.079 / 26.948 / 29.618 / 29.933 (x0.94 / x0.95 / x0.95) | 108.715 vs 110.075 | 2005 / 2004 | 200.0/s | yes | s0: – / – / x1.236 / x0.959 / –; s1: – / – / x1.026 / x0.981 / –; s2: – / – / x1.050 / x0.975 / –; s3: – / – / x1.043 / x0.985 / – | 0 |
| 2 load budget | force_check_budget_7708 | 106.06 vs 107.506 (x0.99) | 15.152 / 21.803 / 25.36 / 25.846 vs 14.77 / 23.9 / 26.891 / 27.303 (x1.03 / x0.91 / x0.94) | 106.061 vs 107.506 | 1349 / 1429 | 200.0/s | yes | s0: – / x1.137 / x0.981 / – / x0.960; s1: – / x1.145 / x1.000 / – / x0.953; s2: – / x1.147 / x0.998 / – / x0.945; s3: – / x1.146 / x1.002 / – / x0.956 | 0 |
| 3 even split | force_check_even_7728 | 105.895 vs 107.506 (x0.99) | 15.057 / 21.657 / 25.185 / 25.663 vs 14.77 / 23.9 / 26.891 / 27.303 (x1.02 / x0.91 / x0.94) | 105.896 vs 107.506 | 1351 / 1429 | 200.0/s | yes | s0: – / x1.189 / x0.981 / – / x0.958; s1: – / x1.042 / x0.999 / – / x0.955; s2: – / x1.010 / x0.999 / – / x0.953; s3: – / x1.014 / x1.004 / – / x0.958 | 0 |
| 4 force-schedule | force_check_good_7708 | 60.903 vs 61.033 (x1.00) | 15.234 / 22.064 / 25.675 / 26.148 vs 14.77 / 24.093 / 27.122 / 27.519 (x1.03 / x0.92 / x0.95) | 106.135 vs 107.504 | 1342 / 1429 | 200.0/s | yes | s0: – / x1.084 / x0.977 / – / x0.961; s1: – / x0.890 / x0.997 / – / x0.953; s2: – / x0.829 / x0.996 / – / x0.946; s3: – / x0.847 / x1.000 / – / x0.949 | 0 |

## Sessions (§5)

| Step | N | Run | Rounds/s | New tokens/s | Computed tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Stage busy 0 / 1 / 2 / 3 | In-run JIT |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 shortest first | 400 | sched_srpt8k_n400_7709 | 25.5 vs 26.8 (x0.95) | 88.2k vs 93.0k (x0.95) | 114.1k vs 118.3k (x0.96) | 98.47% vs 98.58% | 0.33 / 0.41 / 1.94 / 167.7 vs 0.26 / 0.34 / 1.80 / 133.0 (x1.28 / x1.21 / x1.08) | 22 vs 12 | 87 / 80 / 85 / 86% vs 86 / 81 / 83 / 84% | 0 |
| └ sim x1.06 | | | 26.6 | 92.3k | 115.4k | 98.68% | 0.27 / 0.36 / 2.02 / 211.7 (measured x1.20 / x1.14 / x0.96) | 17 | | |
| 1 shortest first | 500 | sched_srpt8k_n500_7727 | 28.3 vs 29.2 (x0.97) | 102.0k vs 104.2k (x0.98) | 122.9k vs 127.7k (x0.96) | 98.85% vs 98.76% | 0.34 / 0.44 / 3.29 / 511.1 vs 0.26 / 0.36 / 3.39 / 564.7 (x1.31 / x1.24 / x0.97) | 101 vs 118 | 91 / 85 / 92 / 91% vs 91 / 86 / 88 / 89% | 0 |
| └ sim x1.06 | | | 28.6 | 102.6k | 123.0k | 98.88% | 0.28 / 0.38 / 4.42 / 573.6 (measured x1.23 / x1.16 / x0.75) | 125 | | |
| 2 load budget | 400 | sched_budget_n400_7728 | 25.4 vs 26.7 (x0.95) | 87.6k vs 92.3k (x0.95) | 115.0k vs 113.9k (x1.01) | 98.37% vs 98.77% | 0.30 / 0.96 / 2.30 / 281.5 vs 0.15 / 0.34 / 1.80 / 232.0 (x2.00 / x2.79 / x1.28) | 19 vs 19 | 88 / 82 / 87 / 88% vs 88 / 85 / 87 / 88% | 0 |
| └ sim x1.06 | | | 26.2 | 91.0k | 111.2k | 98.82% | 0.16 / 0.48 / 2.14 / 289.3 (measured x1.88 / x2.00 / x1.07) | 19 | | |
| 2 load budget | 500 | sched_budget_n500_7730 | 27.5 vs 28.6 (x0.96) | 99.8k vs 102.9k (x0.97) | 117.8k vs 124.4k (x0.95) | 98.97% vs 98.83% | 0.22 / 0.73 / 4.70 / 621.1 vs 0.16 / 0.59 / 4.61 / 577.4 (x1.37 / x1.25 / x1.02) | 146 vs 162 | 92 / 87 / 94 / 93% vs 92 / 88 / 91 / 91% | 4 |
| └ sim x1.06 | | | 28.0 | 101.4k | 120.1k | 98.95% | 0.18 / 0.72 / 6.60 / 604.6 (measured x1.19 / x1.01 / x0.71) | 161 | | |

Stage time / fit by microbatch tokens 0–255 / 256–1023 / 1024–8191 / 8192 / 8193–32768; hit kinds; GPU busy SM MHz / W / power-capped share:

| Step | N | Stage time / fit | Rounds by hit kind | GPU |
|---|---|---|---|---|
| 1 shortest first | 400 | s0: x1.255 / x1.210 / x1.063 / x1.040 / –; s1: x0.916 / x1.083 / x1.037 / x1.022 / –; s2: x0.914 / x1.088 / x1.059 / x1.060 / –; s3: x0.985 / x1.102 / x1.070 / x1.063 / – | evicted 175, full 31529, none_first_round 1426, superseded 416 | g0: 1636 / 896 / 93%; g1: 1657 / 887 / 93%; g2: 1638 / 901 / 94%; g3: 1623 / 903 / 94% |
| 1 shortest first | 500 | s0: x1.256 / x1.212 / x1.041 / x1.031 / –; s1: x0.916 / x1.087 / x1.040 / x1.023 / –; s2: x0.920 / x1.100 / x1.071 / x1.082 / –; s3: x0.989 / x1.102 / x1.065 / x1.058 / – | evicted 283, full 34209, none_first_round 1634, superseded 430 | g0: 1638 / 918 / 96%; g1: 1634 / 917 / 96%; g2: 1571 / 936 / 96%; g3: 1601 / 936 / 96% |
| 2 load budget | 400 | s0: x1.255 / x1.203 / x1.049 / – / x1.034; s1: x0.914 / x1.075 / x1.033 / – / x0.999; s2: x0.905 / x1.080 / x1.064 / – / x1.032; s3: x0.965 / x1.091 / x1.070 / – / x1.034 | evicted 180, full 31130, none_first_round 1415, superseded 411 | g0: 1633 / 905 / 94%; g1: 1656 / 890 / 93%; g2: 1641 / 907 / 94%; g3: 1625 / 910 / 95% |
| 2 load budget | 500 | s0: x1.280 / x1.212 / x1.042 / – / x1.026; s1: x0.915 / x1.078 / x1.049 / – / x1.009; s2: x0.908 / x1.088 / x1.097 / – / x1.066; s3: x0.977 / x1.089 / x1.077 / – / x1.043 | evicted 265, full 33380, none_first_round 1618, superseded 395 | g0: 1645 / 927 / 96%; g1: 1618 / 929 / 96%; g2: 1566 / 942 / 96%; g3: 1596 / 943 / 96% |

## Notes

### Scheduling check
- Every step forms the target's microbatches chunk for chunk (plain 28, srpt8k 28, budget 55, even 58, good 58).
- Until 82b801b, the first microbatch with several fresh prompts (4x1k, 8x1k) cost +16-25 ms of stage-0 host time,
  once. This put srpt8k, budget and even over the ±10% TTFT rule (+10 to +24%), all from that one constant. The warm-up
  now covers 3, 4, 8 and 16 fresh 1k prompts. With it, good (7709) has mb0 at 27.8 ms (host 8.0 ms) and its worst
  TTFT is 5.6%. The kernel behind the cost was not pinned. srpt8k, budget and even rerun when a window has spare time.
- Stage 0 starts mb0 7.6-9.3 ms after the first arrival. The 26 submits spread over 3.5-4.6 ms, and the 128k prompt
  then needs ~3 ms to read, parse and hand over. All 26 are admitted at 6.3-7.8 ms; dispatch is +0.1 ms and the
  stage-0 start +1.3 ms. The target has all 26 at t = 0.

### Force check
- Microbatch count: budget, even and good form 1342-1353 microbatches against a target of 1429. The CPU replay of
  the same engine on fit times forms 1355 (bench.replay run_force). The count therefore comes from the rules
  (engine vs simulator), not from GPU timing. The pass rule does not include it.

### Sessions: srpt8k N=400 (sched_srpt8k_n400_7709)
- **Throughput −4.8% vs target (−4.0% vs sim x1.06): the session slot (issues.md "Session slot").** req-frontend
  holds a finished session's slot through its last tool wait; the simulator frees it at the last token. Active
  sessions counted the simulator's way are 382-397 in the real run vs 400 in the sim, so new sessions start later
  (1312 vs 1383 started by 1200 s). The CPU replay at fit times gives 25.72 r/s with the slot held and 26.71 freed.
  The real run, 25.51, is −0.8% from the held replay.
- **The 1140-1200 s dip** is trajectory drift, not a server event. Matched (session, round) pairs show real submits
  trailing the sim's: lag p50 1.0 s at 480 s, 15 s at 720 s, 35 s at 1080 s, 60 s at 1140 s. The burst of rounds the
  sim submits at 1140-1200 s (904 and 1122 per 30 s) reaches the real run at 1200-1260 s (811, 1013), outside the
  20-min window. 7727's server start (11:37:18 = 1158 s on this run's clock) overlaps the dip, but the server's own
  measures do not move: computed tokens 131.9-134 k/s, busy 98-100%, 8k stage-0 time 60.6-61.6 ms, host 7.8 ms, GPU
  clocks unchanged, client lateness p50 3 ms. At most −1.6% computed for one 30 s window.
- **TTFT p50 +54 ms vs sim x1.06** (rounds at p45-p55: real 325 ms, sim 272 ms):

  | ms, mean of p45-p55 rounds | real | sim x1.06 |
  |---|---|---|
  | arrival → admitted | 1.5 | – |
  | admitted → plan formed | 25.9 | – |
  | plan formed → stage-0 start | 30.5 | – |
  | arrival → stage-0 start of last chunk (wait) | 58.9 | 31.8 |
  | stage-0 start → stage-3 end (traversal) | 264.9 | 239.7 |
  | stage-3 end → token, token → response, client − server | 1.2 + 0.7 + 1.2 | 3.3 |

  - **Traversal, +25 ms.** On 8,192-token microbatches the per-stage compute matches the sim (real 60.4 / 55.8 / 59.8
    / 60.3 ms vs sim 61.2 / 57.6 / 59.4 / 59.8). The difference is waiting before stages 2 and 3 (real 8.0 and 19.4
    ms vs sim 0.9 and 1.4). In every such gap, the stage was still busy with its previous microbatch. Stage 3 is as
    slow as stage 0 in the real run (60.3 vs 60.4), while the sim has stage 0 the slowest by 1.4-1.8 ms. Stage/fit
    ratios: s0 x1.04, s2/s3 x1.06; the sim scales every stage by 1.06. So the pipeline queues at stage 3.
  - **Wait, +27 ms.** Plans form up to 45 ms before stage 0 frees (form lead median 28.5 ms). 45% of rounds arrive
    after the next stage-0 plan is already formed, and they wait a median 76 ms to their first chunk, against 42 ms
    for the others. The sim decides when stage 0 frees. Caveat: the CPU replay at fit times puts the 45 ms lead's
    p50 cost at only +7 ms (0.262 vs 0.255 s; a 12 ms lead gives 0.260), so the lead is not the whole wait gap.
- Proposed fixes:
  1. Form lead per plan, from its predicted host time (p50 7 ms; 41 ms only for 8k microbatches with 56k-134k
     contexts), instead of a fixed 45 ms. Measure it with one srpt8k N=400 run.
  2. The stage-3 queueing comes from stage balance (stages 2 and 3 at x1.06 of fit vs stage 0 at x1.04). Profile what
     makes stages 2 and 3 slower than the fit. The layer split is fixed by the contract.

### Sessions: budget N=400 (sched_budget_n400_7728, before c0a58f3)
- 25.38 r/s, 87.6k new, TTFT 0.30/0.96/2.30 s, against sim x1.06 26.22, 91.0k, 0.16/0.48/2.14. From 720 s the
  microbatches grow from 4096 tokens to 15-29k; the sim stays at 4096.
- **Cause: the backlog B counted waiting rounds by their whole prompt.** The server has no declared prefix, so
  `Request.fresh` was the whole prompt. The simulator counts a waiting round by its new tokens (`fresh_prompt_tokens`,
  pipelined_chunked_prefill_admission.rs `prefill_budget`), even when an eviction means the prefix must be
  recomputed. Waiting rounds in both runs at the same times:

  | t (s) | 780 | 840 | 900 | 960 | 1020 | 1080 |
  |---|---|---|---|---|---|---|
  | real: waiting rounds, B share (M tokens) | 6, 0.72 | 9, 1.04 | 9, 1.48 | 8, 1.57 | 17, 2.33 | 13, 2.12 |
  | sim: waiting rounds, B share / to compute (M) | 5, 0.04 / 0.26 | 6, 0.00 / 0.63 | 11, 0.05 / 0.84 | 3, 0.03 / 0.30 | 10, 0.08 / 0.80 | 6, 0.04 / 0.57 |

  The waiting rounds are mostly evicted or superseded contexts. The sim's B stays below 1M, so its budget stays at
  4096. Ours passed 1M and grew the microbatches. Bigger microbatches meant longer TTFT, longer waits, more
  evictions and more recompute (computed − new 27.4k/s vs target 21.6k/s).
- Fix c0a58f3: `KvManager.shared_prefix` estimates the declared prefix from what the server knows: the HBM hit, the
  common prefix with a kept context the prompt does not extend, or the length of the evicted context. The backlog
  counts a waiting round's tokens beyond it. Replay is unchanged (it has the declared prefix). budget N=400 reruns.
- Per-microbatch cost is not the gap. Real stage 0 runs at x1.04-1.06 of the fit for its own microbatches; the sim's
  x1.06 runs at x1.043-1.051 of the fit for its own (per 120 s windows). By stage, at exactly 4096 tokens:
  s0/s1/s2/s3 = x1.046 / x1.030 / x1.062 / x1.067 of fit, all GPU time (host 5.7-10.9 ms, below the compute). The
  33.5 vs 31.6 ms comparison is between microbatches with different contents: ours carry more attention per token
  in that bin.
- Stages 2 and 3 being relatively slow (x1.06-1.07 vs stage 0's x1.046) makes stage 3 the pipeline's bottleneck.
  With 4 microbatches in flight, stage 0 then idles while work waits: 2.4% of the time at 240-720 s and 6.7% at
  720-840 s, every time with the next plan at in_flight ≥ 4. The sim has stage 0 the slowest, so it never sees
  this. srpt8k shows the same balance as queueing before stage 3.

### Sessions: budget N=500 (sched_budget_n500_7730, with c0a58f3)
- 27.54 r/s, 99.8k new, TTFT 0.22/0.73/4.70 s, against target 28.60, 102.9k, 0.16/0.59/4.61 (throughput x0.96,
  TTFT x1.36/x1.24/x1.02) and sim x1.06 28.03, 101.4k, 0.18/0.72/6.60 (TTFT x1.21/x1.02/x0.71). With the backlog
  fix, p90 matches the sim x1.06. What remains is ~40 ms at p50, the same fixed per-round cost as srpt8k.
- 4 in-run JITs: `_kpool_tail_seed_kernel` at mb 16263 (1 decode row + 6,352 prefill tokens), 44-45 ms per stage.
  The seed gets the tail slot mapping sliced past the decode rows (an int32 pointer off 16-byte alignment) and the
  prefill token count. Here the count was divisible by 16 for the first time with decode rows ahead. Warm-up shapes
  had 2,149 prefill tokens. 53d5563 adds 1-3 decode rows ahead of 2,160 prefill tokens (all four classes of slice
  alignment × count divisible by 16 are now compiled; a 1-token prefill slice cannot occur, since a 1-token row is a
  decode row). Warm-up now prints the compiled variant count of every Triton kernel in the fork's kpool ops and
  topk_canon (worker logs, "own Triton kernel variants after warm-up").
