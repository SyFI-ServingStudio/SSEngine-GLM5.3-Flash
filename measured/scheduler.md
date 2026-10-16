# The scheduler (6_scheduler.md)

Real runs (bench/run.py, each on a freshly started server) against the handoff v2 targets. Written by `python -m bench.sched summarize`; the part from "## Notes" on is kept.

## Scheduling check (§3)

| Step | Run | Microbatches | Chunk for chunk | First difference | TTFT 128k / 32k / 8k alone / 1k alone (ms) | Worst TTFT deviation | In-run JIT |
|---|---|---|---|---|---|---|---|
| 0 plain | sched_check_plain_7708 | 28 / 28 | yes | – | 1142 vs 1118 / 1376 vs 1344 / 212 vs 224 / 52 vs 52 | 5.3% | 0 |
| 1 shortest first | sched_check_srpt8k_7741 | 28 / 28 | yes | – | 1713 vs 1679 / 734 vs 729 / 209 vs 224 / 52 vs 52 | 6.5% | 0 |
| 2 load budget | sched_check_budget_7744 | 55 / 55 | yes | – | 1841 vs 1742 / 741 vs 711 / 154 vs 155 / 53 vs 52 | 5.7% | 0 |
| 3 even split | sched_check_even_7738 | 58 / 58 | yes | – | 1847 vs 1742 / 743 vs 711 / 131 vs 132 / 51 vs 51 | 6.1% | 0 |
| 4 force-schedule | sched_check_good_7738 | 58 / 58 | yes | – | 1934 vs 1742 / 827 vs 711 / 132 vs 132 / 51 vs 51 | 68.1% | 0 |

## Force check (§4)

| Step | Run | 32k TTFT (s) | 1k TTFT p50 / p90 / p99 / max (s) | Last token (s) | Microbatches | 1k submit rate | Pass | Stage time / fit by microbatch tokens 0–255 / 256–1023 / 1024–8191 / 8192–8192 / 8193–32768 | In-run JIT |
|---|---|---|---|---|---|---|---|---|---|
| 0 plain | force_check_plain_7742 | 0.41 vs 0.397 (x1.03) | 15.421 / 27.719 / 30.483 / 30.809 vs 15.261 / 27.131 / 29.802 / 30.109 (x1.01 / x1.02 / x1.02) | 110.774 vs 110.069 | 2005 / 2004 | 200.0/s | yes | s0: – / – / x1.310 / x0.939 / –; s1: – / – / x0.931 / x0.981 / –; s2: – / – / x0.999 / x1.008 / –; s3: – / – / x0.983 / x0.975 / – | 0 |
| 1 shortest first | force_check_srpt8k_7708 | 108.714 vs 110.075 (x0.99) | 14.196 / 25.677 / 28.27 / 28.576 vs 15.079 / 26.948 / 29.618 / 29.933 (x0.94 / x0.95 / x0.95) | 108.715 vs 110.075 | 2005 / 2004 | 200.0/s | yes | s0: – / – / x1.236 / x0.959 / –; s1: – / – / x1.026 / x0.981 / –; s2: – / – / x1.050 / x0.975 / –; s3: – / – / x1.043 / x0.985 / – | 0 |
| 2 load budget | force_check_budget_7708 | 106.06 vs 107.506 (x0.99) | 15.152 / 21.803 / 25.36 / 25.846 vs 14.77 / 23.9 / 26.891 / 27.303 (x1.03 / x0.91 / x0.94) | 106.061 vs 107.506 | 1349 / 1429 | 200.0/s | yes | s0: – / x1.137 / x0.981 / – / x0.960; s1: – / x1.145 / x1.000 / – / x0.953; s2: – / x1.147 / x0.998 / – / x0.945; s3: – / x1.146 / x1.002 / – / x0.956 | 0 |
| 3 even split | force_check_even_7738 | 107.803 vs 107.506 (x1.00) | 16.294 / 22.879 / 27.057 / 27.557 vs 14.77 / 23.9 / 26.891 / 27.303 (x1.10 / x0.96 / x1.01) | 107.804 vs 107.506 | 1276 / 1429 | 200.0/s | no: 1k p50 16.294 vs 14.77 | s0: – / x2.223 / x0.994 / – / x0.957; s1: – / x0.954 / x0.993 / – / x0.960; s2: – / x0.910 / x1.003 / – / x0.956; s3: – / x0.930 / x0.988 / – / x0.937 | 0 |
| 4 force-schedule | force_check_good_7708 | 60.903 vs 61.033 (x1.00) | 15.234 / 22.064 / 25.675 / 26.148 vs 14.77 / 24.093 / 27.122 / 27.519 (x1.03 / x0.92 / x0.95) | 106.135 vs 107.504 | 1342 / 1429 | 200.0/s | yes | s0: – / x1.084 / x0.977 / – / x0.961; s1: – / x0.890 / x0.997 / – / x0.953; s2: – / x0.829 / x0.996 / – / x0.946; s3: – / x0.847 / x1.000 / – / x0.949 | 0 |

## Sessions (§5)

| Step | N | Run | Rounds/s | New tokens/s | Computed tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Stage busy 0 / 1 / 2 / 3 | In-run JIT |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 shortest first | 400 | sched_srpt8k_n400_7709 (slot held, old client) | 25.5 vs 26.8 (x0.95) | 88.2k vs 93.0k (x0.95) | 114.1k vs 118.3k (x0.96) | 98.47% vs 98.58% | 0.33 / 0.41 / 1.94 / 167.7 vs 0.26 / 0.34 / 1.80 / 133.0 (x1.28 / x1.21 / x1.08) | 22 vs 12 | 87 / 80 / 85 / 86% vs 86 / 81 / 83 / 84% | 0 |
| └ sim x1.06 | | | 26.6 | 92.3k | 115.4k | 98.68% | 0.27 / 0.36 / 2.02 / 211.7 (measured x1.20 / x1.14 / x0.96) | 17 | | |
| 1 shortest first | 500 | sched_srpt8k_n500_7727 (slot held, old client) | 28.3 vs 29.2 (x0.97) | 102.0k vs 104.2k (x0.98) | 122.9k vs 127.7k (x0.96) | 98.85% vs 98.76% | 0.34 / 0.44 / 3.29 / 511.1 vs 0.26 / 0.36 / 3.39 / 564.7 (x1.31 / x1.24 / x0.97) | 101 vs 118 | 91 / 85 / 92 / 91% vs 91 / 86 / 88 / 89% | 0 |
| └ sim x1.06 | | | 28.6 | 102.6k | 123.0k | 98.88% | 0.28 / 0.38 / 4.42 / 573.6 (measured x1.23 / x1.16 / x0.75) | 125 | | |
| 2 load budget | 400 | sched_budget_n400_7733 (slot held, old client) | 25.2 vs 26.7 (x0.94) | 86.9k vs 92.3k (x0.94) | 107.4k vs 113.9k (x0.94) | 98.75% vs 98.77% | 0.17 / 0.60 / 2.26 / 298.3 vs 0.15 / 0.34 / 1.80 / 232.0 (x1.15 / x1.75 / x1.26) | 28 vs 19 | 88 / 82 / 89 / 88% vs 88 / 85 / 87 / 88% | 0 |
| └ sim x1.06 | | | 26.2 | 91.0k | 111.2k | 98.82% | 0.16 / 0.48 / 2.14 / 289.3 (measured x1.08 / x1.26 / x1.06) | 19 | | |
| 2 load budget | 500 | sched_budget_n500_7730 (slot held, old client) | 27.5 vs 28.6 (x0.96) | 99.8k vs 102.9k (x0.97) | 117.8k vs 124.4k (x0.95) | 98.97% vs 98.83% | 0.22 / 0.73 / 4.70 / 621.1 vs 0.16 / 0.59 / 4.61 / 577.4 (x1.37 / x1.25 / x1.02) | 146 vs 162 | 92 / 87 / 94 / 93% vs 92 / 88 / 91 / 91% | 4 |
| └ sim x1.06 | | | 28.0 | 101.4k | 120.1k | 98.95% | 0.18 / 0.72 / 6.60 / 604.6 (measured x1.19 / x1.01 / x0.71) | 161 | | |
| 3 even split | 400 | sched_even_n400_7738 (slot held, old client) | 3.8 vs 26.7 (x0.14) | 21.6k vs 92.3k (x0.23) | 21.8k vs 114.0k (x0.19) | 99.89% vs 98.77% | 0.46 / 0.89 / 4.01 / 123.0 vs 0.15 / 0.34 / 1.83 / 179.8 (x3.13 / x2.60 / x2.19) | 5 vs 18 | 17 / 16 / 17 / 16% vs 89 / 86 / 89 / 90% | 0 |
| └ sim x1.06 | | | 26.2 | 91.1k | 111.1k | 98.83% | 0.16 / 0.48 / 1.99 / 304.2 (measured x2.93 / x1.85 / x2.01) | 23 | | |
| 3 even split | 500 | sched_even_n500_7741 (slot held, old client) | 27.3 vs 28.6 (x0.95) | 99.1k vs 102.9k (x0.96) | 116.3k vs 124.5k (x0.93) | 99.00% vs 98.83% | 0.19 / 0.74 / 6.61 / 647.8 vs 0.16 / 0.60 / 4.38 / 579.6 (x1.22 / x1.25 / x1.51) | 159 vs 159 | 93 / 85 / 92 / 91% vs 93 / 89 / 91 / 92% | 0 |
| └ sim x1.06 | | | 28.0 | 101.4k | 120.1k | 98.95% | 0.18 / 0.70 / 6.59 / 610.2 (measured x1.06 / x1.05 / x1.00) | 162 | | |
| 4 force-schedule | 400 | sched_good_n400_7742 (slot held, old client) | 24.4 vs 25.6 (x0.95) | 84.8k vs 89.4k (x0.95) | 107.2k vs 113.7k (x0.94) | 98.60% vs 98.55% | 0.18 / 1.16 / 27.75 / 66.4 vs 0.15 / 0.81 / 9.85 / 63.7 (x1.17 / x1.44 / x2.82) | 236 vs 165 | 90 / 83 / 90 / 89% vs 90 / 87 / 89 / 90% | 0 |
| └ sim x1.06 | | | 25.2 | 87.8k | 111.0k | 98.60% | 0.16 / 1.06 / 18.73 / 63.3 (measured x1.09 / x1.10 / x1.48) | 223 | | |
| 4 force-schedule | 500 | sched_good_n500_7744 (slot held, old client) | 25.3 vs 25.8 (x0.98) | 89.8k vs 92.6k (x0.97) | 122.4k vs 128.2k (x0.95) | 97.97% vs 97.82% | 0.43 / 4.70 / 61.72 / 67.0 vs 0.29 / 4.22 / 61.55 / 68.4 (x1.47 / x1.11 / x1.00) | 1014 vs 979 | 93 / 85 / 92 / 91% vs 93 / 88 / 91 / 92% | 0 |
| └ sim x1.06 | | | 25.4 | 90.9k | 124.4k | 97.90% | 0.38 / 5.50 / 61.89 / 66.9 (measured x1.16 / x0.85 / x1.00) | 1024 | | |

Stage time / fit by microbatch tokens 0–255 / 256–1023 / 1024–8191 / 8192 / 8193–32768; hit kinds; GPU busy SM MHz / W / power-capped share:

| Step | N | Stage time / fit | Rounds by hit kind | GPU |
|---|---|---|---|---|
| 1 shortest first | 400 | s0: x1.255 / x1.210 / x1.063 / x1.040 / –; s1: x0.916 / x1.083 / x1.037 / x1.022 / –; s2: x0.914 / x1.088 / x1.059 / x1.060 / –; s3: x0.985 / x1.102 / x1.070 / x1.063 / – | evicted 175, full 31529, none_first_round 1426, superseded 416 | g0: 1636 / 896 / 93%; g1: 1657 / 887 / 93%; g2: 1638 / 901 / 94%; g3: 1623 / 903 / 94% |
| 1 shortest first | 500 | s0: x1.256 / x1.212 / x1.041 / x1.031 / –; s1: x0.916 / x1.087 / x1.040 / x1.023 / –; s2: x0.920 / x1.100 / x1.071 / x1.082 / –; s3: x0.989 / x1.102 / x1.065 / x1.058 / – | evicted 283, full 34209, none_first_round 1634, superseded 430 | g0: 1638 / 918 / 96%; g1: 1634 / 917 / 96%; g2: 1571 / 936 / 96%; g3: 1601 / 936 / 96% |
| 2 load budget | 400 | s0: x1.298 / x1.227 / x1.060 / – / x1.028; s1: x0.914 / x1.072 / x1.030 / – / x0.993; s2: x0.908 / x1.084 / x1.083 / – / x1.040; s3: x0.968 / x1.084 / x1.065 / – / x1.024 | evicted 146, full 31083, none_first_round 1416, superseded 375 | g0: 1690 / 909 / 94%; g1: 1654 / 915 / 95%; g2: 1595 / 925 / 95%; g3: 1619 / 926 / 95% |
| 2 load budget | 500 | s0: x1.280 / x1.212 / x1.042 / – / x1.026; s1: x0.915 / x1.078 / x1.049 / – / x1.009; s2: x0.908 / x1.088 / x1.097 / – / x1.066; s3: x0.977 / x1.089 / x1.077 / – / x1.043 | evicted 265, full 33380, none_first_round 1618, superseded 395 | g0: 1645 / 927 / 96%; g1: 1618 / 929 / 96%; g2: 1566 / 942 / 96%; g3: 1596 / 943 / 96% |
| 3 even split | 400 | s0: x1.264 / x1.153 / x1.042 / – / x1.039; s1: x0.920 / x1.028 / x1.029 / – / x0.992; s2: x0.909 / x1.025 / x1.071 / – / x1.042; s3: x0.983 / x1.023 / x1.053 / – / x1.023 | full 4041, none_first_round 459, superseded 7 | g0: 1709 / 828 / 86%; g1: 1685 / 829 / 86%; g2: 1650 / 835 / 85%; g3: 1670 / 834 / 86% |
| 3 even split | 500 | s0: x1.232 / x1.164 / x1.070 / – / x1.042; s1: x0.892 / x1.032 / x1.038 / – / x1.002; s2: x0.934 / x1.040 / x1.094 / – / x1.056; s3: x0.937 / x1.029 / x1.076 / – / x1.037 | evicted 255, full 33191, none_first_round 1618, superseded 393 | g0: 1663 / 932 / 96%; g1: 1635 / 929 / 97%; g2: 1572 / 940 / 97%; g3: 1604 / 942 / 97% |
| 4 force-schedule | 400 | s0: x1.453 / x1.163 / x1.064 / – / x1.042; s1: x0.890 / x1.033 / x1.036 / – / x0.996; s2: x0.932 / x1.041 / x1.088 / – / x1.046; s3: x0.934 / x1.030 / x1.068 / – / x1.028 | evicted 145, full 29164, none_first_round 1362, superseded 351 | g0: 1692 / 913 / 94%; g1: 1652 / 916 / 95%; g2: 1596 / 927 / 96%; g3: 1622 / 929 / 96% |
| 4 force-schedule | 500 | s0: x1.247 / x1.167 / x1.068 / – / x1.048; s1: x0.896 / x1.034 / x1.038 / – / x1.010; s2: x0.936 / x1.039 / x1.087 / – / x1.061; s3: x0.937 / x1.027 / x1.069 / – / x1.043 | evicted 405, full 29577, none_first_round 1513, superseded 311 | g0: 1659 / 930 / 96%; g1: 1642 / 925 / 96%; g2: 1586 / 935 / 97%; g3: 1616 / 937 / 97% |

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
- **The ~40 ms at p50 (srpt8k and budget): the extra in-flight plan queues in front of a slower later stage.** The
  dispatch rule lets a plan queue behind stage 0's running one as the 5th in flight (27,284 of 30,370 plans in 7730
  formed at in_flight 5). The sim and the replay cap it at 4. Stage 0 is the slowest stage in the sim; in the real
  runs, stages 2 and 3 are slower relative to the fit (7730 4,096-token microbatches: s0 33.25, s1 32.29, s2 34.62,
  s3 34.32 ms). On GPUs 0-3 (7727, 7730), physical GPU 2 also clocks lower: 1546-1550 MHz vs 1601-1628. So the 5th
  plan waits before stage 2 (7730: 21.0 ms mean; sim 0.25), and every round it carries pays that wait.
  CPU replay of budget N=500 (bench.replay, sim x1.06 cost_log run in wt-pp-hitrule logs/20261016_8_budget_c500_costlog):

  | Replay (budget N=500, form lead 45 ms) | r/s | TTFT p50 / p90 / p99 (s) | queue before stage 2 (ms, mean) |
  |---|---|---|---|
  | stage time / fit x1.06 each, 4 in flight | 27.68 | 0.171 / 0.717 / 4.41 | 0.9 |
  | real stage balance (1.042 / 1.049 / 1.097 / 1.077), 4 in flight | 27.56 | 0.176 / 0.715 / 5.06 | 4.2 |
  | real stage balance, 5 in flight (as the server) | 27.52 | 0.210 / 0.836 / 5.09 | 40.7 |
  | real stage balance, 5 in flight, pacing 2 ms | 27.53 | 0.174 / 0.714 / 5.58 | 2.5 |
  | real 7730 | 27.54 | 0.22 / 0.73 / 4.70 | 21.0 |

  srpt8k N=400 (stage balance 1.040 / 1.022 / 1.060 / 1.063): 5 in flight 0.313 / 0.393 / 1.87, real 0.33 / 0.41 /
  1.94. With pacing 2 ms: 0.272 / 0.363 / 1.91, against sim x1.06 0.27 / 0.36 / 2.02. Throughput is unchanged
  (25.59 vs 25.61).
- **Tried ead8d9a: at most 4 in flight, the early-formed plan counted** (6_scheduler.md §2; reverted, see "Dispatch
  decision" below). Early forming (the form
  lead) stays within that cap. Once the pipeline is full, the next plan forms on the stage-3 exit report, and stage 0
  waits for report + form + first kernel. In 7730 that is 1.1 ms p50 from exit to dispatch, plus 0.5-1.2 ms from
  dispatch to the stage-0 start. Replay throughput (r/s) and TTFT p50 / p90 / p99 (s) by the delay from forming to
  the stage-0 start (host):

  | host (ms) | budget N=500: 4 in flight | 5 in flight | 5 + pacing 2 ms | srpt8k N=400: 4 in flight | 5 in flight | 5 + pacing |
  |---|---|---|---|---|---|---|
  | 0 | 27.56, 0.176 / 0.715 / 5.06 | 27.52, 0.210 / 0.836 / 5.09 | 27.53, 0.174 / 0.714 / 5.58 | 25.62, 0.274 / 0.362 / 1.86 | 25.59, 0.313 / 0.393 / 1.87 | 25.61, 0.272 / 0.363 / 1.91 |
  | 2 | 27.60, 0.188 / 0.753 / 4.94 | 27.53, 0.210 / 0.835 / 4.80 | 27.56, 0.179 / 0.716 / 4.98 | 25.61, 0.276 / 0.362 / 1.85 | 25.58, 0.313 / 0.394 / 1.89 | 25.58, 0.275 / 0.366 / 1.92 |
  | 3 | 27.51, 0.178 / 0.717 / 5.86 | 27.54, 0.209 / 0.835 / 4.76 | 27.53, 0.179 / 0.721 / 4.91 | 25.60, 0.276 / 0.363 / 1.90 | 25.58, 0.313 / 0.394 / 1.87 | 25.60, 0.276 / 0.368 / 2.00 |
  | 7 | 27.32, 0.183 / 0.731 / 6.20 | 27.55, 0.209 / 0.834 / 5.22 | 27.52, 0.181 / 0.722 / 5.60 | 25.55, 0.279 / 0.368 / 1.93 | 25.59, 0.313 / 0.394 / 1.86 | 25.60, 0.280 / 0.371 / 1.97 |
  | 10 | 27.17, 0.191 / 0.745 / 6.87 | 27.55, 0.209 / 0.837 / 4.80 | 27.53, 0.184 / 0.727 / 5.34 | 25.53, 0.283 / 0.374 / 2.05 | 25.58, 0.314 / 0.395 / 1.87 | 25.59, 0.283 / 0.377 / 2.00 |

  At the measured 2-3 ms, 4 in flight is within 0.3% of 5 in flight for both steps (the replay's run-to-run spread is
  of that order). It loses 0.8% only at 7 ms (budget's 4,096-token microbatches). Pacing (a152856, off, no run
  names) was kept in the code; it became the default (1a5c8e7).
- **Stage balance (stages 2/3 slower than the fit's ratio): power cap, per stage.** Under no sustained load the stages
  match the fit equally (force checks, 1,024-8,191-token bin: s0 / s1 / s2 / s3 x0.98 / 1.00 / 1.00 / 1.00). In the
  session runs, every GPU sits at its 1,000 W cap for 93-96% of busy samples. The stages that draw more power clock
  lower:
  - 7709 (GPUs 4-7): SM 1603 / 1624 / 1606 / 1592 MHz at 932 / 924 / 936 / 936 W;
  - 7730 (GPUs 0-3): SM 1628 / 1601 / 1546 / 1579 MHz at 950 / 950 / 963 / 964 W.
  Stage time / fit follows the clocks (7730: s1 x1.049, s2 x1.097). The sim applies one x1.06 to every stage, so it
  keeps stage 0 the slowest. Not a software fix; pacing keeps the imbalance from turning into queueing.

### Sessions: budget N=400 (sched_budget_n400_7733, cap 4 + JIT warm-up)
- 25.16 r/s, 86.9k new, TTFT 0.17/0.60/2.26 s, 0 in-run JITs (the tail seed has 4 variants after warm-up). Sim x1.06:
  26.22, 91.0k, 0.16/0.48/2.14.
- **Throughput −4.0%, mostly the session slot.** Replay of budget N=400 (r/s, new tokens):
  - slot held through the last tool wait (as req-frontend), real stage balance: 25.32, 87.4k. Real is −0.6% from it.
  - slot freed at the last token (as the sim): 25.93 (−2.3% from held).
  - uniform x1.06, slot freed: 26.02 (−0.8% from the sim: engine vs simulator rules).
  - Stage balance moves it at most 0.4%.
  The same rule costs budget −0.6% and srpt8k 0.0% at N=500, hence about −4% at N=400 against −1% at N=500. Per
  120 s: from ~450 s the real run has 386-398 sessions active (sim-way count) against 400; by 1200 s, 1309 sessions
  have started against 1374. From 720 s every microbatch is a full 4,096 tokens in both runs, but the stages are
  94-97% busy against the sim's 99%.
- **p90 (x1.25) comes from the start-up burst.** 70% of the rounds over 480 ms arrive at 120-360 s, in both runs. At
  120-240 s, TTFT p50/p90 is 593/924 ms (sim 459/700, replay 514/772). Real microbatches are larger there (mean
  10.8k vs 9.3k, p90 21.7k vs 17.4k), and stage 2 idles 8.9%. From 360 s, every window's p90 is within ±15% of the sim.
- **Cap 4 bubble while saturated.** Stage-2 idle at 720-1200 s is 3.4% (7733, cap 4) against 0.7% (7730, cap 5,
  N=500). Of it, 1.2% waits on a full plan formed on an exit report and 1.7% on plans delayed by earlier ones. The
  closed loop (sum of the stages + host) / 4 = 34.1 ms is only 0.5 ms below stage 2's 34.6 ms, so host jitter (p90 4.7
  ms) opens bubbles. Estimated at ~1% of the run. Decision: cap 5 + pacing (below).

### Sessions: even N=500 (7741) and good N=400 (7742), cap 4 + expandable segments + regrow
- **even N=400 (7738): stage 2 ran out of memory** at ~240 s, allocating 2 GiB for MLA decode output. The microbatch
  (4649) had 32,698 tokens, 5 requests and contexts up to 69k. PyTorch had 164.4 GB allocated, 7.2 GB reserved but
  unallocated, and 1.8 GB free. Budget runs already showed 8-11 allocator retries per stage at peak_reserved 186 GB.
  Fixes:
  - 848903a: expandable segments. Retries drop to 0 (force_check_even: 12-79 → 0).
  - 25f128a: run the allocator-peak shapes again after graph capture. Without it, the first microbatches allocate
    from the driver: sched_check_good +37 ms (+80 ms with expandable segments), force_check_plain 32k TTFT 0.64 s.
- **even N=500 (7741)**: 27.31 r/s, TTFT 0.19/0.74/6.61 s. Against sim x1.06: TTFT x1.06/x1.05/x1.00, throughput
  −2.6%. Bottleneck stage 2 idles 5.2% at 360-1200 s, against 2.9% in budget N=500 7730 (cap 5). From 480 to 960 s,
  computed is −5%, with busy at 96-97% against the sim's 99%.
- **good N=400 (7742)**: 24.42 r/s, TTFT 0.18/1.16/27.75 s. Against sim x1.06: TTFT x1.09/x1.10/x1.48, throughput
  −3.2% (mostly the session slot, as at budget N=400). The p99 comes from the overload at the end of the run. Rounds
  over 10 s: real 576, sim 405. By arrival window, 840-1200 s holds 467 of them (sim 313); at 1080-1200 s alone it is
  199 against 67. With lower throughput, the backlog in the overload grows faster.
  At 720-1200 s, real computes 109.7-115.4k tokens/s against the sim's 118.9-120.5k (−5% to −8%), with stage 0 busy
  93-96% against 98-99%. Stage 0 idles 4.8% there: 3.6% at 4 in flight (the cap-4 bubble, gap p50 3.0 ms) and 1.0%
  at 3 (host, p50 0.9 ms). The slot rule adds −2.3% of sessions at N=400. 7757 reruns it with both removed.
- force_check_even 7738 (cap 4): 1k TTFT p50 16.29 s vs 14.77 (x1.10). Cap 4 forms on exits, so the 1k prompts
  arriving at 200/s pile into larger microbatches: 1276 microbatches, mean 12.9k tokens, against 1351 and 12.2k
  (7728, cap 5); the target has 1429.

### Dispatch decision: 5 in flight + pacing 2 ms (1a5c8e7)
- Cap 4 left bubbles once saturated (stage-2 idle 5.2% in even N=500 7741 vs 2.9% at cap 5; force_check_even 1k p50
  x1.10). The default is back to 5 in flight (one plan queued behind stage 0's running one), and the queued plan
  forms only when it would reach every later stage no earlier than 2 ms before that stage frees (per-stage EWMAs,
  the next plan assumed as large as the last, host time predicted from stage-0 idle samples). Replay: p50 as cap 4,
  throughput as cap 5 (table above).
- From 16:11 PDT req-frontend frees a session's slot at its last token (365fba07); the replay does the same
  (b9fa4d5). Session runs before that are marked "slot held, old client".

### Sessions: good N=500 (sched_good_n500_7744, cap 4, old client)
- 25.28 r/s, 89.8k new, TTFT 0.43/4.70/61.72 s, max 67.0, 1014 over 60 s. Sim x1.06: 0.38/5.50/61.89, max 66.9, 1024
  (x1.16/x0.85/x1.00).
- **p50 x1.16: stage 0 (the bottleneck in both runs) idles 4.4% at 120-1200 s; the sim's is busy 99-100%.** By
  in-flight count at the gap: 2.4% at 4 (the cap-4 bubble; 3,891 gaps, p50 3.6 ms), 1.7% at 3 (host between exit
  report and the next stage-0 start, p50 1.0 ms), 0.4% with 1-2 in flight. even N=500 7741 (cap 4) shows the same
  (5.2%: 3.2% at 4, 1.7% at 3). Per microbatch, stage 0 is within ±3% of sim x1.06 in every token bin (4,096-8,191:
  36.7 vs 35.7 ms).
- With less capacity the backlog grows, so the good step forms bigger microbatches (16k-32k tokens: 3,051 vs 1,969),
  and every round waits for and traverses bigger ones. By arrival window, p50 is above the sim in the start-up burst
  (120-240 s: 543 vs 424 ms; 240-360 s: 238 vs 166) and at 720-960 s (696 vs 622, 1051 vs 884); computed tokens are
  −5% to −6% at 480-720 s. The slot rule costs ≤0.6% at N=500. Cap 5 + pacing removes the 2.4%; 7758 measures it.

