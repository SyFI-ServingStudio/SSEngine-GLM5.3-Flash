# Prefix cache and chunked prefill (5_prefix_cache.md)

## Prefix bring-up

Measured / target (ratio), medians (p50) over each case's sessions, TTFT also p90 and p99 over them; stage compute and sends from CUDA events (microbatches.jsonl).

| P | A | n | hit = P | Round 0 TTFT p50 (ms) | Round 1 TTFT p50 (ms) | Round 1 server TTFT p50 (ms) | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Each send | Token = no cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8k | 1k | 5 | 5/5 | 210.0 / 223.6 (x0.94); p90 210.4, p99 210.5 | 57.2 / 53.3 (x1.07); p90 84.8, p99 100.9 | – | 13.9 / 12.7 (x1.10) | 12.4 / 13.3 (x0.94) | 12.8 / 13.5 (x0.95) | 13.1 / 13.7 (x0.95) | 0.206 / 0.068 | 5/5 |
| 8k | 8k | 5 | 5/5 | 209.7 / 223.6 (x0.94); p90 210.2, p99 210.3 | 215.7 / 223.8 (x0.96); p90 216.9, p99 217.1 | – | 53.5 / 57.1 (x0.94) | 50.6 / 54.2 (x0.93) | 52.2 / 55.3 (x0.94) | 52.5 / 55.7 (x0.94) | 0.675 / 0.482 | 5/5 |
| 32k | 1k | 5 | 5/5 | 401.2 / 396.8 (x1.01); p90 401.9, p99 402.0 | 59.8 / 53.8 (x1.11); p90 60.0, p99 60.1 | – | 14.1 / 12.8 (x1.10) | 12.5 / 13.4 (x0.94) | 13.0 / 13.6 (x0.96) | 13.3 / 13.8 (x0.97) | 0.208 / 0.068 | 5/5 |
| 32k | 8k | 5 | 5/5 | 401.1 / 396.8 (x1.01); p90 402.2, p99 402.4 | 222.5 / 226.4 (x0.98); p90 222.8, p99 222.9 | – | 54.9 / 57.8 (x0.95) | 51.2 / 54.6 (x0.94) | 53.2 / 56.0 (x0.95) | 53.7 / 56.4 (x0.95) | 0.671 / 0.482 | 5/5 |
| 128k | 1k | 5 | 5/5 | 1185.5 / 1118.2 (x1.06); p90 1267.3, p99 1314.8 | 68.4 / 55.4 (x1.24); p90 68.9, p99 69.1 | – | 15.1 / 13.3 (x1.14) | 13.2 / 13.7 (x0.96) | 13.8 / 14.0 (x0.98) | 14.0 / 14.3 (x0.98) | 0.210 / 0.068 | 5/5 |
| 128k | 8k | 5 | 5/5 | 1180.2 / 1118.2 (x1.06); p90 1186.0, p99 1186.6 | 251.9 / 239.0 (x1.05); p90 303.1, p99 333.5 | – | 61.6 / 61.3 (x1.01) | 55.6 / 56.9 (x0.98) | 59.8 / 59.5 (x1.00) | 60.0 / 59.8 (x1.00) | 0.660 / 0.482 | 5/5 |
| 512k | 1k | 5 | 5/5 | 5110.0 / 4408.1 (x1.16); p90 19109.4, p99 27486.3 | 112.8 / 64.3 (x1.75); p90 116.0, p99 117.4 | – | 20.9 / 15.7 (x1.33) | 16.3 / 15.3 (x1.07) | 18.4 / 16.5 (x1.12) | 18.8 / 16.7 (x1.12) | 0.208 / 0.068 | 5/5 |
| 512k | 8k | 5 | 5/5 | 5117.2 / 4408.1 (x1.16); p90 5128.4, p99 5133.2 | 386.3 / 287.7 (x1.34); p90 389.2, p99 390.5 | – | 91.3 / 74.6 (x1.22) | 73.9 / 65.8 (x1.12) | 87.8 / 72.8 (x1.21) | 88.4 / 73.1 (x1.21) | 0.670 / 0.482 | 5/5 |

## Cache on vs off

40 round-1 prompts, 40 with a hit; 40 tokens equal to the no-cache token; all rounds 80/80.

## Sessions

| N | Rounds/s | New tokens/s | Computed tokens/s | Prefix hit share | TTFT p50 / p90 / p99 (s) | Server TTFT p50 / p90 / p99 (s) | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| 400 (prefix_n400_7683) | 24.2 / 24.9 (x0.97) | 84.4k / 87.5k (x0.96) | 111.2k / 115.4k (x0.96) | 98.28% / 98.28% | 0.64 / 4.43 / 7.57 vs 0.55 / 3.66 / 5.98 (x1.15 / x1.21 / x1.27) | 0.73 / 4.80 / 8.44 (x1.32 / x1.31 / x1.41) | 84 / 79 / 85 / 84% vs 84 / 79 / 82 / 82% |
| └ sim x1.06 | 24.6 (x0.98) | 86.3k (x0.98) | 112.5k (x0.99) | 98.35% | 0.75 / 4.25 / 8.22 (measured x0.85 / x1.04 / x0.92) | | |
| 500 (prefix_n500_7693) | 21.6 / 22.8 (x0.95) | 83.1k / 86.6k (x0.96) | 122.4k / 126.8k (x0.97) | 96.85% / 97.01% | 3.22 / 14.26 / 26.77 vs 3.01 / 13.03 / 23.93 (x1.07 / x1.09 / x1.12) | 3.59 / 17.41 / 26.64 (x1.19 / x1.34 / x1.11) | 90 / 85 / 92 / 91% vs 91 / 86 / 88 / 89% |
| └ sim x1.06 | 22.0 (x0.98) | 84.5k (x0.98) | 121.9k (x1.00) | 97.07% | 3.85 / 14.37 / 24.25 (measured x0.84 / x0.99 / x1.10) | | |

| N | Pool (tokens) | Evicted sessions (tokens) | Preemptions | Prefix shortfall | Rounds failed | Microbatches / fit, median per stage (p10–p90) | In-run JIT | GPU busy: SM MHz / W / power-capped |
|---|---|---|---|---|---|---|---|---|
| 400 | 61,258,368 | 1,135 (78,407,949) | 0 | 0 | 0 | s0: x1.029 (0.98–1.13, n=21584); s1: x1.035 (0.98–1.08, n=21584); s2: x1.086 (1.02–1.12, n=21584); s3: x1.061 (1.01–1.11, n=21584) | 4 (mb 1562 BuildPrefillChunkMetadataKernel.kernel 70.1 ms) | g0: 1678 / 875 / 91%; g1: 1664 / 884 / 93%; g2: 1615 / 896 / 93%; g3: 1640 / 893 / 94% |
| 500 | 61,258,368 | 1,478 (105,739,604) | 0 | 0 | 0 | s0: x1.027 (0.99–1.07, n=21251); s1: x1.031 (0.98–1.07, n=21251); s2: x1.088 (1.05–1.12, n=21251); s3: x1.061 (1.03–1.09, n=21251) | 4 (mb 2015 BuildPrefillChunkMetadataKernel.kernel 70.8 ms) | g0: 1645 / 914 / 95%; g1: 1637 / 921 / 96%; g2: 1577 / 932 / 96%; g3: 1610 / 931 / 96% |

| N | Rounds by hit kind | Superseded: recomputed tokens | Evicted: recomputed tokens | Admit waits |
|---|---|---|---|---|
| 400 | evicted 169, full 28844, none_first_round 1346, superseded 379 | 26,101,156 | 12,947,066 | 0 (max 0 ms) |
| 500 | evicted 495, full 25285, none_first_round 1397, superseded 267 | 15,717,822 | 40,817,277 | 0 (max 0 ms) |

## Sessions smoke

Measured / target over the same window: from the first submit to the last first token of the smoke, the target run (measured/sessions_sim/, bench/sim_rounds.py: the frozen simulator run of data/session_targets.json) cut there by data/metrics.py's rule (rounds whose first token comes inside it).

| Run | N | Window (s) | Rounds | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 (s) |
|---|---|---|---|---|---|---|---|
| prefix_smoke_n400_7508 | 400 | 146 | 1954 / 2063 | 13.35 / 14.09 | 91.1k / 93.4k | 99.71% / 99.73% | 0.58 / 2.64 / 7.92 vs 0.38 / 1.23 / 2.05 (x1.51 / x2.15 / x3.87) |
| └ replay (our engine, fit times; plain, dispatch_ahead 1, form lead 35 ms) | | | 2067 | 14.12 | 93.5k | 99.73% | 0.39 / 1.22 / 2.08 (x1.02 / x0.99 / x1.01 of target) |
| prefix_smoke_n400_7569 | 400 | 149 | 2038 / 2090 | 13.64 / 13.98 | 93.7k / 94.5k | 99.72% / 99.74% | 0.46 / 1.83 / 3.32 vs 0.39 / 1.28 / 2.14 (x1.17 / x1.43 / x1.55) |
| └ replay (our engine, fit times; plain, dispatch_ahead 1, form lead 35 ms) | | | 2090 | 13.98 | 94.5k | 99.74% | 0.40 / 1.26 / 2.14 (x1.02 / x0.99 / x1.00 of target) |
| prefix_smoke_n400_7658 | 400 | 151 | 2040 / 2139 | 13.50 / 14.16 | 93.9k / 95.9k | 99.72% / 99.74% | 0.48 / 2.19 / 4.27 vs 0.40 / 1.36 / 3.57 (x1.19 / x1.62 / x1.20) |
| └ replay (our engine, fit times; plain, dispatch_ahead 1, form lead 35 ms) | | | 2145 | 14.20 | 96.0k | 99.74% | 0.41 / 1.40 / 3.61 (x1.02 / x1.03 / x1.01 of target) |
| prefix_smoke_n400_7683 | 400 | 149 | 2048 / 2090 | 13.74 / 14.02 | 94.1k / 94.7k | 99.73% / 99.74% | 0.45 / 1.73 / 2.82 vs 0.39 / 1.28 / 2.14 (x1.15 / x1.35 / x1.32) |
| └ replay (our engine, fit times; plain, dispatch_ahead 1, form lead 45 ms) | | | 2100 | 14.09 | 94.9k | 99.74% | 0.40 / 1.22 / 2.22 (x1.03 / x0.95 / x1.03 of target) |
| prefix_smoke_n400_7693 | 400 | 151 | 2065 / 2129 | 13.70 / 14.12 | 94.5k / 95.8k | 99.73% / 99.74% | 0.45 / 1.72 / 3.42 vs 0.40 / 1.34 / 3.57 (x1.12 / x1.28 / x0.96) |
| └ replay (our engine, fit times; plain, dispatch_ahead 1, form lead 45 ms) | | | 2138 | 14.18 | 96.1k | 99.74% | 0.41 / 1.33 / 3.43 (x1.03 / x0.99 / x0.96 of target) |

Kernels compiled after warm-up (worker jit_stage<S>.jsonl; any fails the run): prefix_smoke_n400_7508 not monitored; prefix_smoke_n400_7569 not monitored; prefix_smoke_n400_7658 44 (by stage {0: 11, 1: 11, 2: 11, 3: 11}); prefix_smoke_n400_7683 4 (by stage {0: 1, 1: 1, 2: 1, 3: 1}); prefix_smoke_n400_7693 0.

Prefix cache per round (server_requests.jsonl hit_kind): rounds by kind, tokens recomputed because the kept context was superseded or evicted, waits for pool room:

| Run | Rounds by kind | Superseded: recomputed tokens | Evicted: recomputed tokens | Admit waits |
|---|---|---|---|---|
| prefix_smoke_n400_7658 | full 1792, none_first_round 307, superseded 5 | 188,750 | 0 | 0 (max 0 ms) |
| prefix_smoke_n400_7683 | full 1813, none_first_round 302, superseded 5 | 188,750 | 0 | 0 (max 0 ms) |
| prefix_smoke_n400_7693 | full 1824, none_first_round 306, superseded 5 | 188,750 | 0 | 0 (max 0 ms) |

Stage 0, plan formed -> its forward returned with all GPU work enqueued (ms, p50 / p90 / p99); of the plans formed while another ran, how many were enqueued after that one's stage-0 end (late, by how much):

| Run | Formed → enqueued | Formed → received | Forward host | Queued plans late | Late by |
|---|---|---|---|---|---|
| prefix_smoke_n400_7658 | 7.2 / 10.1 / 41.5 | 0.8 / 1.0 / 2.4 | 6.5 / 9.1 / 39.0 | 135 / 1830 | 5.3 / 15.8 / 475.2 |
| prefix_smoke_n400_7683 | 7.4 / 10.1 / 38.6 | 0.7 / 0.9 / 3.5 | 6.7 / 9.1 / 37.8 | 229 / 1795 | 4.9 / 9.9 / 24.1 |
| prefix_smoke_n400_7693 | 7.3 / 10.2 / 37.4 | 0.7 / 0.9 / 4.2 | 6.6 / 9.1 / 36.5 | 270 / 1833 | 5.2 / 10.7 / 26.7 |

Stage time / fit (microbatch_fit.json) by microbatch size in tokens, summed over the run's microbatches (n); stage 0 idle while a round was already waiting (gaps), and from a round's arrival on an idle stage 0 to its start (arrivals):

| Run | Stage | [0, 256) | [256, 1024) | [1024, 8192) | 8192 | Stage 0 idle, waiting | Stage 0 idle, arrival to start |
|---|---|---|---|---|---|---|---|
| prefix_smoke_n400_7508 | 0 | x1.703 (59; +0.3 s) | x1.538 (138; +0.7 s) | x1.291 (336; +2.9 s) | x1.065 (1577; +5.9 s) | 13.84 s (1722 gaps) | 2.37 s (292 arrivals) |
| prefix_smoke_n400_7508 | 1 | x1.308 (59; +0.1 s) | x1.313 (138; +0.5 s) | x1.243 (336; +2.3 s) | x1.091 (1577; +7.8 s) |  |  |
| prefix_smoke_n400_7508 | 2 | x1.364 (59; +0.2 s) | x1.387 (138; +0.6 s) | x1.279 (336; +2.8 s) | x1.126 (1577; +11.1 s) |  |  |
| prefix_smoke_n400_7508 | 3 | x1.350 (59; +0.2 s) | x1.376 (138; +0.6 s) | x1.282 (336; +2.8 s) | x1.121 (1577; +10.7 s) |  |  |
| prefix_smoke_n400_7569 | 0 | x1.380 (72; +0.2 s) | x1.302 (184; +0.5 s) | x1.193 (408; +2.3 s) | x1.026 (1605; +2.4 s) | 2.77 s (1714 gaps) | 3.37 s (403 arrivals) |
| prefix_smoke_n400_7569 | 1 | x1.009 (72; +0.0 s) | x1.117 (184; +0.2 s) | x1.189 (408; +2.2 s) | x1.024 (1605; +2.1 s) |  |  |
| prefix_smoke_n400_7569 | 2 | x0.974 (72; -0.0 s) | x1.144 (184; +0.3 s) | x1.221 (408; +2.7 s) | x1.061 (1605; +5.4 s) |  |  |
| prefix_smoke_n400_7569 | 3 | x0.990 (72; -0.0 s) | x1.138 (184; +0.3 s) | x1.219 (408; +2.7 s) | x1.063 (1605; +5.7 s) |  |  |
| prefix_smoke_n400_7658 | 0 | x2.371 (77; +0.8 s) | x1.266 (179; +0.5 s) | x1.108 (431; +1.4 s) | x1.031 (1610; +2.9 s) | 3.80 s (1741 gaps) | 4.93 s (432 arrivals) |
| prefix_smoke_n400_7658 | 1 | x1.701 (77; +0.4 s) | x1.094 (179; +0.2 s) | x1.094 (431; +1.2 s) | x1.021 (1610; +1.8 s) |  |  |
| prefix_smoke_n400_7658 | 2 | x1.703 (77; +0.5 s) | x1.105 (179; +0.2 s) | x1.119 (431; +1.5 s) | x1.060 (1610; +5.4 s) |  |  |
| prefix_smoke_n400_7658 | 3 | x1.748 (77; +0.5 s) | x1.113 (179; +0.2 s) | x1.122 (431; +1.6 s) | x1.048 (1610; +4.3 s) |  |  |
| prefix_smoke_n400_7683 | 0 | x1.354 (73; +0.2 s) | x1.254 (198; +0.5 s) | x1.032 (448; +0.4 s) | x1.008 (1586; +0.7 s) | 3.47 s (1706 gaps) | 3.58 s (453 arrivals) |
| prefix_smoke_n400_7683 | 1 | x0.921 (73; -0.0 s) | x1.088 (198; +0.2 s) | x1.038 (448; +0.5 s) | x1.018 (1586; +1.6 s) |  |  |
| prefix_smoke_n400_7683 | 2 | x0.918 (73; -0.1 s) | x1.101 (198; +0.2 s) | x1.069 (448; +0.9 s) | x1.062 (1586; +5.4 s) |  |  |
| prefix_smoke_n400_7683 | 3 | x0.993 (73; -0.0 s) | x1.103 (198; +0.2 s) | x1.059 (448; +0.8 s) | x1.037 (1586; +3.3 s) |  |  |
| prefix_smoke_n400_7693 | 0 | x1.327 (76; +0.2 s) | x1.236 (201; +0.5 s) | x1.028 (470; +0.4 s) | x1.010 (1594; +0.9 s) | 3.51 s (1744 gaps) | 3.81 s (453 arrivals) |
| prefix_smoke_n400_7693 | 1 | x0.929 (76; -0.0 s) | x1.087 (201; +0.2 s) | x1.035 (470; +0.5 s) | x1.016 (1594; +1.4 s) |  |  |
| prefix_smoke_n400_7693 | 2 | x0.931 (76; -0.0 s) | x1.101 (201; +0.2 s) | x1.065 (470; +0.9 s) | x1.065 (1594; +5.8 s) |  |  |
| prefix_smoke_n400_7693 | 3 | x1.004 (76; +0.0 s) | x1.100 (201; +0.2 s) | x1.056 (470; +0.8 s) | x1.038 (1594; +3.4 s) |  |  |

## Notes

Runs (measured/runs_m4/, raw, untracked): `prefix_check_on/off` (job 7403), `prefix_bringup` (7404, Nsight window of
microbatches 564–631: the last round-0 chunk and round 1 of session 34, 512k + 1k, and session 35, 512k + 8k).

**Bring-up (7404).** Full schedule (sessions 30 s apart); cayenne's idle-GPU watchdog flagged the job but did not stop
it, so no session was rerun on a compressed schedule. `intake_procs` 16. Uptime load 10–14 before and after. Hits
40/40 equal P; round-1 tokens 40/40 equal the no-cache token (7403, req-frontend's prompts rebuilt by
bench/prefix_check.py and checked by sha256).

Pass (§6, ±10%; unrounded ratios): 8k + 1k, 8k + 8k, 32k + 8k and 128k + 8k on every column; 32k + 1k except round-1
TTFT (x1.11; stage 0 x1.099). Misses: 32k + 1k and 128k + 1k round-1 TTFT, 128k + 1k stage 0 (x1.14), and every 512k
column except stage 1 at 1k (x1.07). Causes:
- Round 1 TTFT at 1k appends grows with P (x1.07 → 1.11 → 1.24 → 1.75 for 8k → 512k) while stages 1–3 stay at
  x0.94–0.98 up to 128k: the gap is host time per request, which grows with the prompt (receiving and parsing its
  ids; 512k: ~27 ms from arrival to first dispatch), and stage 0 at x1.10–1.14.
- 512k: every stage x1.07–1.33 and round 0 x1.16; the targets put the 512k excess in the indexer.
- Both are being measured in lane longctx (host parse, a trace of one 512k session, the stage-0 1k excess, the two
  first-use outliers: session 0's first hit at 8k + 1k, 37 ms on stage 0; session 25, the first 128k + 8k, 143.5 ms
  on stage 0). Session 34's round 0 (28.4 s) is the Nsight capture starting at microbatch 564: a 23.3 s gap before
  it on stage 0.
- Sends: 0.21 ms (1k) and 0.67 ms (8k) against 0.068 / 0.482: the copy-engine transport's known gap (issues.md).

**Unaligned P.** The trace's bring-up P are multiples of 4; the sessions' are not (N=400/500, 7405/7406, crashed on the
first such hit). The indexer fix and its checks are in issues.md ("kpool indexer assumed pool-aligned chunk starts").

**Preflight before the session runs** (all passed):
- `tests/test_kpool_gpu.py` (7481, again 7601 after the sync fix): the fork's kpool ops write the unsplit prefill's
  index cache bytes for hits at P = 8193 + 1/2/3 and for mixed chunkings, bitwise; CPU tests 135 passed.
- kvcheck (7476, 7569): 8,192-token chunks bitwise equal to whole at 32,768 tokens; 8,001- and 8,190-token chunks at
  40,000 (starts inside indexer pools and off KDA's 32-token chunk grid) within 2x the split-K noise baseline per stage
  (0.128 / 0.334 / 0.402 against limits 0.235 / 0.407 / 0.458), KL 8e-6 / 1.5e-5, top-1 equal.
- Unaligned prefix check, first (7476 cache on, 7498 cache off): hit = P 40/40; 3 token differences, all near ties
  (margins 0.375–1.875 nats, 7503; issues.md "Unaligned prefix check").
- Unaligned prefix check with the decode top-k fix (7569, both halves in one job, new req-frontend): hit = P 40/40,
  same round-1 prompt 40/40, round-1 token equal 38/40; the 2 differences (s27 and s32 round 1) are near ties by
  `bench/margin.py --judge` (top 2 the same pair on both servers, margins 0–0.375 nats). Round 0 now matches in all 40.
- Smoke (7508): prefix_n400's first 2.4 min of client time on a fresh server: 1,954 rounds, failed 0, 1,699 hits,
  0 evictions, 0 preemptions; uptime load 35–42.
- Smoke (7569, new req-frontend, decode top-k fix and the unaligned-P sync fix): 2,038 rounds, failed 0, 1,799 hits,
  0 evictions, 0 preemptions; TTFT p50 / p90 / p99 0.46 / 1.83 / 3.32 s against the same-window target 0.37 / 1.17 /
  1.99 (table above); uptime load 65 before, 10 after.

**Where the smoke's TTFT gap goes (7569 against v1's frozen sim run, `bench/ttft_attrib.py`).** All 2,108 rounds the
server answered, matched to the sim run by request id (trace row); each round's server TTFT split along its critical
path. Mean TTFT is +309 ms over the sim (round 0, 304 rounds: 1,034 vs 793 ms; resumed, 1,804: 846 vs 525). By part
(real − sim, round 0 / resumed, ms):

| Part | Round 0 | Resumed |
|---|---|---|
| Arrival → first chunk on stage 0, stage 0 busy with other microbatches (work ahead) | +197 | +267 |
| same, stage 0 idle (admission and dispatch latency) | +12 | +11 |
| Own chunks on stage 0 (and gaps between them) | +6 | +8 |
| Stages 1–3: own microbatch (and own earlier chunks) | +22 | +19 |
| Stages 1–3: behind other microbatches | 0 | +12 |
| Last stage → response | +5 | +4 |

The work ahead is a longer backlog, not slower microbatches in it: 9.3–9.5 microbatches ahead against the sim's
5.2–6.5 (~62 ms each in both). Same chunking (3.98 / 3.99 chunks per round 0), same FIFO order and 8,192-token
microbatches as the sim. A replay of the real stage-0 order as a work-conserving server gives the arrival → stage 0
wait (mean, p50 / p90 / p99 ms): real 600 (173 / 1,740 / 4,125); replay at the real compute 511 (139 / 1,488 / 3,603);
replay at the sim's per-microbatch compute (x0.941) 366 (94 / 1,094 / 2,761); sim 333 (91 / 920 / 2,876). So of the
+268 ms mean wait: ~145 ms is per-microbatch compute, amplified by queueing (stage 0 busy 73% vs 70%); ~90 ms is
stage 0 idle while rounds wait (6.1 s of 154 s vs the sim's 2.8 s: 2.3 s at the in-flight cap of 4 behind slower
stages 2–3, 0.5 s dispatched late, the rest the ~12 ms from arrival to start when stage 0 is idle); ~33 ms is left
for packing and the closed loop's arrivals.

Per-microbatch compute against the sim, by tokens (stage 0 / 3): full 8,192-token microbatches x1.024 / x1.061;
partial ones x1.09–1.35 / x1.10–1.38 (4,097–6,144 tokens: 51.3 vs 38.0 ms). The partial ones are padded to the
power-of-2 CUDA graph buckets of that tree (4,097 → 8,192); lane longctx's 1,024-step buckets (merged, 306107f)
target exactly that. Full microbatches on stages 2–3 (+6%) make stage 3 the slowest stage (59.6 ms vs stage 0's
58.9), where the sim's slowest is stage 0: that is the in-flight-cap idle on stage 0 and the waits behind other
microbatches on stages 2–3. Scheduling order is not the gap (same policy, same chunk shapes); M5's scheduler work
does not explain it.

**Smoke 7658 on the merged tree (longctx da56287, 33 graph buckets: 128 and every 256 to 8,192).** 2,040 rounds,
failed 0; full 1,792, none_first_round 307, superseded 5 (188,750 tokens recomputed), evicted 0, no admit waits.
TTFT p50 / p90 / p99 0.48 / 2.19 / 4.27 s against the same-window v2 target 0.40 / 1.36 / 3.57 (x1.19 / x1.62 /
x1.20); our engine's replay at v2's fit 0.41 / 1.40 / 3.61 (x1.02 / x1.03 / x1.01 of target), so scheduling explains
at most x1.03 and the rest is stage time over the fit, 11 Triton compiles per stage after warm-up among it (5
pipeline stalls of 0.3–3.3 s; fixed in warm-up, issues.md). Against v1 (whose hit rule kept partial hits) the target
was 0.39 / 1.28 / 3.55 and the replay x1.10 / x1.08 / x1.01. Uptime load 4.7 before, 11 after. By size, in v1's
fit and size classes: the 256-step buckets brought 1,024–4,095-token microbatches from x1.18–1.21 (7569) to x1.04–1.07
and 4,096–8,191 from x1.19–1.23 to x1.14–1.16; below 1,024 tokens x1.24–1.53 (+0.6–1.2 s per stage: host-bound small
microbatches, lane longctx's stage-0 host work, and the compile stalls).

**Full 8,192-token microbatches over the fit: the power cap, not the batch.** nvidia-smi every 200 ms (SAMPLE=1,
`slurm/gpu_samples_7658.csv`) joined to each microbatch's GPU interval: the reason is sw_power_cap in 92–94% of the
samples inside full microbatches; SM clock falls over the run from ~1,700 to ~1,570 MHz as power rises from ~720 to
~975 W (limit 1,000 W), and the ratio rises with it (stage 2 by quartile x1.030 / 1.052 / 1.108 / 1.082, stage 3
x1.023 / 1.041 / 1.102 / 1.066). At contexts below 32k, by SM clock (stages 0 / 1 / 2 / 3): below 1,700 MHz x1.035 /
1.036 / 1.069 / 1.055; at 1,900 MHz or more x0.967 / 0.950 / 0.981 / 0.956; ratio against 1965 / SM clock r = 0.70 /
0.75 / 0.77 / 0.78. Temperature does not order it (stage 1 runs hottest, 68 C, with the lowest ratio). At the same
clock stages 2/3 stay ~2–3% above stages 0/1, as perf lane's placement finding (main bdd697e). Requests per
microbatch had no effect (7569). Not a lane-m4 fix: the fit was measured at full clock; under sustained load the
GPUs run at the 1,000 W cap.

**form_lead_ms (7658, worker_host.jsonl from longctx's 1934aff).** Stage 0, plan formed -> forward returned with all
GPU work enqueued: p50 / p90 / p99 7.2 / 10.1 / 41.5 ms (formed -> received 0.8 / 1.0 / 2.4; forward host 6.5 / 9.1 /
39.0). The p99 is 8,192-token microbatches at 56k–134k contexts (101 microbatches over 20 ms of forward host time).
135 of 1,830 queued plans were enqueued after the previous stage-0 end: 69% formed less than 20 ms before that end
(the plan formed late, not slow to enqueue), the rest slow forwards (4 over 100 ms). REAL_FORM_LEAD_MS 35 -> 45
(14d505b). The long-context forward host time is lane longctx's.

**Sessions N=400 and N=500 (the table under Sessions): N=500 passes; N=400 passes on throughput, and its TTFT p50 /
p90 are outside ±10% because of the GPU power cap, which the user decided to record, not chase.** Full 20-minute runs,
plain scheduler, HBM prefix cache, sessions_2ps.csv, against handoff v2 (session_targets.json) and the same simulator
at `gpu_time_multiplier` 1.06 (measured/sessions_sim/mult106.json):
- **N=400** (prefix_n400_7683, 840c1ab; load average 14.2 before, 10.6 after): rounds/s x0.97, new
  tokens/s x0.96, computed x0.96, hit share 98.28% = target, 0 failed. TTFT 0.64 / 4.43 / 7.57 s vs 0.55 / 3.66 /
  5.98 (x1.15 / x1.21 / x1.27): p50 and p90 are outside ±10%, p99 within ±25%. Against the simulator at x1.06
  (0.75 / 4.25 / 8.22 s, 86.3k new tokens/s): x0.85 / x1.04 / x0.92, throughput x0.98.
- **N=500** (prefix_n500_7693, 1fa8068; load average 6.5 before, 6.9 after): rounds/s x0.95, new x0.96, computed x0.97, hit
  96.85% vs 97.01%, 0 failed. TTFT 3.22 / 14.26 / 26.77 s vs 3.01 / 13.03 / 23.93 (x1.07 / x1.09 / x1.12): pass.
  At x1.06: 3.85 / 14.37 / 24.25 s (measured x0.84 / x0.99 / x1.10).
- **The cause is the power cap.** Under this load the four GPUs sit at the 1,000 W cap in 91–96% of busy samples, at
  1,577–1,678 MHz SM clock (table above), and stage time runs over the fit by x1.03 / 1.03 / 1.09 / 1.06 (median per
  stage) on both runs; the same simulator with GPU time x1.06 reproduces N=400's throughput and TTFT p90 within 2–4%
  (issues.md, "Under load the GPUs run at the 1,000 W power cap"). Our scheduler and cache replay to the target on the
  fit (Sessions smoke: replay x1.03 / x0.95 / x1.03), so the rules are not the gap.
- **In-run JIT:** 4 in each run (one BuildPrefillChunkMetadataKernel key per stage, ~70 ms, a request-level indexer
  chunk at an unaligned slice). Fixed by compiling every compile key vLLM's kernels declare at warm-up (163356b,
  cb61603); smoke prefix_smoke_n400_7693 on cb61603 has 0. A full-length run with 0 is M5's first session run.
