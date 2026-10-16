# Prefix cache and chunked prefill (5_prefix_cache.md)

## Prefix bring-up

Measured / target (ratio), medians (p50) over each case's sessions, TTFT also p90 and p99 over them; stage compute and sends from CUDA events (microbatches.jsonl).

| P | A | n | hit = P | Round 0 TTFT p50 (ms) | Round 1 TTFT p50 (ms) | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Each send | Token = no cache |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8k | 1k | 5 | 5/5 | 210.0 / 223.6 (x0.94); p90 210.4, p99 210.5 | 57.2 / 53.3 (x1.07); p90 84.8, p99 100.9 | 13.9 / 12.7 (x1.10) | 12.4 / 13.3 (x0.94) | 12.8 / 13.5 (x0.95) | 13.1 / 13.7 (x0.95) | 0.206 / 0.068 | 5/5 |
| 8k | 8k | 5 | 5/5 | 209.7 / 223.6 (x0.94); p90 210.2, p99 210.3 | 215.7 / 223.8 (x0.96); p90 216.9, p99 217.1 | 53.5 / 57.1 (x0.94) | 50.6 / 54.2 (x0.93) | 52.2 / 55.3 (x0.94) | 52.5 / 55.7 (x0.94) | 0.675 / 0.482 | 5/5 |
| 32k | 1k | 5 | 5/5 | 401.2 / 396.8 (x1.01); p90 401.9, p99 402.0 | 59.8 / 53.8 (x1.11); p90 60.0, p99 60.1 | 14.1 / 12.8 (x1.10) | 12.5 / 13.4 (x0.94) | 13.0 / 13.6 (x0.96) | 13.3 / 13.8 (x0.97) | 0.208 / 0.068 | 5/5 |
| 32k | 8k | 5 | 5/5 | 401.1 / 396.8 (x1.01); p90 402.2, p99 402.4 | 222.5 / 226.4 (x0.98); p90 222.8, p99 222.9 | 54.9 / 57.8 (x0.95) | 51.2 / 54.6 (x0.94) | 53.2 / 56.0 (x0.95) | 53.7 / 56.4 (x0.95) | 0.671 / 0.482 | 5/5 |
| 128k | 1k | 5 | 5/5 | 1185.5 / 1118.2 (x1.06); p90 1267.3, p99 1314.8 | 68.4 / 55.4 (x1.24); p90 68.9, p99 69.1 | 15.1 / 13.3 (x1.14) | 13.2 / 13.7 (x0.96) | 13.8 / 14.0 (x0.98) | 14.0 / 14.3 (x0.98) | 0.210 / 0.068 | 5/5 |
| 128k | 8k | 5 | 5/5 | 1180.2 / 1118.2 (x1.06); p90 1186.0, p99 1186.6 | 251.9 / 239.0 (x1.05); p90 303.1, p99 333.5 | 61.6 / 61.3 (x1.01) | 55.6 / 56.9 (x0.98) | 59.8 / 59.5 (x1.00) | 60.0 / 59.8 (x1.00) | 0.660 / 0.482 | 5/5 |
| 512k | 1k | 5 | 5/5 | 5110.0 / 4408.1 (x1.16); p90 19109.4, p99 27486.3 | 112.8 / 64.3 (x1.75); p90 116.0, p99 117.4 | 20.9 / 15.7 (x1.33) | 16.3 / 15.3 (x1.07) | 18.4 / 16.5 (x1.12) | 18.8 / 16.7 (x1.12) | 0.208 / 0.068 | 5/5 |
| 512k | 8k | 5 | 5/5 | 5117.2 / 4408.1 (x1.16); p90 5128.4, p99 5133.2 | 386.3 / 287.7 (x1.34); p90 389.2, p99 390.5 | 91.3 / 74.6 (x1.22) | 73.9 / 65.8 (x1.12) | 87.8 / 72.8 (x1.21) | 88.4 / 73.1 (x1.21) | 0.670 / 0.482 | 5/5 |

## Cache on vs off

40 round-1 prompts, 40 with a hit; 40 tokens equal to the no-cache token; all rounds 80/80.

## Cache on vs off, unaligned P

P + 1,2,3 (cycled) on the bring-up sessions: 40 sessions, hit = P 40, round-1 token equal to no-cache 38, same prompt 40 (bench/prefix_check.py --p-shift).

## Sessions smoke

Measured / target over the same window: from the first submit to the last first token of the smoke, the target run (measured/sessions_sim/, bench/sim_rounds.py: the frozen simulator run of data/session_targets.json) cut there by data/metrics.py's rule (rounds whose first token comes inside it).

| Run | N | Window (s) | Rounds | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 (s) |
|---|---|---|---|---|---|---|---|
| prefix_smoke_n400_7508 | 400 | 146 | 1954 / 2082 | 13.35 / 14.22 | 91.1k / 94.8k | 99.71% / 100.00% | 0.58 / 2.64 / 7.92 vs 0.37 / 1.14 / 1.86 (x1.59 / x2.33 / x4.25) |
| prefix_smoke_n400_7569 | 400 | 149 | 2038 / 2105 | 13.64 / 14.08 | 93.7k / 94.7k | 99.72% / 100.00% | 0.46 / 1.83 / 3.32 vs 0.37 / 1.17 / 1.99 (x1.23 / x1.57 / x1.67) |

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

**Where the smoke's TTFT gap goes (7569 against the frozen sim, `bench/ttft_attrib.py`).** All 2,108 rounds the
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

**Sessions N=400 and N=500: not run yet.** On hold until three fixes are in lane-m4: lane longctx's graph buckets
(the 3,000-token padding) and its stage-0 host launch gap, and the unaligned-P per-DSA-layer sync (done, 9d73048).
Then one smoke with the new req-frontend and two parallel 4-GPU jobs
(`m4.sbatch prefix_n400`, `prefix_n500`), each failing fast on a server or worker traceback or client failed > 0.
