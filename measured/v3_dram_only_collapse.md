# DRAM-only collapse on the v3 trace (sim scan + 2 real runs)

v3 rerun of the old-trace DRAM-only study (TIERS.md §9 step 1b, jobs 7969/7970). Workload: closed loop of N sessions on
`data/tracelab-v3/m6/sessions_2ps_v3.csv`, 20 min. Design under test: DRAM written by HBM evictions only
(`hbm-evictions-exclusive`), 10% loading areas (st10), **no SSD**, force-schedule at 60 s, scheduler `good`.

## Configs

| | HBM / GPU | HBM pool (tokens) | DRAM / GPU | DRAM (tokens) | C = HBM + DRAM | N |
|---|---|---|---|---|---|---|
| collapsing: h25d50 | 25 GB | 15,273,856 | 50 GB | 33.36M (3,888 slots) | 48.6M | 400 |
| near-edge stable: h25d75 | 25 GB | 15,273,856 | 75 GB | 50.03M (5,833 slots) | 65.3M | 400 |

The two configs differ only in DRAM size. The real `--pool-tokens` is the same as the sim's HBM `capacity_tokens`
(`raw/run_meta.json`). Real DRAM slots x 8,576 tokens per block gives the sim's DRAM capacity.

## W against C

W(t) is the retained context the closed loop needs at time t. It counts every session whose latest served round is
not its last in the trace and whose next round has not been served yet. Each such session holds that round's prompt
plus 24,600 token-equivalents for the KDA end state. Sessions whose next round has arrived and is still waiting are
included. W is computed per 2-minute window end by `windows.py`.

- **v3 needs more retained context than the old trace, and it grows faster.** At N400, W reaches 33M by minute 4
  (old trace: 19M) and 52-65M by minutes 8-20 (old trace: 31-48M).
- **Every scanned N400 config with C ≤ 57M collapses within 20 min.** Collapse comes 1-3 windows (2-6 min) after W
  first exceeds C. Each step of capacity delays it: C = 41M collapses at min 6-8, 48.6M at 12-14, 57M at 18-20.
- **Exceeding C does not always mean collapse within 20 min.** h20d75 (C = 62.3M) has W > C by up to 2.6M from minute
  14 and holds. h25d37 at N300 (C = 40.3M) holds with W up to 47M at a gpu_time_multiplier of 1.0, but collapses at
  min 16-18 at 1.03. Close to the edge, small changes in speed decide whether it tips.
- **h25d75 is the edge:** max W 64.6M vs C 65.3M. It stays stable at gpu_time_multiplier 1.0 and 1.03.

Simulator scan, 22 runs (`first W > C` and `hit < 0.5` are 2-minute window indices):

| run | C (M tok) | max W (M) | first window W > C | first window hit < 0.5 | rounds, min 18-20 |
|---|---|---|---|---|---|
| h12d12 c400 | 16.0 | 42.5 | 1 (min 2-4) | 2 (min 4-6) | 203 |
| h12d19 c400 | 20.1 | 43.1 | 1 (min 2-4) | 2 (min 4-6) | 204 |
| h12d37 c400 | 32.6 | 44.5 | 1 (min 2-4) | 3 (min 6-8) | 197 |
| h12d50 c400 | 41.0 | 47.3 | 2 (min 4-6) | 3 (min 6-8) | 185 |
| h20d20 c400 | 25.6 | 44.9 | 1 (min 2-4) | 3 (min 6-8) | 201 |
| h20d37 c400 | 37.2 | 44.5 | 2 (min 4-6) | 3 (min 6-8) | 182 |
| h20d50 c400 | 45.6 | 52.0 | 3 (min 6-8) | 5 (min 10-12) | 180 |
| h20d50 c400 m1.03 | 45.6 | 52.1 | 3 (min 6-8) | 5 (min 10-12) | 149 |
| h20d62 c400 | 53.9 | 58.2 | 5 (min 10-12) | 7 (min 14-16) | 143 |
| h20d75 c400 | 62.3 | 64.9 | 7 (min 14-16) | - | 3748 |
| h25d37 c400 | 40.3 | 45.7 | 2 (min 4-6) | 3 (min 6-8) | 182 |
| h25d50 c400 | 48.6 | 54.4 | 3 (min 6-8) | 6 (min 12-14) | 153 |
| h25d50 c400 m1.03 | 48.6 | 54.2 | 3 (min 6-8) | 5 (min 10-12) | 158 |
| h25d62 c400 | 57.0 | 61.8 | 6 (min 12-14) | 9 (min 18-20) | 146 |
| h25d62 c400 m1.03 | 57.0 | 60.9 | 6 (min 12-14) | 8 (min 16-18) | 142 |
| h25d75 c400 | 65.3 | 64.6 | - | - | 3143 |
| h25d75 c400 m1.03 | 65.3 | 64.8 | - | - | 3784 |
| h12d19 c300 | 20.1 | 34.1 | 1 (min 2-4) | 2 (min 4-6) | 193 |
| h20d20 c300 | 25.6 | 37.4 | 1 (min 2-4) | 4 (min 8-10) | 168 |
| h20d37 c300 | 37.2 | 46.2 | 3 (min 6-8) | 9 (min 18-20) | 137 |
| h25d37 c300 | 40.3 | 46.9 | 3 (min 6-8) | - | 1660 |
| h25d37 c300 m1.03 | 40.3 | 47.1 | 3 (min 6-8) | 8 (min 16-18) | 118 |

The old settings (h25d37, h12d19) and the eval sizes (tiers 20/20, ssdheavy 12.5/12.5), all without SSD, collapse at
N400 by minute 4-8 on v3. The old trace collapsed at min 14-16 and 10-12 under the old code.

## Real vs sim, per 2-minute window

The real runs used the eval tree at c5d4e09: 128-token HBM pages and hostgate. Job 8491 (h25d50) ran on GPU set A
(bus 1B/43/52/61, the hot set). Job 8492 (h25d75) ran on set B (bus 9D/C3/D1/DF). t = 0 is the first session
round. Rounds are counted by first token.

Column meanings:

- **rounds**: rounds served in the window.
- **hit**: share of rounds with a previous context that found it in HBM or DRAM.
- **D**: rounds served from a DRAM read.
- **lost**: rounds that recomputed a context they had.
- **rec**: recomputed tokens (computed minus new), in M.
- **HBM ev**: HBM evictions. Sim counts prefix_cache_event evictions; real counts `evicted_sessions_at_admit`.
- **W**: retained context at the window's end, in M.

Each cell is sim / real.

**h25d50, N400 (collapses)**

| min | rounds | hit | D | lost | rec | HBM ev | W |
|---|---|---|---|---|---|---|---|
| 0-2 | 2532 / 2395 | 1.00 / 1.00 | 32 / 31 | 0 / 0 | 0.0 / 0.0 | 125 / 165 | 15.9 / 15.3 |
| 2-4 | 2905 / 2860 | 1.00 / 1.00 | 1107 / 1154 | 0 / 0 | 0.0 / 0.0 | 1320 / 1144 | 33.1 / 32.8 |
| 4-6 | 3452 / 3023 | 1.00 / 0.99 | 2553 / 2687 | 12 / 15 | 1.2 / 1.5 | 2735 / 1768 | 42.8 / 42.7 |
| 6-8 | 3610 / 2941 | 0.99 / 0.98 | 2823 / 2698 | 43 / 50 | 4.1 / 4.7 | 2946 / 1794 | 49.5 / 47.6 |
| 8-10 | 2415 / 2139 | 0.96 / 0.95 | 1964 / 1819 | 102 / 103 | 10.1 / 9.4 | 2097 / 1413 | 52.9 / 50.6 |
| **10-12** | **496 / 476** | **0.72 / 0.72** | 322 / 327 | 131 / 130 | 12.5 / 12.9 | 531 / 240 | 54.1 / 51.7 |
| 12-14 | 144 / 402 | 0.00 / 0.60 | 0 / 236 | 144 / 160 | 17.2 / 16.6 | 144 / 198 | 54.3 / 53.0 |
| 14-16 | 162 / 483 | 0.00 / 0.68 | 0 / 326 | 160 / 150 | 16.5 / 16.6 | 162 / 178 | 54.2 / 53.0 |
| 16-18 | 143 / 454 | 0.00 / 0.67 | 0 / 301 | 140 / 145 | 16.8 / 16.1 | 143 / 171 | 54.3 / 54.5 |
| 18-20 | 153 / 427 | 0.00 / 0.68 | 0 / 285 | 149 / 137 | 16.6 / 16.2 | 153 / 150 | 54.4 / 55.6 |

eval_table, 0-20 / 10-20 min:

| run | r/s | recompute/new | TTFT p50/p90/p99 (s) | >60 s |
|---|---|---|---|---|
| sim | 13.34 / 1.83 | 1.25 / 15.9 | 1.34/42.7/284 / 159/286/290 | 1448 / 773 |
| real | 13.00 / 3.74 | 1.24 / 8.47 | 1.98/63.7/161 / 65.8/159/170 | 2948 / 2211 |

**h25d75, N400 (near-edge stable)**

| min | rounds | hit | D | lost | rec | HBM ev | W |
|---|---|---|---|---|---|---|---|
| 0-2 | 2532 / 2407 | 1.00 / 1.00 | 32 / 30 | 0 / 0 | 0.0 / 0.0 | 125 / 159 | 15.9 / 15.2 |
| 2-4 | 2905 / 2762 | 1.00 / 1.00 | 1107 / 1184 | 0 / 0 | 0.0 / 0.0 | 1320 / 1213 | 33.1 / 32.6 |
| 4-6 | 3716 / 3297 | 1.00 / 1.00 | 2719 / 2932 | 2 / 2 | 0.3 / 0.3 | 2896 / 1881 | 42.5 / 42.8 |
| 6-8 | 4933 / 4027 | 1.00 / 1.00 | 4072 / 3657 | 5 / 6 | 0.3 / 0.5 | 4185 / 2513 | 51.8 / 49.2 |
| 8-10 | 4895 / 4325 | 1.00 / 1.00 | 3636 / 4193 | 14 / 10 | 1.3 / 0.8 | 3695 / 2804 | 56.8 / 53.3 |
| 10-12 | 4219 / 3733 | 0.99 / 0.99 | 3161 / 3147 | 27 / 19 | 3.1 / 2.1 | 3258 / 2407 | 58.7 / 56.2 |
| 12-14 | 3073 / 3176 | 0.98 / 0.99 | 2064 / 2589 | 47 / 37 | 6.7 / 4.9 | 2196 / 1996 | 59.7 / 57.0 |
| 14-16 | 3811 / 2965 | 0.99 / 0.99 | 2909 / 2355 | 30 / 33 | 4.9 / 4.7 | 3010 / 1891 | 63.4 / 61.3 |
| 16-18 | 3842 / 2877 | 0.99 / 0.99 | 3027 / 2205 | 38 / 41 | 5.2 / 5.6 | 3129 / 1806 | 64.0 / 61.0 |
| 18-20 | 3143 / 3062 | 0.99 / 0.99 | 2288 / 2625 | 45 / 35 | 7.0 / 5.4 | 2387 / 1996 | 64.6 / 64.1 |

eval_table, 10-20 min:

| run | r/s | recompute/new | TTFT p50/p90/p99 (s) | >60 s | hit H/D (lost) % | s0 busy |
|---|---|---|---|---|---|---|
| sim | 30.15 | 0.451 | 1.12/8.86/61.8 | 516 | 23.1/75.8 (1.1) | 0.992 |
| real | 26.36 | 0.424 | 1.84/11.7/62.5 | 775 | 15.6/83.3 (1.1) | 0.855 |

## Collapse timing and mechanism

- **h25d50 collapses in the same window on both sides (min 10-12).** In that window both have 476-496 rounds and hit
  0.72. Before it, hit share, lost rounds, recompute and W match window by window. Real serves 5-19% fewer rounds at
  min 4-10 and evicts fewer sessions from HBM: the HBM ev count is per session at admission, while the sim count is
  per entry.
- **The trigger is the same on both sides.** W passes C (48.6M) at min 6-8. Lost rounds double every window (12 → 43 →
  102 → 131). Each loss is a full recompute of about 110k tokens, which is admitted at the 60-s bound. That raises
  the backlog and the time contexts wait, so more contexts age out.
- **After the collapse the two sides settle differently.**
  - Sim: total collapse. Hits fall to 0. About 150 rounds per 2 min, each recomputing its whole context (16.5M
    tokens per window). TTFT p50 is 159 s.
  - Real: partial collapse. Hit share stays at 0.60-0.68, with about 450 rounds per 2 min and the same ~16.5M
    recomputed tokens. DRAM-hit rounds get TTFT ≈ 64 s; lost rounds get 163 s.
- **Why the settled states differ (observed, code rule not yet pinned down):**
  - Real (tier_ops): for collapse-window rounds that hit DRAM, the read starts 51/62/68 s (p10/p50/p90) after
    arrival, i.e. at the force bound. The read lands and the round is admitted first, ahead of the recomputes.
  - Sim (prefix_tier_events): 569 of 601 rounds arriving after min 12 had their context in DRAM at arrival. None
    still had it at first token. The context was LRU-evicted from DRAM 163/183/189 s after arrival (p10/p50/p90),
    and no read had started.
  - So the sim's staged loading does not start an overdue round's DRAM read while full recomputes are ahead in the
    overdue queue. To check: `FreshQueues::stage_loads` in `simulator/src/worker/admission/fresh_source.rs` and
    `PrefixFetch::stage` / `load_gate` in `prefix_fetch.rs` (wt-pp-staged).
  - This affects only the regime after collapse. The collapse window and everything before it match.
- **h25d75 is stable on both sides.** Hit is 0.98-0.99 in every window and lost rounds are about 1%. W peaks at
  64.1M real and 64.6M sim, just under C = 65.3M.
  - Real is 12.6% below sim on r/s in min 10-20 (26.36 vs 30.15), with stage-0 busy 0.855 vs 0.992. It is limited by
    the read path, not compute: 83% of real rounds are DRAM reads (sim 76%), and real keeps fewer contexts in HBM
    (H 15.6% vs 23.1%).
  - This is the same class of gap as the eval's read-heavy settings (excl tiers -17%). It also ran on GPU set B.

## Conclusion

On v3 at N400 without SSD, a DRAM tier collapses unless C = HBM + DRAM is about as large as the peak retained context.
The peak is about 65M tokens per GPU at N400, which is 25 GB of HBM plus 75 GB of DRAM per GPU. The old h25d37 and
h12d19 settings, and the eval's 20/20 and 12.5/12.5 sizes, all collapse by minute 4-8.

The sim predicts the onset well. For h25d50 the collapse window matches (min 10-12, rounds 496 vs 476, hit 0.72 on
both), and so does the pre-collapse trajectory (hit share, lost rounds, recompute, W). For h25d75, both sides stay
stable, as the sim predicted.

After the collapse the states differ. The sim goes to total collapse (0 hits); real settles at about 65% DRAM hits,
because it starts overdue rounds' DRAM reads at the force bound. In the stable near-edge config, real throughput is
12.6% below the sim, limited by the read path.

## Reproduction

Sim (a ServingStudioSim checkout of branch pp-staged, CPU only; the presets, scripts and outputs are the sim run
`20261018_15_v3_dramonly` in its `logs/`), from that checkout:

```bash
uv run python -m launcher logs/20261018_15_v3_dramonly/preset.yaml --no-gpu --no-plot --parallelism 15   # 15 runs
uv run python -m launcher logs/20261018_15_v3_dramonly/refine.yaml --no-gpu --no-plot --parallelism 7    # 7 runs
```

Real (this repo at c5d4e09, from its root; T is the TraceLab v3 trace `m6/sessions_2ps_v3.csv`, not in this repo;
h25d75 needs `--mem=500G` for 300 GB of pinned DRAM):

```bash
T=<path to sessions_2ps_v3.csv>
sbatch -p priority -J ppdo50 --time=00:30:00 --mem=400G -o measured/runs_m6/slurm/%j.out bench/slurm/m6.sbatch \
  sched_good_n400 --trace $T --pool-tokens 15273856 --dram-gb 50 --ssd-gb none \
  --tier dram_writes=hbm-evictions-exclusive --tier loading_area=0.10 --run-name do_h25d50_n400    # job 8491
sbatch -p priority -J ppdo75 --time=00:30:00 --mem=500G -o measured/runs_m6/slurm/%j.out bench/slurm/m6.sbatch \
  sched_good_n400 --trace $T --pool-tokens 15273856 --dram-gb 75 --ssd-gb none \
  --tier dram_writes=hbm-evictions-exclusive --tier loading_area=0.10 --run-name do_h25d75_n400    # job 8492
```

Comparison, from this repo's root (a Python with numpy and pyarrow):

```bash
PY=python
S=<ServingStudioSim checkout>/logs/20261018_15_v3_dramonly
R=measured/runs_m6
$PY measured/eval_m6/eval_table.py real:h25d50=$R/do_h25d50_n400 sim:h25d50=$S/h25d50/c400 \
  real:h25d75=$R/do_h25d75_n400 sim:h25d75=$S/h25d75/c400
$PY $S/windows.py real:h25d50=$R/do_h25d50_n400 sim:h25d50=$S/h25d50/c400 \
  real:h25d75=$R/do_h25d75_n400 sim:h25d75=$S/h25d75/c400      # per-window rounds/hit/D/lost/rec/ev/W
```
