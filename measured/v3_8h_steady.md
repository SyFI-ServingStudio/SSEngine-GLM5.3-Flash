# 8 h steady state on the v3 traces: SSD vs DRAM only, base vs hg

GLM-5.3-Flash NVFP4, PP4 on 4x B200, HBM -> DRAM -> SSD retention tiers, ServingStudio Sim, TraceLab v3
new-model traces (8 h closed loop). Date of the runs: 2026-10-18. CPU only (`--no-gpu --no-plot`).

## Setup

- **Deployment.** As in the KDA k study (`kda_ckpt_k_recompute_v3.md`).
  - B200, layers 12/11/11/11, `max_batch_tokens` 32768, HBM KV 100.18 GB/GPU.
  - DRAM 150 GB/GPU at 50 GB/s.
  - Scheduling: `force_schedule_after_ms` 60 s, shortest-prefill-first with SRPT, warm start in the slowest tier,
    tier read wait 1 s.
  - Decode runs externally.
- **Tiers.**
  - `DRAM 150 + SSD 8 TB`: SSD 8 TB at 10 GB/s/GPU. The SSD is DRAM's victim tier: `ssd_tier_writes: dram-evictions`.
  - `DRAM 150 only`: no SSD.
- **Mode.**
  - `st10`: exclusive DRAM (`dram_tier_writes: hbm-evictions-exclusive`) plus a 10% HBM loading area
    (`prefix_tier_loading_area: 0.10`). This is the deployed configuration.
  - `base`: inclusive DRAM (`hbm-evictions`) and no loading area.
- **k.** `recurrent_state_checkpoints`, the number of round-end KDA states each context keeps (see the k doc).
- **Rule.**
  - `base`: overdue requests (waiting over 60 s) go first, oldest first.
  - `hg`: `force_schedule_hits_first` plus `force_schedule_hits_first_when_at_risk`.
    - While serving the overdue requests oldest first would evict a queued request's held context from every tier,
      requests whose context is held go first.
    - Otherwise the order is unchanged.
- **Protocols.**
  - `hb`, KV hand-back (the test protocol): the external decode hands the output KV and the round-end state back.
    The traces are `data/tracelab-v3/traces/closed_c{1563,2000,2500}_s0.csv`, the same files as the k study.
  - `nohb`, no hand-back: each round's outputs become fresh prompt tokens of the next round (`fold.py`).
    - v3 has no such files, so they were derived from the same v3 source:
      `session_closed_loop.py --concurrency N --after 20N --seed 0`, then `fold.py`.
    - c2000 folds `closed_c2000_s0.csv`; c1438 draws a new closed loop from `mono_sessions_d80.csv`.

## Definitions

A round counts in the window its first token falls in. Only completed rounds count. t = 0 is the run start.

| Column | Definition |
|---|---|
| r/s 2-4 h, 4-6 h, 6-8 h | Rounds served per second for the 4-GPU system, per 2 h window |
| new k tok/s | New prompt tokens (`fresh_prompt_tokens`) served per second, in the 2-4 h and 6-8 h windows |
| TTFT p50 / p90 / p99 | Over the rounds served in the last 2 h, [6 h, 8 h) |
| hit % | Rounds served in the last 2 h that reused a cached context (`prefix_cache_hit_tokens > 0`), divided by the non-cold rounds served then. A cold round is a session's first round in the run. |
| extra compute % | Over [1 h, 8 h): (sum of computed tokens - sum of new tokens) / sum of new tokens, where computed = declared prefix + new - cached. This is the k doc's recompute / new. |
| r/s 1-8 h | Rounds per second over [1 h, 8 h), the k doc's r/s |

## Results

### hb (KV hand-back)

| tiers | mode | k | load | r/s 2-4 h | 4-6 h | 6-8 h | new k tok/s 2-4 / 6-8 h | TTFT p50 / p90 / p99 s (6-8 h) | hit % | extra compute % | r/s 1-8 h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DRAM 150 + SSD 8 TB | st10 | k1 | c1563 | 60.0 | 56.2 | 51.0 | 125 / 122 | 3.5 / 20.1 / 63 | 99.3 | 13.3 | 55.39 |
| DRAM 150 + SSD 8 TB | st10 | k1 | c2000 | 57.5 | 53.7 | 47.9 | 121 / 112 | 4.0 / 64.9 / 80 | 99.2 | 19.3 | 53.21 |
| DRAM 150 + SSD 8 TB | st10 | k1 | c2500 | 48.4 | 48.5 | 45.4 | 111 / 107 | 8.3 / 75.5 / 96 | 99.2 | 24.8 | 48.14 |
| DRAM 150 + SSD 8 TB | st10 | k2 | c1563 | 63.7 | 58.2 | 53.0 | 134 / 126 | 3.0 / 16.6 / 63 | 99.3 | 7.2 | 58.60 |
| DRAM 150 + SSD 8 TB | st10 | k2 | c2000 | 63.5 | 57.4 | 53.5 | 132 / 126 | 3.1 / 62.8 / 73 | 99.3 | 8.2 | 58.64 |
| DRAM 150 + SSD 8 TB | st10 | k2 | c2500 | 54.8 | 55.0 | 53.4 | 126 / 127 | 4.2 / 67.1 / 81 | 99.3 | 10.3 | 54.94 |
| DRAM 150 + SSD 8 TB | base | k1 | c1563 | 60.2 | 57.4 | 50.4 | 125 / 120 | 2.5 / 22.9 / 65 | 99.3 | 12.8 | 55.64 |
| DRAM 150 + SSD 8 TB | base | k1 | c2000 | 58.7 | 53.5 | 49.6 | 123 / 116 | 3.2 / 62.7 / 74 | 99.2 | 17.1 | 53.82 |
| DRAM 150 + SSD 8 TB | base | k1 | c2500 | 49.5 | 49.6 | 45.8 | 115 / 110 | 9.6 / 67.8 / 77 | 99.2 | 22.2 | 48.83 |
| DRAM 150 + SSD 8 TB | base | k2 | c1563 | 63.6 | 57.9 | 53.3 | 133 / 126 | 2.0 / 18.5 / 63 | 99.3 | 7.4 | 58.50 |
| DRAM 150 + SSD 8 TB | base | k2 | c2000 | 63.1 | 57.6 | 53.2 | 132 / 127 | 2.5 / 62.0 / 70 | 99.3 | 8.1 | 58.43 |
| DRAM 150 + SSD 8 TB | base | k2 | c2500 | 54.3 | 55.1 | 53.7 | 126 / 127 | 3.7 / 63.7 / 72 | 99.3 | 10.2 | 54.90 |
| DRAM 150 only | st10 | k1 | c2000 | 0.7 | 0.7 | 0.7 | 3 / 3 | 1745 / 1760 / 1769 | 0.0 | 4750 | 0.69 |
| DRAM 150 only | st10 | k1 | c2500 | 0.7 | 0.7 | 0.7 | 3 / 3 | 2157 / 2179 / 2188 | 0.0 | 4600 | 0.69 |
| DRAM 150 only | st10 | k2 | c2000 | 0.7 | 0.7 | 0.7 | 3 / 3 | 1746 / 1762 / 1771 | 0.0 | 4732 | 0.69 |
| DRAM 150 only | st10 | k2 | c2500 | 0.7 | 0.7 | 0.7 | 3 / 3 | 2176 / 2198 / 2211 | 0.0 | 4594 | 0.69 |
| DRAM 150 only | base | k1 | c2000 | 0.6 | 0.6 | 0.6 | 3 / 2 | 2470 / 2500 / 20516 | 1.2 | 5421 | 0.64 |
| DRAM 150 only | base | k1 | c2500 | 0.7 | 0.6 | 0.6 | 2 / 3 | 3109 / 3155 / 3180 | 0.9 | 5498 | 0.64 |
| DRAM 150 only | base | k2 | c2000 | 0.7 | 0.6 | 0.6 | 3 / 2 | 2456 / 2470 / 2480 | 0.4 | 5583 | 0.64 |
| DRAM 150 only | base | k2 | c2500 | 0.6 | 0.7 | 0.6 | 2 / 2 | 3120 / 3168 / 20527 | 1.0 | 5556 | 0.64 |

### nohb (no hand-back, k1)

| tiers | mode | load | r/s 2-4 h | 4-6 h | 6-8 h | new k tok/s 2-4 / 6-8 h | TTFT p50 / p90 / p99 s (6-8 h) | hit % | extra compute % | r/s 1-8 h |
|---|---|---|---|---|---|---|---|---|---|---|
| DRAM 150 + SSD 8 TB | st10 | c1438 | 51.9 | 50.1 | 44.7 | 135 / 130 | 4.9 / 21.9 / 62 | 99.3 | 4.6 | 48.91 |
| DRAM 150 + SSD 8 TB | st10 | c2000 | 51.2 | 49.5 | 45.7 | 133 / 130 | 5.3 / 64.0 / 76 | 99.4 | 5.8 | 49.18 |
| DRAM 150 + SSD 8 TB | base | c1438 | 52.1 | 50.0 | 45.0 | 135 / 130 | 3.2 / 25.9 / 65 | 99.3 | 4.9 | 48.99 |
| DRAM 150 + SSD 8 TB | base | c2000 | 50.5 | 48.5 | 45.4 | 132 / 130 | 4.8 / 63.3 / 73 | 99.3 | 6.2 | 48.40 |
| DRAM 150 only | st10 | c1438 | 0.7 | 0.7 | 0.7 | 3 / 2 | 1280 / 1297 / 1306 | 0.0 | 5332 | 0.68 |
| DRAM 150 only | st10 | c2000 | 0.7 | 0.7 | 0.7 | 3 / 3 | 1747 / 1764 / 1774 | 0.0 | 4116 | 0.69 |
| DRAM 150 only | base | c1438 | 0.6 | 0.6 | 0.6 | 2 / 2 | 1709 / 1728 / 1737 | 0.0 | 5901 | 0.64 |
| DRAM 150 only | base | c2000 | 0.6 | 0.6 | 0.6 | 3 / 3 | 2455 / 2471 / 2482 | 0.4 | 4761 | 0.64 |

### Hit-first rule hg (k1, base mode)

The hg rule exists only on branch `pp-stable` (063c85ae). These runs use that code.

- **DRAM-only base rows.** They are bit-identical (`request_slo.parquet` md5) to the k1 base DRAM-only rows above,
  for all four loads.
- **SSD rows.** With SSD, the pp-stable code writes every finished context to the SSD (write-through), not only
  DRAM's evictions. Its base row therefore differs slightly from the table above.

| tiers | protocol | rule | load | r/s 2-4 h | 4-6 h | 6-8 h | new k tok/s 2-4 / 6-8 h | TTFT p50 / p90 / p99 s (6-8 h) | hit % | extra compute % | r/s 1-8 h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DRAM 150 + SSD 8 TB | hb | base | c2000 | 57.8 | 53.1 | 49.8 | 122 / 117 | 3.3 / 62.7 / 75 | 99.2 | 17.5 | 53.49 |
| DRAM 150 + SSD 8 TB | hb | hg | c2000 | 57.8 | 53.1 | 49.8 | 122 / 117 | 3.3 / 62.7 / 75 | 99.2 | 17.5 | 53.49 |
| DRAM 150 only | hb | base | c2000 | 0.6 | 0.6 | 0.6 | 3 / 2 | 2470 / 2500 / 20516 | 1.2 | 5421 | 0.64 |
| DRAM 150 only | hb | hg | c2000 | 24.4 | 23.3 | 23.0 | 41 / 40 | 1.6 / 17.0 / 2176 | 98.0 | 241 | 23.66 |
| DRAM 150 only | hb | base | c2500 | 0.7 | 0.6 | 0.6 | 2 / 3 | 3109 / 3155 / 3180 | 0.9 | 5498 | 0.64 |
| DRAM 150 only | hb | hg | c2500 | 24.5 | 24.6 | 25.6 | 39 / 43 | 1.5 / 7.1 / 2782 | 98.1 | 238 | 24.80 |
| DRAM 150 + SSD 8 TB | nohb | base | c2000 | 50.1 | 48.5 | 45.1 | 132 / 129 | 4.9 / 63.4 / 76 | 99.3 | 7.0 | 48.02 |
| DRAM 150 + SSD 8 TB | nohb | hg | c2000 | 50.1 | 48.5 | 45.1 | 132 / 129 | 4.9 / 63.4 / 76 | 99.3 | 7.0 | 48.02 |
| DRAM 150 only | nohb | base | c1438 | 0.6 | 0.6 | 0.6 | 2 / 2 | 1709 / 1728 / 1737 | 0.0 | 5901 | 0.64 |
| DRAM 150 only | nohb | hg | c1438 | 20.8 | 20.8 | 21.0 | 46 / 51 | 1.6 / 36.3 / 1357 | 97.9 | 184 | 20.89 |
| DRAM 150 only | nohb | base | c2000 | 0.6 | 0.6 | 0.6 | 3 / 3 | 2455 / 2471 / 2482 | 0.4 | 4761 | 0.64 |
| DRAM 150 only | nohb | hg | c2000 | 22.6 | 23.1 | 22.4 | 48 / 50 | 1.5 / 13.6 / 2347 | 98.0 | 177 | 22.94 |

### With SSD: where r/s moves across the 8 h

- **Compute is flat.** Computed prompt tokens are 138-141k tok/s in every 2 h window of every SSD run.
- **Larger rounds explain the late fall in r/s.** New tokens per round grow across the run in every SSD run,
  in both modes and at every k and load:
  - hb: 2.08-2.11k at 2-4 h, 2.34-2.40k at 6-8 h.
  - nohb: 2.60-2.63k at 2-4 h, 2.85-2.92k at 6-8 h.
- **The 6-8 h window also carries slightly more extra compute.** Example, k2 st10 c2000: 7.0% at 2-4 h, 10.0% at
  6-8 h.
- **Net effect.** r/s at 6-8 h is up to 17% below the 2-4 h value. New-token throughput moves by -8% to +1%
  (k2: -6% to +1%).
- **No backlog builds.**
  - At 8 h, 330-1,560 rounds are waiting.
  - None waits longer than 60-86 s.
  - Hit % stays at 99.2-99.4% in every SSD run.
- **The SSD never fills in 8 h.**
  - It holds 3.1-4.5 TB of 8 TB at 8 h: 39-45% for k1, 54-56% for k2.
  - It evicts nothing.
  - SSD reads are 4.8-8.3 GB/s of the 10 GB/s channel; the 8.3 GB/s is k2 base c2500.
  - The tier stats are in `tiers8h.jsonl`, written by `ckpt_table.py`.

### DRAM only: how the collapse looks

- **No context survives.** Without the SSD, the retained contexts exceed HBM plus 150 GB of DRAM. Under oldest-first
  overdue service, every round waits behind rounds that recompute their whole context (about 150k tokens). By the
  time a round is served, its own context has been evicted.
- **The GPUs stay busy recomputing.** Compute stays at 135-137k tok/s, but nearly all of it is recompute:
  - Extra compute is 41x to 59x the new tokens.
  - 0.6-0.7 r/s from hour 2, in every mode, at every k, load and protocol.
  - TTFT p50 is 1,280-3,120 s.
- **Neither st10 nor k2 changes this.**
- **hg prevents the collapse with DRAM only** (k1, base mode): 21-25 r/s and 98% hit %.
  - The cost: the rounds that still miss wait behind every hit, and TTFT p99 is 1,360-2,780 s.
  - Extra compute is 1.8-2.4x the new tokens, from those misses.
- **With SSD, hg changes nothing.** No held context is ever at risk, and the runs are bit-identical to base
  (md5, both protocols).

## Conclusion

- **With DRAM 150 GB + SSD 8 TB, the system is stable for 8 h on v3.**
  - Both protocols, every load, both modes, k1 and k2.
  - Compute is saturated at about 140k tok/s throughout.
  - Hit % is 99.2-99.4%.
  - Queues stay bounded by the 60 s force-schedule rule. TTFT p90 is 17-26 s at c1563/c1438 and 62-76 s at
    c2000/c2500, where the 60 s bound sets the tail.
  - The r/s drift over the run is round-size mix, not loss of cache. The 8 TB SSD is at most 56% full at 8 h and has
    not started evicting, so runs longer than 8 h are not covered.
- **The deployed configuration (st10, k2, hb)** gives 58.6 / 58.6 / 54.9 r/s over [1 h, 8 h) at c1563 / c2000 / c2500.
  - Extra compute is 7.2 / 8.2 / 10.3%.
  - Last-2 h TTFT p50 is 3.0-4.2 s.
  - k1 gives 55.4 / 53.2 / 48.1 r/s with 13.3 / 19.3 / 24.8% extra compute, matching the k doc.
  - base and st10 are within 0.5 r/s at k2.
- **With DRAM only, throughput collapses in every configuration**: 0.6-0.7 r/s from hour 2, near-zero hits, and
  TTFT in the thousands of seconds. Neither st10 nor k2 helps.
- **hg helps only with DRAM only.**
  - It turns the collapse into 21-25 r/s (k1, base mode), at the cost of a TTFT p99 of 1,360-2,780 s for the
    rounds that recompute.
  - With SSD it never triggers, and runs are bit-identical to the base rule.
  - It is not on the deployed code branch (pp-staged), so hg with st10 or k2 was not run.

## Not covered

- **Attribution of the whole-context hit rule (1de110c0) on v3: not rerun.**
  - Rerunning it would mean rebuilding the pre-rule code (7bb224a6) and running it on the v3 traces.
  - The rule's cost on v3 is the extra compute in the tables above and in the k doc: 12.8-24.8% at k1 and 7.2-10.3%
    at k2 (hb).
- **hg on st10 or k2.** Porting the two pp-stable commits onto pp-staged conflicts with staged loading
  (`fresh_source.rs`), so it was not done here.

## Sources and reproduction

- **KDA checkpoints k.** For the k1/k2/k4 extra compute and the per-class split, see `kda_ckpt_k_recompute_v3.md`
  (logs/20261018_1_v3_katomic, logs/20261018_6_v3_stB). The k1/k2 rows here reproduce it at the newer code:
  - k1 base and k1/k2 st10 are md5-identical to those runs.
  - k2 base differs because of 0d19fb23 (reads move only surviving states): 58.50 / 58.43 / 54.90 r/s here.
- **Main runs.** `/raid/kanzhu/ServingStudio/wt-pp-staged/logs/20261018_15_v3_8h_steady/`
  - Layout: `{hb,nohb}/{k1,k2}/{st10,base}/{s8t,d150}/c*`.
  - Branch pp-staged at e4de4292, repo `profiling/profile.db` (md5 12529b8e, the database the k study used).
  - Commands, from `wt-pp-staged`:
    - `bash logs/20261018_15_v3_8h_steady/make_traces.sh` (nohb traces into `traces/`).
    - `uv run python -m launcher logs/20261018_15_v3_8h_steady/hb_s8t.yaml logs/20261018_15_v3_8h_steady/hb_d150.yaml --no-gpu --no-plot --parallelism 12`
    - `uv run python -m launcher logs/20261018_15_v3_8h_steady/nohb.yaml --no-gpu --no-plot --parallelism 8`
- **hg runs.** `/raid/kanzhu/ServingStudio/wt-pp-stable/logs/20261018_0_v3_hg/`
  - Layout: `{hb,nohb}/{s8t,d150}/{base,hg}/c*`.
  - Branch pp-stable at 063c85ae, same profile.db md5.
  - Command, from `wt-pp-stable`:
    `uv run python -m launcher logs/20261018_0_v3_hg/hb_s8t.yaml logs/20261018_0_v3_hg/hb_d150.yaml logs/20261018_0_v3_hg/nohb_s8t.yaml logs/20261018_0_v3_hg/nohb_d150.yaml --no-gpu --no-plot --parallelism 6`
- **Tables.**
  - `bash logs/20261018_15_v3_8h_steady/rows.sh` runs `steady_table.py` on every completed run of both directories
    and writes `rows.jsonl`.
  - `doc_tables.py rows.jsonl` prints the tables above.
  - SSD occupancy and read rates: `ckpt_table.py 3600000 28800000 <run>=<trace> ...` writes `tiers8h.jsonl`.
