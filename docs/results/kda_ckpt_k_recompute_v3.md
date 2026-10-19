# KDA round-end checkpoints k1 / k2 / k4: extra compute on the v3 trace

GLM-5.3-Flash NVFP4, PP4 on 4x B200, HBM -> DRAM -> SSD retention tiers, ServingStudio Sim, TraceLab v3
new-model traces (8 h closed loop). Date of the runs: 2026-10-18.

## What k means

KDA (linear attention) layers can resume a cached context only from a saved recurrent state at that
exact token position. KV can be cut anywhere; the KDA state cannot. `kN` is
`recurrent_state_checkpoints: N`. Each retained context keeps its latest N round-end states:

- A round's **prefill end** (prompt end, `P = p + i`).
- Its **hand-back decode end** (`D = P + o - 1`, the external decode returns the output KV and the state).

With hand-back, k1 keeps only the decode end. k2 keeps the decode end and the prefill end of the latest
round. k4 also keeps the two ends of the round before.

The rules are the same in every tier:

- **Atomic entry.** An entry is the KV plus its k states, treated as one unit.
- **Eviction.** LRU eviction is whole-entry. Every state is charged its real bytes: 36.9 MB/GPU per state
  in DRAM and SSD.
- **Reads.** A tier read moves only the states that survive the round it serves (0d19fb23). The base runs
  below predate this change and move all k states; it never affects k1.
- **Resume.** A round resumes from the deepest kept state at or before its declared prefix. If no kept state
  is there, the whole prefix is recomputed.

k1 is bit-identical (`request_slo.parquet` md5) across every code version used here.

## What is counted

- **Trace.** `data/tracelab-v3/traces/closed_c{1563,2000,2500}_s0.csv`, closed loop with 1563/2000/2500
  concurrent sessions, 8 h runs.
- **Window.** Rounds count when their first token falls in [1 h, 8 h). Only completed rounds count.
- **Per-round sums.** For each round in `raw/request_slo.parquet`:
  - computed = `declared_prefix_tokens + fresh_prompt_tokens - prefix_cache_hit_tokens`
  - new = `fresh_prompt_tokens`
- **Extra compute (recompute / new)** = (sum computed - sum new) / sum new. This is the tokens prefilled
  beyond the new tokens, as a share of the new tokens.
- **r/s.** Completed rounds per second for the 4-GPU system.
- **Setup.**
  - Hardware: B200, layers 12/11/11/11, `max_batch_tokens` 32768, HBM KV 100.18 GB/GPU, DRAM 150 GB/GPU at
    50 GB/s, SSD 8 TB at 10 GB/s/GPU.
  - Decode runs externally, with output KV handed back.
  - Scheduling: `force_schedule_after_ms` 60 s, warm start in the slowest tier, SSD read wait 1 s.
- **Modes.**
  - `base`: inclusive DRAM (`hbm-evictions`).
  - `st10`: exclusive DRAM (`hbm-evictions-exclusive`) plus a 10% HBM loading area. This is the chosen
    target configuration.
  - `g220`: DRAM 220 GB/GPU.
- **Round classes.** Each round is compared with the previous round of its session:
  - `inside_output`: the prompt ends inside the previous round's handed-back output.
  - `inside_prompt`: the prompt ends inside the previous prompt.
  - `extend`: everything else.

## Results

Extra compute, recompute / new tokens:

| k | mode | c1563 | c2000 | c2500 |
|---|---|---|---|---|
| **k1** | base | **12.83%** | 17.11% | 22.19% |
| **k2** | base | **7.61%** | 8.12% | 11.01% |
| **k4** | base | **4.28%** | 4.90% | 6.51% |
| k1 | st10 g150 | 13.31% | 19.34% | 24.77% |
| k2 | st10 g150 | 7.17% | 8.18% | 10.35% |
| k4 | st10 g150 | 4.29% | 5.06% | 5.86% |
| k1 | st10 g220 | 7.08% | 17.56% | 24.56% |
| k2 | st10 g220 | 5.65% | 7.14% | 9.64% |
| k4 | st10 g220 | 3.71% | 4.86% | 5.74% |

Throughput, r/s for 4 GPUs (TTFT p50/p90/p99 in s):

| k | mode | c1563 | c2000 | c2500 |
|---|---|---|---|---|
| k1 | base | 55.64 (2.02/14.3/62.6) | 53.82 (2.55/62.0/70.7) | 48.83 (6.04/66.0/76.8) |
| k2 | base | 58.43 (1.56/10.4/62.3) | 58.34 (1.86/61.8/67.4) | 54.15 (3.03/63.4/73.2) |
| k4 | base | 56.67 (1.00/9.4/62.1) | 56.96 (1.16/61.6/64.5) | 53.51 (1.67/62.4/69.2) |
| k1 | st10 g150 | 55.39 (2.57/14.5/61.4) | 53.21 (3.31/62.7/75.6) | 48.14 (6.37/72.4/91.5) |
| k2 | st10 g150 | 58.60 (2.06/11.2/61.4) | 58.64 (2.53/62.2/70.2) | 54.94 (3.70/65.5/79.8) |
| k4 | st10 g150 | 59.63 (1.65/8.7/62.5) | 59.63 (1.76/62.1/67.9) | 55.88 (2.42/64.4/77.0) |

GPU compute is about 139-140k tok/s in every run except k4 base, which drops to 133-139k tok/s.
Throughput therefore tracks the extra compute.

Where the k1 extra comes from (base, share of new tokens):

| load | inside_output | inside_prompt | extend | inside rounds / all rounds |
|---|---|---|---|---|
| c1563 | 5.52% (1,043 rounds) | 7.27% (9,213) | 0.04% | 0.74% |
| c2000 | 9.09% (1,165) | 7.97% (8,867) | 0.04% | 0.75% |
| c2500 | 11.46% (1,202) | 10.68% (8,460) | 0.04% | 0.79% |

- **Few rounds, long contexts.** Fewer than 1% of rounds cause the extra. Prefix hit is still
  99.7-99.85%, but contexts are about 150k tokens against about 2.2k new tokens per round. Missing 0.15% of
  the prefix therefore costs about 13% of the new tokens.
- **k2 removes the inside_output part.** It keeps the prefill-end state:
  - At k2, inside_output falls to 0.02-0.03%.
  - What remains is inside_prompt: the prompt is genuinely shorter than the previous prompt.
- **k4 also recovers part of inside_prompt.**

Trace-level check, ideal cache: `v3new_preserving.csv`, all 887,873 rounds. This assumes k1 holds a state
only at the previous decode end, and that an inside round recomputes its whole prefix.

| round class | rounds | k1 extra (% of new) |
|---|---|---|
| prompt end in [prev prompt end, prev decode end) | 1,757 | 32.56% (median shortfall 22 tokens) |
| prompt shorter than previous, drop < 64k | 720 | 6.67% |
| drop >= 64k (compaction) | 5,412 | 3.28% |
| total | | 42.51% |

The sims see fewer such rounds in their [1 h, 8 h) window, and the tiers also hold shorter, older copies
that are partially reusable. That is why the sim numbers are lower than this bound.

The 20-min M6 v3 trace (`data/tracelab-v3/m6/sessions_2ps_v3.csv`) has very few inside rounds. At
trace level, with no cache pressure:

- k1 extra is 1.57% of new tokens, k2 1.18%, k4 0.88%.
- Inside rounds account for 12.80% of k1's computed tokens; this is a different quantity from extra/new.

## Conclusion

- **k1 needs 12.8% more prefill than the new tokens on v3** (c1563, base), rising to 17.1% at c2000 and
  22.2% at c2500. With the target `st10` configuration it is 13.3/19.3/24.8%.
- **k2 cuts the extra to 7.6/8.1/11.0%** (base); `st10` gives 7.2/8.2/10.4%. It removes the
  inside_output class: a prompt that stops a few tokens short of the handed-back decode end can resume from
  the prefill-end state.
  - Throughput gain over k1: +5% at c1563, +8% at c2000, +11% at c2500 (base); +6/+10/+14% with `st10`.
- **k4 roughly halves the extra again**, to 4.3/4.9/6.5%. The throughput gain is small:
  - With `st10` it beats k2 by +1.0 to +0.9 r/s.
  - In base mode it falls below k2, because every read moves 4 states (36.9 MB/GPU each) and SSD reads
    reach about 9.3-9.7 GB/s of 10.
- **A clean k1 is the exclusive-DRAM one.** In base (inclusive) mode, k1 benefits from a stale shorter copy
  left in DRAM, which acts as an accidental second checkpoint. Exclusive k1 at c1563 is 54.12 r/s with
  14.52% extra (`20261018_1_v3_katomic/k1/excl`).
- **What remains at k2 and k4 is the inside_prompt class**, where the prompt is genuinely shorter. Anchor
  retention (`m1s2`/`m2s2`, logs/20261018_10_kda_anchors and 20261018_12_kda_compress) targets this class.

## Sources and reproduction

All `logs/` paths are simulator runs in a ServingStudioSim checkout of branch pp-staged, CPU only.

- **base runs:** `logs/20261018_1_v3_katomic/{k1,k2,k4}/base/hb_c{1563,2000,2500}`.
  - Code: c7dc261f.
  - Command: `uv run python -m launcher logs/20261018_1_v3_katomic/preset.yaml --no-gpu --no-plot`.
  - The k1 rows equal wt-pp-kda-ckpt `logs/20261017_10_v3new_k124`.
- **st10 runs:** `logs/20261018_6_v3_stB/{k1,k2,k4}/st10/{g150,g220}/hb_c*`.
  - Code: k1/k2 at 91264f5f, k4 rerun at 1f9c419d.
  - k1/k2 st10 were checked bit-identical at 1f9c419d on M6.
  - Preset: `logs/20261018_6_v3_stB/preset.yaml`.
- **Table rows:** `logs/20261018_1_v3_katomic/rows8h.jsonl`, `logs/20261018_6_v3_stB/rows8h.jsonl`.
  - Produced by `ckpt_table.py 3600000 28800000 <run>=<trace> ...` (see `logs/20261018_6_v3_stB/table.sh`).
  - recompute/new = `comp_tok_s / new_tok_s - 1`.
  - Rerunning `logs/20261018_1_v3_katomic/ckpt_table.py` on the 27 runs above reproduces r/s, compute and
    new tok/s exactly (checked 2026-10-18).
- **Per-class split and trace-level split:** the two DuckDB queries described above, over `request_slo.parquet`
  joined to the trace by row id, and over `data/tracelab-v3/traces/v3new_preserving.csv`.
  - Scripts: [`kda_ckpt/extra_by_class.py`](kda_ckpt/extra_by_class.py) (per-class split, `<run dir>=<trace>` arguments) and
    [`kda_ckpt/trace_k1_split.py`](kda_ckpt/trace_k1_split.py) (trace-level split); they need `duckdb`.
- **M6 v3 trace-level numbers:** `logs/20261018_10_kda_anchors/step0_sessions_2ps_v3.json` (`step0.py`).
  Extra/new = comp / sum(input_len) - 1, where sum(input_len) = 711,802,609.
