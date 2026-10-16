# Extended prefix bring-up (M4)

More cases for the prefix bring-up of `5_prefix_cache.md` (handoff `glm53_flash_pp4_b200_v1`, frozen at
ServingStudioSim `7bb224a6`), with simulator targets made exactly as the handoff made `data/prefix_targets.json`.
The handoff directory is not changed.

## Protocol change from the handoff: 1 s between rounds

The handoff sends round k + 1 **10 s** after round k's token. The GPU replay here uses **1 s**: the wait is only
idle GPU time, and cayenne stops jobs whose GPUs sit idle. Sessions stay isolated: the next session starts 1 s after
the previous one's last token.

The change moves no target. The simulator, run on the 1 s traces as they are replayed (the compressed parts below),
exports `prefix_targets_ext_w1s.json` and `prefix_multiround_ext_w1s.json` byte-identical to the 10 s
`prefix_targets_ext.json` and `prefix_multiround_ext.json`. No run evicts anything (`prefix_cache_event`: only
miss, hit and retain): the pool holds 61.3M tokens, and the cache peaks at 8.0M tokens in a part and 17.8M in the
whole two-round trace.

## Sessions

Same session shape as `data/bringup/prefix_sessions.csv`: round 0 is a prompt of P tokens; each later round, after
the previous round's token and the round wait, sends the previous round's prompt, that token and A − 1 new tokens.
Round k hits P + (k − 1)·A tokens and computes A. One session is active at a time.

| Trace (10 s wait / 1 s wait) | Rounds | P | A | Sessions | Slot |
|---|---|---|---|---|---|
| `prefix_sessions_ext.csv` / `_w1s.csv` | 2 | 2k, 3,000, 4k, 8k, 16k, 32k, 64k, 100,003, 128k, 256k, 512k | 256, 1k, 8k | 5 per case, 165 | 30 s |
| `prefix_multiround_ext.csv` / `_w1s.csv` | 4 (rounds 0–3) | 8k, 128k, 512k | 256, 1k, 8k | 5 per case, 45 | 60 s |

- P = 3,000 and 100,003 resume at a token that is not a multiple of 4 (the indexer's pool) or of a block.
- The two-round set holds the handoff's 8 cases (P ∈ {8k, 32k, 128k, 512k} × A ∈ {1k, 8k}).
- With the 10 s wait, the longest two-round session (512k + 8k) takes 14.7 s, inside its 30 s slot, and the
  longest four-round one (512k + 3 × 8k) 35.3 s, so the multi-round slot is 60 s. The exporter checks that no two
  sessions of a run overlap.

## Targets

- `prefix_targets_ext.json` (= `prefix_targets_ext_w1s.json`): the two-round set, in exactly the format of
  `data/prefix_targets.json`: one entry per (P, A), with `round0` TTFT and microbatch count, and `round1` TTFT, hit and
  computed tokens, and per microbatch the tokens, context, stage compute and sends.
- `prefix_multiround_ext.json` (= `prefix_multiround_ext_w1s.json`): the four-round set, in the same format plus
  `"rounds": 4` and the keys `round2` and `round3`, shaped like `round1`.
- Reproduction: the handoff's own trace, run with the setup below, exports a file byte-identical to
  `data/prefix_targets.json`, and the 8 handoff cases in `prefix_targets_ext.json` equal its entries exactly.

## What the GPU replays: the 1 s parts

Each part runs on its own freshly started server, so the parts can run as parallel 4-GPU jobs. Sessions start 1 s
after the previous one's last token, rounded up to a whole ms; each part starts at t = 0 and keeps the full trace's
session ids. Wall time is the simulated replay, without server start and the client's corpus tokenization.

| Part | P | Sessions | Wall time |
|---|---|---|---|
| `prefix_sessions_ext_w1s_compressed_p1.csv` | 2k, 3,000, 4k, 8k | 60 | 133.1 s |
| `prefix_sessions_ext_w1s_compressed_p2.csv` | 16k, 32k, 64k | 45 | 113.5 s |
| `prefix_sessions_ext_w1s_compressed_p3.csv` | 100,003, 128k, 256k | 45 | 156.4 s |
| `prefix_sessions_ext_w1s_compressed_p4.csv` | 512k | 15 | 97.1 s |
| `prefix_multiround_ext_w1s_compressed_p1.csv` | 8k | 15 | 67.1 s |
| `prefix_multiround_ext_w1s_compressed_p2.csv` | 128k | 15 | 80.8 s |
| `prefix_multiround_ext_w1s_compressed_p3.csv` | 512k | 15 | 131.0 s |

The whole traces in one run each: `prefix_sessions_ext_w1s_compressed.csv` 503.1 s, and
`prefix_multiround_ext_w1s_compressed.csv` 280.9 s. The 10 s versions, `prefix_sessions_ext_compressed.csv` and
`prefix_multiround_ext_compressed.csv`, take 1,988.1 s and 1,495.9 s.

The 1 s gap absorbs a real system up to about 1 s slower per session than the targets; a slower one overlaps
sessions. Then rewrite the compressed files with a larger `--gap-ms`.

Replay, from the lane root, one part per fresh server:

```
python -m bench.run prefix_bringup --trace bench/targets_ext/prefix_sessions_ext_w1s_compressed_p1.csv \
    --run-name prefix_bringup_ext_p1 ...
```

`bench/run.py` stops the client after max(`--client-timeout`, 25 min); every part is far below that. `bench/prefix.py`
summarizes rounds 0 and 1 only.

## How they were made

Simulator: a worktree of ServingStudioSim at `7bb224a6` (detached), with submodule `req-frontend` at `7f6aa9d8`, its
committed `profiling/profile.db` (identical to `third_party/sim_7bb224a6/profiling/profile.db`); `$SIM` below. The preset is a copy of the run that made the handoff's
prefix targets (wt-pp-main `logs/20261013_1_pp4_sched/pp4_prefix.yaml` with `backends_pp.yaml`, sched `plain`:
greedy 8,192-token microbatches, FIFO, plain chunking; HBM-only pool of 100.18 GB per GPU, 61,258,368 tokens; routing
`corpus` from the TP4/EP4 capture). Only the trace, `duration_ms` and `log_dir` differ; the resolved run config of the
`orig` run equals the frozen one except for the trace's path. Presets, in that worktree:

| Experiment | Traces |
|---|---|
| `logs/20261015_0_prefix_ext` | `orig` (the handoff's trace), `two`, `multi` (10 s wait) |
| `logs/20261015_1_prefix_ext_w1s` | `two`, `multi` (1 s wait, slotted): durations for the compressed traces |
| `logs/20261015_2_prefix_ext_w1s_parts` | the 7 compressed 1 s parts: the `_w1s.json` targets |

```
python bench/targets_ext/make_prefix_ext.py          # the slotted traces, 10 s and 1 s (lane root)

cd $SIM && source <workspace>/.env                 # SIM: that ServingStudioSim worktree
E=<this repo>/bench/targets_ext
X=logs/20261015_0_prefix_ext W=logs/20261015_1_prefix_ext_w1s Q=logs/20261015_2_prefix_ext_w1s_parts
uv run python -m launcher $X/preset.yaml             # CPU only, as are the two below
uv run python $E/prefix_targets_ext.py $E/prefix_targets_ext.json --run $X/two/plain $E/prefix_sessions_ext.csv \
    --compressed $E/prefix_sessions_ext_compressed.csv
uv run python $E/prefix_targets_ext.py $E/prefix_multiround_ext.json --run $X/multi/plain \
    $E/prefix_multiround_ext.csv --compressed $E/prefix_multiround_ext_compressed.csv

uv run python -m launcher $W/preset.yaml
uv run python $E/prefix_targets_ext.py $TMPDIR/two_w1s.json --run $W/two/plain $E/prefix_sessions_ext_w1s.csv \
    --compressed $E/prefix_sessions_ext_w1s_compressed.csv
uv run python $E/prefix_targets_ext.py $TMPDIR/multi_w1s.json --run $W/multi/plain $E/prefix_multiround_ext_w1s.csv \
    --compressed $E/prefix_multiround_ext_w1s_compressed.csv
(cd $E && python3 split_by_p.py prefix_sessions_ext_w1s_compressed.csv $TMPDIR/two_w1s.json 8192 65536 262144 524288 \
  && python3 split_by_p.py prefix_multiround_ext_w1s_compressed.csv $TMPDIR/multi_w1s.json 8192 131072 524288)

uv run python -m launcher $Q/preset.yaml
uv run python $E/prefix_targets_ext.py $E/prefix_targets_ext_w1s.json \
    $(for n in 1 2 3 4; do echo --run $Q/two_p$n/plain $E/prefix_sessions_ext_w1s_compressed_p$n.csv; done)
uv run python $E/prefix_targets_ext.py $E/prefix_multiround_ext_w1s.json \
    $(for n in 1 2 3; do echo --run $Q/multi_p$n/plain $E/prefix_multiround_ext_w1s_compressed_p$n.csv; done)
cmp $E/prefix_targets_ext_w1s.json $E/prefix_targets_ext.json && cmp $E/prefix_multiround_ext_w1s.json $E/prefix_multiround_ext.json

# reproduction check: export the orig run and compare with the handoff's file
H=<ServingStudioSim at the v1 handoff commit>/doc/handoff/glm53_flash_pp4_b200_v1/data
uv run python $E/prefix_targets_ext.py $TMPDIR/orig.json --run $X/orig/plain $H/bringup/prefix_sessions.csv \
    && cmp $TMPDIR/orig.json $H/prefix_targets.json
```

- `prefix_targets_ext.py` is a parameterized copy of the handoff's `data/prefix_targets.py`: the same extraction,
  with the traces and output on the command line, any number of rounds, several runs merged, and the compressed
  trace.
- `split_by_p.py` cuts a compressed trace into parts by P; the bounds above balance the parts' wall times.
- Compressed start times are whole ms. The simulator steps in 100 µs ticks, and an arrival inside a tick waits
  for the next one: with fractional-ms starts, round-0 TTFTs moved by up to 0.09 ms between sessions of one case.

## Notes on the numbers

- With A = 8k, round 1 is the same 223.758 ms (stages 57.130 / 54.159 / 55.325 / 55.698 ms) for P = 2k, 3,000, 4k
  and 8k; with A = 1k and P = 512k, rounds 1 and 2 are equal. The simulator gives the same time to these contexts;
  why was not investigated. The differences are inside the ±10% pass rule.
