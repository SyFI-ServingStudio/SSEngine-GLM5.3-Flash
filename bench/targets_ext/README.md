# Extended prefix bring-up (M4)

More cases for the prefix bring-up of `5_prefix_cache.md` (handoff `glm53_flash_pp4_b200_v1`, frozen at
ServingStudioSim `7bb224a6`), with simulator targets made exactly as the handoff made `data/prefix_targets.json`.
The handoff directory is not changed.

## Sessions

Same session shape as `data/bringup/prefix_sessions.csv`: round 0 is a prompt of P tokens; each later round, 10 s
after the previous round's token, sends the previous round's prompt, that token and A − 1 new tokens. Round k hits
P + (k − 1)·A tokens and computes A. One session is active at a time.

| File | Rounds | P | A | Sessions | Slot |
|---|---|---|---|---|---|
| `prefix_sessions_ext.csv` | 2 | 2k, 3,000, 4k, 8k, 16k, 32k, 64k, 100,003, 128k, 256k, 512k | 256, 1k, 8k | 5 per case, 165 | 30 s |
| `prefix_multiround_ext.csv` | 4 (rounds 0–3) | 8k, 128k, 512k | 256, 1k, 8k | 5 per case, 45 | 60 s |

- P = 3,000 and 100,003 resume at a token that is not a multiple of 4 (the indexer's pool) or of a block.
- The two-round set holds the handoff's 8 cases (P ∈ {8k, 32k, 128k, 512k} × A ∈ {1k, 8k}).
- The longest two-round session (512k + 8k) ends 14.7 s after it starts, inside its 30 s slot. The longest
  four-round one (512k + 3 × 8k) ends 35.3 s after it starts, so the multi-round slot is 60 s. The exporter checks
  that no two sessions overlap.

## Targets

- `prefix_targets_ext.json`: the two-round set, in exactly the format of `data/prefix_targets.json` (one entry per
  (P, A): `round0` TTFT and microbatch count; `round1` TTFT, hit and computed tokens, and per microbatch the tokens,
  context, stage compute and sends).
- `prefix_multiround_ext.json`: the four-round set, in the same format plus `"rounds": 4` and the keys `round2` and
  `round3`, shaped like `round1`.
- Reproduction: the handoff's own trace, run with the setup below, exports a file byte-identical to
  `data/prefix_targets.json`, and the 8 handoff cases in `prefix_targets_ext.json` equal its entries exactly.

## Compressed traces (for the GPU run)

`prefix_sessions_ext_compressed.csv` and `prefix_multiround_ext_compressed.csv`: the same sessions, each starting
1 s after the previous one's last token in the simulated run, so the GPUs never sit idle for long. Rounds and tool
waits are unchanged. Wall time of the replay, by the simulated run:

| Trace | Sessions | Wall time |
|---|---|---|
| `prefix_sessions_ext_compressed.csv` | 165 | 1,988.0 s (33.1 min) |
| `prefix_multiround_ext_compressed.csv` | 45 | 1,495.9 s (24.9 min) |

The 1 s margin absorbs a real system up to about 1 s slower per session than the targets; a slower one overlaps
sessions. Then rewrite the compressed file with a larger `--gap-ms`.

Replay (each on a freshly started server, from the lane root):

```
python -m bench.run prefix_bringup --trace bench/targets_ext/prefix_sessions_ext_compressed.csv \
    --run-name prefix_bringup_ext ...
python -m bench.run prefix_bringup --trace bench/targets_ext/prefix_multiround_ext_compressed.csv \
    --run-name prefix_bringup_multiround ...
```

`bench/run.py` stops the client after `max(--client-timeout, 25 min)`, and `submit` (the Slurm path) does not
forward `--client-timeout`. Both traces run longer than 25 min with the client's corpus tokenization, so pass
`--client-timeout 2700` to an `--inline` run, or forward the option in `submit`.

## How they were made

Simulator: a worktree of ServingStudioSim at `7bb224a6` (detached), with submodule `req-frontend` at `7f6aa9d8`, its
committed `profiling/profile.db` (identical to `third_party/sim_7bb224a6/profiling/profile.db`), at
`/raid/kanzhu/ServingStudio/tmp/sim-7bb224a6-prefix-ext`. The preset is a copy of the run that made the handoff's
prefix targets (wt-pp-main `logs/20261013_1_pp4_sched/pp4_prefix.yaml` with `backends_pp.yaml`, sched `plain`:
greedy 8,192-token microbatches, FIFO, plain chunking; HBM-only pool of 100.18 GB per GPU, 61,258,368 tokens; routing
`corpus` from the TP4/EP4 capture). Only the trace, `duration_ms` and `log_dir` differ; the resolved run config of the
`orig` run equals the frozen one except for the trace's path. Preset: `logs/20261015_0_prefix_ext/preset.yaml` in
that worktree.

```
python bench/targets_ext/make_prefix_ext.py                       # the two traces (lane root)

cd /raid/kanzhu/ServingStudio/tmp/sim-7bb224a6-prefix-ext && source /raid/kanzhu/ServingStudio/.env
uv run python -m launcher logs/20261015_0_prefix_ext/preset.yaml  # runs orig, two, multi (CPU only)
E=/raid/kanzhu/ServingStudio/pp-impl/lanes/longctx/bench/targets_ext
uv run python $E/prefix_targets_ext.py logs/20261015_0_prefix_ext/two/plain $E/prefix_sessions_ext.csv \
    $E/prefix_targets_ext.json --compressed $E/prefix_sessions_ext_compressed.csv
uv run python $E/prefix_targets_ext.py logs/20261015_0_prefix_ext/multi/plain $E/prefix_multiround_ext.csv \
    $E/prefix_multiround_ext.json --compressed $E/prefix_multiround_ext_compressed.csv
# reproduction check: export the orig run and compare with the handoff's file
H=/raid/kanzhu/ServingStudio/wt-pp-main/doc/handoff/glm53_flash_pp4_b200_v1/data
uv run python $E/prefix_targets_ext.py logs/20261015_0_prefix_ext/orig/plain $H/bringup/prefix_sessions.csv \
    $TMPDIR/orig.json --compressed $TMPDIR/orig_compressed.csv && cmp $TMPDIR/orig.json $H/prefix_targets.json
```

`prefix_targets_ext.py` is a parameterized copy of the handoff's `data/prefix_targets.py`: the same extraction, with
the trace and output on the command line, any number of rounds, and the compressed trace.

## Notes on the numbers

- With A = 8k, round 1 is the same 223.758 ms (stages 57.130 / 54.159 / 55.325 / 55.698 ms) for P = 2k, 3,000, 4k
  and 8k; with A = 1k and P = 512k, rounds 1 and 2 are equal. The simulator gives the same time to these contexts;
  why was not investigated. The differences are inside the ±10% pass rule.
