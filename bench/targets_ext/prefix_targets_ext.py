"""Export the extended prefix bring-up targets from simulator runs of make_prefix_ext.py's traces.

A parameterized copy of the handoff's data/prefix_targets.py (frozen at 7bb224a6): the same extraction, with the
trace and the output given on the command line, any number of rounds per session, and the compressed trace.

For each (prefix P, computed A), from the run of a trace whose sessions run one at a time:
  round0        TTFT of the P-token prompt and its microbatch count
  round<k>      (k >= 1) TTFT, hit tokens (P + (k - 1) * A), computed tokens (A), and per microbatch the stage compute
                times and the sends between stages (the gap from one stage's end to the next one's start)
For two-round sessions the entries have exactly the keys of prefix_targets.json. For more rounds they also carry
`rounds` (the round count) and `round<k>` for every round.
The 5 sessions of a case must give the same times; the script checks it.

It also writes the compressed trace <out stem>_compressed.csv (well, `--compressed`): the same sessions, each starting
1 s after the previous one's last token in the simulated run, instead of on a fixed slot. Its total wall time (the
last session's end) is printed.

usage (ServingStudioSim root at 7bb224a6, which has pyarrow in its uv env):
  uv run python <this file> <run dir> <trace csv> <targets json> --compressed <compressed csv>
"""
import argparse, collections, csv, json
from pathlib import Path

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("trace", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--compressed", type=Path, required=True)
ap.add_argument("--gap-ms", type=float, default=1000.0)
args = ap.parse_args()

trace = list(csv.DictReader(open(args.trace)))
# The simulator numbers requests by their row in the trace.
slo = {r["request_id"]: r for r in pq.read_table(args.run / "raw/request_slo.parquet").to_pylist()}
assert len(slo) == len(trace), (len(slo), len(trace))
stages = [pq.read_table(args.run / f"raw/cost_log/worker_stage_{s}.parquet",
                        columns=["wall_start_ms", "total_time_ms", "groups"]).to_pylist() for s in range(4)]


def microbatches(start, end):
    """The microbatches stage 0 starts in [start, end), each followed through the 4 stages (FIFO per stage)."""
    first = [i for i, r in enumerate(stages[0]) if start <= r["wall_start_ms"] < end]
    out = []
    for i in first:
        rows = [stages[s][i] for s in range(4)]
        out.append(dict(
            tokens=sum(g["batch_tokens"] for g in rows[0]["groups"]),
            context=[p for g in rows[0]["groups"] for p in g["prefill_prefix_lens"]],
            compute_ms=[round(r["total_time_ms"], 3) for r in rows],
            send_ms=[round(rows[s + 1]["wall_start_ms"] - rows[s]["wall_start_ms"] - rows[s]["total_time_ms"], 3)
                     for s in range(3)]))
    return out


assert all(len(s) == len(stages[0]) for s in stages), "every microbatch passes all 4 stages"
sessions = collections.defaultdict(list)          # session id -> [(row index, trace row)] in round order
for i, r in enumerate(trace):
    sessions[int(r["session_id"])].append((i, r))
cases = collections.defaultdict(list)
spans = {}                                        # session id -> (start, end) in ms
for sid, rows in sessions.items():
    assert [int(r["round_idx"]) for _, r in rows] == list(range(len(rows))), sid
    s = [slo[i] for i, _ in rows]
    # No other session's microbatch may run inside this one.
    spans[sid] = (s[0]["arrival_time_ms"], s[-1]["arrival_time_ms"] + s[-1]["ttft_ms"])
    prefix, computed = int(rows[0][1]["input_len"]), int(rows[1][1]["input_len"]) + 1
    run = dict(round0=dict(ttft_ms=round(s[0]["ttft_ms"], 3),
                           microbatches=len(microbatches(s[0]["arrival_time_ms"],
                                                         s[0]["arrival_time_ms"] + s[0]["ttft_ms"]))))
    for k in range(1, len(rows)):
        assert int(rows[k][1]["input_len"]) + 1 == computed, sid
        run[f"round{k}"] = dict(ttft_ms=round(s[k]["ttft_ms"], 3), hit_tokens=s[k]["prefix_cache_hit_tokens"],
                                computed_tokens=s[k]["prefill_processed"],
                                microbatches=microbatches(s[k]["arrival_time_ms"],
                                                          s[k]["arrival_time_ms"] + s[k]["ttft_ms"]))
    cases[(prefix, computed, len(rows))].append(run)
order = sorted(spans)
for a, b in zip(order, order[1:]):
    assert spans[a][1] <= spans[b][0], ("sessions overlap", a, b)

out = []
for (prefix, computed, rounds), runs in sorted(cases.items()):
    assert all(r == runs[0] for r in runs), (prefix, computed)
    r = runs[0]
    for k in range(1, rounds):
        hit = prefix + (k - 1) * computed
        assert r[f"round{k}"]["hit_tokens"] == hit and r[f"round{k}"]["computed_tokens"] == computed, (prefix, k)
    out.append(dict(prefix_tokens=prefix, computed_tokens=computed, sessions=len(runs),
                    **({"rounds": rounds} if rounds > 2 else {}), **r))
    line = f"P {prefix:7} A {computed:5}  round0 {r['round0']['ttft_ms']:9.3f} ms ({r['round0']['microbatches']} mb)"
    for k in range(1, rounds):
        mb = r[f"round{k}"]["microbatches"]
        line += (f"  r{k} {r[f'round{k}']['ttft_ms']:8.3f} ms  stages " + " ".join(f"{x:7.3f}" for x in mb[0]["compute_ms"])
                 + f"  sends {mb[0]['send_ms'][0]:.3f} ({len(mb)} mb)")
    print(line)
args.out.write_text(json.dumps(out, indent=1) + "\n")
print(args.out)

# Compressed trace: session k starts gap_ms after session k-1's last token (its simulated duration).
start, header = 0.0, list(trace[0].keys())
with open(args.compressed, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for sid in order:
        for _, r in sessions[sid]:
            w.writerow([f"{start:.6f}" if h == "arrival_time_ms" else r[h] for h in header])
        start += spans[sid][1] - spans[sid][0] + args.gap_ms
end = start - args.gap_ms
print(f"{args.compressed}: {len(order)} sessions, last session ends at {end / 1e3:.1f} s ({end / 60e3:.2f} min)")
