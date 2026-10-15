"""What the node-check misses mean for a stage: per-stage time error at one prompt length.

For every kernel call in data/layer_targets.jsonl (one request of L tokens, one microbatch, by stage), find its
configuration (kind, backend, fixed fields, via the call's position in data/kernel_specs.jsonl) and the three
node-check rows of that configuration (measured/kernels_nodecheck.jsonl). Then, per stage:
  estimate  sum of total_ms x (ratio - 1), with the ratio of the sampled row nearest in log(time) to the call;
  misses    the part of the estimate from calls whose nearest row is outside +-10%;
  bound     sum of total_ms x max |ratio - 1| over the configuration's three rows (worst case, any size);
  uncovered the time in kernels with no node-check row (kinds outside the 35 configurations, e.g. elementwise).
Prints a markdown table.

usage: python stage_bound.py <handoff data dir> <rows.jsonl>[,<more rows.jsonl>] [lengths, default 8192,131072]
  rows files: kernels_nodecheck.jsonl (3 sampled rows per configuration) and/or kernels.jsonl (every row of the
  re-measured configurations); a configuration found in a later file replaces the earlier file's rows.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sample_specs import config_key  # noqa: E402


def main(data, nodecheck, lengths="8192,131072"):
    data = Path(data)
    pos_cfg = {}
    for line in open(data / "kernel_specs.jsonl"):
        row = json.loads(line)
        for p in row.get("positions", []):
            pos_cfg[(row["kind"], row["backend"], p)] = config_key(row)
    samples = {}
    for path in nodecheck.split(","):
        found = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            if r["ratio"] is not None:
                found[config_key(r)].append((r["db_time_ms"], r["ratio"]))
        samples.update(found)
    targets = [json.loads(l) for l in open(data / "layer_targets.jsonl")]
    print("| Prompt | Stage | Stage ms | Estimate ms (%) | From misses ms (%) | Worst-ratio bound ms (%) | Uncovered ms (%) | Largest estimate term |")
    print("|---|---|---|---|---|---|---|---|")
    for L in (int(x) for x in lengths.split(",")):
        for stage in range(4):
            layers = [t for t in targets if t["prompt_tokens"] == L and t["stage"] == stage]
            stage_ms = sum(t["count"] * t["layer_ms"] for t in layers)
            est = miss = bound = unc = 0.0
            terms = defaultdict(float)
            for t in layers:
                for k in t["kernels"]:
                    tot = k["total_ms"] * t["count"]
                    cfg = pos_cfg.get((k["kind"], k["backend"], f"stage {stage}: {k['position']}"))
                    rows = samples.get(cfg) if cfg else None
                    if not rows:
                        unc += tot
                        continue
                    near = min(rows, key=lambda r: abs(math.log(r[0]) - math.log(max(k["time_ms"], 1e-9))))
                    e = tot * (near[1] - 1)
                    est += e
                    if abs(near[1] - 1) > 0.10:
                        miss += e
                    terms[f"{k['kind']}/{k['backend']}"] += e
                    bound += tot * max(abs(r[1] - 1) for r in rows)
            top = max(terms.items(), key=lambda kv: abs(kv[1])) if terms else ("-", 0)
            pct = lambda v: f"{v:+.3f} ({100 * v / stage_ms:+.2f}%)"
            print(f"| {L // 1024}k | {stage} | {stage_ms:.2f} | {pct(est)} | {pct(miss)} | {bound:.3f} ({100 * bound / stage_ms:.2f}%) | "
                  f"{unc:.3f} ({100 * unc / stage_ms:.1f}%) | `{top[0]}` {top[1]:+.3f} ms |")


if __name__ == "__main__":
    main(*sys.argv[1:])
