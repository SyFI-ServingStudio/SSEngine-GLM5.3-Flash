"""Summary of a vLLM DP/EP session run (bench/slurm/vllm_dpep.sbatch): the handoff metrics (data/metrics.py, minutes
0-20, the numbers our M6 tables quote) and the same numbers over minutes 10-20 (bench.tiers' window), failed rounds,
and from vLLM's /metrics snapshots (prom/*.prom, one a minute) the counter deltas over the client's minutes 10-20:
local prefix cache and connector (CPU offload) hits, preemptions, offload transfer bytes, plus the peak queue.

    python -m bench.vllm_dpep_summary measured/vllm_dpep/<run>     # writes <run>/dpep_summary.json and prints it
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


from bench.prefix import real_arrays, window_metrics

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$')


def prom(path: Path) -> dict[str, float]:
    """Metric name (labels other than engine/model dropped) -> value summed over engines."""
    out: dict[str, float] = defaultdict(float)
    for line in path.read_text().splitlines():
        m = LINE.match(line)
        if not m or line.startswith("#"):
            continue
        labels = ",".join(x for x in (m.group(2) or "").strip("{}").split(",")
                          if x and not x.startswith(("engine=", "model_name=")))
        out[m.group(1) + (f"{{{labels}}}" if labels else "")] += float(m.group(3))
    return out


def snapshot(run: Path, at_s: float) -> tuple[float, dict] | None:
    snaps = sorted((int(p.stem[1:]), p) for p in (run / "prom").glob("t*.prom"))
    if not snaps:
        return None
    t, p = min(snaps, key=lambda x: abs(x[0] - at_s))
    return t, prom(p)


def counters(run: Path) -> dict:
    a = snapshot(run, 600)
    end = run / "prom" / "end.prom"
    if a is None or not end.exists():
        return {}
    t, x = a
    y = prom(end)
    keys = [k for k in y if re.search(r"(prefix_cache|connector|offload|kv_transfer|preempt|external|cached|"
                                      r"prompt_tokens|generation_tokens|request_success|kv_bytes)", k)
            and not k.endswith(("_bucket", "_created"))]
    delta = {k: y[k] - x.get(k, 0.0) for k in sorted(keys)}
    peaks: dict[str, float] = {}
    for p in (run / "prom").glob("t*.prom"):
        for k, v in prom(p).items():
            if re.search(r"num_requests_(waiting|running)$|kv_cache_usage_perc$|cpu_cache_usage", k):
                peaks[k] = max(peaks.get(k, 0.0), v)
    return {"from_s": t, "to": "end", "delta": {k: v for k, v in delta.items() if v}, "peak_sum_over_engines": peaks}


def main(run: Path) -> dict:
    rows = real_arrays(run / "requests.jsonl")
    arr = rows[0]
    late = tuple(a[(arr >= 600e3)] for a in rows)            # rounds arriving in minutes 10-20, first token < 20 min
    m1020 = window_metrics(late, 1200e3)
    m1020 = {**m1020, "rounds_per_s": m1020["rounds"] / 600, "new_tokens_per_s": m1020["new_tokens_per_s"] * 2,
             "window_s": 600.0}
    sent = failed = 0
    for line in (run / "requests.jsonl").read_text().splitlines():
        o = json.loads(line).get("outcome", {})
        sent += 1
        failed += str(o.get("status")).upper() != "SUCCESS"
    out = {"run": run.name, "metrics_0_20": json.loads((run / "metrics.json").read_text()),
           "metrics_arrival_10_20": m1020, "rounds_sent": sent, "rounds_failed": failed, "prom": counters(run)}
    (run / "dpep_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
