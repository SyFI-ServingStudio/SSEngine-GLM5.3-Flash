"""Summary of a vLLM DP/EP session run (bench/slurm/vllm_dpep.sbatch): the handoff metrics (data/metrics.py, minutes
0-20, the numbers our M6 tables quote) and the same numbers over minutes 10-20 (bench.tiers' window), failed rounds,
and from vLLM's /metrics snapshots (prom/*.prom, one a minute) the counter deltas over the client's minutes 10-20:
local prefix cache and connector (CPU offload) hits, preemptions, offload transfer bytes, plus the peak queue.

    python -m bench.vllm_dpep_summary measured/vllm_dpep/<run> [--trace CSV]    # writes <run>/dpep_summary.json and prints it
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


from bench.prefix import real_arrays, window_metrics

TRACE = Path("/raid/kanzhu/ServingStudio/data/tracelab-v3/m6/sessions_2ps_v3.csv")
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


def inside_rounds(run: Path, trace: Path = TRACE) -> dict:
    """Rounds arriving in minutes 10-20 (successful, first token < 20 min) split into "inside" rounds (round > 0 whose
    declared 0 < prefix_len < the previous round's prefix + input + output: a context cutoff, resumable only from a
    state inside the old context) and the rest ("extending"; includes prefix 0). Per class: rounds, prefix tokens,
    tokens the server hit (cached_prompt_tokens), new tokens (input_len) and recompute = prompt - cached - new;
    recompute shares are over all new tokens."""
    ctx = {(r["session_id"], int(r["round_idx"])): int(r["prefix_len"]) + int(r["input_len"]) + int(r["output_len"])
           for r in csv.DictReader(trace.open())}
    recs = [json.loads(x) for x in (run / "requests.jsonl").read_text().splitlines()]
    t0 = min(r["outcome"]["submit_timestamp"] for r in recs)
    out = {k: {"rounds": 0, "prefix_tokens": 0, "hit_tokens": 0, "new_tokens": 0, "recompute_tokens": 0}
           for k in ("inside", "extending")}
    inside_hit_rounds = 0
    for r in recs:
        o, d = r["outcome"], r["source"]["data"]
        if not (600e3 <= (o["submit_timestamp"] - t0) * 1e3 < 1200e3 and str(o.get("status")).upper() == "SUCCESS"
                and o.get("first_token_id_ms") is not None
                and (o["post_timestamp"] - t0) * 1e3 + o["first_token_id_ms"] < 1200e3):
            continue
        u = o.get("server_usage") or {}
        k, p, c = int(d["round_idx"]), d.get("prefix_len") or 0, u.get("cached_prompt_tokens") or 0
        cls = "inside" if k > 0 and 0 < p < ctx[(str(d["session_id"]), k - 1)] else "extending"
        inside_hit_rounds += cls == "inside" and c > 0
        b = out[cls]
        b["rounds"] += 1
        b["prefix_tokens"] += p
        b["hit_tokens"] += c
        b["new_tokens"] += d["input_len"]
        b["recompute_tokens"] += (u.get("prompt_tokens") or 0) - c - d["input_len"]
    new = sum(b["new_tokens"] for b in out.values())
    return {**out, "inside_rounds_with_hit": inside_hit_rounds,
            "recompute_over_new": sum(b["recompute_tokens"] for b in out.values()) / max(new, 1),
            "inside_recompute_over_new": out["inside"]["recompute_tokens"] / max(new, 1),
            "extending_recompute_over_new": out["extending"]["recompute_tokens"] / max(new, 1)}


def main(run: Path, trace: Path = TRACE) -> dict:
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
           "metrics_arrival_10_20": m1020, "rounds_sent": sent, "rounds_failed": failed, "inside_rounds": inside_rounds(run, trace),
           "prom": counters(run)}
    (run / "dpep_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--trace", type=Path, default=TRACE, help="the run's session trace (inside-round classification)")
    a = ap.parse_args()
    print(json.dumps(main(a.run, a.trace), indent=1))
