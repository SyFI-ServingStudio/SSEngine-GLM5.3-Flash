"""M6 retention eval: per-window metrics of real runs and the matching simulator runs (docs/results/eval_m6.md).

    <python with pyarrow> benchmarks/eval_table.py real:<label>=<run dir> ... sim:<label>=<sim run dir> ...

Windows [0, 20) and [10, 20) min by first token (handoff data/metrics.py: t = 0 at the first request; a round counts
when its first token comes inside the window). Per window: rounds/s; new tok/s (trace input_len / fresh prompt tokens);
computed tok/s (real: prompt - server-reported cached; sim: prefill_processed); recompute/new = (computed - new) / new;
TTFT p50/p90/p99 and rounds over 60 s of the counted rounds; hit mix = shares of the counted rounds that had a previous
context, by where it came from: HBM / DRAM / SSD, and lost (recomputed from nothing; % of those rounds); stage-0 busy
= stage-0 compute time inside the window / window (real: microbatches.jsonl; sim: raw/cost_log/worker_stage_0.parquet).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

WINDOWS = ((0.0, 20.0), (10.0, 20.0))


def real(run: Path) -> dict:
    rows = [json.loads(x) for x in (run / "requests.jsonl").read_text().splitlines() if x.strip()]
    t0c = min(r.get("outcome", r)["submit_timestamp"] for r in rows) * 1e3
    t0s = json.loads((run / "t0.json").read_text())["t0_unix_ns"] / 1e6
    shift = t0c - t0s                      # server ms -> client ms: subtract this
    arr, first, new, comp, rid = [], [], [], [], []
    for r in rows:
        o = r.get("outcome", r)
        if o.get("first_token_id_ms") is None or o.get("post_timestamp") is None or str(o.get("status")).upper() != "SUCCESS":
            continue
        u = o.get("server_usage") or {}
        arr.append(o["submit_timestamp"] * 1e3 - t0c)
        first.append(o["post_timestamp"] * 1e3 + o["first_token_id_ms"] - t0c)
        new.append(r["source"]["data"]["input_len"])
        comp.append((u.get("prompt_tokens") or 0) - (u.get("cached_prompt_tokens") or 0))
        rid.append(o["request_id"])
    kind = {}
    for x in (run / "server_requests.jsonl").read_text().splitlines():
        s = json.loads(x)
        k = s.get("hit_kind")
        # a tier read: hit_kind "tier", or "evicted" (HBM copy gone) with the context served from a tier
        tier = {"dram": "D", "ssd": "S"}.get(s.get("tier")) if (s.get("tier_tokens") or 0) > 0 else None
        kind[s["id"]] = tier if k in ("tier", "evicted") and tier else {"full": "H", "evicted": "L"}.get(k)
    busy = []
    for x in (run / "microbatches.jsonl").read_text().splitlines():
        m = json.loads(x)
        if m["stage"] == 0:
            busy.append((m["start_ms"] - shift, m["compute_ms"]))
    return {"arr": np.array(arr), "first": np.array(first), "new": np.array(new, float), "comp": np.array(comp, float),
            "kind": np.array([kind.get(i) for i in rid], object), "busy": np.array(busy)}


def sim(run: Path) -> dict:
    import pyarrow.parquet as pq
    t = pq.read_table(run / "raw/request_slo.parquet", columns=["request_id", "completed", "arrival_time_ms",
                      "finish_decode_time_ms", "ttft_ms", "prefill_processed", "fresh_prompt_tokens",
                      "declared_prefix_tokens", "prefix_cache_hit_tokens"]).to_pydict()
    c = np.array(t["completed"])
    # no tiers (no prefix_tiers_w0.csv): a round with a previous context hits HBM or is lost
    kind = {i: ("H" if (h or 0) > 0 else "L") for i, p, h in
            zip(t["request_id"], t["declared_prefix_tokens"], t["prefix_cache_hit_tokens"]) if p > 0}
    tiers = run / "prefix_tiers_w0.csv"
    with open(tiers) if tiers.exists() else open("/dev/null") as f:
        for r in csv.DictReader(f):
            if int(r["declared_tokens"]) == 0:
                continue
            k = {"dram": "D", "ssd": "S"}.get(r["tier"]) or ("H" if int(r["hbm_tokens"]) > 0 else "L")
            kind[int(r["request_id"])] = k
    cl = pq.read_table(run / "raw/cost_log/worker_stage_0.parquet", columns=["wall_start_ms", "total_time_ms"]).to_pydict()
    arr = np.array(t["arrival_time_ms"], float)[c]
    return {"arr": arr, "first": arr + np.array([x or 0.0 for x in t["ttft_ms"]], float)[c],
            "new": np.array(t["fresh_prompt_tokens"], float)[c], "comp": np.array(t["prefill_processed"], float)[c],
            "kind": np.array([kind.get(i) for i in np.array(t["request_id"])[c]], object),
            "busy": np.column_stack([cl["wall_start_ms"], cl["total_time_ms"]])}


def window(d: dict, a: float, b: float) -> dict:
    lo, hi = a * 60e3, b * 60e3
    w = (hi - lo) / 1e3
    m = (d["first"] >= lo) & (d["first"] < hi)
    ttft = (d["first"] - d["arr"])[m] / 1e3
    new, comp = d["new"][m].sum(), d["comp"][m].sum()
    k = d["kind"][m]
    n = max(sum(k == x for x in "HDSL").sum(), 1) if len(k) else 1
    s, e = d["busy"][:, 0], d["busy"][:, 0] + d["busy"][:, 1]
    busy = np.clip(np.minimum(e, hi) - np.maximum(s, lo), 0, None).sum() / (hi - lo)
    return {"rps": m.sum() / w, "new": new / w, "comp": comp / w, "rec": (comp - new) / new,
            "p50": np.percentile(ttft, 50), "p90": np.percentile(ttft, 90), "p99": np.percentile(ttft, 99),
            "over60": int((ttft > 60).sum()), **{x: 100 * (k == x).sum() / n for x in "HDSL"}, "busy": busy}


def main() -> None:
    print("| run | window | r/s | new tok/s | computed tok/s | recompute/new | TTFT p50/p90/p99 (s) | >60 s | hit H/D/S (lost) % | s0 busy |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        side, rest = arg.split(":", 1)
        label, path = rest.split("=", 1)
        d = (real if side == "real" else sim)(Path(path))
        for a, b in WINDOWS:
            r = window(d, a, b)
            print(f"| {side} {label} | {a:.0f}-{b:.0f} | {r['rps']:.2f} | {r['new'] / 1e3:.1f}k | {r['comp'] / 1e3:.1f}k | "
                  f"{r['rec']:.3f} | {r['p50']:.2f}/{r['p90']:.2f}/{r['p99']:.2f} | {r['over60']} | "
                  f"{r['H']:.1f}/{r['D']:.1f}/{r['S']:.1f} ({r['L']:.1f}) | {r['busy']:.3f} |")


if __name__ == "__main__":
    main()
