"""Tier bring-up summary (7_offloading.md §1, TIERS.md §10): per P, round-1 TTFT and the tier read time against the
v2-realistic targets, with what the read moved and when.

    python -m bench.tiers measured/runs_m6/tier_bringup_dram [--target dram] [--reference RUN_DIR]
    python -m bench.tiers RUN_DIR --setting full_dram --n 400     # a session run (step 1b / 2b)

Per session (bench/prefix.py `bringup`: client TTFT, server TTFT, cached tokens, per-stage compute) plus, from
server_requests.jsonl, the tier, tokens read and the read window (read start -> all four TierDone), and from
tier_ops.jsonl the read op's per-GPU copy windows and bytes. Means per P go next to the target's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.common import HANDOFF
from bench.prefix import bringup, server_rows
from bench.report import read_jsonl

VARIANTS = ("hbm_hit", "hbm", "dram", "ssd", "dram_unbal", "ssd_unbal")


def targets(variant: str) -> dict[int, tuple[float, float]]:
    """P -> (round-1 TTFT ms, read ms) of the handoff's data/tier_targets.json (v2-realistic overlay: the coordinator's
    tiercheck runs, TIERS.md §10)."""
    rows = json.loads((HANDOFF / "data/tier_targets.json").read_text())[variant]
    return {r["prefix_tokens"]: (r["round1_ttft_ms"], r["read_ms"]) for r in rows}


def read_parts(o: dict) -> dict | None:
    """Where a read op's time went, on the GPU that reported last: sent -> that worker's tier thread took the op
    (`recv`), its copies' submission (`submit`, overlaps `gpu`), the copies' GPU time from a mark recorded as submission
    began (`gpu`), GPU end -> its TierDone left (`notify`: done thread wake + send), last TierDone -> landed in the
    scheduler (`land`). None for ops logged without the GPU time."""
    spans = o.get("gpu_span_ms")
    if not spans or any(x is None for x in spans):
        return None
    g = max(range(len(o["gpu_ms"])), key=lambda i: o["gpu_ms"][i][1])
    (a, b), span = o["gpu_ms"][g], spans[g]
    return {"recv": round(a - o["sent_ms"], 3), "submit": o["submit_ms"][g], "gpu": span,
            "notify": round(b - a - span, 3), "land": round(o["landed_ms"] - b, 3)}


def summarize(run_dir: Path, target: str | None, reference: Path | None = None) -> dict:
    lines = [x for x in bringup(run_dir, reference) if x.get("round1_ttft_ms") is not None]
    sreq = server_rows(run_dir)
    reads = {o["req"]: o for o in read_jsonl(run_dir / "tier_ops.jsonl") if o["kind"] == "read"} \
        if (run_dir / "tier_ops.jsonl").exists() else {}
    client_ids = {}
    for r in read_jsonl(run_dir / "requests.jsonl"):
        d = r["source"]["data"]
        client_ids[(int(d["session_id"]), int(d["round_idx"]))] = r["outcome"]["request_id"]
    sessions = []
    for x in lines:
        rid = client_ids.get((x["session"], 1))
        s = sreq.get(rid, {})
        o = reads.get(rid)
        row = {"session": x["session"], "P": x["prefix_tokens"], "round0_ttft_ms": x["round0_ttft_ms"],
               "round1_ttft_ms": x["round1_ttft_ms"], "round1_server_ttft_ms": x["round1_server_ttft_ms"],
               "cached": x["round1_cached_tokens"], "computed": x["computed_tokens"], "tier": s.get("tier"),
               "tier_tokens": s.get("tier_tokens"), "stages_ms": [st["compute_ms"] for st in x["round1_stages"] or []],
               "token_matches_reference": x["token_matches_reference"]}
        if s.get("read_start_ms") is not None:
            row["read_ms"] = round(s["read_land_ms"] - s["read_start_ms"], 3)
            row["wait_before_read_ms"] = round(s["read_start_ms"] - s["arrival_ms"] - s["host_ms"]["admitted"], 3) \
                if s.get("host_ms") else None
        if o is not None:
            row["read_gpu_ms"] = [round(b - a, 3) for a, b in o["gpu_ms"]]
            row["read_bytes_per_gpu"] = o["bytes"]
            row["read_gb_s_per_gpu"] = [round(n / max(b - a, 1e-6) / 1e6, 2)
                                        for n, (a, b) in zip(o["bytes"], o["gpu_ms"])]
            row["read_parts_ms"] = read_parts(o)
        sessions.append(row)
    by_p = {}
    for P in sorted({r["P"] for r in sessions}):
        rs = [r for r in sessions if r["P"] == P]
        tgt = targets(target).get(P) if target else None
        reads_ms = [r["read_ms"] for r in rs if "read_ms" in r]
        by_p[P] = {"sessions": len(rs), "round1_ttft_ms": round(float(np.mean([r["round1_ttft_ms"] for r in rs])), 2),
                   "round1_server_ttft_ms": round(float(np.mean([r["round1_server_ttft_ms"] for r in rs
                                                                 if r["round1_server_ttft_ms"] is not None])), 2),
                   "read_ms": round(float(np.mean(reads_ms)), 3) if reads_ms else None,
                   "read_parts_ms": mean_parts([r.get("read_parts_ms") for r in rs]),
                   "tiers": sorted({str(r["tier"]) for r in rs}),
                   "cached_all_P": all(r["cached"] == P for r in rs),
                   "target_ttft_ms": tgt[0] if tgt else None, "target_read_ms": tgt[1] if tgt else None}
    return {"run": str(run_dir), "target": target, "by_P": by_p, "sessions": sessions}


def session_vs_target(run_dir: Path, setting: str, n: int) -> dict:
    """A session run against data/session_targets.json (setting, N): the client metrics, stage busy, where the hit
    tokens came from (rounds arriving in minutes 10-20, as the targets' prefix_read_share: HBM-resident vs read from
    DRAM / SSD, as shares of all hit tokens) and the scheduler's and tiers' counters."""
    from bench.prefix import client_rows, sessions, shift_ms

    s = sessions(run_dir)
    tgt = {(t["setting"], t["max_sessions"]): t for t in json.loads((HANDOFF / "data/session_targets.json")
                                                                   .read_text())}.get((setting, n))
    lo = shift_ms(run_dir, client_rows(run_dir)) + 600e3
    got = {"hbm": 0, "dram": 0, "ssd": 0}
    for r in server_rows(run_dir).values():
        if lo <= r.get("arrival_ms", -1) < lo + 600e3 and r.get("cached"):
            t = r.get("tier_tokens", 0) if r.get("tier") else 0
            got["hbm"] += r["cached"] - t
            if t:
                got[r["tier"]] += t
    hit = max(sum(got.values()), 1)
    share = {k: v / hit for k, v in got.items()}
    want = None
    if tgt:
        w = tgt["prefix_read_share"]
        want = {k: w[k] / max(sum(w.values()), 1e-12) for k in w}
    cache = json.loads((run_dir / "cache.json").read_text())
    reads = [o for o in read_jsonl(run_dir / "tier_ops.jsonl") if o["kind"] == "read"] \
        if (run_dir / "tier_ops.jsonl").exists() else []
    by_tier = {}
    for o in reads:
        by_tier.setdefault(o["tier"], []).append(o)
    read_stats = {t: {"reads": len(os_), "read_ms": {f"p{q}": round(float(np.percentile(
                      [o["landed_ms"] - o["sent_ms"] for o in os_], q)), 3) for q in (50, 90, 99)},
                      "parts_ms": mean_parts([read_parts(o) for o in os_])} for t, os_ in by_tier.items()}
    return {"run": run_dir.name, "reads": read_stats, "setting": setting, "n": n, "metrics": s["metrics"], "stage_busy": s["stage_busy"],
            "server_ttft_s": s["server_ttft_s"], "hit_share_by_tier": share, "target_hit_share_by_tier": want,
            "scheduler": cache.get("scheduler"), "blocked_by_write": cache.get("blocked_by_write"),
            "tiers": cache.get("tiers"), "target": tgt, "rounds_failed": s["rounds_failed"],
            "in_run_jit": s["in_run_jit"]}


def print_session(x: dict):
    m, t = x["metrics"], x["target"] or {}
    tt, wt = m["ttft_s"], t.get("ttft_s", {})
    print(f"{x['run']} vs {x['setting']} N={x['n']}")
    print(f"  rounds/s {m['rounds_per_s']:.2f} vs {t.get('rounds_per_s', float('nan')):.2f}, new tok/s "
          f"{m['new_tokens_per_s'] / 1e3:.1f}k vs {t.get('new_tokens_per_s', float('nan')) / 1e3:.1f}k")
    print("  TTFT p50/p90/p99 s " + " / ".join(f"{tt[k]:.2f}" for k in ("p50", "p90", "p99")) + " vs "
          + " / ".join(f"{wt.get(k, float('nan')):.2f}" for k in ("p50", "p90", "p99"))
          + f", >60 s {m.get('rounds_ttft_over_60s')} vs {t.get('rounds_ttft_over_60s')}")
    print("  stage busy " + " / ".join(f"{100 * b:.0f}" for b in x["stage_busy"]) + "% vs "
          + " / ".join(f"{100 * b:.0f}" for b in t.get("stage_busy", [])) + "%")
    w = x["target_hit_share_by_tier"] or {}
    print("  hit tokens by tier " + ", ".join(f"{k} {100 * v:.2f}% vs {100 * w.get(k, float('nan')):.2f}%"
                                              for k, v in x["hit_share_by_tier"].items()))
    for t, r in x["reads"].items():
        print(f"  {t} reads {r['reads']}: sent -> landed ms p50/p90/p99 " + " / ".join(str(v) for v in r["read_ms"].values())
              + (", parts " + ", ".join(f"{k} {v}" for k, v in r["parts_ms"].items()) if r["parts_ms"] else ""))
    print(f"  scheduler {x['scheduler']}, blocked_by_write {x['blocked_by_write']}, failed {x['rounds_failed']}")


def mean_parts(parts: list) -> dict | None:
    parts = [p for p in parts if p]
    return {k: round(float(np.mean([p[k] for p in parts])), 3) for k in parts[0]} if parts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--target", choices=VARIANTS)
    ap.add_argument("--reference", type=Path, help="a run whose round-1 tokens to compare (same prompts)")
    ap.add_argument("--setting", help="a session run: its data/session_targets.json setting (e.g. full_dram)")
    ap.add_argument("--n", type=int, help="with --setting: max sessions")
    a = ap.parse_args()
    if a.setting:
        x = session_vs_target(a.run_dir, a.setting, a.n)
        (a.run_dir / "tiers_session.json").write_text(json.dumps(x, indent=1) + "\n")
        print_session(x)
        return
    out = summarize(a.run_dir, a.target, a.reference)
    (a.run_dir / "tiers_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'P':>7} {'n':>2} {'TTFT ms':>9} {'target':>7} {'read ms':>8} {'target':>7}  tiers")
    for P, r in out["by_P"].items():
        print(f"{P:>7} {r['sessions']:>2} {r['round1_ttft_ms']:>9} {str(r['target_ttft_ms']):>7} "
              f"{str(r['read_ms']):>8} {str(r['target_read_ms']):>7}  {','.join(r['tiers'])}"
              f"{'' if r['cached_all_P'] else '  (not all P cached)'}")
        if r["read_parts_ms"]:
            print(f"{'':>10} read parts ms: " + ", ".join(f"{k} {v}" for k, v in r["read_parts_ms"].items()))


if __name__ == "__main__":
    main()
