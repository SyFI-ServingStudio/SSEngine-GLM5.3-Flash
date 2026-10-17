"""Tier bring-up summary (7_offloading.md §1, TIERS.md §10): per P, round-1 TTFT and the tier read time against the
v2-realistic targets, with what the read moved and when.

    python -m bench.tiers measured/runs_m6/tier_bringup_dram [--target dram] [--reference RUN_DIR]

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
                   "tiers": sorted({str(r["tier"]) for r in rs}),
                   "cached_all_P": all(r["cached"] == P for r in rs),
                   "target_ttft_ms": tgt[0] if tgt else None, "target_read_ms": tgt[1] if tgt else None}
    return {"run": str(run_dir), "target": target, "by_P": by_p, "sessions": sessions}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--target", choices=VARIANTS)
    ap.add_argument("--reference", type=Path, help="a run whose round-1 tokens to compare (same prompts)")
    a = ap.parse_args()
    out = summarize(a.run_dir, a.target, a.reference)
    (a.run_dir / "tiers_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'P':>7} {'n':>2} {'TTFT ms':>9} {'target':>7} {'read ms':>8} {'target':>7}  tiers")
    for P, r in out["by_P"].items():
        print(f"{P:>7} {r['sessions']:>2} {r['round1_ttft_ms']:>9} {str(r['target_ttft_ms']):>7} "
              f"{str(r['read_ms']):>8} {str(r['target_read_ms']):>7}  {','.join(r['tiers'])}"
              f"{'' if r['cached_all_P'] else '  (not all P cached)'}")


if __name__ == "__main__":
    main()
