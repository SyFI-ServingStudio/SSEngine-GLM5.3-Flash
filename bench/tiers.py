"""Tier bring-up summary (7_offloading.md §1, TIERS.md §10): per P, round-1 TTFT and the tier read time against the
v2-realistic targets, with what the read moved and when.

    python -m bench.tiers measured/runs_m6/tier_bringup_dram [--target dram] [--reference RUN_DIR]
    python -m bench.tiers RUN_DIR --setting full_dram --n 400     # a session run (step 1b / 2b)
    python -m bench.tiers RUN_DIR --setting tiers --n 400         # a victim-chain setting (VICTIM_SETTINGS)

Per session (bench/prefix.py `bringup`: client TTFT, server TTFT, cached tokens, per-stage compute) plus, from
server_requests.jsonl, the tier, tokens read and the read window (read start -> all four TierDone), and from
tier_ops.jsonl the read op's per-GPU copy windows and bytes. Means per P go next to the target's.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from bench.common import HANDOFF, ROOT
from bench.prefix import bringup, server_rows
from bench.report import read_jsonl

VARIANTS = ("hbm_hit", "hbm", "dram", "ssd", "dram_unbal", "ssd_unbal")

# Victim-chain settings (HBM -> DRAM -> SSD, TIERS.md §5): name -> (sim run, HBM GB, `--pool-tokens`, DRAM GB per
# GPU); SSD 128 GB per GPU. Targets: the simulator's rows of these runs, copied from the sim run 20261017_7_victim_chain
# (PPIMPL_VICTIM_TARGETS overrides the path).
VICTIM_SETTINGS = {"full": ("full_d50", 100.18, None, 50.0),
                   "tiers": ("h20_d20", 20.0, 12_220_800, 20.0),
                   "ssd_heavy": ("h12p5_d12p5", 12.5, 7_632_640, 12.5)}
VICTIM_TARGETS = Path(os.environ.get("PPIMPL_VICTIM_TARGETS") or ROOT / "measured" / "victim_chain_rows.jsonl")


def victim_target(setting: str, n: int, rows: Path = VICTIM_TARGETS) -> dict | None:
    """The simulator's row of a victim setting at N = `n`, in data/session_targets.json's keys. prefix_read_share:
    the hit tokens' tiers (HBM / DRAM / SSD) of minutes 10-20 (the row's 2-minute windows 5-9, weighted by rounds);
    the rows carry no stage busy."""
    run = VICTIM_SETTINGS[setting][0]
    for line in rows.read_text().splitlines():
        r = json.loads(line)
        if r["run"].rstrip("/").split("/")[-2:] == [run, f"c{n}"]:
            w = np.array(r["win_rounds"][5:10], float)
            split = (np.array(r["win_tok_split"][5:10]) * w[:, None]).sum(0) / w.sum()
            return {"setting": setting, "max_sessions": n, "rounds_per_s": r["rounds_per_s"],
                    "new_tokens_per_s": r["new_tokens_per_s"], "prefix_hit_share": r["prefix_hit_share"],
                    "ttft_s": r["ttft"], "rounds_ttft_over_60s": r["rounds_ttft_over_60s"],
                    "prefix_read_share": dict(zip(("hbm", "dram", "ssd"), map(float, split[:3]))),
                    "traffic": r["traffic"], "source": f"{rows}:{r['run']}"}
    return None


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
    tgt = victim_target(setting, n) if setting in VICTIM_SETTINGS else \
        {(t["setting"], t["max_sessions"]): t for t in json.loads((HANDOFF / "data/session_targets.json")
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
    reads = [o for o in read_jsonl(run_dir / "tier_ops.jsonl") if o["kind"] in ("read", "fetch")] \
        if (run_dir / "tier_ops.jsonl").exists() else []
    by_tier = {}
    for o in reads:                     # staged loading: an SSD read is a fetch, then a hop (a DRAM read)
        by_tier.setdefault(o["tier"] if o["kind"] == "read" else "ssd fetch", []).append(o)
    rounds = {}                         # a tier round's whole read, start -> landed in HBM (staged: fetch + hop)
    for r in server_rows(run_dir).values():
        if r.get("read_start_ms") is not None:
            rounds.setdefault(r["tier"], []).append(r["read_land_ms"] - r["read_start_ms"])
    read_stats = {t: {"reads": len(os_), "read_ms": {f"p{q}": round(float(np.percentile(
                      [o["landed_ms"] - o["sent_ms"] for o in os_], q)), 3) for q in (50, 90, 99)},
                      "parts_ms": mean_parts([read_parts(o) for o in os_])} for t, os_ in by_tier.items()}
    read_stats |= {f"{t} rounds": {"reads": len(v), "read_ms": {f"p{q}": round(float(np.percentile(v, q)), 3)
                                                              for q in (50, 90, 99)}, "parts_ms": None}
                   for t, v in rounds.items()}
    plans = read_jsonl(run_dir / "plans.jsonl") if (run_dir / "plans.jsonl").exists() else []
    area = {}                           # staged loading: loading area use per microbatch (HBM blocks, DRAM slots)
    for k in ("area_held", "dram_area_used"):
        v = [p[k] for p in plans if p.get(k) is not None]
        if v:
            area[k] = {"mean": round(float(np.mean(v)), 1), "p90": round(float(np.percentile(v, 90)), 1),
                       "max": int(max(v))}
    return {"run": run_dir.name, "reads": read_stats, "area_use": area or None, "trace": trace_summary(run_dir), "setting": setting, "n": n, "metrics": s["metrics"], "stage_busy": s["stage_busy"],
            "server_ttft_s": s["server_ttft_s"], "hit_share_by_tier": share, "target_hit_share_by_tier": want,
            "scheduler": cache.get("scheduler"), "blocked_by_write": cache.get("blocked_by_write"),
            "write_back": cache.get("write_back"), "tiers": cache.get("tiers"), "target": tgt,
            "loading_area": cache.get("loading_area"), "rounds_failed": s["rounds_failed"],
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
    for k, t in (x.get("trace") or {}).items():
        if k == "probe":     # the member's GIL / CPU probe: counts, not per-op stats
            print("  probe (> 20 ms late): " + ", ".join(f"{f} {v}" for f, v in t.items()))
            continue
        print(f"  {k} trace ({t['ops']} ops x GPUs), p50 / mean: " + ", ".join(
            f"{f} {v['p50']} / {v['mean']}" for f, v in t.items() if f != "ops"))
    print(f"  scheduler {x['scheduler']}, blocked_by_write {x['blocked_by_write']}, failed {x['rounds_failed']}"
          + (f", loading area {x['loading_area']}" if x.get("loading_area") else ""))
    if x.get("area_use"):
        print(f"  loading area use per microbatch (mean/p90/max): {x['area_use']}")
    for lvl, w in (x.get("write_back") or {}).items():
        print(f"  {lvl} write-back: " + ", ".join(f"{k} {v}" for k, v in w.items()))


def trace_parts(op: dict) -> dict:
    """One traced op (tier_trace_s<stage>.jsonl) on one GPU, us: host-to-device (or device-to-host) copy time and
    GB/s, the NVLink hop's time, host time outside the batch calls (Python, and waiting for the GIL before a call),
    inside the calls on the CPU (driver) and blocked in them (the GIL back after the call, or the driver blocking; of
    it, runnable without a CPU: `call_runq_us`), the
    time the PCIe stream sat idle between its batches (ring waits or late submission) and how much of that idle was
    waiting for the host (the batch started within 30 us of its call returning), and the GPU span."""
    b = op["batches"]
    pcie = "read" if op["kind"] == "read" else "write"
    hb = [x for x in b if x["stream"] == pcie]
    nb = [x for x in b if x["stream"] != pcie]
    host_end = max(x["host_at_us"] + x["call_us"] for x in b)
    idle = host_wait = 0.0
    for prev, x in zip(hb, hb[1:]):
        gap = x["gpu_start_us"] - prev["gpu_end_us"]
        if gap > 0:
            idle += gap
            if x["gpu_start_us"] - (x["host_at_us"] + x["call_us"]) < 30:
                host_wait += min(gap, x["host_at_us"] + x["call_us"] - prev["gpu_end_us"])
    pcie_us = sum(x["gpu_end_us"] - x["gpu_start_us"] for x in hb)
    return {"pcie_us": round(pcie_us, 1), "pcie_gb_s": round(sum(x["bytes"] for x in hb) / max(pcie_us, 1e-3) / 1e3, 1),
            "nvl_us": round(sum(x["gpu_end_us"] - x["gpu_start_us"] for x in nb), 1),
            "host_outside_calls_us": round(host_end - sum(x["call_us"] for x in b), 1),
            "call_cpu_us": round(sum(x["cpu_us"] for x in b), 1),
            "call_blocked_us": round(sum(x["call_us"] - x["cpu_us"] for x in b), 1),
            "call_runq_us": round(sum(x.get("runq_us", 0.0) for x in b), 1),     # of it: runnable, no CPU
            "call_max_us": round(max(x["call_us"] for x in b), 1),
            "pcie_idle_us": round(idle, 1), "pcie_idle_waiting_host_us": round(host_wait, 1),
            "span_us": round(max(x["gpu_end_us"] for x in b) - min(x["gpu_start_us"] for x in b), 1)}


def trace_summary(run_dir: Path) -> dict | None:
    """Per kind (read / write), p50 and mean of trace_parts over every traced op on every GPU. The trace holds 1 DRAM
    op in `every` (PPIMPL_TIER_TRACE; older traces: every op): `ops` counts the ops it stands for, `traced` the lines;
    counts and totals of tier ops come from tier_ops.jsonl, not from here."""
    rows: dict[str, list] = {}
    every: dict[str, int] = {}
    for f in sorted(run_dir.glob("tier_trace_s*.jsonl")):
        for op in read_jsonl(f):
            rows.setdefault(op["kind"], []).append(trace_parts(op))
            every[op["kind"]] = every.get(op["kind"], 0) + op.get("every", 1)
    if not rows:
        return None
    out = {k: {"ops": every[k], "traced": len(v), **{f: {"p50": round(float(np.median([r[f] for r in v])), 1),
                                     "mean": round(float(np.mean([r[f] for r in v])), 1),
                                     "max": round(float(np.max([r[f] for r in v])), 1)} for f in v[0]}}
           for k, v in rows.items()}
    probes = [dict(p, stage=int(f.stem.rsplit("_s", 1)[1])) for f in sorted(run_dir.glob("tier_probe_s*.jsonl"))
              for p in read_jsonl(f)]
    if probes:   # the member's probe thread woke > 20 ms late: GIL held elsewhere (runq small) or no CPU (runq large)
        out["probe"] = {"late": len(probes), "over_100ms": sum(p["late_ms"] > 100 for p in probes),
                        "max_late_ms": max(p["late_ms"] for p in probes),
                        "runq_share": round(sum(p["runq_ms"] for p in probes) / sum(p["late_ms"] for p in probes), 3),
                        "by_stage": {s: sum(p["stage"] == s for p in probes) for s in sorted({p["stage"] for p in probes})}}
    return out


def mean_parts(parts: list) -> dict | None:
    parts = [p for p in parts if p]
    return {k: round(float(np.mean([p[k] for p in parts])), 3) for k in parts[0]} if parts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--target", choices=VARIANTS)
    ap.add_argument("--reference", type=Path, help="a run whose round-1 tokens to compare (same prompts)")
    ap.add_argument("--setting", help="a session run: its data/session_targets.json setting (e.g. full_dram) or a "
                    f"victim setting ({', '.join(VICTIM_SETTINGS)})")
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
