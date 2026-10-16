"""6_scheduler.md checks of real runs (bench/run.py sched_check_<step>, force_check_<step>, sched_<step>_n<N>).

  python -m bench.sched check <run dir>                 one check run against its step's targets (bench/run.py runs it)
  python -m bench.sched summarize [runs root] [--out measured]
        measured/sched_check_<step>.jsonl, measured/force_check.jsonl, measured/scheduler.md

**Scheduling check** (§3, §7): stage 0's microbatches chunk for chunk against data/sched_check_targets.json (request,
first token, tokens; ids without req-frontend's `independent_` prefix), and each request's TTFT within ±10%.
**Force check** (§4, §7): the 32k TTFT, the 1k TTFT p50 / p90 / p99 / max, the last token (s from the first submit) and
the microbatch count against data/force_check_targets.json; the 1k submit rate (§6: 200 per second). Each run also
records stage time / fit by microbatch size (bench.prefix.stage_fit), since under overload a few percent of stage time
moves TTFT by several times that.
**Sessions** (§5): data/metrics.py's numbers, stage busy and stage time / fit against data/session_targets.json
(setting <step>_hbm), and the simulator at GPU time x1.06 (measured/sessions_sim/mult106.json) where it has the step.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from bench.common import HANDOFF, MEASURED
from bench.report import client_rows, read_jsonl

STEPS = ("plain", "srpt8k", "budget", "even", "good")
LABEL = {"plain": "0 plain", "srpt8k": "1 shortest first", "budget": "2 load budget", "even": "3 even split",
         "good": "4 force-schedule"}
# microbatch sizes for stage time / fit: the M4 classes and the load budget's larger microbatches
BINS = [(0, 256), (256, 1024), (1024, 8192), (8192, 8193), (8193, 32769)]
BIN_LABELS = [str(lo) if hi == lo + 1 else f"{lo}–{hi - 1}" for lo, hi in BINS]
ID_PREFIX = "independent_"


def _id(x: str) -> str:
    return x.removeprefix(ID_PREFIX)


def _near(a: float, b: float, rel: float = 0.10) -> bool:
    return abs(a / b - 1) <= rel


def stage0_microbatches(run_dir: Path) -> list[dict]:
    mbs = sorted((m for m in read_jsonl(run_dir / "microbatches.jsonl") if m["stage"] == 0), key=lambda m: m["index"])
    t0 = mbs[0]["start_ms"] if mbs else 0.0
    key = lambda c: (int(c[0].split("_")[-1]) if c[0].split("_")[-1].isdigit() else -1, c[0], c[1])
    return [{"index": m["index"], "start_ms": round(m["start_ms"] - t0, 3),
             "tokens": sum(q["append"] for q in m["requests"]),
             "chunks": sorted(([_id(q["id"]), q["prefix"], q["append"]] for q in m["requests"]), key=key)}
            for m in mbs]


def client_ttft_ms(run_dir: Path) -> dict[str, float]:
    return {_id(k): v["ttft_ms"] for k, v in client_rows(run_dir).items() if v["ok"]}


def check_sched(run_dir: Path, step: str) -> dict:
    target = json.loads((HANDOFF / "data/sched_check_targets.json").read_text())[step]
    got = stage0_microbatches(run_dir)
    ttft = client_ttft_ms(run_dir)
    diffs, first_diff = [], None
    for i, want in enumerate(target["microbatches"]):
        mine = got[i]["chunks"] if i < len(got) else None
        if mine != want["chunks"]:
            diffs.append(f"mb {i}: {mine} != target {want['chunks']}")
            first_diff = i if first_diff is None else first_diff
    if len(got) != len(target["microbatches"]):
        diffs.append(f"{len(got)} microbatches != target {len(target['microbatches'])}")
    dev = {}
    for k, want in target["ttft_ms"].items():
        if k not in ttft:
            diffs.append(f"TTFT {k}: no answer")
            continue
        dev[k] = ttft[k] / want - 1
        if abs(dev[k]) > 0.10:
            diffs.append(f"TTFT {k} {ttft[k]:.1f} ms vs target {want} ({100 * dev[k]:+.1f}%)")
    return {"kind": "sched_check", "step": step, "pass": not diffs, "diffs": diffs, "first_diff_mb": first_diff,
            "microbatches": got, "microbatch_count": len(got), "target_microbatches": len(target["microbatches"]),
            "ttft_ms": ttft, "target_ttft_ms": target["ttft_ms"],
            "max_ttft_dev": max((abs(v) for v in dev.values()), default=None)}


def force_metrics(run_dir: Path) -> dict:
    rows = {_id(k): v for k, v in client_rows(run_dir).items()}
    ok = {k: v for k, v in rows.items() if v["ok"]}
    t0 = min(v["submit_unix_ms"] for v in rows.values())
    short = np.array([v["ttft_ms"] for k, v in ok.items() if k != "force_long"]) / 1e3
    subs = np.sort([v["submit_unix_ms"] - t0 for k, v in rows.items() if k != "force_long"]) / 1e3
    span = subs[-1] - subs[0] if len(subs) > 1 else 0.0
    mbs = read_jsonl(run_dir / "microbatches.jsonl")
    return {"long_ttft_s": round(ok["force_long"]["ttft_ms"] / 1e3, 3) if "force_long" in ok else None,
            "short_ttft_s": {f"p{q}": round(float(np.percentile(short, q)), 3) for q in (50, 90, 99)}
            | {"max": round(float(short.max()), 3)},
            "last_token_s": round(max(v["submit_unix_ms"] - t0 + v["ttft_ms"] for v in ok.values()) / 1e3, 3),
            "microbatches": len({m["index"] for m in mbs}),
            "answered": len(ok), "requests": len(rows),
            "short_submit_rate_per_s": round((len(subs) - 1) / span, 1) if span else None,
            "short_submit_span_s": round(float(span), 2)}


def check_force(run_dir: Path, step: str) -> dict:
    want = json.loads((HANDOFF / "data/force_check_targets.json").read_text())[step]
    got = force_metrics(run_dir)
    diffs = []
    if got["answered"] != got["requests"]:
        diffs.append(f"{got['requests'] - got['answered']} requests without an answer")
    if got["long_ttft_s"] is None or not _near(got["long_ttft_s"], want["long_ttft_s"]):
        diffs.append(f"32k TTFT {got['long_ttft_s']} vs {want['long_ttft_s']}")
    if step == "good" and not (got["long_ttft_s"] is not None and 60 <= got["long_ttft_s"] <= 65):
        diffs.append(f"32k TTFT {got['long_ttft_s']} not in [60, 65] s")
    for p in ("p50", "p90", "p99"):
        if not _near(got["short_ttft_s"][p], want["short_ttft_s"][p]):
            diffs.append(f"1k {p} {got['short_ttft_s'][p]} vs {want['short_ttft_s'][p]}")
    if not _near(got["last_token_s"], want["last_token_s"]):
        diffs.append(f"last token {got['last_token_s']} vs {want['last_token_s']}")
    if got["short_submit_rate_per_s"] is None or not _near(got["short_submit_rate_per_s"], 200.0, 0.02):
        diffs.append(f"1k submit rate {got['short_submit_rate_per_s']}/s, not 200/s")
    return {"kind": "force_check", "step": step, "pass": not diffs, "diffs": diffs, **got, "target": want}


def check(run_dir: Path) -> dict:
    from bench.prefix import stage_fit
    from bench.run import in_run_jit

    meta = json.loads((run_dir / "run.json").read_text())
    kind, step = meta["kind"], meta.get("scheduler_step", "plain")
    out = check_sched(run_dir, step) if kind == "sched_check" else check_force(run_dir, step)
    out["run"] = run_dir.name
    out["stage_fit"] = stage_fit(run_dir, BINS)
    out["in_run_jit"] = in_run_jit(run_dir)
    out["uptime"] = [meta.get("uptime"), meta.get("uptime_after")]
    (run_dir / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def _fit_str(sf: dict) -> str:
    return "; ".join(f"s{s}: " + " / ".join("–" if c["ratio"] is None else f"x{c['ratio']:.3f}" for c in row)
                     for s, row in enumerate(sf["stages"]))


def _latest(root: Path, pattern: str) -> Path | None:
    """The run dir `<pattern>` or `<pattern>_<job>` with the highest job id that has run.json."""
    best = None
    for d in root.glob(pattern + "*"):
        m = re.fullmatch(re.escape(pattern) + r"(?:_(\d+))?", d.name)
        if m and (d / "run.json").exists():
            job = int(m.group(1) or 0)
            if best is None or job > best[0]:
                best = (job, d)
    return best[1] if best else None


# req-frontend frees a session's slot at its last round's token (as the simulator) since 365fba07, swapped in at
# 2026-10-16 16:11 PDT; session runs whose client started earlier held the slot through the last tool wait
SLOT_AT_LAST_TOKEN_FROM_NS = 1_792_192_260 * 10**9


def session_line(run_dir: Path, step: str, n: int) -> dict:
    from bench.prefix import sessions, stage_fit

    s = sessions(run_dir)
    s["stage_fit_bins"] = stage_fit(run_dir, BINS)
    targets = {(t["setting"], t["max_sessions"]): t for t in json.loads((HANDOFF / "data/session_targets.json")
                                                                        .read_text())}
    mult_path = MEASURED / "sessions_sim" / "mult106.json"
    mult = {(r["setting"], r["max_sessions"]): r for r in json.loads(mult_path.read_text())} \
        if mult_path.exists() else {}
    s.update(step=step, n=n, target=targets.get((f"{step}_hbm", n)), sim_x106=mult.get((f"{step}_hbm", n)),
             old_client=json.loads((run_dir / "t0.json").read_text())["t0_unix_ns"] < SLOT_AT_LAST_TOKEN_FROM_NS)
    return s


def _r(a, b):
    return f"{a / b:.2f}" if a is not None and b else "–"


def summarize(root: Path, out: Path) -> Path:
    parts = ["# The scheduler (6_scheduler.md)", "",
             "Real runs (bench/run.py, each on a freshly started server) against the handoff v2 targets. Written by "
             "`python -m bench.sched summarize`; the part from \"## Notes\" on is kept.", ""]
    # scheduling check
    rows = []
    for step in STEPS:
        d = _latest(root, f"sched_check_{step}")
        if d is None:
            continue
        c = check(d)
        rows.append(c)
        (out / f"sched_check_{step}.jsonl").write_text(
            "".join(json.dumps({k: m[k] for k in ("index", "start_ms", "tokens", "chunks")}) + "\n"
                    for m in c["microbatches"]) + json.dumps({"ttft_ms": c["ttft_ms"]}) + "\n")
    if rows:
        parts += ["## Scheduling check (§3)", "",
                  "| Step | Run | Microbatches | Chunk for chunk | First difference | TTFT 128k / 32k / 8k alone / 1k alone "
                  "(ms) | Worst TTFT deviation | In-run JIT |", "|---|---|---|---|---|---|---|---|"]
        for c in rows:
            t, w = c["ttft_ms"], c["target_ttft_ms"]
            tt = " / ".join(f"{t.get(k, float('nan')):.0f} vs {w[k]:.0f}" for k in ("sched_0", "sched_1", "sched_26",
                                                                                    "sched_27"))
            chunk_ok = not any(x.startswith("mb ") or "microbatches !=" in x for x in c["diffs"])
            parts.append(f"| {LABEL[c['step']]} | {c['run']} | {c['microbatch_count']} / {c['target_microbatches']} "
                         f"| {'yes' if chunk_ok else 'no'} | {c['first_diff_mb'] if c['first_diff_mb'] is not None else '–'} "
                         f"| {tt} | {100 * c['max_ttft_dev']:.1f}% | {c['in_run_jit']['compiles']} |")
        parts.append("")
    # force check
    rows = []
    for step in STEPS:
        d = _latest(root, f"force_check_{step}")
        if d is not None:
            rows.append(check(d))
    if rows:
        (out / "force_check.jsonl").write_text("".join(json.dumps(
            {"step": c["step"], "run": c["run"], **{k: c[k] for k in ("long_ttft_s", "short_ttft_s", "last_token_s",
                                                                        "microbatches")}}) + "\n" for c in rows))
        parts += ["## Force check (§4)", "",
                  "| Step | Run | 32k TTFT (s) | 1k TTFT p50 / p90 / p99 / max (s) | Last token (s) | Microbatches | "
                  "1k submit rate | Pass | Stage time / fit by microbatch tokens " + " / ".join(
                      f"{lo}–{hi - 1}" for lo, hi in BINS) + " | In-run JIT |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in rows:
            w, s, ws = c["target"], c["short_ttft_s"], c["target"]["short_ttft_s"]
            parts.append(
                f"| {LABEL[c['step']]} | {c['run']} | {c['long_ttft_s']} vs {w['long_ttft_s']} "
                f"(x{_r(c['long_ttft_s'], w['long_ttft_s'])}) | {s['p50']} / {s['p90']} / {s['p99']} / {s['max']} vs "
                f"{ws['p50']} / {ws['p90']} / {ws['p99']} / {ws['max']} (x{_r(s['p50'], ws['p50'])} / "
                f"x{_r(s['p90'], ws['p90'])} / x{_r(s['p99'], ws['p99'])}) | {c['last_token_s']} vs "
                f"{w['last_token_s']} | {c['microbatches']} / {w['microbatches']} | {c['short_submit_rate_per_s']}/s "
                f"| {'yes' if c['pass'] else 'no: ' + '; '.join(c['diffs'])} | {_fit_str(c['stage_fit'])} "
                f"| {c['in_run_jit']['compiles']} |")
        parts.append("")
    # sessions
    lines = []
    for step in STEPS:
        for n in (400, 500):
            d = _latest(root, f"sched_{step}_n{n}")
            if d is None and step == "plain":
                d = _latest(root, f"prefix_n{n}")
            if d is not None and (d / "requests.jsonl").exists() and (d / "metrics.json").exists():
                lines.append(session_line(d, step, n))
    if lines:
        (out / "scheduler_sessions.json").write_text(json.dumps(lines, indent=1, default=str) + "\n")
        parts += ["## Sessions (§5)", "",
                  "| Step | N | Run | Rounds/s | New tokens/s | Computed tokens/s | Prefix hit share | TTFT p50 / p90 / "
                  "p99 / max (s) | Over 60 s | Stage busy 0 / 1 / 2 / 3 | In-run JIT |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in lines:
            m, t = s["metrics"], s["target"] or {}
            tt, wt = m["ttft_s"], t.get("ttft_s", {})
            parts.append(
                f"| {LABEL[s['step']]} | {s['n']} | {s['run']}{' (slot held, old client)' if s['old_client'] else ''} "
                f"| {m['rounds_per_s']:.1f} vs {t.get('rounds_per_s', 0):.1f} (x{_r(m['rounds_per_s'], t.get('rounds_per_s'))}) "
                f"| {m['new_tokens_per_s'] / 1e3:.1f}k vs {t.get('new_tokens_per_s', 0) / 1e3:.1f}k "
                f"(x{_r(m['new_tokens_per_s'], t.get('new_tokens_per_s'))}) "
                f"| {m['computed_tokens_per_s'] / 1e3:.1f}k vs {t.get('computed_tokens_per_s', 0) / 1e3:.1f}k "
                f"(x{_r(m['computed_tokens_per_s'], t.get('computed_tokens_per_s'))}) "
                f"| {100 * m['prefix_hit_share']:.2f}% vs {100 * t.get('prefix_hit_share', 0):.2f}% "
                f"| {tt['p50']:.2f} / {tt['p90']:.2f} / {tt['p99']:.2f} / {tt['max']:.1f} vs "
                f"{wt.get('p50', 0):.2f} / {wt.get('p90', 0):.2f} / {wt.get('p99', 0):.2f} / {wt.get('max', 0):.1f} "
                f"(x{_r(tt['p50'], wt.get('p50'))} / x{_r(tt['p90'], wt.get('p90'))} / x{_r(tt['p99'], wt.get('p99'))}) "
                f"| {m['rounds_ttft_over_60s']} vs {t.get('rounds_ttft_over_60s', '–')} "
                f"| {' / '.join(f'{100 * b:.0f}' for b in s['stage_busy'])}% vs "
                f"{' / '.join(f'{100 * b:.0f}' for b in t.get('stage_busy', []))}% "
                f"| {s['in_run_jit']['compiles']} |")
            if (x := s["sim_x106"]) is not None:
                xt = x["ttft_s"]
                parts.append(
                    f"| └ sim x{x['gpu_time_multiplier']:g} | | | {x['rounds_per_s']:.1f} "
                    f"| {x['new_tokens_per_s'] / 1e3:.1f}k | {x['computed_tokens_per_s'] / 1e3:.1f}k "
                    f"| {100 * x['prefix_hit_share']:.2f}% | {xt['p50']:.2f} / {xt['p90']:.2f} / {xt['p99']:.2f} / "
                    f"{xt['max']:.1f} (measured x{_r(tt['p50'], xt['p50'])} / x{_r(tt['p90'], xt['p90'])} / "
                    f"x{_r(tt['p99'], xt['p99'])}) | {x['rounds_ttft_over_60s']} | | |")
        parts += ["", "Stage time / fit by microbatch tokens " + " / ".join(BIN_LABELS)
                  + "; hit kinds; GPU busy SM MHz / W / power-capped share:", "",
                  "| Step | N | Stage time / fit | Rounds by hit kind | GPU |", "|---|---|---|---|---|"]
        for s in lines:
            kinds = ", ".join(f"{k} {v}" for k, v in sorted((s.get("cache_kinds") or {}).get("by_kind", {}).items()))
            gpu = "; ".join(f"g{g}: {v['sm_mhz']} / {v['power_w']} / {100 * v['power_cap_share']:.0f}%"
                            for g, v in (s.get("gpu") or {}).items()) or "–"
            parts.append(f"| {LABEL[s['step']]} | {s['n']} | {_fit_str(s['stage_fit_bins'])} | {kinds} | {gpu} |")
        parts.append("")
    path = out / "scheduler.md"
    notes = ""
    if path.exists() and (m := re.search(r"^## Notes$", path.read_text(), re.M)):
        notes = path.read_text()[m.start():]
    path.write_text("\n".join(parts) + ("\n" + notes if notes else ""))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("run_dir", type=Path)
    s = sub.add_parser("summarize")
    s.add_argument("root", type=Path, nargs="?", default=MEASURED / "runs_m5")
    s.add_argument("--out", type=Path, default=MEASURED)
    a = ap.parse_args()
    if a.cmd == "check":
        r = check(a.run_dir)
        print(json.dumps({k: r[k] for k in r if k not in ("microbatches", "ttft_ms", "target_ttft_ms", "stage_fit")},
                         default=str)[:4000])
        print(f"[{a.run_dir.name}] {'PASS' if r['pass'] else 'FAIL'}: " + "; ".join(r["diffs"][:10]))
    else:
        print(summarize(a.root, a.out))


if __name__ == "__main__":
    main()
