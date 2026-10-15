"""Numbers of 4_iteration.md §6/§8 from run dirs written by bench/run.py.

  python bench/report.py measured/runs/one_8k [more run dirs...]

Per run dir: report.json. Then, over every run dir beside it (the runs root), the summary: `iteration.jsonl`
(one-request runs, one line per request), `saturated.jsonl` (one line per length) and `iteration.md` (both tables
with the targets and ratios). The summary goes to measured/ for measured/runs/<name>, else into the runs root.

Times: microbatches.jsonl is in ms from t0 = the first request's arrival at the server (t0.json has it on the host
clock); req-frontend's requests.jsonl has unix-second submit/post times on the same host clock, so the two align.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import HANDOFF  # noqa: E402

SEND_BYTES_PER_TOKEN = 4 * 4096 * 2


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def client_rows(run_dir: Path) -> dict[str, dict]:
    """req-frontend requests.jsonl, by request id, with times in ms."""
    out = {}
    for r in read_jsonl(run_dir / "requests.jsonl"):
        o = r.get("outcome", r)
        out[o["request_id"]] = {
            "ok": str(o.get("status")).upper() == "SUCCESS" and o.get("first_token_id_ms") is not None,
            "submit_unix_ms": o["submit_timestamp"] * 1e3,
            "post_unix_ms": (o.get("post_timestamp") or o["submit_timestamp"]) * 1e3,
            "ttft_ms": o.get("first_token_id_ms"),
            "prompt_tokens": (o.get("server_usage") or {}).get("prompt_tokens"),
            "trace_id": r.get("source", {}).get("data", {}).get("id"),
        }
    return out


def trace_ids(trace: Path) -> list[str]:
    lines = trace.read_text().splitlines()
    col = lines[0].split(",").index("id")
    return [line.split(",")[col] for line in lines[1:] if line.strip()]


def client_id(trace_id: str) -> str:
    return f"independent_{trace_id}"   # req-frontend's id for an independent request


# --- one request ------------------------------------------------------------------------------------------------------

def load_trace(run_dir: Path) -> dict | None:
    f = run_dir / "trace.json"
    return json.loads(f.read_text()) if f.exists() else None


def trace_request(tr: dict, mb_ids: list[int], nstages: int) -> dict | None:
    """A request's stages, sends and span from the nsys trace (bench/nsys_trace.py), over its microbatches."""
    got = [tr["mbs"].get(str(i)) for i in sorted(set(mb_ids))]
    if not got or any(g is None for g in got):
        return None
    stages = []
    for s in range(nstages):
        xs = [g["stages"][s] for g in got]
        stages.append({"stage": s, "start_ms": xs[0]["start_ms"], "end_ms": xs[-1]["end_ms"],
                       "compute_ms": sum(x["compute_ms"] for x in xs), "gaps_ms": sum(x["gaps_ms"] for x in xs),
                       "nccl_recv_ms": _sum_opt(x["nccl_recv_ms"] for x in xs),
                       "nccl_send_ms": _sum_opt(x["nccl_send_ms"] for x in xs),
                       "host_lead_ms": xs[0].get("host_lead_ms"), "launch_ms": _sum_opt(x.get("launch_ms") for x in xs)})
    return {"stages": stages, "span": stages[-1]["end_ms"] - stages[0]["start_ms"]}


def _sum_opt(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) if xs else None


def one_request(run_dir: Path, meta: dict) -> list[dict]:
    """Stages, sends and GPU span come from the nsys trace when the run has one (trace.json), with the CUDA-event
    numbers of microbatches.jsonl beside them as `event_*`; otherwise from the CUDA events alone."""
    tr = load_trace(run_dir)
    mbs = read_jsonl(run_dir / "microbatches.jsonl")
    sreq = {r["id"]: r for r in read_jsonl(run_dir / "server_requests.jsonl")}
    t0 = json.loads((run_dir / "t0.json").read_text())["t0_unix_ns"] / 1e6 if (run_dir / "t0.json").exists() else None
    client = client_rows(run_dir)
    pool = run_dir / "pool.json"
    nstages = (len(json.loads(pool.read_text())["ready_blocks"]) if pool.exists()
               else 1 + max((m["stage"] for m in mbs), default=0))
    lines = []
    for i, tid in enumerate(trace_ids(Path(meta["trace"])), 1):
        rid = client_id(tid)
        mine = [m for m in mbs if any(q["id"] == rid for q in m["requests"])]
        c = client.get(rid)
        why = ("no client record" if c is None else "client: no token" if not c["ok"]
               else "stages missing" if {m["stage"] for m in mine} != set(range(nstages)) else None)
        if why:   # a failed request is a result, not a crash: it is listed and counted in the tables
            lines.append({"prompt_tokens": meta["prompt_tokens"], "run": i, "id": rid, "failed": why,
                          "stages_done": sorted({m["stage"] for m in mine})})
            continue
        stages = []
        for s in range(nstages):
            ms = sorted((m for m in mine if m["stage"] == s), key=lambda m: m["start_ms"])
            start = ms[0]["start_ms"]
            end = max(m["start_ms"] + m["compute_ms"] for m in ms)
            stages.append({"stage": s, "start_ms": start, "end_ms": end, "compute_ms": sum(m["compute_ms"] for m in ms),
                           "gaps_ms": None})
        origin = stages[0]["start_ms"]
        tokens = sum(q["append"] for m in mine if m["stage"] == 0 for q in m["requests"])
        sends = [{"after_stage": s, "bytes": tokens * SEND_BYTES_PER_TOKEN,
                  "ms": stages[s + 1]["start_ms"] - stages[s]["end_ms"]} for s in range(nstages - 1)]
        span = stages[-1]["end_ms"] - origin
        line = {"prompt_tokens": meta["prompt_tokens"], "run": i, "id": rid, "client_ttft_ms": c["ttft_ms"],
                "gpu_span_ms": span, "host_ms": c["ttft_ms"] - span if c["ttft_ms"] is not None else None,
                "stages": [{"stage": st["stage"], "start_ms": st["start_ms"] - origin, "compute_ms": st["compute_ms"],
                            "gaps_ms": st["gaps_ms"]} for st in stages],
                "sends": sends, "token_matches_reference": None}
        t = trace_request(tr, [m["index"] for m in mine], nstages) if tr else None
        if t is not None:
            o = t["stages"][0]["start_ms"]
            for st, ts in zip(line["stages"], t["stages"]):
                st.update(event_start_ms=st["start_ms"], event_compute_ms=st["compute_ms"],
                          start_ms=ts["start_ms"] - o, compute_ms=ts["compute_ms"], gaps_ms=ts["gaps_ms"],
                          host_lead_ms=ts["host_lead_ms"], launch_ms=ts["launch_ms"])
            for se in line["sends"]:
                a, b = t["stages"][se["after_stage"]], t["stages"][se["after_stage"] + 1]
                se.update(event_ms=se["ms"], ms=b["start_ms"] - a["end_ms"], nccl_send_ms=a["nccl_send_ms"],
                          nccl_recv_ms=b["nccl_recv_ms"])
            line.update(event_gpu_span_ms=span, gpu_span_ms=t["span"], source="nsys")
            if c["ttft_ms"] is not None:
                line["host_ms"] = c["ttft_ms"] - t["span"]
        else:
            line["source"] = "cuda_events"
        sr = sreq.get(rid)
        if t0 is not None and sr is not None and c["ttft_ms"] is not None:   # on the host clock (CUDA events)
            post = c["post_unix_ms"] - t0                 # client send, on the server's t0 axis
            first = post + c["ttft_ms"]                   # client sees the token
            line["host_parts_ms"] = {
                "client_to_server": sr["arrival_ms"] - post,
                "arrival_to_dispatch": sr["first_dispatch_ms"] - sr["arrival_ms"],
                "dispatch_to_stage0": origin - sr["first_dispatch_ms"],
                "last_stage_to_controller": sr["token_ms"] - stages[-1]["end_ms"],
                "controller_to_client": first - sr["token_ms"],
            }
        lines.append(line)
    return lines


# --- saturated --------------------------------------------------------------------------------------------------------

def saturated(run_dir: Path, meta: dict) -> dict:
    mbs = read_jsonl(run_dir / "microbatches.jsonl")
    nstages = 1 + max(m["stage"] for m in mbs)
    by = [[m for m in mbs if m["stage"] == s] for s in range(nstages)]
    start = min(m["start_ms"] for m in by[0])
    end = max(m["start_ms"] + m["compute_ms"] for m in by[-1])
    lo, hi = start + 0.1 * (end - start), start + 0.9 * (end - start)
    tokens = sum(q["append"] for m in by[0] if lo <= m["start_ms"] < hi for q in m["requests"])
    busy = [sum(max(0.0, min(m["start_ms"] + m["compute_ms"], hi) - max(m["start_ms"], lo)) for m in s) / (hi - lo)
            for s in by]
    s0 = sorted(by[0], key=lambda m: m["start_ms"])
    edges = set((meta.get("nsys") or {}).get("profile_mbs") or [])   # cudaProfilerStart/Stop stall the host there (seconds), not the engine
    gaps = [b["start_ms"] - (a["start_ms"] + a["compute_ms"]) for a, b in zip(s0, s0[1:]) if b["index"] not in edges]
    s3_end = {}
    for m in by[-1]:
        for q in m["requests"]:
            s3_end[q["id"]] = max(s3_end.get(q["id"], 0.0), m["start_ms"] + m["compute_ms"])
    lags = [r["token_ms"] - s3_end[r["id"]] for r in read_jsonl(run_dir / "server_requests.jsonl")
            if "token_ms" in r and r["id"] in s3_end]
    client = client_rows(run_dir)
    ids = trace_ids(Path(meta["trace"]))
    rows = [client.get(client_id(t)) for t in ids]
    ok = [r for r in rows if r is not None and r["ok"]]
    ttft = np.array([r["ttft_ms"] for r in ok]) / 1e3
    submits = [r["submit_unix_ms"] for r in rows if r is not None]
    tps = tokens / ((hi - lo) / 1e3)
    out = {
        "prompt_tokens": meta["prompt_tokens"], "requests": len(ids), "answered": len(ok),
        "microbatches": len(by[0]),
        "submit_spread_s": (max(submits) - min(submits)) / 1e3 if submits else None,
        "run_s": (end - start) / 1e3, "tokens_per_s": tps, "requests_per_s": tps / meta["prompt_tokens"],
        "ttft_s": {"first": float(ttft.min()), **{f"p{q}": float(np.percentile(ttft, q)) for q in (50, 90, 99)},
                   "max": float(ttft.max())} if len(ttft) else None,
        "stage_busy": busy,
        "stage0_gap_ms": {"mean": float(np.mean(gaps)), "max": float(np.max(gaps)),
                          "p99": float(np.percentile(gaps, 99))} if gaps else None,
        "stage0_gap_excluded_mbs": sorted(edges),
        # stage-3 end -> the controller handled that report (control thread responsiveness)
        "report_lag_ms": {"p50": float(np.median(lags)), "max": float(np.max(lags))} if lags else None,
        "ttft_ms_by_request": [round(r["ttft_ms"], 3) if r is not None and r["ok"] else None for r in rows],
    }
    tr = load_trace(run_dir)
    if tr and tr["mbs"]:
        out["trace"] = trace_summary(tr, {(m["stage"], m["index"]): m["compute_ms"] for m in mbs})
    return out


def _stats(xs: list[float]) -> dict | None:
    return {"mean": float(np.mean(xs)), "max": float(np.max(xs)), "p99": float(np.percentile(xs, 99))} if xs else None


def trace_summary(tr: dict, event_compute: dict) -> dict:
    """The traced microbatches (nsys window): per-stage compute and gaps, sends, the stage-0 gap between consecutive
    microbatches at kernel level, and the CUDA-event compute of the same microbatches as a cross-check."""
    ms = list(tr["mbs"].items())
    n = tr["stages"]
    return {
        "microbatches": len(ms), "mb_range": [min(int(k) for k, _ in ms), max(int(k) for k, _ in ms)],
        "stage_compute_ms": [float(np.mean([v["stages"][s]["compute_ms"] for _, v in ms])) for s in range(n)],
        "event_stage_compute_ms": [float(np.mean([event_compute[(s, int(k))] for k, _ in ms
                                                  if (s, int(k)) in event_compute])) for s in range(n)],
        "stage_gaps_ms": [float(np.mean([v["stages"][s]["gaps_ms"] for _, v in ms])) for s in range(n)],
        "send_ms": [float(np.mean([v["sends"][s]["ms"] for _, v in ms])) for s in range(n - 1)],
        "host_lead_ms": [_median_opt([v["stages"][s].get("host_lead_ms") for _, v in ms]) for s in range(n)],
        "launch_ms": [_median_opt([v["stages"][s].get("launch_ms") for _, v in ms]) for s in range(n)],
        "stage0_gap_ms": _stats(tr["stage0_gap_ms"]),
        "unattributed_kernels": tr["unattributed_kernels"],
    }


# --- summary ----------------------------------------------------------------------------------------------------------

def _targets():
    it = {t["prompt_tokens"]: t for t in json.loads((HANDOFF / "data/iteration_targets.json").read_text())}
    sat = {t["prompt_tokens"]: t for t in json.loads((HANDOFF / "data/saturated_targets.json").read_text())}
    return it, sat


def _median_opt(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def _mean_opt(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _f(x, nd=3):
    return "—" if x is None else f"{x:,.{nd}f}"


def _ratio(m, t):
    return "—" if m is None or not t else f"{m / t:.2f}"


def iteration_md(one: list[dict], sat: list[dict], title: str) -> str:
    it_t, sat_t = _targets()
    out = [f"# {title}", ""]
    if one:
        out += ["## One request (4_iteration.md §3)", "",
                "Per length: mean of the runs, and the first run alone; ms. Ratio = measured / target.", "",
                "| Prompt | Runs | | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Send 0→1 | Send 1→2 | Send 2→3 | GPU span "
                "| Client TTFT | Host |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for L in sorted({r["prompt_tokens"] for r in one}):
            failed = [r for r in one if r["prompt_tokens"] == L and r.get("failed")]
            if failed:
                out.append(f"| {L // 1024}k | {len(failed)} failed | " + "; ".join(
                    f"#{r['run']}: {r['failed']} (stages done {r['stages_done']})" for r in failed) + " |")
            rs = [r for r in one if r["prompt_tokens"] == L and not r.get("failed")]
            if not rs:
                continue
            t = it_t.get(L)

            def row(label, sel):
                st = [np.mean([r["stages"][s]["compute_ms"] for r in sel]) for s in range(len(sel[0]["stages"]))]
                se = [np.mean([r["sends"][s]["ms"] for r in sel]) for s in range(len(sel[0]["sends"]))]
                span = np.mean([r["gpu_span_ms"] for r in sel])
                ttft = np.mean([r["client_ttft_ms"] for r in sel])
                host = np.mean([r["host_ms"] for r in sel])
                cells = [_f(x) for x in st] + [_f(x) for x in se] + [_f(span), _f(ttft), _f(host)]
                rat = ([_ratio(x, t["stages"][s]["compute_ms"]) for s, x in enumerate(st)]
                       + [_ratio(x, t["sends"][s]["ms"]) for s, x in enumerate(se)]
                       + [_ratio(span, t["ttft_ms"]), _ratio(ttft, t["ttft_ms"]), ""]) if t else [""] * len(cells)
                return [f"| {L // 1024}k | {label} | measured | " + " | ".join(cells) + " |",
                        f"| | | ratio | " + " | ".join(rat) + " |"]

            out += row(f"mean of {len(rs)}", rs) + row("first", rs[:1])
            if t:
                out.append(f"| | | target | " + " | ".join(
                    [_f(s["compute_ms"]) for s in t["stages"]] + [_f(s["ms"]) for s in t["sends"]]
                    + [_f(t["ttft_ms"]), _f(t["ttft_ms"]), "0"]) + " |")
        traced = [r for r in one if r.get("source") == "nsys"]
        if traced:
            out += ["", "From the nsys trace (mean over runs, ms): gaps between kernels inside each stage, NCCL kernel "
                    "time per send, and CUDA-event / trace stage compute.", "",
                    "| Prompt | Gaps 0 / 1 / 2 / 3 | NCCL send 0 / 1 / 2 | NCCL recv 1 / 2 / 3 | Event / trace 0 / 1 / 2 / 3 "
                    "| Host lead 0 / 1 / 2 / 3 | Launch 0 / 1 / 2 / 3 |",
                    "|---|---|---|---|---|---|---|"]
            for L in sorted({r["prompt_tokens"] for r in traced}):
                rs = [r for r in traced if r["prompt_tokens"] == L]
                ns = len(rs[0]["stages"])
                g = [np.mean([r["stages"][s]["gaps_ms"] for r in rs]) for s in range(ns)]
                snd = [_mean_opt([r["sends"][s].get("nccl_send_ms") for r in rs]) for s in range(ns - 1)]
                rcv = [_mean_opt([r["sends"][s].get("nccl_recv_ms") for r in rs]) for s in range(ns - 1)]
                ev = [np.mean([r["stages"][s]["event_compute_ms"] / r["stages"][s]["compute_ms"] for r in rs])
                      for s in range(ns)]
                lead = [_mean_opt([r["stages"][s].get("host_lead_ms") for r in rs]) for s in range(ns)]
                launch = [_mean_opt([r["stages"][s].get("launch_ms") for r in rs]) for s in range(ns)]
                out.append(f"| {L // 1024}k | " + " / ".join(_f(x) for x in g) + " | " + " / ".join(_f(x) for x in snd)
                           + " | " + " / ".join(_f(x) for x in rcv) + " | " + " / ".join(f"{x:.3f}" for x in ev)
                           + " | " + " / ".join(_f(x) for x in lead) + " | " + " / ".join(_f(x) for x in launch) + " |")
        parts = [r for r in one if r.get("host_parts_ms")]
        if parts:
            out += ["", "Host time by part (mean over runs, ms):", "",
                    "| Prompt | client→server | arrival→dispatch | dispatch→stage 0 | last stage→controller "
                    "| controller→client |", "|---|---|---|---|---|---|"]
            for L in sorted({r["prompt_tokens"] for r in parts}):
                ps = [r["host_parts_ms"] for r in parts if r["prompt_tokens"] == L]
                keys = list(ps[0])
                out.append(f"| {L // 1024}k | " + " | ".join(_f(np.mean([p[k] for p in ps])) for k in keys) + " |")
        out.append("")
    if sat:
        out += ["## Saturated (4_iteration.md §4)", "",
                "| Prompt | Requests | Run (s) | Tokens/s | TTFT first / p50 / p90 / p99 / max (s) "
                "| Stage busy 0 / 1 / 2 / 3 | Submit spread (s) | Stage-0 gap mean / max (ms) | Report lag p50 / max (ms) |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(sat, key=lambda r: r["prompt_tokens"]):
            t = sat_t.get(r["prompt_tokens"])
            tt = r["ttft_s"] or {}
            q = ("first", "p50", "p90", "p99", "max")
            g = r.get("stage0_gap_ms") or {}
            lag = r.get("report_lag_ms") or {}
            out.append(f"| {r['prompt_tokens'] // 1024}k | {r['answered']}/{r['requests']} | {_f(r['run_s'], 2)} "
                       f"| {r['tokens_per_s'] / 1e3:.1f}k | " + " / ".join(_f(tt.get(k), 3) for k in q)
                       + " | " + " / ".join(f"{100 * b:.1f}" for b in r["stage_busy"])
                       + f" | {_f(r['submit_spread_s'], 3)} | {_f(g.get('mean'), 3)} / {_f(g.get('max'), 3)}"
                       + f" | {_f(lag.get('p50'), 3)} / {_f(lag.get('max'), 3)} |")
            if t:
                out.append(f"| ratio | | {_ratio(r['run_s'], t['run_s'])} | {_ratio(r['tokens_per_s'], t['tokens_per_s'])}"
                           " | " + " / ".join(_ratio(tt.get(k), t["ttft_s"][k]) for k in q) + " | | | | |")
                out.append(f"| target | {t['requests']} | {_f(t['run_s'], 2)} | {t['tokens_per_s'] / 1e3:.1f}k | "
                           + " / ".join(_f(t["ttft_s"][k], 3) for k in q) + " | "
                           + " / ".join(f"{100 * b:.1f}" for b in t["stage_busy"]) + " | | | |")
        traced = [r for r in sat if r.get("trace")]
        if traced:
            out += ["", "From the nsys trace of each run's traced microbatches (ms): stage compute (CUDA events in "
                    "brackets), gaps inside the stage, sends, and the stage-0 gap between consecutive microbatches.", "",
                    "| Prompt | Traced mbs | Stage compute 0 / 1 / 2 / 3 | Gaps 0 / 1 / 2 / 3 | Sends | "
                    "Stage-0 gap mean / p99 / max | Host lead p50 0 / 1 / 2 / 3 | Launch p50 0 / 1 / 2 / 3 |",
                    "|---|---|---|---|---|---|---|---|"]
            for r in traced:
                t = r["trace"]
                g = t["stage0_gap_ms"] or {}
                out.append(f"| {r['prompt_tokens'] // 1024}k | {t['microbatches']} ({t['mb_range'][0]}–{t['mb_range'][1]}) | "
                           + " / ".join(f"{a:.3f} ({b:.3f})" for a, b in zip(t["stage_compute_ms"],
                                                                           t["event_stage_compute_ms"]))
                           + " | " + " / ".join(_f(x) for x in t["stage_gaps_ms"])
                           + " | " + " / ".join(_f(x) for x in t["send_ms"])
                           + f" | {_f(g.get('mean'))} / {_f(g.get('p99'))} / {_f(g.get('max'))}"
                           + " | " + " / ".join(_f(x) for x in t.get("host_lead_ms", []))
                           + " | " + " / ".join(_f(x) for x in t.get("launch_ms", [])) + " |")
        out.append("")
    return "\n".join(out)


def report_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    rep = {"name": meta["name"], "kind": meta["kind"]}
    if meta["kind"] == "one":
        rep["requests"] = one_request(run_dir, meta)
    else:
        rep["saturated"] = saturated(run_dir, meta)
    tr = load_trace(run_dir)
    if tr:
        rep["kernels"] = [{k: v for k, v in a.items() if k != "demangled"} for a in tr["kernels"]]
    rep["tactics"] = meta.get("tactics")
    rep["tokens"] = token_summary(run_dir)
    (run_dir / "report.json").write_text(json.dumps(rep) + "\n")
    return rep


def token_summary(run_dir: Path) -> dict | None:
    """bench/check_tokens.py's verdicts, with each miss's reference top-1 margin: above 2k the forward is not
    bitwise run to run (sparse-MLA summation order), so a miss with a small margin may be that, not a bug."""
    f = run_dir / "token_check.json"
    if not f.exists():
        return None
    tc = json.loads(f.read_text())
    refs = {}
    ref_file = Path(tc["summary"]["reference"])
    if ref_file.exists():
        refs = {r["id"]: r for r in read_jsonl(ref_file)}
    misses = [{"id": r["id"], "verdict": r["verdict"], "token": r.get("token"),
               "reference_token": r.get("reference_token"),
               "reference_margin": refs.get(r.get("reference_id"), {}).get("margin"),
               "rank_in_reference_top5": r.get("engine_token_rank_in_reference_top5")}
              for r in tc["requests"] if r["verdict"] in ("fail", "no_token")]
    s = tc["summary"]
    return {k: s[k] for k in ("pass", "fail", "no_token", "no_reference", "ok")} | {"misses": misses}


def tactics_note(tactics: list | None) -> str:
    if not tactics:
        return "not recorded"
    src = [(t or {}).get("source", "none") for t in tactics]
    tuned = [s for s, x in enumerate(src) if x == "tuned_in_process"]
    files = {(t or {}).get("sha256", "")[:12] for t in tactics if t}
    return ("**TUNED IN PROCESS on stages " + ", ".join(map(str, tuned)) + "**") if tuned \
        else f"{'/'.join(sorted(set(src)))} ({', '.join(sorted(files)) or '—'})"


def runs_md(runs: list[dict]) -> str:
    """Per run: tokens against the reference (misses with the reference margin) and the FlashInfer tactics used."""
    out = ["## Runs: tokens and tactics", "",
           "| Run | Tokens pass / fail / no token / not in reference | Misses (token vs reference, reference margin) "
           "| Tactics |", "|---|---|---|---|"]
    size = lambda r: int(m.group(1)) if (m := re.search(r"_(\d+)k$", r["name"])) else 0   # noqa: E731
    for r in sorted(runs, key=lambda r: (r["kind"], size(r), r["name"])):
        t = r.get("tokens")
        cell = "—" if t is None else f"{t['pass']} / {t['fail']} / {t['no_token']} / {t['no_reference']}"
        miss = "; ".join(f"{m['id']}: {m['token']} vs {m['reference_token']} (margin {m['reference_margin']})"
                         for m in (t or {}).get("misses", [])) or "—"
        out.append(f"| {r['name']} | {cell} | {miss} | {tactics_note(r.get('tactics'))} |")
    return "\n".join(out) + "\n"


def kernels_md(name: str, kernels: list[dict], top: int = 12) -> str:
    out = [f"### {name}: kernels per stage (nsys, per microbatch)", "",
           "| Stage | Kernel | ms / mb | launches / mb | Manifest | Positions |", "|---|---|---|---|---|---|"]
    stages = sorted({k["stage"] for k in kernels})
    for s in stages:
        ks = [k for k in kernels if k["stage"] == s]
        for k in ks[:top]:
            mbs = k["total_ms"] / k["per_mb_ms"] if k["per_mb_ms"] else 1
            cls = k.get("class", "—") + (f" {k['configs']}" if k.get("configs") else "")
            out.append(f"| {s} | `{k['name'][:60]}` | {k['per_mb_ms']:.3f} | {k['count'] / mbs:.1f} | {cls} "
                       f"| {', '.join(k.get('positions', []))[:80] or '—'} |")
        rest = ks[top:]
        if rest:
            out.append(f"| {s} | ({len(rest)} more) | {sum(k['per_mb_ms'] for k in rest):.3f} | | | |")
    return "\n".join(out + [""])


def summarize(runs_root: Path) -> Path:
    one, sat, kern, runs = [], [], [], []
    for d in sorted(runs_root.iterdir()):
        f = d / "report.json"
        if not f.exists():
            continue
        rep = json.loads(f.read_text())
        runs.append(rep)
        one += rep.get("requests", [])
        if "saturated" in rep:
            sat.append(rep["saturated"])
        if rep.get("kernels"):
            kern.append(kernels_md(rep["name"], rep["kernels"]))
    out = runs_root.parent if runs_root.name == "runs" else runs_root
    one.sort(key=lambda r: (r["prompt_tokens"], r["run"]))
    sat.sort(key=lambda r: r["prompt_tokens"])
    (out / "iteration.jsonl").write_text("".join(json.dumps(r) + "\n" for r in one))
    (out / "saturated.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sat))
    title = "M3 iteration: measured vs targets" + (f" ({runs_root.name} runs)" if out == runs_root else "")
    md = iteration_md(one, sat, title)
    md += runs_md(runs)
    if kern:
        md += "\n## Kernels (nsys)\n\n" + "\n".join(kern)
    (out / "iteration.md").write_text(md)
    return out / "iteration.md"


def main():
    dirs = [Path(a).resolve() for a in sys.argv[1:]]
    if not dirs:
        raise SystemExit(__doc__)
    for d in dirs:
        rep = report_run(d)
        if "saturated" in rep:
            s = rep["saturated"]
            print(f"{d.name}: {s['tokens_per_s'] / 1e3:.1f}k tok/s, run {s['run_s']:.2f} s, busy "
                  + " ".join(f"{100 * b:.1f}" for b in s["stage_busy"])
                  + (f", TTFT p50 {s['ttft_s']['p50']:.3f} s" if s["ttft_s"] else ""))
        else:
            for r in rep["requests"]:
                if r.get("failed"):
                    print(f"{d.name} #{r['run']}: FAILED ({r['failed']}, stages done {r['stages_done']})")
                else:
                    print(f"{d.name} #{r['run']}: span {r['gpu_span_ms']:.3f} ms, TTFT {r['client_ttft_ms']:.3f} ms, "
                          "stages " + " ".join(f"{s['compute_ms']:.3f}" for s in r["stages"]))
    for root in {d.parent for d in dirs}:
        print(summarize(root))


if __name__ == "__main__":
    main()
