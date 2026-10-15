"""Numbers of 4_iteration.md §6/§8 from run dirs written by bench/run.py.

  python bench/report.py measured/runs/one_8k [more run dirs...]

Per run dir: report.json. Then, over every run dir beside it (the runs root), the summary: `iteration.jsonl`
(one-request runs, one line per request), `saturated.jsonl` (one line per length) and `iteration.md` (both tables
with the targets and ratios). The summary goes to measured/ for measured/runs/<name>, else into the runs root.

Times: microbatches.jsonl is in ms from t0 = the first request's arrival at the server (t0.json has it on the host
clock); req-frontend's requests.jsonl has unix-second submit/post times on the same host clock, so the two align.
"""

import json
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

def one_request(run_dir: Path, meta: dict) -> list[dict]:
    mbs = read_jsonl(run_dir / "microbatches.jsonl")
    sreq = {r["id"]: r for r in read_jsonl(run_dir / "server_requests.jsonl")}
    t0 = json.loads((run_dir / "t0.json").read_text())["t0_unix_ns"] / 1e6 if (run_dir / "t0.json").exists() else None
    client = client_rows(run_dir)
    nstages = 1 + max((m["stage"] for m in mbs), default=0)
    lines = []
    for i, tid in enumerate(trace_ids(Path(meta["trace"])), 1):
        rid = client_id(tid)
        mine = [m for m in mbs if any(q["id"] == rid for q in m["requests"])]
        c = client.get(rid)
        if not mine or c is None:
            lines.append({"prompt_tokens": meta["prompt_tokens"], "run": i, "id": rid, "missing": True})
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
        sr = sreq.get(rid)
        if t0 is not None and sr is not None and c["ttft_ms"] is not None:
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
    gaps = [b["start_ms"] - (a["start_ms"] + a["compute_ms"]) for a, b in zip(s0, s0[1:])]
    client = client_rows(run_dir)
    ids = trace_ids(Path(meta["trace"]))
    rows = [client.get(client_id(t)) for t in ids]
    ok = [r for r in rows if r is not None and r["ok"]]
    ttft = np.array([r["ttft_ms"] for r in ok]) / 1e3
    submits = [r["submit_unix_ms"] for r in rows if r is not None]
    tps = tokens / ((hi - lo) / 1e3)
    return {
        "prompt_tokens": meta["prompt_tokens"], "requests": len(ids), "answered": len(ok),
        "microbatches": len(by[0]),
        "submit_spread_s": (max(submits) - min(submits)) / 1e3 if submits else None,
        "run_s": (end - start) / 1e3, "tokens_per_s": tps, "requests_per_s": tps / meta["prompt_tokens"],
        "ttft_s": {"first": float(ttft.min()), **{f"p{q}": float(np.percentile(ttft, q)) for q in (50, 90, 99)},
                   "max": float(ttft.max())} if len(ttft) else None,
        "stage_busy": busy,
        "stage0_gap_ms": {"mean": float(np.mean(gaps)), "max": float(np.max(gaps)),
                          "p99": float(np.percentile(gaps, 99))} if gaps else None,
        "ttft_ms_by_request": [round(r["ttft_ms"], 3) if r is not None and r["ok"] else None for r in rows],
    }


# --- summary ----------------------------------------------------------------------------------------------------------

def _targets():
    it = {t["prompt_tokens"]: t for t in json.loads((HANDOFF / "data/iteration_targets.json").read_text())}
    sat = {t["prompt_tokens"]: t for t in json.loads((HANDOFF / "data/saturated_targets.json").read_text())}
    return it, sat


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
            rs = [r for r in one if r["prompt_tokens"] == L and not r.get("missing")]
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
                "| Stage busy 0 / 1 / 2 / 3 | Submit spread (s) | Stage-0 gap mean / max (ms) |",
                "|---|---|---|---|---|---|---|---|"]
        for r in sorted(sat, key=lambda r: r["prompt_tokens"]):
            t = sat_t.get(r["prompt_tokens"])
            tt = r["ttft_s"] or {}
            q = ("first", "p50", "p90", "p99", "max")
            g = r.get("stage0_gap_ms") or {}
            out.append(f"| {r['prompt_tokens'] // 1024}k | {r['answered']}/{r['requests']} | {_f(r['run_s'], 2)} "
                       f"| {r['tokens_per_s'] / 1e3:.1f}k | " + " / ".join(_f(tt.get(k), 3) for k in q)
                       + " | " + " / ".join(f"{100 * b:.1f}" for b in r["stage_busy"])
                       + f" | {_f(r['submit_spread_s'], 3)} | {_f(g.get('mean'), 3)} / {_f(g.get('max'), 3)} |")
            if t:
                out.append(f"| ratio | | {_ratio(r['run_s'], t['run_s'])} | {_ratio(r['tokens_per_s'], t['tokens_per_s'])}"
                           " | " + " / ".join(_ratio(tt.get(k), t["ttft_s"][k]) for k in q) + " | | | |")
                out.append(f"| target | {t['requests']} | {_f(t['run_s'], 2)} | {t['tokens_per_s'] / 1e3:.1f}k | "
                           + " / ".join(_f(t["ttft_s"][k], 3) for k in q) + " | "
                           + " / ".join(f"{100 * b:.1f}" for b in t["stage_busy"]) + " | | |")
        out.append("")
    return "\n".join(out)


def report_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    rep = {"name": meta["name"], "kind": meta["kind"]}
    if meta["kind"] == "one":
        rep["requests"] = one_request(run_dir, meta)
    else:
        rep["saturated"] = saturated(run_dir, meta)
    (run_dir / "report.json").write_text(json.dumps(rep) + "\n")
    return rep


def summarize(runs_root: Path) -> Path:
    one, sat = [], []
    for d in sorted(runs_root.iterdir()):
        f = d / "report.json"
        if not f.exists():
            continue
        rep = json.loads(f.read_text())
        one += rep.get("requests", [])
        if "saturated" in rep:
            sat.append(rep["saturated"])
    out = runs_root.parent if runs_root.name == "runs" else runs_root
    one.sort(key=lambda r: (r["prompt_tokens"], r["run"]))
    sat.sort(key=lambda r: r["prompt_tokens"])
    (out / "iteration.jsonl").write_text("".join(json.dumps(r) + "\n" for r in one))
    (out / "saturated.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sat))
    title = "M3 iteration: measured vs targets" + (f" ({runs_root.name} runs)" if out == runs_root else "")
    (out / "iteration.md").write_text(iteration_md(one, sat, title))
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
                if not r.get("missing"):
                    print(f"{d.name} #{r['run']}: span {r['gpu_span_ms']:.3f} ms, TTFT {r['client_ttft_ms']:.3f} ms, "
                          "stages " + " ".join(f"{s['compute_ms']:.3f}" for s in r["stages"]))
    for root in {d.parent for d in dirs}:
        print(summarize(root))


if __name__ == "__main__":
    main()
