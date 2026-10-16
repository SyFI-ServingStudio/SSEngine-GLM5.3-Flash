"""Numbers of 5_prefix_cache.md §8 from run dirs written by bench/run.py (prefix_bringup, prefix_check_on/off,
prefix_n<N>).

  python -m bench.prefix run <run dir>                      report.json of one run
  python -m bench.prefix summarize <runs root> --out measured
      measured/prefix_bringup.jsonl (one line per bring-up session, §8) and measured/prefix_cache.md (both tables
      with targets and ratios, the microbatch check, the cache counters); reads <runs root>/prefix_bringup,
      prefix_check_on, prefix_check_off, every prefix_n<N> and every prefix_smoke_n<N>* (against the target run cut
      to the smoke's window: measured/sessions_sim/, written once by bench/sim_rounds.py).

Clocks. The server's logs are in ms from its t0 (t0.json: the first request it received). With the prefix cache on,
req-frontend first sends its preflight (one 8k prompt twice, `PREFLIGHT`), so the server's t0 is earlier than the
workload's t = 0 (1_contract.md §3: the first workload request's submit); stage busy is taken over [0, 20 min) of the
workload's clock (`shift_ms`). The preflight's requests are left out of every number.

Correctness (§5 "Check the output"): round 1 of every bring-up session is compared with the same prompt computed
without the cache (prefix_check_off: the bring-up sessions on a server without `keep_sessions`), by prompt sha256.
prefix_check_on is the same with the cache on: on vs off, prompt for prompt.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from bench.common import HANDOFF, MEASURED
from bench.report import load_trace, read_jsonl, trace_request

PREFLIGHT = "req-frontend-prefix-cache-preflight"
WINDOW_MS = 20 * 60e3
MIN_FIT_TOKENS = 256


def client_rows(run_dir: Path) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(run_dir / "requests.jsonl"):
        o, d = r["outcome"], r["source"]["data"]
        u = o.get("server_usage") or {}
        out[o["request_id"]] = {
            "ok": str(o.get("status")).upper() == "SUCCESS" and o.get("first_token_id_ms") is not None,
            "session": int(d["session_id"]), "round": int(d["round_idx"]), "prefix_len": d["prefix_len"],
            "input_len": d["input_len"], "shortfall": d.get("prefix_shortfall_tokens", 0),
            "submit_unix_ms": o["submit_timestamp"] * 1e3, "ttft_ms": o.get("first_token_id_ms"),
            "prompt_tokens": u.get("prompt_tokens"), "cached": u.get("cached_prompt_tokens")}
    return out


# --- metrics over a window (the handoff's data/metrics.py, ported: a smoke against the same window of the target run)

def real_arrays(requests: Path):
    """(arrival, first token, prompt, new, prefix, cached) of req-frontend's log, ms from the first submit, as
    data/metrics.py's `real` returns them (successful rounds with a first token)."""
    arr, first, prompt, new, prefix, cached, t0 = [], [], [], [], [], [], None
    for r in read_jsonl(requests):
        o = r.get("outcome", r)
        t0 = o["submit_timestamp"] * 1e3 if t0 is None else min(t0, o["submit_timestamp"] * 1e3)
        if o.get("first_token_id_ms") is None or o.get("post_timestamp") is None \
                or str(o.get("status")).upper() != "SUCCESS":
            continue
        u = o.get("server_usage") or {}
        arr.append(o["submit_timestamp"] * 1e3)
        first.append(o["post_timestamp"] * 1e3 + o["first_token_id_ms"])
        prompt.append(u.get("prompt_tokens") or 0)
        cached.append(u.get("cached_prompt_tokens") or 0)
        new.append(r["source"]["data"]["input_len"])
        prefix.append(r["source"]["data"].get("prefix_len") or 0)
    return (np.array(arr) - t0, np.array(first) - t0, np.array(prompt), np.array(new), np.array(prefix),
            np.array(cached))


def sim_arrays(doc: dict):
    """The same arrays of a simulator run exported by bench/sim_rounds.py (data/metrics.py's `sim`)."""
    arr = np.asarray(doc["arrival_ms"], dtype=float)
    new, prefix = np.asarray(doc["new"]), np.asarray(doc["prefix"])
    return arr, arr + np.asarray(doc["ttft_ms"], dtype=float), prefix + new, new, prefix, np.asarray(doc["cached"])


def window_metrics(rows, window_ms: float) -> dict:
    """data/metrics.py's numbers over [0, window_ms): rounds whose first token comes inside it."""
    arr, first, prompt, new, prefix, cached = rows
    m = first < window_ms
    s = window_ms / 1e3
    ttft = (first - arr)[m] / 1e3
    return {"window_s": s, "rounds": int(m.sum()), "rounds_per_s": m.sum() / s,
            "new_tokens_per_s": float(new[m].sum() / s), "prompt_tokens_per_s": float(prompt[m].sum() / s),
            "computed_tokens_per_s": float((prompt[m] - cached[m]).sum() / s),
            "prefix_hit_share": float(cached[m].sum() / max(prefix[m].sum(), 1)),
            "ttft_s": {**{p: float(np.percentile(ttft, q)) for p, q in (("p50", 50), ("p90", 90), ("p99", 99))},
                       "max": float(ttft.max())} if m.any() else None,
            "rounds_ttft_over_60s": int((ttft > 60).sum())}


def sim_rounds(n: int, setting: str = "plain_hbm") -> dict | None:
    import gzip

    f = MEASURED / "sessions_sim" / f"{setting}_c{n}.json.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        return json.load(fh)


def smoke(run_dir: Path, n: int) -> dict:
    """A short sessions run (bench/run.py --client-limit) against the target run cut to the same window: from the
    first submit to the last first token the client logged (it logs no round still waiting when it stops)."""
    real = real_arrays(run_dir / "requests.jsonl")
    window = float(real[1].max()) + 1e-3      # the last logged first token counts
    sim = sim_rounds(n)
    return {"run": run_dir.name, "max_sessions": n, "window_s": window / 1e3,
            "measured": window_metrics(real, window),
            "target": None if sim is None else window_metrics(sim_arrays(sim), window),
            "target_source": None if sim is None else sim["source"]}


def smoke_md(rows: list[dict]) -> str:
    out = ["| Run | N | Window (s) | Rounds | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        m, t = x["measured"], x["target"] or {}
        tt, ts = m["ttft_s"] or {}, (t.get("ttft_s") or {})
        ttft = (" / ".join(f"{tt.get(p, float('nan')):.2f}" for p in ("p50", "p90", "p99")) + " vs "
                + " / ".join(f"{ts.get(p, float('nan')):.2f}" for p in ("p50", "p90", "p99"))
                + (" (" + " / ".join(f"x{_r(tt[p], ts[p])}" for p in ("p50", "p90", "p99")) + ")" if ts else ""))
        out.append(f"| {x['run']} | {x['max_sessions']} | {x['window_s']:.0f} | {m['rounds']} / {t.get('rounds', '–')} | "
                   f"{m['rounds_per_s']:.2f} / {t.get('rounds_per_s', float('nan')):.2f} | "
                   f"{m['new_tokens_per_s'] / 1e3:.1f}k / {t.get('new_tokens_per_s', float('nan')) / 1e3:.1f}k | "
                   f"{100 * m['prefix_hit_share']:.2f}% / {100 * t.get('prefix_hit_share', float('nan')):.2f}% | "
                   f"{ttft} |")
    return "\n".join(out)


def server_rows(run_dir: Path) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(run_dir / "server_requests.jsonl") if not r["id"].startswith(PREFLIGHT)}


def shift_ms(run_dir: Path, client: dict) -> float:
    """Workload t = 0 (first workload submit) on the server's ms axis."""
    t0 = json.loads((run_dir / "t0.json").read_text())["t0_unix_ns"] / 1e6
    return min(c["submit_unix_ms"] for c in client.values()) - t0


def mbs_of(mbs: list[dict]) -> dict[str, list[dict]]:
    out = {}
    for m in mbs:
        for q in m["requests"]:
            out.setdefault(q["id"], []).append(m)
    return out


def stages_of(lines: list[dict], nstages: int = 4) -> tuple[list[dict], list[float]]:
    """A request's per-stage compute (summed over its microbatches) and its sends (one-microbatch requests)."""
    st = []
    for s in range(nstages):
        ms = sorted((m for m in lines if m["stage"] == s), key=lambda m: m["start_ms"])
        st.append({"stage": s, "compute_ms": round(sum(m["compute_ms"] for m in ms), 3),
                   "start_ms": ms[0]["start_ms"], "end_ms": ms[-1]["start_ms"] + ms[-1]["compute_ms"],
                   "microbatches": len(ms)})
    sends = [round(st[s + 1]["start_ms"] - st[s]["end_ms"], 3) for s in range(nstages - 1)] \
        if st[0]["microbatches"] == 1 else []
    return st, sends


# --- bring-up ---------------------------------------------------------------------------------------------------------

def bringup(run_dir: Path, reference: Path | None = None) -> list[dict]:
    """One line per session (§8), from the run's client log, server logs and (if any) nsys trace."""
    client = client_rows(run_dir)
    sreq = server_rows(run_dir)
    by_req = mbs_of(read_jsonl(run_dir / "microbatches.jsonl"))
    tr = load_trace(run_dir)
    ref = {}
    if reference is not None and (reference / "server_requests.jsonl").exists():
        ref = {r["prompt_sha256"]: r for r in server_rows(reference).values()}
    sessions = {}
    for rid, c in client.items():
        sessions.setdefault(c["session"], {})[c["round"]] = (rid, c)
    out = []
    for sid in sorted(sessions):
        rounds = sessions[sid]
        (r0id, r0), (r1id, r1) = rounds[0], rounds.get(1, (None, None))
        line = {"prefix_tokens": r0["input_len"], "computed_tokens": None, "session": sid,
                "round0_ttft_ms": r0["ttft_ms"], "round1_ttft_ms": None, "round1_cached_tokens": None,
                "round1_stages": None, "round1_sends_ms": None, "token_matches_reference": None}
        if r1 is None or not r1["ok"]:
            line["failed"] = "round 1: no token"
            out.append(line)
            continue
        line.update(computed_tokens=r1["prompt_tokens"] - (r1["cached"] or 0), round1_ttft_ms=r1["ttft_ms"],
                    round1_cached_tokens=r1["cached"])
        line["append_tokens"] = r1["input_len"] + 1                      # A: the output token + the new tokens
        mine = by_req.get(r1id, [])
        if mine:
            st, sends = stages_of(mine)
            line["round1_stages"] = [{"stage": s["stage"], "compute_ms": s["compute_ms"]} for s in st]
            line["round1_sends_ms"] = sends
            line["round1_microbatches"] = st[0]["microbatches"]
            t = trace_request(tr, sorted({m["index"] for m in mine}), 4) if tr else None
            if t is not None:
                line["round1_nsys"] = {"stages_ms": [round(s["compute_ms"], 3) for s in t["stages"]],
                                       "sends_ms": [round(t["stages"][s + 1]["start_ms"] - t["stages"][s]["end_ms"], 3)
                                                    for s in range(3)]}
        line["round0_microbatches"] = len({m["index"] for m in by_req.get(r0id, [])})
        s1 = sreq.get(r1id)
        if s1 is not None:
            line["round1_token"] = s1.get("token")
            line["round1_host_ms"] = round(r1["ttft_ms"] - (st[-1]["end_ms"] - st[0]["start_ms"]), 3) if mine else None
            other = ref.get(s1["prompt_sha256"])
            if other is not None:
                line["token_matches_reference"] = other["token"] == s1["token"]
                line["reference_token"] = other["token"]
                line["reference_cached"] = other.get("cached")
        out.append(line)
    return out


def check(on: Path, off: Path) -> dict:
    """Cache on vs off, prompt for prompt: every round's token, by prompt sha256."""
    a = {r["prompt_sha256"]: r for r in server_rows(on).values()}
    b = {r["prompt_sha256"]: r for r in server_rows(off).values()}
    rows = []
    for sha, x in a.items():
        y = b.get(sha)
        rows.append({"id": x["id"], "cached_on": x.get("cached"), "cached_off": y.get("cached") if y else None,
                     "token_on": x["token"], "token_off": y["token"] if y else None,
                     "match": (y is not None and x["token"] == y["token"])})
    r1 = [r for r in rows if r["id"].endswith("round_000001")]
    return {"rounds": len(rows), "missing_in_off": sum(r["token_off"] is None for r in rows),
            "round1": len(r1), "round1_hit": sum(1 for r in r1 if (r["cached_on"] or 0) > 0),
            "round1_match": sum(r["match"] for r in r1), "all_match": sum(r["match"] for r in rows),
            "mismatches": [r for r in rows if not r["match"]]}


# --- sessions ---------------------------------------------------------------------------------------------------------

def fit_coefs() -> dict[int, np.ndarray]:
    return {f["stage"]: np.array([f["a_ms"], f["b_ms_per_1k_tokens"], f["c_ms_per_1e9_pairs"]])
            for f in json.loads((HANDOFF / "data/microbatch_fit.json").read_text())}


def microbatch_check(mbs: list[dict]) -> dict:
    """measured / fit per stage over microbatches of >= 256 tokens (data/microbatch_fit.py check), preflight out."""
    fit = fit_coefs()
    ratios = {}
    for m in mbs:
        reqs = [q for q in m["requests"] if not q["id"].startswith(PREFLIGHT)]
        if len(reqs) != len(m["requests"]):
            continue
        tokens = sum(q["append"] for q in reqs)
        if tokens < MIN_FIT_TOKENS:
            continue
        pairs = sum(q["append"] * (q["prefix"] + q["append"] / 2) for q in reqs)
        x = np.array([1.0, tokens / 1e3, pairs / 1e9])
        ratios.setdefault(m["stage"], []).append(m["compute_ms"] / float(x @ fit[m["stage"]]))
    return {s: {"microbatches": len(r), **{f"p{q}": float(np.percentile(r, q)) for q in (10, 50, 90)}}
            for s, r in sorted(ratios.items())}


def sessions(run_dir: Path) -> dict:
    client = client_rows(run_dir)
    mbs = read_jsonl(run_dir / "microbatches.jsonl")
    shift = shift_ms(run_dir, client)
    nstages = 1 + max(m["stage"] for m in mbs)
    busy = [sum(m["compute_ms"] for m in mbs if m["stage"] == s and shift <= m["start_ms"] < shift + WINDOW_MS)
            / WINDOW_MS for s in range(nstages)]
    cache = json.loads((run_dir / "cache.json").read_text()) if (run_dir / "cache.json").exists() else {}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    sreq = server_rows(run_dir)
    return {"metrics": metrics, "stage_busy": busy, "shift_ms": shift,
            "pool_tokens": cache.get("pool_tokens"), "evicted_sessions": cache.get("evicted_sessions"),
            "evicted_tokens": cache.get("evicted_tokens"), "preemptions": cache.get("preemptions"),
            "hits": cache.get("hits"), "hit_tokens": cache.get("hit_tokens"), "peak_hard_blocks": cache.get("peak_hard"),
            "dropped_stale": cache.get("dropped_stale"), "kept_sessions_at_end": cache.get("kept_sessions"),
            "prefix_shortfall_tokens": sum(c["shortfall"] for c in client.values()),
            "rounds_sent": len(client), "rounds_failed": sum(not c["ok"] for c in client.values()),
            "server_answered": len(sreq), "microbatches": len({m["index"] for m in mbs}),
            "microbatch_check": microbatch_check(mbs)}


def report_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    rep = {"name": meta["name"], "kind": meta["kind"]}
    if not (run_dir / "requests.jsonl").exists():
        rep["failed"] = "no requests.jsonl"
    elif meta["kind"] == "sessions":
        rep["sessions"] = sessions(run_dir)
    else:
        rep["bringup"] = bringup(run_dir)
    (run_dir / "report.json").write_text(json.dumps(rep) + "\n")
    return rep


# --- summary ----------------------------------------------------------------------------------------------------------

def _r(m, t):
    return f"{m / t:.2f}" if m is not None and t else "–"


def _k(n):
    return f"{n // 1024}k"


def bringup_md(lines: list[dict]) -> str:
    targets = {(t["prefix_tokens"], t["computed_tokens"]): t for t in json.loads((HANDOFF / "data/prefix_targets.json").read_text())}
    out = ["| P | A | n | hit = P | Round 0 TTFT p50 (ms) | Round 1 TTFT p50 (ms) | Stage 0 | Stage 1 | Stage 2 | Stage 3 | "
           "Each send | Token = no cache |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    groups = {}
    for x in lines:
        groups.setdefault((x["prefix_tokens"], x.get("append_tokens")), []).append(x)
    for (P, A), xs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        t = targets.get((P, A))
        ok = [x for x in xs if x.get("round1_stages")]
        med = lambda f: float(np.median([f(x) for x in ok])) if ok else None   # noqa: E731
        r0s = [x["round0_ttft_ms"] for x in xs if x["round0_ttft_ms"] is not None]
        r1s = [x["round1_ttft_ms"] for x in ok]
        r0, r1 = float(np.median(r0s)), med(lambda x: x["round1_ttft_ms"])
        tail = lambda v: f"; p90 {np.percentile(v, 90):.1f}, p99 {np.percentile(v, 99):.1f}" if v else ""  # noqa: E731
        st = [med(lambda x, s=s: x["round1_stages"][s]["compute_ms"]) for s in range(4)]
        sends = med(lambda x: float(np.mean(x["round1_sends_ms"]))) if ok and all(x["round1_sends_ms"] for x in ok) else None
        hits = sum(x["round1_cached_tokens"] == P for x in xs)
        toks = [x["token_matches_reference"] for x in xs]
        tok = f"{sum(t is True for t in toks)}/{len(toks)}" + (f" ({sum(t is None for t in toks)} unchecked)"
                                                              if any(t is None for t in toks) else "")
        tr = t["round1"] if t else None
        cell = lambda m, tv: f"{m:.1f} / {tv:.1f} (x{_r(m, tv)})" if m is not None and tv else "–"   # noqa: E731
        send = f"{sends:.3f} / {tr['microbatches'][0]['send_ms'][0]:.3f}" if sends is not None and tr else "–"
        out.append(f"| {_k(P)} | {_k(A) if A else '?'} | {len(xs)} | {hits}/{len(xs)} | "
                   f"{cell(r0, t['round0']['ttft_ms'] if t else None)}{tail(r0s)} | "
                   f"{cell(r1, tr['ttft_ms'] if tr else None)}{tail(r1s)} | "
                   + " | ".join(cell(st[s], tr["microbatches"][0]["compute_ms"][s] if tr else None) for s in range(4))
                   + f" | {send} | {tok} |")
    return "\n".join(out)


def sessions_md(runs: dict[int, dict]) -> str:
    targets = {t["max_sessions"]: t for t in json.loads((HANDOFF / "data/session_targets.json").read_text())
               if t["setting"] == "plain_hbm"}
    out = ["| N | Rounds/s | New tokens/s | Computed tokens/s | Prefix hit share | TTFT p50 / p90 / p99 (s) | "
           "Stage busy 0 / 1 / 2 / 3 |", "|---|---|---|---|---|---|---|"]
    for n, s in sorted(runs.items()):
        m, t = s["metrics"], targets.get(n)
        tt = m["ttft_s"]
        out.append(
            f"| {n} | {m['rounds_per_s']:.1f} / {t['rounds_per_s']:.1f} (x{_r(m['rounds_per_s'], t['rounds_per_s'])}) "
            f"| {m['new_tokens_per_s'] / 1e3:.1f}k / {t['new_tokens_per_s'] / 1e3:.1f}k "
            f"(x{_r(m['new_tokens_per_s'], t['new_tokens_per_s'])}) "
            f"| {m['computed_tokens_per_s'] / 1e3:.1f}k / {t['computed_tokens_per_s'] / 1e3:.1f}k "
            f"(x{_r(m['computed_tokens_per_s'], t['computed_tokens_per_s'])}) "
            f"| {100 * m['prefix_hit_share']:.2f}% / {100 * t['prefix_hit_share']:.2f}% "
            f"| {tt['p50']:.2f} / {tt['p90']:.2f} / {tt['p99']:.2f} vs {t['ttft_s']['p50']:.2f} / "
            f"{t['ttft_s']['p90']:.2f} / {t['ttft_s']['p99']:.2f} (x{_r(tt['p50'], t['ttft_s']['p50'])} / "
            f"x{_r(tt['p90'], t['ttft_s']['p90'])} / x{_r(tt['p99'], t['ttft_s']['p99'])}) "
            f"| {' / '.join(f'{100 * b:.0f}' for b in s['stage_busy'])}% vs "
            f"{' / '.join(f'{100 * b:.0f}' for b in t['stage_busy'])}% |")
    out += ["", "| N | Pool (tokens) | Evicted sessions (tokens) | Preemptions | Prefix shortfall | Rounds failed | "
            "Microbatches / fit, median per stage (p10–p90) |", "|---|---|---|---|---|---|---|"]
    for n, s in sorted(runs.items()):
        mc = s["microbatch_check"]
        fit = "; ".join(f"s{st}: x{v['p50']:.3f} ({v['p10']:.2f}–{v['p90']:.2f}, n={v['microbatches']})"
                        for st, v in mc.items())
        out.append(f"| {n} | {s['pool_tokens']:,} | {s['evicted_sessions']:,} ({s['evicted_tokens']:,}) | "
                   f"{s['preemptions']} | {s['prefix_shortfall_tokens']} | {s['rounds_failed']} | {fit} |")
    return "\n".join(out)


NOTES = "## Notes"   # prefix_cache.md from this heading on is written by hand; summarize keeps it


def summarize(root: Path, out: Path) -> Path:
    parts = ["# Prefix cache and chunked prefill (5_prefix_cache.md)", ""]
    b = root / "prefix_bringup"
    off = root / "prefix_check_off"
    if (b / "requests.jsonl").exists():
        lines = bringup(b, off if off.exists() else None)
        (out / "prefix_bringup.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        parts += ["## Prefix bring-up", "", "Measured / target (ratio), medians (p50) over each case's sessions, "
                  "TTFT also p90 and p99 over them; stage compute and sends from CUDA events (microbatches.jsonl).", "", bringup_md(lines), ""]
    on = root / "prefix_check_on"
    if (on / "server_requests.jsonl").exists() and (off / "server_requests.jsonl").exists():
        c = check(on, off)
        (out / "prefix_check.json").write_text(json.dumps(c, indent=1) + "\n")
        parts += ["## Cache on vs off", "", f"{c['round1']} round-1 prompts, {c['round1_hit']} with a hit; "
                  f"{c['round1_match']} tokens equal to the no-cache token; all rounds {c['all_match']}/{c['rounds']}.",
                  ""]
    u = root / "prefix_check_unaligned.json"
    if u.exists():
        c = json.loads(u.read_text())
        parts += ["## Cache on vs off, unaligned P", "", f"P + {c['p_shift']} (cycled) on the bring-up sessions: "
                  f"{c['sessions']} sessions, hit = P {c['hit_is_P']}, round-1 token equal to no-cache "
                  f"{c['round1_match']}, same prompt {c['same_prompt']} (bench/prefix_check.py --p-shift).", ""]
    runs = {}
    for d in sorted(root.glob("prefix_n*")):
        m = re.fullmatch(r"prefix_n(\d+)", d.name)
        if m and (d / "requests.jsonl").exists() and (d / "metrics.json").exists():
            runs[int(m.group(1))] = sessions(d)
    if runs:
        parts += ["## Sessions", "", sessions_md(runs), ""]
        (out / "prefix_sessions.json").write_text(json.dumps(runs, indent=1) + "\n")
    smokes = []
    for d in sorted(root.glob("prefix_smoke_n*")):
        m = re.match(r"prefix_smoke_n(\d+)", d.name)
        if m and (d / "requests.jsonl").exists():
            smokes.append(smoke(d, int(m.group(1))))
    if smokes:
        parts += ["## Sessions smoke", "", "Measured / target over the same window: from the first submit to the last "
                  "first token of the smoke, the target run (measured/sessions_sim/, bench/sim_rounds.py: the frozen "
                  "simulator run of data/session_targets.json) cut there by data/metrics.py's rule (rounds whose "
                  "first token comes inside it).", "", smoke_md(smokes), ""]
        (out / "prefix_smoke.json").write_text(json.dumps(smokes, indent=1) + "\n")
    path = out / "prefix_cache.md"
    notes = ""
    if path.exists() and NOTES in path.read_text():      # the hand-written part is kept
        text = path.read_text()
        notes = text[text.index(NOTES):]
    path.write_text("\n".join(parts) + ("\n" + notes if notes else ""))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "run":
        rep = report_run(a.path)
        print(json.dumps(rep)[:2000])
    else:
        print(summarize(a.path, a.out or a.path))


if __name__ == "__main__":
    sys.exit(main())
