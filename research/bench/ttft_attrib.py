"""Where each round's server TTFT goes, in a sessions run and in the frozen simulator run of the same trace, matched by
request id (the trace's row: sim request_id k is row k of sessions_2ps.csv, whose request_id names the real request).

  <python with pyarrow> -m research.bench.ttft_attrib <real run dir> <sim run dir> [--window-s 149.5] [--out JSON]

The pp-impl env has no parquet reader; run it with one that has (as research/bench/sim_rounds.py).

Per round, TTFT = arrival -> first token, split along its critical path:
- `queue`: arrival -> its first chunk is dispatched (real: the controller's plan; sim: the chunk starts on stage 0).
- `s0_wait_dispatched`: dispatched -> the first chunk starts on stage 0 (real only; 0 in the sim).
- stage 0, from the first chunk's start to the last chunk's end: `s0_own` (the compute of the microbatches that carry
  this round's chunks, co-batched tokens included), `s0_other` (other microbatches running on stage 0 in that span:
  waiting behind them) and `s0_idle` (stage 0 idle in that span).
- stages 1-3, the last chunk: `s{k}_own` (its microbatch's compute, and any wait behind the round's own earlier
  chunks), `s{k}_other` (waiting behind other microbatches
  on that stage) and `s{k}_idle` (waiting with the stage idle: transfer, host), between its end on stage k-1 and its
  end on stage k.
- `tail`: the last chunk's end on stage 3 -> the first token (real: sampling and the response; sim: 0).

Sim chunks come from raw/cost_log (stage iterations with their (prefix, append) pairs, no request ids): each round's
chain of pairs from its cache hit to its prompt length, in time order; among rounds with the same next prefix the one
whose remaining length equals the append, else the earliest admitted. A match is kept only if its last chunk ends on
stage 3 at the round's first token (within 1 us).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

EXTRA = ["pre_s0_busy", "pre_s0_idle", "pre_s0_microbatches"]   # queue + s0_wait_dispatched, split by stage 0's state
PARTS = ["queue", "s0_wait_dispatched", "s0_own", "s0_other", "s0_idle"] + \
        [f"s{k}_{p}" for k in (1, 2, 3) for p in ("own", "other", "idle")] + ["tail"]


class Busy:
    """One stage's busy intervals (sorted, non-overlapping), with each interval's microbatch key."""

    def __init__(self, ivs):
        ivs = sorted(ivs)
        self.s = [a for a, _, _ in ivs]
        self.e = [b for _, b, _ in ivs]
        self.k = [k for _, _, k in ivs]
        self.cum = np.concatenate([[0.0], np.cumsum(np.array(self.e) - np.array(self.s))])

    def busy(self, a, b, exclude=frozenset(), only=None):
        """Busy time inside [a, b], without the intervals whose key is in `exclude` (with `only`: just those keys)."""
        if b <= a:
            return 0.0
        i, j = bisect.bisect_right(self.e, a), bisect.bisect_left(self.s, b)
        t = 0.0
        for x in range(i, j):
            if self.k[x] not in exclude and (only is None or self.k[x] in only):
                t += max(0.0, min(b, self.e[x]) - max(a, self.s[x]))
        return t


def split_round(a, d, chunks, stages, busy, token):
    """chunks: keys of the round's microbatches in order; stages[s][key] = (start, end)."""
    first, last = chunks[0], chunks[-1]
    s0a, s0b = stages[0][first][0], stages[0][last][1]
    own = set(chunks)
    o = {"queue": d - a, "s0_wait_dispatched": s0a - d}
    # before the first chunk starts: stage 0 busy with other microbatches (work ahead) or idle, and how many ran
    o["pre_s0_busy"] = busy[0].busy(a, s0a, own)
    o["pre_s0_idle"] = (s0a - a) - o["pre_s0_busy"]
    o["pre_s0_microbatches"] = bisect.bisect_left(busy[0].s, s0a) - bisect.bisect_left(busy[0].s, a)
    o["s0_own"] = sum(stages[0][c][1] - stages[0][c][0] for c in chunks)
    o["s0_other"] = busy[0].busy(s0a, s0b, own)
    o["s0_idle"] = (s0b - s0a) - o["s0_own"] - o["s0_other"]
    prev = s0b
    for k in (1, 2, 3):
        st, en = stages[k][last]
        mine = busy[k].busy(prev, st, only=own - {last})    # behind its own earlier chunks: its own work
        o[f"s{k}_own"] = en - st + mine
        o[f"s{k}_other"] = busy[k].busy(prev, st, own)
        o[f"s{k}_idle"] = (st - prev) - o[f"s{k}_other"] - mine
        prev = en
    o["tail"] = token - prev
    return o


def real_rounds(run: Path):
    reqs = {}
    for line in open(run / "server_requests.jsonl"):
        r = json.loads(line)
        if r.get("first_dispatch_ms") is not None and r["id"].startswith("session_"):
            reqs[r["id"]] = r
    stages = [dict() for _ in range(4)]
    tokens = {}
    order0 = []
    chunks = defaultdict(list)
    for line in open(run / "microbatches.jsonl"):
        m = json.loads(line)
        stages[m["stage"]][m["index"]] = (m["start_ms"], m["start_ms"] + m["compute_ms"])
        if m["stage"] == 0:
            tokens[m["index"]] = sum(x["append"] for x in m["requests"])
            order0.append(m)
            for x in m["requests"]:
                chunks[x["id"]].append((m["index"], x["prefix"], x["append"]))
    busy = [Busy([(a, b, i) for i, (a, b) in st.items()]) for st in stages]
    out = {}
    for rid, r in reqs.items():
        cs = sorted(chunks.get(rid, []))
        if not cs or any(c[0] not in stages[3] for c in cs):
            continue
        keys = [c[0] for c in cs]
        o = split_round(r["arrival_ms"], r["first_dispatch_ms"], keys, stages, busy, r["token_ms"])
        o.update(ttft=r["token_ms"] - r["arrival_ms"], arrival=r["arrival_ms"], cached=r["cached"],
                 prompt=r["prompt_tokens"], chunks=[[p, n] for _, p, n in cs], mb_tokens=[tokens[k] for k in keys])
        out[rid] = o
    return out, {"stages": stages, "tokens": tokens, "order0": order0, "reqs": reqs}


def sim_rounds(run: Path, trace: Path, ids: set[str]):
    import pyarrow.parquet as pq

    names = []
    with open(trace) as f:
        for row in csv.DictReader(f):
            names.append(row["request_id"])
    t = pq.read_table(run / "raw/request_slo.parquet",
                      columns=["request_id", "completed", "arrival_time_ms", "ttft_ms", "stage_times_ms", "stage_codes",
                               "declared_prefix_tokens", "fresh_prompt_tokens", "prefix_cache_hit_tokens"]).to_pylist()
    rounds = {}
    for r in t:
        if not r["completed"]:
            continue
        admit = r["stage_times_ms"][r["stage_codes"].index(1)] if 1 in r["stage_codes"] else r["arrival_time_ms"]
        rounds[r["request_id"]] = dict(name=names[r["request_id"]], arrival=r["arrival_time_ms"], admit=admit,
                                       token=r["arrival_time_ms"] + r["ttft_ms"], hit=int(r["prefix_cache_hit_tokens"] or 0),
                                       prompt=r["declared_prefix_tokens"] + r["fresh_prompt_tokens"])
    iters = []
    for s in range(4):
        tt = pq.read_table(run / f"raw/cost_log/worker_stage_{s}.parquet",
                           columns=["iter_id", "wall_start_ms", "total_time_ms", "groups", "section"]).to_pylist()
        rows = sorted((x for x in tt if x["section"] == "iter"), key=lambda x: x["wall_start_ms"])
        iters.append([(x["wall_start_ms"], x["wall_start_ms"] + x["total_time_ms"],
                       [(p, n) for g in x["groups"] for p, n in zip(g["prefill_prefix_lens"], g["prefill_append_lens"])])
                      for x in rows])
    n0 = len(iters[0])
    assert all(len(it) >= n0 - 8 for it in iters), [len(it) for it in iters]
    for s in (1, 2, 3):          # the pipeline keeps microbatch order: the k-th iteration of every stage is one microbatch
        for k in range(min(n0, len(iters[s]))):
            assert sorted(iters[s][k][2]) == sorted(iters[0][k][2]), (s, k)
    stages = [{k: (a, b) for k, (a, b, _) in enumerate(it)} for it in iters]
    tokens = {k: sum(n for _, n in pairs) for k, (_, _, pairs) in enumerate(iters[0])}
    busy = [Busy([(a, b, k) for k, (a, b) in st.items()]) for st in stages]
    # chain each round's chunks through stage 0
    by_admit = sorted(rounds, key=lambda i: rounds[i]["admit"])
    cursor = {}
    pending = defaultdict(list)            # next prefix -> rounds waiting for a chunk there
    nxt = 0
    chunks = defaultdict(list)
    for k, (a, _, pairs) in enumerate(iters[0]):
        while nxt < len(by_admit) and rounds[by_admit[nxt]]["admit"] <= a + 0.05:   # stage_times_ms are float32
            i = by_admit[nxt]
            cursor[i] = rounds[i]["hit"]
            pending[cursor[i]].append(i)
            nxt += 1
        for p, n in pairs:
            cand = pending.get(p, [])
            if not cand:
                continue
            exact = [i for i in cand if rounds[i]["prompt"] - p == n]
            pick = exact[0] if exact else next((i for i in cand if rounds[i]["prompt"] - p > n), None)
            if pick is None:
                continue
            cand.remove(pick)
            chunks[pick].append((k, p, n))
            cursor[pick] = p + n
            if cursor[pick] < rounds[pick]["prompt"]:
                pending[cursor[pick]].append(pick)
    out, bad = {}, {}
    for i, r in rounds.items():
        if r["name"] not in ids:
            continue
        cs = chunks.get(i)
        why = ("no chunk" if not cs else "chain short of the prompt" if cursor[i] != r["prompt"]
               else "last chunk not on stage 3" if cs[-1][0] not in stages[3]
               else "last chunk ends off the first token" if abs(stages[3][cs[-1][0]][1] - r["token"]) > 1e-3 else None)
        if why:
            bad[why] = bad.get(why, 0) + 1
            if os.environ.get("TTFT_ATTRIB_DEBUG") and bad[why] <= 3:
                print(why, i, r, cs, [(stages[3].get(c[0])) for c in cs or []])
            continue
        keys = [c[0] for c in cs]
        start0 = stages[0][keys[0]][0]
        o = split_round(r["arrival"], start0, keys, stages, busy, r["token"])
        o.update(ttft=r["token"] - r["arrival"], arrival=r["arrival"], cached=r["hit"], prompt=r["prompt"],
                 chunks=[[p, n] for _, p, n in cs], mb_tokens=[tokens[k] for k in keys])
        out[r["name"]] = o
    return out, bad, {"stages": stages, "tokens": tokens}


def occupancy(rows: dict, ids: list[str], stages) -> dict:
    """Stage 0 over the span of these rounds (first arrival -> last first token): busy share, and its idle time split
    by whether some round was waiting for its first chunk (idle with backlog: scheduling) or not; per stage, the mean
    compute per microbatch and per token."""
    a0 = min(rows[i]["arrival"] for i in ids)
    a1 = max(rows[i]["arrival"] + rows[i]["ttft"] for i in ids)
    waits = sorted((rows[i]["arrival"], rows[i]["arrival"] + rows[i]["queue"] + rows[i]["s0_wait_dispatched"]) for i in ids)
    merged = []
    for a, b in waits:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        elif b > a:
            merged.append([a, b])
    backlog = Busy([(a, b, n) for n, (a, b) in enumerate(merged)])
    ivs = sorted((a, b) for a, b in stages[0].values() if b > a0 and a < a1)
    busy = sum(min(b, a1) - max(a, a0) for a, b in ivs)
    idle_backlog = 0.0
    prev = a0
    for a, b in ivs + [(a1, a1)]:
        if a > prev:
            idle_backlog += backlog.busy(prev, min(a, a1))
        prev = max(prev, b)
    span = a1 - a0
    out = {"span_s": span / 1e3, "s0_busy_share": busy / span, "s0_idle_with_backlog_s": idle_backlog / 1e3,
           "s0_idle_s": (span - busy) / 1e3, "backlog_s": sum(b - a for a, b in merged) / 1e3}
    for k in range(4):
        c = [b - a for a, b in stages[k].values() if a0 <= a < a1]
        out[f"s{k}_microbatches"] = len(c)
        out[f"s{k}_compute_mean_ms"] = float(np.mean(c)) if c else None
    return out


def compute_by_tokens(real: dict, sim: dict, t1: float) -> list[dict]:
    """Mean microbatch compute per stage and token bin (microbatches starting within t1 ms of each run's first)."""
    edges = [0, 1024, 2048, 4096, 6144, 8191, 8192]
    out = []
    for k in range(4):
        for lo, hi in zip(edges, edges[1:]):
            row = {"stage": k, "tokens": f"({lo}, {hi}]"}
            for name, d in (("real", real), ("sim", sim)):
                st = d["stages"][k]
                a0 = min(a for a, _ in st.values())
                c = [b - a for m, (a, b) in st.items() if lo < d["tokens"].get(m, -1) <= hi and a - a0 <= t1]
                row[name] = {"n": len(c), "mean_ms": float(np.mean(c)) if c else None, "sum_s": sum(c) / 1e3}
            if row["real"]["n"] and row["sim"]["n"]:
                row["ratio"] = row["real"]["mean_ms"] / row["sim"]["mean_ms"]
            out.append(row)
    return out


def replay_queue(real: dict, scale: float) -> dict:
    """The real stage-0 microbatch sequence (same order and contents) replayed as a work-conserving server with each
    microbatch's compute times `scale`: a microbatch starts once stage 0 is free, its rounds have arrived and their
    previous chunks have ended. Arrival -> first chunk start per round."""
    reqs, t, first, last = real["reqs"], 0.0, {}, {}
    for m in sorted(real["order0"], key=lambda m: m["start_ms"]):
        ids = [x["id"] for x in m["requests"] if x["id"] in reqs]
        st = max([t] + [reqs[i]["arrival_ms"] for i in ids] + [last.get(i, 0.0) for i in ids])
        t = st + m["compute_ms"] * scale
        for i in ids:
            first.setdefault(i, st)
            last[i] = t
    q = np.array([first[i] - reqs[i]["arrival_ms"] for i in first])
    return {"scale": scale, "mean_ms": float(q.mean()), **{p: float(np.percentile(q, v)) for p, v in (("p50", 50), ("p90", 90), ("p99", 99))}}


def summary(rows: dict, ids: list[str], top: list[str]) -> dict:
    out = {"n": len(ids)}
    if not ids:
        return out
    tt = np.array([rows[i]["ttft"] for i in ids])
    out["ttft_ms"] = {p: float(np.percentile(tt, q)) for p, q in (("p50", 50), ("p90", 90), ("p99", 99))}
    out["mean_ms"] = {k: float(np.mean([rows[i][k] for i in ids])) for k in PARTS + EXTRA + ["ttft"]}
    out["p90_up_mean_ms"] = {k: float(np.mean([rows[i][k] for i in top])) for k in PARTS + EXTRA + ["ttft"]}
    out["chunks_mean"] = float(np.mean([len(rows[i]["chunks"]) for i in ids]))
    out["mb_tokens_mean"] = float(np.mean([np.mean(rows[i]["mb_tokens"]) for i in ids]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real", type=Path)
    ap.add_argument("sim", type=Path)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--window-s", type=float, help="keep rounds arriving within this many seconds of the first")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    trace = a.trace or Path(json.loads((a.real / "run.json").read_text())["trace"])
    real, real_d = real_rounds(a.real)
    t0 = min(r["arrival"] for r in real.values())
    if a.window_s:
        real = {k: v for k, v in real.items() if v["arrival"] - t0 <= a.window_s * 1e3}
    sim, bad, sim_d = sim_rounds(a.sim, trace, set(real))
    ids = sorted(set(real) & set(sim))
    res = {"real_run": str(a.real.resolve()), "sim_run": str(a.sim.resolve()), "window_s": a.window_s,
           "real_rounds": len(real), "sim_matched": len(sim), "sim_unmatched": bad, "both": len(ids), "groups": {}}
    groups = {"round 0 (cold)": [i for i in ids if i.endswith("_round_000000")],
              "resumed": [i for i in ids if not i.endswith("_round_000000")]}
    for g, gi in groups.items():
        tt = np.array([real[i]["ttft"] for i in gi])
        top = [i for i in gi if real[i]["ttft"] >= np.percentile(tt, 90)]   # the real run's slowest tenth, both sides
        res["groups"][g] = {"real": summary(real, gi, top), "sim": summary(sim, gi, top)}
    print(f"real rounds {len(real)}, sim matched {len(sim)} (unmatched {bad}), both {len(ids)}")
    res["occupancy"] = {"real": occupancy(real, ids, real_d["stages"]), "sim": occupancy(sim, ids, sim_d["stages"])}
    for k in res["occupancy"]["real"]:
        print(f"   {k:28s} real {res['occupancy']['real'][k]:10.3f}   sim {res['occupancy']['sim'][k]:10.3f}")
    span = res["occupancy"]["real"]["span_s"] * 1e3
    res["compute_by_tokens"] = compute_by_tokens(real_d, sim_d, span)
    print("== microbatch compute by tokens, real / sim (mean ms, ratio, n)")
    for r in res["compute_by_tokens"]:
        if "ratio" in r:
            print(f"   stage {r['stage']} {r['tokens']:>14s}: {r['real']['mean_ms']:6.1f} / {r['sim']['mean_ms']:6.1f}  "
                  f"x{r['ratio']:.3f}  n {r['real']['n']}/{r['sim']['n']}")
    s0 = res["occupancy"]["sim"]["s0_compute_mean_ms"] / res["occupancy"]["real"]["s0_compute_mean_ms"]
    wq = np.array([real[i]["queue"] + real[i]["s0_wait_dispatched"] for i in ids])
    sq = np.array([sim[i]["queue"] for i in ids])
    res["queue_replay"] = {"real": {"mean_ms": float(wq.mean()), **{p: float(np.percentile(wq, v)) for p, v in (("p50", 50), ("p90", 90), ("p99", 99))}},
                           "sim": {"mean_ms": float(sq.mean()), **{p: float(np.percentile(sq, v)) for p, v in (("p50", 50), ("p90", 90), ("p99", 99))}},
                           "replays": [replay_queue(real_d, 1.0), replay_queue(real_d, s0)]}
    print("== arrival -> first chunk on stage 0 (mean, p50/p90/p99 ms)")
    for name, q in [("real", res["queue_replay"]["real"]), ("sim", res["queue_replay"]["sim"])] + \
                   [(f"real order, work-conserving, compute x{r['scale']:.3f}", r) for r in res["queue_replay"]["replays"]]:
        print(f"   {name:48s} {q['mean_ms']:7.1f}  {q['p50']:7.1f} / {q['p90']:7.1f} / {q['p99']:7.1f}")
    for g, d in res["groups"].items():
        r, s = d["real"], d["sim"]
        print(f"== {g}: n {r['n']}; TTFT p50/p90/p99 real {r['ttft_ms']['p50']:.0f}/{r['ttft_ms']['p90']:.0f}/"
              f"{r['ttft_ms']['p99']:.0f} ms, sim {s['ttft_ms']['p50']:.0f}/{s['ttft_ms']['p90']:.0f}/{s['ttft_ms']['p99']:.0f}; "
              f"chunks {r['chunks_mean']:.2f} / {s['chunks_mean']:.2f}, microbatch tokens {r['mb_tokens_mean']:.0f} / "
              f"{s['mb_tokens_mean']:.0f}")
        print(f"   {'part':20s} {'real mean':>10s} {'sim mean':>10s} {'diff':>8s} | {'real slow10%':>12s} {'sim same':>9s}")
        for k in PARTS + ["ttft"] + EXTRA:
            print(f"   {k:20s} {r['mean_ms'][k]:10.1f} {s['mean_ms'][k]:10.1f} {r['mean_ms'][k] - s['mean_ms'][k]:+8.1f} | "
                  f"{r['p90_up_mean_ms'][k]:12.1f} {s['p90_up_mean_ms'][k]:9.1f}")
    if a.out:
        res["rounds"] = {i: {"real": real[i], "sim": sim[i]} for i in ids}
        a.out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
