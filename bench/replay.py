"""CPU timing-model executor: the real `ppimpl.engine` (scheduler, KV, tiers) driven by a discrete-event pipeline.

Not the product: the product drives the same engine from the controller with real GPU times. This replays a handoff
trace with stage times from a model, to check the engine's decisions on CPU:
- 4 stages, FIFO each; stage 0 starts a microbatch when it is formed (when stage 0 is free, `dispatch_ahead` = 0);
  stage s pulls the activations once stage s-1 is done and its receive buffer is free:
  pull_start = max(ready on s-1, compute start of the previous microbatch on s), compute_start = max(pull_end,
  compute end of the previous microbatch on s). The send takes `send_ms(tokens)` (fit to iteration_targets.json).
  A microbatch exits when stage 3 finishes; its last chunks' tokens come out then.
- Stage time: `microbatch_fit.json` (a + b·tokens/1k + c·pairs/1e9 per stage), or for the scheduling check the
  target's own stage-0 time per microbatch, with stages 1-3 scaled by the fit's ratio.
- Tier reads land at the time the engine's channel model gives; tier writes are free (as in the target).
- Traces: `text-generation-independent` (id, input_len, output_len, arrival_time in ms) and
  `text-generation-session-execution-v2` (sessions_2ps.csv and the bring-ups), replayed as req-frontend does: a
  session starts at its arrival or when one of the N slots frees (first come first served, file order), holds its slot
  through every round and the tool waits between them (freed at the last round's token), and sends round k+1
  `tool_wait_after_ms` after round k's token.

Output (run dir): `microbatches.jsonl` and `reads.jsonl` (1_contract.md §3), `requests.jsonl` in req-frontend's shape
for `data/metrics.py`, and `summary.json`.

usage:
  python -m bench.replay check-sched [--steps plain,srpt8k,...]      scheduling check vs sched_check_targets.json
  python -m bench.replay check-force [--steps ...]                    force check vs force_check_targets.json
  python -m bench.replay sessions --step good --n 500 [--minutes 20] [--pool-tokens T] [--tiers full|tiers|ssd_heavy]
                                 [--accounting pages|tokens] [--partial-hits] --out <run dir>
  python -m bench.replay writeback --run <server run dir> [--latency-from <run>] [--cap-ms 50] [--measured]
                                 [--t-mode measured|physical|p99] [--dup-first drop|demote] [--ssd-backs]
                                 the victim chain's write-back (slot mode, the real KvManager + BalancedKVPool) on a
                                 server run's logged admissions, each op timed as the reference run measured it
                                 (`OpLatency`, stalls included): HBM waits, write volume, hit split (TIERS.md §5)
"""

import argparse
import csv
import heapq
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from bench.common import HANDOFF
from ppimpl.config import BLOCK_TOKENS, SCHEDULER_STEPS, SchedulerConfig, TierConfig
from ppimpl.engine.kv import KvManager
from ppimpl.engine.scheduler import Request, Scheduler
from ppimpl.kvpool import BalancedKVPool, KVViews

NUM_STAGES = 4
POOL_TOKENS = 61_258_368
MEMORY = {   # 1_contract.md §2.3: KV pool in tokens, DRAM per GPU
    "full": (61_258_368, 150.0),
    "tiers": (15_273_856, 37.5),
    "ssd_heavy": (7_632_640, 18.75),
}
MS = 1_000_000          # ns per ms


# --- stage times ------------------------------------------------------------------------------------------------

class StageModel:
    """Per-stage compute (ms) from the microbatch fit; sends from the iteration targets."""

    def __init__(self, handoff: Path = HANDOFF, stage_mult: list[float] | None = None):
        self.stage_mult = stage_mult or [1.0] * NUM_STAGES     # per-stage time / fit (a real run's stage balance)
        fit = json.loads((handoff / "data/microbatch_fit.json").read_text())
        self.coef = [(f["a_ms"], f["b_ms_per_1k_tokens"], f["c_ms_per_1e9_pairs"]) for f in fit]
        it = json.loads((handoff / "data/iteration_targets.json").read_text())
        (t0, s0), (t1, s1) = [(x["prompt_tokens"], x["sends"][0]["ms"]) for x in (it[0], it[-1])]
        self.send_slope = (s1 - s0) / (t1 - t0)
        self.send_base = s0 - t0 * self.send_slope

    def fit_ms(self, stage: int, chunks) -> float:
        a, b, c = self.coef[stage]
        tokens = sum(n for _, _, n in chunks)
        pairs = sum(n * (start + n / 2) for _, start, n in chunks)
        return a + b * tokens / 1e3 + c * pairs / 1e9

    def times(self, index: int, chunks) -> list[float]:
        return [self.fit_ms(s, chunks) * self.stage_mult[s] for s in range(NUM_STAGES)]

    def send_ms(self, tokens: int) -> float:
        return self.send_base + self.send_slope * tokens


class TargetStage0(StageModel):
    """The scheduling check: stage 0 takes the target's own time for microbatch i; stages 1-3 that time scaled by the
    fit's stage ratio. Past the target's microbatches (a mismatch), the fit."""

    def __init__(self, compute_ms: list[float], handoff: Path = HANDOFF):
        super().__init__(handoff)
        self.compute_ms = compute_ms

    def times(self, index: int, chunks) -> list[float]:
        fit = super().times(index, chunks)
        if index >= len(self.compute_ms):
            return fit
        t0 = self.compute_ms[index]
        return [t0 * f / fit[0] for f in fit]


# --- traces -----------------------------------------------------------------------------------------------------

@dataclass
class Round:
    req_id: str
    session: int | None
    prefix_len: int
    input_len: int
    arrival_ms: float              # independent: the request's arrival; sessions: the session's arrival
    tool_wait_ms: float = 0.0


@dataclass
class Session:
    arrival_ms: float
    rounds: list[Round] = field(default_factory=list)


def load_trace(path: Path) -> tuple[list[Round], list[Session]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if "session_id" not in rows[0]:
        return [Round(r["id"], None, 0, int(r["input_len"]), float(r["arrival_time"])) for r in rows], []
    sessions: dict[str, Session] = {}
    for r in rows:
        s = sessions.setdefault(r["session_id"], Session(float(r["arrival_time_ms"])))
        s.rounds.append(Round(r["request_id"], int(r["session_id"]), int(r["prefix_len"]), int(r["input_len"]),
                              float(r["arrival_time_ms"]), float(r["tool_wait_after_ms"])))
    return [], list(sessions.values())


# --- the replay -------------------------------------------------------------------------------------------------

EXIT, LAND, STAGE0, ARRIVE, SESSION, FORM = range(6)   # same-time order: exits, landings, stage 0, arrivals


@dataclass
class Microbatch:
    index: int
    chunks: list            # (req_id, start, tokens)
    start: list[float]      # per stage, ms
    compute: list[float]
    finishing: list         # Request objects whose last chunk this is


class Replay:
    def __init__(self, sched: Scheduler, stages: StageModel, requests: list[Round] = (), sessions: list[Session] = (),
                 max_sessions: int | None = None, horizon_ms: float | None = None, form_lead_ms: float | None = None,
                 pace_margin_ms: float | None = None, host_ms: float = 0.0):
        self.sched, self.stages = sched, stages
        self.host_ms = host_ms        # a formed plan reaches stage 0 this much later (the server's host time)
        # pacing (None: off): form a microbatch only once it would reach every later stage no more than the margin
        # before that stage frees, judged by the last microbatch's stage times
        self.pace_margin_ns = None if pace_margin_ms is None else int(pace_margin_ms * MS)
        self.pace_event_ns = -1
        self.max_sessions = max_sessions
        self.horizon_ns = None if horizon_ms is None else int(horizon_ms * MS)
        self.form_lead_ns = None if form_lead_ms is None else int(form_lead_ms * MS)
        self.events: list = []
        self._eseq = 0
        self.rounds: dict[str, Round] = {}
        self.reqs: dict[str, Request] = {}
        self.submit_ns: dict[str, int] = {}
        self.first_ns: dict[str, int] = {}
        self.microbatches: list[Microbatch] = []
        self.reads: list[dict] = []
        self.stage_free = [0.0] * NUM_STAGES       # compute end of the last microbatch per stage (ms)
        self.stage_start = [0.0] * NUM_STAGES      # compute start of the last microbatch per stage (ms)
        self.stage0_end_ns = 0
        self.sessions = list(sessions)
        self.active = 0
        self.waiting_sessions: list[int] = []
        self.next_round: dict[int, int] = {}
        for r in requests:
            self._push(int(r.arrival_ms * MS), ARRIVE, r)
        for i, s in enumerate(self.sessions):
            self._push(int(s.arrival_ms * MS), SESSION, i)

    def _push(self, t: int, kind: int, payload):
        heapq.heappush(self.events, (t, kind, self._eseq, payload))
        self._eseq += 1

    # --- workload ---------------------------------------------------------------------------------------------

    def _submit(self, r: Round, now: int):
        self.rounds[r.req_id] = r
        req = Request(r.req_id, r.prefix_len + r.input_len, arrival_ns=now, session=r.session,
                      prefix_tokens=r.prefix_len)
        self.reqs[r.req_id] = req
        self.submit_ns[r.req_id] = now
        if not self.sched.kv.fits_ever(req.num_tokens):
            raise RuntimeError(f"{r.req_id}: {req.num_tokens} tokens never fit the pool")
        self.sched.add(req, now)

    def _start_session(self, i: int, now: int):
        self.active += 1
        self.next_round[i] = 0
        self._submit(self.sessions[i].rounds[0], now)

    def _round_done(self, req: Request, now: int):
        r = self.rounds[req.req_id]
        if r.session is None:
            return
        i = self._session_index[r.session]
        k = self.next_round[i] + 1
        self.next_round[i] = k
        wait = int(r.tool_wait_ms * MS)
        if k < len(self.sessions[i].rounds):
            self._push(now + wait, ARRIVE, self.sessions[i].rounds[k])
        else:   # the session frees its slot at its last round's token (the simulator; req-frontend since 365fba07)
            self._push(now, SESSION, -1 - i)

    # --- pipeline ---------------------------------------------------------------------------------------------

    def _launch(self, chunks, now: int):
        index = len(self.microbatches)
        spec = [(c.req.req_id, c.start, c.num_tokens) for c in chunks]
        compute = self.stages.times(index, spec)
        tokens = sum(n for _, _, n in spec)
        send = self.stages.send_ms(tokens)
        start = [0.0] * NUM_STAGES
        start[0] = max(now / MS + self.host_ms, self.stage_free[0])
        ready = start[0] + compute[0]
        for s in range(1, NUM_STAGES):
            pull_start = max(ready, self.stage_start[s])
            start[s] = max(pull_start + send, self.stage_free[s])
            ready = start[s] + compute[s]
        for s in range(NUM_STAGES):
            self.stage_start[s] = start[s]
            self.stage_free[s] = start[s] + compute[s]
        mb = Microbatch(index, spec, start, compute, [c.req for c in chunks if c.is_last])
        self.microbatches.append(mb)
        for c in chunks:
            if c.start == c.req.cached and c.req.prefix_tokens > 0 and c.req.session is not None:
                self._log_read(c.req)
        self.stage0_end_ns = round(self.stage_free[0] * MS)
        self._push(self.stage0_end_ns, STAGE0, mb)
        self._push(round(self.stage_free[-1] * MS), EXIT, mb)
        if self.form_lead_ns is not None:
            self._push(max(now, self.stage0_end_ns - self.form_lead_ns), FORM, None)

    def _log_read(self, req: Request):
        """reads.jsonl: one line per round with a prefix, when its first chunk is scheduled."""
        tiers = {"dram": 0, "ssd": 0}
        if req.tier:
            tiers[req.tier] = req.tier_tokens
        t = req.read_start_ns if req.read_start_ns is not None else req.admit_ns
        land = req.read_land_ns if req.read_land_ns is not None else req.admit_ns
        self.reads.append(dict(id=req.req_id, arrival_ms=self.submit_ns[req.req_id] / MS,
                               prefix_tokens=req.prefix_tokens, hbm_tokens=req.hbm_tokens,
                               dram_tokens=tiers["dram"], ssd_tokens=tiers["ssd"],
                               read_start_ms=t / MS, read_land_ms=land / MS))

    def _pace_ns(self) -> int:
        """Earliest stage-0 start at which the next microbatch would not queue before a later stage (by more than the
        margin), taking the last microbatch's stage times as the next one's."""
        if self.pace_margin_ns is None or not self.microbatches:
            return 0
        mb = self.microbatches[-1]
        send = self.stages.send_ms(sum(n for _, _, n in mb.chunks))
        t, ahead = 0.0, 0.0
        for s in range(1, NUM_STAGES):
            ahead += mb.compute[s - 1] + send
            t = max(t, self.stage_free[s] - ahead)
        return round((t - self.host_ms) * MS) - self.pace_margin_ns

    def _may_form(self, now: int) -> bool:
        pace = self._pace_ns()
        if now < pace:
            if self.pace_event_ns != pace:
                self.pace_event_ns = pace
                self._push(pace, FORM, None)
            return False
        if self.sched.on_stage0 == 0:
            return True
        return self.form_lead_ns is None or now >= self.stage0_end_ns - self.form_lead_ns

    def _form(self, now: int):
        while self._may_form(now):
            reads = self.sched.stats["reads"]
            chunks = self.sched.next_microbatch(now)
            if self.sched.stats["reads"] > reads:      # the reads it started, in start order: land at their eta
                for r in list(self.sched.reading.values())[reads - self.sched.stats["reads"]:]:
                    self._push(r.read_land_ns, LAND, r.req_id)
            if chunks is None:
                break
            self._launch(chunks, now)

    def run(self):
        self._session_index = {s.rounds[0].session: i for i, s in enumerate(self.sessions)}
        while self.events:
            now = self.events[0][0]
            if self.horizon_ns is not None and now >= self.horizon_ns:
                break
            while self.events and self.events[0][0] == now:
                _, kind, _, payload = heapq.heappop(self.events)
                if kind == EXIT:
                    self.sched.stage_done(NUM_STAGES - 1, NUM_STAGES - 1)
                    for req in payload.finishing:
                        self.first_ns[req.req_id] = now
                        self.sched.finish(req)
                        self._round_done(req, now)
                elif kind == LAND:
                    self.sched.read_landed(payload, now)
                elif kind == STAGE0:
                    self.sched.stage_done(0, NUM_STAGES - 1)
                elif kind == ARRIVE:
                    self._submit(payload, now)
                elif kind == SESSION:
                    if payload >= 0:
                        self.waiting_sessions.append(payload)
                    else:
                        self.active -= 1
                    while self.waiting_sessions and (self.max_sessions is None or self.active < self.max_sessions):
                        self._start_session(self.waiting_sessions.pop(0), now)
            self._form(now)
        return self

    # --- outputs ----------------------------------------------------------------------------------------------

    def ttft_ms(self) -> dict[str, float]:
        return {k: (self.first_ns[k] - self.submit_ns[k]) / MS for k in self.first_ns}

    def write(self, out: Path, window_ms: float | None = None) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "microbatches.jsonl", "w") as f:
            for mb in self.microbatches:
                reqs = [dict(id=i, prefix=s, append=n) for i, s, n in mb.chunks]
                for s in range(NUM_STAGES):
                    f.write(json.dumps(dict(stage=s, index=mb.index, start_ms=round(mb.start[s], 4),
                                            compute_ms=round(mb.compute[s], 4), requests=reqs)) + "\n")
        with open(out / "reads.jsonl", "w") as f:
            for r in self.reads:
                f.write(json.dumps(r) + "\n")
        with open(out / "requests.jsonl", "w") as f:
            for k, req in self.reqs.items():
                r = self.rounds[k]
                o = dict(request_id=k, submit_timestamp=self.submit_ns[k] / 1e9, post_timestamp=self.submit_ns[k] / 1e9,
                         status="SUCCESS" if k in self.first_ns else "TIMEOUT",
                         server_usage=dict(prompt_tokens=req.num_tokens, cached_prompt_tokens=req.cached))
                if k in self.first_ns:
                    o["first_token_id_ms"] = (self.first_ns[k] - self.submit_ns[k]) / MS
                f.write(json.dumps(dict(outcome=o, source=dict(data=dict(input_len=r.input_len,
                                                                         prefix_len=r.prefix_len)))) + "\n")
        summary = self.summary(window_ms)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        return summary

    def summary(self, window_ms: float | None) -> dict:
        end = window_ms if window_ms is not None else max(mb.start[-1] + mb.compute[-1] for mb in self.microbatches)
        busy = [0.0] * NUM_STAGES
        for mb in self.microbatches:
            for s in range(NUM_STAGES):
                if mb.start[s] < end:
                    busy[s] += mb.compute[s]
        lo, hi = 600_000.0, 1_200_000.0
        share = defaultdict(int)
        for r in self.reads:
            if lo <= r["arrival_ms"] < hi:
                for k in ("prefix_tokens", "hbm_tokens", "dram_tokens", "ssd_tokens"):
                    share[k] += r[k]
        kv = self.sched.kv
        return dict(
            microbatches=len(self.microbatches),
            stage_busy=[round(b / end, 4) for b in busy],
            prefix_read_share={k: round(share[f"{k}_tokens"] / share["prefix_tokens"], 4)
                               for k in ("hbm", "dram", "ssd")} if share["prefix_tokens"] else None,
            pool=dict(capacity=kv.capacity, accounting=kv.accounting, **kv.stats),
            scheduler=self.sched.stats,
            tiers=self.sched.pool.summary() if self.sched.pool is not None else None,
            unanswered=len(self.reqs) - len(self.first_ns),
        )


# --- builders -----------------------------------------------------------------------------------------------------

def make_scheduler(step: str | SchedulerConfig, pool_tokens: int = POOL_TOKENS, keep_sessions: bool = True,
                   tiers: TierConfig | None = None, accounting: str = "pages", partial_hits: bool = False,
                   dispatch_ahead: int = 0, max_in_flight: int = NUM_STAGES) -> Scheduler:
    cfg = SCHEDULER_STEPS[step] if isinstance(step, str) else step
    kv = KvManager(pool_tokens // BLOCK_TOKENS, keep_sessions=keep_sessions, accounting=accounting,
                   partial_hits=partial_hits)
    pool = BalancedKVPool(tiers, partial_hits=partial_hits) if tiers is not None else None
    return Scheduler(kv, cfg, pool, max_in_flight=max_in_flight, dispatch_ahead=dispatch_ahead)


def check_sched(step: str, handoff: Path = HANDOFF, dispatch_ahead: int = 0,
                form_lead_ms: float | None = None) -> tuple[list[str], dict]:
    """Replay bringup/sched_check.csv with the target's stage times; return the differences from
    sched_check_targets.json (microbatch chunk for chunk, start, TTFT ±10%) and the replay's TTFTs."""
    target = json.loads((handoff / "data/sched_check_targets.json").read_text())[step]
    requests, _ = load_trace(handoff / "data/bringup/sched_check.csv")
    stages = TargetStage0([m["compute_ms"] for m in target["microbatches"]], handoff)
    rp = Replay(make_scheduler(step, keep_sessions=False, dispatch_ahead=dispatch_ahead), stages, requests,
                form_lead_ms=form_lead_ms).run()
    diffs = []
    key = lambda c: int(c[0].split("_")[1])
    start_dev = 0.0
    for i, (want, got) in enumerate(zip(target["microbatches"], rp.microbatches)):
        mine = sorted([list(c) for c in got.chunks], key=key)
        if mine != want["chunks"]:
            diffs.append(f"mb {i}: chunks {mine} != target {want['chunks']}")
        start_dev = max(start_dev, abs(got.start[0] - want["start_ms"]))
    if len(rp.microbatches) != len(target["microbatches"]):
        diffs.append(f"{len(rp.microbatches)} microbatches != target {len(target['microbatches'])}")
    ttft = rp.ttft_ms()
    for k, want in target["ttft_ms"].items():
        if abs(ttft[k] / want - 1) > 0.10:
            diffs.append(f"TTFT {k} {ttft[k]:.1f} ms vs target {want} (>10%)")
    ttft["_max_start_dev_ms"] = start_dev
    ttft["_max_ttft_dev"] = max(abs(ttft[k] / w - 1) for k, w in target["ttft_ms"].items())
    return diffs, ttft


def run_force(step: str, handoff: Path = HANDOFF) -> dict:
    import numpy as np
    requests, _ = load_trace(handoff / "data/bringup/force_check.csv")
    rp = Replay(make_scheduler(step, keep_sessions=False), StageModel(handoff), requests).run()
    ttft = rp.ttft_ms()
    short = np.array([v for k, v in ttft.items() if k != "force_long"]) / 1e3
    last = max(rp.first_ns.values()) / 1e9
    return dict(long_ttft_s=round(ttft["force_long"] / 1e3, 3),
                short_ttft_s={p: round(float(np.percentile(short, q)), 3) for p, q in (("p50", 50), ("p90", 90),
                                                                                      ("p99", 99))}
                | {"max": round(float(short.max()), 3)},
                last_token_s=round(last, 3), microbatches=len(rp.microbatches))


def check_force(step: str, handoff: Path = HANDOFF) -> tuple[list[str], dict]:
    """6_scheduler.md §7: 32k TTFT ±10% (step 4: in [60, 65] s); 1k p50/p90/p99 ±10%; last token ±10%."""
    want = json.loads((handoff / "data/force_check_targets.json").read_text())[step]
    got = run_force(step, handoff)
    diffs = []
    near = lambda a, b: abs(a / b - 1) <= 0.10
    if not near(got["long_ttft_s"], want["long_ttft_s"]):
        diffs.append(f"32k TTFT {got['long_ttft_s']} vs {want['long_ttft_s']}")
    if step == "good" and not 60 <= got["long_ttft_s"] <= 65:
        diffs.append(f"32k TTFT {got['long_ttft_s']} not in [60, 65] s")
    for p in ("p50", "p90", "p99"):
        if not near(got["short_ttft_s"][p], want["short_ttft_s"][p]):
            diffs.append(f"1k {p} {got['short_ttft_s'][p]} vs {want['short_ttft_s'][p]}")
    if not near(got["last_token_s"], want["last_token_s"]):
        diffs.append(f"last token {got['last_token_s']} vs {want['last_token_s']}")
    return diffs, got


def run_sessions(step: str, n: int | None, out: Path, minutes: float = 20.0, memory: str = "full",
                 tiers: bool = False, accounting: str = "pages", partial_hits: bool = False,
                 trace: Path | None = None, pool_tokens: int | None = None, tier_cfg: TierConfig | None = None,
                 dispatch_ahead: int = 0, form_lead_ms: float | None = None, handoff: Path = HANDOFF,
                 max_in_flight: int = NUM_STAGES,
                 stage_mult: list[float] | None = None, pace_margin_ms: float | None = None,
                 host_ms: float = 0.0) -> dict:
    pool, dram = MEMORY[memory]
    if tiers and tier_cfg is None:
        tier_cfg = TierConfig(dram_gb=dram)
    sched = make_scheduler(step, pool_tokens or pool, True, tier_cfg if tiers else None, accounting, partial_hits,
                           dispatch_ahead, max_in_flight)
    _, sessions = load_trace(trace or handoff / "data/sessions_2ps.csv")
    window = minutes * 60e3
    rp = Replay(sched, StageModel(handoff, stage_mult), sessions=sessions, max_sessions=n, horizon_ms=window,
                form_lead_ms=form_lead_ms,
                pace_margin_ms=pace_margin_ms, host_ms=host_ms).run()
    summary = rp.write(out, window)
    m = subprocess.run([sys.executable, str(handoff / "data/metrics.py"), str(out / "requests.jsonl"),
                        "--minutes", str(minutes)], capture_output=True, text=True, check=True)
    summary["metrics"] = json.loads(m.stdout)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def run_bringup(kind: str, variant: str = "dram", handoff: Path = HANDOFF) -> list[dict]:
    """Replay a bring-up and set it beside its target per (P, A) case: `prefix` (plain scheduler, HBM; targets
    prefix_targets.json) or `tier` (design scheduler, 1,234,944-token pool; `variant` dram_unbal / ssd_unbal = own
    stage storage with DRAM only or SSD only, hbm = no tier, hbm_hit = full pool; tier_targets.json)."""
    if kind == "prefix":
        sched = make_scheduler("plain")
        trace, targets = "prefix_sessions.csv", json.loads((handoff / "data/prefix_targets.json").read_text())
    else:
        pool = POOL_TOKENS if variant == "hbm_hit" else 1_234_944
        tc = {"dram_unbal": TierConfig(ssd_gb=None), "ssd_unbal": TierConfig(dram_gb=None)}.get(variant)
        sched = make_scheduler("good", pool, True, tc)
        trace, targets = "tier_sessions.csv", json.loads((handoff / "data/tier_targets.json").read_text())[variant]
    _, sessions = load_trace(handoff / "data/bringup" / trace)
    rp = Replay(sched, StageModel(handoff), sessions=sessions).run()
    ttft = rp.ttft_ms()
    rows = []
    for t in targets:
        P = t["prefix_tokens"]
        A = t.get("computed_tokens", 1024)
        cases = [s for s in sessions if len(s.rounds) == 2 and s.rounds[0].input_len == P
                 and s.rounds[1].input_len + 1 == A]
        r0 = [ttft[s.rounds[0].req_id] for s in cases]
        r1 = [ttft[s.rounds[1].req_id] for s in cases]
        reqs = [rp.reqs[s.rounds[1].req_id] for s in cases]
        read = [(q.read_land_ns - q.read_start_ns) / MS for q in reqs if q.read_start_ns is not None]
        want0 = t["round0"]["ttft_ms"] if kind == "prefix" else t["round0_ttft_ms"]
        want1 = t["round1"]["ttft_ms"] if kind == "prefix" else t["round1_ttft_ms"]
        rows.append(dict(P=P, A=A, cached=sorted({q.cached for q in reqs}), tier=sorted({str(q.tier) for q in reqs}),
                         round0_ms=round(max(r0), 1), round0_target=want0, round1_ms=round(max(r1), 1),
                         round1_target=want1, read_ms=round(max(read), 2) if read else 0.0,
                         read_target=t.get("read_ms", 0.0)))
    return rows


# --- write-back model on a server run's admissions (TIERS.md §5) ----------------------------------------------------

OP_KINDS = {("write", "dram"): "write", ("write", "ssd"): "demote", ("demote", "ssd"): "demote",
            ("read", "dram"): "read_dram", ("read", "ssd"): "read_ssd"}


class OpLatency:
    """Tier op times (sent -> landed) as a server run measured them (`tier_ops.jsonl`), stalls and queueing included:
    per kind (HBM -> DRAM write, DRAM -> SSD demote, DRAM read, SSD read) a base a + b·slots (least squares on the
    per-slot-count medians) plus the excess over it of the measured op of that kind sent nearest in workload time
    (`at`: ns since the workload's first submit). Driving another run (or policy) with it replays the reference's
    stalls when they happened. `phys_ms`: the write's physical time, the bytes at the link's measured write rate
    (per GPU, the PCIe hop of `tier_trace_s*.jsonl`) plus the per-op overhead (median rest of the writes before the
    first > 50 ms one: no stall, no queue). `cap_ms`: excess over the base capped (the stalls removed: what fixing
    them would give)."""

    def __init__(self, run: Path, cap_ms: float | None = None):
        from bench.prefix import client_rows, shift_ms

        self.run, self.cap_ms = run.name, cap_ms
        self.shift = shift_ms(run, client_rows(run))
        ops = [json.loads(x) for x in (run / "tier_ops.jsonl").read_text().splitlines()]
        sb = json.loads((run / "tiers.json").read_text())["slot_bytes"]
        self.kinds = {}
        for k in set(OP_KINDS.values()):
            xs = sorted((o["sent_ms"] - self.shift, o["landed_ms"] - o["sent_ms"], o["slots"]) for o in ops
                        if o.get("landed_ms") is not None and OP_KINDS.get((o["kind"], o["tier"])) == k)
            if not xs:
                continue
            at, dur, n = (np.array(c, dtype=float) for c in zip(*xs))
            ns = np.unique(n)
            med = np.array([np.median(dur[n == v]) for v in ns])
            w = np.array([np.sum(n == v) for v in ns])
            b, a = np.polyfit(ns, med, 1, w=np.sqrt(w)) if len(ns) > 1 else (0.0, float(med[0]))
            ex = dur - (a + b * n)
            self.kinds[k] = (at, ex if cap_ms is None else np.minimum(ex, cap_ms), a, b)
        trace = [json.loads(x) for f in sorted(run.glob("tier_trace_s*.jsonl")) for x in f.read_text().splitlines()]
        rate = [x["bytes"] / (x["gpu_end_us"] - x["gpu_start_us"]) / 1e3 for o in trace if o["kind"] == "write"
                for x in o["batches"] if x["stream"] == "write" and x["gpu_end_us"] > x["gpu_start_us"]]
        self.write_gb_s = float(np.median(rate)) if rate else 50.0
        w = sorted((o for o in ops if o["kind"] == "write" and o.get("landed_ms") is not None), key=lambda o: o["sent_ms"])
        first = next((i for i, o in enumerate(w) if o["landed_ms"] - o["sent_ms"] > 50), len(w))
        w = w[:first]
        self.write_overhead_ms = float(np.median([o["landed_ms"] - o["sent_ms"] - max(o["bytes"]) / self.write_gb_s / 1e6
                                                  for o in w]))
        self.slot_bytes = sb

    def ms(self, kind: str, slots: int, at_ns: int) -> float:
        at, ex, a, b = self.kinds[kind]
        i = min(int(np.searchsorted(at, at_ns / MS)), len(at) - 1)
        if i and at_ns / MS - at[i - 1] < at[i] - at_ns / MS:
            i -= 1
        base = a + b * slots
        return max(base + ex[i], 0.25 * base)

    def phys_ms(self, slots: int) -> float:
        return self.write_overhead_ms + slots * self.slot_bytes / self.write_gb_s / 1e6

    def describe(self) -> dict:
        return dict(ref=self.run, cap_ms=self.cap_ms, write_gb_s=round(self.write_gb_s, 1), write_overhead_ms=round(self.write_overhead_ms, 2),
                    base_ms={k: [round(v[2], 2), round(v[3], 3)] for k, v in self.kinds.items()})


def hit_share(rows: list[dict], lo: float) -> dict:
    """Hit tokens by where they came from (rounds arriving in minutes 10-20, as `bench.tiers.session_vs_target`)."""
    got = {"hbm": 0, "dram": 0, "ssd": 0}
    for r in rows:
        if lo <= r["arrival_ms"] < lo + 600e3 and r.get("cached"):
            t = r.get("tier_tokens", 0) if r.get("tier") else 0
            got["hbm"] += r["cached"] - t
            if t:
                got[r["tier"]] += t
    hit = max(sum(got.values()), 1)
    return {k: round(100 * v / hit, 1) for k, v in got.items()}


def measured_writeback(run: Path) -> dict:
    """What the server run measured, in `run_writeback`'s terms."""
    from bench.prefix import client_rows, server_rows, shift_ms

    c = json.loads((run / "cache.json").read_text())
    tiers = {t["tier"]: t for t in c["tiers"]}
    wb = c.get("write_back") or {}
    return dict(run=run.name, policy="measured", hbm_waits=(wb.get("hbm") or {}).get("evict_waits"),
                blocked_by_write=c.get("blocked_by_write"),
                write_gb={k: round(v["written_bytes"] / 1e9) for k, v in tiers.items()},
                reads={k: v["reads"] for k, v in tiers.items()},
                hit_share=hit_share(list(server_rows(run).values()), shift_ms(run, client_rows(run)) + 600e3),
                R=(wb.get("hbm") or {}).get("headroom"), T_ms=(wb.get("hbm") or {}).get("write_ms"))


def run_writeback(run: Path, lat: OpLatency | None = None, t_mode: str = "measured", dup_first: str | None = None,
                  ssd_backs: bool = False) -> dict:
    """The server's pool policy (slot mode: `KvManager` with `BalancedKVPool.hbm_level`, the DRAM/SSD index) driven by
    a server run's logged rounds: each is admitted at its logged admission (read start or first dispatch) if it fits
    then, else as soon as it fits (re-checked at every op landing and release; a later round waits behind it), and
    releases its context at its logged token time plus that wait. Every op takes the time `lat` (default: this run's
    own `OpLatency`) gives it at its send, in workload time; drops take none.

    Variants of the write-back policy, applied from here (the pool's code is the server's):
    - `t_mode` "measured": T is the longest write measured (the server's); "physical": T is the longest write's
      `OpLatency.phys_ms` (bytes / link rate + per-op overhead; no stall or queue inside T); "p99": the 99th
      percentile of the last 4096 writes measured.
    - `dup_first`: DRAM evicts first (LRU among them) the entries that are clean copies of contexts HBM keeps;
      "drop": without a demote (HBM's entry is dirty again: `TierManager._removed`), "demote": as any other victim.
    - `ssd_backs`: a context DRAM lets go of that SSD then holds whole stays clean in HBM (backed by SSD); otherwise
      (the server) HBM's entry is dirty again and HBM writes it to DRAM again.
    Policies without `BalancedKVPool(clock=)` (the 0.9 watermarks before 4106992) run with their own write-back."""
    import inspect
    from bench.prefix import client_rows, shift_ms

    lat = lat or OpLatency(run)
    shift = shift_ms(run, client_rows(run))
    cfg = json.loads((run / "tiers.json").read_text())
    srv = yaml.safe_load((run / "server_config.yaml").read_text())
    t = srv["tiers"]
    sb = cfg["slot_bytes"]
    tc = TierConfig(dram_gb=t["dram_gb"], ssd_gb=t["ssd_gb"], slot_bytes=sb, state_slots=cfg["state_slots"],
                    staging_gb=t.get("staging_gb", 8.0), dram_read_gb_s=t["dram_read_gb_s"],
                    ssd_read_gb_s=t["ssd_read_gb_s"], max_read_wait_ms=None)
    now = [0]
    new = "clock" in inspect.signature(BalancedKVPool).parameters
    pool = BalancedKVPool(tc, clock=lambda: now[0]) if new else BalancedKVPool(tc)
    kv = KvManager(json.loads((run / "pool.json").read_text())["num_blocks"], keep_sessions=True)
    kv.lower = pool.hbm_level(kv)
    hwb = pool.hbm.wb
    phys = [0.0]
    if new and t_mode == "p99":
        orig_done, seen = hwb.done, []

        def done(key, token, started_ns=None):
            w = hwb.writing.get(key)
            if w is not None and w[0] is token:
                seen.append(now[0] - (w[2] if started_ns is None else started_ns))
            orig_done(key, token, started_ns)
            hwb.write_ns = int(np.percentile(seen[-4096:], 99)) if seen else 0
        hwb.done = done
    if new and t_mode == "physical":
        orig_done, t_phys = hwb.done, [0]

        def done(key, token, started_ns=None):
            orig_done(key, token, started_ns)
            t_phys[0] = max(t_phys[0], int(phys[0] * MS))
            hwb.write_ns = t_phys[0]
        hwb.done = done
    if dup_first:
        m, dram = pool.manager, pool.manager.tiers[0]
        dups = set()
        orig_store, orig_evict = dram.store, m._evict

        def store(key, tokens, content, keeps):
            dups.clear()
            dups.update(k for k in dram.entries if k != key and k in kv.kept and hwb.evictable(k))
            if not dups:
                return orig_store(key, tokens, content, keeps)
            order = list(dram.entries)
            for k in reversed(order):
                if k in dups:
                    dram.entries.move_to_end(k, last=False)
            out = orig_store(key, tokens, content, keeps)
            rank = {k: i for i, k in enumerate(order)}
            rest = sorted(dram.entries.items(), key=lambda kv_: rank.get(kv_[0], len(rank)))
            dram.entries.clear()
            dram.entries.update(rest)
            return out

        def evict(i, key, e):
            if i == 0 and key in dups and dup_first == "drop":
                m.wb[0].remove(key)
                m._removed(0, key)
                m._emit("drop", dram.name, key)
                return
            orig_evict(i, key, e)
        dram.store, m._evict = store, evict
    if ssd_backs:
        m, orig_removed = pool.manager, pool.manager._removed

        def removed(i, key):
            k, below = kv.kept.get(key), m.tiers[1].entries.get(key) if len(m.tiers) > 1 else None
            if i == 0 and k is not None and below is not None and below.tokens >= k.tokens:
                m.wb[0].remove(key)
                return
            orig_removed(i, key)
        m._removed = removed
    rows = [json.loads(x) for x in (run / "server_requests.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["id"].startswith("session_") and "first_dispatch_ms" in r]
    ev, seq, last = [], 0, {}
    for r in sorted(rows, key=lambda r: r["seq"]):
        s = int(r["id"].split("_")[1])
        resumes = r["hit_kind"] in ("full", "partial", "evicted") and s in last
        req = Request(r["id"], r["prompt_tokens"], session=s, prefix_tokens=last[s] + 1 if resumes else 0)
        last[s] = r["prompt_tokens"]
        at = int((r.get("read_start_ms") or r["first_dispatch_ms"]) * MS)
        ev.append((at, seq, "arrive", (req, int(r["token_ms"] * MS) - at, r["arrival_ms"])))
        seq += 1
    heapq.heapify(ev)
    pending, waits, late_at, got, landing = [], [], [], [], {}

    def admit():
        nonlocal seq
        while pending and kv.fits(pending[0][0]):
            req, hold, arrival, at = pending.pop(0)
            hbm = kv.allocate(req)
            hit = pool.lookup(req.session, req, now[0])
            row = dict(arrival_ms=arrival, cached=hbm)
            if hit is not None and hit.tokens > hbm:
                a = kv.held[req.req_id]
                pool.load(req.session, hit, KVViews(a.kv_pages, a.state_blocks, hit.tokens, hbm), now[0], req.req_id)
                row |= dict(cached=hit.tokens, tier=hit.tier, tier_tokens=hit.tokens - hbm)
            got.append(row)
            waits.append((now[0] - at) / MS)
            if now[0] > at:
                late_at.append(round(at / 1e9, 1))
            heapq.heappush(ev, (now[0] + hold, seq, "release", req))
            seq += 1

    while ev:
        now[0], _, kind, x = heapq.heappop(ev)
        if kind == "arrive":
            pending.append((*x, now[0]))
        elif kind == "release":
            kv.release(x)
        else:
            phys[0] = landing.pop(x, 0.0)
            pool.complete(x, now[0])
        admit()
        for o in pool.ready_ops():
            n = len(o.slots) + len(o.state_slots)
            k = OP_KINDS.get((o.op, o.tier)) if o.moves_bytes else None
            done = now[0] + (int(lat.ms(k, n, now[0] - int(shift * MS)) * MS) if k else 0)
            if k == "write":
                landing[o.op_id] = lat.phys_ms(n)
            heapq.heappush(ev, (done, seq, "op", o.op_id))
            seq += 1
    w = np.array(waits)
    late = w[w > 0]
    tiers = {x["tier"]: x for x in pool.summary()}
    wb = pool.write_back_stats()
    hb = wb.get("hbm") or {}
    return dict(run=run.name, rounds=len(w), hbm_waits=hb.get("evict_waits"), waits=int(len(late)),
                wait_s=round(float(late.sum()) / 1e3, 2), wait_ms_max=round(float(late.max()), 1) if len(late) else 0.0,
                waits_at_s=late_at[:10], write_gb={k: round(v["written_bytes"] / 1e9) for k, v in tiers.items()},
                reads={k: v["reads"] for k, v in tiers.items()}, hit_share=hit_share(got, shift + 600e3),
                R=hb.get("headroom"), T_ms=hb.get("write_ms"), write_back=wb, blocked_by_write=kv.stats["blocked_by_write"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check-sched", "check-force"):
        p = sub.add_parser(name)
        p.add_argument("--steps", default=",".join(SCHEDULER_STEPS))
        p.add_argument("--dispatch-ahead", type=int, default=0)
        p.add_argument("--form-lead-ms", type=float)
    p = sub.add_parser("bringup")
    p.add_argument("kind", choices=["prefix", "tier"])
    p.add_argument("--variant", default="dram_unbal")
    p = sub.add_parser("sessions")
    p.add_argument("--step", default="good")
    p.add_argument("--n", type=int)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--memory", choices=list(MEMORY), default="full")
    p.add_argument("--tiers", action="store_true")
    p.add_argument("--no-ssd", action="store_true")
    p.add_argument("--accounting", choices=["pages", "tokens"], default="pages")
    p.add_argument("--partial-hits", action="store_true", help="the simulator's hit rule (not realizable)")
    p.add_argument("--trace", type=Path)
    p.add_argument("--pool-tokens", type=int)
    p.add_argument("--dispatch-ahead", type=int, default=0)
    p.add_argument("--form-lead-ms", type=float)
    p.add_argument("--max-in-flight", type=int, default=NUM_STAGES)
    p.add_argument("--stage-mult", help="per-stage time / fit, comma-separated (default 1 each)")
    p.add_argument("--pace-margin-ms", type=float, help="pace microbatch formation to the later stages (default off)")
    p.add_argument("--host-ms", type=float, default=0.0, help="a formed plan reaches stage 0 this much later")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("writeback")
    p.add_argument("--run", type=Path, required=True, action="append", help="a server run dir (repeatable)")
    p.add_argument("--latency-from", type=Path, help="the run whose measured op times drive the model (default each)")
    p.add_argument("--cap-ms", type=float, help="cap the measured op times' excess over their base (no stalls)")
    p.add_argument("--t-mode", choices=["measured", "physical", "p99"], default="measured")
    p.add_argument("--dup-first", choices=["drop", "demote"], help="DRAM evicts clean copies of HBM-kept contexts first")
    p.add_argument("--ssd-backs", action="store_true", help="HBM stays clean when SSD holds what DRAM let go of")
    p.add_argument("--measured", action="store_true", help="also print what each run measured")
    a = ap.parse_args()
    if a.cmd == "check-sched":
        for step in a.steps.split(","):
            diffs, t = check_sched(step, dispatch_ahead=a.dispatch_ahead, form_lead_ms=a.form_lead_ms)
            print(f"{step:7} {'PASS' if not diffs else 'FAIL'}  stage-0 start within {t['_max_start_dev_ms']:.3f} ms,"
                  f" TTFT within {100 * t['_max_ttft_dev']:.2f}%", *diffs[:10], sep="\n  ")
    elif a.cmd == "check-force":
        for step in a.steps.split(","):
            diffs, got = check_force(step)
            print(f"{step:7} {'PASS' if not diffs else 'FAIL'} {json.dumps(got)}", *diffs, sep="\n  ")
    elif a.cmd == "writeback":
        lat = OpLatency(a.latency_from or a.run[0], a.cap_ms) if a.latency_from or a.cap_ms is not None else None
        if lat:
            print(json.dumps(lat.describe()), flush=True)
        for run in a.run:
            if a.measured:
                print(json.dumps(measured_writeback(run)), flush=True)
            print(json.dumps(dict(run_writeback(run, lat, a.t_mode, a.dup_first, a.ssd_backs), t_mode=a.t_mode,
                                  dup_first=a.dup_first, ssd_backs=a.ssd_backs)), flush=True)
    elif a.cmd == "bringup":
        for row in run_bringup(a.kind, a.variant):
            print(json.dumps(row))
    else:
        mem = MEMORY[a.memory]
        tc = TierConfig(dram_gb=mem[1], ssd_gb=None if a.no_ssd else 256.0)
        s = run_sessions(a.step, a.n, a.out, a.minutes, a.memory, a.tiers, a.accounting, a.partial_hits, a.trace,
                         a.pool_tokens, tc, a.dispatch_ahead, a.form_lead_ms,
                         max_in_flight=a.max_in_flight,
                         stage_mult=[float(x) for x in a.stage_mult.split(",")] if a.stage_mult else None,
                         pace_margin_ms=a.pace_margin_ms, host_ms=a.host_ms)
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
