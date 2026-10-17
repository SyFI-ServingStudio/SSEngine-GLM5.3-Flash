"""When the controller forms the next microbatch. Pure: the caller passes the clock.

The targets (6_scheduler.md §2, ServingStudioSim `pipeline_head_worker.rs`): stage 0 starts microbatch k at the exact
time it may, the later of stage 0 finishing k-1 and microbatch k-`max_in_flight` leaving the last stage, and forms it
then; at most `max_in_flight` (4) are in flight. The host here learns of a stage end by a report (0.7-1 ms after it)
and a plan takes host time (form, send, receive, launch: 0.7-1 ms) to reach the GPU, so forming on reports leaves
stage 0 idle for both every time. So:
- **The in-flight limit is on the GPU.** Stage 0's stream waits, before microbatch k's forward, until the last stage
  has finished microbatch k-`max_in_flight` (a flag the last stage writes after each forward; `transport.py`). A plan
  may thus reach stage 0 early: it starts exactly when the targets would start it.
- **Forming follows a projection, not reports.** From the in-flight plans' reported stage ends, and predicted ones for
  the rest (per stage, a per-token EWMA of measured times per power-of-two size bucket: `StagePredictor`), the rule
  projects each stage's timeline with the gate and gets the next microbatch's start T = max(stage 0 free, exit of
  microbatch next-`max_in_flight`). It forms the plan at T - host - `lead_ms`, `host` being the predicted time from
  forming to the stage-0 start (an EWMA per size bucket over plans that found stage 0 ready). A report refreshes the
  projection; a stage that ends earlier than predicted is seen by its report and the plan forms at once.
- **The microbatch's contents are the targets'**: the scheduler forms it with the microbatches projected to have left
  the pipeline by T taken out of the in-flight count and the backlog (`exiting`).
Before any stage has a sample the rule forms only with stage 0 free and fewer than `max_in_flight` in flight (by
reports). Backstops, by reports, when a plan forms: at most 2 others not yet done on stage 0 (one may have ended
unreported) and `max_in_flight` + 1 not yet done on the last stage (the gate holds the GPU to `max_in_flight`).
Replaces the earlier queue-one rule (one plan queued behind stage 0's running one as the 5th in flight, formed
`form_lead_ms` before its predicted end, plus pacing to the later stages): it formed on stage-0, stage-3 and pacing
reports, which with plan delivery left stage 0 idle 2.4 s/min at good N=400 (7763) vs the targets' 1.0.
"""

NOW, WAIT, NO = "now", "wait", "no"


class StagePredictor:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.rate: dict[int, float] = {}          # size bucket -> ns per token

    @staticmethod
    def bucket(tokens: int) -> int:
        return max(tokens, 1).bit_length()

    def observe(self, tokens: int, compute_ns: int):
        if tokens <= 0:
            return
        b, r = self.bucket(tokens), compute_ns / tokens
        self.rate[b] = r if b not in self.rate else (1 - self.alpha) * self.rate[b] + self.alpha * r

    def predict_ns(self, tokens: int) -> float | None:
        if not self.rate:
            return None
        b = self.bucket(tokens)
        near = min(self.rate, key=lambda k: (abs(k - b), k))
        return self.rate[near] * tokens


class HostPredictor(StagePredictor):
    """EWMA of a time (not a rate) per size bucket."""

    def observe(self, tokens: int, ns: int):
        b = self.bucket(tokens)
        self.rate[b] = ns if b not in self.rate else (1 - self.alpha) * self.rate[b] + self.alpha * ns

    def predict_ns(self, tokens: int) -> float | None:
        if not self.rate:
            return None
        b = self.bucket(tokens)
        return self.rate[min(self.rate, key=lambda k: (abs(k - b), k))]


class DispatchRule:
    DEFAULT_HOST_NS = 1_000_000                       # forming -> stage-0 start before any sample

    def __init__(self, max_in_flight: int = 4, lead_ms: float = 3.0, num_stages: int = 4):
        self.max_in_flight = max_in_flight
        self.lead_ns = int(lead_ms * 1e6)
        self.num_stages = num_stages
        self.stage_pred = [StagePredictor() for _ in range(num_stages)]
        self.pred = self.stage_pred[0]
        self.host = HostPredictor()
        self.on_stage0: dict[int, None] = {}        # formed, stage 0 not reported done (insertion = mb order)
        self.flight: dict[int, list] = {}           # mb_id -> [dispatch_ns, tokens, stage ends (None: not reported)]
        self.exits: dict[int, int] = {}             # mb_id -> reported last-stage end, the last few
        self.last_end_ns: int | None = None         # stage-0 end of the last plan stage 0 reported
        self.stage_end = [0] * num_stages            # each stage's last reported end
        self.last_tokens = 0
        self.next_mb = 0
        self.projected: dict | None = None          # the last decision's projection (for the plan log)

    @property
    def cap(self) -> int:
        """In-flight count (by reports) the scheduler must allow: the GPU gate holds the pipeline to `max_in_flight`,
        reports lag it."""
        return self.max_in_flight + 2

    @property
    def sched_limits(self) -> tuple[int, int]:
        """The scheduler's (max_in_flight, dispatch_ahead): in flight net of `exiting`; up to 2 other plans not
        reported done on stage 0."""
        return self.max_in_flight, 2

    # --- events -----------------------------------------------------------------------------------------------------

    def dispatched(self, mb_id: int, now_ns: int, tokens: int):
        self.on_stage0[mb_id] = None
        self.flight[mb_id] = [now_ns, tokens, [None] * self.num_stages]
        self.last_tokens = tokens
        self.next_mb = mb_id + 1

    def stage_done(self, stage: int, mb_id: int, start_ns: int, end_ns: int):
        """A stage's report, in any order across stages: each stage's worker reports its microbatches in order, but a
        stage-0 report may reach the controller after later stages' reports of the same microbatch (job 7878: stage 0's
        reporter held ~0.5 s by its main thread's forward of the next one; mb 6279 left stage 3 before its stage-0
        report came). Keyed by mb_id; a stage that finished microbatch k finished every earlier one on it (one
        stream per stage), so any stage's report of k retires stage 0's entries up to k."""
        self.stage_end[stage] = max(self.stage_end[stage], end_ns)
        for k in [k for k in self.on_stage0 if k <= mb_id]:
            del self.on_stage0[k]
        f = self.flight.get(mb_id)
        if f is None:
            return                                   # left the pipeline already (its last-stage report came first)
        dispatch_ns, tokens, ends = f
        self.stage_pred[stage].observe(tokens, end_ns - start_ns)
        ends[stage] = end_ns
        if stage == 0:
            ready = max(self.last_end_ns or 0, self._exit_known(mb_id - self.max_in_flight) or 0)
            if ready <= dispatch_ns and self._exit_known(mb_id - self.max_in_flight) is not False:
                self.host.observe(tokens, max(start_ns - dispatch_ns, 0))     # stage 0 was ready: host time only
            self.last_end_ns = max(self.last_end_ns or 0, end_ns)
        if stage == self.num_stages - 1:
            del self.flight[mb_id]
            self.exits[mb_id] = end_ns
            for old in [k for k in self.exits if k < mb_id - 2 * self.max_in_flight]:
                del self.exits[old]

    def _exit_known(self, mb_id: int):
        """A reported exit: its time; None: long gone (or never existed); False: still in flight."""
        if mb_id < 0:
            return None
        if mb_id in self.flight:
            return False
        return self.exits.get(mb_id)

    # --- projection -------------------------------------------------------------------------------------------------

    def stage_ns(self, stage: int, tokens: int) -> float | None:
        """Predicted compute of `tokens` on `stage`; a later stage without a sample in that size bucket takes stage 0's
        rate for it, if stage 0 has one (stage 0 reports a new size first; a smaller bucket's rate carries the
        per-microbatch fixed cost, force_check_plain 7753)."""
        p = self.stage_pred[stage]
        b = p.bucket(tokens)
        if stage > 0 and b not in p.rate and b in self.pred.rate:
            return self.pred.rate[b] * tokens
        if p.rate:
            return p.predict_ns(tokens)
        return self.pred.predict_ns(tokens)

    def host_ns(self, tokens: int) -> float:
        h = self.host.predict_ns(tokens)
        return self.DEFAULT_HOST_NS if h is None else h

    def project(self) -> dict | None:
        """Projected stage ends of the in-flight plans (reported where known), and the next microbatch's start:
        {"ends": {mb_id: [end per stage]}, "start": ns, "exiting": n}. None: a stage has no sample yet."""
        n, depth = self.num_stages, self.max_in_flight
        ends: dict[int, list[float]] = {}
        free = [float(t) for t in self.stage_end]

        def exit_of(mb: int) -> float:
            if mb in ends:
                return ends[mb][-1]
            return float(self.exits.get(mb, 0))

        for mb in sorted(self.flight):
            dispatch_ns, tokens, rep = self.flight[mb]
            e = []
            for s in range(n):
                if rep[s] is not None:
                    t = float(rep[s])
                else:
                    c = self.stage_ns(s, tokens)
                    if c is None:
                        return None
                    if s == 0:
                        begin = max(dispatch_ns + self.host_ns(tokens), free[0], exit_of(mb - depth))
                    else:
                        begin = max(e[s - 1], free[s])
                    t = begin + c
                e.append(t)
                free[s] = t
            ends[mb] = e
        start = max(free[0], exit_of(self.next_mb - depth))
        exiting = sum(1 for mb in ends if ends[mb][-1] <= start)
        return {"ends": ends, "start": start, "exiting": exiting}

    # --- decision ---------------------------------------------------------------------------------------------------

    def decide(self, now_ns: int) -> tuple[str, int | None]:
        """(NOW, None): form now; (WAIT, t): form at host time t; (NO, None): wait for a report. After NOW,
        `exiting` is the number of in-flight plans (oldest first) that leave the pipeline before the new one starts."""
        self.projected = None
        in_flight = len(self.flight)
        if len(self.on_stage0) > 2 or in_flight > self.max_in_flight + 1:
            return NO, None
        p = self.project() if self.flight else {"ends": {}, "start": 0.0, "exiting": 0}
        if p is None:   # no sample yet: stage 0 free and room, by reports
            self.projected = {"start": None, "exiting": 0}
            return (NOW, None) if not self.on_stage0 and in_flight < self.max_in_flight else (NO, None)
        if in_flight - p["exiting"] >= self.max_in_flight:
            return NO, None
        self.projected = {"start": p["start"], "exiting": p["exiting"]}
        at = int(p["start"] - self.host_ns(self.last_tokens) - self.lead_ns)
        return (NOW, None) if now_ns >= at else (WAIT, at)

    @property
    def exiting(self) -> int:
        return self.projected["exiting"] if self.projected else 0


class QueueOneRule:
    """The rule before a8b80bc (for A/B against `DispatchRule`; RunConfig.dispatch = "queue_one"), with reports keyed
    by mb_id in any order (7878). Stage 0 idle: form at once if fewer than `max_in_flight` are in flight (by reports).
    Stage 0 busy: one more plan may wait queued behind the running one, even as the `max_in_flight + 1`-th; it forms
    `lead_ms` before the running plan's predicted stage-0 end (its start = its dispatch or the previous plan's stage-0
    end, plus a per-token EWMA per size bucket). Pacing (`pace_margin_ms`): a plan forms only when it would reach
    every later stage no earlier than the margin before that stage frees (per-stage EWMAs; the next plan assumed to
    have the last plan's tokens; the predicted host time ahead, `lead_ms` before any sample). The GPU gate is at
    `max_in_flight + 1` (RunConfig.gate_depth), so the queued plan is not held there."""

    def __init__(self, max_in_flight: int = 4, lead_ms: float = 45.0, pace_margin_ms: float | None = 2.0,
                 num_stages: int = 4):
        self.max_in_flight = max_in_flight
        self.lead_ns = int(lead_ms * 1e6)
        self.pace_ns = None if pace_margin_ms is None else int(pace_margin_ms * 1e6)
        self.num_stages = num_stages
        self.stage_pred = [StagePredictor() for _ in range(num_stages)]
        self.pred = self.stage_pred[0]
        self.host = HostPredictor()
        self.on_stage0: dict[int, list[int]] = {}   # mb_id -> [dispatch_ns, tokens], formed, stage 0 not reported
        self.flight: dict[int, list] = {}           # mb_id -> [dispatch_ns, tokens, stage ends (None: not reported)]
        self.last_end_ns: int | None = None
        self.last_tokens = 0
        self.projected = None                       # the plan log's projection fields: none here
        self.exiting = 0

    @property
    def sched_limits(self) -> tuple[int, int]:
        return self.max_in_flight + 1, 1

    def dispatched(self, mb_id: int, now_ns: int, tokens: int):
        self.on_stage0[mb_id] = [now_ns, tokens]
        self.flight[mb_id] = [now_ns, tokens, [None] * self.num_stages]
        self.last_tokens = tokens

    def stage_done(self, stage: int, mb_id: int, start_ns: int, end_ns: int):
        if stage == 0:
            e = self.on_stage0.pop(mb_id, None)
            if e is not None:
                self.pred.observe(e[1], end_ns - start_ns)
                if self.last_end_ns is None or self.last_end_ns <= e[0]:     # stage 0 was idle at its dispatch
                    self.host.observe(e[1], max(start_ns - e[0], 0))
            self.last_end_ns = max(self.last_end_ns or 0, end_ns)
        for k in [k for k in self.on_stage0 if k <= mb_id]:   # a later stage's report of k: stage 0 is past k
            del self.on_stage0[k]
        f = self.flight.get(mb_id)
        if f is None:
            return
        if stage > 0:
            self.stage_pred[stage].observe(f[1], end_ns - start_ns)
        f[2][stage] = end_ns
        if stage == self.num_stages - 1:
            del self.flight[mb_id]

    def stage_ns(self, stage: int, tokens: int) -> float:
        p = self.stage_pred[stage]
        b = p.bucket(tokens)
        if stage > 0 and b not in p.rate and b in self.pred.rate:
            return self.pred.rate[b] * tokens
        return p.predict_ns(tokens)

    def paced_form_ns(self, now_ns: int) -> int | None:
        if self.pace_ns is None or not self.flight or any(not p.rate for p in self.stage_pred):
            return None
        free = [0.0] * self.num_stages
        for mb_id in sorted(self.flight):
            dispatch_ns, tokens, ends = self.flight[mb_id]
            prev = float(dispatch_ns)
            for s in range(self.num_stages):
                e = ends[s]
                if e is None:
                    e = max(max(prev, free[s]) + self.stage_ns(s, tokens), now_ns)
                free[s] = prev = e
        start, ahead = None, 0.0
        for s in range(1, self.num_stages):
            ahead += self.stage_ns(s - 1, self.last_tokens)
            t = free[s] - ahead
            start = t if start is None else max(start, t)
        host = self.host.predict_ns(self.last_tokens)
        return int(start - self.pace_ns - (self.lead_ns if host is None else host))

    def predicted_end_ns(self) -> int | None:
        if not self.on_stage0:
            return None
        dispatch_ns, tokens = next(iter(self.on_stage0.values()))
        p = self.pred.predict_ns(tokens)
        if p is None:
            return None
        return int(max(dispatch_ns, self.last_end_ns or 0) + p)

    def decide(self, now_ns: int) -> tuple[str, int | None]:
        """(NOW, None): form now; (WAIT, t): form at host time t; (NO, None): wait for a report."""
        verdict, at = self._decide(len(self.flight), now_ns)
        if verdict == NO:
            return verdict, at
        paced = self.paced_form_ns(now_ns)
        if paced is None or paced <= now_ns:
            return verdict, at
        return WAIT, paced if at is None else max(at, paced)

    def _decide(self, in_flight: int, now_ns: int) -> tuple[str, int | None]:
        busy = len(self.on_stage0)
        if busy == 0:
            return (NOW, None) if in_flight < self.max_in_flight else (NO, None)
        if busy >= 2 or in_flight > self.max_in_flight:
            return NO, None
        end = self.predicted_end_ns()
        if end is None or now_ns >= end - self.lead_ns:
            return NOW, None
        return WAIT, end - self.lead_ns


def make_rule(cfg, num_stages: int):
    """The controller's rule for RunConfig.dispatch."""
    if cfg.dispatch == "project":
        return DispatchRule(cfg.max_in_flight, cfg.form_lead_ms, num_stages)
    if cfg.dispatch == "queue_one":
        return QueueOneRule(cfg.max_in_flight, cfg.form_lead_ms, cfg.pace_margin_ms, num_stages)
    raise ValueError(f"dispatch: {cfg.dispatch!r} (project | queue_one)")
