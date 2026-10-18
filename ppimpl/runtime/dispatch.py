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
  forming to the stage-0 start (an EWMA per size bucket of every plan's delivery + prepare + launch, from the
  worker's host stamps; GPU waits left out). A report refreshes the
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

    def stage_done(self, stage: int, mb_id: int, start_ns: int, end_ns: int, host_ns: tuple | None = None):
        """A stage's report, in any order across stages: each stage's worker reports its microbatches in order, but a
        stage-0 report may reach the controller after later stages' reports of the same microbatch (job 7878: stage 0's
        reporter held ~0.5 s by its main thread's forward of the next one; mb 6279 left stage 3 before its stage-0
        report came). Keyed by mb_id; a stage that finished microbatch k finished every earlier one on it (one
        stream per stage), so any stage's report of k retires stage 0's entries up to k. `host_ns`: the worker's host
        stamps (`StageDone.host_ns`); with them stage 0's host time is measured on every microbatch (`_host_time`)."""
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
            if host_ns is not None:
                self.host.observe(tokens, self._host_time(dispatch_ns, start_ns, host_ns))
            else:                                    # no stamps: only a plan that found stage 0 ready shows host time
                ready = max(self.last_end_ns or 0, self._exit_known(mb_id - self.max_in_flight) or 0)
                if ready <= dispatch_ns and self._exit_known(mb_id - self.max_in_flight) is not False:
                    self.host.observe(tokens, max(start_ns - dispatch_ns, 0))
            self.last_end_ns = max(self.last_end_ns or 0, end_ns)
        if stage == self.num_stages - 1:
            del self.flight[mb_id]
            self.exits[mb_id] = end_ns
            for old in [k for k in self.exits if k < mb_id - 2 * self.max_in_flight]:
                del self.exits[old]

    def _host_time(self, dispatch_ns: int, start_ns: int, host_ns: tuple) -> int:
        """Forming -> stage-0 start were stage 0 free all along: the plan's delivery (formed -> received), the worker's
        host side (prepare call -> prepared) and its launch (start mark, or stage 0's previous end if later -> first
        kernel). Waits for the GPU (run-ahead, the previous microbatch still computing) are left out, so every
        microbatch is a sample, not only the few that found stage 0 ready (a saturated pipeline gave almost none and
        the estimate stayed at its early value: 8 ms planned vs 19 ms taken at v3 minutes 6-8, job 8242)."""
        plan, call, prep, ready = host_ns[:4]
        launch = start_ns - max(ready, self.last_end_ns or 0)
        return max(plan - dispatch_ns, 0) + max(prep - call, 0) + max(launch, 0)

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
