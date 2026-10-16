"""When the controller forms the next microbatch. Pure: the caller passes the clock.

The targets have no host time between consecutive stage-0 microbatches. With at most `max_in_flight` (4) microbatches
on 4 stages, the next one could only form after the stage-3 exit of microbatch k-3 is reported, and stage 0 would wait
on report -> form -> plan -> enqueue every time. So:
- stage 0 idle: form at once, if fewer than `max_in_flight` are in flight;
- stage 0 busy: one more plan may wait queued behind the running one (`queue_one`), even as the `max_in_flight + 1`-th;
  it forms `lead_ms` before the running plan's predicted stage-0 end, leaving the host time to form, send and enqueue
  it while stage 0 still computes. At most one plan is queued.
Predicted stage-0 end = the running plan's start (its dispatch, or the previous plan's stage-0 end if later) + its
predicted compute: a per-token EWMA of measured stage-0 times, one per power-of-two size bucket (the nearest bucket's
rate when its own has no sample yet). Without any sample the queued plan forms at once.

Pacing (`pace_margin_ms`; the server's default 2 ms, None: off). When a later stage is slower than stage 0 (stage balance differs per GPU and
per stage from the fit), the extra in-flight plan does not wait at stage 0 but queues in front of that stage, and
every round it carries pays the queue in its TTFT (budget N=500, 7730: 21 ms mean before stage 2). So a plan forms
only when it would reach every later stage no earlier than `pace_margin_ms` before that stage frees: the in-flight
plans' stage ends (reported, or predicted from per-stage EWMAs as above) give each stage's free time, and the next
plan is assumed to have the last plan's tokens. A later stage with no sample in a size bucket yet uses stage 0's
rate for it when stage 0 has one: stage 0 reports a new size first, and the nearest smaller bucket's rate carries
the per-microbatch fixed cost (force_check_plain 7753: 1,024-token rates put the first 8,192-token chunk at 94 ms on
stages 1-3, measured 50; stage 0 idled 40 ms twice and the 32k TTFT went from 0.41 to 0.48 s). It forms the predicted host time (dispatch to stage-0 start, an EWMA
per size bucket over plans that found stage 0 idle; `lead_ms` before any sample) ahead of that start. When stage 0
is the slowest, pacing never binds. Chosen over a cap of 4 in flight with the queued plan counted (ead8d9a, reverted):
that cap forms on exit reports, which left the slowest stage idle 5.2% vs 2.9% (even N=500 7741 vs budget N=500 7730)
and made force_check_even's 1k TTFT p50 x1.10 (larger microbatches); replay TTFT is the same for both.
"""

from collections import deque

NOW, WAIT, NO = "now", "wait", "no"


class Stage0Predictor:
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


class HostPredictor(Stage0Predictor):
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
    def __init__(self, max_in_flight: int = 4, queue_one: bool = True, lead_ms: float = 3.0,
                 pace_margin_ms: float | None = None, num_stages: int = 4):
        self.max_in_flight = max_in_flight
        self.queue_one = queue_one
        self.lead_ns = int(lead_ms * 1e6)
        self.pred = Stage0Predictor()
        self.on_stage0: deque[list[int]] = deque()   # [mb_id, dispatch_ns, tokens], head = running on stage 0
        self.last_end_ns: int | None = None          # stage-0 end of the last finished plan
        self.pace_ns = None if pace_margin_ms is None else int(pace_margin_ms * 1e6)
        self.num_stages = num_stages
        self.stage_pred = [self.pred] + [Stage0Predictor() for _ in range(num_stages - 1)]
        self.host = HostPredictor()
        self.flight: dict[int, list] = {}            # mb_id -> [dispatch_ns, tokens, stage ends (None: not reported)]
        self.last_tokens = 0

    @property
    def cap(self) -> int:
        """In-flight limit the scheduler is built with: the controller's rule is the binding one."""
        return self.max_in_flight + (1 if self.queue_one else 0)

    def dispatched(self, mb_id: int, now_ns: int, tokens: int):
        self.on_stage0.append([mb_id, now_ns, tokens])
        self.flight[mb_id] = [now_ns, tokens, [None] * self.num_stages]
        self.last_tokens = tokens

    def stage0_done(self, mb_id: int, start_ns: int, end_ns: int):
        head = self.on_stage0.popleft()
        assert head[0] == mb_id, (head, mb_id)
        self.pred.observe(head[2], end_ns - start_ns)
        if self.last_end_ns is None or self.last_end_ns <= head[1]:      # stage 0 was idle at its dispatch
            self.host.observe(head[2], max(start_ns - head[1], 0))
        self.last_end_ns = end_ns

    def stage_done(self, stage: int, mb_id: int, start_ns: int, end_ns: int):
        """Any stage's report (stage 0's also goes to `stage0_done`): per-stage times for pacing."""
        f = self.flight.get(mb_id)
        if f is None:
            return
        if stage > 0:
            self.stage_pred[stage].observe(f[1], end_ns - start_ns)
        f[2][stage] = end_ns
        if stage == self.num_stages - 1:
            del self.flight[mb_id]

    def stage_ns(self, stage: int, tokens: int) -> float:
        """Predicted compute of `tokens` on `stage`; a later stage without a sample in that size bucket takes stage 0's
        rate for it, if stage 0 has one."""
        p = self.stage_pred[stage]
        b = p.bucket(tokens)
        if stage > 0 and b not in p.rate and b in self.pred.rate:
            return self.pred.rate[b] * tokens
        return p.predict_ns(tokens)

    def paced_form_ns(self, now_ns: int) -> int | None:
        """Pacing: the host time at which the next plan should form (None: no constraint)."""
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
        """Predicted stage-0 end of the running plan."""
        if not self.on_stage0:
            return None
        _, dispatch_ns, tokens = self.on_stage0[0]
        p = self.pred.predict_ns(tokens)
        if p is None:
            return None
        start = max(dispatch_ns, self.last_end_ns or 0)
        return int(start + p)

    def decide(self, in_flight: int, now_ns: int) -> tuple[str, int | None]:
        """(NOW, None): form now; (WAIT, t): form at host time t; (NO, None): wait for a report."""
        verdict, at = self._decide(in_flight, now_ns)
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
        if not self.queue_one or busy >= 2 or in_flight > self.max_in_flight:
            return NO, None
        end = self.predicted_end_ns()
        if end is None or now_ns >= end - self.lead_ns:
            return NOW, None
        return WAIT, end - self.lead_ns
