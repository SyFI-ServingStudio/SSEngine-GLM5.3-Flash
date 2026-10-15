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


class DispatchRule:
    def __init__(self, max_in_flight: int = 4, queue_one: bool = True, lead_ms: float = 3.0):
        self.max_in_flight = max_in_flight
        self.queue_one = queue_one
        self.lead_ns = int(lead_ms * 1e6)
        self.pred = Stage0Predictor()
        self.on_stage0: deque[list[int]] = deque()   # [mb_id, dispatch_ns, tokens], head = running on stage 0
        self.last_end_ns: int | None = None          # stage-0 end of the last finished plan

    @property
    def cap(self) -> int:
        """In-flight limit the scheduler is built with: the controller's rule is the binding one."""
        return self.max_in_flight + (1 if self.queue_one else 0)

    def dispatched(self, mb_id: int, now_ns: int, tokens: int):
        self.on_stage0.append([mb_id, now_ns, tokens])

    def stage0_done(self, mb_id: int, start_ns: int, end_ns: int):
        head = self.on_stage0.popleft()
        assert head[0] == mb_id, (head, mb_id)
        self.pred.observe(head[2], end_ns - start_ns)
        self.last_end_ns = end_ns

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
        busy = len(self.on_stage0)
        if busy == 0:
            return (NOW, None) if in_flight < self.max_in_flight else (NO, None)
        if not self.queue_one or busy >= 2 or in_flight > self.max_in_flight:
            return NO, None
        end = self.predicted_end_ns()
        if end is None or now_ns >= end - self.lead_ns:
            return NOW, None
        return WAIT, end - self.lead_ns
