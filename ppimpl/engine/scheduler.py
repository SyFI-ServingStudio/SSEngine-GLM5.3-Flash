"""Microbatch formation on stage 0. A pure state machine: no I/O, no clock; the controller feeds it events.

Phase A, the plain scheduler (4_iteration.md §2): admitted prompts in arrival order fill a microbatch up to the budget
(8,192 tokens), a longer prompt is cut into budget-sized chunks; a prompt's progress is committed when its chunk is
scheduled, so its next chunk can ride the next microbatch. At most `max_in_flight` microbatches are in flight
(dispatched, last stage not done). A microbatch forms when stage 0 is free, or up to `dispatch_ahead` microbatches
early so stage 0 never waits on the host.

Phase C replaces `_order` and `_budget` with steps 1-4 of 6_scheduler.md §2 (shortest first, load budget, even split,
force-schedule); everything else stays.
"""

from collections import deque
from dataclasses import dataclass

from ppimpl.engine.kv import Allocation, KvManager


@dataclass(eq=False)
class Request:
    req_id: str
    num_tokens: int
    arrival_ns: int = 0
    token_ids: list[int] | None = None
    computed: int = 0                  # prompt tokens scheduled so far
    alloc: Allocation | None = None

    @property
    def remaining(self) -> int:
        return self.num_tokens - self.computed


@dataclass
class ChunkSpec:
    req: Request
    start: int
    num_tokens: int

    @property
    def is_last(self) -> bool:
        return self.start + self.num_tokens == self.req.num_tokens


class PlainScheduler:
    def __init__(self, kv: KvManager, mb_tokens: int, max_in_flight: int = 4, dispatch_ahead: int = 1):
        self.kv = kv
        self.mb_tokens = mb_tokens
        self.max_in_flight = max_in_flight
        self.dispatch_ahead = dispatch_ahead
        self.waiting: deque[Request] = deque()   # not admitted yet, arrival order
        self.running: list[Request] = []         # admitted, prompt not fully scheduled, arrival order
        self.in_flight = 0                       # dispatched, last stage not done
        self.on_stage0 = 0                       # dispatched, stage 0 not done

    # --- events ---------------------------------------------------------------------------------------------------

    def add(self, req: Request):
        self.waiting.append(req)

    def stage_done(self, stage: int, last_stage: int):
        if stage == 0:
            self.on_stage0 -= 1
        if stage == last_stage:
            self.in_flight -= 1

    def finish(self, req: Request):
        """The request's token is out: free its blocks (phase A keeps nothing)."""
        self.kv.release(req.req_id)

    # --- decisions ------------------------------------------------------------------------------------------------

    def can_dispatch(self) -> bool:
        return self.in_flight < self.max_in_flight and self.on_stage0 <= self.dispatch_ahead

    def admit(self):
        """Allocate blocks in arrival order; the first request that does not fit holds back the rest."""
        while self.waiting:
            req = self.waiting[0]
            alloc = self.kv.admit(req.req_id, req.num_tokens)
            if alloc is None:
                return
            req.alloc = alloc
            self.running.append(self.waiting.popleft())

    def next_microbatch(self) -> list[ChunkSpec] | None:
        """Form and commit the next microbatch, or None (nothing to run, or no room in flight)."""
        if not self.can_dispatch():
            return None
        self.admit()
        chunks, left = [], self._budget()
        for req in self._order():
            if left == 0:
                break
            n = min(req.remaining, left)
            chunks.append(ChunkSpec(req, req.computed, n))
            req.computed += n
            left -= n
        if not chunks:
            return None
        self.running = [r for r in self.running if r.remaining > 0]
        self.in_flight += 1
        self.on_stage0 += 1
        return chunks

    def _order(self) -> list[Request]:
        return self.running

    def _budget(self) -> int:
        return self.mb_tokens
