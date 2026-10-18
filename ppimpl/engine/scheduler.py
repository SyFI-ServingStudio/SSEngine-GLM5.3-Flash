"""Microbatch formation on stage 0. A pure state machine: no I/O, no clock of its own; the caller passes `now_ns` and
feeds it events (`add`, `stage_done`, `finish`, `read_landed`).

The scheduler of 6_scheduler.md §2, built on the plain one (`SchedulerConfig`; `SCHEDULER_STEPS` names the steps):
- **In flight.** At most `max_in_flight` microbatches (formed, last stage not done). A microbatch forms when stage 0 is
  free, or up to `dispatch_ahead` microbatches early (0 reproduces the targets, which decide at stage-0-free time);
  a caller forming ahead says how many in-flight ones will have left by then (`exiting`).
- **Progress** is committed when a chunk is scheduled; chunks end at any token; the next chunk can ride the next
  microbatch.
- **Admission.** A request reserves its whole footprint once, when its first chunk is scheduled (`KvManager`); the
  first waiting request that does not fit stops admission for this microbatch. No preemption.
- **Order.** `fifo` (step 0): started prompts in start order, then waiting prompts in arrival order. `srpt` (steps
  1-4): (a) overdue requests first (waiting, then started, oldest first), (b) started prompts fewest remaining tokens
  first, each preceded by the waiting prompts that need no more tokens than it has left, shortest first, (c) the
  other waiting prompts shortest first. A waiting prompt's size is the tokens it must compute, fixed when it arrives
  (cached tokens, HBM or tier, do not count); ties go by arrival.
- **Budget.** `mb_tokens`; with `load_budget` at most 4,096 tokens up to a backlog of 1M rising linearly to
  `mb_tokens` at 3M; with `even_split` at most ceil(B / 4) kept in [512, mb_tokens]. The backlog B is the started
  prompts' unscheduled tokens, the waiting prompts' new tokens (beyond the session's previous context, even if that
  must be recomputed: `Request.fresh`) and the in-flight microbatches' tokens.
- **Force-schedule** (`force_after_ms`): a request whose wait since it arrived has reached the bound goes first.
- **Tiers** (phase D, `pool`, `ppimpl/kvpool`): when the order reaches a waiting round that DRAM or SSD holds more
  of than HBM, its read starts (if its footprint fits; the footprint is held until it lands) and it leaves the queue.
  A read that would wait more than the bound (`TierConfig.max_read_wait_ms`) for its tier (`Hit.wait_ns`) is not
  started: the round is passed over and stays queued. Landed rounds are admitted before every queued request, in
  landing order. Writes are not the scheduler's: the pool is the KV manager's lower level (`pool.hbm_level`), which
  writes kept contexts before HBM evicts them (a request that fits only once such a write lands waits, as for room;
  when a write lands, that round's read starts at once if it fits then).
  A round whose kept context is still being written waits to take it over (`busy_waits`).
- **Staged loading** (`TierConfig.loading_read_gb_s`; ServingStudioSim pp-staged `prefix_tier_loading_area`, rule
  B, not vLLM): pending -> loading -> ready -> batch. An in-flight area of HBM (`KvManager.area`, Little's law:
  channel read rate x one batch period, `BalancedKVPool.loading_area_blocks`) is always counted held, so admission
  and kept contexts leave it free, and the pool writes kept contexts back ahead by what one batch period takes
  (`WriteBack.reserve`), so admission evicts written contexts and never waits for a write. `load` walks the pending queues in admission order (overdue, then
  the policy order) and starts the read of every round a tier holds more of than HBM while its area has room: a
  DRAM read (or an SSD read without a DRAM area) into HBM's area; an SSD read into DRAM's area (the same fraction of
  DRAM, `fetch`), which moves on to HBM's area over DRAM's channel once that has room (`hop`, in fetch landing
  order). Each queue stops at the first read that does not fit (a read larger than the whole area starts when the
  area is empty): later reads never jump it. A read holds its area until it lands (DRAM's until its hop lands); its
  blocks then stay pinned in HBM, counted in occupancy, until its round is admitted. New reads start only while the
  landed rounds' fresh tokens are below `ready_bound` (pp_size x max_batch_tokens). Admission takes only rounds whose context is all in HBM: landed reads first (overdue ones in the
  force-schedule pass too), then the queues, passing over rounds that still need a read; a round found ready in HBM
  that needs a read again later (its context evicted before admission) is read again (`rereads`). The read-wait
  bound is not used.

The microbatch-forming procedure follows the reference implementation of the same design (ServingStudioSim
`pipelined_chunked_prefill_admission.rs`, `srpt.rs`, `fresh_source.rs`, `prefix_fetch.rs`), so that the scheduling
check matches chunk for chunk.
"""

import heapq
import time
from collections import deque
from dataclasses import dataclass

from ppimpl.config import MAX_IN_FLIGHT, SchedulerConfig
from ppimpl.engine.kv import Allocation, KvManager
from ppimpl.kvpool import BalancedKVPool, KVViews


@dataclass(eq=False)
class Request:
    req_id: str
    num_tokens: int                    # whole prompt
    arrival_ns: int = 0
    token_ids: list[int] | None = None
    session: object = None             # kept-context key: the session (replay) or a content key (server)
    prefix_tokens: int = 0             # declared prefix: tokens the prompt shares with its session's previous context
    backlog_prefix: int | None = None  # server (no declared prefix): KvManager.shared_prefix at arrival, for `fresh`
    cached: int = 0                    # prompt tokens already in KV at admission (HBM hit or tier read)
    computed: int = 0                  # prompt tokens scheduled so far
    alloc: Allocation | None = None
    # set by the scheduler
    seq: int = -1
    size: int = 0                      # tokens to compute, as ranked at arrival
    hbm_tokens: int = 0                # where the cached tokens came from
    tier: str | None = None
    tier_tokens: int = 0
    read_start_ns: int | None = None
    read_land_ns: int | None = None
    admit_ns: int | None = None
    hit_memo: tuple | None = None      # (kept context, hit): KvManager's comparison of the prompt with it
    output_token: int | None = None    # the token the prompt got (server; kept with the context)
    room_wait_ns: int | None = None    # first time it was at the head and its footprint did not fit
    admit_info: dict | None = None     # KvManager.allocate: how the prefix cache served it (hit_kind, ...)
    tier_memo: dict | None = None      # TierManager.lookup: tier -> (entry, its tokens, the resumable tokens)
    load_phase: str | None = None      # staged: "read" into HBM's area, "fetch" SSD -> DRAM's area, "hop_wait", "hop"
    dram_area: list[int] | None = None  # staged: the DRAM area slots its SSD read holds
    fetch_land_ns: int | None = None
    hop_start_ns: int | None = None
    hbm_ready: bool = False            # staged: found with its context in HBM at an admission check

    @property
    def remaining(self) -> int:
        return self.num_tokens - self.cached - self.computed

    @property
    def fresh(self) -> int:
        """New tokens: what the request adds to the backlog while it waits (beyond the declared prefix, even when the
        prefix must be recomputed, as the simulator counts it)."""
        return self.num_tokens - (self.prefix_tokens if self.backlog_prefix is None else self.backlog_prefix)


@dataclass
class ChunkSpec:
    req: Request
    start: int                          # tokens of the request already in KV (cached + earlier chunks)
    num_tokens: int

    @property
    def is_last(self) -> bool:
        return self.start + self.num_tokens == self.req.num_tokens


# --- waiting queues ---------------------------------------------------------------------------------------------

class FifoQueue:
    """Arrival order. Removal (to the overdue queue) is lazy."""

    def __init__(self):
        self._q: deque[Request] = deque()
        self._in: set[int] = set()
        self.tokens = 0

    def __len__(self):
        return len(self._in)

    def __iter__(self):
        return (r for r in self._q if id(r) in self._in)

    def push(self, r: Request):
        self._q.append(r)
        self._in.add(id(r))
        self.tokens += r.fresh

    def push_front(self, r: Request):
        self._q.appendleft(r)
        self._in.add(id(r))
        self.tokens += r.fresh

    def peek(self) -> Request | None:
        while self._q and id(self._q[0]) not in self._in:
            self._q.popleft()
        return self._q[0] if self._q else None

    def pop(self) -> Request:
        r = self.peek()
        self._q.popleft()
        self._in.discard(id(r))
        self.tokens -= r.fresh
        return r

    def remove(self, r: Request) -> bool:
        if id(r) not in self._in:
            return False
        self._in.discard(id(r))
        self.tokens -= r.fresh
        return True


class ShortestFirstQueue(FifoQueue):
    """Shortest prefill first: by the tokens to compute fixed at arrival, then by arrival."""

    def __init__(self):
        super().__init__()
        self._h: list[tuple[int, int, Request]] = []

    def __iter__(self):
        return (e[2] for e in sorted(self._h) if id(e[2]) in self._in)

    def push(self, r: Request):
        heapq.heappush(self._h, (r.size, r.seq, r))
        self._in.add(id(r))
        self.tokens += r.fresh

    push_front = push     # the key restores its place

    def peek(self) -> Request | None:
        while self._h and id(self._h[0][2]) not in self._in:
            heapq.heappop(self._h)
        return self._h[0][2] if self._h else None

    def pop(self) -> Request:
        r = self.peek()
        heapq.heappop(self._h)
        self._in.discard(id(r))
        self.tokens -= r.fresh
        return r


class OverdueQueue:
    """Force-schedule: requests that waited past the bound leave the pending order, oldest first."""

    def __init__(self, after_ms: float):
        self.after_ns = int(after_ms * 1e6)
        self.arrivals: deque[Request] = deque()
        self.q: deque[Request] = deque()
        self.tokens = 0

    def is_overdue(self, r: Request, now_ns: int) -> bool:
        return now_ns - r.arrival_ns >= self.after_ns

    def promote(self, policy: FifoQueue, now_ns: int):
        while self.arrivals and self.is_overdue(self.arrivals[0], now_ns):
            r = self.arrivals.popleft()
            if policy.remove(r):
                self.q.append(r)
                self.tokens += r.fresh

    def peek(self) -> Request | None:
        return self.q[0] if self.q else None

    def pop(self) -> Request:
        r = self.q.popleft()
        self.tokens -= r.fresh
        return r

    def push_front(self, r: Request):
        self.q.appendleft(r)
        self.tokens += r.fresh


# --- the scheduler ----------------------------------------------------------------------------------------------

LANDED, OVERDUE, POLICY = "landed", "overdue", "policy"


class Scheduler:
    def __init__(self, kv: KvManager, cfg: SchedulerConfig | None = None, pool: BalancedKVPool | None = None,
                 max_in_flight: int = MAX_IN_FLIGHT, dispatch_ahead: int = 1):
        self.kv = kv
        self.cfg = cfg = cfg or SchedulerConfig()
        self.pool = pool
        wait = None if pool is None else pool.cfg.max_read_wait_ms
        self.max_wait_ns = None if wait is None else int(wait * 1e6)    # the read-wait bound
        self.max_in_flight = max_in_flight
        self.dispatch_ahead = dispatch_ahead
        self.srpt = cfg.order == "srpt"
        assert cfg.order in ("fifo", "srpt")
        self.policy = ShortestFirstQueue() if self.srpt else FifoQueue()
        self.overdue = OverdueQueue(cfg.force_after_ms) if cfg.force_after_ms else None
        self.started: list[Request] = []       # admitted, prompt not fully scheduled; start order
        self.reading: dict[str, Request] = {}  # out of the queue while their tier read runs
        self.landed: deque[Request] = deque()  # read landed, holding their blocks, admitted first
        self.in_flight = 0                     # formed, last stage not done
        self.on_stage0 = 0                     # formed, stage 0 not done
        self.in_flight_tokens = 0
        self._mb_tokens: deque[int] = deque()
        self._seq = 0
        self._chunks: list[ChunkSpec] = []
        self._room_blocked: Request | None = None   # the waiting round whose footprint did not fit at the last check
        self.stats = dict(microbatches=0, passed_over=0, reads=0, blocked_reads=0, busy_waits=0)
        self.staged = False
        self.waiting_hop: deque[Request] = deque()     # staged: fetched into DRAM's area, waiting for HBM's
        self._load_dirty = True
        if pool is not None:
            if not kv.keep_sessions:
                raise ValueError("tiers take what HBM's prefix cache evicts: keep_sessions is required")
            kv.lower = pool.hbm_level(kv)
            if pool.set_loading_area(kv, cfg.mb_tokens):
                self.staged = True
                self.stats.update(fetches=0, hops=0, rereads=0, ready_stops=0, load_passes=0, load_pass_ns=0)
        self.ready_bound = max_in_flight * cfg.mb_tokens   # staged: pp_size x max_batch_tokens of landed fresh work

    # --- events ---------------------------------------------------------------------------------------------------

    def add(self, req: Request, now_ns: int | None = None):
        """A request arrived. Its size for the order is fixed now: what it must compute beyond what HBM or a tier
        holds of its prefix."""
        req.seq = self._seq
        self._seq += 1
        if now_ns is not None and not req.arrival_ns:
            req.arrival_ns = now_ns
        rank = self.kv.peek_hit(req)
        if self.pool is not None and req.session is not None:
            hit = self.pool.lookup(req.session, req)
            if hit is not None:
                rank = max(rank, hit.tokens)
        req.size = req.num_tokens - rank
        if req.token_ids is not None and not req.prefix_tokens:     # server: no declared prefix
            req.backlog_prefix = self.kv.shared_prefix(req)
        self.policy.push(req)
        if self.overdue is not None:
            self.overdue.arrivals.append(req)
        self._load_dirty = True

    def stage_done(self, stage: int, last_stage: int):
        if stage == 0:
            self.on_stage0 -= 1
        if stage == last_stage:
            self.in_flight -= 1
            self.in_flight_tokens -= self._mb_tokens.popleft()

    def finish(self, req: Request):
        """The request's token is out: its context stays kept in HBM, or its blocks are freed."""
        self.kv.release(req)
        self._load_dirty = True

    def op_done(self, op_id: int, now_ns: int):
        """Every worker reported the tier op done (slot tiers): a load done lands its round."""
        _, h = self.pool.complete(op_id, now_ns)
        if self.staged:
            self._staged_done(h, now_ns)
            return
        if h is not None and h.kind == "load":
            self.read_landed(h.req_id, now_ns)
        elif h is not None and h.kind == "store" and self._room_blocked is not None:
            self._retry_read(now_ns)

    def _retry_read(self, now_ns: int):
        """A write-back landed (HBM may evict more now): the round that did not fit at the head of the order starts its
        tier read at once, not at the next microbatch's forming. A round with no read to start waits for the forming
        (it computes only then)."""
        src, r = self._peek(False)
        if r is not self._room_blocked or src == LANDED or self.kv.busy(r):
            return
        if self._at_head(r, now_ns) == "read":
            self._take(src)
            self._room_blocked = None

    def read_landed(self, req_id: str, now_ns: int):
        """Every worker reported the read done: the round is admitted before every queued request."""
        req = self.reading.pop(req_id)
        req.read_land_ns = now_ns
        self.landed.append(req)

    # --- staged loading -----------------------------------------------------------------------------------------

    def _staged_done(self, h, now_ns: int):
        """A staged op is done: a fetch waits for HBM's area, a hop gives DRAM's area back, a read or hop lands."""
        self._load_dirty = True
        if h is None or h.kind != "load":
            return
        r = self.reading[h.req_id]
        if r.load_phase == "fetch":
            r.load_phase, r.fetch_land_ns = "hop_wait", now_ns
            self.waiting_hop.append(r)
            self._start_hops(now_ns)
            return
        if r.load_phase == "hop":
            self.pool.release_dram_area(r.dram_area)
            r.dram_area = None
        r.load_phase = None
        self.kv.read_landed(r)                       # HBM's in-flight area is free again; the blocks stay pinned
        self.read_landed(r.req_id, now_ns)
        self._start_hops(now_ns)

    def _start_hops(self, now_ns: int):
        """Fetched SSD reads move on to HBM's area in fetch landing order while it has room."""
        while self.waiting_hop:
            r = self.waiting_hop[0]
            if not self.kv.area_fits(r.cached):
                return
            alloc = self.kv.hold_read(r, r.cached)
            if alloc is None:
                return
            self.waiting_hop.popleft()
            h = self.pool.hop(r.session, r.cached, r.dram_area, KVViews(alloc.kv_blocks, alloc.state_blocks, r.cached),
                              now_ns, r.req_id)
            r.load_phase, r.hop_start_ns, r.read_land_ns = "hop", now_ns, h.eta_ns
            self.stats["hops"] += 1

    def _needs_read(self, r: Request, now_ns: int):
        """The tier hit of a round a tier holds more of than HBM, else None."""
        if r.session is None:
            return None
        hit = self.pool.lookup(r.session, r, now_ns)
        return hit if hit is not None and hit.tokens > self.kv.peek_hit(r) else None

    def load(self, now_ns: int):
        """Staged loading pass (module docs): start the reads of pending rounds in admission order while their
        loading areas have room. Runs only after something changed (arrival, landing, admission, finish)."""
        if not self.staged or not self._load_dirty:
            return
        t0 = time.perf_counter_ns()
        self._load_dirty = False
        self._start_hops(now_ns)
        if self.overdue is not None:
            self.overdue.promote(self.policy, now_ns)
        if self.ready_fresh() >= self.ready_bound:     # enough landed work to fill the pipeline: no new reads
            self.stats["ready_stops"] += 1
            self.stats["load_passes"] += 1
            self.stats["load_pass_ns"] += time.perf_counter_ns() - t0
            return
        hbm_open, dram_open = True, True
        queues = ([(OVERDUE, list(self.overdue.q))] if self.overdue is not None else []) \
            + [(POLICY, list(self.policy))]
        for src, rs in queues:
            for r in rs:
                if not (hbm_open or dram_open):
                    break
                if self.kv.busy(r):
                    continue
                hit = self._needs_read(r, now_ns)
                if hit is None:
                    continue
                via_dram = hit.tier == "ssd" and self.pool.dram_area_fits(hit.tokens) is not None
                if via_dram:
                    if not dram_open:
                        continue
                    if not self.pool.dram_area_fits(hit.tokens) or not self._start_fetch(r, hit, now_ns):
                        dram_open = False
                        continue
                else:
                    if not hbm_open:
                        continue
                    if not self.kv.area_fits(hit.tokens) or not self._start_read(r, hit, now_ns):
                        hbm_open = False
                        continue
                if src == OVERDUE:
                    self.overdue.q.remove(r)
                    self.overdue.tokens -= r.fresh
                else:
                    self.policy.remove(r)
                if self.ready_fresh() >= self.ready_bound:
                    break
        self.stats["load_passes"] += 1
        self.stats["load_pass_ns"] += time.perf_counter_ns() - t0

    def state(self) -> dict:
        """Counters for a hang report (the controller's watchdog)."""
        kv = self.kv
        x = dict(in_flight=self.in_flight, started=len(self.started), pending=len(self.policy),
                 overdue=len(self.overdue.q) if self.overdue is not None else 0, reading=len(self.reading),
                 landed=len(self.landed), hard=kv.hard, kept=kv.kept_charge, capacity=kv.capacity,
                 held_back=kv.lower.held_back() if kv.lower is not None else 0,
                 kv_stats={k: kv.stats[k] for k in ("blocked_by_write", "area_waits", "live_waits")})
        if self.staged:
            d = self.pool.manager.dram_area
            x.update(area=kv.area, area_held=kv.area_held, landed_held=kv.landed_held,
                     ready_fresh=self.ready_fresh(), ready_bound=self.ready_bound, waiting_hop=len(self.waiting_hop),
                     dram_area=None if d is None else [d.area_used, d.area], load_dirty=self._load_dirty,
                     tier_ops_in_flight=len(self.pool.manager.in_flight))
        return x

    def ready_fresh(self) -> int:
        """New tokens of the landed rounds not yet admitted (the ready set)."""
        return sum(r.fresh for r in self.landed)

    def _reading(self, r: Request, hit, now_ns: int, phase: str):
        if r.hbm_ready or r.read_land_ns is not None:
            self.stats["rereads"] += 1               # its context was in HBM and was evicted before admission
        r.hbm_ready = False
        r.cached, r.tier, r.tier_tokens, r.hbm_tokens = hit.tokens, hit.tier, hit.tokens, 0
        r.read_start_ns, r.load_phase = now_ns, phase
        r.fetch_land_ns = r.hop_start_ns = None
        self.reading[r.req_id] = r
        self.stats["reads"] += 1

    def _start_read(self, r: Request, hit, now_ns: int) -> bool:
        """A tier read into HBM's loading area (the whole context; HBM holds none of it, or a shorter one the read
        supersedes)."""
        alloc = self.kv.hold_read(r, hit.tokens)
        if alloc is None:
            return False
        h = self.pool.load(r.session, hit, KVViews(alloc.kv_blocks, alloc.state_blocks, hit.tokens, 0), now_ns,
                           r.req_id)
        if h is None:                                # the tier let it go meanwhile: compute it
            self.kv.drop_read(r)
            return False
        self._reading(r, hit, now_ns, "read")
        r.read_land_ns = h.eta_ns
        return True

    def _start_fetch(self, r: Request, hit, now_ns: int) -> bool:
        got = self.pool.fetch(r.session, hit, now_ns, r.req_id)
        if got is None:
            return False
        h, r.dram_area = got
        self._reading(r, hit, now_ns, "fetch")
        r.read_land_ns = h.eta_ns
        self.stats["fetches"] += 1
        return True

    @property
    def waiting(self) -> list[Request]:
        return list(self.policy)

    def has_work(self) -> bool:
        return bool(self.started or len(self.policy) or self.landed or (self.overdue and self.overdue.q))

    # --- decisions ------------------------------------------------------------------------------------------------

    def can_dispatch(self, exiting: int = 0) -> bool:
        return self.in_flight - exiting < self.max_in_flight and self.on_stage0 <= self.dispatch_ahead

    def next_microbatch(self, now_ns: int = 0, exiting: int = 0) -> list[ChunkSpec] | None:
        """Form and commit the next microbatch, or None (nothing to run, or no room in flight). May start tier
        reads (`reading`). `exiting`: the oldest in-flight microbatches that leave the last stage before this one
        starts (the controller forms ahead, ppimpl/runtime/dispatch.py): not counted in flight or in the backlog."""
        if not self.can_dispatch(exiting):
            return None
        if self.overdue is not None:
            self.overdue.promote(self.policy, now_ns)
        self.load(now_ns)
        self._chunks = []
        budget = self.budget(exiting)
        if self.srpt:
            self._form_srpt(budget, now_ns)
        else:
            self._form_fifo(budget, now_ns)
        chunks, self._chunks = self._chunks, []
        if self.staged and chunks:
            self._load_dirty = True                   # admission freed area and evicted kept contexts
            self.load(now_ns)
        if not chunks:
            return None
        tokens = sum(c.num_tokens for c in chunks)
        self.in_flight += 1
        self.on_stage0 += 1
        self.in_flight_tokens += tokens
        self._mb_tokens.append(tokens)
        self.stats["microbatches"] += 1
        return chunks

    def backlog(self, exiting: int = 0) -> int:
        """B: started prompts' unscheduled tokens + waiting prompts' new tokens + in-flight microbatches' tokens
        (without the `exiting` oldest)."""
        gone = sum(self._mb_tokens[i] for i in range(min(exiting, len(self._mb_tokens))))
        return (sum(r.remaining for r in self.started) + self.policy.tokens
                + (self.overdue.tokens if self.overdue else 0) + self.in_flight_tokens - gone)

    def budget(self, exiting: int = 0) -> int:
        cfg = self.cfg
        b = cfg.mb_tokens
        if not (cfg.load_budget or cfg.even_split):
            return b
        backlog = self.backlog(exiting)
        if cfg.load_budget:
            rise = min(max((backlog - cfg.backlog_lo) / (cfg.backlog_hi - cfg.backlog_lo), 0.0), 1.0)
            b = min(b, cfg.load_budget_low + int((cfg.mb_tokens - cfg.load_budget_low) * rise + 0.5))
        if cfg.even_split:
            b = min(b, min(max(-(-backlog // cfg.depth), cfg.even_min), cfg.mb_tokens))
        return b

    # --- forming ----------------------------------------------------------------------------------------------------

    def _schedule(self, r: Request, n: int):
        self._chunks.append(ChunkSpec(r, r.cached + r.computed, n))
        r.computed += n

    def _form_fifo(self, budget: int, now_ns: int):
        still = []
        for r in self.started:
            n = min(r.remaining, budget)
            if n:
                self._schedule(r, n)
                budget -= n
            if r.remaining:
                still.append(r)
        self.started = still
        self._admit(budget, now_ns)

    def _form_srpt(self, budget: int, now_ns: int):
        od = self.overdue
        started = []
        for r in self.started:
            overdue = od is not None and od.is_overdue(r, now_ns)
            started.append((not overdue, r.arrival_ns if overdue else r.remaining, r))
        started.sort(key=lambda e: (e[0], e[1]))       # stable: ties keep start order
        self.started = []
        if od is not None and budget > 0:
            budget -= self._admit(budget, now_ns, overdue_only=True)
        still = []
        for not_overdue, _, r in started:
            remaining = r.remaining
            if not_overdue and budget > 0:
                budget -= self._admit(budget, now_ns, no_longer_than=remaining)
            n = min(remaining, budget)
            if n:
                self._schedule(r, n)
                budget -= n
            if n < remaining:
                still.append(r)
        if budget > 0:
            self._admit(budget, now_ns)
        self.started = still + self.started               # admitted now join after the ones already started

    def _peek(self, overdue_only: bool, now_ns: int = 0) -> tuple[str, Request] | tuple[None, None]:
        if self.landed and not overdue_only:
            return LANDED, self.landed[0]
        if self.landed and self.staged and self.overdue is not None:     # force-schedule covers landed reads too
            od = [r for r in self.landed if self.overdue.is_overdue(r, now_ns)]
            if od:
                return LANDED, min(od, key=lambda r: r.arrival_ns)
        if self.overdue is not None and self.overdue.q:
            return OVERDUE, self.overdue.q[0]
        if overdue_only:
            return None, None
        r = self.policy.peek()
        return (POLICY, r) if r is not None else (None, None)

    def _take(self, src: str, r: Request | None = None):
        if src == LANDED:
            if r is None or r is self.landed[0]:
                self.landed.popleft()
            else:
                self.landed.remove(r)
        elif src == OVERDUE:
            self.overdue.pop()
        else:
            self.policy.pop()

    def _admit(self, budget: int, now_ns: int, no_longer_than: int | None = None, overdue_only: bool = False) -> int:
        """Admit fresh prompts into `budget`: landed reads, overdue requests, then the pending order, stopping at the
        first pending one with more than `no_longer_than` tokens to compute. Returns the budget taken."""
        left = budget
        passed = []
        while left > 0:
            src, r = self._peek(overdue_only, now_ns)
            if r is None:
                break
            if src != LANDED and self.staged and self._needs_read(r, now_ns) is not None:
                self.stats["passed_over"] += 1      # not ready: the loading pass reads it
                self._take(src)
                passed.append((src, r))
                continue
            if src != LANDED and self.kv.busy(r):    # its kept context is still being written to a tier
                self.stats["busy_waits"] += 1
                self._take(src)
                passed.append((src, r))
                continue
            if src != LANDED and self.pool is not None and not self.staged:
                outcome = self._at_head(r, now_ns)
                if outcome == "read":
                    self._take(src)
                    continue
                if outcome == "blocked":
                    break
                if outcome == "passed":
                    self._take(src)
                    passed.append((src, r))
                    continue
            hit = r.cached if src == LANDED else self.kv.peek_hit(r)
            remaining = r.num_tokens - hit
            if src == POLICY and no_longer_than is not None and remaining > no_longer_than:
                break
            n = min(remaining, left)
            if n == 0:
                break
            if self.staged and src != LANDED and hit:
                r.hbm_ready = True                  # its context is in HBM now
            if (src != LANDED or self.staged) and not self.kv.fits(r):
                if r.room_wait_ns is None:
                    r.room_wait_ns = now_ns
                self._room_blocked = r
                break
            self._take(src, r)
            if src == LANDED and self.staged:
                self.kv.allocate(r)                 # takes over its read's blocks in the loading area
                r.alloc = self.kv.held[r.req_id]
            elif src != LANDED:
                r.cached = r.hbm_tokens = self.kv.allocate(r)
                r.alloc = self.kv.held[r.req_id]
            r.admit_ns = now_ns
            self._schedule(r, n)
            left -= n
            if n < remaining:
                self.started.append(r)
        for src, r in reversed(passed):
            (self.policy if src == POLICY else self.overdue).push_front(r)
        return budget - left

    def _at_head(self, r: Request, now_ns: int) -> str:
        """Start a tier read if DRAM or SSD holds more of the round's prefix than HBM: "admit" (no read needed),
        "read" (started), "blocked" (its footprint does not fit), "passed" (its tier is past the read-wait bound)."""
        if r.session is None:
            return "admit"
        hbm = self.kv.peek_hit(r)
        hit = self.pool.lookup(r.session, r, now_ns)
        if hit is None or hit.tokens <= hbm:
            return "admit"
        if self.max_wait_ns is not None and hit.wait_ns > self.max_wait_ns:
            self.stats["passed_over"] += 1
            return "passed"
        if not self.kv.fits(r):
            self.stats["blocked_reads"] += 1
            if r.room_wait_ns is None:
                r.room_wait_ns = now_ns
            self._room_blocked = r
            return "blocked"
        r.hbm_tokens = self.kv.allocate(r)
        r.alloc = self.kv.held[r.req_id]
        h = self.pool.load(r.session, hit, KVViews(r.alloc.kv_blocks, r.alloc.state_blocks, hit.tokens, r.hbm_tokens),
                           now_ns, r.req_id)
        r.cached = hit.tokens
        r.tier = hit.tier
        r.tier_tokens = hit.tokens - r.hbm_tokens
        r.read_start_ns = now_ns
        r.read_land_ns = h.eta_ns
        self.reading[r.req_id] = r
        self.stats["reads"] += 1
        return "read"


class PlainScheduler(Scheduler):
    """Phase-A constructor: the plain scheduler with a fixed budget."""

    def __init__(self, kv: KvManager, mb_tokens: int, max_in_flight: int = MAX_IN_FLIGHT, dispatch_ahead: int = 1):
        super().__init__(kv, SchedulerConfig(mb_tokens=mb_tokens), None, max_in_flight, dispatch_ahead)
