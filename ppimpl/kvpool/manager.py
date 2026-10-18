"""Layer 5 of the KV pool (lanes/m6 TIERS.md §3-§7, §13.2): the DRAM and SSD index behind the HBM prefix cache
(7_offloading.md §2). Pure policy on the leader, no torch: it decides, the members move the bytes. Every decision
that moves bytes becomes a `TierOp` the leader sends to all n members (n = PP degree).

- **What a tier keeps.** One entry per key (a session): its latest context, KV of the prompt's tokens plus the KDA
  state at the prompt's end. Each tier evicts on its own, least recently used first, a whole entry at a time; a read
  refreshes its entry.
- **Accounting.** Bytes (`TierConfig.slot_bytes` None: replay, the target's own accounting, bytes per token plus one
  state per context) or slots (the server, TIERS.md §3): a tier is a free list of fixed slots, one slot per KV block
  of the context plus `state_slots` for its state, the same slot ids on every GPU. Each GPU's slot holds its 1/n of
  that block on all n stages (balanced storage, `placement.py`).
- **Victim chain HBM -> DRAM -> SSD** (user, 2026-10-17). Each tier is written only with what the level above lets
  go of, and only what its entry for the key lacks: the fastest tier with HBM's kept contexts
  (`prepare_write_back`, driven by the HBM level's `writeback.WriteBack` in `BalancedKVPool`), SSD with DRAM's
  entries (`demote`, driven by DRAM's `WriteBack` here: ahead of need above `dram_write_ahead`, or at eviction).
  A level letting go of what the one below already holds whole only refreshes it there (`hbm_evicted`); an entry
  stored where the tier below already holds it whole is clean. A read back into HBM leaves the entry in place (a clean
  copy); an SSD read goes through the staging area, not into DRAM. **Exclusive DRAM** (`TierConfig.dram_writes`
  "hbm-evictions-exclusive"): a DRAM read releases its entry (its slots are reused only after the read, `deps`), and
  a round admitted on HBM's kept context releases the DRAM copy write-ahead made of it (`taken_over`): DRAM holds only
  what HBM lacks, and HBM evicting the context again writes it whole. SSD stays inclusive. When a
  session's next round is admitted, every entry it does not resume from is dropped (`drop_stale`), so entries stay
  prefixes of the session's latest context (a KDA state exists only at a context's end). Writes are free in the
  target; the engine issues them and goes on.
- **Reads** (`lookup`, `prepare_load`). One channel per tier: reads queue one after another at the tier's bandwidth;
  the two tiers read in parallel. A read moves the tokens HBM lacks plus the KDA state. `Hit.wait_ns` is how long a
  read started now waits for its channel; the read-wait bound is the scheduler's. An SSD read reads the whole
  context into staging slots and copies to HBM what HBM lacks.
- **Slot order.** A context that grows keeps the slots of its full blocks and takes new ones for its last (partial)
  block, the new blocks and its state. A freed slot is reused at once; the ops keep their order through `TierOp.deps`:
  an op waits for the ops in flight that write a slot it reads or writes, or read a slot it writes (an evicted
  context's demote is emitted before the write that reuses its slots). The leader sends an op once its deps are
  done (`ready_ops`) and reports it done (`complete`).
"""

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np

from ppimpl.config import BLOCK_TOKENS, TierConfig
from ppimpl.engine.kv import kv_blocks_for, resumable_tokens
from ppimpl.kvpool.writeback import WriteBack

NS = 1_000_000_000

def states_to_move(states: list[int], resume: int, k: int, added: int) -> list[int]:
    """Which of an entry's round-end states (token positions, ascending) a read moves into HBM: the state it resumes
    from (`resume`) plus only the earlier ones that stay among the latest `k` once the round adds its own `added`
    (its prefill end and decode end, after `resume`). The engine keeps one state per entry (k = 1: the resume state
    only); with exclusive DRAM the states not moved are dropped with the released entry."""
    assert resume in states and k >= 1
    keep = max(k - added - 1, 0)
    earlier = [x for x in states if x < resume]
    return (earlier[-keep:] if keep else []) + [resume]


@dataclass
class KVViews:
    """A context's blocks in the KV caches the members registered: KV block ids from block 0 and the state blocks.
    One KVViews names the same blocks on every stage (this engine: one allocation for all); a list of n names each
    stage's own. A load fills tokens [hbm_tokens, tokens): HBM already holds the first hbm_tokens."""
    kv_blocks: list[int]
    state_blocks: list[int]
    tokens: int = 0
    hbm_tokens: int = 0


@dataclass(frozen=True)
class Hit:
    """`lookup`: the fastest tier holding the longest context of the key the prompt resumes from, and how long a
    read started now waits for that tier's channel."""
    tier: str
    tokens: int
    wait_ns: int


def _blocks(views: KVViews | list[KVViews] | None) -> dict:
    """A TierOp's block lists: one list for every stage, or one per stage."""
    if views is None:
        return {}
    vs = views if isinstance(views, list) else [views]
    return dict(kv_blocks=[list(v.kv_blocks) for v in vs], state_blocks=[list(v.state_blocks) for v in vs])


@dataclass
class TierOp:
    """One byte move for the members. `kv_blocks`/`state_blocks` are HBM block ids, one list for every stage or one
    per stage (reads: the held blocks to fill; writes: the kept blocks to copy from). Tokens [token_start, token_end)
    of KV plus the KDA state move.

    Slot mode: `slots[i]` is `tier`'s slot of the context's KV block `block_start + i`, `state_slots` its state.
    `host_slots`/`host_state` are the host side of an SSD op on the same blocks: the DRAM entry (a demote's source)
    or the staging area (an SSD read's target; an SSD write from HBM without DRAM) (`host_tier`). An SSD read reads all of `slots` (from block
    0) into the host side and copies to HBM only the blocks from token_start // BLOCK_TOKENS on."""
    op_id: int
    op: str                    # "write" HBM -> tier, "read" tier -> HBM, "demote" DRAM -> SSD, "fetch" SSD ->
                               # DRAM's loading area (staged loading), "drop" forget the tier's entry
    tier: str                  # destination tier for write/demote, source tier for read, the tier for drop
    entry: int                 # the key's entry id (same on every member)
    token_start: int = 0
    token_end: int = 0
    kv_blocks: list[list[int]] = field(default_factory=list)
    state_blocks: list[list[int]] = field(default_factory=list)
    req_id: str | None = None  # read: the request it is for
    land_ns: int | None = None  # read: when the engine expects it to land
    block_start: int = 0
    slots: list[int] = field(default_factory=list)
    state_slots: list[int] = field(default_factory=list)
    host_tier: str | None = None
    host_slots: list[int] = field(default_factory=list)
    host_state: list[int] = field(default_factory=list)
    deps: list[int] = field(default_factory=list)

    @property
    def moves_bytes(self) -> bool:
        return self.op != "drop"

    @property
    def blocks(self) -> tuple[int, int]:
        """HBM blocks [b_lo, b_hi) the op moves: from the block holding token_start to the last one (none for a
        demote)."""
        return (0, 0) if self.op in ("demote", "fetch") else (self.token_start // BLOCK_TOKENS, kv_blocks_for(self.token_end))


@dataclass
class TierEntry:
    tokens: int
    content: object | None             # the context's token ids (kv.prefix_content), None in replay
    slots: list[int] = field(default_factory=list)        # slot mode: one per KV block
    state: list[int] = field(default_factory=list)


class SlotGuard:
    """The last op that writes each slot and the ops that read it, among the ops not yet done."""

    def __init__(self):
        self.writer: dict[tuple, int] = {}
        self.readers: dict[tuple, set[int]] = {}
        self.touch: dict[int, tuple[list, list]] = {}      # op id -> (written keys, read keys)

    def add(self, op_id: int, writes: list, reads: list) -> list[int]:
        deps = {self.writer[k] for k in (*reads, *writes) if k in self.writer}
        for k in writes:
            deps |= self.readers.get(k, set())
        deps.discard(op_id)
        for k in writes:
            self.writer[k] = op_id
        for k in reads:
            self.readers.setdefault(k, set()).add(op_id)
        self.touch[op_id] = (writes, reads)
        return sorted(deps)

    def done(self, op_id: int):
        writes, reads = self.touch.pop(op_id, ((), ()))
        for k in writes:
            if self.writer.get(k) == op_id:
                del self.writer[k]
        for k in reads:
            r = self.readers.get(k)
            if r is not None:
                r.discard(op_id)
                if not r:
                    del self.readers[k]


class TierStore:
    """One tier's LRU of session contexts, in one GPU's bytes (byte mode) or slots."""

    def __init__(self, name: str, capacity_bytes: float, read_bytes_per_s: float, bytes_per_token: int,
                 state_bytes: int, slot_bytes: int | None = None, state_slots: int = 0):
        self.name = name
        self.capacity = capacity_bytes
        self.read_bps = read_bytes_per_s
        self.bpt = bytes_per_token
        self.state_bytes = state_bytes
        self.slot_bytes = slot_bytes
        self.state_slots = state_slots
        self.free: deque[int] = deque(range(int(capacity_bytes // slot_bytes))) if slot_bytes else deque()
        self._ring = 0
        self.entries: OrderedDict[object, TierEntry] = OrderedDict()
        self.used = 0.0
        self.area = 0                   # slot mode: loading area (staged loading) in slots, held whether used or not
        self.area_used = 0              # slots its reads hold
        self.channel_free_ns = 0
        self.stats = dict(stored=0, written_bytes=0, evicted=0, evicted_tokens=0, reads=0, read_tokens=0,
                          read_bytes=0, dropped=0, released=0,      # released: exclusive DRAM, moved into HBM
                          hops=0, hop_bytes=0)                      # DRAM: SSD reads moved on from its area to HBM

    @property
    def num_slots(self) -> int:
        return int(self.capacity // self.slot_bytes) if self.slot_bytes else 0

    def charge(self, tokens: int) -> float:
        if self.slot_bytes:
            return (kv_blocks_for(tokens) + self.state_slots) * self.slot_bytes
        return tokens * self.bpt + self.state_bytes

    def move_bytes(self, token_start: int, token_end: int) -> float:
        """What a read or write of tokens [token_start, token_end) plus the state moves on one GPU."""
        if self.slot_bytes:
            return (kv_blocks_for(token_end) - token_start // BLOCK_TOKENS + self.state_slots) * self.slot_bytes
        return (token_end - token_start) * self.bpt + self.state_bytes

    def release(self, slots: list[int]):
        self.free.extend(slots)

    def _short(self, c: float, need: int) -> bool:
        """Whether an entry of charge `c` taking `need` free slots does not fit beside the entries and the loading
        area (held whole, at least; its unused slots stay free)."""
        sb = self.slot_bytes or 0
        return (self.used + c > self.capacity - max(self.area_used, self.area) * sb
                or len(self.free) - max(self.area - self.area_used, 0) < need)

    def _evict_lru(self) -> tuple[object, TierEntry]:
        victim, e = self.entries.popitem(last=False)
        self.used -= self.charge(e.tokens)
        self.release(e.slots + e.state)
        self.stats["evicted"] += 1
        self.stats["evicted_tokens"] += e.tokens
        return victim, e

    def area_fits(self, n: int) -> bool:
        """Whether a read of `n` slots may start in the loading area: it fits, or the area is empty."""
        return self.area_used == 0 or self.area_used + n <= self.area

    def area_take(self, n: int) -> tuple[list[int], list[tuple[object, TierEntry]]]:
        """`n` slots of the loading area (the caller checked `area_fits`), and the entries evicted for them (a read
        larger than the area)."""
        self.area_used += n
        left = []
        while self.entries and self._short(0, n):
            left.append(self._evict_lru())
        assert len(self.free) >= n, (self.name, n, len(self.free))
        return [self.free.popleft() for _ in range(n)], left

    def area_give(self, slots: list[int]):
        self.area_used -= len(slots)
        self.release(slots)

    def remove(self, key) -> TierEntry | None:
        e = self.entries.pop(key, None)
        if e is not None:
            self.used -= self.charge(e.tokens)
        return e

    def touch(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)

    def store(self, key, tokens: int, content, keeps: int) -> tuple[bool, list[tuple[object, TierEntry]]]:
        """Store the session's context; `keeps` tokens of the old entry are still valid (written already).
        Returns (stored, what left the tier: LRU victims, or the context itself when it exceeds the tier).
        Slot mode: the old entry's full blocks below `keeps` keep their slots; everything else gets new slots, which
        may be the victims' (the caller emits the victims' demotes before this context's write)."""
        old = self.remove(key)
        c = self.charge(tokens)
        reuse = old.slots[:keeps // BLOCK_TOKENS] if (old is not None and self.slot_bytes) else []
        if old is not None and self.slot_bytes:
            self.release(old.slots[len(reuse):] + old.state)
        need = kv_blocks_for(tokens) - len(reuse) + self.state_slots if self.slot_bytes else 0
        if c > self.capacity - self.area * (self.slot_bytes or 0):
            self.release(reuse)
            return False, [(key, TierEntry(tokens, content))]
        left = []
        while self._short(c, need):
            left.append(self._evict_lru())
        e = TierEntry(tokens, content)
        if self.slot_bytes:
            fresh = [self.free.popleft() for _ in range(need)]
            n_kv = kv_blocks_for(tokens) - len(reuse)
            e.slots, e.state = reuse + fresh[:n_kv], fresh[n_kv:]
        self.entries[key] = e
        self.used += c
        self.stats["stored"] += 1
        self.stats["written_bytes"] += self.move_bytes(keeps, tokens)
        return True, left

    def take(self, n: int) -> list[int]:
        """The staging area of an SSD op without DRAM: the next n slots of a ring. A slot an earlier op still uses is
        taken anyway; the new op waits for that one (`SlotGuard` deps)."""
        if n > self.num_slots:
            raise RuntimeError(f"{self.name}: {n} slots wanted, {self.num_slots} in all; raise staging_gb")
        out = [self.free[i % self.num_slots] for i in range(self._ring, self._ring + n)]
        self._ring = (self._ring + n) % self.num_slots
        return out



class _TierLevel:
    """A tier as the upper level of `WriteBack` (its LRU; demotes to the tier below)."""

    def __init__(self, m: "TierManager", i: int):
        self.m, self.i, self.t = m, i, m.tiers[i]
        self.capacity = self.t.capacity

    def fixed(self) -> float:
        return max(self.t.area_used, self.t.area) * (self.t.slot_bytes or 0)     # its loading area

    def lru(self):
        return iter(self.t.entries)

    def write(self, key, entry=None):
        e = entry or self.t.entries[key]
        _, op = self.m._store_in(self.i + 1, key, e.tokens, e.content, "demote", src=e)
        return op


class TierManager:
    """The DRAM/SSD index of one pipeline, fastest first, and the ops in flight (slot mode).

    `prepare_load` / `prepare_write_back` return the ops they emitted, in order. Slot mode keeps the byte-moving ones
    for `ready_ops` / `complete`; byte mode (replay) keeps none: nothing executes them, reads land at their `land_ns`,
    and a level writes at eviction (writes take no time)."""

    def __init__(self, cfg: TierConfig, partial_hits: bool = False):
        self.cfg = cfg
        self.partial_hits = partial_hits
        if cfg.dram_writes not in ("hbm-evictions", "hbm-evictions-exclusive"):
            raise ValueError(f"tiers.dram_writes: {cfg.dram_writes!r} (hbm-evictions | hbm-evictions-exclusive)")
        self.exclusive = cfg.dram_writes == "hbm-evictions-exclusive" and bool(cfg.dram_gb)
        self.slot_mode = bool(cfg.slot_bytes)
        self.guard = SlotGuard() if self.slot_mode else None
        kw = dict(slot_bytes=cfg.slot_bytes, state_slots=cfg.state_slots)
        self.tiers: list[TierStore] = []
        if cfg.dram_gb:
            self.tiers.append(TierStore("dram", cfg.dram_gb * 1e9, cfg.dram_read_gb_s * 1e9, cfg.bytes_per_token,
                                        cfg.state_bytes, **kw))
        if cfg.ssd_gb:
            self.tiers.append(TierStore("ssd", cfg.ssd_gb * 1e9, cfg.ssd_read_gb_s * 1e9, cfg.bytes_per_token,
                                        cfg.state_bytes, **kw))
        self.staging = TierStore("staging", cfg.staging_gb * 1e9, 0, 0, 0, **kw) if self.slot_mode else None
        self.level = {t.name: i for i, t in enumerate(self.tiers)}
        self._out: list[TierOp] = []                              # emitted by the current prepare_* call
        self.waiting: OrderedDict[int, TierOp] = OrderedDict()   # slot mode: emitted, deps not done yet
        self.in_flight: dict[int, TierOp] = {}
        self._entry_ids: dict[object, int] = {}
        self._keys: dict[int, object] = {}
        self._next_op = 0
        self.gone: list = []        # keys whose entry left the fastest tier (HBM's kept copy is then no longer held)
        # slot mode: each tier with one below writes back to it (DRAM -> SSD); the same policy as HBM -> DRAM
        self.wb: dict[int, WriteBack] = {i: WriteBack(t.name, _TierLevel(self, i), cfg.dram_write_ahead)
                                         for i, t in enumerate(self.tiers[:-1])} if self.slot_mode else {}

    def _take(self) -> list[TierOp]:
        ops, self._out = self._out, []
        return ops

    def _entry(self, key) -> int:
        i = self._entry_ids.setdefault(key, len(self._entry_ids))
        self._keys[i] = key
        return i

    def _emit(self, op: str, tier: str, key, **kw) -> TierOp:
        o = TierOp(self._next_op, op, tier, self._entry(key), **kw)
        self._next_op += 1
        if self.slot_mode and o.moves_bytes:
            o.deps = self.guard.add(o.op_id, *self._touches(o))
            self.waiting[o.op_id] = o
        self._out.append(o)
        return o

    @staticmethod
    def _touches(o: TierOp) -> tuple[list, list]:
        """(written, read) slot keys of a byte-moving op."""
        t = [(o.tier, s) for s in o.slots + o.state_slots]
        h = [(o.host_tier, s) for s in o.host_slots + o.host_state]
        if o.op == "write":
            return t + h, []            # SSD write through staging: staging written, then read, by this op
        if o.op in ("read", "fetch"):
            return h, t
        if o.op == "demote":
            return t, h
        return [], []

    # --- the leader's side of slot mode ---------------------------------------------------------------------------

    def step(self) -> list[TierOp]:
        """Slot mode: each tier writes ahead to the one below above its watermark (`WriteBack.step`)."""
        for wb in self.wb.values():
            wb.step()
        return self._take()

    def ready_ops(self) -> list[TierOp]:
        """Slot mode: the emitted ops whose deps are done (to send now), in emission order."""
        ready = [o for o in self.waiting.values() if not any(d in self.waiting or d in self.in_flight
                                                                 for d in o.deps)]
        for o in ready:
            del self.waiting[o.op_id]
            self.in_flight[o.op_id] = o
        return ready

    def complete(self, op_id: int, now_ns: int) -> TierOp:
        """Every member reported the op done at `now_ns`: free what it held, its dependents may go. A read's channel
        is free no earlier than that (real reads may run slower than the configured bandwidth; `Hit.wait_ns` then
        sees the real queue)."""
        o = self.in_flight.pop(op_id)
        self.guard.done(op_id)
        if o.op == "read":
            t = self.tiers[self.level[o.tier]]
            t.channel_free_ns = max(t.channel_free_ns, now_ns)
        elif o.op == "demote":
            wb = self.wb.get(self.level[o.tier] - 1)
            if wb is not None:
                wb.done(self._keys[o.entry], o)
        return o

    # --- lookups ------------------------------------------------------------------------------------------------

    def lookup(self, key, prompt, now_ns: int = 0) -> Hit | None:
        """The fastest tier holding the longest context of `key` that `prompt` (a `scheduler.Request`) resumes from."""
        best = None
        memo = getattr(prompt, "tier_memo", False) if self.slot_mode else False
        if memo is None:
            memo = prompt.tier_memo = {}
        for i, t in enumerate(self.tiers):
            e = t.entries.get(key)
            if e is not None:
                m = memo.get(i) if memo is not False else None
                if m is not None and m[0] is e and m[1] == e.tokens:   # compared before (a long prompt: ~0.1 ms)
                    n = m[2]
                else:
                    n = resumable_tokens(e.tokens, e.content, prompt, self.partial_hits)
                    if memo is not False:
                        memo[i] = (e, e.tokens, n)
                if n > 0 and (best is None or n > best[1]):
                    best = (i, n)
        return None if best is None else Hit(self.tiers[best[0]].name, best[1], self.wait_ns(best[0], now_ns))

    def wait_ns(self, tier: int | str, now_ns: int) -> int:
        """How long a read started at `now_ns` waits for the tier's channel."""
        return max(0, self.tiers[self.level.get(tier, tier)].channel_free_ns - now_ns)

    def read_ns(self, tier: int | str, tokens: int, hbm_tokens: int = 0) -> int:
        t = self.tiers[self.level.get(tier, tier)]
        if self.slot_mode and t.name == "ssd":
            hbm_tokens = 0              # an SSD read reads the whole context (into staging)
        return round(t.move_bytes(hbm_tokens, tokens) / t.read_bps * NS)

    # --- reads --------------------------------------------------------------------------------------------------

    def prepare_load(self, key, hit: Hit, views: KVViews | list[KVViews] | None, now_ns: int,
                     req_id: str | None = None) -> tuple[list[TierOp], int] | None:
        """Queue a read of the key's first `hit.tokens` from `hit.tier` into `views` (HBM already has their first
        `hbm_tokens`); the entry stays. Returns (the ops, the read's expected landing time), or None if the tier no
        longer holds the key."""
        tier = self.level[hit.tier]
        t = self.tiers[tier]
        e = t.entries.get(key)
        if e is None:
            return None
        tokens = hit.tokens
        hbm_tokens = 0 if views is None else (views if isinstance(views, KVViews) else views[0]).hbm_tokens
        start = max(t.channel_free_ns, now_ns)
        land = start + self.read_ns(tier, tokens, hbm_tokens)
        t.channel_free_ns = land
        t.stats["reads"] += 1
        t.stats["read_tokens"] += tokens - hbm_tokens
        t.stats["read_bytes"] += t.move_bytes(hbm_tokens, tokens)
        t.touch(key)
        kw = dict(token_start=hbm_tokens, token_end=tokens, req_id=req_id, land_ns=land, **_blocks(views))
        if not self.slot_mode:
            self._emit("read", t.name, key, **kw)
            self._release_read(tier, key)
            return self._take(), land
        n = kv_blocks_for(tokens)
        if t.name == "dram":
            b0 = hbm_tokens // BLOCK_TOKENS
            self._emit("read", t.name, key, block_start=b0, slots=e.slots[b0:n], state_slots=list(e.state), **kw)
            self._release_read(tier, key)
            return self._take(), land
        hs = self.staging.take(n + self.staging.state_slots)       # SSD: the whole context through staging
        self._emit("read", t.name, key, slots=e.slots[:n], state_slots=list(e.state), host_tier="staging",
                   host_slots=hs[:n], host_state=hs[n:], **kw)
        return self._take(), land

    def _release_read(self, tier: int, key):
        """Exclusive DRAM: a read moves the entry into HBM; its slots are free for ops after the read (`deps`)."""
        t = self.tiers[tier]
        if not (self.exclusive and t.name == "dram"):
            return
        e = t.remove(key)
        t.release(e.slots + e.state)
        t.stats["released"] += 1
        self._removed(tier, key)

    def taken_over(self, key, tokens: int):
        """A round was admitted on HBM's kept context of `key` (`tokens` long). Exclusive DRAM releases the copy
        write-ahead made of it (a prefix of it, `drop_stale`): HBM holds it."""
        if not self.exclusive or tokens <= 0:
            return
        t = self.tiers[0]
        e = t.entries.get(key)
        if e is not None and e.tokens <= tokens:
            t.remove(key)
            t.release(e.slots + e.state)
            t.stats["released"] += 1
            self._removed(0, key)

    # --- staged loading: SSD reads through DRAM's loading area -------------------------------------------------------

    def set_dram_area(self, fraction: float):
        """Carve `fraction` of DRAM's slots (rounded down, as the simulator) out as its loading area (slot mode, an SSD
        behind DRAM)."""
        if not (self.slot_mode and len(self.tiers) == 2 and self.tiers[0].name == "dram"):
            return
        d = self.tiers[0]
        d.area = int(fraction * d.num_slots)

    @property
    def dram_area(self) -> TierStore | None:
        d = self.tiers[0] if self.tiers else None
        return d if d is not None and d.name == "dram" and d.area else None

    def fetch_slots(self, tokens: int) -> int:
        return kv_blocks_for(tokens) + self.cfg.state_slots

    def prepare_fetch(self, key, hit: Hit, now_ns: int, req_id: str | None = None
                      ) -> tuple[list[TierOp], int, list[int]] | None:
        """Read the key's first `hit.tokens` from SSD into slots of DRAM's loading area (the caller checked
        `area_fits`), evicting DRAM entries for a read larger than the area. Returns (the ops, the expected landing,
        the area slots: KV blocks from block 0, then the state), or None if the SSD no longer holds the key."""
        t = self.tiers[self.level["ssd"]]
        e = t.entries.get(key)
        if e is None:
            return None
        d = self.dram_area
        tokens = hit.tokens
        n = kv_blocks_for(tokens)
        area, left = d.area_take(n + self.cfg.state_slots)
        for victim, ve in left:
            self._evict(0, victim, ve)
        start = max(t.channel_free_ns, now_ns)
        land = start + self.read_ns(self.level["ssd"], tokens)
        t.channel_free_ns = land
        t.stats["reads"] += 1
        t.stats["read_tokens"] += tokens
        t.stats["read_bytes"] += t.move_bytes(0, tokens)
        t.touch(key)
        self._emit("fetch", "ssd", key, token_start=0, token_end=tokens, req_id=req_id, land_ns=land,
                   slots=e.slots[:n], state_slots=list(e.state), host_tier="dram", host_slots=area[:n],
                   host_state=area[n:])
        return self._take(), land, area

    def prepare_hop(self, key, tokens: int, area: list[int], views: KVViews, now_ns: int,
                    req_id: str | None = None) -> tuple[list[TierOp], int]:
        """Move a fetched context from DRAM's loading area `area` into `views` (HBM's loading area) on DRAM's read
        channel. The area slots are given back when it is done (`area_give`)."""
        d = self.dram_area
        n = kv_blocks_for(tokens)
        start = max(d.channel_free_ns, now_ns)
        land = start + round(d.move_bytes(0, tokens) / d.read_bps * NS)
        d.channel_free_ns = land
        d.stats["hops"] += 1
        d.stats["hop_bytes"] += d.move_bytes(0, tokens)
        self._emit("read", "dram", key, token_start=0, token_end=tokens, req_id=req_id, land_ns=land, block_start=0,
                   slots=area[:n], state_slots=area[n:], **_blocks(views))
        return self._take(), land

    # --- writes: the victim chain ---------------------------------------------------------------------------------

    def prepare_write_back(self, key, tokens: int, content, views=None) -> list[TierOp]:
        """HBM lets go of (or will soon let go of) its kept context of `key`, `tokens` long, in `views`: write what
        the fastest tier's entry lacks (nothing if it holds it all: the entry is refreshed). Returns the ops; a write
        among them reads `views` until done."""
        t = self.tiers[0]
        e = t.entries.get(key)
        if e is not None and e.tokens >= tokens:
            t.touch(key)
            return []
        self._store_in(0, key, tokens, content, "write", views)
        return self._take()

    def hbm_evicted(self, key, tokens: int, content) -> list[TierOp]:
        """HBM evicted its kept context of `key`. A tier holding it all is refreshed. Slot mode: HBM evicts only
        contexts already written back, so nothing moves (one no tier could hold is lost); byte mode (replay, writes
        take no time) writes it now, as the simulator does."""
        for t in self.tiers:
            e = t.entries.get(key)
            if e is not None and e.tokens >= tokens:
                t.touch(key)
                return []
        return [] if self.slot_mode else self.prepare_write_back(key, tokens, content)

    def drop_stale(self, key, req) -> list[TierOp]:
        """The session's next round `req` was admitted: an entry it does not resume from is dropped (byte mode with
        partial hits: cut to what it resumes), so every entry stays a prefix of the session's latest context."""
        for i, t in enumerate(self.tiers):
            e = t.entries.get(key)
            if e is None:
                continue
            n = resumable_tokens(e.tokens, e.content, req, self.partial_hits)
            if n == e.tokens:
                continue
            if n and not self.slot_mode:
                t.used -= t.charge(e.tokens) - t.charge(n)
                e.tokens = n
            else:
                self._drop(i, key)
        return self._take()

    def take_gone(self) -> list:
        g, self.gone = self.gone, []
        return g

    def _removed(self, i: int, key):
        """Tier i no longer holds `key`'s entry: the level above (HBM for tier 0) holds it dirty again."""
        if i in self.wb:
            self.wb[i].remove(key)
        if i == 0:
            self.gone.append(key)
        elif (wb := self.wb.get(i - 1)) is not None and (up := self.tiers[i - 1].entries.get(key)) is not None:
            wb.lost(key, self.tiers[i - 1].charge(up.tokens))

    def _drop(self, i: int, key):
        t = self.tiers[i]
        e = t.remove(key)
        if e is not None:
            t.release(e.slots + e.state)
            self._removed(i, key)
        t.stats["dropped"] += 1
        self._emit("drop", t.name, key)

    @staticmethod
    def _keeps(old: TierEntry | None, tokens: int, content) -> int:
        """Tokens of the tier's old entry for the key that the new context still starts with (not rewritten). Entries
        are prefixes of the session's latest context (`drop_stale`); the content check guards the rest."""
        if old is None or old.tokens > tokens:
            return 0
        if content is not None and old.content is not None:
            return old.tokens if np.array_equal(content[:old.tokens], old.content) else 0
        return old.tokens

    def _evict(self, i: int, key, e: TierEntry):
        """Tier i lets go of `key`'s entry `e` for room: written back to the tier below unless it holds it already
        (slot mode: `WriteBack.evicting`, which counts a wait when a demote is in flight or must start)."""
        t = self.tiers[i]
        if i + 1 < len(self.tiers):
            nxt = self.tiers[i + 1]
            if i in self.wb:
                state = self.wb[i].evicting(key, e)
            else:
                held = nxt.entries.get(key)
                state = "clean" if held is not None and held.tokens >= e.tokens else "dirty"
                if state == "dirty":
                    self._store_in(i + 1, key, e.tokens, e.content, "demote", src=e)
            if state == "clean":
                nxt.touch(key)
        self._removed(i, key)
        self._emit("drop", t.name, key)

    def _store_in(self, i: int, key, tokens: int, content, op: str, views=None,
                  src: TierEntry | None = None) -> tuple[TierEntry | None, TierOp | None]:
        """Store into tier i (op "write" from HBM, "demote" DRAM -> SSD from `src`), only what its entry lacks. What
        the tier evicts for it goes down to the next tier first. A context larger than the whole tier goes straight
        to the next one. Returns (the new entry or None, the op that moves its bytes or None)."""
        t = self.tiers[i]
        old = t.entries.get(key)
        keeps = self._keeps(old, tokens, content)
        stored, left = t.store(key, tokens, content, keeps)
        entry = t.entries.get(key) if stored else None
        too_large = False
        for victim, e in left:
            if victim == key:
                too_large = True
            else:
                self._evict(i, victim, e)
        moved = None
        if stored:
            if i in self.wb:
                below = self.tiers[i + 1].entries.get(key)
                if below is not None and below.tokens >= tokens:
                    self.wb[i].remove(key)          # the tier below holds it whole already: clean
                else:
                    self.wb[i].add(key, t.charge(tokens))
            kw = _blocks(views)
            if self.slot_mode:
                b0 = keeps // BLOCK_TOKENS
                kw |= dict(block_start=b0, slots=entry.slots[b0:], state_slots=list(entry.state))
                if src is not None:                 # demote: from DRAM's slots of the same blocks
                    kw |= dict(host_tier="dram", host_slots=src.slots[b0:kv_blocks_for(tokens)],
                               host_state=list(src.state))
                elif t.name == "ssd":               # write to SSD from HBM (no DRAM): through the staging area
                    hs = self.staging.take(len(entry.slots) - b0 + self.staging.state_slots)
                    kw |= dict(host_tier="staging", host_slots=hs[:len(entry.slots) - b0],
                               host_state=hs[len(entry.slots) - b0:])
            moved = self._emit(op, t.name, key, token_start=keeps, token_end=tokens, **kw)
        elif old is not None:
            self._removed(i, key)
            self._emit("drop", t.name, key)
        if too_large and i + 1 < len(self.tiers) and op == "write":
            return self._store_in(i + 1, key, tokens, content, "write", views)
        return entry, moved

    # --- reporting ----------------------------------------------------------------------------------------------

    def summary(self) -> list[dict]:
        out = [dict(tier=t.name, capacity_gb=t.capacity / 1e9, used_gb=t.used / 1e9, entries=len(t.entries),
                    **t.stats) for t in self.tiers]
        if self.slot_mode:
            for i, (d, t) in enumerate(zip(out, self.tiers)):
                d |= dict(slots=t.num_slots, free_slots=len(t.free))
                if i in self.wb:                    # its write-back to the tier below (TIERS.md §5)
                    d["write_back"] = dict(self.wb[i].stats)
        return out
