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
- **Writes** (`prepare_store`). When a round ends its context goes to the fastest tier (DRAM), only what the key's
  entry there does not already hold: the new tokens' KV and the new state. With `ssd_writes = "dram-evictions"` the
  SSD is written only with what DRAM evicts (or cannot hold), again only what SSD's entry lacks; a context SSD
  already holds whole is not written again. A context whose session moved on with a shorter one (a prompt that keeps
  only part of the previous context) no longer resumes from its state, so a victim tier's older entry for that key
  is dropped. With `"every-context"`, SSD is written through like DRAM. Writes are free in the target; the engine
  issues them and goes on.
- **Reads** (`lookup`, `prepare_load`). One channel per tier: reads queue one after another at the tier's bandwidth;
  the two tiers read in parallel. A read moves the tokens HBM lacks plus the KDA state. `Hit.wait_ns` is how long a
  read started now waits for its channel; the read-wait bound is the scheduler's. A read from SSD also puts the
  entry back in DRAM (copy-up); in slot mode that is the same op: SSD -> the new DRAM slots -> HBM.
- **Slot order.** A context that grows keeps the slots of its full blocks and takes new ones for its last (partial)
  block, the new blocks and its state. A freed slot is reused at once; the ops keep their order through `TierOp.deps`:
  an op waits for the ops in flight that write a slot it reads or writes, or read a slot it writes (an evicted
  context's demote is emitted before the write that reuses its slots). The leader sends an op once its deps are
  done (`ready_ops`) and reports it done (`complete`).
"""

from collections import OrderedDict, deque
from dataclasses import dataclass, field

from ppimpl.config import BLOCK_TOKENS, TierConfig
from ppimpl.engine.kv import kv_blocks_for, prefix_content, resumable_tokens

NS = 1_000_000_000


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
    `host_slots`/`host_state` are the host side of an SSD op on the same blocks: the DRAM entry (demote: the source;
    an SSD read: the copy-up target) or the staging area (`host_tier`). An SSD read reads all of `slots` (from block
    0) into the host side and copies to HBM only the blocks from token_start // BLOCK_TOKENS on."""
    op_id: int
    op: str                    # "write" HBM -> tier, "read" tier -> HBM, "demote" DRAM -> SSD, "promote" SSD -> DRAM,
                               # "drop" forget the tier's entry
    tier: str                  # destination tier for write/demote/promote, source tier for read, the tier for drop
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
        return (0, 0) if self.op == "demote" else (self.token_start // BLOCK_TOKENS, kv_blocks_for(self.token_end))


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
                 state_bytes: int, victim: bool, slot_bytes: int | None = None, state_slots: int = 0):
        self.name = name
        self.capacity = capacity_bytes
        self.read_bps = read_bytes_per_s
        self.bpt = bytes_per_token
        self.state_bytes = state_bytes
        self.victim = victim                 # written only with what the tier above evicts
        self.slot_bytes = slot_bytes
        self.state_slots = state_slots
        self.free: deque[int] = deque(range(int(capacity_bytes // slot_bytes))) if slot_bytes else deque()
        self._ring = 0
        self.entries: OrderedDict[object, TierEntry] = OrderedDict()
        self.used = 0.0
        self.channel_free_ns = 0
        self.stats = dict(stored=0, written_bytes=0, evicted=0, evicted_tokens=0, reads=0, read_tokens=0,
                          read_bytes=0, dropped=0)

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
        if c > self.capacity:
            self.release(reuse)
            return False, [(key, TierEntry(tokens, content))]
        left = []
        while self.used + c > self.capacity or len(self.free) < need:
            victim, e = self.entries.popitem(last=False)
            self.used -= self.charge(e.tokens)
            self.release(e.slots + e.state)
            self.stats["evicted"] += 1
            self.stats["evicted_tokens"] += e.tokens
            left.append((victim, e))
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


class TierManager:
    """The DRAM/SSD index of one pipeline, fastest first, and the ops in flight (slot mode).

    `prepare_load` / `prepare_store` return the ops they emitted, in order. Slot mode keeps the byte-moving ones for
    `ready_ops` / `complete`; byte mode (replay) keeps none: nothing executes them, reads land at their `land_ns`."""

    def __init__(self, cfg: TierConfig, partial_hits: bool = False):
        self.cfg = cfg
        self.partial_hits = partial_hits
        self.slot_mode = bool(cfg.slot_bytes)
        self.guard = SlotGuard() if self.slot_mode else None
        kw = dict(slot_bytes=cfg.slot_bytes, state_slots=cfg.state_slots)
        self.tiers: list[TierStore] = []
        if cfg.dram_gb:
            self.tiers.append(TierStore("dram", cfg.dram_gb * 1e9, cfg.dram_read_gb_s * 1e9, cfg.bytes_per_token,
                                        cfg.state_bytes, victim=False, **kw))
        if cfg.ssd_gb:
            self.tiers.append(TierStore("ssd", cfg.ssd_gb * 1e9, cfg.ssd_read_gb_s * 1e9, cfg.bytes_per_token,
                                        cfg.state_bytes, victim=bool(self.tiers) and cfg.ssd_writes == "dram-evictions",
                                        **kw))
        self.staging = TierStore("staging", cfg.staging_gb * 1e9, 0, 0, 0, False, **kw) if self.slot_mode else None
        assert cfg.ssd_writes in ("dram-evictions", "every-context")
        self.level = {t.name: i for i, t in enumerate(self.tiers)}
        self._out: list[TierOp] = []                              # emitted by the current prepare_* call
        self.waiting: OrderedDict[int, TierOp] = OrderedDict()   # slot mode: emitted, deps not done yet
        self.in_flight: dict[int, TierOp] = {}
        self._entry_ids: dict[object, int] = {}
        self._next_op = 0

    def _take(self) -> list[TierOp]:
        ops, self._out = self._out, []
        return ops

    def _entry(self, key) -> int:
        return self._entry_ids.setdefault(key, len(self._entry_ids))

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
        if o.op == "read":
            return h, t
        if o.op in ("demote", "promote"):
            return t, h
        return [], []

    # --- the leader's side of slot mode ---------------------------------------------------------------------------

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
        return o

    # --- lookups ------------------------------------------------------------------------------------------------

    def lookup(self, key, prompt, now_ns: int = 0) -> Hit | None:
        """The fastest tier holding the longest context of `key` that `prompt` (a `scheduler.Request`) resumes from."""
        best = None
        for i, t in enumerate(self.tiers):
            e = t.entries.get(key)
            if e is not None:
                n = resumable_tokens(e.tokens, e.content, prompt, self.partial_hits)
                if n > 0 and (best is None or n > best[1]):
                    best = (i, n)
        return None if best is None else Hit(self.tiers[best[0]].name, best[1], self.wait_ns(best[0], now_ns))

    def wait_ns(self, tier: int | str, now_ns: int) -> int:
        """How long a read started at `now_ns` waits for the tier's channel."""
        return max(0, self.tiers[self.level.get(tier, tier)].channel_free_ns - now_ns)

    def read_ns(self, tier: int | str, tokens: int, hbm_tokens: int = 0) -> int:
        t = self.tiers[self.level.get(tier, tier)]
        if self.slot_mode and t.name == "ssd":
            hbm_tokens = 0              # an SSD read reads the whole context (copy-up)
        return round(t.move_bytes(hbm_tokens, tokens) / t.read_bps * NS)

    # --- reads --------------------------------------------------------------------------------------------------

    def prepare_load(self, key, hit: Hit, views: KVViews | list[KVViews] | None, now_ns: int,
                     req_id: str | None = None) -> tuple[list[TierOp], int] | None:
        """Queue a read of the key's first `hit.tokens` from `hit.tier` into `views` (HBM already has their first
        `hbm_tokens`). Copies an SSD context up into DRAM. Returns (the ops, the read's expected landing time), or
        None if the tier no longer holds the key."""
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
            for faster in range(tier):
                self._store_in(faster, key, tokens, e.content, "promote", views)
            return self._take(), land
        n = kv_blocks_for(tokens)
        if t.name == "dram":
            b0 = hbm_tokens // BLOCK_TOKENS
            self._emit("read", t.name, key, block_start=b0, slots=e.slots[b0:n], state_slots=list(e.state), **kw)
            return self._take(), land
        host = self._store_in(0, key, tokens, e.content, "promote", emit=False) if tier > 0 else None
        if host is None:                # no DRAM, or it cannot hold the context: through the staging area
            hs = self.staging.take(n + self.staging.state_slots)
            host_tier, host_slots, host_state = "staging", hs[:n], hs[n:]
        else:
            host_tier, host_slots, host_state = "dram", host.slots[:n], host.state
        self._emit("read", t.name, key, slots=e.slots[:n], state_slots=list(e.state), host_tier=host_tier,
                   host_slots=host_slots, host_state=host_state, **kw)
        return self._take(), land

    # --- writes -------------------------------------------------------------------------------------------------

    def prepare_store(self, key, prompt, views: KVViews | list[KVViews] | None = None) -> list[TierOp]:
        """A round of `prompt` (a `scheduler.Request`) ended in `views`: write its context to the fastest tier and
        every write-through tier; a victim tier's older entry stays valid only if the new context starts with it.
        `key`: the session (replay) or the context's content key (the server, as the HBM cache). Returns the ops."""
        req = prompt
        tokens = req.num_tokens
        content = prefix_content(req.token_ids, tokens) if req.token_ids is not None else None
        for t in self.tiers[1:]:
            e = t.entries.get(key)
            if t.victim and e is not None and resumable_tokens(e.tokens, e.content, req, self.partial_hits) != e.tokens:
                self._drop(t, key)
        for i, t in enumerate(self.tiers):
            if i == 0 or not t.victim:
                self._store_in(i, key, tokens, content, "write", views, req)
        return self._take()

    def _drop(self, t: TierStore, key):
        e = t.remove(key)
        if e is not None:
            t.release(e.slots + e.state)
        t.stats["dropped"] += 1
        self._emit("drop", t.name, key)

    def _keeps(self, old: TierEntry | None, tokens: int, req) -> int:
        """Tokens of the tier's old entry for the session that the new context still starts with (not rewritten).
        A tier's entries for a session are always prefixes of its latest context (a victim tier drops one that is
        not, `store`), so for copy-up and demotion that is the old entry whenever it is not longer."""
        if old is None:
            return 0
        if req is not None:
            return old.tokens if resumable_tokens(old.tokens, old.content, req, self.partial_hits) == old.tokens else 0
        return old.tokens if old.tokens <= tokens else 0

    def _store_in(self, i: int, key, tokens: int, content, op: str, views=None, req=None,
                  src: TierEntry | None = None, emit: bool = True) -> TierEntry | None:
        """Store into tier i (op "write" from HBM, "promote" SSD -> DRAM, "demote" DRAM -> SSD from `src`). What the
        tier evicts for it goes down to the next tier if that one is a victim tier, then is dropped here. Returns the
        new entry (None if not stored). `emit` False: the caller's op moves the bytes (an SSD read's copy-up)."""
        t = self.tiers[i]
        old = t.entries.get(key)
        keeps = self._keeps(old, tokens, req)
        stored, left = t.store(key, tokens, content, keeps)
        entry = t.entries.get(key) if stored else None
        nxt = i + 1 if i + 1 < len(self.tiers) and self.tiers[i + 1].victim else None
        too_large = False
        for victim, e in left:
            if victim == key:                       # larger than this whole tier: straight to the victim tier
                too_large = True
                continue
            if nxt is not None:
                held = self.tiers[nxt].entries.get(victim)
                if held is not None and held.tokens >= e.tokens:
                    self.tiers[nxt].touch(victim)
                else:
                    self._store_in(nxt, victim, e.tokens, e.content, "demote", src=e)
            self._emit("drop", t.name, victim)
        if stored and emit:
            kw = _blocks(views)
            if self.slot_mode:
                b0 = keeps // BLOCK_TOKENS
                kw |= dict(block_start=b0, slots=entry.slots[b0:], state_slots=list(entry.state))
                if src is not None:                 # demote: from DRAM's slots of the same blocks
                    kw |= dict(host_tier="dram", host_slots=src.slots[b0:kv_blocks_for(tokens)],
                               host_state=list(src.state))
                elif t.name == "ssd":               # write to SSD from HBM: through the staging area
                    hs = self.staging.take(len(entry.slots) - b0 + self.staging.state_slots)
                    kw |= dict(host_tier="staging", host_slots=hs[:len(entry.slots) - b0],
                               host_state=hs[len(entry.slots) - b0:])
            self._emit(op, t.name, key, token_start=keeps, token_end=tokens, **kw)
        elif not stored and old is not None:
            self._emit("drop", t.name, key)
        if too_large and nxt is not None and op == "write":
            self._store_in(nxt, key, tokens, content, "write", views, req)
        return entry

    # --- reporting ----------------------------------------------------------------------------------------------

    def summary(self) -> list[dict]:
        out = [dict(tier=t.name, capacity_gb=t.capacity / 1e9, used_gb=t.used / 1e9, entries=len(t.entries),
                    **t.stats) for t in self.tiers]
        if self.slot_mode:
            for d, t in zip(out, self.tiers):
                d |= dict(slots=t.num_slots, free_slots=len(t.free))
        return out
