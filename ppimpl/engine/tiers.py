"""DRAM and SSD behind the HBM prefix cache (7_offloading.md §2). A pure index: the engine decides, the workers move
the bytes. Every decision that moves bytes becomes a `TierOp` the controller sends to all four workers.

- **What a tier keeps.** One entry per session: its latest context, KV of the prompt's tokens plus the KDA state at
  the prompt's end. Each tier evicts on its own, least recently used first, a whole session at a time; a read
  refreshes its session.
- **Own-stage storage.** Each GPU stores and reads its own stage's bytes. Stage 0 holds the most (1,635 B/token +
  39.1 MB per context), so it sets every tier's size and read time (`TierConfig`; balanced storage is a config).
- **Writes.** When a round ends its context goes to the fastest tier (DRAM), only what the session's entry there does
  not already hold: the new tokens' KV and the new state. With `ssd_writes = "dram-evictions"` the SSD is written only
  with what DRAM evicts (or cannot hold), again only what SSD's entry lacks; a context SSD already holds whole is not
  written again. A context whose session moved on with a shorter one (a prompt that keeps only part of the previous
  context) no longer resumes from its state, so a victim tier's older entry for that session is dropped. With
  `"every-context"`, SSD is written through like DRAM. Writes are free in the target; the engine issues them and
  goes on (workers order them before any later reuse of the blocks, see the README proposal).
- **Reads.** One channel per tier: reads queue one after another at the tier's bandwidth; the two tiers read in
  parallel. A read moves the tokens HBM lacks plus the KDA state. A read that would wait more than `max_read_wait_ms`
  for its channel is not started (the scheduler passes the round over). A read from SSD also puts the session back
  in DRAM (copy-up).
"""

from collections import OrderedDict
from dataclasses import dataclass, field

from ppimpl.config import TierConfig
from ppimpl.engine.kv import Allocation, prefix_content, resumable_tokens

NS = 1_000_000_000


@dataclass
class TierOp:
    """One byte move for the workers. `kv_blocks`/`state_blocks` are HBM block ids (reads: the held blocks to fill;
    writes: the kept blocks to copy from). Tokens [token_start, token_end) of KV plus the KDA state move."""
    op_id: int
    op: str                    # "write" HBM -> tier, "read" tier -> HBM, "demote" DRAM -> SSD, "promote" SSD -> DRAM,
                               # "drop" forget the tier's entry
    tier: str                  # destination tier for write/demote/promote, source tier for read, the tier for drop
    entry: int                 # the session's entry id (same on every worker)
    token_start: int = 0
    token_end: int = 0
    kv_blocks: list[int] = field(default_factory=list)
    state_blocks: list[int] = field(default_factory=list)
    req_id: str | None = None  # read: the request it is for
    land_ns: int | None = None  # read: when the engine expects it to land


@dataclass
class TierEntry:
    tokens: int
    content: object | None             # the context's token ids (kv.prefix_content), None in replay


class TierStore:
    """One tier's LRU of session contexts, in one GPU's bytes."""

    def __init__(self, name: str, capacity_bytes: float, read_bytes_per_s: float, bytes_per_token: int,
                 state_bytes: int, victim: bool):
        self.name = name
        self.capacity = capacity_bytes
        self.read_bps = read_bytes_per_s
        self.bpt = bytes_per_token
        self.state_bytes = state_bytes
        self.victim = victim                 # written only with what the tier above evicts
        self.entries: OrderedDict[object, TierEntry] = OrderedDict()
        self.used = 0.0
        self.channel_free_ns = 0
        self.stats = dict(stored=0, written_bytes=0, evicted=0, evicted_tokens=0, reads=0, read_tokens=0,
                          read_bytes=0, dropped=0)

    def charge(self, tokens: int) -> float:
        return tokens * self.bpt + self.state_bytes

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
        Returns (stored, what left the tier: LRU victims, or the context itself when it exceeds the tier)."""
        self.remove(key)
        c = self.charge(tokens)
        if c > self.capacity:
            return False, [(key, TierEntry(tokens, content))]
        left = []
        while self.used + c > self.capacity:
            victim, e = self.entries.popitem(last=False)
            self.used -= self.charge(e.tokens)
            self.stats["evicted"] += 1
            self.stats["evicted_tokens"] += e.tokens
            left.append((victim, e))
        self.entries[key] = TierEntry(tokens, content)
        self.used += c
        self.stats["stored"] += 1
        self.stats["written_bytes"] += (tokens - keeps) * self.bpt + self.state_bytes
        return True, left


class Tiers:
    """The DRAM/SSD index of one pipeline, fastest first, and the reads in flight."""

    def __init__(self, cfg: TierConfig, partial_hits: bool = False):
        self.cfg = cfg
        self.partial_hits = partial_hits
        self.tiers: list[TierStore] = []
        if cfg.dram_gb:
            self.tiers.append(TierStore("dram", cfg.dram_gb * 1e9, cfg.dram_read_gb_s * 1e9, cfg.bytes_per_token,
                                        cfg.state_bytes, victim=False))
        if cfg.ssd_gb:
            self.tiers.append(TierStore("ssd", cfg.ssd_gb * 1e9, cfg.ssd_read_gb_s * 1e9, cfg.bytes_per_token,
                                        cfg.state_bytes, victim=bool(self.tiers) and cfg.ssd_writes == "dram-evictions"))
        assert cfg.ssd_writes in ("dram-evictions", "every-context")
        self.max_wait_ns = None if cfg.max_read_wait_ms is None else int(cfg.max_read_wait_ms * 1e6)
        self.ops: list[TierOp] = []
        self._entry_ids: dict[object, int] = {}
        self._next_op = 0

    def take_ops(self) -> list[TierOp]:
        ops, self.ops = self.ops, []
        return ops

    def _entry(self, key) -> int:
        return self._entry_ids.setdefault(key, len(self._entry_ids))

    def _emit(self, op: str, tier: str, key, **kw) -> TierOp:
        o = TierOp(self._next_op, op, tier, self._entry(key), **kw)
        self._next_op += 1
        self.ops.append(o)
        return o

    # --- lookups ------------------------------------------------------------------------------------------------

    def lookup(self, req) -> tuple[int, int] | None:
        """(tier index, tokens): the fastest tier holding the longest resumable context of the request's session."""
        best = None
        for i, t in enumerate(self.tiers):
            e = t.entries.get(req.session)
            if e is not None:
                n = resumable_tokens(e.tokens, e.content, req, self.partial_hits)
                if n > 0 and (best is None or n > best[1]):
                    best = (i, n)
        return best

    def queue_wait_ns(self, tier: int, now_ns: int) -> int:
        return max(0, self.tiers[tier].channel_free_ns - now_ns)

    def read_ns(self, tier: int, tokens: int) -> int:
        t = self.tiers[tier]
        return round(t.charge(tokens) / t.read_bps * NS)

    def passed_over(self, tier: int, now_ns: int) -> bool:
        return self.max_wait_ns is not None and self.queue_wait_ns(tier, now_ns) > self.max_wait_ns

    # --- reads --------------------------------------------------------------------------------------------------

    def read(self, req, tier: int, tokens: int, hbm_tokens: int, alloc: Allocation, now_ns: int) -> TierOp:
        """Queue a read of the session's first `tokens` into the request's held blocks; HBM already has the first
        `hbm_tokens`. Copies an SSD context up into DRAM."""
        t = self.tiers[tier]
        start = max(t.channel_free_ns, now_ns)
        land = start + self.read_ns(tier, tokens - hbm_tokens)
        t.channel_free_ns = land
        t.stats["reads"] += 1
        t.stats["read_tokens"] += tokens - hbm_tokens
        t.stats["read_bytes"] += t.charge(tokens - hbm_tokens)
        t.touch(req.session)
        op = self._emit("read", t.name, req.session, token_start=hbm_tokens, token_end=tokens,
                        kv_blocks=list(alloc.kv_blocks), state_blocks=list(alloc.state_blocks), req_id=req.req_id,
                        land_ns=land)
        content = t.entries[req.session].content
        for faster in range(tier):
            self._store_in(faster, req.session, tokens, content, "promote", alloc)
        return op

    # --- writes -------------------------------------------------------------------------------------------------

    def store(self, req, alloc: Allocation):
        """A round ended: write its context to the fastest tier and every write-through tier; a victim tier's older
        entry stays valid only if the new context starts with it."""
        tokens = req.num_tokens
        content = prefix_content(req.token_ids, tokens) if req.token_ids is not None else None
        for t in self.tiers[1:]:
            e = t.entries.get(req.session)
            if t.victim and e is not None and resumable_tokens(e.tokens, e.content, req, self.partial_hits) != e.tokens:
                t.remove(req.session)
                t.stats["dropped"] += 1
                self._emit("drop", t.name, req.session)
        for i, t in enumerate(self.tiers):
            if i == 0 or not t.victim:
                self._store_in(i, req.session, tokens, content, "write", alloc, req)

    def _keeps(self, old: TierEntry | None, tokens: int, req) -> int:
        """Tokens of the tier's old entry for the session that the new context still starts with (not rewritten).
        A tier's entries for a session are always prefixes of its latest context (a victim tier drops one that is
        not, `store`), so for copy-up and demotion that is the old entry whenever it is not longer."""
        if old is None:
            return 0
        if req is not None:
            return old.tokens if resumable_tokens(old.tokens, old.content, req, self.partial_hits) == old.tokens else 0
        return old.tokens if old.tokens <= tokens else 0

    def _store_in(self, i: int, key, tokens: int, content, op: str, alloc: Allocation | None = None, req=None):
        """Store into tier i (op "write" from HBM, "promote" SSD -> DRAM, "demote" DRAM -> SSD). What the tier
        evicts for it goes down to the next tier if that one is a victim tier, then is dropped here (the worker
        copies before it frees)."""
        t = self.tiers[i]
        old = t.entries.get(key)
        keeps = self._keeps(old, tokens, req)
        stored, left = t.store(key, tokens, content, keeps)
        blocks = dict(kv_blocks=list(alloc.kv_blocks), state_blocks=list(alloc.state_blocks)) if alloc else {}
        if stored:
            self._emit(op, t.name, key, token_start=keeps, token_end=tokens, **blocks)
        elif old is not None:
            self._emit("drop", t.name, key)
        nxt = i + 1 if i + 1 < len(self.tiers) and self.tiers[i + 1].victim else None
        for victim, e in left:
            if victim == key:                       # larger than this whole tier: straight to the victim tier
                if nxt is not None and op == "write":
                    self._store_in(nxt, key, tokens, content, "write", alloc, req)
                continue
            if nxt is not None:
                held = self.tiers[nxt].entries.get(victim)
                if held is not None and held.tokens >= e.tokens:
                    self.tiers[nxt].touch(victim)
                else:
                    self._store_in(nxt, victim, e.tokens, e.content, "demote")
            self._emit("drop", t.name, victim)

    # --- reporting ----------------------------------------------------------------------------------------------

    def summary(self) -> list[dict]:
        return [dict(tier=t.name, capacity_gb=t.capacity / 1e9, used_gb=t.used / 1e9, entries=len(t.entries),
                     **t.stats) for t in self.tiers]
