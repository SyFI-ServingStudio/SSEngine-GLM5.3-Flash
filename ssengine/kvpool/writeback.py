"""Write-back of one cache level to the level below it (lanes/m6 TIERS.md §5): the victim chain HBM -> DRAM -> SSD.

The same class serves both levels: HBM's kept contexts written to the fastest tier (`BalancedKVPool.HbmLevel`, over
the KV manager's LRU of kept sessions) and DRAM's entries demoted to SSD (`manager.TierManager`, over DRAM's LRU).
A level keeps whole entries in LRU order, each with a charge, and holds `fixed()` charge outside its entries (HBM:
the running requests). An entry is **dirty** until the level below holds it whole, **writing** while a write is in
flight, then **clean**; only a clean entry may be evicted without waiting.

- **Write-ahead** (`step`): above `watermark` x capacity of fixed + dirty charge, the least recently used dirty
  entries are written ahead, until fixed + dirty is back under it. LRU evicts the tail first, so the tail is clean
  when eviction reaches it: an eviction rarely waits for a write.
- **Reserve** (staged loading, `reserve` set): instead of the watermark, the least recently used entries holding
  `reserve` charge are kept written (clean or being written): fixed + dirty <= capacity - reserve. The reserve is what
  the level must free in one batch period (the scheduler sets it from the loading area and the microbatch budget),
  so admission evicts clean entries and does not wait for a write.
- **Demand** (`make_room`): a level that must evict `short` more charge than its clean entries hold starts the
  writes of its least recently used dirty entries (`evict_writes`) and counts a wait (`evict_waits`). Those writes,
  and the level's writes already in flight, are urgent (`Level.urge`): they do not yield to reads (kvpool/member.py). HBM's caller
  then waits (the request is admitted later); DRAM evicts at once and the op that reuses its slots waits for the
  demote (`SlotGuard`), counted by `evicting`.
- The level below reports a write done (`done`) or that it no longer holds an entry (`lost`: dirty again).

Slot mode only (the server): in byte mode (replay) writes take no time, so a level writes at eviction instead.
"""

from __future__ import annotations

from typing import Iterable, Protocol


class Level(Protocol):
    capacity: float

    def fixed(self) -> float:
        """Charge held outside the entries (not evictable at all)."""

    def lru(self) -> Iterable:
        """The entries' keys, least recently used first."""

    def write(self, key, entry=None) -> object | None:
        """Start writing `key`'s entry (or `entry`, one the level just let go of) to the level below: a token that
        `done` will be called with, or None if nothing had to move."""

    def urge(self, token):
        """An eviction waits for the write `token`: it must not yield to reads (optional)."""


class WriteBack:
    def __init__(self, name: str, level: Level, watermark: float | None):
        self.name, self.level, self.watermark = name, level, watermark
        self.reserve: float | None = None           # staged loading: replaces the watermark (module docs)
        self.dirty: dict[object, float] = {}        # key -> charge, not held below
        self.writing: dict[object, tuple] = {}      # key -> (token, charge), write in flight
        self.dirty_charge = 0.0
        self.stats = dict(ahead_writes=0, evict_writes=0, evict_waits=0)

    # --- the level's changes ----------------------------------------------------------------------------------------

    def add(self, key, charge: float):
        """An entry appeared or changed: dirty until written again."""
        self.dirty_charge += charge - self.dirty.get(key, 0.0)
        self.dirty[key] = charge

    def remove(self, key):
        """The entry left the level (evicted, dropped, taken over); a write in flight still lands (`done` ignores
        it)."""
        self.dirty_charge -= self.dirty.pop(key, 0.0)
        self.writing.pop(key, None)

    def done(self, key, token):
        """The write `token` of `key` landed: clean, unless the entry changed since (then it is dirty)."""
        w = self.writing.get(key)
        if w is not None and w[0] is token:
            del self.writing[key]

    def lost(self, key, charge: float):
        """The level below no longer holds `key`'s entry whole: dirty again (if it is still here)."""
        if key not in self.dirty and key not in self.writing:
            self.add(key, charge)

    # --- queries --------------------------------------------------------------------------------------------------

    def evictable(self, key) -> bool:
        return key not in self.dirty and key not in self.writing

    def held_back(self) -> float:
        """Charge of the entries that cannot be evicted now (dirty or being written)."""
        return self.dirty_charge + sum(c for k, (_, c) in self.writing.items() if k not in self.dirty)

    # --- writes ---------------------------------------------------------------------------------------------------

    def _write(self, key, entry=None, urgent: bool = False) -> bool:
        charge = self.dirty.pop(key)
        self.dirty_charge -= charge
        token = self.level.write(key, entry)
        if token is not None:
            self.writing[key] = (token, charge)
            if urgent:
                self._urge(token)
        return token is not None

    def _urge(self, token):
        if (urge := getattr(self.level, "urge", None)) is not None:
            urge(token)

    def _write_lru(self, over: float, kind: str):
        """Write the least recently used dirty entries until `over` charge of them is written or writing."""
        for key in [k for k in self.level.lru() if k in self.dirty]:
            if over <= 0:
                break
            over -= self.dirty[key]
            if self._write(key, urgent=kind == "evict_writes"):
                self.stats[kind] += 1

    def step(self):
        """Write ahead above the watermark."""
        if self.reserve is not None:
            limit = self.level.capacity - self.reserve
        elif self.watermark is None:
            return
        else:
            limit = self.watermark * self.level.capacity
        over = self.level.fixed() + self.dirty_charge - limit
        if over > 0:
            self._write_lru(over, "ahead_writes")

    def make_room(self, short: float):
        """An eviction lacks `short` charge of clean entries: start the writes that make it up; the caller waits."""
        self.stats["evict_waits"] += 1
        for token, _ in self.writing.values():
            self._urge(token)
        self._write_lru(short - sum(c for k, (_, c) in self.writing.items() if k not in self.dirty), "evict_writes")

    def evicting(self, key, entry) -> str:
        """The level evicts `key` now (DRAM: its slots' reuse waits for a write in flight): "clean" (nothing to do),
        "writing" (a wait), or "dirty" (its write starts now from `entry`, and the reuse waits for it)."""
        if key in self.writing and key not in self.dirty:
            self.stats["evict_waits"] += 1
            self.remove(key)
            return "writing"
        if key not in self.dirty:
            return "clean"
        self.stats["evict_waits"] += 1
        self.stats["evict_writes"] += 1
        self._write(key, entry)
        self.remove(key)
        return "dirty"
