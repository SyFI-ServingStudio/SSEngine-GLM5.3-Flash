"""Write-back of one cache level to the level below it (lanes/m6 TIERS.md §5): the victim chain HBM -> DRAM -> SSD.

The same class serves both levels: HBM's kept contexts written to the fastest tier (`BalancedKVPool.HbmLevel`, over
the KV manager's LRU of kept sessions) and DRAM's entries demoted to SSD (`manager.TierManager`, over DRAM's LRU).
A level keeps whole entries in LRU order, each with a charge, and holds `fixed()` charge outside its entries (HBM:
the running requests). An entry is **dirty** until the level below holds it whole, **writing** while a write is in
flight, then **clean**; only a clean entry may be evicted without waiting. Free plus clean charge is what the level
can give to new demand at once: C - F - D - W (capacity, fixed, dirty, writing).

- **Ahead of need** (`ahead=True`, HBM; `step`, every scheduler step). Invariant: demand never waits for a write. A
  write lands within T after it starts, T the longest write seen (from its start, after the demotes of the DRAM
  slots it reuses, to its landing); the writes still in flight then answer demand of the last T, so free + clean
  covers the next T's demand if C - F - D >= R, the largest demand seen in any window of 2T. Demand is the growth of
  F + D + W from one step to the next (admissions, contexts the level below lost; writes landing are not counted as
  supply). After each step the least recently used dirty entries are written until C - F - D >= R. T and R are
  running maxima of what the run measured: no threshold to tune. T leaves out the wait for a demote: with it, a
  burst of write-backs queued behind demotes raised T, R and the next burst (`bench.replay writeback`: DRAM writes
  +60% in ssd_heavy); a write that does wait for a demote is the rare residual wait.
- **At eviction only** (`ahead=False`, DRAM): SSD is written only with what DRAM evicts (`evicting`).
- **Demand** (`make_room`): a level that must evict `short` more charge than its clean entries hold starts the
  writes of its least recently used dirty entries beyond those in flight (`evict_writes`) and counts a wait
  (`evict_waits`). HBM's caller then waits (the request is admitted later; a tier read starts as soon as a write
  lands, `Scheduler.op_done`); DRAM evicts at once and the op that reuses its slots waits for the demote
  (`SlotGuard`), counted by `evicting`.
- The level below reports a write done (`done`) or that it no longer holds an entry (`lost`: dirty again).

Slot mode only (the server): in byte mode (replay) writes take no time, so a level writes at eviction instead.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Iterable, Protocol


class Level(Protocol):
    capacity: float

    def fixed(self) -> float:
        """Charge held outside the entries (not evictable at all)."""

    def lru(self) -> Iterable:
        """The entries' keys, least recently used first."""

    def write(self, key, entry=None) -> object | None:
        """Start writing `key`'s entry (or `entry`, one the level just let go of) to the level below: a token that
        `done` will be called with, or None if nothing had to move."""


class WriteBack:
    def __init__(self, name: str, level: Level, ahead: bool, clock: Callable[[], int] = time.monotonic_ns):
        self.name, self.level, self.ahead, self.clock = name, level, ahead, clock
        self.dirty: dict[object, float] = {}        # key -> charge, not held below
        self.writing: dict[object, tuple] = {}      # key -> (token, charge, issued ns), write in flight
        self.dirty_charge = 0.0
        self.write_ns = 0                           # T: the longest write, start to landing
        self.headroom = 0.0                         # R: the largest demand in any window of 2T
        self._held = 0.0                            # F + D + W after the last step, less what landed since
        self._demand: deque[tuple[int, float]] = deque()   # (step ns, demand) within the last 2T
        self._window = 0.0
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

    def done(self, key, token, started_ns: int | None = None):
        """The write `token` of `key` landed: clean, unless the entry changed since (then it is dirty). `started_ns`:
        when the write itself started (HBM: sent, once the demotes of the DRAM slots it reuses landed); default its
        issue."""
        w = self.writing.get(key)
        if w is not None and w[0] is token:
            del self.writing[key]
            self.write_ns = max(self.write_ns, self.clock() - (w[2] if started_ns is None else started_ns))
            if key not in self.dirty:
                self._held -= w[1]

    def lost(self, key, charge: float):
        """The level below no longer holds `key`'s entry whole: dirty again (if it is still here)."""
        if key not in self.dirty and key not in self.writing:
            self.add(key, charge)

    # --- queries --------------------------------------------------------------------------------------------------

    def evictable(self, key) -> bool:
        return key not in self.dirty and key not in self.writing

    def in_flight(self) -> float:
        """Charge of the entries being written (and not dirty again)."""
        return sum(w[1] for k, w in self.writing.items() if k not in self.dirty)

    def held_back(self) -> float:
        """Charge of the entries that cannot be evicted now (dirty or being written)."""
        return self.dirty_charge + self.in_flight()

    def summary(self) -> dict:
        return dict(self.stats, headroom=self.headroom, write_ms=round(self.write_ns / 1e6, 3))

    # --- writes ---------------------------------------------------------------------------------------------------

    def _write(self, key, entry=None) -> bool:
        charge = self.dirty.pop(key)
        self.dirty_charge -= charge
        token = self.level.write(key, entry)
        if token is not None:
            self.writing[key] = (token, charge, self.clock())
        else:
            self._held -= charge                    # nothing moved: clean at once, as a landed write
        return token is not None

    def _write_lru(self, over: float, kind: str):
        """Write the least recently used dirty entries until `over` charge of them is written or writing."""
        for key in [k for k in self.level.lru() if k in self.dirty]:
            if over <= 0:
                break
            over -= self.dirty[key]
            if self._write(key):
                self.stats[kind] += 1

    def step(self):
        """Measure this step's demand and write ahead until C - F - D >= R (`ahead` only)."""
        if not self.ahead:
            return
        now = self.clock()
        fixed = self.level.fixed()
        held = fixed + self.held_back()
        demand, self._held = held - self._held, held
        if demand > 0:
            self._demand.append((now, demand))
            self._window += demand
        while self._demand and self._demand[0][0] < now - 2 * self.write_ns:
            self._window -= self._demand.popleft()[1]
        self.headroom = max(self.headroom, self._window)
        over = fixed + self.dirty_charge - (self.level.capacity - self.headroom)
        if over > 0:
            self._write_lru(over, "ahead_writes")

    def make_room(self, short: float):
        """An eviction lacks `short` charge of clean entries: start the writes that make it up beyond those in flight;
        the caller waits."""
        self.stats["evict_waits"] += 1
        self._write_lru(short - self.in_flight(), "evict_writes")

    def evicting(self, key, entry) -> str:
        """The level evicts `key` now (DRAM: its slots' reuse waits for a write in flight): "clean" (nothing to do),
        "writing" (a wait), or "dirty" (its write starts now from `entry`, and the reuse waits for it)."""
        if key in self.writing and key not in self.dirty:
            self.stats["evict_waits"] += 1
            self.remove(key)
            return "writing"
        if key not in self.dirty:
            return "clean"
        moved = self._write(key, entry)
        self.remove(key)
        if not moved:                               # the level below held it whole already
            return "clean"
        self.stats["evict_waits"] += 1
        self.stats["evict_writes"] += 1
        return "dirty"
