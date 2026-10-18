"""The KV pool: prefix KV and KDA states of n pipeline stages in DRAM and SSD behind HBM (lanes/m6 TIERS.md §13).

`BalancedKVPool` is the leader's object (here the controller's, where the scheduler runs). The scheduler sees only
`lookup` (a `Hit` with `wait_ns`), `load` -> `Handle` (with `eta_ns`) and a load's completion. Writes are the victim
chain HBM -> DRAM -> SSD (TIERS.md §5): the HBM KV manager reaches the pool through `hbm_level(kv)`, an `HbmLevel`
it asks whether a kept context may be evicted and tells about kept, taken-over and evicted contexts; the same
`writeback.WriteBack` policy writes HBM's LRU tail to the fastest tier and DRAM's to SSD. Shares, slots, tiers' LRU
and the ring stay inside. The controller posts every op to all n members (`ready_ops`, which also writes ahead,
`member.tier_move`) and reports each one done when all n members did (`complete`). The read-wait bound is the
scheduler's (from `Hit.wait_ns`), not the pool's.

Layers, each with its own file: 1 `layout` (which bytes a block or a state owns), 2 `placement` (GPU q's share of
every range and where it sits in a host slot), 3 `backends` (host arena, SSD file), 4 `mover` (copy engines, two hops
through the own-HBM ring), 5 `manager` (the leader's policy: LRU, slots, op order). `member.PoolMember` is each
stage process's data plane over layers 1-4.

Byte mode (`TierConfig.slot_bytes` None, replay): no members; a write is done at once (a level writes what it
evicts at the eviction, as the simulator) and a load lands at its `eta_ns` on the caller's clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ppimpl.config import TierConfig
from ppimpl.engine.kv import blocks_for
from ppimpl.kvpool.manager import Hit, KVViews, TierManager, TierOp
from ppimpl.kvpool.writeback import WriteBack

__all__ = ["BalancedKVPool", "Handle", "Hit", "KVViews"]


@dataclass(eq=False)
class Handle:
    """One `load` or `store`: pending until every member reported each of its byte moves (a store's writes from HBM,
    a load's read); `t_land` is then the leader's clock."""
    kind: str                       # "load" | "store"
    key: object
    tokens: int
    tier: str | None = None         # load: the tier read from
    req_id: str | None = None       # load: the request it is for
    eta_ns: int | None = None       # load: expected landing time from the tier's read channel
    t_start: int | None = None      # load: when it was posted
    t_land: int | None = None
    status: str = "pending"         # "pending" | "done"
    ops: set[int] = field(default_factory=set)

    def done(self) -> bool:
        return self.status == "done"


class BalancedKVPool:
    def __init__(self, cfg: TierConfig, partial_hits: bool = False):
        self.cfg = cfg
        self.manager = TierManager(cfg, partial_hits)
        self._of: dict[int, Handle] = {}            # slot mode: op id -> the handle waiting for it
        self.hbm: HbmLevel | None = None

    def lookup(self, key, prompt, now_ns: int = 0) -> Hit | None:
        """The fastest tier holding the longest context of `key` that `prompt` (a `scheduler.Request`) can resume
        from. Advisory: `load` checks again."""
        return self.manager.lookup(key, prompt, now_ns)

    def load(self, key, hit: Hit, views: KVViews | list[KVViews], now_ns: int, req_id: str | None = None
             ) -> Handle | None:
        """Read `hit` into `views` (one KVViews for every stage, or one per stage); None if no tier holds the key any
        more (the caller computes the prefix)."""
        got = self.manager.prepare_load(key, hit, views, now_ns, req_id)
        if got is None:
            return None
        ops, eta = got
        h = Handle("load", key, hit.tokens, hit.tier, req_id, eta, now_ns)
        self._track(h, [o for o in ops if o.op == "read"])
        return h

    # --- staged loading -------------------------------------------------------------------------------------------

    def set_loading_area(self, kv, mb_tokens: int) -> int:
        """Stage loads (engine/scheduler.py): HBM's loading area in `kv` (`KvManager.area`), `TierConfig.loading_area`
        of its blocks (rounded down, as the simulator), the same fraction of DRAM with an SSD behind it, and each
        level's write-ahead reserve: what it frees in one batch period, the area plus one microbatch's footprint.
        Returns the area in blocks (0: off)."""
        fraction = self.cfg.loading_area
        if not fraction:
            return 0
        if not self.manager.slot_mode:
            raise ValueError("staged loading needs slot mode")
        if not 0 < fraction < 0.5:
            raise ValueError(f"tiers.loading_area {fraction}: must be in (0, 0.5)")
        area = int(kv.capacity * fraction)
        kv.area = area
        self.manager.set_dram_area(fraction)
        reserve = area + blocks_for(mb_tokens)
        if self.hbm is not None and self.hbm.wb is not None:
            self.hbm.wb.reserve = reserve
        for wb in self.manager.wb.values():
            wb.reserve = reserve * self.cfg.slot_bytes
        return area

    def dram_area_fits(self, tokens: int) -> bool | None:
        """Whether an SSD read of `tokens` may start into DRAM's loading area now; None without one (the SSD read
        goes to HBM's area directly)."""
        d = self.manager.dram_area
        n = self.manager.fetch_slots(tokens)
        if d is None or n > d.num_slots:
            return None
        return d.area_fits(n)

    def fetch(self, key, hit: Hit, now_ns: int, req_id: str | None = None) -> tuple[Handle, list[int]] | None:
        """SSD -> DRAM's loading area (`dram_area_fits` said yes): (the handle, the area slots), or None if the SSD
        no longer holds the key."""
        got = self.manager.prepare_fetch(key, hit, now_ns, req_id)
        if got is None:
            return None
        ops, eta, area = got
        h = Handle("load", key, hit.tokens, "ssd", req_id, eta, now_ns)
        self._track(h, [o for o in ops if o.op == "fetch"])
        return h, area

    def hop(self, key, tokens: int, area: list[int], views: KVViews, now_ns: int, req_id: str | None = None
            ) -> Handle:
        """DRAM's loading area -> HBM's: the second hop of a staged SSD read."""
        ops, eta = self.manager.prepare_hop(key, tokens, area, views, now_ns, req_id)
        h = Handle("load", key, tokens, "dram", req_id, eta, now_ns)
        self._track(h, [o for o in ops if o.op == "read"])
        return h

    def release_dram_area(self, area: list[int]):
        self.manager.dram_area.area_give(area)

    def hbm_level(self, kv) -> "HbmLevel":
        """The tiers as HBM's lower level, for the KV manager `kv` (`KvManager.lower`): HBM's kept contexts are written
        here before HBM evicts them."""
        self.hbm = HbmLevel(self, kv)
        return self.hbm

    def write_back(self, key, tokens: int, content, views: KVViews) -> Handle:
        """Write HBM's kept context of `key` (`tokens` long, in `views`) to the fastest tier, what its entry lacks.
        The caller keeps the views unchanged until the handle is done (at once if nothing moves, and in byte
        mode)."""
        h = Handle("store", key, tokens)
        self._track(h, [o for o in self.manager.prepare_write_back(key, tokens, content, views) if o.op == "write"])
        if not h.ops:
            h.status = "done"
        return h

    def _track(self, h: Handle, ops: list[TierOp]):
        if self.manager.slot_mode:                  # byte mode: nothing reports ops done
            h.ops = {o.op_id for o in ops}
            for o in ops:
                self._of[o.op_id] = h

    # --- the leader's executor (the controller) -----------------------------------------------------------------

    def ready_ops(self) -> list[TierOp]:
        """Write ahead where a level is over its watermark (HBM -> DRAM, DRAM -> SSD), then the ops whose
        dependencies are done, to post to all n members now."""
        m = self.manager
        if self.hbm is not None and self.hbm.wb is not None:
            for key in m.take_gone():
                self.hbm.lost(key)
            self.hbm.wb.step()
        m.step()
        return m.ready_ops()

    def complete(self, op_id: int, now_ns: int) -> tuple[TierOp, Handle | None]:
        """Every member reported the op done: (the op, the handle that is done with it, if any)."""
        o = self.manager.complete(op_id, now_ns)
        h = self._of.pop(op_id, None)
        if h is not None:
            h.ops.discard(op_id)
            if h.ops:
                h = None
            else:
                h.status, h.t_land = "done", now_ns
                if h.kind == "store" and self.hbm is not None:
                    self.hbm.landed(h)
        return o, h

    def summary(self) -> list[dict]:
        return self.manager.summary()

    def write_back_stats(self) -> dict:
        """Per level: writes ahead of need, writes at eviction, evictions that waited for a write (TIERS.md §5)."""
        out = {} if self.hbm is None or self.hbm.wb is None else {"hbm": dict(self.hbm.wb.stats)}
        return out | {self.manager.tiers[i].name: dict(wb.stats) for i, wb in self.manager.wb.items()}


class HbmLevel:
    """The tiers below HBM as the KV manager sees them (`KvManager.lower`), and HBM as the upper level of a
    `WriteBack` to the fastest tier (slot mode; the same policy as DRAM -> SSD, `TierConfig.hbm_write_ahead`).

    The KV manager calls `kept`, `unkept`, `evicted` and `admitted` as its kept contexts change, evicts only what
    `evictable` allows, counts `held_back` as not evictable, and calls `make_room` when a request fits only by evicting
    contexts not yet written. A kept context being written is pinned in the KV manager (its next round waits to take
    it over, `KvManager.busy`). Byte mode (replay): writes take no time, so every kept context is evictable and is
    written when evicted."""

    def __init__(self, pool: BalancedKVPool, kv):
        self.pool, self.kv = pool, kv
        self.wb = WriteBack("hbm", self, pool.cfg.hbm_write_ahead) if pool.manager.slot_mode else None
        self._writes: dict[Handle, object] = {}

    # --- WriteBack's level --------------------------------------------------------------------------------------

    @property
    def capacity(self) -> float:
        return self.kv.capacity

    def fixed(self) -> float:
        return self.kv.hard + self.kv.area_charge()     # running and held requests, the loading area

    def lru(self):
        return iter(self.kv.kept)

    def urge(self, h: Handle):
        """An eviction waits for the write `h`: its ops not yet sent go urgent (they do not yield to reads)."""
        for op_id in h.ops or ():
            if (o := self.pool.manager.waiting.get(op_id)) is not None:
                o.urgent = True

    def write(self, key, entry=None):
        k = self.kv.kept[key]
        h = self.pool.write_back(key, k.tokens, k.content, KVViews(k.alloc.kv_blocks, k.alloc.state_blocks, k.tokens))
        if h.done():
            return None
        self._writes[h] = key
        self.kv.pin(key)
        return h

    # --- the KV manager's calls ---------------------------------------------------------------------------------

    def kept(self, key):
        if self.wb is not None:
            self.wb.add(key, self.kv.kept_cost(self.kv.kept[key].tokens))

    def unkept(self, key):
        if self.wb is not None:
            self.wb.remove(key)

    def evictable(self, key) -> bool:
        return self.wb is None or self.wb.evictable(key)

    def held_back(self) -> float:
        return 0 if self.wb is None else self.wb.held_back()

    def make_room(self, short: float):
        if self.wb is not None:
            self.wb.make_room(short)

    def evicted(self, key, k):
        self.unkept(key)
        self.pool.manager.hbm_evicted(key, k.tokens, k.content)

    def admitted(self, key, req, hit: int = 0):
        """`req` was admitted, `hit` tokens of it taken over from HBM's kept context."""
        self.pool.manager.drop_stale(key, req)
        self.pool.manager.taken_over(key, hit)

    # --- the pool's -----------------------------------------------------------------------------------------------

    def landed(self, h: Handle):
        key = self._writes.pop(h)
        self.kv.unpin(key)
        self.wb.done(key, h)

    def lost(self, key):
        """The fastest tier no longer holds `key`: HBM's kept copy, if any, is dirty again."""
        k = self.kv.kept.get(key)
        if k is not None:
            self.wb.lost(key, self.kv.kept_cost(k.tokens))
