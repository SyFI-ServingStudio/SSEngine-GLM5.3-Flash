"""The KV pool: prefix KV and KDA states of n pipeline stages in DRAM and SSD behind HBM (lanes/m6 TIERS.md §13).

`BalancedKVPool` is the leader's object (here the controller's, where the scheduler runs): the scheduler sees only
`lookup` (a `Hit` with `wait_ns`), `load` -> `Handle` (with `eta_ns`), `store` -> `Handle`, and a handle's
completion. Shares, slots, tiers' LRU and the ring stay inside. The controller posts every op to all n members
(`ready_ops`, `member.tier_move`) and reports each one done when all n members did (`complete`). The read-wait bound
is the scheduler's (from `Hit.wait_ns`), not the pool's.

Layers, each with its own file: 1 `layout` (which bytes a block or a state owns), 2 `placement` (GPU q's share of
every range and where it sits in a host slot), 3 `backends` (host arena, SSD file), 4 `mover` (copy engines, two hops
through the own-HBM ring), 5 `manager` (the leader's policy: LRU, slots, op order). `member.PoolMember` is each
stage process's data plane over layers 1-4.

Byte mode (`TierConfig.slot_bytes` None, replay): no members; a store is done at once and a load lands at its
`eta_ns` on the caller's clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ppimpl.config import TierConfig
from ppimpl.kvpool.manager import Hit, KVViews, TierManager, TierOp

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

    def store(self, key, prompt, views: KVViews | list[KVViews] | None) -> Handle:
        """Write the context of `prompt`'s round, held in `views`, under `key`. The caller keeps the views unchanged
        until the handle is done (at once without a write from HBM, and in byte mode)."""
        ops = self.manager.prepare_store(key, prompt, views)
        h = Handle("store", key, prompt.num_tokens)
        self._track(h, [o for o in ops if o.op == "write"])
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
        """Ops whose dependencies are done, to post to all n members now."""
        return self.manager.ready_ops()

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
        return o, h

    def summary(self) -> list[dict]:
        return self.manager.summary()
