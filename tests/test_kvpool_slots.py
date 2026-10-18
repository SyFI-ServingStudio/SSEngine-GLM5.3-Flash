"""Slot-mode index of the pool's leader (ppimpl/kvpool/manager.py; the server, lanes/m6 TIERS.md §3, §5-§7): slots per
block, op order by dependencies, write before evict in HBM, SSD eviction at a small SSD, SSD reads through DRAM or
staging. A byte model executes the ops in any order `ready_ops` allows and checks that every op reads what the
context holds."""

import random

import pytest

from ppimpl.config import BLOCK_TOKENS as B, SchedulerConfig, TierConfig
from ppimpl.engine.kv import KvManager
from ppimpl.engine.scheduler import Request, Scheduler
from ppimpl.kvpool import BalancedKVPool, Hit, KVViews
from ppimpl.kvpool.manager import TierManager

SLOT = 1_000_000          # 1 MB slots: dram_gb=0.01 is 10 slots


def cfg(dram=0.010, ssd=0.020, state_slots=1, staging=0.010, **kw):
    return TierConfig(dram_gb=dram, ssd_gb=ssd, slot_bytes=SLOT, state_slots=state_slots, staging_gb=staging,
                      dram_read_gb_s=10.0, ssd_read_gb_s=1.0, **kw)


_n = 0


def rnd(session, prefix, new):
    global _n
    _n += 1
    return Request(f"r{_n}", prefix + new, session=session, prefix_tokens=prefix)


def store(t, session, tokens, prefix=0):
    """A round of `session` (sharing `prefix` tokens with its previous context) admitted, then its context written
    back from HBM."""
    t.drop_stale(session, rnd(session, prefix, tokens - prefix))
    return t.prepare_write_back(session, tokens, None, KVViews([], []))


def read(t, r, tier, tokens, views=None):
    """Load `r`'s session's first `tokens` from `tier`; (its read op, every op emitted)."""
    ops, _ = t.prepare_load(r.session, Hit(tier, tokens, 0), views or KVViews([], [], tokens), 0, r.req_id)
    return next(o for o in ops if o.op == "read"), ops


def hit(t, r):
    h = t.lookup(r.session, r)
    return None if h is None else (h.tier, h.tokens)


def drain(t):
    """Run every op to completion, in dependency order."""
    while True:
        ready = t.ready_ops()
        if not ready:
            assert not t.waiting
            return
        for o in ready:
            t.complete(o.op_id, 0)


def slots_ok(t):
    """Every slot is free or in exactly one entry."""
    for s in t.tiers:
        held = [x for e in s.entries.values() for x in e.slots + e.state]
        assert len(held) == len(set(held)) and not set(held) & set(s.free), s.name
        assert len(held) + len(s.free) == s.num_slots, s.name


def test_write_takes_one_slot_per_block_plus_state():
    t = TierManager(cfg())
    (op,) = store(t, 1, 2 * B)
    assert op.op == "write" and op.tier == "dram" and len(op.slots) == 2 and len(op.state_slots) == 1
    assert t.tiers[0].used == 3 * SLOT and t.tiers[0].stats["written_bytes"] == 3 * SLOT
    slots_ok(t)


def test_growth_keeps_full_blocks_and_rewrites_the_partial_one():
    t = TierManager(cfg())
    store(t, 1, B + B // 2)                                   # blocks 0 (full), 1 (half)
    first = t.tiers[0].entries[1]
    s0, s1, st = first.slots[0], first.slots[1], first.state[0]
    (op,) = store(t, 1, 3 * B, prefix=B + B // 2 + 1)        # extends it: block 0 kept, 1 and 2 new
    e = t.tiers[0].entries[1]
    assert e.slots[0] == s0 and op.block_start == 1 and op.token_start == B + B // 2
    assert op.slots == e.slots[1:] and len(op.slots) == 2
    assert t.tiers[0].stats["written_bytes"] == 3 * SLOT + 3 * SLOT     # 2 blocks + state, then 2 blocks + state
    slots_ok(t)
    assert {s1, st} <= set(t.tiers[0].free) | set(e.slots) | set(e.state)


def test_eviction_demotes_first_and_the_reusing_write_waits_for_it():
    t = TierManager(cfg(dram=0.006))                                # 6 slots: two 2-block contexts (3 slots each)
    store(t, 1, 2 * B)
    store(t, 2, 2 * B)
    drain(t)
    ops = [o for o in store(t, 3, 2 * B) if o.moves_bytes]   # evicts 1: demote to SSD, then 3's write into its slots
    assert [o.op for o in ops] == ["demote", "write"]
    demote, write = ops
    assert demote.host_tier == "dram" and set(demote.host_slots) & set(write.slots)
    assert write.deps == [demote.op_id]                       # write-after-read on the reused slots
    assert [o.op for o in t.ready_ops()] == ["demote"]
    t.complete(demote.op_id, 0)
    assert [o.op for o in t.ready_ops()] == ["write"]
    slots_ok(t)


def test_small_ssd_evicts_lru_and_reuses_slots():
    t = TierManager(cfg(dram=0.003, ssd=0.006))                     # DRAM 1 context, SSD 2 contexts of 2 blocks
    for s in range(1, 5):
        store(t, s, 2 * B)
    assert set(t.tiers[0].entries) == {4} and set(t.tiers[1].entries) == {2, 3}
    assert t.tiers[1].stats["evicted"] == 1                   # session 1 left the SSD
    assert hit(t, rnd(1, 2 * B + 1, 10)) is None              # computed again
    slots_ok(t)


def test_ssd_read_goes_through_staging_and_dram_keeps_no_copy():
    t = TierManager(cfg(dram=0.003, ssd=0.020))
    store(t, 1, 2 * B)
    ops = store(t, 2, 2 * B)                                  # 1 -> SSD
    assert [o.op for o in ops] == ["demote", "drop", "write"]
    drain(t)
    r = rnd(1, 2 * B + 1, 10)
    assert hit(t, r) == ("ssd", 2 * B)
    op, ops = read(t, r, "ssd", 2 * B, KVViews([7, 8, 9], [1, 2, 3, 4, 5, 6], 2 * B))
    assert op.tier == "ssd" and op.host_tier == "staging" and [o.op for o in ops] == ["read"]
    assert set(t.tiers[0].entries) == {2}                     # the victim DRAM takes no copy-up
    assert t.read_ns(1, 2 * B) == round(3 * SLOT / 1e9 * 1e9)  # whole context: 2 blocks + state at 1 GB/s
    slots_ok(t)


def test_ssd_only_writes_and_reads_through_staging():
    t = TierManager(cfg(dram=None, ssd=0.020, staging=0.004))
    store(t, 1, 2 * B)
    (w,) = t.ready_ops()
    assert w.op == "write" and w.tier == "ssd" and w.host_tier == "staging" and w.host_slots == [0, 1]
    assert w.host_state == [2]
    r = rnd(1, 2 * B + 1, 10)
    op, _ = read(t, r, "ssd", 2 * B, KVViews([7, 8, 9], [1, 2, 3, 4, 5, 6], 2 * B))
    assert op.host_tier == "staging" and op.slots == t.tiers[0].entries[1].slots
    assert op.host_slots == [3, 0] and op.deps == [w.op_id]   # the ring wraps onto the write's staging slot


def finish(s, r, now=0):
    s.add(r, now)
    assert s.next_microbatch(now) is not None
    s.stage_done(0, 3)
    s.stage_done(3, 3)
    s.finish(r)


def test_kept_context_pinned_until_its_write_is_done():
    """A kept context being written back is pinned: not evicted, and its next round waits to take it over."""
    kv = KvManager(20, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram=1.0))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8 * B), pool, dispatch_ahead=0)
    finish(s, rnd(1, 0, 6 * B + 100))                         # 13 blocks of demand: R = 13 > 20 - 13, written ahead
    (w,) = pool.ready_ops()
    assert w.op == "write" and kv.pinned[1] == 1
    big = Request("big", 14 * B)                              # needs every block: session 1 is pinned, it waits
    assert not kv.fits(big)
    nxt = rnd(1, 6 * B + 101, 50)                             # session 1's next round may not take over yet
    s.add(nxt, 0)
    assert s.next_microbatch(0) is None and s.stats["busy_waits"] >= 1
    s.op_done(w.op_id, 1)
    assert not kv.pinned and kv.fits(big)
    mb = s.next_microbatch(1)
    assert [(c.req.req_id, c.start) for c in mb] == [(nxt.req_id, 6 * B + 100)]


def test_read_lands_on_op_done():
    kv = KvManager(20, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram=1.0))
    t = pool.manager
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    store(t, 5, 2 * B)
    drain(t)
    r = rnd(5, 2 * B + 1, 99)
    s.add(r, 0)
    assert s.next_microbatch(0) is None
    (op,) = pool.ready_ops()
    assert op.op == "read" and op.req_id == r.req_id and op.block_start == 0
    s.op_done(op.op_id, 50_000_000)
    assert r.read_land_ns == 50_000_000 and t.tiers[0].channel_free_ns >= 50_000_000
    mb = s.next_microbatch(50_000_000)
    assert [(c.req.req_id, c.start) for c in mb] == [(r.req_id, 2 * B)]


# --- byte model: run random sessions, execute ready ops in a random order, check every read --------------------------

def valid(tokens, j):
    return min(tokens, (j + 1) * B) - j * B


class Bytes:
    """What each slot holds: (session, block, valid tokens) for KV, (session, tokens) for a state."""

    def __init__(self):
        self.mem = {}
        self.count = {}

    def check(self, tier, slots, state, key, tokens, b0):
        for i, s in enumerate(slots):
            j = b0 + i
            assert self.mem.get((tier, s)) == (key, j, valid(tokens, j)), (tier, s, key, j)
        for s in state:
            assert self.mem.get((tier, s)) == (key, tokens), (tier, s, key)

    def put(self, tier, slots, state, key, tokens, b0):
        for i, s in enumerate(slots):
            self.mem[tier, s] = (key, b0 + i, valid(tokens, b0 + i))
        for s in state:
            self.mem[tier, s] = (key, tokens)

    def run(self, o, keys):
        self.count[o.op] = self.count.get(o.op, 0) + 1
        key = keys[o.entry]
        if o.op == "write":
            if o.host_tier:
                self.put(o.host_tier, o.host_slots, o.host_state, key, o.token_end, o.block_start)
            self.put(o.tier, o.slots, o.state_slots, key, o.token_end, o.block_start)
        elif o.op == "demote":
            self.check("dram", o.host_slots, o.host_state, key, o.token_end, o.block_start)
            self.put("ssd", o.slots, o.state_slots, key, o.token_end, o.block_start)
        elif o.op == "read":
            self.check(o.tier, o.slots, o.state_slots, key, o.token_end, o.block_start)
            if o.host_tier:
                self.put(o.host_tier, o.host_slots, o.host_state, key, o.token_end, o.block_start)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dram", [0.012, None])
def test_random_sessions_every_read_sees_its_context(seed, dram):
    """HBM writes contexts back (what the entry lacks), each round's admission drops what it does not resume from,
    DRAM demotes at eviction."""
    rng = random.Random(seed)
    t = TierManager(cfg(dram=dram, ssd=0.020, state_slots=2, staging=0.012))
    mem, keys, ctx = Bytes(), {}, {}
    pending = []
    for step in range(300):
        s = rng.randrange(8)
        if s in ctx and rng.random() < 0.5:
            r = rnd(s, ctx[s] + 1, 10)
            h = t.lookup(s, r, step)
            if h is not None:
                t.prepare_load(s, h, KVViews([], [], h.tokens), step, r.req_id)
        else:
            old = ctx.get(s, 0)
            grow = old > 0 and rng.random() < 0.8
            tokens = (old if grow else 0) + rng.randrange(1, 3 * B)
            if tokens > 4 * B:
                tokens = rng.randrange(1, 2 * B)
                grow = False
            r = rnd(s, old + 1 if grow else 0, tokens - (old if grow else 0))
            t.drop_stale(s, r)
            t.prepare_write_back(s, tokens, None, KVViews([], []))
            ctx[s] = tokens
        keys.update({v: k for k, v in t._entry_ids.items()})
        pending += t.ready_ops()
        rng.shuffle(pending)
        for o in pending[:rng.randrange(len(pending) + 1)]:
            mem.run(o, keys)
            t.complete(o.op_id, step)
            pending.remove(o)
        slots_ok(t)
    while pending or t.waiting:
        pending += t.ready_ops()
        o = pending.pop()
        mem.run(o, keys)
        t.complete(o.op_id, 0)
    assert not t.in_flight and not t.guard.writer and not t.guard.readers
    assert mem.count["read"] > 20 and mem.count["write"] > 50
    assert dram is None or mem.count["demote"] > 20
    assert t.tiers[-1].stats["evicted"] > 5                   # the SSD evicts (small SSD)
    if dram:
        assert t.wb[0].stats["evict_writes"] > 5 and t.wb[0].stats["ahead_writes"] == 0


# --- the victim chain's write-back ---------------------------------------------------------------------------------

def test_dram_takes_only_write_backs_and_only_what_it_lacks():
    t = TierManager(cfg(dram=1.0, ssd=None))
    (w,) = t.prepare_write_back(1, 2 * B, None, KVViews([], []))
    assert w.op == "write" and w.tier == "dram" and len(w.slots) == 2
    assert t.prepare_write_back(1, 2 * B, None, KVViews([], [])) == []     # held whole: only refreshed
    t.drop_stale(1, rnd(1, 2 * B + 1, 50))                            # the next round extends it: kept
    (w2,) = t.prepare_write_back(1, 3 * B, None, KVViews([], []))
    assert w2.block_start == 2 and len(w2.slots) == 1                   # only the new block and the state
    t.drop_stale(1, rnd(1, 0, 4 * B))                                  # a round that does not resume it: dropped
    assert 1 not in t.tiers[0].entries and t.take_gone() == [1]


def test_dram_demotes_only_at_eviction_and_not_what_ssd_holds():
    t = TierManager(cfg(dram=0.006, ssd=0.020))             # 6 slots: two 2-block contexts
    for k in (1, 2):
        t.prepare_write_back(k, 2 * B, None, KVViews([], []))
    drain(t)
    assert not t.wb[0].evictable(1) and not t.wb[0].evictable(2)       # dirty: SSD does not hold them
    ops = t.prepare_write_back(3, 2 * B, None, KVViews([], []))      # evicts 1: its demote, then the write
    demote, write = [o for o in ops if o.moves_bytes]
    assert (demote.op, write.op) == ("demote", "write") and demote.op_id in write.deps
    assert 1 in t.tiers[1].entries and 1 not in t.tiers[0].entries
    assert t.wb[0].stats == dict(ahead_writes=0, evict_writes=1, evict_waits=1)
    drain(t)
    t.prepare_write_back(1, 2 * B, None, KVViews([], []))            # back in DRAM (evicts 2: demoted)
    t.prepare_write_back(4, 2 * B, None, KVViews([], []))            # evicts 3: demoted
    drain(t)
    ops = t.prepare_write_back(5, 2 * B, None, KVViews([], []))      # evicts 1: SSD holds it whole, nothing moves
    assert [o.op for o in ops] == ["drop", "write"] and t.wb[0].stats["evict_writes"] == 3


class Clock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns


def test_scheduler_writes_back_ahead_and_evicts_only_written_contexts():
    """HBM of 21 blocks; sessions of 1 KV + 6 state blocks. The LRU tail is written ahead to keep C - F - D >= R (the
    largest demand in 2T); a request that needs the room of a context not yet written starts its write and waits."""
    kv = KvManager(21, keep_sessions=True)
    clock = Clock()
    pool = BalancedKVPool(cfg(dram=1.0, ssd=None), clock=clock)
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8 * B), pool, dispatch_ahead=0)
    hbm = pool.hbm.wb
    finish(s, rnd(1, 0, 100))
    assert pool.ready_ops() == [] and hbm.headroom == 7 and hbm.dirty_charge == 7   # 21 - 7 >= 7: nothing written
    clock.ns = 1
    finish(s, rnd(2, 0, 100))
    finish(s, rnd(3, 0, 100))                                           # demand 14 (T = 0: this step alone)
    ws = pool.ready_ops()                                               # 21 - 21 < 14: write 1 and 2 ahead
    assert [w.op for w in ws] == ["write", "write"] and hbm.stats["ahead_writes"] == 2 and kv.pinned[1] == 1
    clock.ns = 3
    for w in ws:
        s.op_done(w.op_id, 3)
    assert hbm.write_ns == 2 and hbm.evictable(1) and hbm.evictable(2) and not kv.pinned
    big = Request("big", 2 * B, session=9)                              # 2 + 6 blocks: evicts 1 and 2, written
    s.add(big, 3)
    assert s.next_microbatch(3) is not None and 1 not in kv.kept and 2 not in kv.kept
    assert kv.stats["blocked_by_write"] == 0
    hbm.headroom = 0                                                    # nothing ahead: 3 stays dirty
    r = Request("r", 6 * B, session=8)                                  # 6 + 6 blocks: fits only once 3 is written
    s.add(r, 3)
    s.stage_done(0, 3)
    s.next_microbatch(3)
    assert r.alloc is None
    assert kv.stats["blocked_by_write"] >= 1 and hbm.stats["evict_writes"] == 1
    assert set(pool.write_back_stats()) == {"hbm"}


def test_a_landed_write_starts_the_blocked_read_at_once():
    """A round that resumes from DRAM but does not fit until a kept context is written starts its read when that
    write lands, not at the next microbatch's forming."""
    kv = KvManager(21, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram=1.0, ssd=None), clock=Clock())
    t = pool.manager
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    store(t, 5, 7 * B)                                                  # session 5's context in DRAM: 7 + 6 blocks
    drain(t)
    finish(s, rnd(1, 0, 100))                                           # 7 kept, dirty
    hog = Request("hog", 100)                                           # holds 7 blocks
    s.add(hog, 0)
    assert s.next_microbatch(0) is not None
    r = rnd(5, 7 * B + 1, 10)                                           # needs 14 > 21 - 7 - 7 (1 not written)
    s.add(r, 0)
    s.stage_done(0, 3)
    s.next_microbatch(0)
    assert kv.stats["blocked_by_write"] >= 1 and r.req_id not in s.reading
    (w,) = [o for o in pool.manager.ready_ops() if o.op == "write"]   # (no pool.ready_ops: no write-ahead step)
    s.op_done(w.op_id, 5)                                               # lands: 1 evictable, the read starts now
    assert r.req_id in s.reading and r.read_start_ns == 5
    assert [o.op for o in pool.manager.ready_ops()] == ["read"]


def test_a_round_resuming_its_dirty_kept_context_does_not_count_it_twice():
    """`fits`: the round's footprint includes the kept context it takes over; that context, dirty, is not also held
    back from it."""
    kv = KvManager(21, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram=1.0, ssd=None), clock=Clock())
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8 * B), pool, dispatch_ahead=0)
    finish(s, rnd(1, 0, 100))                                           # 7 kept, dirty
    hog = Request("hog", 100)                                           # holds 7 blocks
    s.add(hog, 0)
    assert s.next_microbatch(0) is not None
    nxt = rnd(1, 101, 6 * B)                                            # 7 + 6 blocks, 7 of them its kept context
    assert kv.fits(nxt) and kv.stats["blocked_by_write"] == 0
