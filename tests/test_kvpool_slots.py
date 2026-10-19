"""Slot-mode index of the pool's leader (ppimpl/kvpool/manager.py; the server, lanes/m6 TIERS.md §3, §5-§7): slots per
block, op order by dependencies, write before evict in HBM, SSD eviction at a small SSD, SSD reads through DRAM or
staging. A byte model executes the ops in any order `ready_ops` allows and checks that every op reads what the
context holds."""

import random

import pytest

from ppimpl.config import BLOCK_PAGES, BLOCK_TOKENS as B, SchedulerConfig, TierConfig
from ppimpl.engine.kv import KvManager, pages_for
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
    pool = BalancedKVPool(cfg(dram=1.0, hbm_write_ahead=0.1))   # 7 of 20 blocks > 10%: written ahead at once
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    finish(s, rnd(1, 0, 100))
    (w,) = pool.ready_ops()
    assert w.op == "write" and kv.pinned[1] == 1
    big = Request("big", 14 * B)                              # needs every block: session 1 is pinned, it waits
    assert not kv.fits(big)
    nxt = rnd(1, 101, 50)                                     # session 1's next round may not take over yet
    s.add(nxt, 0)
    assert s.next_microbatch(0) is None and s.stats["busy_waits"] >= 1
    s.op_done(w.op_id, 1)
    assert not kv.pinned and kv.fits(big)
    mb = s.next_microbatch(1)
    assert [(c.req.req_id, c.start) for c in mb] == [(nxt.req_id, 100)]


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
@pytest.mark.parametrize("dram,writes", [(0.012, "hbm-evictions"), (0.012, "hbm-evictions-exclusive"),
                                         (None, "hbm-evictions")])
def test_random_sessions_every_read_sees_its_context(seed, dram, writes):
    """HBM writes contexts back (what the entry lacks), each round's admission drops what it does not resume from,
    DRAM demotes ahead above its watermark and at eviction (exclusive: a DRAM read releases the entry)."""
    rng = random.Random(seed)
    t = TierManager(cfg(dram=dram, ssd=0.020, state_slots=2, staging=0.012, dram_write_ahead=0.7, dram_writes=writes))
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
            t.step()
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
        assert t.wb[0].stats["ahead_writes"] > 5


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


def test_dram_demotes_ahead_and_its_eviction_then_moves_nothing():
    t = TierManager(cfg(dram=0.009, ssd=0.020, dram_write_ahead=0.5))   # 9 slots: three 2-block contexts
    for k in (1, 2):
        t.prepare_write_back(k, 2 * B, None, KVViews([], []))
    drain(t)
    t.step()                            # 6 of 9 dirty > 4.5: the LRU entry (1) goes to SSD, stays in DRAM
    (d,) = t.ready_ops()
    assert d.op == "demote" and 1 in t.tiers[1].entries and 1 in t.tiers[0].entries
    assert not t.wb[0].evictable(1)                         # being written
    t.complete(d.op_id, 0)
    assert t.wb[0].evictable(1) and not t.wb[0].evictable(2)
    assert t.wb[0].stats == dict(ahead_writes=1, evict_writes=0, evict_waits=0)
    t.prepare_write_back(3, 2 * B, None, KVViews([], []))
    ops = t.prepare_write_back(4, 2 * B, None, KVViews([], []))      # evicts 1: SSD holds it already
    assert [o.op for o in ops] == ["drop", "write"] and t.wb[0].stats["evict_waits"] == 0
    ops = t.prepare_write_back(5, 2 * B, None, KVViews([], []))      # evicts 2, not written: demote, then the write
    assert [o.op for o in ops if o.moves_bytes] == ["demote", "write"]
    assert t.wb[0].stats == dict(ahead_writes=1, evict_writes=1, evict_waits=1)


def test_ssd_loss_makes_the_dram_entry_dirty_again():
    t = TierManager(cfg(dram=1.0, ssd=0.003, dram_write_ahead=0.0))      # SSD: one 2-block context
    t.prepare_write_back(1, 2 * B, None, KVViews([], []))
    drain(t)
    t.step()                                                          # 1 -> SSD ahead
    drain(t)
    assert t.wb[0].evictable(1)
    t.prepare_write_back(2, 2 * B, None, KVViews([], []))
    drain(t)
    t.step()                                                          # 2 -> SSD evicts 1 there
    assert 1 not in t.tiers[1].entries and not t.wb[0].evictable(1)


def test_scheduler_writes_back_ahead_and_evicts_only_written_contexts():
    """HBM of 20 blocks; sessions of 1 KV + 6 state blocks. Above 70% the LRU tail is written ahead (the same
    `WriteBack` as DRAM's); a request that needs the room of a context not yet written starts its write and waits."""
    kv = KvManager(21, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram=1.0, ssd=None, hbm_write_ahead=0.7))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    hbm = pool.hbm.wb
    finish(s, rnd(1, 0, 100))
    assert pool.ready_ops() == [] and hbm.dirty_charge == pages_for(100)   # 403 of 1,407 pages: nothing written
    finish(s, rnd(2, 0, 100))
    finish(s, rnd(3, 0, 100))                                           # 1,209 > 0.7 x 1,407: write 1 ahead
    (w,) = pool.ready_ops()
    assert hbm.stats["ahead_writes"] == 1 and kv.pinned[1] == 1 and not hbm.evictable(1)
    s.op_done(w.op_id, 0)
    assert hbm.evictable(1) and not kv.pinned and hbm.dirty_charge == 2 * pages_for(100)
    big = Request("big", 3 * B, session=9)                              # 201 + 402 pages: evicts 1 (written) and 2
    s.add(big, 0)
    assert s.next_microbatch(0) is None                                 # 2 is not written yet: its write starts
    assert kv.stats["blocked_by_write"] >= 1 and hbm.stats["evict_writes"] == 1 and hbm.stats["evict_waits"] >= 1
    (w,) = pool.ready_ops()
    s.op_done(w.op_id, 0)
    assert s.next_microbatch(0) is not None and 1 not in kv.kept and 2 not in kv.kept
    r = rnd(1, 101, 10)                         # session 1 resumes from DRAM, once 3 (not written) makes room
    s.add(r, 0)
    rd = None
    for _ in range(6):                          # big's prompt runs on
        if s.on_stage0:
            s.stage_done(0, 3)
        s.next_microbatch(0)
        for o in pool.ready_ops():
            if o.op == "read":
                rd = o
            else:
                s.op_done(o.op_id, 0)
        if rd is not None:
            break
    assert rd is not None and rd.req_id == r.req_id and 3 not in kv.kept
    assert hbm.stats["ahead_writes"] + hbm.stats["evict_writes"] == 3
    assert set(pool.write_back_stats()) == {"hbm"}


# --- exclusive DRAM (dram_writes: hbm-evictions-exclusive) ---------------------------------------------------------

EXCL = "hbm-evictions-exclusive"


def test_exclusive_dram_read_releases_the_entry_and_its_slots_wait_for_the_read():
    t = TierManager(cfg(dram=0.006, dram_writes=EXCL))
    store(t, 1, 2 * B)
    drain(t)
    slots = t.tiers[0].entries[1].slots + t.tiers[0].entries[1].state
    op, _ = read(t, rnd(1, 2 * B + 1, 10), "dram", 2 * B, KVViews([7, 8], [1, 2, 3, 4, 5, 6], 2 * B))
    assert 1 not in t.tiers[0].entries and t.tiers[0].used == 0 and t.tiers[0].stats["released"] == 1
    assert t.take_gone() == [1]
    slots_ok(t)
    store(t, 2, 2 * B)
    (w,) = store(t, 3, 2 * B)                                # reuses the released slots: waits for the read
    assert set(w.slots + w.state_slots) == set(slots) and w.deps == [op.op_id]
    drain(t)
    t.drop_stale(1, rnd(1, 2 * B + 1, B // 2))              # session 1 ran on and HBM evicts it again: written whole
    (w1,) = [o for o in t.prepare_write_back(1, 2 * B + B // 2, None, KVViews([], [])) if o.op == "write"]
    assert w1.token_start == 0 and w1.block_start == 0 and len(w1.slots) == 3


def test_inclusive_dram_read_keeps_the_entry():
    t = TierManager(cfg(dram=0.006))
    store(t, 1, 2 * B)
    drain(t)
    read(t, rnd(1, 2 * B + 1, 10), "dram", 2 * B, KVViews([7, 8], [1, 2, 3, 4, 5, 6], 2 * B))
    assert 1 in t.tiers[0].entries and t.tiers[0].stats["released"] == 0


def test_exclusive_dram_ssd_read_leaves_both_tiers():
    """An SSD read: the SSD entry stays (inclusive); DRAM had no copy."""
    t = TierManager(cfg(dram=0.003, ssd=0.020, dram_writes=EXCL))
    store(t, 1, 2 * B)
    store(t, 2, 2 * B)                                       # 1 -> SSD
    drain(t)
    read(t, rnd(1, 2 * B + 1, 10), "ssd", 2 * B, KVViews([7, 8, 9], [1, 2, 3, 4, 5, 6], 2 * B))
    assert 1 in t.tiers[1].entries and set(t.tiers[0].entries) == {2}


def test_dram_entry_the_ssd_holds_whole_is_clean():
    """A context SSD still holds whole (read from SSD, evicted from HBM unchanged) is stored in DRAM clean: its
    eviction demotes nothing."""
    t = TierManager(cfg(dram=0.003, ssd=0.020, dram_writes=EXCL))
    store(t, 1, 2 * B)
    store(t, 2, 2 * B)                                       # 1 -> SSD
    drain(t)
    t.prepare_write_back(1, 2 * B, None, KVViews([], []))   # 1 back in DRAM (evicts 2 -> SSD)
    drain(t)
    assert t.wb[0].evictable(1)
    before = dict(t.wb[0].stats)
    ops = t.prepare_write_back(3, 2 * B, None, KVViews([], []))   # evicts 1: SSD has it, nothing moves for it
    assert [o.op for o in ops if o.moves_bytes] == ["write"] and t.wb[0].stats == before


def test_inclusive_hbm_hit_keeps_the_write_ahead_copy():
    """Inclusive DRAM writes ahead: a kept context taken over by its next round (an HBM hit) stays in DRAM."""
    kv = KvManager(21, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram=1.0, ssd=None, hbm_write_ahead=0.0))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    finish(s, rnd(1, 0, 100))
    (w,) = pool.ready_ops()
    s.op_done(w.op_id, 0)
    assert 1 in pool.manager.tiers[0].entries
    r = rnd(1, 101, 10)
    s.add(r, 0)
    assert s.next_microbatch(0) is not None and r.cached == 100
    assert 1 in pool.manager.tiers[0].entries


def test_exclusive_writes_at_eviction_and_the_blocks_wait_for_it():
    """Exclusive DRAM (the simulator's rule): nothing is written ahead, an HBM eviction writes the context at once
    (urgent), the eviction does not wait, and the evicted blocks are busy until the write is done; a read into them
    depends on it."""
    kv = KvManager(21, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram=1.0, ssd=None, hbm_write_ahead=0.0, dram_writes=EXCL))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    finish(s, rnd(1, 0, 100))
    assert pool.ready_ops() == [] and 1 not in pool.manager.tiers[0].entries      # not written ahead
    k = kv.kept[1]
    pages, states = list(k.alloc.kv_pages), list(k.alloc.state_blocks)
    big = rnd(2, 0, 9 * B - 100)                    # 9 + 6 blocks' pages: fits only by evicting session 1
    s.add(big, 0)
    assert s.next_microbatch(0) is not None and 1 not in kv.kept and kv.stats["blocked_by_write"] == 0
    assert 1 in pool.manager.tiers[0].entries and pool.hbm_busy(pages) and pool.hbm_busy([], states)
    (w,) = pool.ready_ops()
    assert w.op == "write" and w.urgent and w.kv_pages[0] == pages and w.state_blocks[0] == states
    reuse = [states[0] * BLOCK_PAGES + 5]           # a page of an evicted state block, reused as KV
    assert pool.hbm_busy(reuse) and not pool.hbm_busy([max(pages) + 1])
    o = pool.manager._emit("read", "dram", 9, token_start=0, token_end=B, kv_pages=[reuse], state_blocks=[[]])
    assert w.op_id in o.deps
    s.op_done(w.op_id, 0)
    assert not pool.hbm_busy(pages, states)
    r = rnd(1, 101, 10)                              # its next round reads it back from DRAM
    assert pool.lookup(1, r).tier == "dram"


def test_a_read_moves_only_the_states_that_survive():
    from ppimpl.kvpool.manager import states_to_move
    assert states_to_move([100], 100, 1, 2) == [100]                       # the engine: k = 1
    s = [10, 20, 30, 40]
    assert states_to_move(s, 30, 1, 2) == [30]
    assert states_to_move(s, 30, 3, 2) == [30]                             # 2 new + the resume state fill k = 3
    assert states_to_move(s, 30, 4, 2) == [20, 30]
    assert states_to_move(s, 40, 8, 2) == [10, 20, 30, 40]
