"""Slot-mode tier index (the server, lanes/m6 TIERS.md §3, §5-§7): slots per block, op order by dependencies, write
before evict in HBM, SSD eviction at a small SSD, SSD reads through DRAM or staging. A byte model executes the ops in
any order `ready_ops` allows and checks that every op reads what the context holds."""

import random

import pytest

from ppimpl.config import BLOCK_TOKENS as B, SchedulerConfig, TierConfig
from ppimpl.engine.kv import Allocation, KvManager, kv_blocks_for
from ppimpl.engine.scheduler import Request, Scheduler
from ppimpl.engine.tiers import Tiers

SLOT = 1_000_000          # 1 MB slots: dram_gb=0.01 is 10 slots


def cfg(dram=0.010, ssd=0.020, state_slots=1, staging=0.010, writes="dram-evictions"):
    return TierConfig(dram_gb=dram, ssd_gb=ssd, slot_bytes=SLOT, state_slots=state_slots, staging_gb=staging,
                      ssd_writes=writes, dram_read_gb_s=10.0, ssd_read_gb_s=1.0)


_n = 0


def rnd(session, prefix, new):
    global _n
    _n += 1
    return Request(f"r{_n}", prefix + new, session=session, prefix_tokens=prefix)


def store(t, session, tokens, prefix=0):
    t.store(rnd(session, prefix, tokens - prefix), Allocation([], []))


def drain(t):
    """Run every op to completion, in dependency order."""
    while True:
        ready = t.ready_ops()
        if not ready:
            assert not t.waiting
            return
        for o in ready:
            t.op_done(o.op_id)


def slots_ok(t):
    """Every slot is free or in exactly one entry."""
    for s in t.tiers:
        held = [x for e in s.entries.values() for x in e.slots + e.state]
        assert len(held) == len(set(held)) and not set(held) & set(s.free), s.name
        assert len(held) + len(s.free) == s.num_slots, s.name


def test_write_takes_one_slot_per_block_plus_state():
    t = Tiers(cfg())
    store(t, 1, 2 * B)
    (op,) = t.take_ops()
    assert op.op == "write" and op.tier == "dram" and len(op.slots) == 2 and len(op.state_slots) == 1
    assert t.tiers[0].used == 3 * SLOT and t.tiers[0].stats["written_bytes"] == 3 * SLOT
    slots_ok(t)


def test_growth_keeps_full_blocks_and_rewrites_the_partial_one():
    t = Tiers(cfg())
    store(t, 1, B + B // 2)                                   # blocks 0 (full), 1 (half)
    first = t.tiers[0].entries[1]
    s0, s1, st = first.slots[0], first.slots[1], first.state[0]
    t.take_ops()
    store(t, 1, 3 * B, prefix=B + B // 2 + 1)                 # extends it: block 0 kept, 1 and 2 new
    (op,) = t.take_ops()
    e = t.tiers[0].entries[1]
    assert e.slots[0] == s0 and op.block_start == 1 and op.token_start == B + B // 2
    assert op.slots == e.slots[1:] and len(op.slots) == 2
    assert t.tiers[0].stats["written_bytes"] == 3 * SLOT + 3 * SLOT     # 2 blocks + state, then 2 blocks + state
    slots_ok(t)
    assert {s1, st} <= set(t.tiers[0].free) | set(e.slots) | set(e.state)


def test_eviction_demotes_first_and_the_reusing_write_waits_for_it():
    t = Tiers(cfg(dram=0.006))                                # 6 slots: two 2-block contexts (3 slots each)
    store(t, 1, 2 * B)
    store(t, 2, 2 * B)
    drain(t)
    store(t, 3, 2 * B)                                        # evicts 1: demote to SSD, then 3's write into its slots
    ops = [o for o in t.take_ops() if o.moves_bytes]
    t.ops = ops
    assert [o.op for o in ops] == ["demote", "write"]
    demote, write = ops
    assert demote.host_tier == "dram" and set(demote.host_slots) & set(write.slots)
    assert write.deps == [demote.op_id]                       # write-after-read on the reused slots
    assert [o.op for o in t.ready_ops()] == ["demote"]
    t.op_done(demote.op_id)
    assert [o.op for o in t.ready_ops()] == ["write"]
    slots_ok(t)


def test_small_ssd_evicts_lru_and_reuses_slots():
    t = Tiers(cfg(dram=0.003, ssd=0.006))                     # DRAM 1 context, SSD 2 contexts of 2 blocks
    for s in range(1, 5):
        store(t, s, 2 * B)
    assert set(t.tiers[0].entries) == {4} and set(t.tiers[1].entries) == {2, 3}
    assert t.tiers[1].stats["evicted"] == 1                   # session 1 left the SSD
    assert t.lookup(rnd(1, 2 * B + 1, 10)) is None            # computed again
    slots_ok(t)


def test_ssd_read_copies_up_through_new_dram_slots_in_one_op():
    t = Tiers(cfg(dram=0.003, ssd=0.020))
    store(t, 1, 2 * B)
    store(t, 2, 2 * B)                                        # 1 -> SSD
    drain(t)
    r = rnd(1, 2 * B + 1, 10)
    assert t.lookup(r) == (1, 2 * B)
    op = t.read(r, 1, 2 * B, 0, Allocation([7, 8, 9], [1, 2, 3, 4, 5, 6]), 0)
    e = t.tiers[0].entries[1]
    assert op.tier == "ssd" and op.host_tier == "dram" and op.host_slots == e.slots and op.host_state == e.state
    demote = next(o for o in t.ops if o.op == "demote")       # 2 left DRAM for the copy-up
    assert op.deps == [demote.op_id]
    assert t.read_ns(1, 2 * B) == round(3 * SLOT / 1e9 * 1e9)  # whole context: 2 blocks + state at 1 GB/s
    slots_ok(t)


def test_ssd_only_writes_and_reads_through_staging():
    t = Tiers(cfg(dram=None, ssd=0.020, staging=0.004))
    store(t, 1, 2 * B)
    (w,) = t.ready_ops()
    assert w.op == "write" and w.tier == "ssd" and w.host_tier == "staging" and w.host_slots == [0, 1]
    assert w.host_state == [2]
    r = rnd(1, 2 * B + 1, 10)
    op = t.read(r, 0, 2 * B, 0, Allocation([7, 8, 9], [1, 2, 3, 4, 5, 6]), 0)
    assert op.host_tier == "staging" and op.slots == t.tiers[0].entries[1].slots
    assert op.host_slots == [3, 0] and op.deps == [w.op_id]   # the ring wraps onto the write's staging slot


def test_kept_context_pinned_until_its_write_is_done():
    kv = KvManager(20, keep_sessions=True)
    t = Tiers(cfg(dram=1.0))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), t, dispatch_ahead=0)
    a = rnd(1, 0, 100)
    s.add(a, 0)
    s.next_microbatch(0)
    s.stage_done(0, 3)
    s.stage_done(3, 3)
    s.finish(a)
    (w,) = t.ready_ops()
    assert kv.pinned[1] == 1
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
    t = Tiers(cfg(dram=1.0))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), t, dispatch_ahead=0)
    store(t, 5, 2 * B)
    drain(t)
    r = rnd(5, 2 * B + 1, 99)
    s.add(r, 0)
    assert s.next_microbatch(0) is None
    (op,) = t.ready_ops()
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
    rng = random.Random(seed)
    t = Tiers(cfg(dram=dram, ssd=0.020, state_slots=2, staging=0.012))
    mem, keys, ctx = Bytes(), {}, {}
    pending = []
    for step in range(300):
        s = rng.randrange(8)
        if s in ctx and rng.random() < 0.5:
            r = rnd(s, ctx[s] + 1, 10)
            hit = t.lookup(r)
            if hit is not None:
                t.read(r, hit[0], hit[1], 0, Allocation([], []), step)
        else:
            old = ctx.get(s, 0)
            grow = old > 0 and rng.random() < 0.8
            tokens = (old if grow else 0) + rng.randrange(1, 3 * B)
            if tokens > 4 * B:
                tokens = rng.randrange(1, 2 * B)
                grow = False
            t.store(rnd(s, old + 1 if grow else 0, tokens - (old if grow else 0)), Allocation([], []))
            ctx[s] = tokens
        keys.update({v: k for k, v in t._entry_ids.items()})
        pending += t.ready_ops()
        rng.shuffle(pending)
        for o in pending[:rng.randrange(len(pending) + 1)]:
            mem.run(o, keys)
            t.op_done(o.op_id)
            pending.remove(o)
        slots_ok(t)
    while pending or t.waiting:
        pending += t.ready_ops()
        o = pending.pop()
        mem.run(o, keys)
        t.op_done(o.op_id)
    assert not t.in_flight and not t.guard.writer and not t.guard.readers
    assert mem.count["read"] > 20 and mem.count["write"] > 50
    assert dram is None or mem.count["demote"] > 20
    assert t.tiers[-1].stats["evicted"] > 5                   # the SSD evicts (small SSD)
