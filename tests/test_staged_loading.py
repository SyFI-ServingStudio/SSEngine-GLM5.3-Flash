"""Staged loading (ppimpl/engine/scheduler.py "Staged loading"; ServingStudioSim pp-staged): pending -> loading ->
ready -> batch, one test per invariant. HBM is a KvManager of a few dozen blocks; a context of one block costs 7
(1 KV + 6 state blocks), in DRAM/SSD 2 slots (1 KV + 1 state). Ops are completed by hand (`op_done`)."""

import math

import pytest

from ppimpl.config import BLOCK_TOKENS as B, SchedulerConfig, TierConfig
from ppimpl.engine.kv import KvManager, blocks_for
from ppimpl.engine.scheduler import Request, Scheduler
from ppimpl.kvpool import BalancedKVPool, KVViews

SLOT = 1_000_000
MB = 8192
EXCL = "hbm-evictions-exclusive"


def cfg(area=14, blocks=60, dram=0.020, ssd=0.100, **kw):
    """`area` HBM blocks of a `blocks`-block HBM (`tiers.loading_area` = (area + 0.5) / blocks, rounded down)."""
    return TierConfig(dram_gb=dram, ssd_gb=ssd, slot_bytes=SLOT, state_slots=1, staging_gb=0.010,
                      dram_read_gb_s=10.0, ssd_read_gb_s=1.0, dram_writes=EXCL,
                      loading_area=(area + 0.5) / blocks, **kw)


def sched(blocks=60, force_ms=None, **kw):
    """The byte gate counts what HBM holds here (a block per 8,576 tokens and 6 state blocks, a slot each), so the
    tests below reason in blocks; `test_area_is_a_fraction_of_hbm` checks the gate's real sizing."""
    kv = KvManager(blocks, keep_sessions=True)
    pool = BalancedKVPool(cfg(blocks=blocks, **kw))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=MB, force_after_ms=force_ms), pool, dispatch_ahead=0)
    if s.staged:
        kv.area_bytes = kv.area * SLOT
        kv.read_bytes = lambda tokens: -(-tokens // B) * SLOT + 6 * SLOT
    return s, kv, pool, pool.manager


_n = 0


def rnd(session, prefix, new, arrival=0):
    global _n
    _n += 1
    return Request(f"r{_n}", prefix + new, session=session, prefix_tokens=prefix, arrival_ns=arrival)


def put(t, session, tokens, tier="dram"):
    """`session`'s context of `tokens` in DRAM (or, `tier` "ssd", demoted to SSD and gone from DRAM)."""
    t.drop_stale(session, rnd(session, 0, tokens))
    t.prepare_write_back(session, tokens, None, KVViews([], []))
    drain(t)
    if tier == "ssd":
        e = t.tiers[0].remove(session)
        t._store_in(1, session, tokens, None, "demote", src=e)
        t.tiers[0].release(e.slots + e.state)
        t.wb[0].remove(session)
        drain(t)


def drain(t):
    while ready := t.ready_ops():
        for o in ready:
            t.complete(o.op_id, 0)


def sent(pool):
    return pool.ready_ops()


def done(s, ops, now=0):
    for o in ops:
        s.op_done(o.op_id, now)


def run_mb(s, now=0):
    """Form a microbatch and finish its stages (not its requests)."""
    mb = s.next_microbatch(now)
    if mb is not None:
        s.stage_done(0, 3)
        s.stage_done(3, 3)
    return mb


# --- 6: the area is a fixed fraction of HBM (simulator `prefix_tier_loading_area`) ---------------------------------

def test_area_is_a_fraction_of_hbm():
    """HBM's area is `tiers.loading_area` of its blocks, rounded down; DRAM's the same fraction of its slots."""
    s, kv, pool, t = sched(area=14)
    assert s.staged and kv.area == 14 == int(60 * pool.cfg.loading_area)
    assert t.tiers[0].area == int(pool.cfg.loading_area * t.tiers[0].num_slots)
    assert pool.hbm.wb.reserve == 14 + blocks_for(MB)                      # write-ahead: one batch period's room
    with pytest.raises(ValueError):
        sched(area=30)                                                     # half of HBM: not a loading area
    off = Scheduler(KvManager(60, keep_sessions=True), SchedulerConfig(mb_tokens=MB),
                    BalancedKVPool(TierConfig(dram_gb=0.02, ssd_gb=None, slot_bytes=SLOT)))
    assert not off.staged and off.kv.area == 0


def test_area_gates_reads_by_tier_bytes_as_the_simulator():
    """Reads start while their bytes as a tier holds them (KV per token + one state) fit the area's bytes (the
    area's tokens at HBM's bytes per token); HBM counts max(blocks held, area), so more reads fit than the area's
    blocks when a state costs fewer bytes in a tier than its blocks in HBM."""
    kv = KvManager(60, keep_sessions=True)
    c = cfg(area=14, bytes_per_token=200, dram=None, ssd=0.1)
    pool = BalancedKVPool(c)
    pool.state_bytes = 2 * SLOT
    s = Scheduler(kv, SchedulerConfig(mb_tokens=MB), pool, dispatch_ahead=0)
    assert kv.area == 14 and kv.area_bytes == int(60 * B * c.loading_area) * 200
    assert kv.read_bytes(B) == int(B * SLOT / B) + 2 * SLOT
    tok = B                                              # 3 slots of tier bytes, 7 HBM blocks
    n = 0
    while kv.area_fits(tok):
        assert kv.hold_read(rnd(n, 0, tok), tok) is not None
        n += 1
    assert n == kv.area_bytes // (3 * SLOT)            # 8 reads of 3 slots in 24.5 MB, not 14 // 7 = 2
    assert kv.area_held == 7 * n and kv.area_charge() == 7 * n > kv.area


# --- 1: the area is deducted; write-back stays ahead so admission does not wait ------------------------------------

def test_area_is_never_used_by_admission_or_kept_contexts():
    s, kv, pool, t = sched(blocks=40, area=14)
    big = Request("big", 4 * B)                         # 4 + 6 = 10 blocks, fits beside the area
    s.add(big, 0)
    assert run_mb(s) is not None and kv.hard == 10
    huge = Request("huge", 20 * B)                      # 26 blocks: fits in 40 - 10 but not beside the 14 area
    s.add(huge, 0)
    run_mb(s)
    assert huge.req_id not in kv.held and kv.hard + kv.area > 40 - 26
    s.policy.remove(huge)
    for i in range(1, 6):                               # kept contexts: evicted to keep the area free
        s.add(r := rnd(100 + i, 0, 100), 0)
        while r.req_id not in kv.held:                  # after big's chunks
            run_mb(s)
        s.finish(r)
        done(s, sent(pool))
        assert kv.hard + kv.kept_charge + kv.area <= kv.capacity


def test_write_back_runs_ahead_so_admission_never_waits_for_a_write():
    """Sessions finish and new ones arrive: kept contexts are written while HBM still has room (the reserve), so
    every admission evicts written contexts: no blocked_by_write, every eviction clean."""
    s, kv, pool, t = sched(blocks=60, area=14)
    for i in range(30):
        s.add(r := rnd(i, 0, 100), 0)
        assert run_mb(s) is not None, i                 # admitted at once every time
        s.finish(r)
        done(s, sent(pool))                             # writes ahead land before the next arrival
        assert pool.hbm.wb.held_back() + kv.hard + kv.area <= kv.capacity - pool.hbm.wb.reserve + 7
    assert kv.stats["blocked_by_write"] == 0 and pool.hbm.wb.stats["evict_waits"] == 0
    assert pool.hbm.wb.stats["ahead_writes"] > 20 and kv.stats["evicted_sessions"] > 20


# --- 2: reads start in FIFO admission order; no jumping; a read larger than the area when it is empty ---------------

def test_reads_start_in_order_and_stop_at_the_first_that_does_not_fit():
    s, kv, pool, t = sched(blocks=80, area=14, dram=0.050)
    put(t, 1, 1 * B)                                    # 7 blocks
    put(t, 2, 2 * B)                                    # 8 blocks: does not fit beside 1
    put(t, 3, 1 * B)                                    # 7: would fit, must not jump 2
    rs = [rnd(k, n * B + 1, 10, arrival=k) for k, n in ((1, 1), (2, 2), (3, 1))]
    s.cfg.order = "fifo"
    for r in rs:
        s.add(r, 0)
    s.load(0)
    reads = sent(pool)
    assert [o.req_id for o in reads] == [rs[0].req_id] and kv.area_held == 7
    assert rs[1].load_phase is None and rs[2].load_phase is None
    done(s, reads)
    assert run_mb(s) is not None and rs[0].req_id in kv.held   # 1 admitted: area free; 2 starts, 3 does not fit
    reads = sent(pool)
    assert [o.req_id for o in reads] == [rs[1].req_id] and rs[2].load_phase is None and kv.area_held == 8


def test_read_larger_than_the_area_starts_when_it_is_empty():
    s, kv, pool, t = sched(blocks=80, area=14, dram=0.100)
    put(t, 1, 1 * B)
    put(t, 2, 12 * B)                                   # 18 blocks > the 14-block area
    a, b = rnd(1, B + 1, 10), rnd(2, 12 * B + 1, 10)
    s.cfg.order = "fifo"
    s.add(a, 0)
    s.add(b, 0)
    s.load(0)
    assert [o.req_id for o in sent(pool)] == [a.req_id] and b.load_phase is None
    s._load_dirty = True
    s.load(0)
    assert not sent(pool)                               # the area is not empty: b waits
    s.op_done(next(iter(t.in_flight)), 0)
    run_mb(s)                                           # a admitted: the area is empty; b starts
    assert [o.req_id for o in sent(pool)] == [b.req_id] and kv.area_held == 18


# --- 3: only rounds whose context is all in HBM join a batch; rereads ----------------------------------------------

def test_only_ready_rounds_join_a_batch():
    s, kv, pool, t = sched(blocks=80, area=7)
    put(t, 1, B)
    s.add(c := rnd(5, 0, 100), 0)                       # nothing to read: ready
    s.add(r := rnd(1, B + 1, 10), 0)
    put(t, 2, B)
    s.add(w := rnd(2, B + 1, 10), 0)                    # needs a read; the area holds r's
    mb = run_mb(s)
    assert {c_.req.req_id for c_ in mb} == {c.req_id}   # r is loading, w is pending: neither joins
    assert r.load_phase == "read" and w.load_phase is None and s.stats["passed_over"] >= 1
    done(s, sent(pool))
    mb = run_mb(s)
    assert [x.req.req_id for x in mb] == [r.req_id] and r.cached == B and mb[0].start == B


def test_a_round_ready_in_hbm_whose_context_is_evicted_is_read_again():
    s, kv, pool, t = sched(blocks=80, area=7)
    s.add(r0 := rnd(1, 0, 100), 0)
    run_mb(s)
    s.finish(r0)                                        # session 1 kept in HBM
    done(s, sent(pool))
    s.add(r := rnd(1, 101, 10), 0)
    r.hbm_ready = True                                  # as an admission check that found it ready and did not admit
    put(t, 1, 100)                                      # then HBM evicts it, written to DRAM
    k = kv.kept.pop(1)
    kv._unkeep(k, evicted=True)
    kv.pool.give(k.alloc.kv_blocks + k.alloc.state_blocks)
    s._load_dirty = True
    s.load(0)
    assert r.load_phase == "read" and s.stats["rereads"] == 1


# --- 4: SSD reads land in DRAM's area, hop to HBM's; a full DRAM area stalls the SSD queue -------------------------

def test_ssd_read_lands_in_the_dram_area_then_hops_to_hbm():
    s, kv, pool, t = sched(blocks=60, area=14, dram=0.030)
    d = t.tiers[0]
    assert d.area == 7                                  # 14/60 of 30 slots
    put(t, 1, 2 * B, "ssd")
    s.add(r := rnd(1, 2 * B + 1, 10), 0)
    s.load(0)
    (f,) = sent(pool)
    assert f.op == "fetch" and f.tier == "ssd" and f.host_tier == "dram" and f.blocks == (0, 0)
    assert d.area_used == 3 and kv.area_held == 0 and r.load_phase == "fetch"
    done(s, [f], 10)
    (h,) = sent(pool)                                   # the hop: DRAM's area -> HBM's
    assert h.op == "read" and h.tier == "dram" and h.slots + h.state_slots == f.host_slots + f.host_state
    assert kv.area_held == 8 and r.load_phase == "hop" and r.fetch_land_ns == 10
    assert 1 in t.tiers[1].entries                     # SSD stays inclusive
    done(s, [h], 20)
    assert d.area_used == 0 and kv.area_held == 8       # DRAM's area back; HBM's held until admission (5)
    mb = run_mb(s)
    assert mb[0].req is r and mb[0].start == 2 * B and kv.area_held == 0
    assert t.tiers[0].stats["hops"] == 1


def test_a_full_dram_area_stalls_the_ssd_queue_without_bypass():
    s, kv, pool, t = sched(blocks=60, area=14, dram=0.030)    # DRAM area 7 slots
    put(t, 1, 5 * B, "ssd")                             # 6 slots
    put(t, 2, 2 * B, "ssd")                             # 3 slots: does not fit beside 1's
    put(t, 3, B)                                        # a DRAM read: HBM's area has room
    s.cfg.order = "fifo"
    a, b, c = rnd(1, 5 * B + 1, 10), rnd(2, 2 * B + 1, 10), rnd(3, B + 1, 10)
    for r in (a, b, c):
        s.add(r, 0)
    s.load(0)
    ops = sent(pool)
    assert [(o.op, o.req_id) for o in ops] == [("fetch", a.req_id), ("read", c.req_id)]
    assert b.load_phase is None                         # waits for DRAM's area; not read straight into HBM
    done(s, [ops[0]])                                   # a's fetch lands; its hop (11 blocks) waits for c's 7
    assert not sent(pool) and a.load_phase == "hop_wait"
    done(s, [ops[1]])
    run_mb(s)                                           # c admitted: HBM's area empty, a hops
    (h,) = sent(pool)
    assert h.req_id == a.req_id and t.tiers[0].area_used == 6
    done(s, [h])                                        # DRAM's area back: b's fetch starts
    s.load(0)
    assert [o.op for o in sent(pool)] == ["fetch"] and b.load_phase == "fetch"


# --- 5: the area is held until admission -----------------------------------------------------------------------------

def test_area_is_held_until_the_round_is_admitted():
    s, kv, pool, t = sched(blocks=60, area=14)
    put(t, 1, B)
    s.add(r := rnd(1, B + 1, 10), 0)
    s.load(0)
    done(s, sent(pool))
    assert r in s.landed and kv.area_held == 7 and r.req_id in kv.reads
    s.on_stage0 = 5                                     # stage 0 busy: nothing forms
    assert s.next_microbatch(0) is None and kv.area_held == 7
    s.on_stage0 = 0
    run_mb(s)
    assert kv.area_held == 0 and r.req_id in kv.held and not kv.reads


# --- 7: force-schedule covers loading rounds -------------------------------------------------------------------------

def test_overdue_rounds_read_first_and_landed_overdue_rounds_go_first():
    s, kv, pool, t = sched(blocks=80, area=7, force_ms=1.0)
    put(t, 1, B)
    put(t, 2, B)
    s.add(old := rnd(2, B + 1, 10000, arrival=1), 1)   # longer: srpt puts it last, but it is overdue at 5 ms
    s.add(young := rnd(1, B + 1, 10, arrival=5_000_000), 5_000_000)
    s.load(5_000_000)
    (o,) = sent(pool)
    assert o.req_id == old.req_id                       # the overdue round reads first (7 blocks: one fits)
    s.add(st := Request("started", 3 * B, arrival_ns=5_000_000), 5_000_000)
    done(s, [o], 5_000_000)
    mb = s.next_microbatch(5_000_000)
    assert mb[0].req is old                             # landed and overdue: before every other round
