"""The leader's index in byte mode (ppimpl/kvpool/manager.py, replay's accounting): placement, LRU, read channels,
write-back to SSD, and the scheduler's read path through the pool (read-wait bound, landed rounds first)."""

import json

import pytest
from bench.replay import HANDOFF

from ppimpl.config import BLOCK_TOKENS, SchedulerConfig, TierConfig
from ppimpl.engine.kv import KvManager
from ppimpl.engine.scheduler import Request, Scheduler
from ppimpl.kvpool import BalancedKVPool, Hit, KVViews
from ppimpl.kvpool.manager import TierManager

MB = 1_000_000          # 1 MB per token below: 1 GB holds 1,000 tokens
MS = 1_000_000


def cfg(dram_gb=1.0, ssd_gb=10.0, state_bytes=0, wait_ms=1000.0):
    return TierConfig(dram_gb=dram_gb, dram_read_gb_s=10.0, ssd_gb=ssd_gb, ssd_read_gb_s=1.0,
                      bytes_per_token=MB, state_bytes=state_bytes, max_read_wait_ms=wait_ms)


_n = 0


def rnd(session, prefix, new):
    """A session round: `prefix` tokens shared with the session's previous context, then `new` tokens."""
    global _n
    _n += 1
    return Request(f"r{_n}", prefix + new, session=session, prefix_tokens=prefix)


def store(t, session, tokens, prefix=0):
    """A round of `session` (sharing `prefix` tokens with its previous context) admitted, then its `tokens`-token
    context written back from HBM (byte mode: when HBM evicts it)."""
    t.drop_stale(session, rnd(session, prefix, tokens - prefix))
    return t.prepare_write_back(session, tokens, None)


def read(t, r, tier, tokens, hbm=0, now=0):
    """The read of `r`'s session's first `tokens` from tier `tier` (index); its op."""
    ops, _ = t.prepare_load(r.session, Hit(t.tiers[tier].name, tokens, 0), _views(tokens, hbm), now, r.req_id)
    return next(o for o in ops if o.op == "read")


def hit(t, r):
    h = t.lookup(r.session, r)
    return None if h is None else (t.level[h.tier], h.tokens)


def entries(t, i):
    return {k: e.tokens for k, e in t.tiers[i].entries.items()}


def test_placement_dram_first_ssd_only_from_dram_evictions():
    t = TierManager(cfg())
    store(t, 1, 400)
    assert entries(t, 0) == {1: 400} and entries(t, 1) == {}
    assert hit(t, rnd(1, 401, 10)) == (0, 400)
    assert t.prepare_write_back(1, 400, None) == [] and t.hbm_evicted(1, 400, None) == []   # held: refreshed only


def test_lru_eviction_order_whole_sessions_and_read_refreshes():
    t = TierManager(cfg())
    store(t, 1, 400)
    store(t, 2, 400)
    read(t, rnd(1, 401, 1), 0, 400)                      # refreshes session 1: 2 is now the oldest
    store(t, 3, 400)                                     # 1 GB holds 1,000 tokens: one session must go
    assert entries(t, 0) == {1: 400, 3: 400}
    assert entries(t, 1) == {2: 400}                     # whole session, to SSD


def _views(tokens=0, hbm=0):
    return KVViews([1, 2], [3, 4, 5, 6, 7, 8], tokens, hbm)


def test_read_time_bandwidth_and_channel_queue():
    t = TierManager(cfg(state_bytes=50 * MB))
    store(t, 1, 300)
    store(t, 2, 300)
    a = read(t, rnd(1, 301, 1), 0, 300)
    assert a.land_ns == pytest.approx((300 + 50) * MB / 10e9 * 1e9)    # 35 ms at 10 GB/s, state included
    b = read(t, rnd(2, 301, 1), 0, 300, hbm=100)                      # HBM has 100: 200 tokens + state cross
    assert b.land_ns - a.land_ns == pytest.approx(250 * MB / 10e9 * 1e9)  # queued behind the first read
    assert (b.token_start, b.token_end) == (100, 300)
    assert t.wait_ns("dram", 10 * MS) == b.land_ns - 10 * MS
    assert t.lookup(1, rnd(1, 301, 1), 10 * MS).wait_ns == b.land_ns - 10 * MS


@pytest.mark.parametrize("variant,bpt,state", [("dram_unbal", 1635, 39.1e6), ("ssd_unbal", 1635, 39.1e6),
                                               ("dram", 1499, 36.9e6), ("ssd", 1499, 36.9e6)])
def test_read_time_matches_tier_targets(variant, bpt, state):
    t = TierManager(TierConfig(bytes_per_token=bpt, state_bytes=int(state),
                         dram_gb=None if variant.startswith("ssd") else 150.0))
    for case in json.loads((HANDOFF / "data/tier_targets.json").read_text())[variant]:
        assert t.read_ns(0, case["prefix_tokens"]) / MS == pytest.approx(case["read_ms"], rel=0.01)


def test_read_wait_bound_passes_the_round_over():
    """The pool reports the channel's wait; the scheduler passes a round over above its bound (1 s here)."""
    kv = KvManager(100, keep_sessions=True)
    pool = BalancedKVPool(cfg(wait_ms=1000.0))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    t = pool.manager
    store(t, 1, 900)
    store(t, 2, 50)
    read(t, rnd(1, 901, 1), 0, 900)                         # 900 MB at 10 GB/s: the channel is busy 90 ms
    assert t.wait_ns("dram", 0) == 90 * MS
    t.tiers[0].channel_free_ns = 1001 * MS                  # a channel 1.001 s behind
    r = rnd(2, 51, 1)
    assert t.lookup(2, r, 0).wait_ns == 1001 * MS and t.lookup(2, r, 2 * MS).wait_ns == 999 * MS
    s.add(r, 0)
    assert s.next_microbatch(0) is None and s.stats["passed_over"] == 1 and s.waiting == [r]
    mb = s.next_microbatch(2 * MS)                          # 0.999 s: the read starts
    assert mb is None and s.stats["reads"] == 1 and r.read_land_ns == 1001 * MS + 5 * MS


def test_scheduler_reads_passes_over_and_admits_landed_first():
    kv = KvManager(100, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram_gb=1000.0, wait_ms=1000.0))
    t = pool.manager
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    for sess in (1, 2):
        store(t, sess, 8000)                                 # contexts in DRAM only
    a, b, c = rnd(1, 8001, 100), rnd(2, 8001, 100), Request("fresh", 500)
    for r in (a, b, c):
        s.add(r, 0)
    assert a.size == 101 and b.size == 101                   # a tier hit counts as cached for the order
    t.tiers[0].channel_free_ns = 0
    mb = s.next_microbatch(0)
    # a reads (8,000 tokens at 10 GB/s: 800 ms), b would wait 0.8 s < 1 s and reads too, c runs now
    assert [c_.req.req_id for c_ in mb] == ["fresh"]
    assert list(s.reading) == [a.req_id, b.req_id]
    assert a.read_land_ns == 800 * MS and b.read_land_ns == 1600 * MS
    store(t, 3, 8000)
    e = rnd(3, 8001, 50)
    s.add(e, 0)
    s.stage_done(0, 3)
    s.stage_done(3, 3)
    assert s.next_microbatch(1 * MS) is None                 # e's read would wait 1.599 s: passed over
    assert s.stats["passed_over"] == 1 and s.waiting == [e]
    s.read_landed(a.req_id, 800 * MS)
    mb = s.next_microbatch(800 * MS)
    assert [(x.req.req_id, x.start, x.num_tokens) for x in mb] == [(a.req_id, 8000, 101)]   # landed: admitted
    assert a.cached == 8000 and a.tier == "dram" and a.tier_tokens == 8000


def test_ssd_written_only_with_dram_evictions_and_only_what_it_lacks():
    t = TierManager(cfg())
    store(t, 1, 400)
    store(t, 1, 450, prefix=401)                            # session 1 grows to 450 in DRAM: only the new tokens
    assert t.tiers[0].stats["written_bytes"] == 450 * MB
    store(t, 2, 400)
    assert t.tiers[1].stats["stored"] == 0
    ops = store(t, 3, 400)                                  # evicts 1 from DRAM: SSD takes its latest context
    assert entries(t, 1) == {1: 450}
    assert t.tiers[1].stats["written_bytes"] == 450 * MB
    demote = [o for o in ops if o.op == "demote"]
    assert [(o.tier, o.token_start, o.token_end) for o in demote] == [("ssd", 0, 450)]
    assert ops.index(demote[0]) < next(i for i, o in enumerate(ops) if o.op == "drop" and o.tier == "dram")
    # read 1 back from SSD (no copy-up into DRAM), grow it, evict it from DRAM again: SSD writes only the new 100
    read(t, rnd(1, 451, 1), 1, 450)
    assert 1 not in entries(t, 0)
    store(t, 1, 550, prefix=451)
    ops = store(t, 4, 900)                                  # evicts everything else from DRAM
    assert entries(t, 1)[1] == 550
    assert any(o.op == "demote" and (o.token_start, o.token_end) == (450, 550) for o in ops)


def test_unchanged_copy_is_not_written_back():
    t = TierManager(cfg())
    store(t, 1, 400)
    store(t, 2, 400)
    store(t, 3, 400)                                        # 1 -> SSD
    t.prepare_write_back(1, 400, None)                      # read back, evicted from HBM again: DRAM rewrites it
    stored = t.tiers[1].stats["stored"]                     # (2 -> SSD)
    store(t, 4, 400)                                        # evicts 3 -> SSD
    store(t, 5, 400)                                        # evicts 1, which SSD holds whole: not rewritten
    assert t.tiers[1].stats["stored"] == stored + 1
    assert hit(t, rnd(1, 401, 1)) == (1, 400)


def test_shrinking_context_drops_the_older_entries():
    t = TierManager(cfg())
    store(t, 1, 600)
    store(t, 2, 600)                                        # 1 -> SSD at 600
    assert entries(t, 1) == {1: 600}
    # session 1's next round keeps only its first 300 tokens: SSD's 600-token context has no state at 300
    store(t, 1, 450, prefix=300)
    assert 1 not in entries(t, 1) and t.tiers[1].stats["dropped"] == 1
    assert hit(t, rnd(1, 451, 1)) == (0, 450)
    store(t, 3, 900)                                        # evicts 1 (450) -> SSD, written whole
    assert entries(t, 1)[1] == 450
    assert hit(t, rnd(1, 301, 1)) is None                   # a shorter prompt cannot resume inside the context


def test_context_larger_than_dram_goes_straight_to_ssd():
    t = TierManager(cfg())
    ops = store(t, 1, 2000)
    assert entries(t, 0) == {} and entries(t, 1) == {1: 2000}
    assert [o.op for o in ops] == ["write"]


def test_pool_blocks_and_read_into_held_blocks():
    """A tier read holds the round's whole footprint and fills it from the HBM part it lacks."""
    kv = KvManager(50, keep_sessions=True)
    pool = BalancedKVPool(cfg(dram_gb=100.0))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), pool, dispatch_ahead=0)
    store(pool.manager, 7, 2 * BLOCK_TOKENS)
    r = rnd(7, 2 * BLOCK_TOKENS + 1, 99)
    s.add(r, 0)
    loads = []
    real = pool.manager.prepare_load
    pool.manager.prepare_load = lambda *a: loads.append(real(*a)) or loads[-1]
    assert s.next_microbatch(0) is None
    ((op,), _) = loads[0]
    assert op.op == "read" and op.kv_blocks == [r.alloc.kv_blocks] and len(r.alloc.kv_blocks) == 3
    assert kv.hard == 3 + 6 and kv.pool.num_free == 50 - 9
