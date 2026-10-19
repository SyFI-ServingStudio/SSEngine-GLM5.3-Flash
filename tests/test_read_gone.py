"""A tier read whose copy is gone by the time it starts (job 8449: excl tiers, no loading area). `_at_head` looks the
hit up, then HBM makes room for the round, and those evictions can push the round's own DRAM entry out; `pool.load`
then returns None and the round computes from what HBM held."""

from ppimpl.config import SchedulerConfig, TierConfig
from ppimpl.engine.kv import KvManager
from ppimpl.engine.scheduler import Scheduler
from ppimpl.kvpool import BalancedKVPool

from tests.test_staged_loading import EXCL, MB, SLOT, put, rnd


def test_a_read_whose_tier_copy_is_gone_lands_at_once_and_computes_the_prefix():
    kv = KvManager(60, keep_sessions=True)
    pool = BalancedKVPool(TierConfig(dram_gb=0.020, ssd_gb=0.100, slot_bytes=SLOT, state_slots=1, staging_gb=0.010,
                                     dram_read_gb_s=10.0, ssd_read_gb_s=1.0, dram_writes=EXCL))
    s = Scheduler(kv, SchedulerConfig(mb_tokens=MB), pool, dispatch_ahead=0)
    assert not s.staged
    put(pool.manager, "a", 4000)
    r = rnd("a", 4000, 100)
    s.add(r, 0)
    pool.load = lambda *a, **k: None
    mb = s.next_microbatch(0)
    assert s.stats["reads_gone"] == 1 and r.req_id not in s.reading
    assert r.cached == 0 and r.tier is None
    (c,) = [c for c in mb if c.req is r]
    assert c.start == 0 and c.is_last
