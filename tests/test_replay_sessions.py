"""Session replay smoke: a few minutes of sessions_2ps.csv through the engine with the microbatch fit."""

import json

import pytest
from benchmarks.replay import HANDOFF, MEMORY, Replay, StageModel, load_trace, make_scheduler

from ssengine.config import BLOCK_PAGES, TierConfig


class Watched:
    """Checks the pool after every engine call that changes it."""

    def __init__(self, kv):
        self.kv, self.max_used = kv, 0
        for name in ("allocate", "release"):
            f = getattr(kv, name)
            setattr(kv, name, self._wrap(f))

    def _wrap(self, f):
        def g(*a, **k):
            out = f(*a, **k)
            kv = self.kv
            assert kv.hard + kv.kept_charge <= kv.capacity
            held = sum(x.num_pages for x in kv.held.values()) + sum(x.alloc.num_pages for x in kv.kept.values())
            assert held + kv.pool.num_free == kv.num_blocks * BLOCK_PAGES and held == kv.hard + kv.kept_charge
            self.max_used = max(self.max_used, kv.hard + kv.kept_charge)
            return out
        return g


@pytest.mark.parametrize("memory,tiers", [("full", False), ("ssd_heavy", True)])
def test_sessions_smoke(tmp_path, memory, tiers):
    pool, dram = MEMORY[memory]
    sched = make_scheduler("good", pool, True, TierConfig(dram_gb=dram) if tiers else None)
    watch = Watched(sched.kv)
    _, sessions = load_trace(HANDOFF / "data/sessions_2ps.csv")
    minutes = 3
    rp = Replay(sched, StageModel(), sessions=sessions[:400], max_sessions=400, horizon_ms=minutes * 60e3).run()
    summary = rp.write(tmp_path, minutes * 60e3)
    # no lost rounds: every submitted round is answered or still in the engine
    pending = {r.req_id for r in sched.policy} | set(sched.reading) | {r.req_id for r in sched.landed}
    pending |= {r.req_id for r in sched.started} | {c[0] for mb in rp.microbatches[-4:] for c in mb.chunks}
    assert set(rp.reqs) - set(rp.first_ns) <= pending
    assert len(rp.first_ns) > 1500
    assert watch.max_used <= sched.kv.capacity
    lines = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    done = [x for x in lines if x["outcome"]["status"] == "SUCCESS"]
    prefix = sum(x["source"]["data"]["prefix_len"] for x in done)
    cached = sum(x["outcome"]["server_usage"]["cached_prompt_tokens"] for x in done)
    assert 0.95 < cached / prefix < 1.0                       # hits: everything but output tokens and rewinds
    reads = [json.loads(x) for x in (tmp_path / "reads.jsonl").read_text().splitlines()]
    assert all(r["hbm_tokens"] + r["dram_tokens"] + r["ssd_tokens"] <= r["prefix_tokens"] for r in reads)
    if tiers:
        assert sum(r["dram_tokens"] + r["ssd_tokens"] for r in reads) > 0
        assert summary["scheduler"]["reads"] > 0
    mb_lines = (tmp_path / "microbatches.jsonl").read_text().splitlines()
    assert len(mb_lines) == 4 * len(rp.microbatches)
    first = json.loads(mb_lines[0])
    assert set(first) == {"stage", "index", "start_ms", "compute_ms", "requests"}
