from ppimpl.config import BLOCK_PAGES, BLOCK_TOKENS, STATE_BLOCKS
from ppimpl.engine.kv import KvManager, pages_for
from ppimpl.engine.scheduler import PlainScheduler, Request


def make(lengths, mb=8192, in_flight=4, ahead=10, blocks=10_000):
    s = PlainScheduler(KvManager(blocks), mb, in_flight, ahead)
    reqs = [Request(f"r{i}", n) for i, n in enumerate(lengths)]
    for r in reqs:
        s.add(r)
    return s, reqs


def drain(s, last_stage=3):
    """Form microbatches; complete each right away (all stages) so the in-flight limit never binds."""
    mbs = []
    while (mb := s.next_microbatch()) is not None:
        mbs.append([(c.req.req_id, c.start, c.num_tokens, c.is_last) for c in mb])
        for st in range(last_stage + 1):
            s.stage_done(st, last_stage)
    return mbs


def test_fills_to_budget_in_arrival_order():
    s, _ = make([1024] * 20)
    mbs = drain(s)
    assert [sum(c[2] for c in mb) for mb in mbs] == [8192, 8192, 4096]
    assert [c[0] for mb in mbs for c in mb] == [f"r{i}" for i in range(20)]
    assert all(c[1] == 0 and c[3] for mb in mbs for c in mb)


def test_long_prompt_is_cut_into_budget_chunks_with_start():
    s, _ = make([20000, 3000])
    mbs = drain(s)
    assert mbs[0] == [("r0", 0, 8192, False)]
    assert mbs[1] == [("r0", 8192, 8192, False)]
    # the tail of r0 and the next request share a microbatch
    assert mbs[2] == [("r0", 16384, 3616, True), ("r1", 0, 3000, True)]


def test_partial_fill_then_next_request_continues():
    s, _ = make([5000, 5000, 5000])
    mbs = drain(s)
    assert mbs[0] == [("r0", 0, 5000, True), ("r1", 0, 3192, False)]
    assert mbs[1] == [("r1", 3192, 1808, True), ("r2", 0, 5000, True)]


def test_budget_holds_whole_prompt():
    s, _ = make([131072], mb=131072)
    assert drain(s) == [[("r0", 0, 131072, True)]]


def test_in_flight_limit_and_stage0_dispatch_ahead():
    s, _ = make([8192] * 10, ahead=1)
    assert s.next_microbatch() is not None   # running on stage 0
    assert s.next_microbatch() is not None   # queued behind it (dispatch ahead 1)
    assert s.next_microbatch() is None       # stage 0 has two: wait
    s.stage_done(0, 3)
    assert s.next_microbatch() is not None
    s.stage_done(0, 3)
    assert s.next_microbatch() is not None   # 4 in flight now
    s.stage_done(0, 3)
    assert s.in_flight == 4
    assert s.next_microbatch() is None       # in-flight limit
    s.stage_done(3, 3)
    assert s.next_microbatch() is not None


def test_dispatch_ahead_zero_forms_only_when_stage0_free():
    s, _ = make([8192] * 3, ahead=0)
    assert s.next_microbatch() is not None
    assert s.next_microbatch() is None
    s.stage_done(0, 3)
    assert s.next_microbatch() is not None


def test_admission_waits_in_arrival_order_and_resumes():
    per = -(-20000 // BLOCK_TOKENS) + STATE_BLOCKS          # 3 + 6 = 9 blocks
    s, reqs = make([20000, 20000, 1000], blocks=per + 7)     # r1 does not fit while r0 holds its blocks
    mbs = drain(s)
    assert {c[0] for mb in mbs for c in mb} == {"r0"}        # r2 would fit but waits behind r1 (arrival order)
    assert [r.req_id for r in s.waiting] == ["r1", "r2"]
    s.finish(reqs[0])                                        # r0's blocks free: r1, then r2, are admitted
    assert drain(s) == [[("r1", 0, 8192, False)], [("r1", 8192, 8192, False)],
                        [("r1", 16384, 3616, True), ("r2", 0, 1000, True)]]
    assert s.kv.pool.num_free == (per + 7) * BLOCK_PAGES - pages_for(20000) - pages_for(1000)
