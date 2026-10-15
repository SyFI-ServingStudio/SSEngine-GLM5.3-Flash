"""Phase B: the exact prefix cache (5_prefix_cache.md §2) and chunked prefill continuation."""

import pytest

from ppimpl.config import BLOCK_TOKENS, STATE_BLOCKS, SchedulerConfig
from ppimpl.engine.kv import STATE_TOKENS, KvManager, blocks_for
from ppimpl.engine.scheduler import Request, Scheduler


def rnd(name, session, prefix, new, token_ids=None):
    return Request(name, prefix + new, session=session, prefix_tokens=prefix, token_ids=token_ids)


def serve(kv, req):
    """Admit, then finish: the request's prompt stays as its session's kept context."""
    assert kv.fits(req)
    req.cached = kv.allocate(req)
    req.alloc = kv.held[req.req_id]
    kv.release(req)
    return req.cached


def check_pool(kv):
    held = sum(a.num_blocks for a in kv.held.values())
    kept = sum(k.alloc.num_blocks for k in kv.kept.values())
    assert held + kept + kv.pool.num_free == kv.num_blocks
    assert kv.hard + kv.kept_charge <= kv.capacity
    ids = [b for a in kv.held.values() for b in a.kv_blocks + a.state_blocks]
    ids += [b for k in kv.kept.values() for b in k.alloc.kv_blocks + k.alloc.state_blocks]
    assert len(set(ids)) == len(ids) and 0 not in ids


@pytest.mark.parametrize("P,A", [(8192, 1024), (32768, 8192), (131072, 1024), (524288, 8192)])
def test_hit_is_the_whole_previous_prompt(P, A):
    """Bring-up: round 1 sends round 0's prompt, its output token and A - 1 new tokens; P hits, A are computed."""
    kv = KvManager(1000, keep_sessions=True)
    assert serve(kv, rnd("r0", 1, 0, P)) == 0
    r1 = rnd("r1", 1, P + 1, A - 1)
    assert kv.peek_hit(r1) == P
    assert serve(kv, r1) == P and r1.num_tokens - r1.cached == A
    check_pool(kv)


def test_kept_blocks_move_to_the_next_round():
    kv = KvManager(100, keep_sessions=True)
    r0 = rnd("r0", 1, 0, BLOCK_TOKENS + 100)
    serve(kv, r0)
    kept = kv.kept[1].alloc
    kv_ids, state_ids = list(kept.kv_blocks), list(kept.state_blocks)
    r1 = rnd("r1", 1, BLOCK_TOKENS + 101, 3 * BLOCK_TOKENS)
    r1.cached = kv.allocate(r1)
    a = kv.held["r1"]
    assert a.kv_blocks[:2] == kv_ids and a.state_blocks == state_ids    # same blocks: the KDA state continues in place
    assert len(a.kv_blocks) == 5 and 1 not in kv.kept
    check_pool(kv)


def test_kda_state_is_only_at_the_prompt_end():
    """A prompt that keeps only part of the previous context cannot resume inside it (one KDA state per context);
    the simulator's rule (`partial_hits`) does, and is kept only to reproduce the targets."""
    kv = KvManager(100, keep_sessions=True)
    serve(kv, rnd("r0", 1, 0, 50_000))
    shrunk = rnd("r1", 1, 49_000, 10)
    assert kv.peek_hit(shrunk) == 0
    assert serve(kv, shrunk) == 0 and kv.stats["dropped_stale"] == 1      # the stale context is freed
    assert kv.kept[1].tokens == 49_010
    same = rnd("r2", 1, 49_010, 0)                                          # nothing new: nothing to resume into
    assert kv.peek_hit(same) == 0
    sim = KvManager(100, keep_sessions=True, partial_hits=True)
    serve(sim, rnd("r0", 1, 0, 50_000))
    assert sim.peek_hit(rnd("r1", 1, 49_000, 10)) == 49_000


def test_content_match_by_token_ids():
    kv = KvManager(100, keep_sessions=True)
    ids = list(range(1000))
    serve(kv, Request("r0", 1000, session="k", token_ids=ids))
    assert kv.peek_hit(Request("r1", 1500, session="k", token_ids=ids + [7] * 500)) == 1000
    other = ids[:999] + [5] + [7] * 500
    assert kv.peek_hit(Request("r2", 1500, session="k", token_ids=other)) == 0


def test_lru_evicts_whole_sessions_for_running_requests():
    per = blocks_for(2 * BLOCK_TOKENS)                       # 2 KV + 6 state blocks
    kv = KvManager(3 * per, keep_sessions=True)
    n = 2 * BLOCK_TOKENS - 100
    for s in (1, 2, 3):
        serve(kv, rnd(f"s{s}", s, 0, n))
    assert list(kv.kept) == [1, 2, 3] and kv.pool.num_free == 0
    serve(kv, rnd("s1b", 1, n + 1, 9))                       # a hit (within its 2 blocks) refreshes session 1
    assert list(kv.kept) == [2, 3, 1]
    big = rnd("new", 4, 0, BLOCK_TOKENS * 2 + 5000)          # needs 3 + 6 blocks: sessions 2 then 3 go, whole
    assert kv.fits(big)
    big.cached = kv.allocate(big)
    assert list(kv.kept) == [1] and kv.stats["evicted_sessions"] == 2
    assert kv.stats["evicted_tokens"] == 2 * n
    check_pool(kv)
    assert not kv.fits(rnd("x", 5, 0, 30 * BLOCK_TOKENS))     # running requests are never evicted


def test_pool_accounting_tokens_mode_matches_the_simulator_ledger():
    kv = KvManager(100, keep_sessions=True, accounting="tokens")
    assert kv.capacity == 100 * BLOCK_TOKENS
    r = rnd("r0", 1, 0, 10_000)
    assert kv.footprint(10_000) == 10_000 + 1 + STATE_TOKENS
    serve(kv, r)
    assert kv.kept_charge == 10_000 + STATE_TOKENS and kv.hard == 0
    blocks = KvManager(100, keep_sessions=True)
    serve(blocks, rnd("r0", 1, 0, 10_000))
    assert blocks.used_tokens == (2 + STATE_BLOCKS) * BLOCK_TOKENS


def test_chunked_continuation_after_a_hit_and_cached_token_accounting():
    kv = KvManager(1000, keep_sessions=True)
    s = Scheduler(kv, SchedulerConfig(mb_tokens=8192), dispatch_ahead=0)
    r0 = rnd("r0", 9, 0, 20_000)
    s.add(r0, 0)
    mbs = []
    while (mb := s.next_microbatch(0)) is not None:
        mbs.append([(c.req.req_id, c.start, c.num_tokens, c.is_last) for c in mb])
        s.stage_done(0, 3)
        s.stage_done(3, 3)
    assert mbs == [[("r0", 0, 8192, False)], [("r0", 8192, 8192, False)], [("r0", 16384, 3616, True)]]
    s.finish(r0)
    r1 = rnd("r1", 9, 20_001, 9000)                           # 20,000 cached; 9,001 to compute
    s.add(r1, 0)
    mb1 = s.next_microbatch(0)
    s.stage_done(0, 3)
    mb2 = s.next_microbatch(0)
    assert [(c.start, c.num_tokens) for c in mb1] == [(20_000, 8192)]
    assert [(c.start, c.num_tokens, c.is_last) for c in mb2] == [(28_192, 809, True)]
    assert r1.cached == r1.hbm_tokens == 20_000


def test_waiting_size_counts_what_is_cached_at_arrival():
    kv = KvManager(1000, keep_sessions=True)
    s = Scheduler(kv, SchedulerConfig(order="srpt"), dispatch_ahead=0)
    serve(kv, rnd("a0", 1, 0, 100_000))
    hit, miss = rnd("a1", 1, 100_001, 2000), rnd("b1", 2, 50_001, 500)    # b's context is gone: 50,501 to compute
    s.add(miss, 0)
    s.add(hit, 0)
    assert (hit.size, miss.size) == (2001, 50_501)
    assert hit.fresh == 2000 and s.backlog() == 2500                     # the backlog counts new tokens
    mb = s.next_microbatch(0)
    assert [c.req.req_id for c in mb] == ["a1", "b1"] and mb[0].start == 100_000
