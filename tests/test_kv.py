import random

import numpy as np
import pytest

from ssengine.config import BLOCK_PAGES, BLOCK_TOKENS, PAGE_TOKENS
from ssengine.engine.kv import (KvManager, PagePool, StateShape, default_state, kv_blocks_for, kv_pages_for, pages_for,
                                pool_size)

STATE_PAGES = default_state().pages
from ssengine.engine.scheduler import Request


def test_block_and_page_counts():
    assert BLOCK_PAGES * PAGE_TOKENS == BLOCK_TOKENS == 8576 and BLOCK_PAGES == 67
    assert kv_blocks_for(1) == 1
    assert kv_blocks_for(BLOCK_TOKENS) == 1
    assert kv_blocks_for(BLOCK_TOKENS + 1) == 2
    assert kv_blocks_for(131072) == 16
    assert kv_pages_for(1) == 1 and kv_pages_for(PAGE_TOKENS) == 1 and kv_pages_for(PAGE_TOKENS + 1) == 2
    assert default_state() == StateShape(5, 64)       # GLM-5.3-Flash compact state (model/state_layout.py)
    assert pages_for(8192) == 64 + STATE_PAGES == 64 + 320 and pages_for(8192, 2) == 64 + 640


def test_pool_size_null_block_and_token_cap():
    assert pool_size([7144, 7200, 7144, 7300]) == 7143
    assert pool_size([7144] * 4) * BLOCK_TOKENS == 61_258_368
    assert pool_size([7144] * 4, pool_tokens=15_273_856) == 1781       # 1_contract.md §2.3, Tiers
    assert pool_size([7144] * 4, pool_tokens=7_632_640) * BLOCK_TOKENS == 7_632_640
    with pytest.raises(ValueError):
        pool_size([1, 5])


def check_pool(kv: KvManager):
    """The charge is exactly what the pool hands out, every page and state run once, never the null block."""
    allocs = list(kv.held.values()) + [k.alloc for k in kv.kept.values()] + list(kv.reads.values())
    used = sum(a.num_pages for a in allocs)
    assert used + kv.pool.num_free == kv.num_blocks * BLOCK_PAGES
    assert kv.hard + kv.kept_charge + kv.area_held == used
    pages = [p for a in allocs for p in a.kv_pages]
    pages += [p for a in allocs for p in kv.pool.run_pages(a.state_blocks + a.base)]
    c = kv.state.run_pages
    assert c <= BLOCK_PAGES and all(0 < b <= kv.pool.num_blocks              # a run: the first c pages of a block
                                    for a in allocs for b in a.state_blocks + a.base)
    assert len(set(pages)) == len(pages) and min(pages, default=BLOCK_PAGES) >= BLOCK_PAGES
    assert max(pages, default=0) < (kv.num_blocks + 1) * BLOCK_PAGES


def test_admit_release_never_hands_out_null_block():
    kv = KvManager(20)                               # 1,340 pages; a state is 5 runs of 64 = 320
    ra, rb, rc = Request("a", 3 * BLOCK_TOKENS), Request("b", 8 * BLOCK_TOKENS), Request("c", 7 * BLOCK_TOKENS)
    a = kv.admit(ra)
    assert len(a.kv_pages) == 3 * BLOCK_PAGES and len(a.state_blocks) == 5
    assert all(b != 0 for b in a.state_blocks) and min(a.kv_pages) >= BLOCK_PAGES
    check_pool(kv)
    assert kv.pool.num_free == 1340 - 201 - 320
    assert kv.admit(rb) is None                     # 536 + 320 > 819: waits
    b = kv.admit(rc)
    assert b is not None and kv.pool.num_free == 819 - 469 - 320
    kv.release(ra)
    assert kv.pool.num_free == 30 + 521
    kv.release(rc)
    assert kv.pool.num_free == 20 * BLOCK_PAGES
    assert kv.fits_ever(1020 * PAGE_TOKENS) and not kv.fits_ever(1020 * PAGE_TOKENS + 1)
    check_pool(kv)


def test_charge_is_tokens_rounded_to_a_page_not_a_block():
    """A request of L tokens holds ceil(L / 128) pages plus its state's pages: its KV rounded up to 128 tokens, never
    to 8,576."""
    kv = KvManager(100)
    for i, L in enumerate([1, 127, 128, 129, BLOCK_TOKENS - 1, BLOCK_TOKENS + 1, 3 * BLOCK_TOKENS + 4000]):
        r = Request(f"r{i}", L)
        a = kv.admit(r)
        assert len(a.kv_pages) == kv_pages_for(L) and kv._charge[r.req_id] == kv_pages_for(L) + STATE_PAGES
        lost = len(a.kv_pages) * PAGE_TOKENS - L
        assert 0 <= lost < PAGE_TOKENS
        check_pool(kv)
    assert kv.used_tokens == sum(kv_pages_for(L) * PAGE_TOKENS for L in [1, 127, 128, 129, BLOCK_TOKENS - 1,
                                                                         BLOCK_TOKENS + 1, 3 * BLOCK_TOKENS + 4000]) \
        + 7 * STATE_PAGES * PAGE_TOKENS


def test_many_short_contexts_fit_where_whole_blocks_did_not():
    """20 blocks: two requests of 1 token each held 2 x (1 + 6) = 14 blocks under block rounding and a third had to
    wait; with pages and compact states each costs 8 pages + 320, and the KV pages fill the pages the states' blocks
    leave (3 per block), not blocks of their own."""
    kv = KvManager(20, keep_sessions=True)
    for i in range(3):
        r = Request(f"r{i}", 1000, session=i)
        assert kv.admit(r) is not None
        r.alloc = kv.held[r.req_id]
        kv.release(r)
    assert len(kv.kept) == 3 and kv.stats["evicted_sessions"] == 0
    kv_blocks = {p // BLOCK_PAGES for k in kv.kept.values() for p in k.alloc.kv_pages}
    state_blocks = {b for k in kv.kept.values() for b in k.alloc.state_blocks}
    assert kv_blocks <= state_blocks and kv.pool.num_whole == 20 - 15
    check_pool(kv)


def test_page_pool_takes_split_blocks_fullest_first_and_whole_blocks_oldest_first():
    pool = PagePool(6, state=StateShape(2, 30))      # a run: pages 0-29 of a block, the other 37 are KV pages
    s = pool.take_state()                            # the oldest whole blocks
    assert s == [1, 2] and pool.num_whole == 4
    a = pool.take_pages(10)                          # partly used blocks first: blocks 1 and 2 (37 free), the lower
    assert a == list(range(BLOCK_PAGES + 30, BLOCK_PAGES + 40))
    assert pool.free_runs() == 4                     # whole blocks 3-6
    pool.give_state(s)                               # block 2 is whole again (the newest whole block); block 1's
    assert pool.num_whole == 5 and pool.free_runs() == 6   # run pages are free though its KV pages are not
    assert pool.can_take(0, 3) and not pool.can_take(0, 4)
    s2 = pool.take_state()                           # block 1 (partly used first), then the oldest whole: 3
    assert s2 == [1, 3]
    pool.give_state(s2)
    pool.give_pages(a)
    assert pool.num_free == 6 * BLOCK_PAGES and pool.num_whole == 6
    assert pool.can_take(6 * BLOCK_PAGES - 60, 1) and not pool.can_take(6 * BLOCK_PAGES - 59, 1)


def churn(seed: int, steps: int = 600, num_blocks: int = 40, keep: bool = True):
    """Random sessions on a small pool, rounds of mixed sizes (a few pages to three blocks) extending their session's
    last context; after every step the charge equals the pages held, no page twice."""
    rng = random.Random(seed)
    kv = KvManager(num_blocks, keep_sessions=keep)
    running: list[Request] = []
    ctx = {}
    for step in range(steps):
        if running and rng.random() < 0.5:
            r = running.pop(rng.randrange(len(running)))
            kv.release(r)
            ctx[r.session] = r.num_tokens
        else:
            s = rng.randrange(16)
            if any(r.session == s for r in running):
                continue
            prev = ctx.get(s, 0)
            L = prev + rng.choice([rng.randrange(1, 600), rng.randrange(1, 3 * BLOCK_TOKENS)])
            r = Request(f"r{step}", L, session=s, prefix_tokens=prev)
            charge_fits = kv.hard + kv.kept_charge + kv.footprint(L) <= kv.capacity
            if kv.fits(r):
                r.cached = kv.allocate(r)              # asserts fits again; _make_room asserts the pool can hand out
                r.alloc = kv.held[r.req_id]
                assert len(r.alloc.kv_pages) == kv_pages_for(L)
                running.append(r)
            elif not keep and charge_fits:             # turned down only for free run slots for its state
                assert kv.pool.free_runs() < kv.state.runs
        check_pool(kv)
        assert kv.hard + kv.kept_charge <= kv.capacity
    return kv


def test_random_sessions_keep_the_charge_equal_to_the_pages_held():
    """Every admission `fits` allows succeeds, including those whose charge fit but whose states found too few free
    run slots (`states_short`): more kept contexts are evicted for them."""
    short = 0
    for seed in range(6):
        kv = churn(seed)
        assert kv.stats["admitted"] > 150 and kv.stats["hits"] > 10
        short += kv.stats["states_short"]
    assert short > 0


def test_without_kept_contexts_a_request_short_of_whole_blocks_waits_for_a_running_one():
    """No kept contexts to evict: a request whose charge fits but whose state finds too few free run slots is turned
    down (`states_short`) and fits once running requests end (their states free run slots)."""
    for seed in (5, 39):                               # seeds where it happens (rare without kept contexts)
        kv = churn(seed, keep=False)
        assert kv.stats["states_short"] > 0
        assert not kv.kept and kv.stats["evicted_sessions"] == 0
