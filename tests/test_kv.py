import random

import numpy as np
import pytest

from ppimpl.config import BLOCK_PAGES, BLOCK_TOKENS, PAGE_TOKENS, STATE_BLOCKS
from ppimpl.engine.kv import (STATE_PAGES, KvManager, PagePool, kv_blocks_for, kv_pages_for, pages_for, pool_size)
from ppimpl.engine.scheduler import Request


def test_block_and_page_counts():
    assert BLOCK_PAGES * PAGE_TOKENS == BLOCK_TOKENS == 8576 and BLOCK_PAGES == 67
    assert kv_blocks_for(1) == 1
    assert kv_blocks_for(BLOCK_TOKENS) == 1
    assert kv_blocks_for(BLOCK_TOKENS + 1) == 2
    assert kv_blocks_for(131072) == 16
    assert kv_pages_for(1) == 1 and kv_pages_for(PAGE_TOKENS) == 1 and kv_pages_for(PAGE_TOKENS + 1) == 2
    assert pages_for(8192) == 64 + STATE_PAGES and STATE_PAGES == STATE_BLOCKS * BLOCK_PAGES


def test_pool_size_null_block_and_token_cap():
    assert pool_size([7144, 7200, 7144, 7300]) == 7143
    assert pool_size([7144] * 4) * BLOCK_TOKENS == 61_258_368
    assert pool_size([7144] * 4, pool_tokens=15_273_856) == 1781       # 1_contract.md §2.3, Tiers
    assert pool_size([7144] * 4, pool_tokens=7_632_640) * BLOCK_TOKENS == 7_632_640
    with pytest.raises(ValueError):
        pool_size([1, 5])


def check_pool(kv: KvManager):
    """The charge is exactly what the pool hands out, every page and state block once, never the null block."""
    allocs = list(kv.held.values()) + [k.alloc for k in kv.kept.values()] + list(kv.reads.values())
    used = sum(a.num_pages for a in allocs)
    assert used + kv.pool.num_free == kv.num_blocks * BLOCK_PAGES
    assert kv.hard + kv.kept_charge + kv.area_held == used
    pages = [p for a in allocs for p in a.kv_pages]
    pages += [b * BLOCK_PAGES + i for a in allocs for b in a.state_blocks for i in range(BLOCK_PAGES)]
    assert len(set(pages)) == len(pages) and min(pages, default=BLOCK_PAGES) >= BLOCK_PAGES
    assert max(pages, default=0) < (kv.num_blocks + 1) * BLOCK_PAGES


def test_admit_release_never_hands_out_null_block():
    kv = KvManager(20)
    ra, rb, rc = Request("a", 3 * BLOCK_TOKENS), Request("b", 6 * BLOCK_TOKENS), Request("c", 5 * BLOCK_TOKENS)
    a = kv.admit(ra)
    assert len(a.kv_pages) == 3 * BLOCK_PAGES and len(a.state_blocks) == STATE_BLOCKS
    assert 0 not in a.state_blocks and min(a.kv_pages) >= BLOCK_PAGES
    check_pool(kv)
    assert kv.pool.num_free == 11 * BLOCK_PAGES
    assert kv.admit(rb) is None                     # 12 blocks' worth > 11: waits
    b = kv.admit(rc)
    assert b is not None and kv.pool.num_free == 0
    kv.release(ra)
    assert kv.pool.num_free == 9 * BLOCK_PAGES
    kv.release(rc)
    assert kv.pool.num_free == 20 * BLOCK_PAGES
    assert kv.fits_ever(14 * BLOCK_TOKENS) and not kv.fits_ever(14 * BLOCK_TOKENS + 1)
    check_pool(kv)


def test_charge_is_tokens_rounded_to_a_page_not_a_block():
    """A request of L tokens holds ceil(L / 128) pages plus its state blocks: what the simulator charges (L + 6 x 8,576
    tokens) rounded up to 128 tokens, never to 8,576."""
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
    wait; with pages each costs 1 page + 6 state blocks, so the KV pages of many contexts share one block."""
    kv = KvManager(20, keep_sessions=True)
    for i in range(3):
        r = Request(f"r{i}", 1000, session=i)
        assert kv.admit(r) is not None
        r.alloc = kv.held[r.req_id]
        kv.release(r)
    assert len(kv.kept) == 3 and kv.stats["evicted_sessions"] == 0     # 3 x 6 states + 1 shared block of pages
    kv_blocks = {p // BLOCK_PAGES for k in kv.kept.values() for p in k.alloc.kv_pages}
    assert len(kv_blocks) == 1
    check_pool(kv)


def test_page_pool_takes_split_blocks_fullest_first_and_whole_blocks_oldest_first():
    pool = PagePool(6)
    a = pool.take_pages(10)                          # splits block 1
    assert a == list(range(BLOCK_PAGES, BLOCK_PAGES + 10)) and pool.num_whole == 5
    b = pool.take_pages(BLOCK_PAGES)                 # the rest of block 1, then 10 pages of block 2
    assert {p // BLOCK_PAGES for p in b} == {1, 2} and pool.num_whole == 4
    s = pool.take_states(2)
    assert s == [3, 4] and pool.num_whole == 2
    pool.give_pages(a[:4])                           # block 1: 4 free; block 2: 57 free
    c = pool.take_pages(5)                           # the fullest partly used block first: block 1's 4, then block 2
    assert c[:4] == a[:4] and c[4] // BLOCK_PAGES == 2
    pool.give_states([3])
    assert pool.num_whole == 3 and pool.take_states(1) == [5]          # oldest whole first: 5, 6 before 3
    assert pool.num_free == 6 * BLOCK_PAGES - 10 - BLOCK_PAGES - 2 * BLOCK_PAGES + 4 - 5 + BLOCK_PAGES - BLOCK_PAGES
    pool.give_pages(a[4:] + b + c)
    pool.give_states([4, 5])
    assert pool.num_free == 6 * BLOCK_PAGES and pool.num_whole == 6
    assert pool.can_take(6 * BLOCK_PAGES, 0) and pool.can_take(0, 6) and not pool.can_take(1, 6)


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
            elif not keep and charge_fits:             # turned down only for whole blocks for its states
                assert kv.pool.num_whole < STATE_BLOCKS
        check_pool(kv)
        assert kv.hard + kv.kept_charge <= kv.capacity
    return kv


def test_random_sessions_keep_the_charge_equal_to_the_pages_held():
    """Every admission `fits` allows succeeds, including those whose charge fit but whose states found no whole free
    block (`states_short`): one kept context more is evicted for them."""
    short = 0
    for seed in range(6):
        kv = churn(seed)
        assert kv.stats["admitted"] > 150 and kv.stats["hits"] > 10
        short += kv.stats["states_short"]
    assert short > 0


def test_without_kept_contexts_a_request_short_of_whole_blocks_waits_for_a_running_one():
    """No kept contexts to evict: a request whose charge fits but whose states find no whole free block is turned
    down (`states_short`) and fits once running requests end (their states free whole blocks)."""
    for seed in (18, 47):                              # seeds where it happens (rare without kept contexts)
        kv = churn(seed, keep=False)
        assert kv.stats["states_short"] > 0
        assert not kv.kept and kv.stats["evicted_sessions"] == 0
