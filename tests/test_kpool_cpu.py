"""kpool indexer insert with chunks that start inside a pool (pp_patch `install_kpool_unaligned`), on CPU.

The fork's Triton ops are replaced by torch references with the same contract: compress = softmax(gate + ape)-weighted
sum of each pool's 4 K rows, written at the pool's slot; seed = copy a request's last 4 rows into its tail ring. A
prompt prefilled in chunks that start anywhere must write the same pools as the unsplit prefill.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from ppimpl.model import pp_patch

KPOOL, HD = 4, 8


def ref_compress(kv_cache, slot_k, slot_score, ape, loc, pool_size, head_dim, write_mask=None, round_scale=True,
                 write_cache=True, return_compressed=False):
    p = torch.softmax(slot_score.float() + ape[None], dim=1)
    x = (p * slot_k.float()).sum(1)
    keep = torch.ones(len(loc), dtype=torch.bool) if write_mask is None else write_mask
    kv_cache[loc[keep]] = x[keep]


def ref_seed(tail, key, gate, tslot, kpool, head_dim):
    n = len(tslot)
    for i in range(n):
        t = int(tslot[i])
        if t < 0:
            continue
        ahead = int(tslot[i + kpool]) if i + kpool < n else -1
        if ahead >= 0 and ahead // kpool == t // kpool:
            continue
        tail[t // kpool, 0, t % kpool] = key[i]
        tail[t // kpool, 1, t % kpool] = gate[i]


OPS = types.SimpleNamespace(kpool_compress_and_write_cache=ref_compress)


def indexer_call(chunks, k_all, g_all, ape, cache, tail, override=True):
    """One prefill microbatch: chunks = [(request, start, num_tokens)]; request r's pools live at slots r*10000+."""
    pos = np.concatenate([np.arange(s, s + n) for _, s, n in chunks])
    req = np.concatenate([np.full(n, r) for r, _, n in chunks])
    slot = np.where(pos % KPOOL == KPOOL - 1, req * 10000 + pos // KPOOL, -1)
    tslot = req * KPOOL + pos % KPOOL                      # tail block = request id
    k = torch.stack([k_all[r][p] for r, p in zip(req, pos)])
    g = torch.stack([g_all[r][p] for r, p in zip(req, pos)])
    sm = torch.from_numpy(slot)
    pp_patch.KPOOL_HEADS.clear()
    if override:
        starts = np.array([s for _, s, _ in chunks])
        q_lens = np.array([n for _, _, n in chunks])
        row_starts = np.concatenate([[0], np.cumsum(q_lens)[:-1]])
        h = pp_patch.kpool_heads(starts, q_lens, row_starts, np.array([r for r, _, _ in chunks]), 0, 0, KPOOL)
        if h is not None:
            pp_patch.KPOOL_HEADS[sm.data_ptr()] = pp_patch.KpoolHeads(*(torch.from_numpy(a) for a in h))
    pp_patch.kpool_compress_insert(k, g, ape, cache, sm, KPOOL, HD, True, ops=OPS)
    pp_patch.kpool_seed_tail_cache(tail, k, g, torch.from_numpy(tslot), KPOOL, HD, ops=OPS, seed=ref_seed)
    assert not pp_patch._KPOOL_PENDING


def run(schedule, lens, override=True):
    """schedule: microbatches of (request, start, num_tokens). Returns the pool cache."""
    gen = torch.Generator().manual_seed(0)
    k_all = [torch.randn(L, HD, generator=gen).bfloat16() for L in lens]
    g_all = [torch.randn(L, HD, generator=gen).bfloat16() for L in lens]
    ape = torch.randn(KPOOL, HD, generator=gen)
    cache = torch.zeros(10000 * len(lens), HD)
    tail = torch.zeros(len(lens), 2, KPOOL, HD, dtype=torch.bfloat16)
    for mb in schedule:
        indexer_call(mb, k_all, g_all, ape, cache, tail, override)
    return cache


def whole(lens):
    return run([[(r, 0, L)] for r, L in enumerate(lens)], lens)


@pytest.mark.parametrize("hit", [8193 + 1, 8193 + 2, 8193 + 3, 8200])
def test_resume_at_unaligned_hit_writes_the_unsplit_pools(hit):
    """A kept prompt of `hit` tokens, then round 1 resumes there (prefix hit) in 8192-token chunks."""
    L = hit + 9000
    sched = [[(0, 0, 8192)], [(0, 8192, hit - 8192)]]
    s = hit
    while s < L:
        n = min(8192, L - s)
        sched.append([(0, s, n)])
        s += n
    assert torch.equal(run(sched, [L]), whole([L]))


@pytest.mark.parametrize("cuts", [[1], [2], [3], [5, 6, 7], [9, 10, 23], [1, 2, 3, 4, 5, 6, 7, 30]])
def test_any_chunking_writes_the_unsplit_pools(cuts):
    L = 40
    bounds = [0, *cuts, L]
    sched = [[(0, a, b - a)] for a, b in zip(bounds, bounds[1:])]
    assert torch.equal(run(sched, [L]), whole([L]))


def test_unaligned_request_after_another_in_one_microbatch():
    """Request 1 resumes at 13 right after request 0's rows: its first pool must not borrow request 0's rows."""
    lens = [24, 30]
    sched = [[(1, 0, 13)], [(0, 0, 24), (1, 13, 17)]]
    assert torch.equal(run(sched, lens), whole(lens))
    # a microbatch shorter than one pool still completes the straddling pool
    sched = [[(1, 0, 13)], [(1, 13, 3)], [(1, 16, 14), (0, 0, 24)]]
    assert torch.equal(run(sched, lens), whole(lens))


def test_without_the_override_the_straddling_pool_is_wrong():
    """The fork's insert alone (no heads): the pool around an unaligned start is missing or mixed."""
    L = 40
    sched = [[(0, 0, 13)], [(0, 13, 27)]]
    got, ref = run(sched, [L], override=False), whole([L])
    assert not torch.equal(got, ref)
    assert torch.equal(run(sched, [L]), ref)


def test_heads_host_rows():
    rows, ring, blocks, last = pp_patch.kpool_heads(np.array([0, 13, 6, 5]), np.array([8, 10, 1, 3]),
                                                    np.array([0, 8, 18, 19]), np.array([7, 8, 9, 10]), 0, 0, KPOOL)
    # request 1: start 13 (m=1), pool 12..15 = ring 12, rows 8..10; request 2: 6+1 < 8, no completion;
    # request 3: start 5 (m=1), 3 tokens complete 4..7 at row 21
    assert rows.tolist() == [[8, 8, 9, 10], [19, 19, 20, 21]]
    assert ring.tolist() == [[True, False, False, False], [True, False, False, False]]
    assert blocks.tolist() == [8, 10] and last.tolist() == [10, 21]


def test_heads_suffix_equals_heads_of_the_prefill_slice():
    """Heads built for every request before the H2D copy, then cut to the indexer's prefill requests, equal the heads
    built for those requests directly (random batches, every decode/prefill split)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = int(rng.integers(1, 7))
        starts = rng.integers(0, 40, R)
        q_lens = rng.integers(1, 12, R)
        qsl = np.concatenate([[0], np.cumsum(q_lens)])
        tails = rng.integers(0, 50, R)
        allh, req = pp_patch.kpool_heads_all(starts, q_lens, qsl[:-1], tails, KPOOL)
        for nd in range(R + 1):
            want = pp_patch.kpool_heads(starts, q_lens, qsl[:-1], tails, nd, int(qsl[nd]), KPOOL)
            got = pp_patch.kpool_heads_suffix(allh, req, nd, int(qsl[nd]))
            assert (want is None) == (got is None)
            if want is not None:
                for a, b in zip(want, got):
                    np.testing.assert_array_equal(a, b)
