"""The g0 page table on CPU (ppimpl/model/metadata.py `g0_table`): KV in 128-token pages (`config.PAGE_TOKENS`)
addresses the same kernel blocks and slots as the block layout did when the pages are whole blocks in order, and
vLLM's indexer builder (every second column // 2: its 128-token storage pages) recovers the pages themselves."""

import random

import numpy as np

from ppimpl.config import BLOCK_PAGES, BLOCK_TOKENS as B, PAGE_TOKENS as P
from ppimpl.engine.kv import block_pages, kv_pages_for
from ppimpl.model.metadata import KERNEL_BLOCK, g0_table

KPB = B // KERNEL_BLOCK                 # 134 kernel blocks per block


def rows(specs):
    """(seq_lens, positions, qsl) of rows (start, tokens)."""
    seq = np.array([s + n for s, n in specs], dtype=np.int64)
    pos = np.concatenate([np.arange(s, s + n) for s, n in specs])
    qsl = np.r_[0, np.cumsum([n for _, n in specs])].astype(np.int64)
    return seq, pos, qsl


def test_whole_blocks_in_order_give_the_block_layout():
    specs = [(0, 9000), (8000, 1500), (17_000, 1)]
    blocks = [[4, 9], [2, 7], [11, 3, 5]]
    seq, pos, qsl = rows(specs)
    width = 4 * KPB
    table, slots = g0_table([block_pages(b) for b in blocks], seq, pos, qsl, width, P)
    j = np.arange(KPB)
    for r, (b, (s, n)) in enumerate(zip(blocks, specs)):
        nb = -(-(s + n) // B)
        old = (np.asarray(b[:nb])[:, None] * KPB + j).reshape(-1)          # the block layout's table
        np.testing.assert_array_equal(table[r, :kv_pages_for(s + n) * 2], old[:kv_pages_for(s + n) * 2])
        p = pos[qsl[r]:qsl[r + 1]]
        np.testing.assert_array_equal(slots[qsl[r]:qsl[r + 1]], np.asarray(b)[p // B] * B + p % B)


def test_scattered_pages_address_their_own_slots_and_the_indexer_recovers_them():
    rng = random.Random(3)
    pool = block_pages(range(1, 40))
    rng.shuffle(pool)
    specs = [(0, 5000), (8571, 700), (20_000, 255)]
    pages, k = [], 0
    for s, n in specs:
        m = kv_pages_for(s + n)
        pages.append(pool[k:k + m])
        k += m
    seq, pos, qsl = rows(specs)
    table, slots = g0_table(pages, seq, pos, qsl, 8 * KPB, P)
    for r, (s, n) in enumerate(specs):
        m = kv_pages_for(s + n)
        np.testing.assert_array_equal(table[r, : 2 * m:2] // 2, pages[r])         # the indexer's page table
        np.testing.assert_array_equal(table[r, 1: 2 * m:2], np.asarray(pages[r]) * 2 + 1)
        assert not table[r, 2 * m:].any()
        p = pos[qsl[r]:qsl[r + 1]]
        sl = slots[qsl[r]:qsl[r + 1]]
        np.testing.assert_array_equal(sl // P, np.asarray(pages[r])[p // P])        # each token in its own page
        np.testing.assert_array_equal(sl % P, p % P)
        # the same slot as block * 8,576 + offset in the views (page p is page p % 67 of block p // 67)
        pg = sl // P
        np.testing.assert_array_equal(sl, (pg // BLOCK_PAGES) * B + (pg % BLOCK_PAGES) * P + p % P)
    assert len(set(slots.tolist())) == len(slots)
