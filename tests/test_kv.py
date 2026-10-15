import pytest

from ppimpl.config import BLOCK_TOKENS, STATE_BLOCKS
from ppimpl.engine.kv import KvManager, blocks_for, kv_blocks_for, pool_size


def test_block_counts():
    assert kv_blocks_for(1) == 1
    assert kv_blocks_for(BLOCK_TOKENS) == 1
    assert kv_blocks_for(BLOCK_TOKENS + 1) == 2
    assert kv_blocks_for(131072) == 16
    assert blocks_for(8192) == 1 + STATE_BLOCKS


def test_pool_size_null_block_and_token_cap():
    assert pool_size([7144, 7200, 7144, 7300]) == 7143
    assert pool_size([7144] * 4) * BLOCK_TOKENS == 61_258_368
    assert pool_size([7144] * 4, pool_tokens=15_273_856) == 1781       # 1_contract.md §2.3, Tiers
    assert pool_size([7144] * 4, pool_tokens=7_632_640) * BLOCK_TOKENS == 7_632_640
    with pytest.raises(ValueError):
        pool_size([1, 5])


def test_admit_release_never_hands_out_null_block():
    kv = KvManager(20)
    a = kv.admit("a", 3 * BLOCK_TOKENS)
    assert len(a.kv_blocks) == 3 and len(a.state_blocks) == STATE_BLOCKS
    ids = a.kv_blocks + a.state_blocks
    assert 0 not in ids and len(set(ids)) == 9 and max(ids) <= 20
    assert kv.pool.num_free == 11
    assert kv.admit("b", 6 * BLOCK_TOKENS) is None                     # 12 > 11: waits
    b = kv.admit("c", 5 * BLOCK_TOKENS)
    assert b is not None and kv.pool.num_free == 0
    kv.release("a")
    assert kv.pool.num_free == 9
    kv.release("c")
    assert kv.pool.num_free == 20
    assert kv.fits_ever(14 * BLOCK_TOKENS) and not kv.fits_ever(14 * BLOCK_TOKENS + 1)
