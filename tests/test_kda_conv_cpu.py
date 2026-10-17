"""CPU test of `kda_conv.tables`: the token blocks of `_conv_kernel` cover every row of every sequence exactly once,
never cross a sequence, and carry its (start, end, state block, has initial state); one row per sequence for
`_state_kernel`. tests/test_kda_conv_gpu.py checks the kernels against the Dao calls."""

import numpy as np

from ppimpl.model import kda_conv


def test_tables_cover_rows_once():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lens = rng.integers(1, 3 * kda_conv.BLOCK_T + 5, rng.integers(1, 8))
        seqs, at = [], 0
        for i, n in enumerate(lens):
            seqs.append((at, at + int(n), int(rng.integers(0, 7000)), bool(rng.integers(0, 2))))
            at += int(n)
        blocks, sq = kda_conv.tables(seqs)
        assert blocks.dtype == np.int32 and sq.dtype == np.int32 and sq.shape == (len(seqs), 3)
        seen = np.zeros(at, dtype=np.int64)
        for r, s, e, si in blocks:
            assert s <= r < e
            match = [q for q in seqs if q[0] == s]
            assert len(match) == 1 and match[0][1] == e and si == 2 * match[0][2] + int(match[0][3])
            seen[r:min(r + kda_conv.BLOCK_T, e)] += 1
        assert (seen == 1).all()
        for (s, e, slot, init), row in zip(seqs, sq):
            assert tuple(row) == (s, e, 2 * slot + int(init))
