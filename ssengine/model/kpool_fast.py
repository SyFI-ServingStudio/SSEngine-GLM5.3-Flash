"""The indexer's pool-to-token expansion on the prefill path, in one launch written in place.

The fork's `sparse_attn_indexer_kpool` (glm5next nvidia `sparse_indexer.py`), per prefill chunk after the pool top-k:
`pool_topk.to(int64)`, `positions[a:b].to(int32) + 1`, `expand_pools_and_append_tail` (128 columns per program)
into a new [rows, topk + pool - 1] tensor, then a copy into `topk_indices_buffer[a:b]`: five launches and a 34 MB
round trip per DSA layer at 4,096 rows (serving trace 7787, stage 2: copy 23.7 us, expand 45.4 us, 3 per mb).
Here: one kernel reads the int32 pool ids and the int64 positions and writes the buffer rows directly, a whole row
per program. Same values (pure index arithmetic). `install` rewrites the fork function's source at these lines,
checking each one is there (a fork change fails loudly at start-up).
"""

from __future__ import annotations

import inspect
import textwrap

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_rows_kernel(pool_ids_ptr, pos_ptr, out_ptr, topk, out_cols, pid_s0, out_s0,
                        POOL_SIZE: tl.constexpr, BLOCK_COLS: tl.constexpr):
    row = tl.program_id(0)
    seq_len = (tl.load(pos_ptr + row) + 1).to(tl.int32)
    pool_len = seq_len // POOL_SIZE
    tail_start = pool_len * POOL_SIZE
    tail_count = seq_len - tail_start
    for c0 in range(0, out_cols, BLOCK_COLS):
        cols = c0 + tl.arange(0, BLOCK_COLS)
        mask = cols < out_cols
        is_history = cols < topk
        g = cols // POOL_SIZE
        o = cols % POOL_SIZE
        pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
        hist_out = tl.where(pid >= 0, (pid * POOL_SIZE + o).to(tl.int32), -1)
        tail_off = cols - topk
        is_tail = (tail_off >= 0) & (tail_off < tail_count)
        tail_out = tl.where(is_tail, (tail_start + tail_off).to(tl.int32), -1)
        tl.store(out_ptr + row * out_s0 + cols, tl.where(is_history, hist_out, tail_out), mask=mask)


BLOCK_COLS = 512   # research/bench/kpool_expand_bench.py, job 7881: 12.3 us vs the fork's 77.8 at 4,096 rows (1,024: 14.5)
NUM_WARPS = 4


def expand_into(pool_ids: torch.Tensor, positions: torch.Tensor, pool_size: int, out: torch.Tensor) -> torch.Tensor:
    """`expand_pools_and_append_tail(pool_ids.to(int64), positions.to(int32) + 1, pool_size)` written into the first
    `topk + pool_size - 1` columns of `out` (rows = pool_ids' rows); returns that view of `out`."""
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    assert out.shape[0] == rows and out.shape[1] >= out_cols and out.stride(1) == 1 and out.dtype == torch.int32
    assert positions.shape[0] == rows and positions.stride(0) == 1 and pool_ids.stride(1) == 1
    if rows:
        _expand_rows_kernel[(rows,)](pool_ids, positions, out, topk, out_cols, pool_ids.stride(0), out.stride(0),
                                     POOL_SIZE=pool_size, BLOCK_COLS=BLOCK_COLS, num_warps=NUM_WARPS)
    return out[:, :out_cols]


_EDITS = [
    ("""                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:""",
     """                pool_ids = pool_topk
                if positions is not None:"""),
    ("""                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0""",
     """                    _pp_expand_into(
                        pool_ids,
                        positions[chunk.token_start : chunk.token_end],
                        index_kpool,
                        topk_indices_buffer[chunk.token_start : chunk.token_end],
                    )
                else:
                    pool_ids = pool_ids.to(torch.int64)
                    valid = pool_ids >= 0"""),
    ("""                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded""",
     """                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                    topk_indices_buffer[
                        chunk.token_start : chunk.token_end, : expanded.shape[-1]
                    ] = expanded"""),
]
_ORIG = None


def install() -> None:
    """Replace glm5next nvidia `sparse_indexer.sparse_attn_indexer_kpool` by its source with `_EDITS` applied,
    defined in the module itself (process-wide; its other names, patched or not, are looked up as before)."""
    global _ORIG
    import vllm.models.glm5next.nvidia.sparse_indexer as si

    if _ORIG is not None:
        return
    orig = si.sparse_attn_indexer_kpool
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(orig)))
    for old, new in _EDITS:
        assert src.count(old) == 1, f"kpool_fast: the fork's sparse_attn_indexer_kpool changed:\n{old}"
        src = src.replace(old, new)
    si._pp_expand_into = expand_into
    exec(compile(src, f"<kpool_fast:{si.__file__}>", "exec"), si.__dict__)   # rebinds it in the module
    assert si.sparse_attn_indexer_kpool is not orig
    _ORIG = orig
