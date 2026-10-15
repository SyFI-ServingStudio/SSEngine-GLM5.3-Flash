"""Deterministic indexer top-k: the same set and order for the same logits.

vLLM's prefill top-k (`top_k_per_row_prefill`) returns a valid top-k, but in rows with more than k keys neither its
order nor, when keys tie at the k-th value, its set is fixed run to run (measured/issues.md, job 7315: an 8k prompt,
~5% of forwards). The sparse attention sums in that order, so the same input gave different bits. `canonicalize`
rewrites the kernel's answer in place: every key above the k-th largest value plus, among the keys equal to it, the
lowest indices, in ascending order, -1 fills last. Rows of at most k keys come back as the kernel gives them (all
their keys, ascending).

The CUDA path is one Triton program per row: two passes over the row's logits (count the keys above the k-th value,
then compact the kept indices with prefix sums), ~2 reads of the logits. `canonical_topk_torch` is the same rule in
torch ops (CPU tests, and the reference the GPU test compares against).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments
    triton = None


def canonical_topk_torch(logits, ks, ke, dst: torch.Tensor, k: int, block_elems: int = 1 << 24) -> None:
    """Torch reference of `canonicalize` (rows in blocks of about `block_elems` logits)."""
    n, width = dst.shape[0], logits.shape[1]
    big = torch.iinfo(torch.int32).max
    cols = torch.arange(width, dtype=torch.int32, device=dst.device)
    step = max(1, block_elems // max(width, 1))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        lg, d = logits[r0:r1], dst[r0:r1]
        lo, hi = ks[r0:r1].to(torch.int32)[:, None], ke[r0:r1].to(torch.int32)[:, None]
        inr = (cols >= lo) & (cols < hi)
        sel = d.to(torch.int32)
        vals = lg.gather(1, (sel + lo).clamp_(0, width - 1).long())
        kth = torch.where(sel >= 0, vals, torch.full_like(vals, float("inf"))).amin(1, keepdim=True)
        above = inr & (lg > kth)
        tie = inr & (lg == kth)
        take = above | (tie & (tie.cumsum(1, dtype=torch.int32) <= k - above.sum(1, keepdim=True, dtype=torch.int32)))
        rel = torch.where(take, cols - lo, big)
        out = rel.topk(k, dim=1, largest=False, sorted=True).values
        d.copy_(out.masked_fill_(out == big, -1))


if triton is not None:
    @triton.jit
    def _canonical_topk_kernel(logits_ptr, stride_l, ks_ptr, ke_ptr, dst_ptr, stride_d,
                               K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        lo = tl.load(ks_ptr + r).to(tl.int32)
        hi = tl.load(ke_ptr + r).to(tl.int32)
        row = logits_ptr + r * stride_l
        drow = dst_ptr + r * stride_d
        offs_k = tl.arange(0, KP)
        sel = tl.load(drow + offs_k, mask=offs_k < K, other=-1)
        valid = sel >= 0
        v = tl.load(row + lo + sel, mask=valid, other=float("inf"))
        kth = tl.min(v, axis=0)
        n_valid = tl.sum(valid.to(tl.int32), axis=0)
        n_above = tl.zeros((), dtype=tl.int32)
        for c0 in range(lo, hi, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            m = cols < hi
            x = tl.load(row + cols, mask=m, other=float("-inf"))
            n_above += tl.sum(((x > kth) & m).to(tl.int32), axis=0)
        need = n_valid - n_above
        pos = tl.zeros((), dtype=tl.int32)
        ties = tl.zeros((), dtype=tl.int32)
        for c0 in range(lo, hi, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            m = cols < hi
            x = tl.load(row + cols, mask=m, other=float("-inf"))
            tie = ((x == kth) & m).to(tl.int32)
            take = ((x > kth) & m) | ((tie != 0) & (tl.cumsum(tie, axis=0) + ties <= need))
            slot = tl.cumsum(take.to(tl.int32), axis=0) - 1 + pos
            tl.store(drow + slot, cols - lo, mask=take)
            pos += tl.sum(take.to(tl.int32), axis=0)
            ties += tl.sum(tie, axis=0)
        tl.store(drow + offs_k, tl.full((KP,), -1, tl.int32), mask=(offs_k >= pos) & (offs_k < K))


def canonicalize(logits, ks, ke, dst: torch.Tensor, k: int) -> None:
    """In place on `dst` [rows, k] int32 (key indices relative to each row's `ks`); `logits` [>= rows, keys] fp32."""
    n = dst.shape[0]
    if n == 0:
        return
    if triton is None or not dst.is_cuda:
        canonical_topk_torch(logits, ks, ke, dst, k)
        return
    assert dst.dtype == torch.int32 and dst.stride(1) == 1 and logits.stride(1) == 1, (dst.dtype, dst.stride())
    _canonical_topk_kernel[(n,)](logits, logits.stride(0), ks, ke, dst, dst.stride(0),
                                 K=k, KP=triton.next_power_of_2(k), BLOCK=1024, num_warps=4)
