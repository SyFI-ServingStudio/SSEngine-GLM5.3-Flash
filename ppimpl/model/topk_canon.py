"""Deterministic indexer top-k: the same set and order for the same logits.

vLLM's prefill top-k (`top_k_per_row_prefill`) returns a valid top-k, but in rows with more than k keys neither its
order nor, when keys tie at the k-th value, its set is fixed run to run (measured/issues.md, job 7315: an 8k prompt,
~5% of forwards). The sparse attention sums in that order, so the same input gave different bits. `canonicalize`
rewrites the kernel's answer in place: every key above the k-th largest value plus, among the keys equal to it, the
lowest indices, in ascending order, -1 fills last. Rows of at most k keys come back as the kernel gives them (all
their keys, ascending).

The CUDA path reads the logits once, spread over the whole GPU, and touches a row again only when it has a tie:
1. `_kth_kernel`, one program per row: the k-th value (the smallest selected logit) and the number of selected keys.
2. `_count_kernel`, one program per (row, 4,096 keys): counts the row's keys at or above the k-th value.
3. `_finish_kernel`, one program per row: if that count equals the number selected, the kernel's set is already the
   canonical one (every key at or above the k-th value), and only its order is fixed: the 512 indices sorted. Else
   (keys tied at the k-th value outside the set) the row is rewritten in two passes over its logits: count the keys
   above the k-th value, then compact the kept indices with prefix sums.
At 512k context (131k pools) the old one-program-per-row two-pass kernel took 0.66 ms for 1,022 rows (job 7404),
more than the logits and the top-k together. `canonical_topk_torch` is the same rule in torch ops (CPU tests, and the
reference the GPU test compares against).
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
    def _kth_kernel(logits_ptr, stride_l, ks_ptr, dst_ptr, stride_d, kth_ptr, nvalid_ptr, cnt_ptr,
                    K: tl.constexpr, KP: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        lo = tl.load(ks_ptr + r).to(tl.int32)
        offs_k = tl.arange(0, KP)
        sel = tl.load(dst_ptr + r * stride_d + offs_k, mask=offs_k < K, other=-1)
        valid = sel >= 0
        v = tl.load(logits_ptr + r * stride_l + lo + sel, mask=valid, other=float("inf"))
        tl.store(kth_ptr + r, tl.min(v, axis=0))
        tl.store(nvalid_ptr + r, tl.sum(valid.to(tl.int32), axis=0))
        tl.store(cnt_ptr + r, 0)

    @triton.jit
    def _count_kernel(logits_ptr, stride_l, ks_ptr, ke_ptr, kth_ptr, cnt_ptr, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        c0 = tl.program_id(1) * BLOCK
        lo = tl.load(ks_ptr + r).to(tl.int32)
        hi = tl.load(ke_ptr + r).to(tl.int32)
        if c0 + BLOCK > lo and c0 < hi:
            kth = tl.load(kth_ptr + r)
            cols = c0 + tl.arange(0, BLOCK)
            m = (cols >= lo) & (cols < hi)
            x = tl.load(logits_ptr + r * stride_l + cols, mask=m, other=float("-inf"))
            n = tl.sum(((x >= kth) & m).to(tl.int32), axis=0)
            tl.atomic_add(cnt_ptr + r, n, sem="relaxed")

    @triton.jit
    def _finish_kernel(logits_ptr, stride_l, ks_ptr, ke_ptr, dst_ptr, stride_d, kth_ptr, nvalid_ptr, cnt_ptr,
                       K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        drow = dst_ptr + r * stride_d
        offs_k = tl.arange(0, KP)
        n_valid = tl.load(nvalid_ptr + r)
        if tl.load(cnt_ptr + r) == n_valid:
            # the set is every key at or above the k-th value: sort it, -1 last
            big = 2147483647
            sel = tl.load(drow + offs_k, mask=offs_k < K, other=-1)
            s = tl.sort(tl.where(sel >= 0, sel, big), dim=0)
            tl.store(drow + offs_k, tl.where(s == big, -1, s), mask=offs_k < K)
        else:
            # keys tie at the k-th value outside the set: every key above it, then the lowest tied indices
            lo = tl.load(ks_ptr + r).to(tl.int32)
            hi = tl.load(ke_ptr + r).to(tl.int32)
            row = logits_ptr + r * stride_l
            kth = tl.load(kth_ptr + r)
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


COUNT_BLOCK = 4096


def canonicalize(logits, ks, ke, dst: torch.Tensor, k: int) -> None:
    """In place on `dst` [rows, k] int32 (key indices relative to each row's `ks`); `logits` [>= rows, keys] fp32."""
    n = dst.shape[0]
    if n == 0:
        return
    if triton is None or not dst.is_cuda:
        canonical_topk_torch(logits, ks, ke, dst, k)
        return
    assert dst.dtype == torch.int32 and dst.stride(1) == 1 and logits.stride(1) == 1, (dst.dtype, dst.stride())
    kp = triton.next_power_of_2(k)
    scratch = torch.empty(3, n, dtype=torch.int32, device=dst.device)
    kth, n_valid, cnt = scratch[0].view(torch.float32), scratch[1], scratch[2]
    _kth_kernel[(n,)](logits, logits.stride(0), ks, dst, dst.stride(0), kth, n_valid, cnt, K=k, KP=kp, num_warps=4)
    _count_kernel[(n, triton.cdiv(logits.shape[1], COUNT_BLOCK))](logits, logits.stride(0), ks, ke, kth, cnt,
                                                                  BLOCK=COUNT_BLOCK, num_warps=8)
    _finish_kernel[(n,)](logits, logits.stride(0), ks, ke, dst, dst.stride(0), kth, n_valid, cnt,
                         K=k, KP=kp, BLOCK=1024, num_warps=4)
