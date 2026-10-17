"""KDA recurrent state gather / scatter of one microbatch's requests, as plain wide copies.

`pp_patch.kda_prefill_forward` gathers the requests' FP32 [H, 128, 128] states (4 MB each at H = 64) from the state
pool for FlashInfer's persistent KDA, zeroes the fresh requests' rows, and scatters the final states back. Torch's
index_select / index_copy_ ran at 12.7 + 9.4 us per layer for two requests (job 7885, stage 2: 1.3 TB/s); these
kernels move the same bytes with 16-byte accesses, and the gather writes the fresh rows' zeros itself.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 4096
NUM_WARPS = 4


@triton.jit
def _gather_kernel(pool_ptr, pool_s0, slots_ptr, init_ptr, out_ptr, E, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    slot = tl.load(slots_ptr + i)
    init = tl.load(init_ptr + i)
    v = tl.load(pool_ptr + slot * pool_s0 + offs, mask=m & (init != 0), other=0.0)
    tl.store(out_ptr + i.to(tl.int64) * E + offs, v, mask=m)


@triton.jit
def _scatter_kernel(state_ptr, slots_ptr, pool_ptr, pool_s0, E, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    slot = tl.load(slots_ptr + i)
    v = tl.load(state_ptr + i.to(tl.int64) * E + offs, mask=m)
    tl.store(pool_ptr + slot * pool_s0 + offs, v, mask=m)


def gather(pool: torch.Tensor, slots: torch.Tensor, init: torch.Tensor) -> torch.Tensor:
    """pool [blocks, ...] (each block contiguous, any block stride: vLLM's state pages), slots int64 [N], init int32
    [N] on the device -> [N, ...]: pool[slots[i]] where init[i], else zeros."""
    assert pool[0].is_contiguous() and slots.dtype == torch.int64
    N = slots.shape[0]
    E = pool[0].numel()
    out = torch.empty((N,) + tuple(pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    if N:
        _gather_kernel[(N, triton.cdiv(E, BLOCK))](pool, pool.stride(0), slots, init, out, E, BLOCK=BLOCK,
                                                    num_warps=NUM_WARPS)
    return out


def scatter(state: torch.Tensor, slots: torch.Tensor, pool: torch.Tensor) -> None:
    """pool[slots[i]] = state[i] (slots distinct)."""
    assert pool[0].is_contiguous() and state.is_contiguous() and slots.dtype == torch.int64
    N = slots.shape[0]
    E = pool[0].numel()
    if N:
        _scatter_kernel[(N, triton.cdiv(E, BLOCK))](state, slots, pool, pool.stride(0), E, BLOCK=BLOCK,
                                                     num_warps=NUM_WARPS)
