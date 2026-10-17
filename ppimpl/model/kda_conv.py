"""KDA short conv of a whole microbatch in two launches, bitwise equal to the Dao kernel's per-sequence calls.

`pp_patch.kda_prefill_forward` ran Dao causal-conv1d (channel-last, width 4, SiLU, no bias) once per sequence and per
q / k / v, with a clone of the request's conv state as initial state and the final state written in place: ~45 launches
per stage microbatch on stage 2, 1.12 ms of kernels plus 0.18 ms of launch gaps at 4,096 rows (7808, minutes 12-19),
108 us per layer for one 4,096-row sequence where reading and writing the 4,096 x 24,576 BF16 rows bounds it at ~57 us.

Here, per layer:
- `_conv_kernel`: one program per (token block of one sequence, channel block), its rows in order with the previous 3
  rows in registers and the next 4 loaded ahead; x rows before the sequence start come from its conv state (or are 0
  for a fresh request); out = SiLU(sum_w weight[c, w] * x[t - 3 + w, c]) written into three contiguous [T, P] buffers
  (q, k, v; FlashInfer's KDA takes them contiguous).
- `_state_kernel`: one program per (sequence, channel block) after it: the last 3 x rows of the sequence (from its old
  state where the sequence is shorter) into its conv state.
The arithmetic is the Dao kernel's SASS (sm_100, built with --use_fast_math): fma.rn.ftz from 0 over w = 0..3, then
x * rcp.approx.ftz(1 + ex2.approx.ftz(x * -log2(e))), in inline PTX so the compiler cannot reorder or refuse FTZ.
Measured (bench/kda_conv_bench.py, job 7927, B200, BLOCK_T 16 / BLOCK_C 512 / 4 warps, within 1% of the best of 36):
~78 us per layer at 4,096 rows whatever the sequence count, vs Dao 111 (1 sequence), 137 (2), 166 (4), 225 (7 short);
q, k, v and the conv states bitwise equal in every case. Earlier versions, also bitwise: 2D tiles, each output reading
4 shifted rows, 171 us (job 7910); the row loop without the lookahead, 111 us (job 7917).
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

WIDTH = 4
BLOCK_T = 16          # token block (a block never crosses a sequence); the host table is built with it
BLOCK_C = 512
NUM_WARPS = 4
STATE_BLOCK_C = 512


@triton.jit
def _silu_conv4(x0, x1, x2, x3, w0, w1, w2, w3):
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .f32 a, e;
        fma.rn.ftz.f32 a, $5, $1, 0f00000000;
        fma.rn.ftz.f32 a, $6, $2, a;
        fma.rn.ftz.f32 a, $7, $3, a;
        fma.rn.ftz.f32 a, $8, $4, a;
        mul.ftz.f32 e, a, 0fBFB8AA3B;
        ex2.approx.ftz.f32 e, e;
        add.ftz.f32 e, e, 0f3F800000;
        rcp.approx.ftz.f32 e, e;
        mul.ftz.f32 $0, a, e;
        }""",
        constraints="=r,r,r,r,r,r,r,r,r",
        args=[x0, x1, x2, x3, w0, w1, w2, w3], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _x_row(x_ptr, x_s0, st_ptr, st_s0, P, t, cols, s, slot, init, cmask):
    """Row t of one sequence starting at s, columns `cols`, float32: t >= s from x, t < s from its conv state (row
    t - s + 3) when it has one, else 0."""
    xv = tl.load(x_ptr + t.to(tl.int64) * x_s0 + cols, mask=cmask & (t >= s), other=0.0)
    sv = tl.load(st_ptr + slot.to(tl.int64) * st_s0 + (t - s + 3) * (3 * P) + cols,
                 mask=cmask & (t < s) & (init != 0), other=0.0)
    return tl.where(t >= s, xv, sv).to(tl.float32)


@triton.jit(do_not_specialize=["T"])   # one compile for every microbatch size (the warm-up's; no in-run JIT)
def _conv_kernel(x_ptr, x_s0, w_ptr, st_ptr, st_s0, out_ptr, T, P, tab_ptr,
                 BT: tl.constexpr, BC: tl.constexpr):
    """One program: BT rows of one sequence x BC channels. The 3 rows before the block are loaded once and slide in
    registers, so every x row is read once (plus 3 per block)."""
    b = tl.program_id(0)
    row0 = tl.load(tab_ptr + 4 * b)
    s = tl.load(tab_ptr + 4 * b + 1)
    e = tl.load(tab_ptr + 4 * b + 2)
    si = tl.load(tab_ptr + 4 * b + 3)
    slot, init = si >> 1, si & 1
    C = 3 * P
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    cmask = cols < C
    wb = w_ptr + cols * 4
    w0 = tl.load(wb, mask=cmask, other=0.0).to(tl.float32)
    w1 = tl.load(wb + 1, mask=cmask, other=0.0).to(tl.float32)
    w2 = tl.load(wb + 2, mask=cmask, other=0.0).to(tl.float32)
    w3 = tl.load(wb + 3, mask=cmask, other=0.0).to(tl.float32)
    h0 = _x_row(x_ptr, x_s0, st_ptr, st_s0, P, row0 - 3, cols, s, slot, init, cmask)
    h1 = _x_row(x_ptr, x_s0, st_ptr, st_s0, P, row0 - 2, cols, s, slot, init, cmask)
    h2 = _x_row(x_ptr, x_s0, st_ptr, st_s0, P, row0 - 1, cols, s, slot, init, cmask)
    j = cols // P                                  # q, k or v
    off = j.to(tl.int64) * T * P + (cols - j * P)
    # rows load 4 ahead of the row being stored: a load placed after a store cannot move above it (out may alias x
    # for all the compiler knows), so without the lookahead each row's load waits for the previous row's store
    a0 = _ld(x_ptr, x_s0, row0, cols, cmask, e)
    a1 = _ld(x_ptr, x_s0, row0 + 1, cols, cmask, e)
    a2 = _ld(x_ptr, x_s0, row0 + 2, cols, cmask, e)
    a3 = _ld(x_ptr, x_s0, row0 + 3, cols, cmask, e)
    for i in tl.static_range(BT):
        t = row0 + i
        nxt = _ld(x_ptr, x_s0, t + 4, cols, cmask, tl.minimum(e, row0 + BT))
        y = _silu_conv4(h0, h1, h2, a0, w0, w1, w2, w3)
        tl.store(out_ptr + t.to(tl.int64) * P + off, y.to(tl.bfloat16), mask=cmask & (t < e))
        h0, h1, h2 = h1, h2, a0
        a0, a1, a2, a3 = a1, a2, a3, nxt


@triton.jit
def _ld(x_ptr, x_s0, t, cols, cmask, end):
    return tl.load(x_ptr + t.to(tl.int64) * x_s0 + cols, mask=cmask & (t < end), other=0.0).to(tl.float32)


@triton.jit
def _state_kernel(x_ptr, x_s0, st_ptr, st_s0, seq_ptr, C, BC: tl.constexpr):
    r = tl.program_id(0)
    s = tl.load(seq_ptr + 3 * r)
    e = tl.load(seq_ptr + 3 * r + 1)
    si = tl.load(seq_ptr + 3 * r + 2)
    slot, init = si >> 1, si & 1
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    cmask = cols < C
    base = st_ptr + slot.to(tl.int64) * st_s0
    v0 = _state_row(x_ptr, x_s0, base, C, e - 3, cols, s, init, cmask)
    v1 = _state_row(x_ptr, x_s0, base, C, e - 2, cols, s, init, cmask)
    v2 = _state_row(x_ptr, x_s0, base, C, e - 1, cols, s, init, cmask)
    tl.store(base + cols, v0, mask=cmask)             # every load above happens before any store
    tl.store(base + C + cols, v1, mask=cmask)
    tl.store(base + 2 * C + cols, v2, mask=cmask)


@triton.jit
def _state_row(x_ptr, x_s0, base, C, t, cols, s, init, cmask):
    xv = tl.load(x_ptr + t.to(tl.int64) * x_s0 + cols, mask=cmask & (t >= s), other=0.0)
    sv = tl.load(base + (t - s + 3) * C + cols, mask=cmask & (t < s) & (init != 0), other=0.0)
    return tl.where(t >= s, xv, sv)


def tables(seqs: list[tuple[int, int, int, bool]]) -> tuple[np.ndarray, np.ndarray]:
    """Host tables for `conv_qkv` from (first row, end row, state block, has initial state) per sequence: int32
    [blocks, 4] (block first row, sequence start, end, slot * 2 + has_init) and [sequences, 3]."""
    blocks, sq = [], []
    for s, e, slot, init in seqs:
        si = 2 * int(slot) + int(bool(init))
        sq.append((s, e, si))
        blocks += [(r, s, e, si) for r in range(s, e, BLOCK_T)]
    return (np.asarray(blocks, dtype=np.int32).reshape(-1, 4), np.asarray(sq, dtype=np.int32).reshape(-1, 3))


def conv_qkv(x: torch.Tensor, weight: torch.Tensor, conv_state: torch.Tensor, block_tab: torch.Tensor,
             seq_tab: torch.Tensor, P: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [T, >= 3P] BF16 rows (q|k|v in its first 3P columns), weight [3P, 4] BF16, conv_state [blocks, 3, 3P] BF16
    (each block contiguous, any block stride; updated in place), tables from `tables` on the device -> q, k, v, each
    a contiguous [T, P]."""
    T = x.shape[0]
    C = 3 * P
    assert x.stride(1) == 1 and weight.is_contiguous() and weight.shape == (C, WIDTH)
    assert conv_state.shape[1:] == (WIDTH - 1, C) and conv_state[0].is_contiguous()   # blocks: any stride (vLLM pages)
    out = torch.empty(3, T, P, dtype=x.dtype, device=x.device)
    nb = block_tab.shape[0]
    if T and nb:
        _conv_kernel[(nb, triton.cdiv(C, BLOCK_C))](x, x.stride(0), weight, conv_state, conv_state.stride(0), out,
                                                     T, P, block_tab, BT=BLOCK_T, BC=BLOCK_C, num_warps=NUM_WARPS)
        _state_kernel[(seq_tab.shape[0], triton.cdiv(C, STATE_BLOCK_C))](x, x.stride(0), conv_state,
                                                                         conv_state.stride(0), seq_tab, C,
                                                                         BC=STATE_BLOCK_C, num_warps=4)
    return out[0], out[1], out[2]
