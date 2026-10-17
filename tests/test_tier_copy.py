"""Balanced storage by quarters (ppimpl/runtime/tier_copy.py) on CPU: four fake stages' cache views in one process,
four GPUs' host arenas. A context written out by all four quarters and read back into other blocks is byte-equal,
every byte of every range is covered exactly once, and each GPU stores the same bytes per block."""

import ctypes

import numpy as np
import pytest
import torch

from ppimpl.runtime.tier_copy import ALIGN, Layout, ViewRange, op_blocks

NB = 12                                  # logical blocks per fake stage


def fake_stage(stage: int, dsa: int, kda: int, seed: int):
    """Views shaped like the real ones (scaled down): MLA [NB * 2 kernel blocks, 64, 48] u8, indexer [NB * 3 pages,
    16, 32] u8, tail [NB, 2, 4, 8] bf16, KDA conv [NB, 3, 64] bf16 and recurrent [NB, 4, 32, 32] fp32."""
    g = torch.Generator().manual_seed(seed)
    views, ranges = [], []

    def add(t, name, kind, group=-1, rows=1):
        views.append(t)
        rb = t[0].numel() * t.element_size()
        ranges.append(ViewRange(stage, name, kind, t.data_ptr(), t.stride(0) * t.element_size() * rows, rb * rows,
                                group))

    for i in range(dsa):
        add(torch.randint(0, 255, (NB * 2, 64, 48), dtype=torch.uint8, generator=g), f"s{stage}.mla{i}", "kv", rows=2)
        add(torch.randint(0, 255, (NB * 3, 16, 32), dtype=torch.uint8, generator=g), f"s{stage}.idx{i}", "kv", rows=3)
        add(torch.randn(NB, 2, 4, 8, generator=g).bfloat16(), f"s{stage}.tail{i}", "state", 5)
    for i in range(kda):
        add(torch.randn(NB, 3, 64, generator=g).bfloat16(), f"s{stage}.kda{i}#0", "state", i % 5)
        add(torch.randn(NB, 4, 32, 32, generator=g), f"s{stage}.kda{i}#1", "state", i % 5)
    return views, ranges


def stages():
    out = [fake_stage(s, d, k, s) for s, (d, k) in enumerate([(3, 9), (2, 9), (3, 8), (3, 8)])]
    return [v for vs, _ in out for v in vs], [r for _, rs in out for r in rs]


def block_bytes(views, ranges, kv_blocks, state_blocks):
    """Every byte the context owns, per range, as bytes."""
    out = {}
    for r in ranges:
        blocks = kv_blocks if r.kind == "kv" else [state_blocks[r.group]]
        out[r.name] = b"".join(ctypes.string_at(r.base + b * r.stride, r.nbytes) for b in blocks)
    return out


def copy(dev, host, lens, arena, to_host):
    for d, h, n in zip(dev.tolist(), host.tolist(), lens.tolist()):
        a = arena.ctypes.data + h
        ctypes.memmove(a, d, n) if to_host else ctypes.memmove(d, a, n)


def test_layout_sizes_balanced_and_aligned():
    _, ranges = stages()
    lay = [Layout(ranges, q) for q in range(4)]
    kv_total = sum(r.nbytes for r in ranges if r.kind == "kv")
    st_total = sum(r.nbytes for r in ranges if r.kind == "state")
    assert all(x.slot_bytes == lay[0].slot_bytes for x in lay) and lay[0].slot_bytes % ALIGN == 0
    assert sum(x.kv_len.sum() for x in lay) == kv_total                   # each byte once over the 4 GPUs
    assert all(x.state_bytes * 4 == st_total for x in lay)
    assert all(np.all(x.kv_dev % 16 == 0) and np.all(x.st_off % 16 == 0) for x in lay)


@pytest.mark.parametrize("slot_bytes", [None, 28672])           # state spans several slots, pieces split
def test_write_then_read_into_other_blocks_is_byte_equal(slot_bytes):
    views, ranges = stages()
    lay = [Layout(ranges, q, slot_bytes=slot_bytes) for q in range(4)]
    L = lay[0]
    tokens, BT = 3 * 100 + 37, 100                                     # 4 blocks of 100 tokens, last partial
    kv_src, st_src = [1, 5, 7, 2], [3, 4, 6, 8, 9, 10]
    want = block_bytes(views, ranges, kv_src, st_src)
    n = -(-tokens // BT)
    slots, sslots = [4, 0, 2, 6], list(range(7, 7 + L.state_slots))
    arenas = [np.zeros((7 + L.state_slots) * L.slot_bytes, dtype=np.uint8) for _ in range(4)]

    class Op:
        token_start, token_end = 0, tokens
    b_lo, b_hi = op_blocks(Op, BT)
    assert (b_lo, b_hi) == (0, n)
    for q in range(4):
        copy(*lay[q].pieces(kv_src, st_src, slots, sslots, b_lo, b_hi, 0), arenas[q], to_host=True)
    for v in views:                                                    # HBM scrambled
        v.view(torch.uint8).random_(0, 255) if v.dtype == torch.uint8 else v.normal_()
    kv_dst, st_dst = [9, 0 + 11, 3, 4], [1, 2, 5, 6, 7, 8]
    for q in range(4):
        copy(*lay[q].pieces(kv_dst, st_dst, slots, sslots, b_lo, b_hi, 0), arenas[q], to_host=False)
    assert block_bytes(views, ranges, kv_dst, st_dst) == want


def test_partial_range_moves_only_its_blocks():
    _, ranges = stages()
    L = Layout(ranges, 1)
    sslots = list(range(20, 20 + L.state_slots))
    dev, host, lens = L.pieces([1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [10, 11], sslots, 2, 4, 2)
    nkv = len(L.kv_len)
    assert len(dev) == 2 * nkv + len(L.st_len)
    assert set((host[:2 * nkv] // L.slot_bytes).tolist()) == {10, 11}
    assert dev[0] == L.kv_dev[0] + 3 * L.kv_stride[0]                    # block 2 -> HBM block 3
