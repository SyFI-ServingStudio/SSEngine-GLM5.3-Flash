"""Balanced storage, 1/n per GPU (ppimpl/kvpool/placement.py) on CPU: n fake stages' cache views in one process, n
GPUs' host arenas, n = 4 (PP4) and 8 (PP8). A context written out by all n shares and read back into other blocks
(the same on every stage, or each stage its own) is byte-equal, every byte of every range is covered exactly once,
and each GPU stores the same bytes per block."""

import ctypes

import numpy as np
import pytest
import torch

from ppimpl.config import BLOCK_TOKENS as B
from ppimpl.kvpool.layout import ViewRange
from ppimpl.kvpool.manager import TierOp
from ppimpl.kvpool.placement import ALIGN, BalancedPlacement

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


SPLITS = {4: [(3, 9), (2, 9), (3, 8), (3, 8)],                      # (DSA, KDA) layers per stage, GLM-5.3-like
          8: [(2, 4), (1, 5), (2, 4), (1, 5), (2, 4), (1, 4), (1, 4), (1, 4)]}


def stages(n: int = 4):
    out = [fake_stage(s, d, k, s) for s, (d, k) in enumerate(SPLITS[n])]
    return [v for vs, _ in out for v in vs], [r for _, rs in out for r in rs]


def block_bytes(views, ranges, kv_blocks, state_blocks):
    """Every byte the context owns, per range, as bytes. `kv_blocks` / `state_blocks`: one list for every stage, or
    {stage: list}."""
    out = {}
    for r in ranges:
        kv = kv_blocks[r.stage] if isinstance(kv_blocks, dict) else kv_blocks
        st = state_blocks[r.stage] if isinstance(state_blocks, dict) else state_blocks
        blocks = kv if r.kind == "kv" else [st[r.group]]
        out[r.name] = b"".join(ctypes.string_at(r.base + b * r.stride, r.nbytes) for b in blocks)
    return out


def copy(dev, host, lens, arena, to_host):
    for d, h, n in zip(dev.tolist(), host.tolist(), lens.tolist()):
        a = arena.ctypes.data + h
        ctypes.memmove(a, d, n) if to_host else ctypes.memmove(d, a, n)


@pytest.mark.parametrize("n", [4, 8])
def test_layout_sizes_balanced_and_aligned(n):
    _, ranges = stages(n)
    lay = [BalancedPlacement(ranges, n, q) for q in range(n)]
    kv_total = sum(r.nbytes for r in ranges if r.kind == "kv")
    st_total = sum(r.nbytes for r in ranges if r.kind == "state")
    assert all(x.slot_bytes == lay[0].slot_bytes for x in lay) and lay[0].slot_bytes % ALIGN == 0
    assert sum(x.kv_len.sum() for x in lay) == kv_total                   # each byte once over the n GPUs
    assert all(x.state_bytes * n == st_total for x in lay)
    assert all(np.all(x.kv_dev % 16 == 0) and np.all(x.st_off % 16 == 0) for x in lay)


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("slot_bytes", [None, 28672])           # state spans several slots, pieces split
def test_write_then_read_into_other_blocks_is_byte_equal(slot_bytes, n):
    views, ranges = stages(n)
    lay = [BalancedPlacement(ranges, n, q, slot_bytes=slot_bytes) for q in range(n)]
    L = lay[0]
    tokens = 3 * B + 37                                                # 4 blocks, the last partial
    kv_src, st_src = [1, 5, 7, 2], [3, 4, 6, 8, 9, 10]
    want = block_bytes(views, ranges, kv_src, st_src)
    slots, sslots = [4, 0, 2, 6], list(range(7, 7 + L.state_slots))
    arenas = [np.zeros((7 + L.state_slots) * L.slot_bytes, dtype=np.uint8) for _ in range(n)]
    b_lo, b_hi = TierOp(0, "write", "dram", 0, token_end=tokens).blocks
    assert (b_lo, b_hi) == (0, 4)
    for q in range(n):
        copy(*lay[q].pieces(kv_src, st_src, slots, sslots, b_lo, b_hi, 0), arenas[q], to_host=True)
    scramble(views)                                                    # HBM scrambled
    kv_dst, st_dst = [9, 0 + 11, 3, 4], [1, 2, 5, 6, 7, 8]
    for q in range(n):
        copy(*lay[q].pieces([kv_dst], [st_dst], slots, sslots, b_lo, b_hi, 0), arenas[q], to_host=False)
    assert block_bytes(views, ranges, kv_dst, st_dst) == want
    scramble(views)                                                    # each stage reads into its own blocks
    kv_own = [[(s + j * 3) % NB for j in range(4)] for s in range(n)]
    st_own = [[(s + g) % NB for g in range(6)] for s in range(n)]
    for q in range(n):
        copy(*lay[q].pieces(kv_own, st_own, slots, sslots, b_lo, b_hi, 0), arenas[q], to_host=False)
    assert block_bytes(views, ranges, dict(enumerate(kv_own)), dict(enumerate(st_own))) == want


def scramble(views):
    for v in views:
        v.view(torch.uint8).random_(0, 255) if v.dtype == torch.uint8 else v.normal_()


def test_partial_range_moves_only_its_blocks():
    _, ranges = stages()
    L = BalancedPlacement(ranges, 4, 1)
    sslots = list(range(20, 20 + L.state_slots))
    dev, host, lens = L.pieces([1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [10, 11], sslots, 2, 4, 2)
    nkv = len(L.kv_len)
    assert len(dev) == 2 * nkv + len(L.st_len)
    assert set((host[:2 * nkv] // L.slot_bytes).tolist()) == {10, 11}
    assert dev[0] == L.kv_dev[0] + 3 * L.kv_stride[0]                    # block 2 -> HBM block 3
