"""Balanced storage, 1/n per GPU (ppimpl/kvpool/placement.py) on CPU: n fake stages' cache views in one process, n
GPUs' host arenas, n = 4 (PP4) and 8 (PP8). A context written out by all n shares and read back into other pages
(the same on every stage, or each stage its own; whole blocks in order or pages scattered over blocks) is byte-equal,
every byte of every range is covered exactly once, and each GPU stores the same bytes per block."""

import ctypes
import random

import numpy as np
import pytest
import torch

from ppimpl.config import BLOCK_PAGES as P, BLOCK_TOKENS as B
from ppimpl.engine.kv import block_pages, kv_pages_for
from ppimpl.kvpool.layout import ViewRange
from ppimpl.kvpool.manager import TierOp
from ppimpl.kvpool.placement import ALIGN, BalancedPlacement

NB = 12                                  # logical blocks per fake stage


def fake_stage(stage: int, dsa: int, kda: int, seed: int):
    """Views shaped like the real ones (scaled down): MLA [NB * 134 kernel blocks, 4, 48] u8 (2 per page), indexer
    [NB * 67 pages, 8, 16] u8, tail [NB, 2, 4, 8] bf16, KDA conv [NB, 3, 64] bf16 and recurrent [NB, 4, 32, 32]
    fp32."""
    g = torch.Generator().manual_seed(seed)
    views, ranges = [], []

    def add(t, name, kind, group=-1, rows=1):
        views.append(t)
        rb = t[0].numel() * t.element_size()
        ranges.append(ViewRange(stage, name, kind, t.data_ptr(), t.stride(0) * t.element_size() * rows, rb * rows,
                                group))

    for i in range(dsa):
        add(torch.randint(0, 255, (NB * 2 * P, 4, 48), dtype=torch.uint8, generator=g), f"s{stage}.mla{i}", "kv",
            rows=2 * P)
        add(torch.randint(0, 255, (NB * P, 8, 16), dtype=torch.uint8, generator=g), f"s{stage}.idx{i}", "kv", rows=P)
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


def block_bytes(views, ranges, kv_pages, state_blocks):
    """Every byte the context owns, per range, as bytes: its KV pages' rows and its state blocks' rows. `kv_pages` /
    `state_blocks`: one list for every stage, or {stage: list}."""
    out = {}
    for r in ranges:
        kv = kv_pages[r.stage] if isinstance(kv_pages, dict) else kv_pages
        st = state_blocks[r.stage] if isinstance(state_blocks, dict) else state_blocks
        if r.kind == "kv":
            pb = r.nbytes // P
            out[r.name] = b"".join(ctypes.string_at(r.base + (p // P) * r.stride + (p % P) * pb, pb) for p in kv)
        else:
            out[r.name] = ctypes.string_at(r.base + st[r.group] * r.stride, r.nbytes)
    return out


def scattered(blocks, n: int, seed: int) -> list[int]:
    """n pages over `blocks`, in a random order (no two adjacent in memory, mostly)."""
    pages = [b * P + i for b in blocks for i in range(P)]
    random.Random(seed).shuffle(pages)
    return pages[:n]


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
    assert all(np.all(x._t["off"] % 16 == 0) and np.all(x._t["len"] % 16 == 0) and np.all(x.st_off % 16 == 0)
               for x in lay)


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("extra", [None, 4096])                 # a slot larger than the KV; the state spans slots
def test_write_then_read_into_other_pages_is_byte_equal(extra, n):
    views, ranges = stages(n)
    slot_bytes = None if extra is None else BalancedPlacement(ranges, n, 0).kv_slot_bytes + extra
    lay = [BalancedPlacement(ranges, n, q, slot_bytes=slot_bytes) for q in range(n)]
    L = lay[0]
    tokens = 3 * B + 37                                                # 4 blocks, the last one page
    npg = kv_pages_for(tokens)
    kv_src, st_src = block_pages([1, 5, 7, 2])[:npg], [3, 4, 6, 8, 9, 10]   # whole blocks in order
    want = block_bytes(views, ranges, kv_src, st_src)
    slots, sslots = [4, 0, 2, 6], list(range(7, 7 + L.state_slots))
    arenas = [np.zeros((7 + L.state_slots) * L.slot_bytes, dtype=np.uint8) for _ in range(n)]
    b_lo, b_hi = TierOp(0, "write", "dram", 0, token_end=tokens).blocks
    assert (b_lo, b_hi) == (0, 4)
    for q in range(n):
        copy(*lay[q].pieces(kv_src, st_src, slots, sslots, b_lo, b_hi, 0)[:3], arenas[q], to_host=True)
    scramble(views)                                                    # HBM scrambled
    kv_dst, st_dst = scattered([9, 11, 0, 3, 4], npg, 1), [1, 2, 5, 6, 7, 8]   # pages scattered over 5 blocks
    for q in range(n):
        copy(*lay[q].pieces([kv_dst], [st_dst], slots, sslots, b_lo, b_hi, 0)[:3], arenas[q], to_host=False)
    assert block_bytes(views, ranges, kv_dst, st_dst) == want
    scramble(views)                                                    # each stage reads into its own pages
    kv_own = [scattered([(s + j * 5) % NB for j in range(5)], npg, s) for s in range(n)]   # 5 distinct blocks
    st_own = [[(s + g) % NB for g in range(6)] for s in range(n)]
    for q in range(n):
        copy(*lay[q].pieces(kv_own, st_own, slots, sslots, b_lo, b_hi, 0)[:3], arenas[q], to_host=False)
    assert block_bytes(views, ranges, dict(enumerate(kv_own)), dict(enumerate(st_own))) == want
    for q in range(n):                                                 # and written back out from scattered pages
        arenas[q][:] = 0
        copy(*lay[q].pieces(kv_own, st_own, slots, sslots, b_lo, b_hi, 0)[:3], arenas[q], to_host=True)
    scramble(views)
    for q in range(n):
        copy(*lay[q].pieces(kv_src, st_src, slots, sslots, b_lo, b_hi, 0)[:3], arenas[q], to_host=False)
    assert block_bytes(views, ranges, kv_src, st_src) == want


def scramble(views):
    for v in views:
        v.view(torch.uint8).random_(0, 255) if v.dtype == torch.uint8 else v.normal_()


def test_partial_range_moves_only_its_blocks():
    _, ranges = stages()
    L = BalancedPlacement(ranges, 4, 1)
    sslots = list(range(20, 20 + L.state_slots))
    kv = block_pages([1, 2, 3, 4])
    dev, host, lens, stage = L.pieces(kv, [1, 2, 3, 4, 5, 6], [10, 11], sslots, 2, 4, 2)
    nkv = len(L.kv_len)
    assert len(dev) == 2 * nkv + len(L.st_len)                          # whole blocks: one piece per range
    assert set((host[:2 * nkv] // L.slot_bytes).tolist()) == {10, 11}
    r0 = min((r for r in ranges if r.kind == "kv"), key=lambda r: (r.stage, r.name))
    assert dev[0] == r0.base + 3 * r0.stride + L.q * r0.nbytes // 4    # block 2 -> HBM block 3, GPU 1's quarter
    assert lens[:2 * nkv].tolist() == np.tile(L.kv_len, 2).tolist()


def test_a_block_cut_into_scattered_pages_moves_the_same_bytes_in_more_pieces():
    _, ranges = stages()
    L = BalancedPlacement(ranges, 4, 0)
    whole = L.pieces(block_pages([2]), [], [5], [], 0, 1, 0)
    cut = L.pieces(scattered([3, 4, 6], P, 0), [], [5], [], 0, 1, 0)
    assert whole[2].sum() == cut[2].sum() == L.kv_len.sum() and len(cut[0]) > len(whole[0])
    assert sorted(whole[1].tolist()) == sorted(cut[1][np.isin(cut[1], whole[1])].tolist())
    part = L.pieces(block_pages([2])[:10], [], [5], [], 0, 1, 0)       # a last block of 10 pages: only theirs
    assert 0 < part[2].sum() < L.kv_len.sum()


def test_stage_marks_the_pieces_in_the_stage_s_own_views():
    """`pieces`' stage is the stage whose views each piece's device address is in (blocks, then the state)."""
    views, ranges = stages()
    L = BalancedPlacement(ranges, 4, 2)
    sslots = list(range(20, 20 + L.state_slots))
    dev, _, lens, stage = L.pieces(scattered([1, 2, 3, 4, 5], 3 * P, 3), [1, 2, 3, 4, 5, 6], [10, 11, 12], sslots,
                                   0, 3, 0)
    owner = {}
    for v, r in zip(views, ranges):
        lo = v.data_ptr()
        owner[(lo, lo + v.numel() * v.element_size())] = r.stage
    want = [next(s for (a, b), s in owner.items() if a <= d and d + n <= b)
            for d, n in zip(dev.tolist(), lens.tolist())]
    assert stage.tolist() == want
