"""Layer 2 of the KV pool (dev:research/TIERS.md §3, §13.2): balanced storage, 1/n of every range per GPU. Pure numpy.

n is the PP degree (`RunConfig.num_stages`), never fixed: PP4 has n = 4 (quarters), PP8 n = 8.

GPU q of n moves share q of every range (`layout.ViewRange`): bytes [q m / n, (q + 1) m / n) of a range of m bytes.
Its host side (a DRAM or staging slot, an SSD slot at the same offset in that GPU's file) packs its shares of one
logical block's KV ranges of all stages into one slot (page_first), and of the context's state ranges into
`state_slots` slots. The slot ids come from the leader's index (`manager.py`) and are the same on every GPU.

`BalancedPlacement` is built once per member from all stages' ranges (device addresses valid in that process: its
own and the other stages' KV, opened by CUDA IPC); `pieces` turns one op into (device address, host offset, length,
stage) arrays for `mover.Mover`, the HBM side of a block from its pages. Own-stage placement (every GPU stores only
its own stage, dev:research/TIERS.md §2.4) is not built: it would need per-GPU slot sizes in the index.
"""

from __future__ import annotations

import numpy as np

from ssengine.config import BLOCK_PAGES
from ssengine.kvpool.layout import ViewRange

ALIGN = 4096                  # O_DIRECT: slot sizes and file offsets are multiples of this
PIECE_ALIGN = 16              # every piece starts 16-byte aligned on both sides


def _round(n: int, a: int) -> int:
    return -(-n // a) * a


def _shared(blocks) -> np.ndarray:
    a = np.asarray(blocks, dtype=np.int64)
    return a[0] if a.ndim == 2 and len(a) == 1 else a


class BalancedPlacement:
    """GPU q's pieces of every range and where they sit in the host slots.

    HBM holds a context's KV in pages (`config.PAGE_TOKENS`, engine/kv.py): logical block j of the context is its pages
    j x BLOCK_PAGES .. (j + 1) x BLOCK_PAGES - 1, anywhere in the pool, and the last block has only the pages its
    tokens reach. A KV range's block row is BLOCK_PAGES page rows (page p at `base + (p // BLOCK_PAGES) * stride +
    (p % BLOCK_PAGES) * page_bytes`); GPU q's share of a logical block's row is cut where its pages are not adjacent.
    A logical block whose pages are one block's, in order, moves as one piece per range, as a whole block did."""

    def __init__(self, ranges: list[ViewRange], n: int, q: int, slot_bytes: int | None = None):
        self.n, self.q = n, q
        kv = sorted((r for r in ranges if r.kind == "kv"), key=lambda r: (r.stage, r.name))
        st = sorted((r for r in ranges if r.kind == "state"), key=lambda r: (r.stage, r.name))
        for r in kv + st:
            if r.nbytes % (n * PIECE_ALIGN):
                raise ValueError(f"{r.name} (stage {r.stage}): {r.nbytes} B per block is not a multiple of "
                                 f"{n} x {PIECE_ALIGN}")
        for r in kv:
            if r.nbytes % (BLOCK_PAGES * PIECE_ALIGN):
                raise ValueError(f"{r.name} (stage {r.stage}): {r.nbytes} B per block is not {BLOCK_PAGES} pages of a "
                                 f"multiple of {PIECE_ALIGN} B")
        self.kv_slot_bytes = _round(sum(r.nbytes // n for r in kv), ALIGN)
        self.state_bytes = sum(r.nbytes // n for r in st)
        self.slot_bytes = slot_bytes or self.kv_slot_bytes
        assert self.slot_bytes >= self.kv_slot_bytes and self.slot_bytes % ALIGN == 0
        self.state_slots = -(-self.state_bytes // self.slot_bytes)
        # KV: GPU q's share [s0, s1) of each range's block row, cut at page boundaries: one template piece per (range,
        # page it touches), packed in range order on the host side
        m = np.array([r.nbytes // n for r in kv], dtype=np.int64)
        self.kv_stage = np.array([r.stage for r in kv], dtype=np.int64)
        self.kv_len = m
        self.kv_host = np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int64)
        t_range, t_page, t_dev, t_host, t_len = [], [], [], [], []
        for i, r in enumerate(kv):
            pb = r.nbytes // BLOCK_PAGES
            s0, s1 = q * m[i], (q + 1) * m[i]
            for pg in range(s0 // pb, -(-s1 // pb)):
                lo, hi = max(s0, pg * pb), min(s1, (pg + 1) * pb)
                t_range.append(i)
                t_page.append(pg)
                t_dev.append(lo - pg * pb)              # offset in the page row
                t_host.append(self.kv_host[i] + lo - s0)
                t_len.append(hi - lo)
        self._t = {k: np.array(v, dtype=np.int64) for k, v in
                   dict(range=t_range, page=t_page, off=t_dev, host=t_host, len=t_len).items()}
        self._base = np.array([r.base for r in kv], dtype=np.int64)
        self._stride = np.array([r.stride for r in kv], dtype=np.int64)
        self._pb = np.array([r.nbytes // BLOCK_PAGES for r in kv], dtype=np.int64)
        # state: pieces packed over state_slots slots, split where they cross a slot boundary
        stage, dev, stride, group, host, length = [], [], [], [], [], []
        off = 0
        for r in st:
            k = r.nbytes // n
            d = r.base + q * k
            while k:
                take = min(k, self.slot_bytes - off % self.slot_bytes)
                stage.append(r.stage)
                dev.append(d)
                stride.append(r.stride)
                group.append(r.group)
                host.append(off)
                length.append(take)
                d, off, k = d + take, off + take, k - take
        self.st_stage = np.array(stage, dtype=np.int64)
        self.st_dev = np.array(dev, dtype=np.int64)
        self.st_stride = np.array(stride, dtype=np.int64)
        self.st_group = np.array(group, dtype=np.int64)
        self.st_slot = np.array(host, dtype=np.int64) // self.slot_bytes
        self.st_off = np.array(host, dtype=np.int64) % self.slot_bytes
        self.st_len = np.array(length, dtype=np.int64)

    def _kv_pieces(self, kv_pages, slots, b_lo: int, b_hi: int, block_start: int):
        """KV pieces of logical blocks [b_lo, b_hi): (dev, host, len, stage), block by block, then range by range,
        page by page, adjacent pieces of one (block, range) merged."""
        t = self._t
        per_stage = (isinstance(kv_pages, (list, tuple)) and len(kv_pages) > 0
                     and isinstance(kv_pages[0], (list, tuple, np.ndarray)))
        if per_stage and len(kv_pages) == 1:
            kv_pages, per_stage = kv_pages[0], False
        nb = b_hi - b_lo
        j = np.arange(b_lo, b_hi, dtype=np.int64)[:, None]                  # [nb, 1]
        idx = j * BLOCK_PAGES + t["page"][None, :]                           # [nb, K]: index into the page list
        r = np.broadcast_to(t["range"], idx.shape)
        if per_stage:
            lists = [np.asarray(x, dtype=np.int64) for x in kv_pages]
            lens = np.array([len(x) for x in lists], dtype=np.int64)
            st = self.kv_stage[r]
            valid = idx < lens[st]
            pg = np.zeros(idx.shape, dtype=np.int64)
            for s, arr in enumerate(lists):
                sel = valid & (st == s)
                pg[sel] = arr[idx[sel]]
        else:
            arr = np.asarray(kv_pages, dtype=np.int64)
            valid = idx < len(arr)
            pg = np.where(valid, arr[np.minimum(idx, max(len(arr) - 1, 0))] if len(arr) else 0, 0)
        sl = np.asarray(slots[b_lo - block_start:b_hi - block_start], dtype=np.int64)[:, None]
        dev = self._base[r] + (pg // BLOCK_PAGES) * self._stride[r] + (pg % BLOCK_PAGES) * self._pb[r] + t["off"]
        host = sl * self.slot_bytes + t["host"]
        ln = np.broadcast_to(t["len"], idx.shape)
        group = j * len(self.kv_len) + r
        dev, host, ln, group, r = (x[valid] for x in (dev, np.broadcast_to(host, idx.shape), ln, group, r))
        if len(ln) > 1:                                  # merge pieces adjacent on both sides within (block, range)
            cut = np.flatnonzero(np.r_[True, (group[1:] != group[:-1]) | (dev[1:] != dev[:-1] + ln[:-1])
                                       | (host[1:] != host[:-1] + ln[:-1])])
            dev, host, ln, r = dev[cut], host[cut], np.add.reduceat(ln, cut), r[cut]
        return dev, host, ln, self.kv_stage[r]

    def pieces(self, kv_pages, state_blocks, slots: list[int], state_slots: list[int], b_lo: int, b_hi: int,
               block_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(device address, host offset, length, stage) of every piece of the context's blocks [b_lo, b_hi) and the
        state: block j's rows at its HBM pages kv_pages[j * BLOCK_PAGES:(j + 1) * BLOCK_PAGES] (those that exist),
        its host side in slot slots[j - block_start] (offsets are slot * slot_bytes + offset within the slot, from the
        start of the tier's arena or file). `kv_pages` / `state_blocks`: one list for every stage (flat, or a list of
        one list), or one list per stage (n lists). `stage`: the stage whose views a piece is in."""
        devs, hosts, lens, stages = [], [], [], []
        if b_hi > b_lo:
            for a, x in zip((devs, hosts, lens, stages), self._kv_pieces(kv_pages, slots, b_lo, b_hi, block_start)):
                a.append(x)
        if len(state_slots):
            st = _shared(state_blocks)
            sb = st[self.st_group] if st.ndim == 1 else st[self.st_stage, self.st_group]
            ss = np.asarray(state_slots, dtype=np.int64)[self.st_slot]
            devs.append(self.st_dev + sb * self.st_stride)
            hosts.append(ss * self.slot_bytes + self.st_off)
            lens.append(self.st_len)
            stages.append(self.st_stage)
        if not devs:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z, z
        return np.concatenate(devs), np.concatenate(hosts), np.concatenate(lens), np.concatenate(stages)
