"""Layer 2 of the KV pool (lanes/m6 TIERS.md §3, §13.2): balanced storage, 1/n of every range per GPU. Pure numpy.

n is the PP degree (`RunConfig.num_stages`), never fixed: PP4 has n = 4 (quarters), PP8 n = 8.

GPU q of n moves share q of every range (`layout.ViewRange`): bytes [q m / n, (q + 1) m / n) of a range of m bytes.
Its host side (a DRAM or staging slot, an SSD slot at the same offset in that GPU's file) packs its shares of one
block's KV ranges of all stages into one slot (page_first), and of the context's state ranges into `state_slots`
slots. The slot ids come from the leader's index (`manager.py`) and are the same on every GPU.

`BalancedPlacement` is built once per member from all stages' ranges (device addresses valid in that process: its
own and the other stages' KV, opened by CUDA IPC); `pieces` turns one op into (device address, host offset, length)
arrays for `mover.Mover`. Own-stage placement (every GPU stores only its own stage, TIERS.md §2.4) is not built: it
would need per-GPU slot sizes in the index.
"""

from __future__ import annotations

import numpy as np

from ppimpl.kvpool.layout import ViewRange

ALIGN = 4096                  # O_DIRECT: slot sizes and file offsets are multiples of this
PIECE_ALIGN = 16              # every piece starts 16-byte aligned on both sides


def _round(n: int, a: int) -> int:
    return -(-n // a) * a


def _shared(blocks) -> np.ndarray:
    a = np.asarray(blocks, dtype=np.int64)
    return a[0] if a.ndim == 2 and len(a) == 1 else a


class BalancedPlacement:
    """GPU q's pieces of every range and where they sit in the host slots."""

    def __init__(self, ranges: list[ViewRange], n: int, q: int, slot_bytes: int | None = None):
        self.n, self.q = n, q
        kv = sorted((r for r in ranges if r.kind == "kv"), key=lambda r: (r.stage, r.name))
        st = sorted((r for r in ranges if r.kind == "state"), key=lambda r: (r.stage, r.name))
        for r in kv + st:
            if r.nbytes % (n * PIECE_ALIGN):
                raise ValueError(f"{r.name} (stage {r.stage}): {r.nbytes} B per block is not a multiple of "
                                 f"{n} x {PIECE_ALIGN}")
        self.kv_slot_bytes = _round(sum(r.nbytes // n for r in kv), ALIGN)
        self.state_bytes = sum(r.nbytes // n for r in st)
        self.slot_bytes = slot_bytes or self.kv_slot_bytes
        assert self.slot_bytes >= self.kv_slot_bytes and self.slot_bytes % ALIGN == 0
        self.state_slots = -(-self.state_bytes // self.slot_bytes)
        # KV: one piece per range per block, packed in range order
        m = np.array([r.nbytes // n for r in kv], dtype=np.int64)
        self.kv_stage = np.array([r.stage for r in kv], dtype=np.int64)
        self.kv_dev = np.array([r.base for r in kv], dtype=np.int64) + q * m
        self.kv_stride = np.array([r.stride for r in kv], dtype=np.int64)
        self.kv_len = m
        self.kv_host = np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int64)
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

    def pieces(self, kv_blocks, state_blocks, slots: list[int], state_slots: list[int], b_lo: int, b_hi: int,
               block_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(device address, host offset, length) of every piece of blocks [b_lo, b_hi) and the state: block j's
        rows at HBM block kv_blocks[j], its host side in slot slots[j - block_start] (offsets are slot * slot_bytes +
        offset within the slot, from the start of the tier's arena or file). `kv_blocks` / `state_blocks`: one list
        for every stage (flat, or a list of one list), or one list per stage (n lists)."""
        devs, hosts, lens = [], [], []
        if b_hi > b_lo:
            kv = _shared(kv_blocks)
            hb = kv[b_lo:b_hi, None] if kv.ndim == 1 else kv[self.kv_stage, b_lo:b_hi].T
            sl = np.asarray(slots[b_lo - block_start:b_hi - block_start], dtype=np.int64)[:, None]
            devs.append((self.kv_dev + hb * self.kv_stride).ravel())
            hosts.append((sl * self.slot_bytes + self.kv_host).ravel())
            lens.append(np.broadcast_to(self.kv_len, (b_hi - b_lo, len(self.kv_len))).ravel())
        if len(state_slots):
            st = _shared(state_blocks)
            sb = st[self.st_group] if st.ndim == 1 else st[self.st_stage, self.st_group]
            ss = np.asarray(state_slots, dtype=np.int64)[self.st_slot]
            devs.append(self.st_dev + sb * self.st_stride)
            hosts.append(ss * self.slot_bytes + self.st_off)
            lens.append(self.st_len)
        if not devs:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z
        return np.concatenate(devs), np.concatenate(hosts), np.concatenate(lens)

    def local(self, stage: int, blocks: int, state: bool) -> np.ndarray:
        """Which of `pieces`' pieces (`blocks` blocks, then the state if `state`) are in stage `stage`'s KV."""
        return np.r_[np.tile(self.kv_stage == stage, blocks), (self.st_stage == stage) if state else []].astype(bool)
