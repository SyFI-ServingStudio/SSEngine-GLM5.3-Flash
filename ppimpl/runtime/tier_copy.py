"""Which bytes a tier op moves on one GPU: balanced storage by quarters (lanes/m6 TIERS.md §3).

Every byte range a context owns in HBM is one logical block's row of one cache view on one stage:
- KV ranges: for each DSA layer, the MLA latent pages and the indexer K pages of one logical block (`kind="kv"`);
- state ranges: for each KDA layer, its conv state and recurrent state rows of one state block, and for each DSA
  layer its kpool tail row (`kind="state"`, `group` = which of the request's 6 state blocks backs it).
GPU q of `parts` moves quarter q of every range: bytes [q n / parts, (q + 1) n / parts). Its host side (a DRAM or
staging slot, an SSD slot at the same offset in that GPU's file) packs its quarters of one block's KV ranges of all
stages into one slot (page_first), and of the context's state ranges into `state_slots` slots. The slot ids come
from the controller's index (`ppimpl/engine/tiers.py`) and are the same on every GPU.

`Layout` is built once per worker from the four stages' ranges (device addresses valid in this worker's process:
its own and the other stages' KV, opened by CUDA IPC); `Layout.pieces` turns one op into (device address, host
offset, length) arrays for one `cuMemcpyBatchAsync` (vLLM `ops.swap_blocks_batch`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALIGN = 4096                  # O_DIRECT: slot sizes and file offsets are multiples of this
PIECE_ALIGN = 16              # every piece starts 16-byte aligned on both sides


@dataclass(frozen=True)
class ViewRange:
    stage: int
    name: str
    kind: str                 # "kv" or "state"
    base: int                 # device address of logical block 0's row
    stride: int               # bytes from one logical block's row to the next
    nbytes: int               # bytes of one logical block's row
    group: int = -1           # state: index into the request's state_blocks


def _round(n: int, a: int) -> int:
    return -(-n // a) * a


class Layout:
    """GPU q's pieces of every range and where they sit in the host slots."""

    def __init__(self, ranges: list[ViewRange], q: int, parts: int = 4, slot_bytes: int | None = None):
        self.q, self.parts = q, parts
        kv = sorted((r for r in ranges if r.kind == "kv"), key=lambda r: (r.stage, r.name))
        st = sorted((r for r in ranges if r.kind == "state"), key=lambda r: (r.stage, r.name))
        for r in kv + st:
            if r.nbytes % (parts * PIECE_ALIGN):
                raise ValueError(f"{r.name} (stage {r.stage}): {r.nbytes} B per block is not a multiple of "
                                 f"{parts} x {PIECE_ALIGN}")
        self.kv_slot_bytes = _round(sum(r.nbytes // parts for r in kv), ALIGN)
        self.state_bytes = sum(r.nbytes // parts for r in st)
        self.slot_bytes = slot_bytes or self.kv_slot_bytes
        assert self.slot_bytes >= self.kv_slot_bytes and self.slot_bytes % ALIGN == 0
        self.state_slots = -(-self.state_bytes // self.slot_bytes)
        # KV: one piece per range per block, packed in range order
        n = np.array([r.nbytes // parts for r in kv], dtype=np.int64)
        self.kv_dev = np.array([r.base for r in kv], dtype=np.int64) + q * n
        self.kv_stride = np.array([r.stride for r in kv], dtype=np.int64)
        self.kv_len = n
        self.kv_host = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
        # state: pieces packed over state_slots slots, split where they cross a slot boundary
        dev, stride, group, host, length = [], [], [], [], []
        off = 0
        for r in st:
            m = r.nbytes // parts
            d = r.base + q * m
            while m:
                take = min(m, self.slot_bytes - off % self.slot_bytes)
                dev.append(d)
                stride.append(r.stride)
                group.append(r.group)
                host.append(off)
                length.append(take)
                d, off, m = d + take, off + take, m - take
        self.st_dev = np.array(dev, dtype=np.int64)
        self.st_stride = np.array(stride, dtype=np.int64)
        self.st_group = np.array(group, dtype=np.int64)
        self.st_slot = np.array(host, dtype=np.int64) // self.slot_bytes
        self.st_off = np.array(host, dtype=np.int64) % self.slot_bytes
        self.st_len = np.array(length, dtype=np.int64)

    def pieces(self, kv_blocks: list[int], state_blocks: list[int], slots: list[int], state_slots: list[int],
               b_lo: int, b_hi: int, block_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(device address, host offset, length) of every piece of blocks [b_lo, b_hi) and the state: block j's
        rows at HBM block kv_blocks[j], its host side in slot slots[j - block_start] (offsets are slot * slot_bytes +
        offset within the slot, from the start of the tier's arena or file)."""
        devs, hosts, lens = [], [], []
        if b_hi > b_lo:
            hb = np.asarray(kv_blocks[b_lo:b_hi], dtype=np.int64)[:, None]
            sl = np.asarray(slots[b_lo - block_start:b_hi - block_start], dtype=np.int64)[:, None]
            devs.append((self.kv_dev + hb * self.kv_stride).ravel())
            hosts.append((sl * self.slot_bytes + self.kv_host).ravel())
            lens.append(np.broadcast_to(self.kv_len, (b_hi - b_lo, len(self.kv_len))).ravel())
        if len(state_slots):
            sb = np.asarray(state_blocks, dtype=np.int64)[self.st_group]
            ss = np.asarray(state_slots, dtype=np.int64)[self.st_slot]
            devs.append(self.st_dev + sb * self.st_stride)
            hosts.append(ss * self.slot_bytes + self.st_off)
            lens.append(self.st_len)
        if not devs:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z
        return np.concatenate(devs), np.concatenate(hosts), np.concatenate(lens)

    def slot_pieces(self, slots: list[int], state_slots: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(offset, length) of whole slots: the host <-> SSD side, one piece per slot."""
        s = np.asarray(list(slots) + list(state_slots), dtype=np.int64)
        return s * self.slot_bytes, np.full(len(s), self.slot_bytes, dtype=np.int64)


def op_blocks(op, block_tokens: int) -> tuple[int, int]:
    """The KV blocks a write, demote or DRAM read moves: from the block holding token_start to the last one."""
    return op.token_start // block_tokens, -(-op.token_end // block_tokens)


def stage_ranges(stage: int, ctx: dict, kv) -> list[ViewRange]:
    """This stage's ranges, from its vLLM layer objects (`static_forward_context`) and `kv_tensors.StageKV`: every
    view's dim 0 is the cache's block (or kernel block) index; a logical block is `rows` consecutive dim-0 rows,
    copied whole when the rows are contiguous (MLA, indexer pages), else row by row is not supported (state views
    hold one row per state block)."""
    import torch

    def view(name):
        v = ctx[name].kv_cache
        return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v

    def rng(t: torch.Tensor, name: str, kind: str, group: int = -1, blocks: int | None = None) -> ViewRange:
        rows = t.shape[0] // blocks if blocks else 1
        row = t[0]
        if not row.is_contiguous():
            raise ValueError(f"{name}: a block row is not contiguous ({tuple(t.shape)}, {t.stride()})")
        row_bytes = row.numel() * t.element_size()
        stride = t.stride(0) * t.element_size()
        if rows > 1 and stride != row_bytes:
            raise ValueError(f"{name}: {rows} rows per block are not contiguous")
        return ViewRange(stage, name, kind, t.data_ptr(), stride * rows, row_bytes * rows, group)

    out = []
    nb = kv.num_blocks
    for name in kv.mla_names + kv.indexer_names:
        out.append(rng(view(name), name, "kv", blocks=nb))
    for name in kv.tail_names:
        out.append(rng(view(name), name, "state", group=5))
    for g, names in enumerate(kv.kda_groups):
        for name in names:
            for i, t in enumerate(view(name)):          # (conv, recurrent)
                out.append(rng(t, f"{name}#{i}", "state", group=g))
    return out
