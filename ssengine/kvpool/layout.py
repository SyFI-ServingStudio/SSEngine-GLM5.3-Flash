"""Layer 1 of the KV pool (dev:research/TIERS.md §13.2): which bytes a KV block or a context's state owns on one stage.

Every byte range a context owns in HBM is one logical block's row of one cache view on one stage:
- KV ranges: for each DSA layer, the MLA latent pages and the indexer K pages of one logical block (`kind="kv"`);
- state ranges: for each KDA layer, its conv state and recurrent state rows of one state block, and for each DSA
  layer its kpool tail row (`kind="state"`, `group` = which of the request's 6 state blocks backs it).
Nothing here knows tiers, slots or GPUs: `placement.py` decides which share of each range a GPU moves and where it
sits on the host.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewRange:
    stage: int
    name: str
    kind: str                 # "kv" or "state"
    base: int                 # device address of logical block 0's row
    stride: int               # bytes from one logical block's row to the next
    nbytes: int               # bytes of one logical block's row
    group: int = -1           # state: index into the request's state_blocks


def from_stage_kv(stage: int, ctx: dict, kv) -> list[ViewRange]:
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
