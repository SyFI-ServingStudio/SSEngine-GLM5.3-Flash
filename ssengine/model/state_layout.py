"""Compact KDA state layout (docs/design/kept_states.md, "Compact state allocation").

A state is R runs of c pages, each run the first c pages of a block (the rest of the block is ordinary KV pages), and
a state is named by its runs' block ids, the same on every stage. On a stage, the first c pages of a block are one
contiguous region of c x page_bytes in each of its KV tensors (MLA and indexer of each DSA layer: a block's pages are
contiguous whatever vLLM's block stride), and every state piece (each KDA layer's conv and recurrent state, each DSA
layer's kpool tail) has a fixed (run, region, byte offset). The piece's view is an `as_strided` over its region with
one row per block (row stride = the region's block stride): row b is the piece of the state whose run is block b.

`geometry` derives (c, R) and the placements from all stages' sizes, so every stage computes the same; the workers
report (c, R) in `Ready`. `StateViews` binds a stage's views and copies states (copy-on-resume); pure torch, CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ssengine.config import BLOCK_PAGES

ALIGN = 16                    # every piece starts 16-byte aligned (the tiers' pieces too: placement.PIECE_ALIGN)


@dataclass(frozen=True)
class StagePieces:
    """One stage's KV regions and state pieces, in bytes."""
    regions: tuple[tuple[str, int], ...]             # (region name, bytes per page): every KV tensor of the stage
    kda: tuple[tuple[str, int, int], ...]            # (KDA layer, conv bytes, recurrent bytes), layer order
    tails: tuple[tuple[str, int], ...]               # (DSA layer's tail name, bytes)


@dataclass(frozen=True)
class Place:
    run: int
    region: str
    offset: int               # bytes from the run's first page in the region


@dataclass
class Geometry:
    run_pages: int            # c
    runs: int                 # R
    places: list[dict[str, Place]] = field(default_factory=list)   # per stage: piece name -> place

    @property
    def pages(self) -> int:
        return self.run_pages * self.runs


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _pack(st: StagePieces, c: int) -> tuple[int, dict[str, Place]] | None:
    """Runs a stage needs at run size c, and where each piece goes; None if a piece fits no region."""
    cap = {name: c * pb for name, pb in st.regions}
    runs: list[dict[str, int]] = []                  # per run: region -> bytes used
    places: dict[str, Place] = {}

    def put(run: int, size: int) -> str | None:
        """Best fit: the region of the run with the least room left that holds `size` (first in order on ties)."""
        used = runs[run]
        room = [(cap[name] - _align(used[name]), i, name) for i, (name, _) in enumerate(st.regions)]
        fits = [x for x in room if x[0] >= size]
        return min(fits)[2] if fits else None

    def new_run():
        runs.append({name: 0 for name, _ in st.regions})

    def take(run: int, name: str, size: int) -> Place:
        off = _align(runs[run][name])
        runs[run][name] = off + size
        return Place(run, name, off)

    if st.tails:
        new_run()
        for name, size in st.tails:                  # one run holds every tail (one tail block table)
            r = put(0, size)
            if r is None:
                return None
            places[name] = take(0, r, size)
    def fit(run: int, name: str, conv: int, rec: int) -> bool:
        """Recurrent (the larger) first, then conv, both in `run`; nothing taken if they do not both fit."""
        r_rec = put(run, rec)
        if r_rec is None:
            return False
        saved = dict(runs[run])
        p_rec = take(run, r_rec, rec)
        r_conv = put(run, conv)
        if r_conv is None:
            runs[run] = saved
            return False
        places[f"{name}#0"] = take(run, r_conv, conv)
        places[f"{name}#1"] = p_rec
        return True

    for name, conv, rec in st.kda:                   # conv and recurrent in the same run (one sequence table)
        if any(fit(run, name, conv, rec) for run in range(len(runs))):
            continue
        new_run()
        if not fit(len(runs) - 1, name, conv, rec):
            return None                              # not even an empty run holds the layer
    return len(runs), places


def geometry(stages: list[StagePieces]) -> Geometry:
    """The run size c in 1..BLOCK_PAGES with the fewest pages per state (c x max runs over stages; ties: fewer runs)
    and every stage's placements."""
    best = None
    for c in range(1, BLOCK_PAGES + 1):
        packed = [_pack(st, c) for st in stages]
        if any(p is None for p in packed):
            continue
        R = max(p[0] for p in packed)
        key = (c * R, R)
        if best is None or key < best[0]:
            best = (key, Geometry(c, R, [p[1] for p in packed]))
    if best is None:
        raise ValueError("a KDA state piece fits no run of a block's pages")
    return best[1]


# --- this model's sizes without vLLM (the fake model and the KV manager's default) --------------------------------

def glm53_pieces(split=None) -> list[StagePieces]:
    """GLM-5.3-Flash per stage, in the sizes vLLM's specs give (tests/test_model_cpu.py checks them against the specs):
    MLA 512 B and indexer 33 B per token per DSA layer, KDA conv bf16 [3, 3 x 64 x 128] and recurrent fp32
    [64, 128, 128], kpool tail bf16 [2, 4, 128]."""
    from ssengine import kernels
    from ssengine.config import PAGE_TOKENS

    split = tuple(split or kernels.SPLIT)
    out = []
    for s in range(len(split)):
        regions, kda, tails = [], [], []
        for i in kernels.stage_layers(s, split):
            if i in kernels.DSA_LAYERS:
                regions += [(f"{i}.mla", 512 * PAGE_TOKENS), (f"{i}.indexer", 33 * PAGE_TOKENS)]
                tails.append((f"{i}.tail", 2 * 4 * 128 * 2))
            else:
                kda.append((f"{i}.kda", 3 * 3 * 64 * 128 * 2, 64 * 128 * 128 * 4))
        out.append(StagePieces(tuple(regions), tuple(kda), tuple(tails)))
    return out


def glm53_geometry(split=None) -> Geometry:
    return geometry(glm53_pieces(split))


# --- one stage's views ----------------------------------------------------------------------------------------------

class StateViews:
    """A stage's state piece views over its KV regions, and the state copy (copy-on-resume).

    `regions`: region name -> (a tensor whose block b starts at byte b x block_stride from the tensor's start (only
    its storage and data_ptr are used), block_stride); `num_blocks`: blocks of the pool (incl. the null block);
    `pieces`: piece name -> (shape, dtype); `places`: this stage's placements."""

    def __init__(self, regions: dict, num_blocks: int, geo: Geometry, places: dict[str, Place], pieces: dict):
        import torch

        self.geo = geo
        self.views: dict[str, torch.Tensor] = {}
        self.run_of: dict[str, int] = {}
        for name, (shape, dtype) in pieces.items():
            p = places[name]
            t, stride = regions[p.region]
            esize = torch.empty(0, dtype=dtype).element_size()
            assert stride % esize == 0 and p.offset % esize == 0, (name, stride, p.offset)
            st = t.untyped_storage()
            off = t.data_ptr() - st.data_ptr()
            assert off % esize == 0 and off + num_blocks * stride <= st.nbytes() + stride, (name, off, st.nbytes())
            n_el = (st.nbytes() - off) // esize
            flat = torch.empty(0, dtype=dtype, device=t.device).set_(st, off // esize, (n_el,), (1,))
            inner = [1] * len(shape)
            for d in range(len(shape) - 2, -1, -1):
                inner[d] = inner[d + 1] * shape[d + 1]
            self.views[name] = torch.as_strided(flat, (num_blocks, *shape), (stride // esize, *inner),
                                                off // esize + p.offset // esize)
            self.run_of[name] = p.run
        self._by_run = [[n for n in self.views if self.run_of[n] == r] for r in range(geo.runs)]

    def copy(self, pairs: list[tuple[list[int], list[int]]]) -> None:
        """For each (src runs, dst runs) (block ids): every piece of the state at src into the state at dst."""
        import torch

        dst, src = [], []
        for s_runs, d_runs in pairs:
            for r, names in enumerate(self._by_run):
                for n in names:
                    v = self.views[n]
                    dst.append(v[d_runs[r]])
                    src.append(v[s_runs[r]])
        if dst:
            torch._foreach_copy_(dst, src)
