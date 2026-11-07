"""Which bytes a block or a state owns (ssengine/kvpool/layout.py) on CPU: fake layer objects shaped like one stage's
(MLA pages of 2 kernel blocks per logical block, indexer pages of 1) and compact state views over them
(model/state_layout.py: one row per block, a state addressed by its runs' blocks). The real StageKV's ranges are
checked by tests/test_kvpool_bytes_gpu.py."""

from types import SimpleNamespace as NS

import pytest
import torch

from ssengine.config import BLOCK_PAGES
from ssengine.kvpool.layout import from_stage_kv
from ssengine.model.state_layout import Geometry, Place, StateViews

NB = 5


def stage_kv():
    ctx = {"mla0": NS(kv_cache=[torch.zeros(NB * 2, BLOCK_PAGES, 48, dtype=torch.uint8)]),   # 96 B per page
           "idx0": NS(kv_cache=torch.zeros(NB, BLOCK_PAGES, 32, dtype=torch.uint8))}         # 32 B per page
    kv = NS(num_blocks=NB, mla_names=["mla0"], indexer_names=["idx0"], tail_names=["tail0"],
            kda_groups=[["kda0"], ["kda1"]])
    geo = Geometry(16, 2)                               # 2 runs of 16 pages: 1,536 B of mla0, 512 B of idx0 each
    places = {"kda0#1": Place(0, "mla0", 0), "kda0#0": Place(0, "idx0", 0), "tail0": Place(0, "idx0", 384),
              "kda1#1": Place(1, "mla0", 0), "kda1#0": Place(1, "idx0", 0)}
    pieces = {"kda0#0": ((3, 64), torch.bfloat16), "kda0#1": ((4, 8, 8), torch.float32), "tail0": ((2, 4, 8), torch.bfloat16),
              "kda1#0": ((3, 64), torch.bfloat16), "kda1#1": ((4, 8, 8), torch.float32)}
    regions = {"mla0": (ctx["mla0"].kv_cache[0], BLOCK_PAGES * 96), "idx0": (ctx["idx0"].kv_cache, BLOCK_PAGES * 32)}
    kv.state = StateViews(regions, NB, geo, places, pieces)
    return ctx, kv


def test_ranges_of_one_stage():
    ctx, kv = stage_kv()
    got = {r.name: r for r in from_stage_kv(2, ctx, kv)}
    assert set(got) == {"mla0", "idx0", "tail0", "kda0#0", "kda0#1", "kda1#0", "kda1#1"}
    assert all(r.stage == 2 for r in got.values())
    mla = got["mla0"]                                   # 2 rows per logical block, copied whole
    assert (mla.kind, mla.stride, mla.nbytes) == ("kv", 2 * BLOCK_PAGES * 48, 2 * BLOCK_PAGES * 48)
    assert mla.base == ctx["mla0"].kv_cache[0].data_ptr()
    assert (got["idx0"].kind, got["idx0"].nbytes) == ("kv", BLOCK_PAGES * 32)
    tail = got["tail0"]                                 # a state piece: one row per block, at its offset in the run
    assert (tail.kind, tail.group, tail.nbytes, tail.stride) == ("state", 0, 2 * 4 * 8 * 2, BLOCK_PAGES * 32)
    assert tail.base == ctx["idx0"].kv_cache.data_ptr() + 384
    assert (got["kda1#1"].group, got["kda1#1"].nbytes, got["kda1#1"].stride) == (1, 4 * 8 * 8 * 4, BLOCK_PAGES * 96)
    assert got["kda0#0"].base == ctx["idx0"].kv_cache.data_ptr() and got["kda0#0"].group == 0


def test_non_contiguous_rows_are_refused():
    ctx, kv = stage_kv()
    ctx["idx0"] = NS(kv_cache=torch.zeros(16, NB, 32, dtype=torch.uint8).transpose(0, 1))
    with pytest.raises(ValueError, match="not contiguous"):
        from_stage_kv(0, ctx, kv)
