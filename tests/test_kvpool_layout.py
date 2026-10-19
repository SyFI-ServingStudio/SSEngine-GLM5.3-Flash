"""Which bytes a block or a state owns (ssengine/kvpool/layout.py) on CPU: fake layer objects shaped like one stage's
(MLA pages of 2 kernel blocks per logical block, indexer pages of 1, a tail row and KDA (conv, recurrent) rows per
state block). The real StageKV's ranges are checked by tests/test_kvpool_bytes_gpu.py."""

from types import SimpleNamespace as NS

import pytest
import torch

from ssengine.kvpool.layout import from_stage_kv

NB = 5


def stage_kv():
    ctx = {"mla0": NS(kv_cache=[torch.zeros(NB * 2, 64, 48, dtype=torch.uint8)]),
           "idx0": NS(kv_cache=torch.zeros(NB, 16, 32, dtype=torch.uint8)),
           "tail0": NS(kv_cache=torch.zeros(NB, 2, 4, 8, dtype=torch.bfloat16)),
           "kda0": NS(kv_cache=(torch.zeros(NB, 3, 64, dtype=torch.bfloat16), torch.zeros(NB, 4, 32, 32))),
           "kda1": NS(kv_cache=(torch.zeros(NB, 3, 64, dtype=torch.bfloat16), torch.zeros(NB, 4, 32, 32)))}
    kv = NS(num_blocks=NB, mla_names=["mla0"], indexer_names=["idx0"], tail_names=["tail0"],
            kda_groups=[["kda0"], [], ["kda1"], [], []])
    return ctx, kv


def test_ranges_of_one_stage():
    ctx, kv = stage_kv()
    got = {r.name: r for r in from_stage_kv(2, ctx, kv)}
    assert set(got) == {"mla0", "idx0", "tail0", "kda0#0", "kda0#1", "kda1#0", "kda1#1"}
    assert all(r.stage == 2 for r in got.values())
    mla = got["mla0"]                                   # 2 rows per logical block, copied whole
    assert (mla.kind, mla.stride, mla.nbytes) == ("kv", 2 * 64 * 48, 2 * 64 * 48)
    assert mla.base == ctx["mla0"].kv_cache[0].data_ptr()
    assert (got["idx0"].kind, got["idx0"].nbytes) == ("kv", 16 * 32)
    assert (got["tail0"].kind, got["tail0"].group, got["tail0"].nbytes) == ("state", 5, 2 * 4 * 8 * 2)
    assert (got["kda1#1"].group, got["kda1#1"].nbytes) == (2, 4 * 32 * 32 * 4)
    assert got["kda0#0"].base == ctx["kda0"].kv_cache[0].data_ptr() and got["kda0#0"].group == 0


def test_non_contiguous_rows_are_refused():
    ctx, kv = stage_kv()
    ctx["idx0"] = NS(kv_cache=torch.zeros(16, NB, 32, dtype=torch.uint8).transpose(0, 1))
    with pytest.raises(ValueError, match="not contiguous"):
        from_stage_kv(0, ctx, kv)
