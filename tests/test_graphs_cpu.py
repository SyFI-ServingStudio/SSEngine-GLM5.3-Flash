"""CPU tests of the CUDA-graph plumbing (ppimpl/model/stage.py, metadata.py, pp_patch.py): bucket choice, the padded
static rows a captured graph reads, the run-config flag, and the deterministic top-k that makes repeated forwards bitwise equal."""

import numpy as np

from ppimpl.model.metadata import PAD_SLOT_ID, graph_rows
from ppimpl.model.stage import GRAPH_BUCKETS, graph_bucket


def test_bucket_is_smallest_that_holds_the_microbatch():
    assert graph_bucket(1, GRAPH_BUCKETS) == 128
    assert graph_bucket(128, GRAPH_BUCKETS) == 128
    assert graph_bucket(129, GRAPH_BUCKETS) == 256
    assert graph_bucket(1000, GRAPH_BUCKETS) == 1024
    assert graph_bucket(1024, GRAPH_BUCKETS) == 1024
    assert graph_bucket(3000, GRAPH_BUCKETS) == 3072       # not 4096
    assert graph_bucket(2100, GRAPH_BUCKETS) == 2304       # 256-token steps
    assert graph_bucket(7937, GRAPH_BUCKETS) == 8192
    assert len(GRAPH_BUCKETS) == 33
    assert graph_bucket(8192, GRAPH_BUCKETS) == 8192
    assert graph_bucket(8193, GRAPH_BUCKETS) is None        # eager above the largest bucket
    assert graph_bucket(300, {1024: 0, 256: 0}) == 1024      # any container of buckets, unsorted


def test_graph_rows_pad_positions_slots_and_tokens():
    pos = np.array([0, 1, 2, 5, 6], dtype=np.int64)
    slots = np.array([8576, 8577, 8578, 17157, 17158], dtype=np.int64)
    tok = np.array([11, 12, 13, 14, 15], dtype=np.int64)
    r = graph_rows(pos, slots, tok, 8)
    assert r.dtype == np.int64 and r.shape == (3, 8)
    assert r[0].tolist() == [0, 1, 2, 5, 6, 0, 0, 0]
    assert r[1].tolist() == [8576, 8577, 8578, 17157, 17158] + [PAD_SLOT_ID] * 3   # padded rows write no KV
    assert r[2].tolist() == [11, 12, 13, 14, 15, 0, 0, 0]
    r2 = graph_rows(pos, slots, None, 5)          # stages > 0: no token ids; exact bucket: no padding
    assert r2.shape == (2, 5) and (r2[1] == slots).all()


def test_run_config_flag():
    from ppimpl.config import RunConfig
    from ppimpl.model.stage import bench_config, stage_settings

    assert stage_settings(RunConfig(model="glm53"))["cuda_graphs"] is True
    assert stage_settings(RunConfig(model="glm53", cuda_graphs=False))["cuda_graphs"] is False
    assert stage_settings(bench_config(8192))["cuda_graphs"] is False          # benches run eagerly unless asked
    assert stage_settings(bench_config(8192, cuda_graphs=True))["cuda_graphs"] is True


def test_topk_canonical_set_and_order():
    import torch

    from ppimpl.model.topk_canon import canonical_topk_torch, canonicalize

    # row 0: keys 0..5, k=3, a tie at the 3rd value (cols 1, 3, 5 all 2.0): the lowest index wins, whatever the
    # kernel picked. row 1: keys start at col 2 (indices relative to it), only 2 keys: all of them, then -1.
    # row 2: no tie, the kernel's order scrambled.
    logits = torch.tensor([[9., 2., 7., 2., 0., 2.],
                           [5., 5., 1., 3., 8., 8.],
                           [1., 6., 4., 0., 5., 3.]])
    ks = torch.tensor([0, 2, 0], dtype=torch.int32)
    ke = torch.tensor([6, 4, 6], dtype=torch.int32)
    dst = torch.tensor([[2, 5, 0], [1, 0, -1], [4, 1, 2]], dtype=torch.int32)
    canonical_topk_torch(logits, ks, ke, dst, 3, block_elems=6)    # one row per block
    assert dst.tolist() == [[0, 1, 2], [0, 1, -1], [1, 2, 4]]
    dst2 = torch.tensor([[3, 2, 0], [0, 1, -1], [2, 4, 1]], dtype=torch.int32)
    canonicalize(logits, ks, ke, dst2, 3)                          # another valid kernel answer, one block
    assert dst2.tolist() == dst.tolist()
