"""CPU test of StageMetadata's packed staging (ssengine/model/metadata.py `Pack`): every view equals the array it was
built from (dtype, shape, values), segments are aligned and disjoint, empty arrays work."""

import numpy as np
import torch

from ssengine.model.metadata import PACK_ALIGN, Pack


def test_pack_views_round_trip():
    rng = np.random.default_rng(0)
    arrays = [
        (rng.integers(0, 1 << 40, 1027), torch.int64),
        (np.arange(5, dtype=np.int64), torch.int32),
        (rng.integers(0, 1 << 20, (3, 134 * 2)), torch.int32),
        (np.full(7, -1, dtype=np.int64), torch.int64),
        (np.ones(3, dtype=np.bool_), torch.bool),
        ([], torch.int64),
        ([9, 17, 18], torch.int64),
    ]
    pk = Pack()
    hs = [pk.add(a, dt) for a, dt in arrays]
    pk.copy(torch.device("cpu"), pin=False)
    spans = []
    for h, (a, dt) in zip(hs, arrays):
        want = torch.from_numpy(np.asarray(a, dtype=np.int64 if dt != torch.bool else np.bool_)).to(dt)
        for v in (pk.host(h), pk.dev(h)):
            assert v.dtype == dt and v.shape == want.shape and v.is_contiguous()
            assert torch.equal(v, want)
        off = pk.items[h][0]
        assert off % PACK_ALIGN == 0
        spans.append((off, off + pk.items[h][1].nbytes))
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
