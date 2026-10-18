"""pp_patch `install_kda_split_heads`: the C++ head-split search returns what FlashInfer's Python `_split_heads`
returns (span and every slot's pieces, in order), on CPU."""
from __future__ import annotations

import random

import pytest
import torch

fw = pytest.importorskip("flashinfer.kda_kernels.persistent.forward")

from ppimpl.model import kda_schedule  # noqa: E402

H, SMS = 64, 148    # GLM-5.3-Flash KDA heads; B200 SMs (the plan's slots)
PY_SPLIT = fw._split_heads if kda_schedule._ORIG is None else kda_schedule._ORIG


def microbatch(rng: random.Random) -> list[int]:
    """Request lengths of a served-like microbatch: 1-16 chunks, up to 32,768 rows, some chunk-aligned."""
    total = rng.choice([128, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    n = rng.randint(1, 16)
    cuts = sorted(rng.sample(range(1, total), min(n - 1, total - 1)))
    lens = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    return [(x // 32) * 32 or x if rng.random() < 0.2 else x for x in lens]


def search_args(lens: list[int], nslots: int, rng: random.Random):
    chunks = [(lens[c // H] + 31) // 32 for c in range(len(lens) * H)]
    full = [lens[c // H] // 32 for c in range(len(lens) * H)]
    span_limit = rng.choice([float("inf"), 1e9, max(chunks) * rng.uniform(1.0, 3.0), rng.randint(1, 400)])
    return chunks, full, min(len(chunks), nslots), span_limit


def test_split_heads_equals_python_on_random_microbatches():
    rng = random.Random(0)
    found = 0
    for _ in range(1000):
        chunks, full, nslots, limit = search_args(microbatch(rng), rng.choice([SMS, 132, 64, 8]), rng)
        want = PY_SPLIT(chunks, full, nslots, limit)
        got = kda_schedule.split_heads(chunks, full, nslots, limit)
        assert got == want, (len(chunks), nslots, limit)
        found += want is not None
    assert found > 100          # the search found a split often enough to compare schedules, not only None


def test_split_heads_ties():
    """Equal lengths everywhere: every comparison is a tie, decided by the tie-breaking rules."""
    for n_req, length, nslots in [(4, 1024, SMS), (16, 64, SMS), (3, 4096, 64), (8, 320, 100)]:
        lens = [length] * n_req
        chunks = [(x + 31) // 32 for x in lens for _ in range(H)]
        full = [x // 32 for x in lens for _ in range(H)]
        for limit in (float("inf"), 100.5, 50):
            assert kda_schedule.split_heads(chunks, full, min(len(chunks), nslots), limit) == \
                PY_SPLIT(chunks, full, min(len(chunks), nslots), limit)


def test_installed_plan_equals_python_plan(monkeypatch):
    """`_piece_schedule` (what make_plan builds) with the C++ search installed equals the Python one's."""
    rng = random.Random(1)
    cases = [microbatch(rng) for _ in range(200)]
    want = [fw._piece_schedule(lens, H, SMS, torch.device("cpu")) for lens in cases]
    monkeypatch.setattr(fw, "_split_heads", kda_schedule.split_heads)
    splits = 0
    for lens, w in zip(cases, want):
        g = fw._piece_schedule(lens, H, SMS, torch.device("cpu"))
        assert len(g) == len(w)
        for a, b in zip(g, w):
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
        splits += w[7] > 0
    assert splits >= 10


def test_host_plan_equals_flashinfer_make_plan(monkeypatch):
    """pp_patch `make_kda_plan_host` (the schedule StageMetadata stages in its Pack) equals FlashInfer's `make_plan`
    (device tensors) array for array, with the same G, scratch sizes, has_split and full_chunks."""
    from ppimpl.model.pp_patch import make_kda_plan_host

    monkeypatch.setattr(fw, "_sm_count", lambda dev: SMS)
    rng = random.Random(1)
    split = 0
    for _ in range(200):
        lens = microbatch(rng)
        offsets = [0]
        for n in lens:
            offsets.append(offsets[-1] + n)
        want = fw.make_plan(offsets, H, torch.device("cpu"))
        arrays, G, nbuf, has_split, full = make_kda_plan_host(offsets, H, SMS)
        for a, w in zip(arrays, want[:7]):
            assert a.dtype.name == "int32" and a.tolist() == w.tolist()
        assert (G, has_split, full) == (want[7], want[10], want[11])
        assert max(nbuf, 1) == want[8].shape[0] == want[9].shape[0]
        split += has_split
    assert split > 0
