"""kpool insert with chunks that start inside a pool, on the fork's real Triton ops (1 GPU): the indexer K cache bytes
(fp8 pools and their scales) written by any chunking equal the unsplit prefill's, bitwise. tests/test_kpool_cpu.py
has the same cases on torch reference ops."""
from __future__ import annotations

import types

import numpy as np
import pytest

from ppimpl.model import pp_patch

pytestmark = pytest.mark.gpu
KPOOL, HD, PAGE = 4, 128, 64


def ops():
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kops

    orig = (pp_patch._ORIG_KPOOL or {}).get("seed", kops.kpool_seed_tail_cache)
    return types.SimpleNamespace(kpool_compress_and_write_cache=kops.kpool_compress_and_write_cache), orig


def run(schedule, lens, override=True):
    import torch

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    k_all = [torch.randn(L, HD, generator=gen, device=dev).bfloat16() for L in lens]
    g_all = [torch.randn(L, HD, generator=gen, device=dev).bfloat16() for L in lens]
    ape = torch.randn(KPOOL, HD, generator=gen, device=dev)
    pools_per_req = max(lens) // KPOOL + PAGE
    cache = torch.zeros(-(-pools_per_req * len(lens) // PAGE), PAGE, HD + 4, dtype=torch.uint8, device=dev)
    tail = torch.zeros(len(lens), 2, KPOOL, HD, dtype=torch.bfloat16, device=dev)
    o, seed = ops()
    for mb in schedule:
        pos = np.concatenate([np.arange(s, s + n) for _, s, n in mb])
        req = np.concatenate([np.full(n, r) for r, _, n in mb])
        slot = np.where(pos % KPOOL == KPOOL - 1, req * pools_per_req + pos // KPOOL, -1)
        k = torch.stack([k_all[r][p] for r, p in zip(req, pos)])
        g = torch.stack([g_all[r][p] for r, p in zip(req, pos)])
        sm = torch.from_numpy(slot).to(dev)
        pp_patch.KPOOL_HEADS.clear()
        if override:
            q_lens = np.array([n for _, _, n in mb])
            h = pp_patch.kpool_heads(np.array([s for _, s, _ in mb]), q_lens,
                                     np.concatenate([[0], np.cumsum(q_lens)[:-1]]), np.array([r for r, _, _ in mb]),
                                     0, 0, KPOOL)
            if h is not None:
                pp_patch.KPOOL_HEADS[sm.data_ptr()] = pp_patch.KpoolHeads(*(torch.from_numpy(a).to(dev) for a in h))
        pp_patch.kpool_compress_insert(k, g, ape, cache, sm, KPOOL, HD, True, ops=o)
        pp_patch.kpool_seed_tail_cache(tail, k, g, torch.from_numpy(req * KPOOL + pos % KPOOL).to(dev), KPOOL, HD,
                                       ops=o, seed=seed)
        assert not pp_patch._KPOOL_PENDING
    torch.cuda.synchronize()
    return cache.cpu()


def whole(lens):
    return run([[(r, 0, L)] for r, L in enumerate(lens)], lens)


@pytest.mark.parametrize("hit", [8193 + 1, 8193 + 2, 8193 + 3])
def test_unaligned_resume_bitwise_on_gpu(hit):
    L = hit + 9000
    sched = [[(0, 0, 8192)], [(0, 8192, hit - 8192)]]
    s = hit
    while s < L:
        n = min(8192, L - s)
        sched.append([(0, s, n)])
        s += n
    import torch

    ref = whole([L])
    assert torch.equal(run(sched, [L]), ref)
    assert not torch.equal(run(sched, [L], override=False), ref)


def test_two_requests_and_short_chunks_bitwise_on_gpu():
    import torch

    lens = [24, 30]
    sched = [[(1, 0, 13)], [(1, 13, 3)], [(1, 16, 14), (0, 0, 24)]]
    assert torch.equal(run(sched, lens), whole(lens))
    sched = [[(0, 0, 8001)], [(0, 8001, 8190)], [(0, 16191, 3)], [(0, 16194, 7806)]]
    assert torch.equal(run(sched, [24000]), whole([24000]))
