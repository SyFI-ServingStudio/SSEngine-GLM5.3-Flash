"""`kda_conv.conv_qkv` (at its default blocks, as served) vs the per-sequence Dao causal-conv1d calls it replaces, and
`kda_state` vs torch, on 1 GPU: q, k, v and the whole conv-state pool bitwise equal, on one 4,096-row sequence (fresh
and continuing) and on multi-sequence microbatches where every or most sequences continue a conv state, including
sequences shorter than the conv width; state gather / scatter bitwise equal."""
from __future__ import annotations

import pytest

from bench.kda_conv_bench import CASES, P, STATE_CASES, dao_loop, paged, pool_clone

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("name", sorted(CASES))
def test_conv_qkv_bitwise(name):
    import torch

    from ppimpl.model import kda_conv

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1)
    C = 3 * P
    lens = CASES[name]
    T = sum(n for n, _ in lens)
    w = (torch.randn(C, 4, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    pool0 = paged(torch.randn(32, 3 * C + 256, generator=g, device=dev).to(torch.bfloat16), (32, 3, C))
    x = torch.randn(T, C + 4096, generator=g, device=dev).to(torch.bfloat16)   # q|k|v in front of more columns
    seqs, at = [], 0
    for i, (n, init) in enumerate(lens):
        seqs.append((at, at + n, 30 - 3 * i, init))
        at += n
    ref = pool_clone(pool0)
    rq, rk, rv = dao_loop(x, w, ref, seqs)
    pool = pool_clone(pool0)
    tabs = tuple(torch.from_numpy(a).to(dev) for a in kda_conv.tables(seqs))
    q, k, v = kda_conv.conv_qkv(x, w, pool, tabs[0], tabs[1], P)
    for a, b in ((q, rq), (k, rk), (v, rv)):
        assert a.is_contiguous() and torch.equal(a, b)
    assert torch.equal(pool, ref)


@pytest.mark.parametrize("name", sorted(STATE_CASES))
def test_kda_state_gather_scatter_bitwise(name):
    """`kda_state.gather` / `scatter` vs torch's index_select (+ zero_ for fresh requests) / index_copy_, on a pool
    paged as vLLM binds it."""
    import torch

    from ppimpl.model import kda_state

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(2)
    inits = STATE_CASES[name]
    pool0 = paged(torch.randn(40, 64 * 128 * 128 + 1024, generator=g, device=dev), (40, 64, 128, 128))
    slots = torch.tensor([37 - 5 * i for i in range(len(inits))], dtype=torch.int64, device=dev)
    ref = pool0.index_select(0, slots)
    for i, h in enumerate(inits):
        if not h:
            ref[i].zero_()
    got = kda_state.gather(pool0, slots, torch.tensor(inits, dtype=torch.int32, device=dev))
    assert torch.equal(got, ref)
    new = torch.randn(len(inits), 64, 128, 128, generator=g, device=dev)
    a, b = pool_clone(pool0), pool_clone(pool0)
    a.index_copy_(0, slots, new)
    kda_state.scatter(new, slots, b)
    assert torch.equal(a, b)
