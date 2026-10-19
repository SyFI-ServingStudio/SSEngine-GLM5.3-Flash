"""GPU test of `ssengine.model.sm_h2d.SmH2D` (kernel copies from mapped pinned memory): bytes equal for every size,
word width and alignment; a source freed by its caller right after the copy is issued is not reused before the
kernel has read it (the copy queued behind a GPU sleep while the host allocates and overwrites pinned buffers)."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_sm_h2d_bytes_equal():
    from ssengine.model.sm_h2d import SmH2D

    dev = torch.device("cuda", 0)
    h = SmH2D(dev)
    h.warmup()
    g = torch.Generator().manual_seed(0)
    for n in (1, 7, 8, 63, 4096, 4099, 1 << 20, (3 << 20) + 5):
        for so in (0, 1, 4, 8):
            for do in (0, 1, 4, 8):
                src = torch.randint(0, 256, (n + so,), dtype=torch.uint8, generator=g).pin_memory()
                dst = torch.zeros(n + do, dtype=torch.uint8, device=dev)
                h.copy_(dst[do:], src[so:])
                torch.cuda.synchronize()
                assert torch.equal(dst[do:].cpu(), src[so:]), (n, so, do)
                assert not dst[:do].any()
    x = torch.arange(1000, dtype=torch.int64).pin_memory()
    assert torch.equal(h.to_device(x).cpu(), x)


def test_sm_h2d_source_kept_until_read():
    from ssengine.model.sm_h2d import SmH2D

    dev = torch.device("cuda", 0)
    h = SmH2D(dev)
    h.warmup()
    # fill torch's pinned block cache first: below, allocations reuse freed blocks (a new cudaHostAlloc may wait
    # for the device, which would end the sleep)
    warm = [torch.empty(1 << 18, dtype=torch.int64).pin_memory() for _ in range(48)]
    del warm
    outs, want = [], []
    torch.cuda._sleep(int(1e9))               # ~0.5 s: every copy below waits behind it
    for i in range(20):
        src = torch.full((1 << 18,), i, dtype=torch.int64).pin_memory()
        outs.append(h.to_device(src))
        want.append(i)
        del src                                 # the caller's last reference
        junk = torch.empty(1 << 18, dtype=torch.int64).pin_memory()   # would take a reused block
        junk.fill_(-1)
    assert h.pending() > 0
    torch.cuda.synchronize()
    for o, w in zip(outs, want):
        assert bool((o == w).all()), w
    assert h.pending() == 0
