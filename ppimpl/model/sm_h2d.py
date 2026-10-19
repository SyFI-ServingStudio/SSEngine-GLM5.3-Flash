"""Host -> device copies of a microbatch's metadata by SMs, not by a copy engine.

A stage's host-to-device copies share the GPU's H2D copy engine with the KV tier reads (`kvpool.mover`: DRAM/SSD ->
HBM batches, tens of MB each, queued back to back). A small copy on the compute stream queued behind them starts when
they are done: job 8367 (nsys, stage 0) had each microbatch's 0.5-2 MB metadata copy (0.04-0.08 ms of copying) start
p50 12-15 / p90 47-58 ms after its stream was free, ~29 s of stage-0 idle per 10 min; with the tier ops as no-ops
(8369) that delay was gone. Throttling or reordering the reads would cost them bandwidth.

Here the copy is a kernel: pinned host memory is mapped into the device's address space (UVA), so a kernel on the
compute stream reads it over PCIe and writes the device buffer. It is ordered on the compute stream like the copy
it replaces, needs no copy engine, and shares only the PCIe link with the reads (a few MB per microbatch against
their ~50 GB/s). `research/bench/h2d_queue.py` measures both against a tier-like read stream.

The host buffer must outlive the kernel. Pinned blocks from torch's caching host allocator are reused as soon as
their tensor is freed when no copy recorded a use of them (only torch's own copies record one), so `SmH2D` keeps a
reference to each source until an event recorded after its kernel has completed.
"""

from __future__ import annotations

import collections

import torch
import triton
import triton.language as tl

BLOCK = 1024          # elements per program (8 KB of int64: 128 threads x four 16-byte loads)


@triton.jit(do_not_specialize=["n"])
def _copy_kernel(src, dst, n, BLOCK: tl.constexpr, MASK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if MASK:
        m = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)
    else:                     # whole blocks: unmasked, so the loads and stores vectorize
        tl.store(dst + offs, tl.load(src + offs))


def _words(t: torch.Tensor) -> torch.Tensor:
    """A flat view of `t`'s bytes in the widest of int64 / int32 / uint8 that its size and address allow."""
    b = t.reshape(-1).view(torch.uint8)
    for dt, w in ((torch.int64, 8), (torch.int32, 4)):
        if b.numel() % w == 0 and b.data_ptr() % w == 0:
            return b.view(dt)
    return b


def kernel_copy_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """dst <- src (both contiguous, same byte count; src on dst's device or pinned host memory) by `_copy_kernel` on
    the current stream: no copy engine. A same-GPU device-to-device copy also goes to a copy engine, where it waits
    behind the tier reads' NVLink scatters (the graph-row copy before each replay, job 8384: p99 11.6 ms)."""
    nbytes = src.numel() * src.element_size()
    assert src.is_contiguous() and dst.is_contiguous() and nbytes == dst.numel() * dst.element_size(), \
        (src.shape, src.dtype, dst.shape, dst.dtype)
    if nbytes == 0:
        return dst
    s, d = _words(src), _words(dst)
    if s.dtype != d.dtype:                     # alignments differ: copy bytes
        s, d = s.view(torch.uint8), d.view(torch.uint8)
    n = s.numel()
    full = n // BLOCK
    with torch.cuda.device(dst.device):
        if full:
            _copy_kernel[(full,)](s, d, n, BLOCK=BLOCK, MASK=False, num_warps=4)
        if n > full * BLOCK:
            k = full * BLOCK
            _copy_kernel[(1,)](s[k:], d[k:], n - k, BLOCK=BLOCK, MASK=True, num_warps=4)
    return dst


class SmH2D:
    """Kernel copies from pinned host tensors to device tensors on the current stream of `device`."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._pending: collections.deque = collections.deque()    # (event, refs) in launch order
        self._free: list = []                                     # completed events, for reuse

    def _release(self) -> None:
        while self._pending and self._pending[0][0].query():
            ev, _ = self._pending.popleft()
            self._free.append(ev)

    def copy_(self, dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """dst (device, contiguous) <- src (pinned host, contiguous, same byte count), ordered on the current stream.
        `src` is kept alive until the copy has run; like a non-blocking copy, the caller must not change it before."""
        assert src.is_pinned() and src.is_contiguous() and dst.is_contiguous() and dst.device == self.device, \
            (src.is_pinned(), src.is_contiguous(), dst.is_contiguous(), dst.device)
        kernel_copy_(dst, src)
        with torch.cuda.device(self.device):
            self._release()
            ev = self._free.pop() if self._free else torch.cuda.Event()
            ev.record()
        self._pending.append((ev, src))
        return dst

    def to_device(self, src: torch.Tensor) -> torch.Tensor:
        """A new device tensor with `src`'s contents (pinned host, contiguous)."""
        dst = torch.empty(src.shape, dtype=src.dtype, device=self.device)
        return self.copy_(dst, src)

    def warmup(self) -> None:
        """Compile every kernel variant (word width x source / destination 16-byte alignment, which Triton
        specializes on) now, not in the middle of a run."""
        src = torch.zeros(BLOCK * 8 + 16, dtype=torch.uint8).pin_memory()
        dst = torch.empty(BLOCK * 8 + 16, dtype=torch.uint8, device=self.device)
        for w in (8, 4, 1):
            for so in (0, w if w < 16 else 8):
                for do in (0, w if w < 16 else 8):
                    for n in (32 - (0 if w > 1 else 1), BLOCK * w):
                        self.copy_(dst[do:do + n], src[so:so + n])
        torch.cuda.synchronize(self.device)
        self._release()

    def pending(self) -> int:
        self._release()
        return len(self._pending)


_BY_DEVICE: dict = {}


def for_device(device) -> SmH2D:
    """The process's `SmH2D` of a CUDA device (one per device: its pending sources are released in launch order)."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    h = _BY_DEVICE.get(device)
    if h is None:
        h = _BY_DEVICE[device] = SmH2D(device)
    return h


def async_tensor_h2d(data, device=None, dtype=None, out=None):
    """vLLM's `vllm.utils.torch_utils.async_tensor_h2d` (list / numpy array / tensor -> device, async), the copy
    done by `SmH2D` instead of a copy engine. The source is staged in a pinned buffer of the target dtype unless it
    already is one (as vLLM does; the caller must not change a pinned source before the copy, as with vLLM's)."""
    import numpy as np

    if dtype is None and out is not None:
        dtype = out.dtype
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        t = data
        if not (t.is_pinned() and t.is_contiguous() and (dtype is None or t.dtype == dtype)):
            t = torch.empty(t.shape, dtype=dtype or t.dtype, pin_memory=True).copy_(t)
    else:
        t = torch.tensor(data, dtype=dtype, pin_memory=True)
    if out is not None:
        assert out.dtype == t.dtype and out.is_contiguous(), (out.dtype, t.dtype)
        return for_device(out.device).copy_(out, t)
    assert device is not None, "must provide destination tensor or device"
    return for_device(device).to_device(t)


def install_vllm() -> int:
    """Route vLLM's `async_tensor_h2d` (its metadata builders' host -> device copies, e.g. the MLA chunked-context
    tables, MBs per microbatch with long cached prefixes) through `async_tensor_h2d` here: replaced in
    `vllm.utils.torch_utils` and in every loaded module that imported the name. Returns how many bindings were
    replaced. Call after the model's modules are imported."""
    import sys

    from vllm.utils import torch_utils

    orig = getattr(torch_utils, "_ppimpl_orig_async_tensor_h2d", torch_utils.async_tensor_h2d)
    torch_utils._ppimpl_orig_async_tensor_h2d = orig
    n = 0
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "async_tensor_h2d", None) is orig:
            mod.async_tensor_h2d = async_tensor_h2d
            n += 1
    return n
