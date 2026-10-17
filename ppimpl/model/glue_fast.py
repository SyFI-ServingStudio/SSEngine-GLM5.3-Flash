"""Speed-only rewrites of small glue in the fork's DSA layers, each bitwise equal to the code it replaces.

- `to_float`: the indexer's `hidden_states.float()` (glm5next `Indexer.forward`, before the FP32 head-weights GEMM).
  torch runs that BF16 -> FP32 cast with its unrolled (non-vectorized) copy kernel: 41 us at 4,096 x 4,096 in the
  serving trace (7808, stage 2, 3 per microbatch, ~2.3 TB/s). Here one Triton kernel: 49.2 -> 20.5 us at 4,096 rows
  alone at 1,965 MHz (job 7914, bench/glue_bench.py).
  BF16 -> FP32 is exact, so the values are the same bits whatever the kernel.

`install_indexer_float` rewrites `Indexer.forward`'s source at that line (checking it is there once: a fork change
fails loudly at start-up) and rebinds the method (process-wide).
"""

from __future__ import annotations

import inspect
import textwrap

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_to_f32_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m).to(tl.float32), mask=m)


CAST_BLOCK = 4096
CAST_WARPS = 8


def to_float(x: torch.Tensor) -> torch.Tensor:
    """`x.float()` for a contiguous BF16 CUDA tensor (anything else: `x.float()` itself)."""
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return x.float()
    y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    if n:
        _bf16_to_f32_kernel[(triton.cdiv(n, CAST_BLOCK),)](x, y, n, BLOCK=CAST_BLOCK, num_warps=CAST_WARPS)
    return y


_INDEXER_EDITS = [("weights = torch.mm(hidden_states.float(), self._wp_fp32)",
                   "weights = torch.mm(_pp_to_float(hidden_states), self._wp_fp32)")]
_ORIG_INDEXER_FORWARD = None


def _rewrite_method(cls, name: str, edits, extra: dict, tag: str):
    """`cls.name` recompiled from its source with `edits` applied, in its module's globals plus `extra`."""
    import sys

    fn = getattr(cls, name)
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    for old, new in edits:
        assert src.count(old) == 1, f"{tag}: the fork's {cls.__name__}.{name} changed:\n{old}"
        src = src.replace(old, new)
    ns = dict(sys.modules[fn.__module__].__dict__) | extra
    exec(compile(src, f"<{tag}:{inspect.getsourcefile(fn)}>", "exec"), ns)
    return fn, ns[name]


def install_indexer_float() -> None:
    global _ORIG_INDEXER_FORWARD
    from vllm.models.glm5next.common import attention

    if _ORIG_INDEXER_FORWARD is not None:
        return
    orig, new = _rewrite_method(attention.Indexer, "forward", _INDEXER_EDITS, {"_pp_to_float": to_float},
                                "glue_fast")
    attention.Indexer.forward = new
    _ORIG_INDEXER_FORWARD = orig


def uninstall_indexer_float() -> None:
    global _ORIG_INDEXER_FORWARD
    from vllm.models.glm5next.common import attention

    if _ORIG_INDEXER_FORWARD is not None:
        attention.Indexer.forward = _ORIG_INDEXER_FORWARD
        _ORIG_INDEXER_FORWARD = None
