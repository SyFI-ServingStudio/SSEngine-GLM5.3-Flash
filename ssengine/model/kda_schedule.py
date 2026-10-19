"""FlashInfer persistent KDA's head-split search (`flashinfer.kda_kernels.persistent.forward._split_heads`) in C++.

`make_plan` (StageMetadata._kda_plan, once per distinct sequence offsets) runs `_piece_schedule`; when cutting chains
pays, `_split_heads` simulates an LPT packing per (split count, head length) of a 10 x 10 grid in Python: 20-35 ms per
multi-request microbatch, 110 ms at worst, on stage 0's host path (7895: 1.1 s/min of stage-0 GPU idle; plans reach
stage 0 ~2 ms before it is ready, so most of it is not hidden behind earlier microbatches).
`ssengine/native/kda_split_heads.cpp` is the same search with the same tie-breaking (jobs by (-cost, kind, chain), slots
from a (load, slot) min-heap, each slot stably by (kind, -cost), a better span only if strictly smaller); results are
equal (tests/test_kda_schedule_cpu.py). Built with g++ on first use into a cache directory keyed by the source hash.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "native" / "kda_split_heads.cpp"
MAX_SPLIT = 256        # the search's largest split count: at most len(chunks) + MAX_SPLIT jobs
_lib = None
_ORIG = None


def _build_dir() -> Path:
    d = os.environ.get("SSENGINE_NATIVE_DIR")
    if d:
        return Path(d)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ssengine" / "native"


def _load():
    global _lib
    if _lib is not None:
        return _lib
    src = SRC.read_bytes()
    d = _build_dir()
    d.mkdir(parents=True, exist_ok=True)
    so = d / f"kda_split_heads_{hashlib.sha256(src).hexdigest()[:16]}.so"
    if not so.exists():                                   # stages build at once: each to its own file, then rename
        tmp = so.with_name(f"{so.name}.{os.getpid()}.tmp")
        subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-o", str(tmp), str(SRC)], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    i32p, i64p = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)
    lib.kda_split_heads.argtypes = [ctypes.c_int, i32p, i32p, ctypes.c_int, ctypes.c_double,
                                    i64p, i32p, i32p, i32p, i32p, i32p]
    lib.kda_split_heads.restype = ctypes.c_int
    _lib = lib
    return lib


def split_heads(chunks, full, nslots, span_limit):
    """`_split_heads(chunks, full, nslots, span_limit)`: (span, slots of (chain, first chunk, end chunk)) or None."""
    lib = _load()
    n = len(chunks)
    i32p = ctypes.POINTER(ctypes.c_int32)
    ch = np.ascontiguousarray(chunks, dtype=np.int32)
    fu = np.ascontiguousarray(full, dtype=np.int32)
    cap = n + MAX_SPLIT
    out = np.empty((4, cap), dtype=np.int32)
    span, count = ctypes.c_int64(0), ctypes.c_int32(0)
    ok = lib.kda_split_heads(n, ch.ctypes.data_as(i32p), fu.ctypes.data_as(i32p), int(nslots), float(span_limit),
                             ctypes.byref(span), *(out[i].ctypes.data_as(i32p) for i in range(4)),
                             ctypes.byref(count))
    if not ok:
        return None
    slots: list[list[tuple[int, int, int]]] = [[] for _ in range(nslots)]
    for s, c, c0, c1 in out[:, :count.value].T.tolist():
        slots[s].append((c, c0, c1))
    return span.value, slots


def install() -> None:
    """Replace `_split_heads` in FlashInfer's persistent KDA forward module (process-wide; `_piece_schedule` looks
    the name up at call time)."""
    global _ORIG
    import flashinfer.kda_kernels.persistent.forward as fw

    if _ORIG is not None:
        return
    _load()                                               # build at start-up, not inside the first microbatch
    _ORIG = fw._split_heads
    fw._split_heads = split_heads
