"""Layer 3 of the KV pool (lanes/m6 TIERS.md §4, §13.2, §14.1): the host media one member stores its slots in. No torch.

Slot ids are not allocated here: the leader's index (`manager.py`) does, so they are the same on all n GPUs. A
backend only maps a slot to a host address (`HostTier.ptr`) or a file offset (`SsdTier`, slot i at i x slot_bytes).

DRAM (`HostTier`): one arena per member, preferred (not bound) on its GPU's NUMA node, 4 KiB pages (MADV_HUGEPAGE
made the first touch compact memory synchronously: 150 GB took ~500 s on a host whose page cache filled the node;
PPIMPL_TIER_HUGEPAGE=1 turns it back on), first touch from TOUCH_THREADS threads on the node's CPUs of the job, on a
side thread while the model loads, then `cudaHostRegister` in chunks of whole slots (a piece never crosses a slot, so
never a registration). Not torch's pinned allocator: it rounds 150 GB up to 256 GB. Every process that makes an arena
sets oom_score_adj 1000, so an OOM takes it before anyone else's (§14.1).

SSD (`SsdTier`): one preallocated file per member, O_DIRECT `preadv`/`pwritev` of whole slots from a pool of
IO_THREADS threads (GIL released), into and out of host slots. The file is deleted at close.
"""

from __future__ import annotations

import ctypes
import errno
import mmap
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

MADV_HUGEPAGE = 14
MPOL_PREFERRED = 1
REGISTER_CHUNK = 1 << 30      # cudaHostRegister per ~1 GiB of whole slots, from 4 threads
TOUCH_THREADS = 16           # first-touch threads per arena, on the job's CPUs of the arena's node (page-cache reclaim
                             # runs in the touching threads: 4 threads on the worker's 2 cores were the limit)
HUGEPAGE = os.environ.get("PPIMPL_TIER_HUGEPAGE", "0") == "1"
IO_THREADS = 8                # SSD preadv/pwritev pool per member (fio: 8 psync threads, 53 GB/s on one file)


def num_slots(capacity_gb: float | None, slot_bytes: int) -> int:
    """Slots of a tier of `capacity_gb` per GPU (as `manager.TierStore.num_slots`)."""
    return int(capacity_gb * 1e9 // slot_bytes) if capacity_gb else 0


def _ok(ret):
    err, *rest = ret if isinstance(ret, tuple) else (ret,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return rest[0] if len(rest) == 1 else rest


# --- host memory placement ------------------------------------------------------------------------------------------

def node_cpus(cpus: set[int] | None, node: int | None) -> set[int] | None:
    """The CPUs of `cpus` on NUMA node `node`, or all of `cpus` when the node has fewer than 8 of them."""
    if not cpus or node is None:
        return cpus
    try:
        mine = cpus & set(_cpulist(open(f"/sys/devices/system/node/node{node}/cpulist").read()))
    except OSError:
        return cpus
    return mine if len(mine) >= 8 else cpus


def _cpulist(lst: str) -> list[int]:
    out = []
    for part in lst.strip().split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def numa_resident_gb(ptr: int, nbytes: int, maps: str = "/proc/self/numa_maps") -> dict[int, float]:
    """GB resident per NUMA node of the mappings starting in [ptr, ptr + nbytes) (`N<node>=<pages>` of each line, in
    its kernelpagesize_kB)."""
    out: dict[int, float] = {}
    with open(maps) as f:
        for line in f:
            parts = line.split()
            if not parts or not ptr <= int(parts[0], 16) < ptr + nbytes:
                continue
            kb = next((int(x.split("=")[1]) for x in parts if x.startswith("kernelpagesize_kB=")), 4)
            for x in parts:
                if x.startswith("N") and "=" in x and x[1:x.index("=")].isdigit():
                    node = int(x[1:x.index("=")])
                    out[node] = out.get(node, 0.0) + int(x.split("=")[1]) * kb * 1024 / 1e9
    return {k: round(v, 2) for k, v in sorted(out.items())}


def prefer_node(numa, ptr: int, nbytes: int, node: int):
    """Place [ptr, ptr + nbytes) on `node`, preferred, not bound: on a node short of memory the pages go to another
    node (the start-up line's resident GB by node shows it). A strict bind (numa_alloc_onnode) does not fall back:
    on cayenne it ran node 0 out and the kernel's OOM killer killed other users' processes (7996)."""
    numa.mbind.restype = ctypes.c_long
    numa.mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.POINTER(ctypes.c_ulong),
                           ctypes.c_ulong, ctypes.c_uint]
    words = node // 64 + 1
    mask = (ctypes.c_ulong * words)()
    mask[node // 64] = 1 << (node % 64)
    if numa.mbind(ptr, nbytes, MPOL_PREFERRED, mask, words * 64 + 1, 0) != 0:
        raise OSError(ctypes.get_errno(), f"mbind(MPOL_PREFERRED, node {node})")


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return libc


def anon_map(nbytes: int) -> int:
    """nbytes of anonymous memory with no page faulted in (libnuma's numa_alloc touches every page under the current
    policy, so a policy set afterwards is too late)."""
    libc = _libc()
    ptr = libc.mmap(None, nbytes, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    if ptr in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), f"mmap({nbytes})")
    return ptr


def anon_unmap(ptr: int, nbytes: int):
    _libc().munmap(ptr, nbytes)


def oom_first():
    """Make this process the OOM killer's first choice (oom_score_adj 1000; raising it needs no privilege): a process
    pinning 100s of GB must not take other users' processes down with it."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def cgroup_peak_gb() -> float | None:
    """This process's cgroup memory.peak (cgroup v2), GB, or None where it cannot be read."""
    try:
        rel = next(x.split("::", 1)[1].strip() for x in open("/proc/self/cgroup") if x.startswith("0::"))
        d = "/sys/fs/cgroup" + rel
        while d.startswith("/sys/fs/cgroup"):
            if os.path.exists(d + "/memory.peak"):
                return int(open(d + "/memory.peak").read()) / 1e9
            d = os.path.dirname(d)
    except (OSError, StopIteration, ValueError):
        pass
    return None


# --- the tiers ------------------------------------------------------------------------------------------------------

class HostTier:
    """`nbytes` of host memory on NUMA node `node` (CUDA: pinned once `register` ran; CPU: a plain numpy buffer), slot
    i at `ptr(i)` once the slot size is known (`register`)."""

    def __init__(self, nbytes: int, node: int | None, cuda: bool):
        self.nbytes = -(-nbytes // (2 << 20)) * (2 << 20)
        self.cuda, self.node = cuda, node
        self.slot_bytes = 0
        self._chunks: list[tuple[int, int]] = []
        self.touch_s = self.register_s = 0.0
        if not cuda:                       # 4 KiB-aligned (O_DIRECT)
            self._buf = np.zeros(self.nbytes + 4096, dtype=np.uint8)
            self.base = self._buf.ctypes.data + (-self._buf.ctypes.data) % 4096
            return
        oom_first()
        self._numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
        self.base = anon_map(self.nbytes)          # no pages yet: the policy below decides where touch puts them
        if node is not None:
            prefer_node(self._numa, self.base, self.nbytes, node)
        if HUGEPAGE:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.madvise(self.base, self.nbytes, MADV_HUGEPAGE)   # advisory: no THP is not an error

    def ptr(self, slot: int) -> int:
        return self.base + slot * self.slot_bytes

    def touch(self, cpus: set[int] | None = None):
        """First touch from TOUCH_THREADS threads, each on `cpus` (pages land on the arena's node whatever CPU touches
        them; ctypes.memset releases the GIL)."""
        if not self.cuda:
            return
        t0 = time.perf_counter()

        def run(a, n):
            if cpus:
                os.sched_setaffinity(0, cpus)       # this thread only
            ctypes.memset(self.base + a, 0, n)

        step = -(-self.nbytes // TOUCH_THREADS)
        step = -(-step // (2 << 20)) * (2 << 20)
        ts = [threading.Thread(target=run, args=(a, min(step, self.nbytes - a)), name="tier-touch")
              for a in range(0, self.nbytes, step)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.touch_s = time.perf_counter() - t0

    def register(self, slot_bytes: int):
        """Fix the slot size; pin in chunks of whole slots (CUDA). Chunks register from 4 threads."""
        self.slot_bytes = slot_bytes
        if not self.cuda:
            return
        from cuda.bindings import runtime as rt
        t0 = time.perf_counter()
        chunk = max(1, REGISTER_CHUNK // slot_bytes) * slot_bytes
        spans = [(a, min(chunk, self.nbytes - a)) for a in range(0, self.nbytes, chunk)]
        flags = rt.cudaHostRegisterPortable | rt.cudaHostRegisterMapped
        errs = []

        def run(part, dev):
            try:
                _ok(rt.cudaSetDevice(dev))
                for a, n in part:
                    _ok(rt.cudaHostRegister(self.base + a, n, flags))
            except Exception as e:    # noqa: BLE001 - re-raised below
                errs.append(e)

        import torch
        dev = torch.cuda.current_device()
        ts = [threading.Thread(target=run, args=(spans[i::4], dev)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if errs:
            raise errs[0]
        self._chunks = spans
        self.register_s = time.perf_counter() - t0

    def node_gb(self) -> dict[int, float]:
        """GB of the arena resident on each NUMA node (/proc/self/numa_maps, the mappings inside the arena)."""
        return numa_resident_gb(self.base, self.nbytes)

    def placement(self) -> str:
        """`resident GB by node {..}`, plus `REMOTE SPILL x GB` when over 1 GB lies off the arena's node (the
        preferred node was short of memory: reads from there cross the socket link)."""
        gb = self.node_gb()
        off = sum(v for k, v in gb.items() if k != self.node)
        return f"resident GB by node {gb}" + (f", REMOTE SPILL {off:.1f} GB off node {self.node}" if off > 1 else "")

    def close(self):
        if not self.cuda or not self.base:
            return
        from cuda.bindings import runtime as rt
        for a, _ in self._chunks:
            rt.cudaHostUnregister(self.base + a)
        self._chunks = []
        anon_unmap(self.base, self.nbytes)
        self.base = 0


class SsdTier:
    """One member's SSD tier: a preallocated file of whole slots, read and written with O_DIRECT (buffered where the
    file system has no O_DIRECT, as tmpfs in CPU tests) from IO_THREADS threads."""

    def __init__(self, path: str, nbytes: int, slot_bytes: int):
        self.path, self.slot_bytes = path, slot_bytes
        os.makedirs(os.path.dirname(path), exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        try:
            self.fd = os.open(path, flags | os.O_DIRECT, 0o600)
            self.direct = True
        except OSError as e:
            if e.errno != errno.EINVAL:
                raise
            self.fd = os.open(path, flags, 0o600)
            self.direct = False
        os.posix_fallocate(self.fd, 0, nbytes)
        self.pool = ThreadPoolExecutor(IO_THREADS, thread_name_prefix="tier-io")

    def read(self, slots: list[int], host_ptrs: list[int]) -> list[Future]:
        """File slots -> host addresses, one future per slot, submitted in order."""
        return [self.pool.submit(self._read, s, p) for s, p in zip(slots, host_ptrs)]

    def write(self, slots: list[int], host_ptrs: list[int]) -> list[Future]:
        """Host addresses -> file slots, one future per slot, submitted in order."""
        return [self.pool.submit(self._write, s, p) for s, p in zip(slots, host_ptrs)]

    def _read(self, slot: int, ptr: int):
        buf = (ctypes.c_char * self.slot_bytes).from_address(ptr)
        n = os.preadv(self.fd, [buf], slot * self.slot_bytes)
        if n != self.slot_bytes:
            raise OSError(f"{self.path}: slot {slot}: read {n} of {self.slot_bytes} B")

    def _write(self, slot: int, ptr: int):
        buf = (ctypes.c_char * self.slot_bytes).from_address(ptr)
        n = os.pwritev(self.fd, [buf], slot * self.slot_bytes)
        if n != self.slot_bytes:
            raise OSError(f"{self.path}: slot {slot}: wrote {n} of {self.slot_bytes} B")

    def close(self):
        self.pool.shutdown()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            os.unlink(self.path)
