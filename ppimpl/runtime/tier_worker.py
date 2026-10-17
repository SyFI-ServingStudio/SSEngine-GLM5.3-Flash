"""Worker side of the DRAM/SSD tiers (lanes/m6 TIERS.md §3-§6).

Every worker moves its quarter of every tier op (balanced storage, `ppimpl/runtime/tier_copy.py`). The controller
sends a `TierBatch` of ops whose dependencies are done to all four workers' tier sockets; a worker's tier thread
builds the (device, host, length) pieces of its quarter over all four stages' KV (the other stages' KV opened by CUDA
IPC at start-up), submits them as one `cuMemcpyBatchAsync` (vLLM `ops.swap_blocks_batch`) on its read stream (H2D)
or write stream (D2H), and a done thread per stream reports `TierDone` when the copies completed. Copy engines only,
no SMs, no wait on the compute stream: a write's blocks are final and pinned until it is done, a read's blocks are
held by a request that computes only after all four `TierDone`s.

DRAM: one pinned arena per worker on its GPU's NUMA node: libnuma, MADV_HUGEPAGE, first touch on a side thread
while the model loads, then `cudaHostRegister` in chunks of whole slots (a piece never crosses a slot, so never a
registration). Not torch's pinned allocator: it rounds 150 GB up to 256 GB.

The fake model has no KV: its engine moves nothing and reports each op after the time the configured bandwidth
gives its bytes, one op after another per direction (CPU tests, fake runs).
"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
import traceback

import numpy as np

from ppimpl.runtime.tier_copy import Layout, ViewRange
from ppimpl.runtime.transport import Shutdown, TierBatch, TierDone, TierMove, WorkerError, decode_tier, encode

MADV_HUGEPAGE = 14
REGISTER_CHUNK = 1 << 30      # cudaHostRegister per ~1 GiB of whole slots, from 4 threads
TOUCH_THREADS = 4
FAKE_SLOT_BYTES = 12_857_344  # 8,576 x 1,499 B rounded up to 4 KiB: the real balanced slot (TIERS.md §3)


def tier_num_slots(capacity_gb: float | None, slot_bytes: int) -> int:
    """Slots of a tier of `capacity_gb` per GPU (as `ppimpl.engine.tiers.TierStore.num_slots`)."""
    return int(capacity_gb * 1e9 // slot_bytes) if capacity_gb else 0


def _ok(ret):
    err, *rest = ret if isinstance(ret, tuple) else (ret,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return rest[0] if len(rest) == 1 else rest


# --- host memory ----------------------------------------------------------------------------------------------------

class HostArena:
    """`nbytes` of host memory on NUMA node `node` (CUDA: pinned once `register` ran; CPU: a plain numpy buffer)."""

    def __init__(self, nbytes: int, node: int | None, cuda: bool):
        self.nbytes = -(-nbytes // (2 << 20)) * (2 << 20)
        self.cuda, self.node = cuda, node
        self._chunks: list[tuple[int, int]] = []
        self.touch_s = self.register_s = 0.0
        if not cuda:
            self._buf = np.zeros(self.nbytes, dtype=np.uint8)
            self.ptr = self._buf.ctypes.data
            return
        self._numa = ctypes.CDLL("libnuma.so.1")
        self._numa.numa_alloc_onnode.restype = ctypes.c_void_p
        self._numa.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
        self._numa.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.ptr = self._numa.numa_alloc_onnode(self.nbytes, node)
        if not self.ptr:
            raise MemoryError(f"numa_alloc_onnode({self.nbytes}, {node})")
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise(self.ptr, self.nbytes, MADV_HUGEPAGE)   # advisory: no THP is not an error

    def touch(self):
        """First touch from a few threads (pages land on the arena's node; ctypes.memset releases the GIL)."""
        if not self.cuda:
            return
        t0 = time.perf_counter()
        step = -(-self.nbytes // TOUCH_THREADS)
        ts = [threading.Thread(target=ctypes.memset, args=(self.ptr + a, 0, min(step, self.nbytes - a)))
              for a in range(0, self.nbytes, step)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.touch_s = time.perf_counter() - t0

    def register(self, slot_bytes: int):
        """Pin in chunks of whole slots (CUDA). Chunks register from 4 threads."""
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
                    _ok(rt.cudaHostRegister(self.ptr + a, n, flags))
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

    def close(self):
        if not self.cuda or not self.ptr:
            return
        from cuda.bindings import runtime as rt
        for a, _ in self._chunks:
            rt.cudaHostUnregister(self.ptr + a)
        self._chunks = []
        self._numa.numa_free(self.ptr, self.nbytes)
        self.ptr = 0


# --- the other stages' KV -------------------------------------------------------------------------------------------

def share_ranges(stage: int, n: int, ranges: list[ViewRange]) -> tuple[list[ViewRange], list[int]]:
    """All stages' ranges with device addresses valid in this process: each worker exports the allocations behind its
    ranges (cuMemGetAddressRange base, cudaIpcGetMemHandle), and opens the others' (lazy peer access). Collective
    over the workers' gloo group (the pipe's). Returns (ranges, opened pointers to close)."""
    import torch.distributed as dist
    from cuda.bindings import driver, runtime as rt

    allocs: dict[int, bytes] = {}
    mine = []
    for r in ranges:
        base, _size = _ok(driver.cuMemGetAddressRange(driver.CUdeviceptr(r.base)))
        base = int(base)
        if base not in allocs:
            try:
                h = _ok(rt.cudaIpcGetMemHandle(base))
            except RuntimeError as e:
                raise RuntimeError(f"stage {stage}: {r.name} is not IPC-shareable (KV must be allocated without "
                                   f"expandable segments: StageModel(ipc_kv=True))") from e
            allocs[base] = ctypes.string_at(h.getPtr(), 64)
        mine.append((r.name, r.kind, r.group, base, r.base - base, r.stride, r.nbytes))
    allh = [None] * n
    dist.all_gather_object(allh, {"allocs": allocs, "ranges": mine})
    out, opened = list(ranges), []
    for s in range(n):
        if s == stage:
            continue
        local = {}
        for base, raw in allh[s]["allocs"].items():
            h = rt.cudaIpcMemHandle_t()
            ctypes.memmove(h.getPtr(), raw, 64)
            local[base] = int(_ok(rt.cudaIpcOpenMemHandle(h, rt.cudaIpcMemLazyEnablePeerAccess)))
            opened.append(local[base])
        for name, kind, group, base, off, stride, nbytes in allh[s]["ranges"]:
            out.append(ViewRange(s, name, kind, local[base] + off, stride, nbytes, group))
    return out, opened


# --- the engine -----------------------------------------------------------------------------------------------------

class TierEngine:
    """One worker's tier side: its tier socket, host arenas, streams, and the threads that run and report ops.

    Built in two steps so the arena's first touch overlaps the model load: `TierEngine(...)` allocates and starts
    touching; `attach(ranges)` (after the model, collective over the workers) opens the peers' KV, builds the layout,
    registers the arena and returns (slot_bytes, state_slots) for `Ready`; `start()` begins serving ops."""

    def __init__(self, cfg, stage: int, device, zmq_ctx):
        tc = cfg.tiers
        self.cfg, self.tc, self.stage, self.n = cfg, tc, stage, cfg.num_stages
        self.device, self.cuda = device, device.type == "cuda"
        self.ctx = zmq_ctx
        self.layout: Layout | None = None
        self.slot_bytes = tc.slot_bytes or 0
        self.opened: list[int] = []
        self.dram: HostArena | None = None
        self._touch: threading.Thread | None = None
        self.real = False
        self.threads: list[threading.Thread] = []
        self.stop = threading.Event()
        self.q = {"read": queue.Queue(), "write": queue.Queue()}
        self._due: dict[str, float] = {}           # fake engine: when each direction's last op is done
        if tc.dram_gb and self.cuda and cfg.model != "fake":
            from ppimpl.runtime.cpus import _gpu_node
            self.dram = HostArena(int(tc.dram_gb * 1e9), _gpu_node(device.index), True)
            self._touch = threading.Thread(target=self.dram.touch, name="tier-touch", daemon=True)
            self._touch.start()

    def attach(self, ranges: list[ViewRange] | None) -> tuple[int, int]:
        """`ranges`: this stage's KV ranges (`tier_copy.stage_ranges`), None for a model without KV (fake)."""
        if ranges is None:
            self.slot_bytes = self.slot_bytes or FAKE_SLOT_BYTES
            return self.slot_bytes, self.tc.state_slots
        if self.tc.ssd_gb or not self.tc.dram_gb:
            raise NotImplementedError("SSD tier: lanes/m6 step 2")
        import torch
        self.real = True
        allr, self.opened = share_ranges(self.stage, self.n, ranges) if self.cuda else (ranges, [])
        self.layout = Layout(allr, self.stage, self.n, slot_bytes=self.tc.slot_bytes)
        self.slot_bytes = self.layout.slot_bytes
        if self.dram is None:
            self.dram = HostArena(tier_num_slots(self.tc.dram_gb, self.slot_bytes) * self.slot_bytes, None, self.cuda)
        if self._touch is not None:
            self._touch.join()
        self.dram.register(self.slot_bytes)
        assert tier_num_slots(self.tc.dram_gb, self.slot_bytes) * self.slot_bytes <= self.dram.nbytes
        if self.cuda:
            from vllm import _custom_ops
            self.swap = _custom_ops.swap_blocks_batch
            self.streams = {"read": torch.cuda.Stream(self.device), "write": torch.cuda.Stream(self.device)}
        print(f"stage {self.stage} tiers: slot {self.slot_bytes} B, state {self.layout.state_slots} slots, "
              f"{len(self.layout.kv_len)} kv + {len(self.layout.st_len)} state pieces per block/context, dram "
              f"{self.dram.nbytes / 1e9:.1f} GB (touch {self.dram.touch_s:.1f} s, register "
              f"{self.dram.register_s:.1f} s), {len(self.opened)} peer allocations", flush=True)
        return self.slot_bytes, self.layout.state_slots

    def start(self):
        for name in ("read", "write"):
            t = threading.Thread(target=self._done_loop, args=(name,), name=f"tier-{name}", daemon=True)
            t.start()
            self.threads.append(t)
        t = threading.Thread(target=self._recv_loop, name="tier-recv", daemon=True)
        t.start()
        self.threads.append(t)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=10)
        if self.cuda:
            import torch
            torch.cuda.synchronize(self.device)
            from cuda.bindings import runtime as rt
            for p in self.opened:
                rt.cudaIpcCloseMemHandle(p)
        if self.dram is not None:
            self.dram.close()

    # --- threads ----------------------------------------------------------------------------------------------------

    def _recv_loop(self):
        import zmq
        sock = self.ctx.socket(zmq.PULL)
        sock.bind(self.cfg.tier_addr(self.stage))
        err = None
        try:
            if self.cuda:
                import torch
                torch.cuda.set_device(self.device)
            while not self.stop.is_set():
                if not sock.poll(200):
                    continue
                msg = decode_tier(sock.recv())
                if isinstance(msg, Shutdown):
                    break
                for op in msg.ops:
                    self.submit(op)
        except BaseException:   # noqa: BLE001 - reported to the controller, which fails the run
            err = WorkerError(self.stage, "tier thread:\n" + traceback.format_exc())
        finally:
            sock.close(linger=0)
            for q in self.q.values():
                q.put(err)
                err = None

    def _done_loop(self, direction: str):
        import zmq
        push = self.ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 5000)
        push.connect(self.cfg.report_addr())
        try:
            while (item := self.q[direction].get()) is not None:
                if isinstance(item, WorkerError):
                    push.send(encode(item))
                    break
                op_id, wait, start_ns, nbytes = item
                if isinstance(wait, float):     # fake engine: due time (monotonic)
                    d = wait - time.monotonic()
                    if d > 0:
                        time.sleep(d)
                elif wait is not None:
                    wait.synchronize()
                push.send(encode(TierDone(self.stage, op_id, start_ns, time.time_ns(), nbytes)))
        finally:
            push.close()

    # --- one op -----------------------------------------------------------------------------------------------------

    @staticmethod
    def direction(op: TierMove) -> str:
        return "read" if op.op == "read" else "write"

    def submit(self, op: TierMove):
        start_ns = time.time_ns()
        d = self.direction(op)
        if not self.real:
            ssd = "ssd" in (op.tier, op.host_tier)
            blocks = len(op.slots) if ssd else op.b_hi - op.b_lo     # an SSD op moves whole slots
            nbytes = (blocks + len(op.state_slots)) * self.slot_bytes
            bw = (self.tc.ssd_read_gb_s if ssd else self.tc.dram_read_gb_s) * 1e9
            self._due[d] = max(self._due.get(d, 0.0), time.monotonic()) + nbytes / bw
            self.q[d].put((op.op_id, self._due[d], start_ns, nbytes))
            return
        if op.tier != "dram" or op.host_tier is not None or op.op not in ("read", "write"):
            raise NotImplementedError(f"tier op {op.op} {op.tier} via {op.host_tier}: lanes/m6 step 2")
        dev, host, lens = self.layout.pieces(op.kv_blocks, op.state_blocks, op.slots, op.state_slots, op.b_lo,
                                             op.b_hi, op.block_start)
        host = host + self.dram.ptr
        src, dst = (host, dev) if d == "read" else (dev, host)
        nbytes = int(lens.sum())
        if not self.cuda:
            for s, t, k in zip(src.tolist(), dst.tolist(), lens.tolist()):
                ctypes.memmove(t, s, k)
            self.q[d].put((op.op_id, None, start_ns, nbytes))
            return
        import torch
        stream = self.streams[d]
        with torch.cuda.stream(stream):
            self.swap(torch.from_numpy(src), torch.from_numpy(dst), torch.from_numpy(lens))
        ev = torch.cuda.Event(blocking=True)
        ev.record(stream)
        self.q[d].put((op.op_id, ev, start_ns, nbytes))

