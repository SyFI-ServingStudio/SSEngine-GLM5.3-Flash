"""Control messages (msgspec, msgpack over zmq) and the stage-to-stage activation pipes.

Interface: ppimpl/runtime/README.md. Controller -> worker s: one PUSH/PULL pair per worker, plans in mb_id order.
Workers -> controller: one PULL socket at the controller. Activations go worker s -> s+1 by copy engine into the
receiver's ring (CePipe; on CPU by gloo, GlooPipe); every worker handles plans in mb_id order, so sends and receives
match without tags.
"""

import ctypes
from collections import deque

import msgspec
import torch
import torch.distributed as dist
import zmq


# --- controller -> worker -------------------------------------------------------------------------------------------

class Chunk(msgspec.Struct, array_like=True):
    req_id: str
    start: int                    # tokens of this request already in KV (cached + computed by earlier chunks)
    num_tokens: int               # tokens computed in this chunk
    kv_blocks: list[int]          # DSA blocks covering [0, start + num_tokens)
    state_blocks: list[int]       # 6 KDA state blocks
    is_last: bool                 # last chunk of the prompt: stage 3 returns this request's token
    token_ids: list[int] | None   # the chunk's prompt tokens; stage 0 only


class Plan(msgspec.Struct, tag=True):
    mb_id: int                    # 0, 1, 2, ... in dispatch order
    chunks: list[Chunk]           # hidden rows concatenated in this order

    @property
    def num_tokens(self) -> int:
        return sum(c.num_tokens for c in self.chunks)


class Shutdown(msgspec.Struct, tag=True):
    pass


class TierMove(msgspec.Struct, array_like=True):
    """One tier op as every worker runs it on its quarter (ppimpl/engine/tiers.py TierOp, lanes/m6 TIERS.md §3)."""
    op_id: int
    op: str                       # "write" HBM -> tier, "read" tier -> HBM, "demote" DRAM -> SSD
    tier: str                     # "dram" | "ssd": destination of write/demote, source of read
    host_tier: str | None         # SSD ops: the host side ("dram" | "staging")
    b_lo: int                     # HBM blocks [b_lo, b_hi) of the context move (an SSD read moves all of its slots)
    b_hi: int
    block_start: int              # slots[i] holds block block_start + i
    kv_blocks: list[int]          # the context's HBM blocks from block 0
    state_blocks: list[int]
    slots: list[int]
    state_slots: list[int]
    host_slots: list[int]
    host_state: list[int]


class TierBatch(msgspec.Struct, tag=True):
    ops: list[TierMove]           # dependencies done: a worker may run them in any order


# --- worker -> controller -------------------------------------------------------------------------------------------

class Ready(msgspec.Struct, tag=True):
    stage: int
    num_blocks: int               # blocks this stage can back with its KV tensors (incl. the null block)
    slot_bytes: int = 0           # tiers: one tier slot (this GPU's quarter of one KV block of all stages)
    state_slots: int = 0          # tiers: slots of one context's state


class StageDone(msgspec.Struct, tag=True):
    stage: int
    mb_id: int
    start_ns: int                 # GPU start of the stage's first kernel (host clock)
    end_ns: int                   # GPU end of its last kernel, before the send
    tokens: list[tuple[str, int]] | None = None   # stage 3: (req_id, greedy token) per is_last chunk
    host_ns: tuple[int, int, int] | None = None    # host clock: plan received, prepare called, run returned


class TierDone(msgspec.Struct, tag=True):
    stage: int
    op_id: int
    start_ns: int                 # host clock: copies submitted
    end_ns: int                   # host clock: seen complete
    nbytes: int                   # moved by this GPU


class WorkerError(msgspec.Struct, tag=True):
    stage: int
    error: str


ToWorker = Plan | Shutdown
ToController = Ready | StageDone | TierDone | WorkerError

_enc = msgspec.msgpack.Encoder()
_dec_worker = msgspec.msgpack.Decoder(ToWorker)
_dec_controller = msgspec.msgpack.Decoder(ToController)
_dec_tier = msgspec.msgpack.Decoder(TierBatch | Shutdown)


def encode(msg) -> bytes:
    return _enc.encode(msg)


def decode_to_worker(buf: bytes) -> Plan | Shutdown:
    return _dec_worker.decode(buf)


def decode_to_controller(buf: bytes) -> Ready | StageDone | TierDone | WorkerError:
    return _dec_controller.decode(buf)


def decode_tier(buf: bytes) -> TierBatch | Shutdown:
    return _dec_tier.decode(buf)


def for_stage(plan: Plan, stage: int) -> Plan:
    """The plan as sent to `stage`: token ids go to stage 0 only."""
    if stage == 0:
        return plan
    return Plan(plan.mb_id, [msgspec.structs.replace(c, token_ids=None) for c in plan.chunks])


# --- zmq sockets ----------------------------------------------------------------------------------------------------

def worker_sockets(ctx: zmq.Context, plan_addr: str) -> zmq.Socket:
    """Worker side: PULL plans (bound by the worker, the controller connects)."""
    pull = ctx.socket(zmq.PULL)
    pull.bind(plan_addr)
    return pull


def report_socket(ctx: zmq.Context, report_addr: str) -> zmq.Socket:
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 5000)
    push.connect(report_addr)
    return push


# --- activations ----------------------------------------------------------------------------------------------------

class GlooPipe:
    """CPU stages (tests, `device: cpu`): activations between neighbouring stages by gloo send/recv."""

    EXIT_TAG = 1

    def __init__(self, stage: int, num_stages: int, init_method: str, max_in_flight: int):
        self.stage, self.num_stages, self.depth = stage, num_stages, max_in_flight
        dist.init_process_group("gloo", init_method=init_method, rank=stage, world_size=num_stages)
        self._sends: deque = deque()   # (work, tensor): keep sent tensors alive until the send completes
        self.n_gate = 0                # stage 0: exits received

    def recv(self, num_tokens: int, width: int) -> torch.Tensor:
        buf = torch.empty(num_tokens, width, dtype=torch.bfloat16)
        dist.recv(buf, self.stage - 1)
        return buf

    def send(self, hidden: torch.Tensor, after=None):
        while self._sends and self._sends[0][0].is_completed():
            self._sends.popleft()
        self._sends.append((dist.isend(hidden, self.stage + 1), hidden))

    def gate(self, n: int):
        """Stage 0, before its n-th forward: block until the last stage has finished forward n - max_in_flight."""
        while self.num_stages > 1 and self.n_gate < n - self.depth:
            self._recv_exit()

    def _recv_exit(self):
        buf = torch.zeros(1, dtype=torch.int64)
        dist.recv(buf, self.num_stages - 1, tag=self.EXIT_TAG)
        self.n_gate += 1
        assert int(buf) == self.n_gate, (int(buf), self.n_gate)

    def exited(self, n: int):
        if self.num_stages > 1:
            t = torch.tensor([n], dtype=torch.int64)
            self._sends.append((dist.isend(t, 0, tag=self.EXIT_TAG), t))

    def release(self):
        pass

    def close(self, mbs: int = 0):
        """`mbs`: microbatches this stage ran (stage 0 takes the exits it has not yet received)."""
        if self.stage == 0:
            while self.num_stages > 1 and self.n_gate < mbs:
                self._recv_exit()
        for work, _ in self._sends:
            work.wait()
        self._sends.clear()
        dist.destroy_process_group()


class CePipe:
    """Activations by copy engine, no SMs: the sender's cudaMemcpyAsync writes straight into a ring of K slots on the
    receiver (CUDA IPC), and two flag words per slot order it, by stream memory operations (no kernels):

      sender, copy stream:      wait end event; wait free[slot] >= n-K+1; copy; write ready[slot] = n
      receiver, compute stream: wait ready[slot] >= n; forward; write free[slot] = n          (release())

    n counts microbatches from 1 on both sides, slot = (n-1) % K. ready[] lives on the receiver and free[] on the
    sender, so each side spins on its own memory and writes the other's over NVLink. Handles are exchanged once, at
    start-up, over gloo, which also serves the barriers.

    The in-flight gate (ppimpl/runtime/dispatch.py) is one more flag word, on stage 0:
      last stage, compute stream:  after forward n: write exits = n                              (exited())
      stage 0, compute stream:     before forward n: wait exits >= n - max_in_flight              (gate())
    """

    RING_BYTES = 4 << 30   # K = max_in_flight + 1 slots of mb_tokens rows, fewer when that exceeds this (one-request
                           # runs: one 128k microbatch; K >= 1 only throttles the sender, never breaks the order)

    def __init__(self, stage: int, num_stages: int, device: torch.device, init_method: str, max_in_flight: int,
                 max_rows: int, width: int):
        from cuda.bindings import driver, runtime
        self.drv, self.rt = driver, runtime
        self.stage, self.num_stages, self.device = stage, num_stages, device
        self.rows, self.width = max_rows, width
        self.slot_bytes = max_rows * width * 2
        self.K = max(1, min(max_in_flight + 1, self.RING_BYTES // self.slot_bytes))
        self.depth = max_in_flight
        dist.init_process_group("gloo", init_method=init_method, rank=stage, world_size=num_stages)
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)           # the primary context is current on this thread (driver calls)
        self.copy_stream = torch.cuda.Stream(device)
        self._local, self._remote = [], []
        mine = {}
        self.ring = self.ready = self.free = None
        if stage > 0:                           # receiver: ring + ready[]
            self.ring_ptr = self._alloc(self.K * self.slot_bytes)
            self.ready = self._alloc(256, zero=True)
            mine |= {"ring": self._export(self.ring_ptr), "ready": self._export(self.ready)}
            self.ring = torch.as_tensor(_DevMem(self.ring_ptr, self.K * self.slot_bytes), device=device) \
                .view(torch.bfloat16).view(self.K, max_rows, width)
        if stage < num_stages - 1:              # sender: free[]
            self.free = self._alloc(256, zero=True)
            mine["free"] = self._export(self.free)
        self.exits = self.peer_exits = None
        if stage == 0 and num_stages > 1:       # the in-flight gate
            self.exits = self._alloc(256, zero=True)
            mine["exits"] = self._export(self.exits)
        torch.cuda.synchronize(device)
        allh = [None] * num_stages
        dist.all_gather_object(allh, mine)
        if stage < num_stages - 1:
            self.peer_ring = self._open(allh[stage + 1]["ring"])
            self.peer_ready = self._open(allh[stage + 1]["ready"])
        if stage > 0:
            self.peer_free = self._open(allh[stage - 1]["free"])
        if stage == num_stages - 1 and num_stages > 1:
            self.peer_exits = self._open(allh[0]["exits"])
        dist.barrier()
        self.n_recv = self.n_send = 0

    # -- CUDA helpers --
    def _ok(self, ret):
        err, *rest = ret if isinstance(ret, tuple) else (ret,)
        if int(err) != 0:
            raise RuntimeError(f"CUDA error {err}")
        return rest[0] if len(rest) == 1 else rest

    def _alloc(self, nbytes: int, zero: bool = False) -> int:
        ptr = int(self._ok(self.rt.cudaMalloc(nbytes)))
        if zero:
            self._ok(self.rt.cudaMemset(ptr, 0, nbytes))
        self._local.append(ptr)
        return ptr

    def _export(self, ptr: int) -> bytes:
        h = self._ok(self.rt.cudaIpcGetMemHandle(ptr))
        return ctypes.string_at(h.getPtr(), 64)

    def _open(self, raw: bytes) -> int:
        h = self.rt.cudaIpcMemHandle_t()
        ctypes.memmove(h.getPtr(), raw, 64)
        ptr = int(self._ok(self.rt.cudaIpcOpenMemHandle(h, self.rt.cudaIpcMemLazyEnablePeerAccess)))
        self._remote.append(ptr)
        return ptr

    def _wait(self, stream: torch.cuda.Stream, addr: int, value: int):
        self._ok(self.drv.cuStreamWaitValue32(self.drv.CUstream(stream.cuda_stream), self.drv.CUdeviceptr(addr),
                                              value, self.drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ))

    def _write(self, stream: torch.cuda.Stream, addr: int, value: int):
        self._ok(self.drv.cuStreamWriteValue32(self.drv.CUstream(stream.cuda_stream), self.drv.CUdeviceptr(addr),
                                               value, self.drv.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT))

    # -- pipe --
    def recv(self, num_tokens: int, width: int) -> torch.Tensor:
        assert width == self.width and num_tokens <= self.rows, (num_tokens, width, self.rows, self.width)
        self.n_recv += 1
        slot = (self.n_recv - 1) % self.K
        self._wait(torch.cuda.current_stream(self.device), self.ready + 4 * slot, self.n_recv)
        return self.ring[slot, :num_tokens]

    def release(self):
        """After the forward that read the last recv()'s rows is queued: the sender may reuse that slot."""
        slot = (self.n_recv - 1) % self.K
        self._write(torch.cuda.current_stream(self.device), self.peer_free + 4 * slot, self.n_recv)

    def gate(self, n: int):
        """Stage 0, before its n-th forward (from 1): the stream waits until the last stage finished forward
        n - max_in_flight."""
        if self.exits is not None and n > self.depth:
            self._wait(torch.cuda.current_stream(self.device), self.exits, n - self.depth)

    def exited(self, n: int):
        """Last stage, after its n-th forward is queued."""
        if self.peer_exits is not None:
            self._write(torch.cuda.current_stream(self.device), self.peer_exits, n)

    def send(self, hidden: torch.Tensor, after: "torch.cuda.Event | None"):
        assert hidden.is_contiguous() and hidden.dtype == torch.bfloat16 and hidden.shape[1] == self.width
        assert hidden.shape[0] <= self.rows, (hidden.shape, self.rows)
        self.n_send += 1
        slot = (self.n_send - 1) % self.K
        s = self.copy_stream
        s.wait_event(after)
        if self.n_send > self.K:
            self._wait(s, self.free + 4 * slot, self.n_send - self.K)
        self._ok(self.rt.cudaMemcpyAsync(self.peer_ring + slot * self.slot_bytes, hidden.data_ptr(),
                                         hidden.numel() * 2, self.rt.cudaMemcpyKind.cudaMemcpyDefault, s.cuda_stream))
        self._write(s, self.peer_ready + 4 * slot, self.n_send)
        hidden.record_stream(s)

    def close(self, mbs: int = 0):
        torch.cuda.synchronize(self.device)
        dist.barrier()                          # peers are done with our ring and flags
        for ptr in self._remote:
            self._ok(self.rt.cudaIpcCloseMemHandle(ptr))
        self.ring = None
        for ptr in self._local:
            self._ok(self.rt.cudaFree(ptr))
        dist.destroy_process_group()


class _DevMem:
    """Raw device bytes as a __cuda_array_interface__ object, for torch.as_tensor (the pipe frees it)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def open_pipe(cfg, stage: int, device: torch.device, width: int):
    if device.type == "cuda":
        return CePipe(stage, cfg.num_stages, device, cfg.dist_init_method(), cfg.gate_depth(),
                      cfg.scheduler.mb_tokens, width)
    return GlooPipe(stage, cfg.num_stages, cfg.dist_init_method(), cfg.gate_depth())
