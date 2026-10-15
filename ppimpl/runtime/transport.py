"""Control messages (msgspec, msgpack over zmq) and the stage-to-stage activation pipe (torch.distributed p2p).

Interface: ppimpl/runtime/README.md. Controller -> worker s: one PUSH/PULL pair per worker, plans in mb_id order.
Workers -> controller: one PULL socket at the controller. Activations go worker s -> s+1 by NCCL (gloo on CPU);
every worker handles plans in mb_id order, so sends and receives match without tags.
"""

from collections import deque

import msgspec
import torch
import torch.distributed as dist
import zmq


# --- controller -> worker -------------------------------------------------------------------------------------------

class Chunk(msgspec.Struct, array_like=True):
    req_id: str
    start: int                    # tokens of this request already in KV (computed by earlier chunks)
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


# --- worker -> controller -------------------------------------------------------------------------------------------

class Ready(msgspec.Struct, tag=True):
    stage: int
    num_blocks: int               # blocks this stage can back with its KV tensors (incl. the null block)


class StageDone(msgspec.Struct, tag=True):
    stage: int
    mb_id: int
    start_ns: int                 # GPU start of the stage's first kernel (host clock)
    end_ns: int                   # GPU end of its last kernel, before the send
    tokens: list[tuple[str, int]] | None = None   # stage 3: (req_id, greedy token) per is_last chunk


class WorkerError(msgspec.Struct, tag=True):
    stage: int
    error: str


ToWorker = Plan | Shutdown
ToController = Ready | StageDone | WorkerError

_enc = msgspec.msgpack.Encoder()
_dec_worker = msgspec.msgpack.Decoder(ToWorker)
_dec_controller = msgspec.msgpack.Decoder(ToController)


def encode(msg) -> bytes:
    return _enc.encode(msg)


def decode_to_worker(buf: bytes) -> Plan | Shutdown:
    return _dec_worker.decode(buf)


def decode_to_controller(buf: bytes) -> Ready | StageDone | WorkerError:
    return _dec_controller.decode(buf)


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

class Pipe:
    """Point-to-point activations between neighbouring stages.

    recv(): posts the receive on its own stream, so it can land while the stage still computes the previous
    microbatch; the compute stream then waits for it on the GPU (no host sync).
    send(): issued on a side stream after the stage's end event, so the next plan's compute is not blocked by it.
    """

    def __init__(self, stage: int, num_stages: int, device: torch.device, init_method: str):
        self.stage, self.num_stages, self.device = stage, num_stages, device
        self.cuda = device.type == "cuda"
        # No device_id (eager init): lazily created p2p communicators are per pair, so the receive from s-1 and the
        # send to s+1 run on different NCCL streams and the receive can land while the send waits for compute.
        dist.init_process_group("nccl" if self.cuda else "gloo", init_method=init_method, rank=stage,
                                world_size=num_stages)
        self.recv_stream = torch.cuda.Stream(device) if self.cuda else None
        self.send_stream = torch.cuda.Stream(device) if self.cuda else None
        self._sends: deque = deque()   # (work, tensor): keep sent tensors alive until the send completes

    def recv(self, num_tokens: int, width: int) -> torch.Tensor:
        src = self.stage - 1
        if not self.cuda:
            buf = torch.empty(num_tokens, width, dtype=torch.bfloat16)
            dist.recv(buf, src)
            return buf
        with torch.cuda.stream(self.recv_stream):
            buf = torch.empty(num_tokens, width, dtype=torch.bfloat16, device=self.device)
            work = dist.irecv(buf, src)
        work.wait()                                   # current (compute) stream waits on the receive; host goes on
        buf.record_stream(torch.cuda.current_stream(self.device))
        return buf

    def send(self, hidden: torch.Tensor, after: "torch.cuda.Event | None"):
        dst = self.stage + 1
        self._reap()
        if not self.cuda:
            self._sends.append((dist.isend(hidden, dst), hidden))
            return
        with torch.cuda.stream(self.send_stream):
            self.send_stream.wait_event(after)
            work = dist.isend(hidden, dst)
        hidden.record_stream(self.send_stream)
        self._sends.append((work, hidden))

    def _reap(self):
        while self._sends and self._sends[0][0].is_completed():
            self._sends.popleft()

    def close(self):
        for work, _ in self._sends:
            work.wait()
        self._sends.clear()
        if self.cuda:
            torch.cuda.synchronize(self.device)
        dist.destroy_process_group()
