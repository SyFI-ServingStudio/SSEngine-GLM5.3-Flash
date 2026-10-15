"""Intake process (RunConfig.intake_procs > 0): one of N HTTP servers on the same port (SO_REUSEPORT, the kernel
spreads connections over them), like vLLM's api-server-count. It parses a request's token ids into int32 bytes and
passes them to the controller's control thread over zmq; the token comes back the same way. The controller process
then serves no HTTP, so a burst of requests no longer queues behind one event loop.

DEALER (identity = index) <-> controller ROUTER (RunConfig.intake_addr), frames:
  intake -> controller: [b"R"] ready | [b"Q", msgpack (key, req_id, arrival_ns), int32 ids] request
                        | [b"D", msgpack (req_id, response_ns)] response sent | [b"S"] /shutdown
  controller -> intake: [b"G"] all intakes ready (/health 200 from now on)
                        | [b"A", msgpack (key, req_id, token)] | [b"E", msgpack (key, why)]
"""

import asyncio
import os
import signal
import socket
import time
import uuid
import warnings

import msgspec
import numpy as np
import uvicorn
import zmq
import zmq.asyncio

from ppimpl.config import RunConfig
from ppimpl.http import SamplingParams, error, json_body, read_body, reply, sse_body


class RawRequest(msgspec.Struct):
    token_ids: msgspec.Raw                # parsed by numpy straight into int32, without a list of ints
    sampling_params: SamplingParams = msgspec.field(default_factory=SamplingParams)
    stream: bool = False
    model: str | None = None
    rid: str | None = None


def parse_ids(raw) -> np.ndarray:
    """A JSON array of ints (msgspec.Raw) -> int32 array; ValueError for anything else."""
    text = bytes(raw)
    if text[:1] != b"[":
        raise ValueError("token_ids: not a list")
    inner = text[1:-1]
    if not inner.strip():
        return np.zeros(0, np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")    # numpy warns (and stops early) on what is not an int list
        try:
            ids = np.fromstring(inner, dtype=np.int64, sep=",")
        except (DeprecationWarning, ValueError) as e:
            raise ValueError(f"token_ids: {e}") from None
    if len(ids) != inner.count(b",") + 1 or (len(ids) and (ids.min() < 0 or ids.max() >= 2**31)):
        raise ValueError("token_ids: not a list of int32 ids")
    return ids.astype(np.int32)


class Intake:
    def __init__(self, cfg: RunConfig, index: int):
        self.cfg, self.index = cfg, index
        self.dec = msgspec.json.Decoder(RawRequest)
        self.pack = msgspec.msgpack.Encoder()
        self.unpack = msgspec.msgpack.Decoder()
        self.pending: dict[int, asyncio.Future] = {}
        self.next_key = 0
        self.go = False

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self._lifespan(receive, send)
        method, path = scope["method"], scope["path"]
        if path == "/inference/v1/generate" and method == "POST":
            return await self._generate(scope, receive, send)
        if path == "/health" and method == "GET":
            return await reply(send, 200 if self.go else 503, b"")
        if path == "/shutdown" and method == "POST":
            await self.sock.send_multipart([b"S"])
            return await reply(send, 200, b"")
        await reply(send, 404, b"")

    async def _lifespan(self, receive, send):
        while True:
            kind = (await receive())["type"].rsplit(".", 1)[1]
            if kind == "startup":
                self.ctx = zmq.asyncio.Context()
                self.sock = self.ctx.socket(zmq.DEALER)
                self.sock.setsockopt(zmq.IDENTITY, str(self.index).encode())
                self.sock.setsockopt(zmq.SNDHWM, 0)
                self.sock.setsockopt(zmq.RCVHWM, 0)
                self.sock.connect(self.cfg.intake_addr())
                await self.sock.send_multipart([b"R"])
                self.tasks = [asyncio.create_task(self._answers()), asyncio.create_task(self._watch_parent())]
            await send({"type": f"lifespan.{kind}.complete"})
            if kind == "shutdown":
                for t in self.tasks:
                    t.cancel()
                self.sock.close(linger=0)
                self.ctx.term()
                return

    async def _answers(self):
        while True:
            frames = await self.sock.recv_multipart()
            kind = frames[0]
            if kind == b"A":
                key, req_id, tok = self.unpack.decode(frames[1])
                fut = self.pending.pop(key, None)
                if fut is not None and not fut.done():
                    fut.set_result((req_id, tok))
            elif kind == b"E":
                key, why = self.unpack.decode(frames[1])
                fut = self.pending.pop(key, None)
                if fut is not None and not fut.done():
                    fut.set_exception(RuntimeError(why))
            elif kind == b"G":
                self.go = True

    async def _watch_parent(self):
        parent = os.getppid()
        while os.getppid() == parent:
            await asyncio.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    async def _generate(self, scope, receive, send):
        arrival_ns = time.time_ns()
        raw = await read_body(receive)
        try:
            body = self.dec.decode(raw)
            ids = parse_ids(body.token_ids)
        except (msgspec.ValidationError, msgspec.DecodeError, ValueError) as e:
            return await error(send, str(e))
        if body.sampling_params.max_tokens != 1:
            return await error(send, "this server computes prefill and one output token (max_tokens 1)")
        if not len(ids):
            return await error(send, "empty prompt")
        req_id = next((v.decode() for k, v in scope["headers"] if k == b"x-request-id"), None) \
            or body.rid or uuid.uuid4().hex
        key, self.next_key = self.next_key, self.next_key + 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[key] = fut
        await self.sock.send_multipart([b"Q", self.pack.encode((key, req_id, arrival_ns)), ids.data], copy=False)
        try:
            final_id, tok = await fut
        except RuntimeError as e:
            return await error(send, str(e))
        if body.stream:
            await reply(send, 200, sse_body(req_id, tok, len(ids)), b"text/event-stream")
        else:
            await reply(send, 200, json_body(req_id, tok, len(ids)))
        await self.sock.send_multipart([b"D", self.pack.encode((final_id, time.time_ns()))])


def run_intake(cfg: RunConfig, index: int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the controller stops us (SIGTERM)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind((cfg.host, cfg.port))
    s.listen(4096)
    config = uvicorn.Config(Intake(cfg, index), loop="uvloop", http="httptools", lifespan="on", log_level="warning",
                            access_log=False, timeout_keep_alive=3600)
    uvicorn.Server(config).run(sockets=[s])
