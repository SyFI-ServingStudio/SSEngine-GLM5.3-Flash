"""Intake process (RunConfig.intake_procs > 0): one of N HTTP servers on the same port (SO_REUSEPORT, the kernel
spreads connections over them), like vLLM's api-server-count. It parses a request's token ids into int32 bytes
(ppimpl/serve/ids.py) and
passes them to the controller's control thread over zmq; the token comes back the same way. The OpenAI-compatible
endpoints (ppimpl/serve/openai.py) are served here too, their text tokenized here. The controller process
then serves no HTTP, so a burst of requests no longer queues behind one event loop.

DEALER (identity = index) <-> controller ROUTER (RunConfig.intake_addr), frames:
  intake -> controller: [b"R", slab name] ready
                        | [b"Q", msgpack (key, req_id, arrival_ns, body_ns, parsed_ns), int32 ids] request
                        | [b"M", the same msgpack, msgpack (slab name, offset, count)] request whose ids are in the
                          intake's shared-memory slab (ppimpl/serve/slab.py), freed when the request is answered
                        (body_ns: body read; parsed_ns: ids parsed, just before the send)
                        | [b"D", msgpack (req_id, response_ns)] response sent | [b"S"] /shutdown
  controller -> intake: [b"G"] all intakes ready (/health 200 from now on)
                        | [b"A", msgpack (key, req_id, token, cached tokens)] | [b"E", msgpack (key, why)]
"""

import asyncio
import os
import signal
import socket
import sys
import time
import uuid

import msgspec
import numpy as np
import uvicorn
import zmq
import zmq.asyncio

from ppimpl.config import RunConfig
from ppimpl.serve import ids as ids_mod
from ppimpl.serve.slab import SLAB_FROM, Slab
from ppimpl.serve.http import SamplingParams, error, json_body, read_body, reply, sse_body
from ppimpl.serve.openai import PATHS as OPENAI_PATHS, OpenAI


class RawRequest(msgspec.Struct):
    token_ids: msgspec.Raw                # parsed by numpy straight into int32, without a list of ints
    sampling_params: SamplingParams = msgspec.field(default_factory=SamplingParams)
    stream: bool = False
    model: str | None = None
    rid: str | None = None


class RestRequest(msgspec.Struct):
    """The body with its token array cut out (`ids.parse_fast`): token_ids must decode to []."""
    token_ids: list[int]
    sampling_params: SamplingParams = msgspec.field(default_factory=SamplingParams)
    stream: bool = False
    model: str | None = None
    rid: str | None = None


parse_ids = ids_mod.parse_slow_ids


def parse_body(raw: bytes, dec: msgspec.json.Decoder, rest_dec: msgspec.json.Decoder):
    """(request, int32 ids): the fast path of ppimpl/serve/ids.py, else msgspec + numpy (same answer, or the error)."""
    got = ids_mod.parse_fast(raw, rest_dec)
    if got is not None:
        return got
    body = dec.decode(raw)
    return body, parse_ids(body.token_ids)


class Intake:
    def __init__(self, cfg: RunConfig, index: int):
        self.cfg, self.index = cfg, index
        self.dec = msgspec.json.Decoder(RawRequest)
        self.rest_dec = msgspec.json.Decoder(RestRequest)
        self.pack = msgspec.msgpack.Encoder()
        self.unpack = msgspec.msgpack.Decoder()
        self.pending: dict[int, asyncio.Future] = {}
        self.in_slab: dict[int, int] = {}      # key -> slab offset of its ids
        self.next_key = 0
        self.go = False
        self.openai = OpenAI(cfg) if cfg is not None else None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self._lifespan(receive, send)
        method, path = scope["method"], scope["path"]
        if path == "/inference/v1/generate" and method == "POST":
            return await self._generate(scope, receive, send)
        if path in OPENAI_PATHS:
            return await self.openai.handle(scope, receive, send, self._submit, self._sent)
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
                ids_mod.warm()
                self.slab = Slab()
                await self.sock.send_multipart([b"R", self.slab.name.encode()])
                self.tasks = [asyncio.create_task(self._answers()), asyncio.create_task(self._watch_parent())]
            await send({"type": f"lifespan.{kind}.complete"})
            if kind == "shutdown":
                for t in self.tasks:
                    t.cancel()
                self.sock.close(linger=0)
                self.ctx.term()
                self.slab.unlink()
                return

    async def _answers(self):
        while True:
            frames = await self.sock.recv_multipart()
            kind = frames[0]
            if kind == b"A":
                key, req_id, tok, cached = self.unpack.decode(frames[1])
                self._free(key)
                fut = self.pending.pop(key, None)
                if fut is not None and not fut.done():
                    fut.set_result((req_id, tok, cached))
            elif kind == b"E":
                key, why = self.unpack.decode(frames[1])
                self._free(key)
                fut = self.pending.pop(key, None)
                if fut is not None and not fut.done():
                    fut.set_exception(RuntimeError(why))
            elif kind == b"G":
                self.go = True
                self.slab.unlink()        # the controller has it mapped

    async def _watch_parent(self):
        parent = os.getppid()
        while os.getppid() == parent:
            await asyncio.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    def _free(self, key: int) -> None:
        off = self.in_slab.pop(key, None)
        if off is not None:
            self.slab.free(off)

    def _alloc(self, n: int):
        off = self.slab.alloc(4 * n) if 4 * n >= SLAB_FROM else None
        return None if off is None else (off, self.slab.ids(off, n))

    async def _generate(self, scope, receive, send):
        arrival_ns = time.time_ns()
        size = next((int(v) for k, v in scope["headers"] if k == b"content-length"), 0)
        off = None
        if size >= ids_mod.STREAM_FROM:      # parse the ids while the body arrives
            sp = ids_mod.StreamParser(self.rest_dec)
            while True:
                msg = await receive()
                sp.feed(msg.get("body", b""))
                if not msg.get("more_body"):
                    break
            body_ns = time.time_ns()
            got = sp.finish(self._alloc)
            raw = None if got is not None else sp.body()
        else:
            raw = await read_body(receive)
            body_ns = time.time_ns()
        if raw is not None:
            try:
                body, ids = parse_body(raw, self.dec, self.rest_dec)
            except (msgspec.ValidationError, msgspec.DecodeError, ValueError) as e:
                return await error(send, str(e))
            dst = self._alloc(len(ids))
            if dst is not None:
                off, dst[1][:] = dst[0], ids
        else:
            body, ids, off = got
        if body.sampling_params.max_tokens != 1 or not len(ids):
            if off is not None:
                self.slab.free(off)
            return await error(send, "empty prompt" if body.sampling_params.max_tokens == 1 else
                               "this server computes prefill and one output token (max_tokens 1)")
        req_id = next((v.decode() for k, v in scope["headers"] if k == b"x-request-id"), None) \
            or body.rid or uuid.uuid4().hex
        try:
            final_id, tok, cached = await self._send(ids, off, req_id, arrival_ns, body_ns)
        except RuntimeError as e:
            return await error(send, str(e))
        if body.stream:
            await reply(send, 200, sse_body(req_id, tok, len(ids), cached), b"text/event-stream")
        else:
            await reply(send, 200, json_body(req_id, tok, len(ids), cached))
        await self._sent(final_id)

    async def _send(self, ids: np.ndarray, off: int | None, req_id: str, arrival_ns: int, body_ns: int):
        """Pass a request to the controller (its ids at slab offset `off`, else in the message); (final request id,
        token, cached prompt tokens) when answered, RuntimeError with the controller's reason when refused."""
        key, self.next_key = self.next_key, self.next_key + 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[key] = fut
        meta = self.pack.encode((key, req_id, arrival_ns, body_ns, time.time_ns()))
        if off is not None:
            self.in_slab[key] = off
            await self.sock.send_multipart([b"M", meta, self.pack.encode((self.slab.name, off, len(ids)))])
        else:
            await self.sock.send_multipart([b"Q", meta, ids.data], copy=False)
        return await fut

    async def _submit(self, ids: np.ndarray, req_id: str, arrival_ns: int, body_ns: int):
        """openai.Submit: the ids in the slab when large, as a parsed /inference/v1/generate body."""
        dst = self._alloc(len(ids))
        off = None
        if dst is not None:
            off, dst[1][:] = dst[0], ids
        final_id, tok, cached = await self._send(ids, off, req_id, arrival_ns, body_ns)
        return tok, cached, final_id

    async def _sent(self, final_id: str):
        await self.sock.send_multipart([b"D", self.pack.encode((final_id, time.time_ns()))])


# The ids parse runs on threads while the event loop reads the body (ids.StreamParser); each piece needs the GIL to
# start and to return. Python's default switch interval (5 ms) let them wait for a busy event loop that long.
SWITCH_INTERVAL_S = 5e-5


def run_intake(cfg: RunConfig, index: int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the controller stops us (SIGTERM)
    sys.setswitchinterval(SWITCH_INTERVAL_S)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind((cfg.host, cfg.port))
    s.listen(4096)
    config = uvicorn.Config(Intake(cfg, index), loop="uvloop", http="httptools", lifespan="on", log_level="warning",
                            access_log=False, timeout_keep_alive=3600)
    uvicorn.Server(config).run(sockets=[s])
