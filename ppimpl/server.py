"""HTTP front of the controller: req-frontend's `vllm-tokens` backend and `/health`.

POST /inference/v1/generate  {"token_ids": [...], "sampling_params": {"max_tokens": 1, ...}, "stream": true}
  stream: SSE, one event with the token id (`choices[0].token_ids`), one usage event, `[DONE]`;
  otherwise one JSON body with both. `usage.prompt_tokens_details.cached_tokens` is 0 in phase A.
  The request id is the `x-request-id` header (req-frontend sends the trace id), else the body's `rid`.
GET /health  200 once all stages are ready.
POST /shutdown  graceful stop (bench/run.py).
"""

import asyncio
import concurrent.futures
import os
import signal
import sys
import time
import traceback
import uuid

import msgspec
import uvicorn

from ppimpl.config import RunConfig
from ppimpl.http import SamplingParams, error, json_body, read_body, reply, sse_body
from ppimpl.runtime.controller import Controller, RequestError


class GenerateRequest(msgspec.Struct):
    token_ids: list[int]
    sampling_params: SamplingParams = msgspec.field(default_factory=SamplingParams)
    stream: bool = False
    model: str | None = None
    rid: str | None = None


class App:
    """A bare ASGI app: a burst of 2048 requests is the hot path, and a framework's per-request layers cost more
    there than decoding the body (py-spy, saturated_8k)."""

    def __init__(self, cfg: RunConfig):
        self.ctl = Controller(cfg)
        self.dec = msgspec.json.Decoder(GenerateRequest)
        # Bodies are decoded off the event loop: a burst arrives within one loop iteration, and decoding 2048 bodies
        # of 8k tokens there (~0.2 ms each) holds back every answer that becomes ready meanwhile.
        self.decoder = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="decode")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self._lifespan(receive, send)
        method, path = scope["method"], scope["path"]
        if path == "/inference/v1/generate" and method == "POST":
            return await self._generate(scope, receive, send)
        if path == "/health" and method == "GET":
            ok = self.ctl.ready and not self.ctl.failed
            return await reply(send, 200 if ok else 503, b"")
        if path == "/shutdown" and method == "POST":
            # Graceful stop by request: under `srun nsys profile`, signalling the server process itself is not
            # possible from the client side, and nsys writes its report only when the server exits on its own.
            asyncio.get_running_loop().call_later(0.1, os.kill, os.getpid(), signal.SIGTERM)
            return await reply(send, 200, b"")
        await reply(send, 404, b"")

    async def _lifespan(self, receive, send):
        while True:
            msg = await receive()
            kind = msg["type"].rsplit(".", 1)[1]
            try:
                await (self.ctl.start() if kind == "startup" else self.ctl.stop())
            except Exception:
                await send({"type": f"lifespan.{kind}.failed", "message": traceback.format_exc()})
                raise
            await send({"type": f"lifespan.{kind}.complete"})
            if kind == "shutdown":
                return

    async def _generate(self, scope, receive, send):
        arrival_ns = time.time_ns()
        raw = await read_body(receive)
        try:
            body = await asyncio.get_running_loop().run_in_executor(self.decoder, self.dec.decode, raw)
        except (msgspec.ValidationError, msgspec.DecodeError) as e:
            return await error(send, str(e))
        if body.sampling_params.max_tokens != 1:
            return await error(send, "this server computes prefill and one output token (max_tokens 1)")
        req_id = next((v.decode() for k, v in scope["headers"] if k == b"x-request-id"), None) \
            or body.rid or uuid.uuid4().hex
        try:
            fut = self.ctl.submit(req_id, body.token_ids, arrival_ns)
        except RequestError as e:
            return await error(send, str(e))
        n = len(body.token_ids)
        if body.stream:
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            tok, info = await fut
            await send({"type": "http.response.body", "body": sse_body(req_id, tok, n)})
        else:
            tok, info = await fut
            await reply(send, 200, json_body(req_id, tok, n))
        self.ctl.log_response(info)


def build_app(cfg: RunConfig) -> App:
    return App(cfg)


def serve(cfg: RunConfig):
    sys.setswitchinterval(0.0005)   # the control thread gets the GIL within 0.5 ms while the event loop takes a burst
    if cfg.intake_procs:            # HTTP in N intake processes (ppimpl/intake.py); here only the controller
        import uvloop
        return uvloop.run(_serve_controller(cfg))
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, loop="uvloop", http="httptools", lifespan="on",
                log_level="warning", access_log=False, backlog=4096, timeout_keep_alive=3600)


async def _serve_controller(cfg: RunConfig):
    """Until SIGTERM (bench/run.py, or /shutdown at an intake, or a failed pipeline): then a graceful stop."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    ctl = Controller(cfg)
    try:
        await ctl.start()
        await stop.wait()
    finally:
        await ctl.stop()
