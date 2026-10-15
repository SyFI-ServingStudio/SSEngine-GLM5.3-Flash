"""HTTP front of the controller: req-frontend's `vllm-tokens` backend and `/health`.

POST /inference/v1/generate  {"token_ids": [...], "sampling_params": {"max_tokens": 1, ...}, "stream": true}
  stream: SSE, one event with the token id (`choices[0].token_ids`), one usage event, `[DONE]`;
  otherwise one JSON body with both. `usage.prompt_tokens_details.cached_tokens` is 0 in phase A.
  The request id is the `x-request-id` header (req-frontend sends the trace id), else the body's `rid`.
GET /health  200 once all stages are ready.
POST /shutdown  graceful stop (bench/run.py).
"""

import asyncio
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager

import msgspec
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ppimpl.config import RunConfig
from ppimpl.runtime.controller import Controller, RequestError


class SamplingParams(msgspec.Struct):
    max_tokens: int = 1
    temperature: float = 0.0
    ignore_eos: bool = True


class GenerateRequest(msgspec.Struct):
    token_ids: list[int]
    sampling_params: SamplingParams = msgspec.field(default_factory=SamplingParams)
    stream: bool = False
    model: str | None = None
    rid: str | None = None


def _usage(prompt_tokens: int, cached: int = 0) -> dict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1,
            "prompt_tokens_details": {"cached_tokens": cached}}


def build_app(cfg: RunConfig) -> FastAPI:
    ctl = Controller(cfg)
    enc = msgspec.json.Encoder()

    @asynccontextmanager
    async def lifespan(app):
        await ctl.start()
        yield
        await ctl.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.controller = ctl

    @app.get("/health")
    async def health():
        ok = ctl.ready and not ctl.failed
        return Response(status_code=200 if ok else 503)

    @app.post("/shutdown")
    async def shutdown():
        """Graceful stop by request: under `srun nsys profile`, signalling the server process itself is not possible
        from the client side, and nsys writes its report only when the server exits on its own."""
        asyncio.get_running_loop().call_later(0.1, os.kill, os.getpid(), signal.SIGTERM)
        return Response(status_code=200)

    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        arrival_ns = time.time_ns()
        try:
            body = msgspec.json.decode(await request.body(), type=GenerateRequest)
        except msgspec.ValidationError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if body.sampling_params.max_tokens != 1:
            return JSONResponse({"error": "this server computes prefill and one output token (max_tokens 1)"},
                                status_code=400)
        req_id = request.headers.get("x-request-id") or body.rid or uuid.uuid4().hex
        try:
            fut = ctl.submit(req_id, body.token_ids, arrival_ns)
        except RequestError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        n = len(body.token_ids)
        if not body.stream:
            tok, info = await fut
            ctl.log_response(info)
            return Response(enc.encode({"id": req_id, "choices": [{"index": 0, "token_ids": [tok],
                                        "finish_reason": "length"}], "usage": _usage(n)}),
                            media_type="application/json")

        async def events():
            tok, info = await fut
            token_event = {"id": req_id, "choices": [{"index": 0, "token_ids": [tok], "finish_reason": "length"}],
                           "usage": None}
            usage_event = {"id": req_id, "choices": [], "usage": _usage(n)}
            yield b"data: " + enc.encode(token_event) + b"\n\ndata: " + enc.encode(usage_event) + b"\n\ndata: [DONE]\n\n"
            ctl.log_response(info)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def serve(cfg: RunConfig):
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, loop="uvloop", http="httptools",
                log_level="warning", access_log=False, backlog=4096, timeout_keep_alive=3600)
