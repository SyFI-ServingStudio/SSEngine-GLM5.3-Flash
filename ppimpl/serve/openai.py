"""The OpenAI-compatible endpoints, served by the intake processes (intake.py) and by the in-process server
(server.py, intake_procs 0) next to req-frontend's token endpoint:

GET  /v1/models             the one model, under RunConfig.served_model_name
POST /v1/completions        `prompt`: a string or a list of token ids
POST /v1/chat/completions   `messages` through the checkpoint's chat template (`tools`, `chat_template_kwargs` too)

The engine computes prefill and one output token (greedy, PLAN.md §1: no decode path), so `max_tokens` (or
`max_completion_tokens`) must be 1 and defaults to 1; `temperature`, `top_p` and the other sampling fields are taken
and ignored. `n` > 1, `echo` and `logprobs` are refused. With `"stream": true` the answer is one chunk, then the usage
chunk if `stream_options.include_usage`, then `[DONE]`.

Text goes through the checkpoint's tokenizer.json (`tokenizers`, loaded by each process at its first request here,
~0.2 GB; transformers would bring torch into every intake) and its chat template, rendered as transformers renders it
(`render_chat`; tests/test_openai.py compares the two).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import msgspec
import numpy as np

from ppimpl.config import CHECKPOINT, RunConfig
from ppimpl.serve.http import read_body, reply

MAX_MODEL_LEN = 1_048_576       # ppimpl/model/stage.py MAX_MODEL_LEN
PATHS = ("/v1/models", "/v1/completions", "/v1/chat/completions")


class StreamOptions(msgspec.Struct):
    include_usage: bool = False


class CompletionRequest(msgspec.Struct):
    prompt: str | list[int]
    model: str | None = None
    max_tokens: int | None = None
    n: int = 1
    echo: bool = False
    logprobs: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None


class ChatRequest(msgspec.Struct):
    messages: list[dict[str, Any]]
    model: str | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int = 1
    logprobs: bool = False
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "BadRequestError"):
        super().__init__(message)
        self.status, self.kind = status, kind


_enc = msgspec.json.Encoder()

# submit(ids, request id, arrival ns, body ns) -> (token, cached prompt tokens, ticket); sent(ticket) after the reply
Submit = Callable[[np.ndarray, str, int, int], Awaitable[tuple[int, int, Any]]]
Sent = Callable[[Any], Awaitable[None]]


def render_chat(template: str, messages: list, tools: list | None, kwargs: dict | None) -> str:
    """A chat template rendered as transformers' `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`
    does (its sandboxed jinja environment, loop controls, and a `tojson` that takes json.dumps' arguments)."""
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
    return env.from_string(template).render(messages=messages, tools=tools, add_generation_prompt=True,
                                            **(kwargs or {}))


class Text:
    """The checkpoint's tokenizer and chat template."""

    def __init__(self, checkpoint: str):
        from tokenizers import Tokenizer

        d = Path(checkpoint)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.loads((d / "tokenizer_config.json").read_text())
        jinja = d / "chat_template.jinja"
        self.template = jinja.read_text() if jinja.is_file() else cfg.get("chat_template")
        gen = d / "generation_config.json"
        eos = json.loads(gen.read_text()).get("eos_token_id") if gen.is_file() else None
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        if cfg.get("eos_token") and self.tok.token_to_id(cfg["eos_token"]) is not None:
            eos.append(self.tok.token_to_id(cfg["eos_token"]))
        self.eos = set(eos)

    def encode(self, text: str, special: bool = True) -> list[int]:
        return self.tok.encode(text, add_special_tokens=special).ids

    def chat(self, req: ChatRequest) -> list[int]:
        if not self.template:
            raise ApiError(400, "the checkpoint has no chat template: use /v1/completions")
        # the template writes the special tokens itself ([gMASK]<sop>), so none are added (as vLLM)
        return self.encode(render_chat(self.template, req.messages, req.tools, req.chat_template_kwargs), False)

    def decode(self, tok: int) -> str:
        return self.tok.decode([tok], skip_special_tokens=False)


class OpenAI:
    def __init__(self, cfg: RunConfig):
        self.name = cfg.served_model_name or cfg.model
        self.max_len = cfg.max_model_len or MAX_MODEL_LEN
        self.checkpoint = cfg.checkpoint or CHECKPOINT
        self.created = int(time.time())
        self._text: asyncio.Future | None = None
        self.completion_dec = msgspec.json.Decoder(CompletionRequest)
        self.chat_dec = msgspec.json.Decoder(ChatRequest)

    async def text(self) -> Text:
        if self._text is None:
            self._text = asyncio.ensure_future(asyncio.to_thread(Text, self.checkpoint))
        return await self._text

    async def handle(self, scope, receive, send, submit: Submit, sent: Sent):
        """One request to a path of PATHS."""
        arrival_ns = time.time_ns()
        method, path = scope["method"], scope["path"]
        try:
            if path == "/v1/models":
                if method != "GET":
                    raise ApiError(405, f"{method} {path}", "MethodNotAllowed")
                return await reply(send, 200, _enc.encode(self.models()))
            if method != "POST":
                raise ApiError(405, f"{method} {path}", "MethodNotAllowed")
            raw = await read_body(receive)
            body_ns = time.time_ns()
            chat = path == "/v1/chat/completions"
            try:
                req = (self.chat_dec if chat else self.completion_dec).decode(raw)
            except (msgspec.ValidationError, msgspec.DecodeError) as e:
                raise ApiError(400, str(e))
            self.check(req, chat)
            text = await self.text()
            if chat:
                ids = await asyncio.to_thread(text.chat, req)
            else:
                ids = req.prompt if isinstance(req.prompt, list) else await asyncio.to_thread(text.encode, req.prompt)
            if not ids:
                raise ApiError(400, "empty prompt")
            if len(ids) + 1 > self.max_len:
                raise ApiError(400, f"This model's maximum context length is {self.max_len} tokens. However, you "
                               f"requested {len(ids) + 1} tokens ({len(ids)} in the prompt, 1 in the completion).")
            if max(ids) >= text.tok.get_vocab_size(with_added_tokens=True) or min(ids) < 0:
                raise ApiError(400, "prompt token id out of the vocabulary")
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            try:
                tok, cached, ticket = await submit(np.asarray(ids, np.int32), rid, arrival_ns, body_ns)
            except RuntimeError as e:
                raise ApiError(400, str(e))
        except ApiError as e:
            return await reply(send, e.status, _enc.encode(
                {"error": {"message": str(e), "type": e.kind, "param": None, "code": e.status}}))
        out = self.answer(chat, rid, tok, text, len(ids), cached, req)
        await reply(send, 200, out, b"text/event-stream" if req.stream else b"application/json")
        await sent(ticket)

    def models(self) -> dict:
        return {"object": "list", "data": [{"id": self.name, "object": "model", "created": self.created,
                                            "owned_by": "ppimpl", "root": self.checkpoint, "parent": None,
                                            "max_model_len": self.max_len}]}

    def check(self, req, chat: bool) -> None:
        if req.model is not None and req.model != self.name:
            raise ApiError(404, f"The model `{req.model}` does not exist.", "NotFoundError")
        max_tokens = (req.max_completion_tokens or req.max_tokens) if chat else req.max_tokens
        if max_tokens not in (None, 1):
            raise ApiError(400, f"max_tokens {max_tokens}: this engine computes prefill and one output token "
                                "(no decode), so max_tokens must be 1")
        if req.n != 1:
            raise ApiError(400, "n must be 1")
        if req.logprobs or (not chat and req.echo):
            raise ApiError(400, "logprobs and echo are not supported")

    def answer(self, chat: bool, rid: str, tok: int, text: Text, n: int, cached: int, req) -> bytes:
        piece = text.decode(tok)
        finish = "stop" if tok in text.eos else "length"
        usage = {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        head = {"id": rid, "created": int(time.time()), "model": self.name}
        if not req.stream:
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": piece}, "logprobs": None,
                          "finish_reason": finish}
                return _enc.encode({**head, "object": "chat.completion", "choices": [choice], "usage": usage})
            choice = {"index": 0, "text": piece, "logprobs": None, "finish_reason": finish}
            return _enc.encode({**head, "object": "text_completion", "choices": [choice], "usage": usage})
        obj = "chat.completion.chunk" if chat else "text_completion"
        choice = {"index": 0, "delta": {"role": "assistant", "content": piece}} if chat else \
            {"index": 0, "text": piece}
        events = [{**head, "object": obj, "choices": [{**choice, "logprobs": None, "finish_reason": finish}]}]
        if req.stream_options is not None and req.stream_options.include_usage:
            events.append({**head, "object": obj, "choices": [], "usage": usage})
        return b"".join(b"data: " + _enc.encode(e) + b"\n\n" for e in events) + b"data: [DONE]\n\n"
