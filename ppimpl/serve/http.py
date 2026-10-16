"""HTTP pieces shared by the in-process server (server.py) and the intake processes (intake.py)."""

import msgspec


class SamplingParams(msgspec.Struct):
    max_tokens: int = 1
    temperature: float = 0.0
    ignore_eos: bool = True


_enc = msgspec.json.Encoder()


def usage(prompt_tokens: int, cached: int = 0) -> dict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1,
            "prompt_tokens_details": {"cached_tokens": cached}}


def sse_body(req_id: str, tok: int, n: int) -> bytes:
    """One event with the token, one usage event, [DONE]."""
    token_event = {"id": req_id, "choices": [{"index": 0, "token_ids": [tok], "finish_reason": "length"}],
                   "usage": None}
    usage_event = {"id": req_id, "choices": [], "usage": usage(n)}
    return b"data: " + _enc.encode(token_event) + b"\n\ndata: " + _enc.encode(usage_event) + b"\n\ndata: [DONE]\n\n"


def json_body(req_id: str, tok: int, n: int) -> bytes:
    return _enc.encode({"id": req_id, "choices": [{"index": 0, "token_ids": [tok], "finish_reason": "length"}],
                        "usage": usage(n)})


async def read_body(receive) -> bytes:
    chunks = []
    while True:
        msg = await receive()
        chunks.append(msg.get("body", b""))
        if not msg.get("more_body"):
            return b"".join(chunks)


async def reply(send, status: int, body: bytes, ctype: bytes = b"application/json"):
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", ctype)]})
    await send({"type": "http.response.body", "body": body})


async def error(send, why: str):
    await reply(send, 400, _enc.encode({"error": why}))
