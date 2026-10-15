"""Recording pass-through proxy for the correctness reference.

Sits between req-frontend and the vLLM server. Every request is forwarded
unchanged and its response streamed back unchanged; for POST
/inference/v1/generate it also records what req-frontend sent and what the
server returned, because req-frontend's requests.jsonl keeps neither the prompt
nor the output token ids:

  {"request_id", "prompt_len", "prompt_sha256", "prompt_head", "prompt_tail",
   "output_token_ids", "t_post", "t_first_token", "t_done"}

usage: python record_proxy.py --listen-port P --upstream http://127.0.0.1:Q --log out.jsonl
       [--save-prompts DIR]   # writes <request_id>.npy (int32) per prompt
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import web

HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


def prompt_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--save-prompts")
    args = ap.parse_args()
    log = open(args.log, "a", buffering=1)
    save = Path(args.save_prompts) if args.save_prompts else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    async def on_startup(app):
        app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def on_cleanup(app):
        await app["session"].close()

    async def handle(request: web.Request):
        body = await request.read()
        record = None
        if request.method == "POST" and request.path.endswith("/inference/v1/generate"):
            req = json.loads(body)
            ids = req.get("token_ids") or []
            rid = req.get("request_id") or req.get("rid") or request.headers.get("x-request-id")
            record = {"request_id": rid, "prompt_len": len(ids),
                      "prompt_sha256": prompt_sha256(ids), "prompt_head": ids[:8],
                      "prompt_tail": ids[-8:], "sampling_params": req.get("sampling_params"),
                      "output_token_ids": [], "t_post": time.time()}
            if save and record["request_id"]:
                np.save(save / f"{record['request_id']}.npy", np.asarray(ids, dtype=np.int32))
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        async with request.app["session"].request(
            request.method, args.upstream + request.path_qs, data=body, headers=headers
        ) as up:
            resp = web.StreamResponse(status=up.status, headers={
                k: v for k, v in up.headers.items() if k.lower() not in HOP})
            await resp.prepare(request)
            buf = b""
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
                if record is not None:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            continue
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        for ch in ev.get("choices") or []:
                            toks = ch.get("token_ids") or []
                            if toks and "t_first_token" not in record:
                                record["t_first_token"] = time.time()
                            record["output_token_ids"].extend(toks)
                        if ev.get("usage"):
                            record["usage"] = ev["usage"]
            if record is not None and buf.strip() and not record["output_token_ids"]:
                # Non-streamed response: one JSON body.
                try:
                    ev = json.loads(buf)
                    for ch in ev.get("choices") or []:
                        record["output_token_ids"].extend(ch.get("token_ids") or [])
                except json.JSONDecodeError:
                    pass
            await resp.write_eof()
        if record is not None:
            record["status"] = up.status
            record["t_done"] = time.time()
            log.write(json.dumps(record) + "\n")
        return resp

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host="127.0.0.1", port=args.listen_port, access_log=None, print=None)


if __name__ == "__main__":
    main()
