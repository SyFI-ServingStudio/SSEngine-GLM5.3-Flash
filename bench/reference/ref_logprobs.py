"""Greedy first token and its top-k logprobs for saved prompts, straight from the server.

Sends each prompt (<request_id>.npy from record_proxy.py --save-prompts) to POST
/inference/v1/generate, not streamed, with max_tokens 1, temperature 0 and
`logprobs` = k, one request at a time. Writes one JSON line per prompt:
{request_id, prompt_len, prompt_sha256, token_id, top_logprobs: [{token_id, token, logprob}], raw_logprobs}

usage: python ref_logprobs.py --base-url http://127.0.0.1:P --prompts DIR --out out.jsonl [--k 5]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import requests


def token_id(entry):
    """Token id of a logprob entry: tokens-only servers render it as 'token_id:N'."""
    tok = entry.get("token")
    if isinstance(tok, str):
        m = re.fullmatch(r"token_id:(\d+)", tok)
        if m:
            return int(m.group(1))
    return entry.get("token_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model")
    a = ap.parse_args()
    with open(a.out, "w") as out:
        for path in sorted(Path(a.prompts).glob("*.npy")):
            ids = np.load(path).astype(np.int32)
            body = {"token_ids": ids.tolist(), "stream": False,
                    "sampling_params": {"max_tokens": 1, "temperature": 0.0, "ignore_eos": True,
                                        "logprobs": a.k}}
            if a.model:
                body["model"] = a.model
            r = requests.post(a.base_url + "/inference/v1/generate", json=body, timeout=600)
            rec = {"request_id": path.stem, "prompt_len": int(ids.size),
                   "prompt_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "status": r.status_code}
            if r.ok:
                ch = r.json()["choices"][0]
                rec["token_id"] = (ch.get("token_ids") or [None])[0]
                lp = ch.get("logprobs") or {}
                content = (lp.get("content") or [{}])[0]
                rec["top_logprobs"] = [{"token_id": token_id(e), "token": e.get("token"),
                                        "logprob": e.get("logprob")}
                                       for e in content.get("top_logprobs") or []]
                rec["raw_logprobs"] = content
            else:
                rec["error"] = r.text[:2000]
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec["request_id"], rec.get("token_id"), [t["token_id"] for t in rec.get("top_logprobs", [])],
                  flush=True)


if __name__ == "__main__":
    main()
