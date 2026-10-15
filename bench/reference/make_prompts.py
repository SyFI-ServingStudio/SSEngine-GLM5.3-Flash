"""Rebuild the exact prompt token ids req-frontend sends for an independent trace.

req-frontend (alignment/load_generator/req-frontend, src/tokens.rs + executor/independent.rs)
tokenizes the corpus line by line (blank lines skipped, no special tokens) into one
token pool, and gives the trace's k-th row (0-based, file order) the `input_len`
tokens starting at pool offset k * 9973 (mod pool length). The pool is a prefix of
the tokenized corpus, so only the first max(k * 9973 + input_len) tokens matter.

usage: python make_prompts.py <trace.csv> <corpus> <tokenizer.json> [--out DIR]
Prints one JSON line per row: {id, ordinal, prompt_len, prompt_sha256}; --out also
writes <id>.npy (int32).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

STRIDE = 9_973
BATCH_LINES = 256


def build_pool(corpus, tokenizer, need):
    pool, batch = [], []
    with open(corpus, encoding="utf-8", newline="") as f:
        for raw in f:
            line = raw[:-1] if raw.endswith("\n") else raw
            if line.endswith("\r"):
                line = line[:-1]
            if not line.strip():
                continue
            batch.append(line)
            if len(batch) == BATCH_LINES:
                for enc in tokenizer.encode_batch(batch, add_special_tokens=False):
                    pool.extend(enc.ids)
                batch = []
                if len(pool) >= need:
                    break
    if batch and len(pool) < need:
        for enc in tokenizer.encode_batch(batch, add_special_tokens=False):
            pool.extend(enc.ids)
    return np.asarray(pool[:max(need, 1)], dtype=np.int32)


def prompts(trace, corpus, tokenizer_path):
    rows = list(csv.DictReader(open(trace)))
    need = max(i * STRIDE + int(r["input_len"]) for i, r in enumerate(rows))
    pool = build_pool(corpus, Tokenizer.from_file(tokenizer_path), need)
    assert len(pool) >= need, "pool shorter than the trace needs (wraparound not reproduced)"
    for i, r in enumerate(rows):
        start = i * STRIDE
        yield r["id"], i, pool[start:start + int(r["input_len"])]


def sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("corpus")
    ap.add_argument("tokenizer")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for rid, i, ids in prompts(a.trace, a.corpus, a.tokenizer):
        if out:
            np.save(out / f"{rid}.npy", ids)
        print(json.dumps({"id": rid, "ordinal": i, "prompt_len": len(ids), "prompt_sha256": sha256(ids)}))
