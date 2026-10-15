"""The exact prompt token ids req-frontend sent for the uni_1k requests `bench/batchflip.py` uses (CPU, ~80 s).

    python bench/batchflip_prompts.py <tokenizer.json> measured/batchflip/prompts.npz

req-frontend's rule (`bench/reference/make_prompts.py`): the corpus tokenized line by line into a pool of 100M tokens
(its default floor), row k of the trace starts at pool offset k * 9973 mod the pool length. Prints each prompt's
sha256 (int32 little-endian), which matches `prompt_sha256` in the run's server_requests.jsonl.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from make_prompts import STRIDE, build_pool  # noqa: E402

CORPUS = "/raid/kanzhu/ServingStudio/data/corpus/enwik9"
POOL = 100_000_000
ORDINALS = [3, 15424, 6, 14880, 11, 7908, 29, 10544, 5, 10, 4803, 5328, 8, 7953, 16, *range(100, 107), 0, 1, 2, 4]

if __name__ == "__main__":
    pool = build_pool(CORPUS, Tokenizer.from_file(sys.argv[1]), POOL)[:POOL]
    out = {f"uni_1k_{k}": pool[(k * STRIDE) % POOL:(k * STRIDE) % POOL + 1024] for k in ORDINALS}
    assert all(len(v) == 1024 for v in out.values())
    np.savez(sys.argv[2], **out)
    for k, v in out.items():
        print(k, len(v), hashlib.sha256(np.asarray(v, dtype=np.int32).tobytes()).hexdigest())
