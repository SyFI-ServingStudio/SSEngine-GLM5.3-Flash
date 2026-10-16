"""Shared by the bench harnesses: repo paths, the handoff inputs, and one request on enwik9 as a stage plan.

Every harness runs from the repo root as a module (`python -m bench.<name>`), so `ppimpl` and `bench` import
normally.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
# The tree that holds the shared env (.venv, third_party/): the repo itself, or the main tree for a lane worktree
# under lanes/.
MAIN = ROOT.parents[1] if ROOT.parent.name == "lanes" else ROOT
MEASURED = ROOT / "measured"
HANDOFF = Path(os.environ.get("PPIMPL_HANDOFF",       # the handoff package (read-only): data/ traces and targets
                              "/raid/kanzhu/ServingStudio/wt-pp-main/doc/handoff/glm53_flash_pp4_b200_v1"))
ENWIK9 = Path(os.environ.get("PPIMPL_ENWIK9", "/raid/kanzhu/ServingStudio/data/corpus/enwik9"))

KV_BLOCKS = 16           # ceil(131072 / 8576)
NUM_BLOCKS = 1 + KV_BLOCKS + 6


def enwik9_tokens(checkpoint: str, n: int) -> list[int]:
    """The first n tokens of enwik9 under the checkpoint's tokenizer."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
    nbytes = 8 * n
    while True:
        with open(ENWIK9, "rb") as f:
            text = f.read(nbytes).decode("utf-8", errors="ignore")
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) >= n:
            return ids[:n]
        nbytes *= 2


def one_request_plan(tokens: list[int]):
    """One request, the whole prompt in one microbatch, on KV blocks 1.. and the 6 state blocks after them."""
    L = len(tokens)
    n = -(-L // 8576)
    chunk = SimpleNamespace(req_id="r0", start=0, num_tokens=L, kv_blocks=list(range(1, 1 + n)),
                            state_blocks=list(range(1 + KV_BLOCKS, 1 + KV_BLOCKS + 6)), is_last=True,
                            token_ids=tokens)
    return SimpleNamespace(mb_id=0, chunks=[chunk])


def sleep_cycles_per_ms() -> float:
    """GPU clock cycles per ms of `torch.cuda._sleep`, for queuing a lead sleep before timed work."""
    import torch

    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cycles = 50_000_000
    a.record()
    torch.cuda._sleep(cycles)
    b.record()
    b.synchronize()
    return cycles / a.elapsed_time(b)
