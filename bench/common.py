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
# A ServingStudioSim checkout of branch pp-main (env/make_env.sh clones it): req-frontend, the client.
SIM = Path(os.environ.get("PPIMPL_SIM") or MAIN / "third_party" / "ServingStudioSim")
# The frozen handoff in use (read-only: data/ traces, targets and the microbatch fit): glm53_flash_pp4_b200_v2 at
# ServingStudioSim 5153041a, linked at third_party/handoff (PPIMPL_HANDOFF overrides it).
HANDOFF = Path(os.environ.get("PPIMPL_HANDOFF") or MAIN / "third_party" / "handoff")
REQ_FRONTEND = SIM / "alignment" / "load_generator" / "req-frontend"
# The prompt corpus (http://mattmahoney.net/dc/enwik9.zip, unzipped; env/make_env.sh fetches it).
ENWIK9 = Path(os.environ.get("PPIMPL_ENWIK9") or MAIN / "third_party" / "enwik9")
# The ServingStudioSim profiler package (`profiling`) at the handoff's frozen commit 7bb224a6 (env/make_env.sh).
PROFILER = MAIN / "third_party" / "sim_7bb224a6"

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


def free_port_block(start: int, width: int = 10, step: int = 10, below: int = 32768) -> int:
    """The first base port from `start` (in steps of `step`) whose `width` ports nothing listens on: a server uses
    HTTP = base, then base + 1 .. base + 2 + stages (zmq, torch.distributed, intake). Other jobs on the node (other
    lanes, other users, a leftover server) can hold any of them: 7476's cache-off server lost its intake port to one.
    Below the ephemeral range, so client connections never hold them."""
    import socket

    for base in range(start, below - width, step):
        free = True
        for p in range(base, base + width):
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    free = False
                    break
        if free:
            return base
    raise RuntimeError(f"no {width} free ports from {start}")
