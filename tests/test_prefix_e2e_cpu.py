"""The prefix cache end to end on CPU (5_prefix_cache.md §2): `python -m ppimpl serve` with `keep_sessions` and
FakeStageModel over gloo, sessions sent over HTTP one round at a time, as req-frontend builds them: round k+1 is round
k's prompt, its output token, then new tokens."""

import json
import random

import httpx
import pytest

from conftest import free_port_block
from e2e import Server
from ppimpl.config import BLOCK_TOKENS, STATE_BLOCKS
from ppimpl.model.fake import fake_token

# 30 blocks reported, 29 usable: four 1-block contexts (7 blocks each with their state blocks) fit, a fifth does not.
CPU = {"device": "cpu", "keep_sessions": True,
       "fake": {"us_per_token": 0.5, "hidden_width": 64, "num_blocks": 30}}


class Client:
    def __init__(self, port: int):
        self.c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120)
        self.rng = random.Random(0)
        self.n = 0

    def new_tokens(self, n: int) -> list[int]:
        return [self.rng.randrange(1, 150_000) for _ in range(n)]

    def send(self, rid: str, ids: list[int], token_of: str | None = None) -> tuple[int, int]:
        """One round; returns (token, cached tokens), streamed and not, alternately. The fake model's token is
        fake_token(request id, prompt length): `token_of` names the request whose token a repeat gets."""
        stream = self.n % 2 == 0
        self.n += 1
        r = self.c.post("/inference/v1/generate", headers={"x-request-id": rid},
                        json={"token_ids": ids, "sampling_params": {"max_tokens": 1}, "stream": stream})
        assert r.status_code == 200, r.text
        if stream:
            events = [json.loads(x[6:]) for x in r.text.splitlines() if x.startswith("data: ") and x != "data: [DONE]"]
            tok, usage = events[0]["choices"][0]["token_ids"], events[-1]["usage"]
        else:
            j = r.json()
            tok, usage = j["choices"][0]["token_ids"], j["usage"]
        assert tok == [fake_token(token_of or rid, len(ids))] and usage["prompt_tokens"] == len(ids)
        return tok[0], usage["prompt_tokens_details"]["cached_tokens"]


def chunks_of(run_dir) -> dict[str, list[tuple[int, int]]]:
    """Per request: its (prefix, append) chunks, from stage 0's lines of microbatches.jsonl."""
    out = {}
    for x in map(json.loads, (run_dir / "microbatches.jsonl").read_text().splitlines()):
        if x["stage"] == 0:
            for q in x["requests"]:
                out.setdefault(q["id"], []).append((q["prefix"], q["append"]))
    return out


@pytest.fixture(params=[0, 2], ids=["intake0", "intake2"])
def server(request, tmp_path):
    port = free_port_block()
    srv = Server(tmp_path / "run", port, intake_procs=request.param, **CPU)
    srv.wait_ready()
    yield srv, Client(port)
    assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()


def test_round1_hits_the_whole_previous_prompt(server):
    """Bring-up shape: round 1 = round 0's prompt + its token + A - 1 new tokens; exactly P tokens are cached, not
    rounded to a block, and only A are computed, from position P on."""
    srv, c = server
    P, A = BLOCK_TOKENS + 1000, 3000            # P ends inside its second block
    r0 = c.new_tokens(P)
    tok, cached = c.send("s0_r0", r0)
    assert cached == 0
    r1 = r0 + [tok] + c.new_tokens(A - 1)
    tok1, cached = c.send("s0_r1", r1)
    assert cached == P
    r2 = r1 + [tok1] + c.new_tokens(10)          # round 2 continues from round 1's whole prompt
    assert c.send("s0_r2", r2)[1] == len(r1)
    srv.stop()
    ch = chunks_of(srv.run_dir)
    assert ch["s0_r0"] == [(0, 8192), (8192, P - 8192)] and ch["s0_r1"] == [(P, A)] and ch["s0_r2"] == [(P + A, 11)]
    sreq = {r["id"]: r for r in map(json.loads, (srv.run_dir / "server_requests.jsonl").read_text().splitlines())}
    assert [sreq[i]["cached"] for i in ("s0_r0", "s0_r1", "s0_r2")] == [0, P, P + A]
    cache = json.loads((srv.run_dir / "cache.json").read_text())
    assert cache["hits"] == 2 and cache["hit_tokens"] == 2 * P + A and cache["evicted_sessions"] == 0
    assert cache["kept_sessions"] == 1 and cache["preemptions"] == 0


def test_unaligned_hits_resume_exactly(server):
    """P = 8193 + 1, 2, 3 (not multiples of the indexer's 4-token pool): the hit is still exactly P and round 1's
    first chunk starts at position P (the kpool insert handles the straddling pool, tests/test_kpool_cpu.py)."""
    srv, c = server
    for i, P in enumerate((8193 + 1, 8193 + 2, 8193 + 3)):
        r0 = c.new_tokens(P)
        tok, cached = c.send(f"u{i}_r0", r0)
        assert cached == 0
        assert c.send(f"u{i}_r1", r0 + [tok] + c.new_tokens(999))[1] == P
    srv.stop()
    ch = chunks_of(srv.run_dir)
    for i, P in enumerate((8193 + 1, 8193 + 2, 8193 + 3)):
        assert ch[f"u{i}_r1"][0] == (P, ch[f"u{i}_r1"][0][1]) and sum(a for _, a in ch[f"u{i}_r1"]) == 1000


def test_lru_evicts_whole_sessions_and_an_evicted_session_recomputes(server):
    """29 blocks; a 5,000-token context holds 1 + 6 blocks. A..D fill 28; A's round 1 refreshes it; a new session
    E evicts B (least recently used), not A, and all of B; B's round 1 recomputes its whole prompt and evicts C; D
    still hits."""
    srv, c = server
    n = 5000
    assert 4 * (1 + STATE_BLOCKS) <= 29 < 5 * (1 + STATE_BLOCKS)
    r0 = {s: c.new_tokens(n) for s in "ABCDE"}
    tok = {}
    for s in "ABCD":
        tok[s], cached = c.send(f"{s}_r0", r0[s])
        assert cached == 0
    ra1 = r0["A"] + [tok["A"]] + c.new_tokens(99)              # stays within A's own block: no eviction
    assert c.send("A_r1", ra1)[1] == n
    assert c.send("E_r0", r0["E"])[1] == 0                     # needs 7 blocks, 1 free: B goes
    rb1 = r0["B"] + [tok["B"]] + c.new_tokens(99)
    assert c.send("B_r1", rb1)[1] == 0                          # evicted: no hit; C goes for it
    rd1 = r0["D"] + [tok["D"]] + c.new_tokens(99)
    assert c.send("D_r1", rd1)[1] == n
    rc1 = r0["C"] + [tok["C"]] + c.new_tokens(99)
    assert c.send("C_r1", rc1)[1] == 0
    srv.stop()
    ch = chunks_of(srv.run_dir)
    assert ch["B_r1"] == [(0, n + 100)] and ch["C_r1"] == [(0, n + 100)]   # the whole prompt again, from 0
    assert ch["D_r1"] == [(n, 100)]
    cache = json.loads((srv.run_dir / "cache.json").read_text())
    # B (for E), C (for B_r1), then A for C_r1 (LRU: A_r1 < E_r0 < B_r1 < D_r1): a whole session each time
    assert cache["evicted_sessions"] == 3 and cache["preemptions"] == 0
    assert cache["evicted_tokens"] == 2 * n + (n + 100)


def test_no_hit_unless_the_prompt_starts_with_the_whole_kept_context(server):
    srv, c = server
    r0 = c.new_tokens(3000)
    tok, _ = c.send("x_r0", r0)
    changed = list(r0)
    changed[2000] = changed[2000] % 1000 + 1                    # same first tokens, differs inside the context
    x1 = changed + [tok] + c.new_tokens(50)
    tok1, cached = c.send("x_r1", x1)
    assert cached == 0                                          # and x_r0's context is dropped
    assert c.send("x_r2", r0 + [tok] + c.new_tokens(50))[1] == 0      # x_r0's context is gone
    srv.stop()
    cache = json.loads((srv.run_dir / "cache.json").read_text())
    assert cache["hits"] == 0 and cache["dropped_stale"] == 2


def test_short_contexts_and_exact_repeat(server):
    """A context shorter than the content key still hits. A prompt equal to a kept context (req-frontend's
    preflight sends one prompt twice) has nothing new to compute: it gets that prompt's token, all of it cached,
    and runs no microbatch."""
    srv, c = server
    r0 = c.new_tokens(10)
    tok, _ = c.send("y_r0", r0)
    r1 = r0 + [tok] + c.new_tokens(5)
    assert c.send("y_r1", r1)[1] == 10
    assert c.send("y_r2", list(r1), token_of="y_r1")[1] == len(r1)
    assert c.send("y_r3", list(r0) + [tok])[1] == 0              # a prefix of the kept context: computed again
    srv.stop()
    ch = chunks_of(srv.run_dir)
    assert "y_r2" not in ch and ch["y_r3"] == [(0, 11)]
    cache = json.loads((srv.run_dir / "cache.json").read_text())
    assert cache["repeats"] == 1 and cache["hits"] == 1


def test_off_by_default(tmp_path):
    port = free_port_block()
    srv = Server(tmp_path / "run", port, intake_procs=0, **{**CPU, "keep_sessions": False})
    try:
        srv.wait_ready()
        c = Client(port)
        r0 = c.new_tokens(2000)
        tok, _ = c.send("z_r0", r0)
        assert c.send("z_r1", r0 + [tok] + c.new_tokens(10))[1] == 0
    finally:
        assert srv.stop() == 0
    cache = json.loads((tmp_path / "run/cache.json").read_text())
    assert cache["keep_sessions"] is False and cache["hits"] == 0 and cache["kept_sessions"] == 0
