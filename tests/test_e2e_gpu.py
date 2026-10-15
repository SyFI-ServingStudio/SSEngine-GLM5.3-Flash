"""End to end on 4 GPUs: FakeStageModel, NCCL p2p of the real 32 KiB/token activations. Run through Slurm."""

import asyncio
import json

import pytest

from conftest import free_port_block
from e2e import Server, check_microbatches, check_responses, send_all

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("transport", ["nccl", "ce"])
def test_serve_http_gpu(tmp_path, transport):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    us = 1.0
    lengths = [1, 100, 1000, 5000, 8191, 8192, 8193, 20000, 3000, 12000, 7, 16384, 65536] * 3
    prompts = {f"req{i}": n for i, n in enumerate(lengths)}
    port = free_port_block()
    srv = Server(tmp_path / "run", port, device="cuda", transport=transport, fake={"us_per_token": us})
    try:
        srv.wait_ready()
        res = asyncio.run(send_all(port, prompts))
        one = asyncio.run(send_all(port, {"solo": 131072}))     # a whole 128k prompt, chunked to 16 microbatches
    finally:
        assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()
    check_responses(prompts, res)                 # tokens carry the transport check (fake.py)
    check_responses({"solo": 131072}, one)
    by_mb, peak = check_microbatches(tmp_path / "run", {**prompts, "solo": 131072})
    assert peak >= 3
    off = []                                      # stage time = the fake's per-token delay + launching its few kernels
    for i, mb in by_mb.items():
        T = sum(q["append"] for q in mb[0]["requests"])
        ms = T * us / 1e3
        off += [(i, s, T, mb[s]["compute_ms"]) for s in range(4) if not ms * 0.95 <= mb[s]["compute_ms"] <= ms * 1.1 + 0.5]
    assert not off, off                           # (microbatch, stage, tokens, compute_ms)
    pool = json.loads((tmp_path / "run/pool.json").read_text())
    assert pool["pool_tokens"] == 61_258_368
