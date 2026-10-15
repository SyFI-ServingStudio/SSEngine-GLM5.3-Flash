"""End to end on 4 GPUs: FakeStageModel, NCCL p2p of the real 32 KiB/token activations. Run through Slurm."""

import asyncio
import json

import pytest

from conftest import free_port_block
from e2e import Server, check_microbatches, check_responses, send_all

pytestmark = pytest.mark.gpu


def test_serve_http_gpu(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    us = 1.0
    lengths = [1, 100, 1000, 5000, 8191, 8192, 8193, 20000, 3000, 12000, 7, 16384, 65536] * 3
    prompts = {f"req{i}": n for i, n in enumerate(lengths)}
    port = free_port_block()
    srv = Server(tmp_path / "run", port, device="cuda", fake={"us_per_token": us})
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
    for mb in by_mb.values():                     # stage time = the fake's per-token delay
        T = sum(q["append"] for q in mb[0]["requests"])
        for s in range(4):
            assert mb[s]["compute_ms"] == pytest.approx(T * us / 1e3, rel=0.1, abs=0.05)
    pool = json.loads((tmp_path / "run/pool.json").read_text())
    assert pool["pool_tokens"] == 61_258_368
