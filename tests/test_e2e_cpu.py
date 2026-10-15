"""End to end on CPU: `python -m ppimpl serve` with FakeStageModel over gloo, driven over HTTP and by req-frontend."""

import asyncio
import json
from pathlib import Path

import pytest

import run as bench_run
from conftest import free_port_block
from e2e import Server, blocks_per_request, check_microbatches, check_responses, send_all

CPU = {"device": "cpu", "fake": {"us_per_token": 2.0, "hidden_width": 64, "num_blocks": 30}}


def test_serve_http_cpu(tmp_path):
    lengths = [1, 100, 1000, 5000, 8191, 8192, 8193, 20000, 3000, 12000, 7, 16384] * 2
    prompts = {f"req{i}": n for i, n in enumerate(lengths)}
    assert max(blocks_per_request(n) for n in lengths) <= 29 < sum(blocks_per_request(n) for n in lengths)
    port = free_port_block()
    srv = Server(tmp_path / "run", port, **CPU)
    try:
        srv.wait_ready()
        res = asyncio.run(send_all(port, prompts))
    finally:
        assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()
    check_responses(prompts, res)
    by_mb, peak = check_microbatches(tmp_path / "run", prompts)
    assert peak >= 2                       # the stages did overlap
    # pool of 29 blocks: requests had to wait for blocks and were admitted later, in arrival order
    first_mb = {}
    for idx in sorted(by_mb):
        for q in by_mb[idx][0]["requests"]:
            first_mb.setdefault(q["id"], idx)
    sreq = [json.loads(x) for x in (tmp_path / "run/server_requests.jsonl").read_text().splitlines()]
    assert {r["id"] for r in sreq} == set(prompts)
    by_seq = [first_mb[r["id"]] for r in sorted(sreq, key=lambda r: r["seq"])]
    assert by_seq == sorted(by_seq)        # first chunks go out in arrival order
    assert by_seq[-1] > 4                  # and late ones waited for blocks
    pool = json.loads((tmp_path / "run/pool.json").read_text())
    assert pool["num_blocks"] == 29


def test_rejects_what_it_cannot_serve(tmp_path):
    port = free_port_block()
    srv = Server(tmp_path / "run", port, **CPU)
    try:
        srv.wait_ready()
        import httpx
        u = f"http://127.0.0.1:{port}/inference/v1/generate"
        assert httpx.post(u, json={"token_ids": [1, 2], "sampling_params": {"max_tokens": 4}}).status_code == 400
        assert httpx.post(u, json={"token_ids": [1] * (24 * 8576)}).status_code == 400   # larger than the pool
        assert httpx.post(u, json={"token_ids": []}).status_code == 400
        assert httpx.post(u, json={"token_ids": [1, 2, 3]}).status_code == 200
    finally:
        assert srv.stop() == 0


def test_req_frontend_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if not bench_run.CLIENT_BIN.exists():
        pytest.skip(f"req-frontend not built: {bench_run.CLIENT_BIN}")
    n, length = 24, 3000
    trace = tmp_path / "uni_tiny.csv"
    trace.write_text("id,input_len,output_len,arrival_time\n"
                     + "".join(f"uni_tiny_{i},{length},1,0\n" for i in range(n)))
    run_dir = tmp_path / "runs" / "saturated_tiny"
    port = free_port_block()
    meta = bench_run.execute(
        run_dir, {"model": "fake", "scheduler": {"mb_tokens": 8192}, **CPU},
        bench_run.client_config(trace, port, run_dir, token_pool_limit=200_000),
        port=port, meta={"name": "saturated_tiny", "kind": "saturated", "prompt_tokens": length, "fake": True,
                         "trace": str(trace)},
        client_timeout_s=300, health_timeout_s=120, gpus=None)
    assert meta["client_returncode"] == 0, (run_dir / "client.log").read_text()
    assert meta["server_returncode"] == 0, (run_dir / "server.log").read_text()
    rows = [json.loads(x) for x in (run_dir / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == n
    for r in rows:
        o = r["outcome"]
        assert o["status"] == "SUCCESS", o
        assert o["server_usage"]["prompt_tokens"] == length and o["output_len_actual"] == 1
    prompts = {f"independent_uni_tiny_{i}": length for i in range(n)}
    check_microbatches(run_dir, prompts)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["rounds"] == n
    rep = json.loads((run_dir / "report.json").read_text())["saturated"]
    assert rep["answered"] == n and rep["microbatches"] >= -(-n * length // 8192)   # early ones form before all arrive
    assert (tmp_path / "iteration.md").exists() and (tmp_path / "saturated.jsonl").exists()
