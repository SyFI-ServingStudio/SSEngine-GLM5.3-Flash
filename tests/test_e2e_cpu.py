"""End to end on CPU: `python -m ppimpl serve` with FakeStageModel over gloo, driven over HTTP and by req-frontend."""

import asyncio
import json
from pathlib import Path

import pytest

import run as bench_run
from conftest import free_port_block
from e2e import Server, blocks_per_request, check_microbatches, check_responses, send_all

CPU = {"device": "cpu", "fake": {"us_per_token": 2.0, "hidden_width": 64, "num_blocks": 30}}


# lanes/measure/measured/reference.jsonl, id one_1k_0: row 0 of a trace, 1,024 tokens of enwik9 (GLM tokenizer).
ONE_1K_0_SHA256 = "bf4c22dedc655d529f3c205732200a134076384f6cc1df3545d8cb2952ea3abb"


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
    lengths = [1024] + [length] * (n - 1)   # row 0: req-frontend sends it the tokens of reference prompt one_1k_0
    trace = tmp_path / "uni_tiny.csv"
    trace.write_text("id,input_len,output_len,arrival_time\n"
                     + "".join(f"uni_tiny_{i},{lengths[i]},1,0\n" for i in range(n)))
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
    prompts = {f"independent_uni_tiny_{i}": lengths[i] for i in range(n)}
    for r in rows:
        o = r["outcome"]
        assert o["status"] == "SUCCESS", o
        assert o["server_usage"]["prompt_tokens"] == prompts[o["request_id"]] and o["output_len_actual"] == 1
    # The server logs the prompt's sha256 (int32 little-endian), the key of measured/reference.jsonl.
    sreq = {r["id"]: r for r in map(json.loads, (run_dir / "server_requests.jsonl").read_text().splitlines())}
    assert sreq["independent_uni_tiny_0"]["prompt_sha256"] == ONE_1K_0_SHA256
    assert len({r["prompt_sha256"] for r in sreq.values()}) == n
    check_microbatches(run_dir, prompts)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["rounds"] == n
    rep = json.loads((run_dir / "report.json").read_text())["saturated"]
    assert rep["answered"] == n and rep["microbatches"] >= -(-sum(lengths) // 8192)   # early ones form before all arrive
    assert (tmp_path / "iteration.md").exists() and (tmp_path / "saturated.jsonl").exists()


def report_lags_ms(run_dir: Path) -> list[float]:
    """Per request: when the control thread handled the last stage's report - that stage's end (host clock, ms)."""
    s3_end = {}
    for x in map(json.loads, (run_dir / "microbatches.jsonl").read_text().splitlines()):
        if x["stage"] == 3:
            for q in x["requests"]:
                s3_end[q["id"]] = x["start_ms"] + x["compute_ms"]   # its last chunk is in the last microbatch logged
    return [r["token_ms"] - s3_end[r["id"]]
            for r in map(json.loads, (run_dir / "server_requests.jsonl").read_text().splitlines())]


def test_burst_does_not_delay_reports(tmp_path):
    """A burst of requests keeps the event loop busy; reports and dispatch run in the control thread meanwhile."""
    prompts = {f"b{i}": 100 for i in range(1500)}
    port = free_port_block()
    srv = Server(tmp_path / "run", port, **{**CPU, "fake": {**CPU["fake"], "num_blocks": 2000}})
    try:
        srv.wait_ready()
        res = asyncio.run(send_all(port, prompts))
    finally:
        assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()
    check_responses(prompts, res)
    sreq = [json.loads(x) for x in (tmp_path / "run/server_requests.jsonl").read_text().splitlines()]
    lags = report_lags_ms(tmp_path / "run")
    first = [lag for r, lag in zip(sreq, lags) if r["first_dispatch_ms"] == min(q["first_dispatch_ms"] for q in sreq)]
    lags.sort()
    print(f"arrivals over {max(r['arrival_ms'] for r in sreq):.0f} ms, first microbatch's report lag "
          f"{max(first):.2f} ms, all: p50 {lags[len(lags) // 2]:.2f} max {lags[-1]:.2f} ms")
    # one event loop (before the control thread): the first microbatch's report waited out the burst, ~115 ms here
    assert max(first) < 10 and lags[len(lags) // 2] < 5


def test_worker_failure_stops_server(tmp_path):
    import subprocess
    port = free_port_block()
    srv = Server(tmp_path / "run", port, **CPU)
    srv.wait_ready()
    workers = subprocess.run(["pgrep", "-P", str(srv.proc.pid)], capture_output=True, text=True).stdout.split()
    assert len(workers) == 4
    subprocess.run(["kill", "-9", workers[2]])
    srv.proc.wait(30)                                       # the control thread notices and stops the server
    assert "worker 2 exited" in (tmp_path / "run/server.log").read_text()
    srv.stop()
