"""bench/report.py on synthetic run dirs."""

import json

import numpy as np
import pytest

import report


def write_run(d, kind, length, mbs, client, trace_ids, t0_unix_ns=1_000_000_000_000_000_000, sreq=None):
    d.mkdir(parents=True)
    trace = d / "trace.csv"
    trace.write_text("id,input_len,output_len,arrival_time\n" + "".join(f"{t},{length},1,0\n" for t in trace_ids))
    (d / "run.json").write_text(json.dumps({"name": d.name, "kind": kind, "prompt_tokens": length,
                                            "trace": str(trace)}))
    (d / "t0.json").write_text(json.dumps({"t0_unix_ns": t0_unix_ns}))
    (d / "microbatches.jsonl").write_text("".join(json.dumps(m) + "\n" for m in mbs))
    (d / "requests.jsonl").write_text("".join(json.dumps(c) + "\n" for c in client))
    if sreq:
        (d / "server_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sreq))


def outcome(rid, submit_s, ttft_ms, n):
    return {"source": {"type": "independent_request", "data": {"id": rid[12:], "input_len": n}},
            "outcome": {"request_id": rid, "status": "SUCCESS", "submit_timestamp": submit_s, "post_timestamp": submit_s,
                        "first_token_id_ms": ttft_ms, "server_usage": {"prompt_tokens": n, "cached_prompt_tokens": 0}}}


def test_saturated_window_tokens_busy_and_ttft(tmp_path):
    L, n, stage_ms, lag = 8192, 100, 10.0, 10.5
    mbs = [{"stage": s, "index": i, "start_ms": i * stage_ms + s * lag, "compute_ms": stage_ms,
            "requests": [{"id": f"independent_u_{i}", "prefix": 0, "append": L}]} for i in range(n) for s in range(4)]
    ttft = np.arange(1, n + 1) * 7.0 + 3.0                     # ms, request i
    trace = [f"u_{i}" for i in range(n)]
    client = [outcome(f"independent_u_{i}", 1000.0 + 0.001 * i, float(ttft[i]), L) for i in reversed(range(n))]
    write_run(tmp_path / "runs/saturated_8k", "saturated", L, mbs, client, trace)
    rep = report.report_run(tmp_path / "runs/saturated_8k")["saturated"]
    end = (n - 1) * stage_ms + 3 * lag + stage_ms
    lo, hi = 0.1 * end, 0.9 * end
    in_win = [i for i in range(n) if lo <= i * stage_ms < hi]
    assert rep["run_s"] == pytest.approx(end / 1e3)
    assert rep["tokens_per_s"] == pytest.approx(len(in_win) * L / ((hi - lo) / 1e3))
    assert rep["requests_per_s"] == pytest.approx(rep["tokens_per_s"] / L)
    assert rep["stage_busy"][0] == pytest.approx(1.0)          # stage 0 back to back
    assert rep["stage0_gap_ms"]["max"] == pytest.approx(0.0)
    assert rep["ttft_s"]["first"] == pytest.approx(0.010)
    for q in (50, 90, 99):
        assert rep["ttft_s"][f"p{q}"] == pytest.approx(np.percentile(ttft, q) / 1e3)
    assert rep["ttft_s"]["max"] == pytest.approx(ttft.max() / 1e3)
    assert rep["ttft_ms_by_request"] == [pytest.approx(x) for x in ttft]   # trace order, not log order
    assert rep["submit_spread_s"] == pytest.approx(0.099)
    md = report.summarize(tmp_path / "runs")
    assert md == tmp_path / "iteration.md"
    assert "| 8k | 100/100 |" in md.read_text()
    assert json.loads((tmp_path / "saturated.jsonl").read_text())["prompt_tokens"] == L


def test_one_request_stages_sends_span_host(tmp_path):
    L, t0_ms = 8192, 5_000_000.0
    stage = [57.0, 54.0, 55.0, 56.0]
    send = 0.5
    mbs, client, sreq, trace = [], [], [], []
    for k in range(3):
        rid = f"independent_one_8k_{k}"
        trace.append(f"one_8k_{k}")
        t = 10_000.0 * k + 2.0                                  # stage 0 start, ms from t0
        for s in range(4):
            mbs.append({"stage": s, "index": k, "start_ms": t, "compute_ms": stage[s],
                        "requests": [{"id": rid, "prefix": 0, "append": L}]})
            t += stage[s] + send
        end = t - send
        arrival = 10_000.0 * k
        client.append(outcome(rid, (t0_ms + arrival - 0.3) / 1e3, end - arrival + 0.3 + 1.2, L))
        sreq.append({"id": rid, "seq": k, "arrival_ms": arrival, "first_dispatch_ms": arrival + 1.0,
                     "token_ms": end + 0.4})
    write_run(tmp_path / "runs/one_8k", "one", L, mbs, client, trace, t0_unix_ns=int(t0_ms * 1e6), sreq=sreq)
    lines = report.report_run(tmp_path / "runs/one_8k")["requests"]
    assert [r["run"] for r in lines] == [1, 2, 3]
    for r in lines:
        assert [s["compute_ms"] for s in r["stages"]] == stage
        assert r["stages"][0]["start_ms"] == 0.0
        assert [s["ms"] for s in r["sends"]] == [pytest.approx(send)] * 3
        assert r["sends"][0]["bytes"] == L * 32768
        assert r["gpu_span_ms"] == pytest.approx(sum(stage) + 3 * send)
        assert r["host_ms"] == pytest.approx(r["client_ttft_ms"] - r["gpu_span_ms"])
        h = r["host_parts_ms"]
        assert h["client_to_server"] == pytest.approx(0.3, abs=1e-3)
        assert h["arrival_to_dispatch"] == pytest.approx(1.0)
        assert h["dispatch_to_stage0"] == pytest.approx(1.0)
        assert h["last_stage_to_controller"] == pytest.approx(0.4)
        assert h["controller_to_client"] == pytest.approx(0.8, abs=1e-3)
        assert sum(h.values()) == pytest.approx(r["host_ms"], abs=1e-3)
    report.summarize(tmp_path / "runs")
    md = (tmp_path / "iteration.md").read_text()
    assert "| 8k | mean of 3 | measured | 57.000 | 54.000 | 55.000 | 56.000 | 0.500 |" in md
    assert len((tmp_path / "iteration.jsonl").read_text().splitlines()) == 3


def test_misses_follow_the_pass_rule():
    it_t, sat_t = report._targets()
    t = it_t[8192]

    def req(scale_stage0=1.0, send_extra=0.0):
        return {"prompt_tokens": 8192, "run": 1, "gpu_span_ms": t["ttft_ms"], "client_ttft_ms": t["ttft_ms"],
                "stages": [{"compute_ms": s["compute_ms"] * (scale_stage0 if i == 0 else 0.9)}
                           for i, s in enumerate(t["stages"])],
                "sends": [{"ms": s["ms"] + send_extra} for s in t["sends"]]}

    st = sat_t[8192]
    sat = [{"prompt_tokens": 8192, "tokens_per_s": 0.95 * st["tokens_per_s"],
            "ttft_s": {q: st["ttft_s"][q] * (1.2 if q == "p99" else 1.0) for q in ("p50", "p90", "p99")}}]
    runs = [{"name": "saturated_8k", "tokens": {"fail": 1, "no_token": 0}}]
    ms = report.misses([req(1.05, 0.04), req(1.2, 0.06)], sat, runs)
    got = {(m["run"], m["what"], m["count"]) for m in ms}
    assert ("one_8k", "stage 0", "1 of 2 requests") in got           # +20% misses, +5% does not
    assert ("one_8k", "send 0→1", "1 of 2 requests") in got          # +0.06 ms > max(10%, 0.05 ms) = 0.05
    assert ("saturated_8k", "TTFT p99", "") in got and not any(m["what"] == "tokens/s" for m in ms)
    assert ("saturated_8k", "output", "1 wrong, 0 without a token") in got
    assert len(ms) == 6                                             # stage 0, three sends, TTFT p99, output
