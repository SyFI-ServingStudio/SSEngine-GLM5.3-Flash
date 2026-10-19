"""ssengine/runtime/host_stamps.py on a stand-in model: segment starts, inclusive function times, worker output."""

import json
import time
from types import SimpleNamespace as NS

from ssengine.runtime import host_stamps


def _leaf():
    time.sleep(0.002)


def _outer():
    _leaf()
    _leaf()


def _model():
    class Graph:
        def replay(self):
            _outer()
    capture = NS(segments=[Graph().replay, lambda: _leaf(), Graph().replay])
    return NS(graphs={256: NS(capture=capture)})


def _run(model):
    for f in model.graphs[256].capture.segments:
        f()


def test_stamps_time_functions_and_segments():
    names = [_outer, _leaf, "no.such:thing"]
    model = _model()
    st = host_stamps.HostStamps(model, names)
    assert st.missing == ["no.such:thing: ModuleNotFoundError"]
    _run(model)                                    # inactive: nothing recorded
    st.begin()
    _run(model)
    rec = st.end()
    assert [(i, k) for i, k, _ in rec["seg"]] == [(0, "g"), (1, "e"), (2, "g")]
    assert rec["seg"][0][2] < rec["seg"][1][2] < rec["seg"][2][2]
    assert rec["fn"]["_outer"][1] == 2 and rec["fn"]["_leaf"][1] == 5
    assert rec["fn"]["_outer"][0] >= 8 and rec["host_ms"] >= rec["fn"]["_outer"][0]
    st.begin()
    rec2 = st.end()
    assert rec2["fn"] == {} and rec2["seg"] == []


def test_worker_stamps_writes_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("SSENGINE_HOST_STAMPS", "1:2@0")
    monkeypatch.setattr(host_stamps, "FUNCTIONS", [])
    model = _model()
    ws = host_stamps.WorkerStamps(0, str(tmp_path), model)
    off = host_stamps.WorkerStamps(1, str(tmp_path), model)
    assert ws.on and not off.on
    for mb in range(3):
        plan = NS(mb_id=mb, chunks=[NS(start=2048, num_tokens=256)])
        ws.before(plan)
        _run(model)
        ws.after()
    rows = [json.loads(x) for x in (tmp_path / "host_stamps_stage0.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rows] == [1] and rows[0]["chunks"] == [[2048, 256]] and len(rows[0]["seg"]) == 3


def test_worker_stamps_slow_mode_keeps_only_slow_forwards(tmp_path, monkeypatch):
    monkeypatch.setenv("SSENGINE_HOST_STAMPS", "slow:20@0")
    model = _model()
    ws = host_stamps.WorkerStamps(0, str(tmp_path), model)
    assert ws.on and ws.slow_ms == 20
    for mb, pause in enumerate([0.0, 0.03, 0.0]):
        ws.before(NS(mb_id=mb, chunks=[NS(start=0, num_tokens=128)]))
        _run(model)
        time.sleep(pause)
        ws.after()
    rows = [json.loads(x) for x in (tmp_path / "host_stamps_stage0.jsonl").read_text().splitlines()]
    assert [r["index"] for r in rows] == [1]
    assert rows[0]["host_ms"] > 20 and rows[0]["gc_long"] == [] and rows[0]["device_allocs"] == 0
