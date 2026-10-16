"""bench/nsys_trace.py on a synthetic Nsight SQLite export (the tables and columns `nsys export --type sqlite` writes)."""

import json
import sqlite3
import sys
import types

import pytest

from bench import nsys_trace as tr
from bench import report
from test_report import outcome, write_run

MS = 1_000_000


def A(x):
    return pytest.approx(x, abs=1e-5)   # kernel times are whole ns
STAGE_T, LAG = 10.0, 10.5


def pid(s):
    return (1000 + s) << 24


def build_db(path, nmb=3):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INT, end INT, eventType INT, text TEXT, globalTid INT, textId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT, globalTid INT, correlationId INT, nameId INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, streamId INT, globalPid INT,
                                                 correlationId INT, shortName INT, demangledName INT);
    """)
    names = {}

    def sid(n):
        if n not in names:
            names[n] = len(names) + 1
            con.execute("INSERT INTO StringIds VALUES (?, ?)", (names[n], n))
        return names[n]

    cid = [0]

    def launch(s, host_ms, kernels):
        cid[0] += 1
        tid = pid(s) + 1000 + s
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                    (int(host_ms * MS), int(host_ms * MS) + 1000, tid, cid[0], sid("cudaLaunchKernel")))
        for a, b, name in kernels:
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?)",
                        (int(a * MS), int(b * MS), s, 7, pid(s), cid[0], sid(name), sid("void " + name + "<int>()")))

    for k in range(nmb):
        for s in range(4):
            T = k * STAGE_T + s * LAG + 100.0       # GPU time of the stage's first compute kernel
            H = T - 50.0                            # host time the plan is processed (well ahead of the GPU)
            text = f"stage {s} mb {k}"
            # half the ranges use interned text (textId), like newer exports
            con.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,?)",
                        (int(H * MS), int((H + 2) * MS), 59, None if k % 2 else text, pid(s) + 1000 + s,
                         sid(text) if k % 2 else None))
            if s > 0:
                launch(s, H + 0.1, [(T - 0.4, T - 0.05, "ncclDevKernel_SendRecv")])
            launch(s, H + 0.2, [(T, T + 3, "gemm_a")])
            launch(s, H + 0.3, [(T + 3.5, T + 6, "rmsnorm")])
            launch(s, H + 0.4, [(T + 6, T + 8, "gemm_a"), (T + 8.2, T + 10, "moe")])   # a graph launch: one id
            if s < 3:
                launch(s, H + 1.0, [(T + 10.05, T + 10.4, "ncclDevKernel_SendRecv")])
    launch(0, 1.0, [(1.0, 2.0, "outside_any_range")])                              # not in an mb range
    con.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?,?)",                     # stage 0 only: partial mb
                (int(500 * MS), int(502 * MS), 59, f"stage 0 mb {nmb}", pid(0) + 1000, None))
    launch(0, 500.5, [(600.0, 610.0, "gemm_a")])
    con.commit()
    con.close()


def test_trace_per_microbatch_numbers(tmp_path):
    db = tmp_path / "trace.sqlite"
    build_db(db)
    res = tr.analyze(db)
    assert res["stages"] == 4 and res["microbatches"] == 3 and res["partial_microbatches"] == 1
    assert res["unattributed_kernels"] == 1
    for k in range(3):
        m = res["mbs"][str(k)]
        for s, st in enumerate(m["stages"]):
            assert st["start_ms"] == A(k * STAGE_T + s * LAG)
            assert st["compute_ms"] == A(10.0)
            assert st["busy_ms"] == A(9.3)
            assert st["gaps_ms"] == A(0.7)
            assert st["kernels"] == 4
            assert st["nccl_recv_ms"] == (None if s == 0 else A(0.35))
            assert st["nccl_send_ms"] == (None if s == 3 else A(0.35))
        assert [x["ms"] for x in m["sends"]] == [A(LAG - STAGE_T)] * 3
        assert m["gpu_span_ms"] == A(3 * LAG + STAGE_T)
    assert res["stage0_gap_ms"] == [A(0.0)] * 2
    k0 = {k["name"]: k for k in res["kernels"] if k["stage"] == 0}
    assert k0["gemm_a"]["count"] == 6 and k0["gemm_a"]["per_mb_ms"] == A(5.0)
    assert k0["moe"]["mean_ms"] == A(1.8)
    assert "outside_any_range" not in k0


def test_manifest_classes_via_kernels_check_trace(tmp_path, monkeypatch):
    class TC:
        def __init__(self, name):
            self.matched = {14: [name]} if "gemm" in name else {}
            self.glue = [name] if "norm" in name else []
            self.forbidden = []

    fake = types.SimpleNamespace(
        check_trace=lambda names, stage: TC(names[0]),
        by_config=lambda c: types.SimpleNamespace(config=c, layer_types=("dsa_moe",), positions=("dsa.q_b_proj",)))
    monkeypatch.setitem(sys.modules, "ppimpl.kernels", fake)
    import ppimpl
    monkeypatch.setattr(ppimpl, "kernels", fake, raising=False)
    db = tmp_path / "trace.sqlite"
    build_db(db, nmb=1)
    k = {(x["stage"], x["name"]): x for x in tr.analyze(db)["kernels"]}
    assert k[(1, "gemm_a")]["class"] == "manifest" and k[(1, "gemm_a")]["positions"] == ["dsa_moe.dsa.q_b_proj"]
    assert k[(1, "rmsnorm")]["class"] == "glue"
    assert k[(1, "moe")]["class"] == "unknown"


def test_report_uses_trace_with_events_beside(tmp_path):
    L, n = 8192, 3
    run = tmp_path / "runs/one_8k"
    mbs, client, ids = [], [], []
    for k in range(n):
        rid = f"independent_one_8k_{k}"
        ids.append(f"one_8k_{k}")
        for s in range(4):   # CUDA-event numbers: slightly different from the trace
            mbs.append({"stage": s, "index": k, "start_ms": 1000.0 * k + s * 10.6, "compute_ms": 10.05,
                        "requests": [{"id": rid, "prefix": 0, "append": L}]})
        client.append(outcome(rid, 1000.0 + k, 50.0, L))
    write_run(run, "one", L, mbs, client, ids)
    build_db(run / "trace.sqlite", nmb=n)
    (run / "trace.json").write_text(json.dumps(tr.analyze(run / "trace.sqlite")))
    rep = report.report_run(run)
    for line in rep["requests"]:
        assert line["source"] == "nsys"
        assert [s["compute_ms"] for s in line["stages"]] == [A(10.0)] * 4
        assert [s["event_compute_ms"] for s in line["stages"]] == [A(10.05)] * 4
        assert [s["gaps_ms"] for s in line["stages"]] == [A(0.7)] * 4
        assert [s["start_ms"] for s in line["stages"]] == [A(s * LAG) for s in range(4)]
        assert [x["ms"] for x in line["sends"]] == [A(0.5)] * 3
        assert [x["event_ms"] for x in line["sends"]] == [A(0.55)] * 3
        assert line["sends"][0]["nccl_send_ms"] == A(0.35)
        assert line["sends"][2]["nccl_recv_ms"] == A(0.35)
        assert line["gpu_span_ms"] == A(41.5) and line["event_gpu_span_ms"] == A(41.85)
        assert line["host_ms"] == A(50.0 - 41.5)
    assert {k["name"] for k in rep["kernels"]} >= {"gemm_a", "rmsnorm", "moe", "ncclDevKernel_SendRecv"}
    md = report.summarize(tmp_path / "runs").read_text()
    assert "Kernels (nsys)" in md and "| 8k | 0.700 / 0.700 / 0.700 / 0.700 |" in md


def test_saturated_trace_summary(tmp_path):
    L, n = 8192, 3
    run = tmp_path / "runs/saturated_8k"
    mbs = [{"stage": s, "index": k, "start_ms": k * STAGE_T + s * LAG, "compute_ms": 10.1,
            "requests": [{"id": f"independent_u_{k}", "prefix": 0, "append": L}]} for k in range(n) for s in range(4)]
    client = [outcome(f"independent_u_{k}", 1000.0, 40.0 + k, L) for k in range(n)]
    write_run(run, "saturated", L, mbs, client, [f"u_{k}" for k in range(n)])
    build_db(run / "trace.sqlite", nmb=n)
    (run / "trace.json").write_text(json.dumps(tr.analyze(run / "trace.sqlite")))
    t = report.report_run(run)["saturated"]["trace"]
    assert t["microbatches"] == 3 and t["mb_range"] == [0, 2]
    assert t["stage_compute_ms"] == [A(10.0)] * 4
    assert t["event_stage_compute_ms"] == [A(10.1)] * 4
    assert t["stage0_gap_ms"]["max"] == A(0.0)
    assert "Stage-0 gap mean / p99 / max" in report.summarize(tmp_path / "runs").read_text()
