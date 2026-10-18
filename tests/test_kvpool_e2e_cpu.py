"""DRAM tier end to end on CPU (lanes/m6 TIERS.md §5-§6): `python -m ppimpl serve` with `keep_sessions` and `tiers`,
FakeStageModel over gloo (its tier engines move no bytes and take the configured bandwidth's time). DRAM is HBM's
victim tier (kept contexts written back ahead of or at their eviction); a session evicted from HBM resumes from DRAM: a read, then only the new tokens are computed."""

import json

import pytest

from conftest import free_port_block
from e2e import Server
from ppimpl.config import STATE_BLOCKS
from test_prefix_e2e_cpu import CPU, Client, chunks_of, sreq_of

TIERS = {**CPU, "tiers": {"dram_gb": 0.5, "ssd_gb": None, "dram_read_gb_s": 5.0}}


@pytest.fixture(params=[0, 2], ids=["intake0", "intake2"])
def server(request, tmp_path):
    port = free_port_block()
    srv = Server(tmp_path / "run", port, intake_procs=request.param, **TIERS)
    srv.wait_ready()
    yield srv, Client(port)
    assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()


def test_hbm_evicted_session_resumes_from_dram(server):
    """As the prefix cache's LRU test: A..D fill HBM, E evicts B. B's round 1 now reads its 5,000 tokens from DRAM
    (not recomputed) and computes 100; C, evicted for it, also reads."""
    srv, c = server
    n = 5000
    assert 4 * (1 + STATE_BLOCKS) <= 29 < 5 * (1 + STATE_BLOCKS)
    r0 = {s: c.new_tokens(n) for s in "ABCDE"}
    tok = {}
    for s in "ABCD":
        tok[s], cached = c.send(f"{s}_r0", r0[s])
        assert cached == 0
    assert c.send("A_r1", r0["A"] + [tok["A"]] + c.new_tokens(99))[1] == n
    assert c.send("E_r0", r0["E"])[1] == 0                     # B leaves HBM
    assert c.send("B_r1", r0["B"] + [tok["B"]] + c.new_tokens(99))[1] == n    # from DRAM
    assert c.send("D_r1", r0["D"] + [tok["D"]] + c.new_tokens(99))[1] == n    # HBM
    assert c.send("C_r1", r0["C"] + [tok["C"]] + c.new_tokens(99))[1] == n    # from DRAM
    srv.stop()
    ch = chunks_of(srv.run_dir)
    assert ch["B_r1"] == [(n, 100)] and ch["C_r1"] == [(n, 100)] and ch["D_r1"] == [(n, 100)]
    sr = sreq_of(srv.run_dir)
    assert sr["B_r1"]["tier"] == "dram" and sr["B_r1"]["tier_tokens"] == n and sr["B_r1"]["hbm_tokens"] == 0
    assert sr["B_r1"]["read_land_ms"] >= sr["B_r1"]["read_start_ms"]
    assert "tier" not in sr["D_r1"]
    ops = [json.loads(x) for x in (srv.run_dir / "tier_ops.jsonl").read_text().splitlines()]
    kinds = [o["kind"] for o in ops]
    reads = [o for o in ops if o["kind"] == "read"]
    assert {o["req"] for o in reads} == {"B_r1", "C_r1"}                 # B and C read back
    cache = json.loads((srv.run_dir / "cache.json").read_text())
    # written back only near eviction (the LRU tail above 90% of the pool, or when evicting), not every round's
    hb = cache["write_back"]["hbm"]
    assert 0 < kinds.count("write") < 9 and hb["ahead_writes"] + hb["evict_writes"] == kinds.count("write")
    for o in reads:          # each read entry was written before the read was sent
        assert any(w["kind"] == "write" and w["entry"] == o["entry"] and w["landed_ms"] <= o["sent_ms"] for w in ops)
    assert all(o["slots"] == 1 + 3 and len(set(o["bytes"])) == 1 for o in reads)   # 1 block + 3 state slots
    tiers = json.loads((srv.run_dir / "tiers.json").read_text())
    assert tiers["state_slots"] == 3 and tiers["slots"]["dram"] == int(0.5e9 // tiers["slot_bytes"])
    assert cache["tiers"][0]["reads"] == 2 and cache["tier_ops_in_flight"] == 0


def test_dram_evictions_go_to_ssd_and_resume_from_there(tmp_path):
    """DRAM holds one context (4 slots of 7): each write demotes the previous one to SSD. B, evicted from HBM and
    DRAM, resumes from SSD, through the staging area (a victim DRAM takes no copy-up)."""
    port = free_port_block()
    cfg = {**CPU, "tiers": {"dram_gb": 0.1, "ssd_gb": 1.0, "dram_read_gb_s": 5.0, "ssd_read_gb_s": 2.0,
                            "staging_gb": 0.1}}
    srv = Server(tmp_path / "run", port, intake_procs=0, **cfg)
    try:
        srv.wait_ready()
        c = Client(port)
        n = 5000
        r0 = {s: c.new_tokens(n) for s in "ABCDE"}
        tok = {s: c.send(f"{s}_r0", r0[s])[0] for s in "ABCD"}
        assert c.send("A_r1", r0["A"] + [tok["A"]] + c.new_tokens(99))[1] == n
        assert c.send("E_r0", r0["E"])[1] == 0
        assert c.send("B_r1", r0["B"] + [tok["B"]] + c.new_tokens(99))[1] == n
    finally:
        assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()
    sr = sreq_of(srv.run_dir)
    assert sr["B_r1"]["tier"] == "ssd" and sr["B_r1"]["tier_tokens"] == n
    assert chunks_of(srv.run_dir)["B_r1"] == [(n, 100)]
    ops = [json.loads(x) for x in (srv.run_dir / "tier_ops.jsonl").read_text().splitlines()]
    kinds = [o["kind"] for o in ops]
    assert kinds.count("demote") >= 1 and kinds.count("read") == 1
    rd = next(o for o in ops if o["kind"] == "read")
    assert rd["tier"] == "ssd" and rd["host_tier"] == "staging"


def test_staged_loading_end_to_end(tmp_path):
    """Staged loading (exclusive DRAM, loading areas of 0.25): HBM's 29 blocks keep a 7-block area (one 5,000-token
    context), so 3 contexts stay kept; DRAM (8 slots) keeps a 2-slot area, the rest for contexts. A..E: A and B leave HBM and DRAM.
    A's next round is fetched from SSD into DRAM's area (larger than the area: it starts with the area empty) and
    hops to HBM's."""
    from ppimpl.kvpool.member import FAKE_BLOCK_BYTES
    slot = -(-FAKE_BLOCK_BYTES // 4 // 4096) * 4096
    port = free_port_block()
    cfg = {**CPU, "tiers": {"dram_gb": 8 * slot / 1e9, "ssd_gb": 1.0, "dram_read_gb_s": 5.0, "ssd_read_gb_s": 2.0,
                            "staging_gb": 0.1, "dram_writes": "hbm-evictions-exclusive",
                            "loading_area": 0.25}}
    srv = Server(tmp_path / "run", port, intake_procs=0, **cfg)
    try:
        srv.wait_ready()
        c = Client(port)
        n = 5000
        r0 = {s: c.new_tokens(n) for s in "ABCDE"}
        tok = {s: c.send(f"{s}_r0", r0[s])[0] for s in "ABCDE"}
        assert c.send("B_r1", r0["B"] + [tok["B"]] + c.new_tokens(99))[1] == n
        assert c.send("A_r1", r0["A"] + [tok["A"]] + c.new_tokens(99))[1] == n
    finally:
        assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()
    pool = json.loads((srv.run_dir / "pool.json").read_text())
    assert pool["loading_area_blocks"] == 7 and pool["dram_area_slots"] == 2
    sr = sreq_of(srv.run_dir)
    assert sr["A_r1"]["tier"] == "ssd" and sr["A_r1"]["fetch_land_ms"] <= sr["A_r1"]["hop_start_ms"]
    assert sr["B_r1"]["tier"] in ("dram", "ssd")    # DRAM's one context: B's or one evicted later
    ch = chunks_of(srv.run_dir)
    assert ch["A_r1"] == [(n, 100)] and ch["B_r1"] == [(n, 100)]
    ops = [json.loads(x) for x in (srv.run_dir / "tier_ops.jsonl").read_text().splitlines()]
    f = next(o for o in ops if o["kind"] == "fetch" and o["req"] == "A_r1")
    hop = next(o for o in ops if o["kind"] == "read" and o["req"] == "A_r1")
    assert f["req"] == "A_r1" and f["host_tier"] == "dram" and hop["tier"] == "dram" and hop["sent_ms"] >= f["landed_ms"]
    cache = json.loads((srv.run_dir / "cache.json").read_text())
    assert cache["loading_area"]["held"] == 0 and cache["loading_area"]["reads"] == 0
    assert cache["tiers"][0]["hops"] == sum(o["kind"] == "fetch" for o in ops) >= 1
    assert "hbm" not in cache["write_back"]           # exclusive DRAM: HBM writes at eviction, never ahead
    assert cache["tiers"][0]["stored"] == cache["evicted_sessions"] >= 2
