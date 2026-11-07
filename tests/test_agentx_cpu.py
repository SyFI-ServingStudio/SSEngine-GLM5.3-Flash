"""benchmarks/agentx.py: layouts and the per-engine commands (no server, no GPU)."""

from pathlib import Path

import pytest

from benchmarks.agentx import Layout, client_argv, parse_layout, router_argv, server_argv, server_ready_path

RD = Path("/tmp/run")


def flag(argv, name):
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("text,lay,gpus", [
    ("pp4", Layout(pp=4), 4), ("tp4", Layout(tp=4), 4), ("dp4-ep", Layout(dp=4, ep=True), 4),
    ("tp2-pp2", Layout(tp=2, pp=2), 4), ("tp4-ep", Layout(tp=4, ep=True), 4), ("tp2-dp4-ep", Layout(2, 1, 4, True), 8),
    ("tp1", Layout(), 1)])
def test_parse(text, lay, gpus):
    assert parse_layout(text) == lay and lay.gpus == gpus and parse_layout(str(lay)) == lay


@pytest.mark.parametrize("bad", ["tp4-tp2", "ep", "pp4-ep", "tp0", "x4", "tp4-ep-ep"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_layout(bad)


def test_ssengine_pp4_only():
    argv = server_argv("ssengine", parse_layout("pp4"), "/ck", 20060, RD, "py", ["--dram-gb", "50"])
    assert argv[:5] == ["py", "-m", "ssengine", "serve", "/ck"] and argv[-2:] == ["--dram-gb", "50"]
    with pytest.raises(ValueError):
        server_argv("ssengine", parse_layout("tp4"), "/ck", 20060, RD, "py")


def test_ssengine_image():
    a = server_argv("ssengine", parse_layout("pp4"), "/ck", 20060, RD, "py", ["--dram-gb", "50"], ssengine_image="ss:arm")
    i = a.index("ss:arm")
    assert a[:2] == ["docker", "run"] and a[i + 1:i + 3] == ["serve", "/ck"] and f"{RD}:{RD}" in a
    assert flag(a, "--run-dir") == str(RD / "server") and a[-2:] == ["--dram-gb", "50"]


def test_vllm_layouts():
    a = server_argv("vllm", parse_layout("dp4-ep"), "/ck", 20060, RD, "py")
    assert (flag(a, "--tensor-parallel-size"), flag(a, "--data-parallel-size"), flag(a, "--pipeline-parallel-size")) \
        == ("1", "4", "1")
    assert "--enable-expert-parallel" in a and flag(a, "--data-parallel-rpc-port") == "20070"
    assert "--tokens-only" not in a and "--enable-prompt-tokens-details" in a
    b = server_argv("vllm", parse_layout("tp2-pp2"), "/ck", 20060, RD, "py")
    assert (flag(b, "--tensor-parallel-size"), flag(b, "--pipeline-parallel-size")) == ("2", "2")
    assert "--enable-expert-parallel" not in b and "--data-parallel-rpc-port" not in b
    d = server_argv("vllm", parse_layout("tp4"), "/ck", 20060, RD, "py", vllm_image="img:v1")
    assert d[0] == "docker" and d[d.index("--entrypoint") + 1:d.index("--entrypoint") + 4] == ["vllm", "img:v1", "serve"]
    assert "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600" in d and flag(d, "--tensor-parallel-size") == "4"
    r = router_argv("vllm", parse_layout("dp4-ep"), 20060, RD)
    assert flag(r, "--intra-node-data-parallel-size") == "4" and flag(r, "--port") == "20080"


def test_sglang_layouts():
    a = server_argv("sglang", parse_layout("dp4-ep"), "/ck", 20060, RD, "py")
    assert a[0] == "docker" and flag(a, "--tp-size") == "4" and flag(a, "--dp-size") == "4"
    assert "--enable-dp-attention" in a and flag(a, "--ep-size") == "4" and "--enable-cache-report" in a
    b = server_argv("sglang", parse_layout("tp2-pp2"), "/ck", 20060, RD, "py")
    assert (flag(b, "--tp-size"), flag(b, "--pp-size")) == ("2", "2")
    assert "--dp-size" not in b and "--ep-size" not in b
    r = router_argv("sglang", parse_layout("dp4-ep"), 20060, RD)
    assert "sglang_router.launch_router" in r and "--dp-aware" in r and flag(r, "--policy") == "cache_aware"
    assert router_argv("sglang", parse_layout("tp4"), 20060, RD) is None
    assert router_argv("ssengine", parse_layout("pp4"), 20060, RD) is None


def test_client_prefill_only():
    c = client_argv("http://h:1", "/ck", 32, 1800, RD, smoke=False)
    assert flag(c, "--extra-inputs") == "max_completion_tokens:1" and flag(c, "--concurrency") == "32"
    assert flag(c, "--public-dataset").endswith("_256k") and "--unsafe-override" not in c
    assert "--unsafe-override" in client_argv("http://h:1", "/ck", 2, 300, RD, smoke=True, entries=8)
    w = client_argv("http://h:1", "/ck", 512, 900, RD, smoke=False, warmup_per_lane=1)
    assert flag(w, "--warmup-requests-per-lane") == "1" and "--unsafe-override" in w


def test_engine_cache_mount(monkeypatch, tmp_path):
    import benchmarks.agentx as ax
    monkeypatch.setattr(ax, "CACHE_DIR", tmp_path)
    a = server_argv("sglang", parse_layout("pp4"), "/ck", 20060, RD, "py")
    mounts = [a[i + 1] for i, x in enumerate(a) if x == "-v"]
    cache = [m for m in mounts if m.endswith(":/root/.cache")]
    assert len(cache) == 1 and cache[0].startswith(str(tmp_path)) and not any(tmp_path.iterdir())   # argv only
    d = server_argv("vllm", parse_layout("tp4"), "/ck", 20060, RD, "py", vllm_image="img:v1")
    assert f"{tmp_path / 'img_v1'}:/root/.cache" in d
    assert ":/root/.cache" not in " ".join(server_argv("vllm", parse_layout("tp4"), "/ck", 20060, RD, "py"))
    assert ":/root/.cache" not in " ".join(router_argv("sglang", parse_layout("dp4-ep"), 20060, RD))


def test_ready_path():
    assert server_ready_path("sglang") == "/health"   # 503 until SGLang's own warm-up is done (job 18150)
    assert server_ready_path("vllm") == server_ready_path("ssengine") == "/v1/models"
