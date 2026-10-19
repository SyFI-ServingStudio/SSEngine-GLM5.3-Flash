"""`ppimpl serve <model>` (ppimpl/serve/launch.py) and the OpenAI-compatible endpoints (ppimpl/serve/openai.py): the
model check, the config it builds, the chat template as transformers renders it, and the endpoints end to end on CPU
(FakeStageModel, a small word-level tokenizer)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from conftest import free_port_block
from e2e import ROOT, Server
from ppimpl.config import CHECKPOINT
from ppimpl.model.fake import fake_token
from ppimpl.serve import launch

# config.json of nvidia/GLM-5.3-Flash-NVFP4, the fields the check reads
GLM53_FLASH = {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
               "text_config": dict(launch.TEXT_CONFIG),
               "quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt"}}
CPU = {"model": "fake", "device": "cpu", "fake": {"us_per_token": 2.0, "hidden_width": 64, "num_blocks": 30}}
WORDS = ["<|endoftext|>", "[gMASK]", "<|user|>", "<|assistant|>", "hello", "world", "what", "is", "2+2?"]
TEMPLATE = ("[gMASK] {% for m in messages %}<|{{ m.role }}|> {{ m.content }} {% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def model_dir(d: Path, config: dict = GLM53_FLASH) -> Path:
    """A checkpoint directory without weights: config.json and a word-level tokenizer of WORDS."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}))
    (d / "chat_template.jinja").write_text(TEMPLATE)
    return d


def args(model: str, **kw) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    launch.add_arguments(ap)
    a = ap.parse_args([model])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_check_model_takes_only_glm53_flash_nvfp4():
    launch.check_model(GLM53_FLASH, "glm")
    other = {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 36}
    with pytest.raises(launch.UnsupportedModel, match="unsupported model q: architecture Qwen3ForCausalLM"):
        launch.check_model(other, "q")
    bf16 = {**GLM53_FLASH, "quantization_config": None}
    with pytest.raises(launch.UnsupportedModel, match="quantization none"):
        launch.check_model(bf16, "bf16")
    bigger = json.loads(json.dumps(GLM53_FLASH))
    bigger["text_config"]["num_hidden_layers"] = 78
    with pytest.raises(launch.UnsupportedModel, match="num_hidden_layers = 78 \\(want 45\\)"):
        launch.check_model(bigger, "big")


@pytest.mark.skipif(not Path(CHECKPOINT, "config.json").is_file(), reason="no checkpoint")
def test_the_checkpoint_passes_the_check():
    launch.check_model(json.loads(Path(CHECKPOINT, "config.json").read_text()), "ckpt")


def test_build_config_is_the_design_setting(tmp_path):
    d = model_dir(tmp_path / "m")
    cfg = launch.build_config(args(str(d)))
    assert (cfg.model, cfg.checkpoint, cfg.keep_sessions, cfg.tiers) == ("glm53", str(d), True, None)
    assert (cfg.host, cfg.port, cfg.served_model_name, cfg.max_model_len) == ("0.0.0.0", 8000, str(d), None)
    good = cfg.scheduler
    assert (good.order, good.mb_tokens, good.load_budget, good.even_split, good.force_after_ms) == \
        ("srpt", 32768, True, True, 60_000.0)
    cfg = launch.build_config(args(str(d), dram_gb=50.0, ssd_gb=128.0, ssd_dir="/ssd", port=9000,
                                   served_model_name="glm", max_model_len=65536))
    t = cfg.tiers
    assert (t.dram_gb, t.ssd_gb, t.ssd_dir, t.dram_writes, t.loading_area, t.ssd_backend) == \
        (50.0, 128.0, "/ssd", "hbm-evictions-exclusive", 0.1, "aio")
    assert (cfg.port, cfg.served_model_name, cfg.max_model_len) == (9000, "glm", 65536)
    assert launch.build_config(args(str(d), dram_gb=20.0)).tiers.ssd_gb is None
    # a YAML on top of the design setting, the options on top of both
    (tmp_path / "o.yaml").write_text(yaml.safe_dump({"pool_tokens": 1000, "scheduler": {"mb_tokens": 8192},
                                                     "port": 1234}))
    cfg = launch.build_config(args(str(d), config=str(tmp_path / "o.yaml"), port=4321))
    assert (cfg.pool_tokens, cfg.scheduler.mb_tokens, cfg.scheduler.order, cfg.port) == (1000, 8192, "srpt", 4321)
    cfg = launch.build_config(args(str(d), dram_gb=50.0, ssd_gb=128.0, ssd_dir="/ssd", ssd_backend="uring"))
    assert cfg.tiers.ssd_backend == "uring"
    with pytest.raises(SystemExit, match="--ssd-backend without an SSD tier"):
        launch.build_config(args(str(d), dram_gb=20.0, ssd_backend="uring"))
    with pytest.raises(SystemExit, match="SSD tier needs --ssd-dir"):
        launch.build_config(args(str(d), ssd_gb=128.0))
    with pytest.raises(SystemExit, match="at most 1048576"):
        launch.build_config(args(str(d), max_model_len=2_000_000))
    with pytest.raises(launch.UnsupportedModel, match="architecture LlamaForCausalLM"):
        launch.build_config(args(str(model_dir(tmp_path / "llama", {"architectures": ["LlamaForCausalLM"]}))))
    with pytest.raises(launch.UnsupportedModel, match="neither a local directory"):
        launch.build_config(args(str(tmp_path / "missing")))


def test_cli_refuses_another_model_before_starting(tmp_path):
    d = model_dir(tmp_path / "llama", {"architectures": ["LlamaForCausalLM"]})
    r = subprocess.run([sys.executable, "-m", "ppimpl", "serve", str(d)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "unsupported model" in r.stderr and "LlamaForCausalLM" in r.stderr, r.stderr


@pytest.mark.skipif(not Path(CHECKPOINT, "chat_template.jinja").is_file(), reason="no checkpoint")
def test_render_chat_equals_transformers():
    """The GLM-5.3 template rendered here (jinja2 only) and by transformers' apply_chat_template."""
    from transformers import AutoTokenizer

    from ppimpl.serve.openai import Text, render_chat

    hf = AutoTokenizer.from_pretrained(CHECKPOINT)
    text = Text(CHECKPOINT)
    tools = [{"type": "function", "function": {"name": "add", "description": "Add two numbers",
                                               "parameters": {"type": "object", "properties": {"a": {"type": "number"},
                                                                                              "b": {"type": "number"}}}}}]
    cases = [([{"role": "user", "content": "What is 2+2?"}], None, None),
             ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
               {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "And now — ünïcode?"}],
              None, {"reasoning_effort": "low"}),
             ([{"role": "user", "content": "add 1 and 2"}], tools, None)]
    for messages, tl, kw in cases:
        want = hf.apply_chat_template(messages, tools=tl, add_generation_prompt=True, tokenize=False, **(kw or {}))
        assert render_chat(text.template, messages, tl, kw) == want
        assert text.encode(want, False) == hf(want, add_special_tokens=False).input_ids


@pytest.mark.parametrize("intake_procs", [0, 2])
def test_openai_endpoints_cpu(tmp_path, intake_procs):
    d = model_dir(tmp_path / "m")
    port = free_port_block()
    srv = Server(tmp_path / "run", port, intake_procs=intake_procs, checkpoint=str(d), served_model_name="glm",
                 max_model_len=100, **CPU)
    u = f"http://127.0.0.1:{port}"
    try:
        srv.wait_ready()
        models = httpx.get(f"{u}/v1/models").json()
        assert [m["id"] for m in models["data"]] == ["glm"] and models["data"][0]["max_model_len"] == 100
        r = httpx.post(f"{u}/v1/completions", json={"model": "glm", "prompt": "hello world hello"})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["object"] == "text_completion" and j["model"] == "glm" and j["id"].startswith("cmpl-")
        assert j["usage"] == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                              "prompt_tokens_details": {"cached_tokens": 0}}
        assert j["choices"][0]["finish_reason"] in ("length", "stop")
        # token ids as the prompt; the answer is the fake model's token of (request id, length)
        r = httpx.post(f"{u}/v1/completions", json={"prompt": [4, 5, 6, 7], "max_tokens": 1, "temperature": 0.7,
                                                     "stream": True, "stream_options": {"include_usage": True}})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream"), r.text
        events = [json.loads(x[6:]) for x in r.text.splitlines() if x.startswith("data: {")]
        assert r.text.rstrip().endswith("data: [DONE]") and len(events) == 2
        assert events[1]["usage"]["prompt_tokens"] == 4 and events[0]["choices"][0]["text"] is not None
        r = httpx.post(f"{u}/v1/chat/completions", json={"messages": [{"role": "user", "content": "what is 2+2?"}]})
        assert r.status_code == 200, r.text
        j = r.json()
        # [gMASK] <|user|> what is 2+2? <|assistant|>
        assert j["object"] == "chat.completion" and j["usage"]["prompt_tokens"] == 6
        assert j["choices"][0]["message"]["role"] == "assistant"
        r = httpx.post(f"{u}/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}],
                                                          "stream": True})
        chunk = json.loads(r.text.splitlines()[0][6:])
        assert chunk["object"] == "chat.completion.chunk" and chunk["choices"][0]["delta"]["role"] == "assistant"
        # what it cannot serve
        bad = [({"prompt": "hello", "max_tokens": 16}, 400, "max_tokens must be 1"),
               ({"prompt": "hello", "model": "other"}, 404, "does not exist"),
               ({"prompt": "hello", "n": 2}, 400, "n must be 1"),
               ({"prompt": ""}, 400, "empty prompt"),
               ({"prompt": [1] * 100}, 400, "maximum context length is 100"),
               ({"prompt": [len(WORDS)]}, 400, "out of the vocabulary"),
               ({"prompt": 7}, 400, "Expected")]
        for body, status, why in bad:
            r = httpx.post(f"{u}/v1/completions", json=body)
            assert r.status_code == status and why in r.json()["error"]["message"], (body, r.text)
        # the token endpoint is unchanged
        r = httpx.post(f"{u}/inference/v1/generate", json={"token_ids": [1, 2, 3]}, headers={"x-request-id": "x"})
        assert r.json()["choices"][0]["token_ids"] == [fake_token("x", 3)]
    finally:
        assert srv.stop() == 0, (tmp_path / "run/server.log").read_text()
    logged = [json.loads(x) for x in (tmp_path / "run/server_requests.jsonl").read_text().splitlines()]
    assert len(logged) == 5


def test_cli_serves_a_model_dir_cpu(tmp_path):
    """`python -m ppimpl serve <dir> --config <yaml>`: the model's config built, the YAML on top (the fake model)."""
    d = model_dir(tmp_path / "m")
    (tmp_path / "fake.yaml").write_text(yaml.safe_dump({**CPU, "intake_procs": 1}))
    port = free_port_block()
    log = open(tmp_path / "serve.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "ppimpl", "serve", str(d), "--config", str(tmp_path / "fake.yaml"),
                          "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "glm",
                          "--run-dir", str(tmp_path / "run")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    try:
        import time
        t = time.monotonic()
        while time.monotonic() - t < 300:
            assert p.poll() is None, (tmp_path / "serve.log").read_text()
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        assert httpx.get(f"http://127.0.0.1:{port}/v1/models").json()["data"][0]["id"] == "glm"
        r = httpx.post(f"http://127.0.0.1:{port}/v1/completions", json={"model": "glm", "prompt": "hello"})
        assert r.status_code == 200 and r.json()["usage"]["prompt_tokens"] == 1, r.text
        cfg = yaml.safe_load((tmp_path / "run/server_config.yaml").read_text())
        assert cfg["keep_sessions"] and cfg["scheduler"]["order"] == "srpt" and cfg["checkpoint"] == str(d)
    finally:
        import os
        import signal
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(60)
        log.close()
