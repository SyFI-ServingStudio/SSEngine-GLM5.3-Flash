"""SemiAnalysis AgentX (InferenceX v3), prefill only, against SSEngine, vLLM or SGLang at any parallel layout.

  python -m benchmarks.agentx run --engine vllm --layout dp4-ep --concurrency 32 [--duration 1800] [--smoke]
  python -m benchmarks.agentx argv --engine sglang --layout tp4-ep --concurrency 32      print the commands only
  python -m benchmarks.agentx summarize <run dir>                                       summary.json of one run
  python -m benchmarks.agentx table <run dir>...                                        one row per run
  python -m benchmarks.agentx compare <run dir>...                                      engines side by side, per
                                                                                        concurrency and warm-up

The client is AgentX's own: `agentx profile` of agentx-harness (NVIDIA AIPerf + SemiAnalysis's scenario), at
AGENTX_VERSION in third_party/agentx-venv (env/make_env.sh step `agentx`). Its `agentx` scenario fixes the replay
rules: the Weka Claude-Code traces (dataset DATASET: the 06-21-26 corpus, requests over 256k dropped), closed-loop
sessions with their subagent SPAWN/JOIN dependencies, the recorded tool gaps (compressed to <= 10 s while nothing is in
flight), synthetic prompt tokens from each request's 64-token hash blocks (so prefix reuse is the trace's), a cache-bust
marker per replay, a random start point per lane with a full-history warm-up request, `ignore_eos`, and >= 900 s of
measurement. `--concurrency` counts live session trees, not requests.
The one change: every request asks for one output token (`--extra-inputs max_completion_tokens:1`, which AIPerf applies
over the trace's output length; the scenario does not lock it). SSEngine computes prefill and one token only
(ssengine/serve/openai.py), and vLLM and SGLang get the same requests, so the runs compare prefill: TTFT is queueing +
prefill, and the token rates are prompt tokens.

Layouts: `-`-joined tp<N>, pp<N>, dp<N> and `ep`; GPUs = tp x pp x dp. `ep` puts the MoE experts on all tp x dp ranks
of a stage. Per engine:
  ssengine  pp4 only (its design: 4 stages, ssengine/serve/launch.py's st10 setting; --dram-gb/--ssd-gb via --server-arg)
  vllm      --tensor/--pipeline/--data-parallel-size, ep -> --enable-expert-parallel; dp > 1 puts vllm-router
            (cache_aware, as benchmarks/jobs/vllm_dpep.sbatch) in front, otherwise the client talks to vLLM's own
            load-only DP balancer and loses prefix affinity. Server flags as that job's, without --tokens-only
            (AgentX sends chat requests).
  sglang    SGLANG_IMAGE in a container (podman inside a Slurm job); --tp-size tp x dp, --pp-size, dp > 1 ->
            --dp-size + --enable-dp-attention, ep -> --ep-size tp x dp. SGLang's own defaults otherwise, plus the
            shared ones: fp8 KV, the 1M context, and the cached-token report. dp > 1 puts the image's sglang-router
            (cache_aware, --dp-aware) in front, for the same reason as vLLM's: SGLang's own DP balancing
            (--load-balance-method) has no prefix-aware choice.

Run dir (default runs/agentx/<engine>_<layout>_c<N>[_smoke]_<job>): run.json (both commands, versions), server.log,
router.log, gpu_samples.csv (5 s), aiperf/ (AIPerf's exports: profile_export.jsonl has one record per request with its
TTFT, prompt tokens and cache-read tokens), summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from benchmarks.common import MAIN, ROOT

AGENTX_VERSION = "1.0.7"
AGENTX = Path(os.environ.get("SSENGINE_AGENTX") or MAIN / "third_party" / "agentx-venv" / "bin" / "agentx")
DATASET = "semianalysis_cc_traces_weka_062126_256k"
SEED = 20260707                     # AgentX's quick-start seed: the same session sample, start points and payloads
DURATION_S = 1800                   # the scenario's default; >= 900 unless --smoke
SMOKE_DURATION_S = 300
SMOKE_ENTRIES = 8
SERVED_NAME = "glm53-flash"
SGLANG_IMAGE = os.environ.get("SSENGINE_SGLANG_IMAGE") or "docker.io/lmsysorg/sglang:v0.5.23-cu130-runtime"
# vLLM from an image (an upstream release) instead of the fork in .venv; None: the fork
VLLM_IMAGE = os.environ.get("SSENGINE_VLLM_IMAGE") or None
# SSEngine from its image (docker/Dockerfile; x86_64 or aarch64 build) instead of --python; None: --python
SSENGINE_IMAGE = os.environ.get("SSENGINE_IMAGE") or None
# Extra `docker run` arguments for every container (engines, router), e.g. "--gpus all -v /data/hf:/data/hf" on a plain
# Docker host. Empty on cayenne, whose docker wrapper adds the job's GPUs and /scratch/hf by itself.
DOCKER_ARGS = shlex.split(os.environ.get("AGENTX_DOCKER_ARGS", ""))


# Host directory for each engine image's /root/.cache (FlashInfer/DeepGEMM JIT and autotune, torch.compile). Without it
# every run starts from an empty cache: SGLang pp4 re-tuned its FP4 MoE for ~40 min per run (job 18149).
CACHE_DIR = Path(os.environ.get("AGENTX_CACHE_DIR") or Path.home() / ".cache" / "agentx-engines")


def _docker(name: str, image: str | None = None) -> list[str]:
    """`docker run` head for one container; an engine `image` also gets its own persistent /root/.cache."""
    argv = ["docker", "run", "--rm", "--name", f"agentx-{name}-{os.environ.get('SLURM_JOB_ID', 'local')}", *DOCKER_ARGS]
    if image is not None:
        argv += ["-v", f"{CACHE_DIR / re.sub(r'[^A-Za-z0-9_.-]+', '_', image)}:/root/.cache"]   # run() creates it
    return argv
VLLM_KEEP_ALIVE_S = "600"   # see run(): vLLM's 5 s keep-alive close races AIPerf's pooled reuse (job 18115)
ROUTER = MAIN / "tmp" / "vllm-router-venv" / "bin" / "vllm-router"
MAX_MODEL_LEN = 1_048_576
READY_TIMEOUT_S = 2700


@dataclass(frozen=True)
class Layout:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: bool = False

    @property
    def gpus(self) -> int:
        return self.tp * self.pp * self.dp

    def __str__(self) -> str:
        parts = [f"{k}{v}" for k, v in (("tp", self.tp), ("pp", self.pp), ("dp", self.dp)) if v > 1]
        return "-".join(parts + (["ep"] if self.ep else [])) or "tp1"


def parse_layout(text: str) -> Layout:
    vals: dict = {}
    for tok in text.lower().split("-"):
        m = re.fullmatch(r"(tp|pp|dp)([1-9][0-9]*)", tok)
        if m and m[1] not in vals:
            vals[m[1]] = int(m[2])
        elif tok == "ep" and "ep" not in vals:
            vals["ep"] = True
        else:
            raise ValueError(f"layout {text!r}: {tok!r} is not tp<N>, pp<N>, dp<N> or ep (each once)")
    lay = Layout(**vals)
    if lay.ep and lay.tp * lay.dp == 1:
        raise ValueError(f"layout {text!r}: ep needs tp x dp > 1 (the experts are spread over those ranks)")
    return lay


def ports(base: int) -> dict:
    """SSEngine takes base .. base+11 (HTTP, then controller/workers/tiers); vLLM DP RPC +10, router +20."""
    return {"server": base, "dp_rpc": base + 10, "router": base + 20}


def server_argv(engine: str, lay: Layout, ckpt: str, port: int, run_dir: Path, py: str,
                extra: list[str] = (), vllm_image: str | None = None, ssengine_image: str | None = None) -> list[str]:
    p = ports(port)
    if engine == "ssengine":
        if lay != Layout(pp=4):
            raise ValueError(f"ssengine runs pp4 only (its 4-stage design), not {lay}")
        head = [py, "-m", "ssengine"] if ssengine_image is None else \
            [*_docker("ssengine", ssengine_image), "--network", "host", "--ipc", "host", "--cap-add", "SYS_NICE",
             "-v", f"{run_dir}:{run_dir}", ssengine_image]   # docker/compose.yaml's settings; --run-dir is mounted
        return [*head, "serve", ckpt, "--host", "127.0.0.1", "--port", str(p["server"]),
                "--served-model-name", SERVED_NAME, "--run-dir", str(run_dir / "server"), *extra]
    if engine == "vllm":
        head = [py, "-m", "vllm.entrypoints.cli.main"] if vllm_image is None else \
            [*_docker("vllm", vllm_image), "--network", "host", "--ipc", "host", "-e", f"VLLM_HTTP_TIMEOUT_KEEP_ALIVE={VLLM_KEEP_ALIVE_S}",
             "--entrypoint", "vllm", vllm_image]
        argv = [*head, "serve", ckpt, "--served-model-name", SERVED_NAME,
                "--host", "127.0.0.1", "--port", str(p["server"]),
                "--kv-cache-dtype", "fp8_e4m3", "--enable-prefix-caching", "--mamba-cache-mode", "align",
                "--max-model-len", str(MAX_MODEL_LEN), "--language-model-only", "--enable-prompt-tokens-details",
                # 1016, not 1024: the fork's KDA decode grid z = seqs x 64 heads must stay <= 65535 (vllm_dpep.sbatch)
                "--max-num-seqs", "1016", "--max-num-batched-tokens", "32768",
                "--tensor-parallel-size", str(lay.tp), "--pipeline-parallel-size", str(lay.pp),
                "--data-parallel-size", str(lay.dp)]
        if lay.ep:
            argv.append("--enable-expert-parallel")
        if lay.dp > 1:
            argv += ["--data-parallel-rpc-port", str(p["dp_rpc"])]
        return argv + list(extra)
    if engine == "sglang":
        world = lay.tp * lay.dp
        argv = [*_docker("sglang", SGLANG_IMAGE),
                # --ipc host shares the host's /dev/shm: Podman refuses --shm-size with it (container create fails)
                "--network", "host", "--ipc", "host", "-v", f"{run_dir}:{run_dir}",
                SGLANG_IMAGE, "python3", "-m", "sglang.launch_server", "--model-path", ckpt,
                "--served-model-name", SERVED_NAME, "--host", "127.0.0.1", "--port", str(p["server"]),
                "--tp-size", str(world), "--pp-size", str(lay.pp), "--context-length", str(MAX_MODEL_LEN),
                "--kv-cache-dtype", "fp8_e4m3", "--enable-cache-report"]
        if lay.dp > 1:
            argv += ["--dp-size", str(lay.dp), "--enable-dp-attention"]
        if lay.ep:
            argv += ["--ep-size", str(world)]
        return argv + list(extra)
    raise ValueError(f"engine {engine!r}: ssengine, vllm or sglang")


def router_argv(engine: str, lay: Layout, port: int, run_dir: Path) -> list[str] | None:
    """A cache-aware, DP-rank-aware router in front of a DP engine (prefix affinity per rank), else None."""
    if lay.dp == 1 or engine == "ssengine":
        return None
    p = ports(port)
    if engine == "sglang":
        return [*_docker("router"), "--network", "host", SGLANG_IMAGE, "python3", "-m", "sglang_router.launch_router",
                "--host", "127.0.0.1", "--port", str(p["router"]), "--worker-urls", f"http://127.0.0.1:{p['server']}",
                "--dp-aware", "--policy", "cache_aware"]
    return [str(ROUTER), "--host", "127.0.0.1", "--port", str(p["router"]),
            "--worker-urls", f"http://127.0.0.1:{p['server']}", "--intra-node-data-parallel-size", str(lay.dp),
            "--policy", "cache_aware", "--max-tree-size", "200000000",
            "--log-dir", str(run_dir / "router"), "--log-level", "info"]


def client_argv(url: str, ckpt: str, concurrency: int, duration_s: int, out: Path, smoke: bool,
                entries: int | None = None, seed: int = SEED, warmup_per_lane: int | None = None) -> list[str]:
    argv = [str(AGENTX), "profile", "--url", url, "--endpoint", "/v1/chat/completions", "--model", SERVED_NAME,
            "--tokenizer", ckpt, "--tokenizer-trust-remote-code", "--concurrency", str(concurrency), "--public-dataset", DATASET,
            "--use-server-token-count", "--extra-inputs", "max_completion_tokens:1",
            "--benchmark-duration", str(duration_s), "--random-seed", str(seed), "--ui", "simple",
            "--output-artifact-dir", str(out)]
    if entries:
        argv += ["--num-dataset-entries", str(entries)]
    if warmup_per_lane is not None:   # the scenario's 10 per lane: at c512 ~5.6k warm-up requests, hours before profiling
        argv += ["--warmup-requests-per-lane", str(warmup_per_lane)]
    if smoke or duration_s < 900 or warmup_per_lane is not None:
        argv.append("--unsafe-override")   # off the scenario's settings; stamps submission_valid: false
    return argv


def server_ready_path(engine: str) -> str:
    """The path that answers 200 once the engine takes load. SGLang serves /v1/models before its own start-up warm-up
    request has returned; /health is 503 until then (server_status Starting -> Up, v0.5.23 http_server.py). Loading
    the client earlier starves that request: at c512 it timed out after 600 s and the server exited (job 18150)."""
    return "/health" if engine == "sglang" else "/v1/models"


def _ready(url: str, path: str = "/v1/models") -> bool:
    """The engines answer /v1/models once loaded; vllm-router has no /v1/models (an HTML 404), only /health."""
    try:
        with urllib.request.urlopen(f"{url}{path}", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _start(argv: list[str], log: Path, env: dict | None = None) -> subprocess.Popen:
    return subprocess.Popen(argv, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True,
                            env=env, cwd=ROOT)


def _stop(proc: subprocess.Popen | None, name: str | None = None) -> None:
    if name:   # a container: stop it by name, its `docker run` client then exits
        subprocess.run(["docker", "stop", "-t", "30", name], capture_output=True)
    if proc is None or proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            pass


def _gpu_sampler(path: Path) -> subprocess.Popen:
    q = "timestamp,index,utilization.gpu,power.draw,memory.used,clocks.sm"
    return subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-l", "5"],
                            stdout=path.open("w"), stderr=subprocess.DEVNULL, start_new_session=True)


def _versions(engine: str, vllm_image: str | None = None, ssengine_image: str | None = None) -> dict:
    v = {"repo": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=10", "HEAD"], capture_output=True,
                                text=True).stdout.strip()}
    if AGENTX.exists():
        v["agentx_harness"] = subprocess.run([str(AGENTX.parent / "python"), "-c",
                                              "from importlib.metadata import version; print(version('agentx-harness'))"],
                                             capture_output=True, text=True).stdout.strip()
    if engine == "ssengine" and ssengine_image:
        v["ssengine_image"] = ssengine_image
    if engine == "vllm" and vllm_image:
        v["vllm_image"] = vllm_image
    elif engine == "vllm":
        v["vllm"] = subprocess.run(["git", "-C", str(MAIN / "third_party" / "vllm"), "rev-parse", "--short=10", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()
    if engine == "sglang":
        v["sglang_image"] = SGLANG_IMAGE
    return v


def default_port() -> int:
    job = int(os.environ.get("SLURM_JOB_ID", "0"))
    return 20000 + job % 120 * 100 + 60          # job-derived, below 32768, beside vllm_dpep's +50


def run(a) -> Path:
    from ssengine.config import CHECKPOINT
    lay = parse_layout(a.layout)
    ckpt = a.checkpoint or str(CHECKPOINT)
    job = os.environ.get("SLURM_JOB_ID", "local")
    name = a.run_name or f"{a.engine}_{lay}_c{a.concurrency}{'_smoke' if a.smoke else ''}_{job}"
    rd = Path(a.out).resolve() / name
    rd.mkdir(parents=True, exist_ok=False)
    port = a.port or default_port()
    p = ports(port)
    duration = a.duration or (SMOKE_DURATION_S if a.smoke else DURATION_S)
    entries = a.entries or (SMOKE_ENTRIES if a.smoke else None)
    srv = server_argv(a.engine, lay, ckpt, port, rd, a.python, a.server_arg, a.vllm_image, a.ssengine_image)
    for i, x in enumerate(srv[:-1]):
        if x == "-v" and srv[i + 1].endswith(":/root/.cache"):
            Path(srv[i + 1].rsplit(":", 1)[0]).mkdir(parents=True, exist_ok=True)
    rtr = router_argv(a.engine, lay, port, rd)
    url = f"http://127.0.0.1:{p['router'] if rtr else p['server']}"
    cli = client_argv(url, ckpt, a.concurrency, duration, rd / "aiperf", a.smoke, entries, a.seed, a.warmup_per_lane)
    meta = {"engine": a.engine, "layout": str(lay), "gpus": lay.gpus, "concurrency": a.concurrency,
            "duration_s": duration, "entries": entries, "smoke": a.smoke, "dataset": DATASET, "seed": a.seed,
            "checkpoint": ckpt, "job": job, "host": os.uname().nodename, "url": url,
            "server_argv": srv, "router_argv": rtr, "client_argv": cli, "versions": _versions(a.engine, a.vllm_image, a.ssengine_image),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    (rd / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    if a.engine == "vllm" and lay.dp > 1 and not ROUTER.exists():
        raise SystemExit(f"{ROUTER} missing: uv venv --python 3.12 {ROUTER.parents[1]} && "
                         f"uv pip install --python {ROUTER.parents[1]}/bin/python vllm-router==0.1.15")
    if not AGENTX.exists():
        raise SystemExit(f"{AGENTX} missing: STEPS=agentx env/make_env.sh")
    for q in {p["server"], p["dp_rpc"], p["router"]}:
        if subprocess.run(["ss", "-ltnH", f"sport = :{q}"], capture_output=True, text=True).stdout.strip():
            raise SystemExit(f"port {q} in use: not started")
    print("server:", shlex.join(srv), flush=True)
    container = srv[srv.index("--name") + 1] if srv[0] == "docker" else None
    rtr_container = rtr[rtr.index("--name") + 1] if rtr and rtr[0] == "docker" else None
    sampler = _gpu_sampler(rd / "gpu_samples.csv") if not a.no_gpu_samples else None
    server = router = None
    t0 = time.time()
    try:
        # vLLM closes an idle keep-alive connection after 5 s (VLLM_HTTP_TIMEOUT_KEEP_ALIVE) while AIPerf keeps pooled
        # connections up to 300 s (AIPERF_HTTP_KEEPALIVE_TIMEOUT): a request sent on one vLLM is closing gets a
        # connection reset, and one failed root warm-up request aborts an AgentX run (job 18115).
        env = dict(os.environ, VLLM_HTTP_TIMEOUT_KEEP_ALIVE=VLLM_KEEP_ALIVE_S) if a.engine == "vllm" else None
        server = _start(srv, rd / "server.log", env)
        while not _ready(f"http://127.0.0.1:{p['server']}", server_ready_path(a.engine)):
            if server.poll() is not None:
                raise SystemExit(f"server exited during start-up ({server.returncode}): {rd / 'server.log'}")
            if time.time() - t0 > READY_TIMEOUT_S:
                raise SystemExit(f"server not ready in {READY_TIMEOUT_S} s")
            time.sleep(5)
        meta["ready_s"] = round(time.time() - t0, 1)
        print(f"ready after {meta['ready_s']} s", flush=True)
        if rtr:
            router = _start(rtr, rd / "router.log")
            # sglang-router's /health is OK before it has registered the worker (it does that in the background, and
            # routes 503 until then); its /v1/models goes through a worker. vllm-router has only /health.
            while not _ready(url, "/v1/models" if a.engine == "sglang" else "/health"):
                if router.poll() is not None:
                    raise SystemExit(f"router exited ({router.returncode}): {rd / 'router.log'}")
                time.sleep(2)
        # The client downloads the dataset once (then ~/.cache/aiperf), so it runs online. AIPerf treats any
        # non-empty HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE, "0" included, as offline, and offline it cannot load a
        # tokenizer from a local directory: drop both (env.sh sets them for the server).
        env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        env.update({"AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800", "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
                    "CUDA_VISIBLE_DEVICES": ""})
        print("client:", shlex.join(cli), flush=True)
        t1 = time.time()
        rc = subprocess.run(cli, stdout=(rd / "client.log").open("w"), stderr=subprocess.STDOUT, env=env,
                            cwd=ROOT).returncode
        meta["client_rc"], meta["client_s"] = rc, round(time.time() - t1, 1)
    finally:
        _stop(router, rtr_container)
        _stop(server, container)
        if sampler:
            _stop(sampler)
        (rd / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    s = summarize(rd)
    print(json.dumps(s, indent=1))
    if meta.get("client_rc"):
        raise SystemExit(f"client exited {meta['client_rc']}: {rd / 'client.log'}")
    return rd


def _stat(d: dict, key: str, *fields: str) -> dict:
    v = d.get(key) or {}
    return {f: v.get(f) for f in fields} if isinstance(v, dict) else {}


def summarize(rd: Path) -> dict:
    """summary.json: AIPerf's aggregate plus prompt-token rates from the profiling-phase records."""
    rd = Path(rd)
    meta = json.loads((rd / "run.json").read_text())
    agg_path = rd / "aiperf" / "profile_export_aiperf.json"
    if not agg_path.exists():
        s = {"run": rd.name, "error": "no AIPerf export"}
        (rd / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        return s
    agg = json.loads(agg_path.read_text())
    recs = [json.loads(line) for line in (rd / "aiperf" / "profile_export.jsonl").read_text().splitlines() if line]
    prof = [r for r in recs if r["metadata"].get("benchmark_phase") == "profiling"]
    ok = [r for r in prof if "time_to_first_token" in r["metrics"]]

    def m(r, k):
        return (r["metrics"].get(k) or {}).get("value") or 0

    dur = (agg.get("benchmark_duration") or {}).get("avg") or 0
    prompt = sum(m(r, "input_sequence_length") for r in ok)
    cached = sum(m(r, "usage_prompt_cache_read_tokens") for r in ok)
    gpus = meta["gpus"]
    s = {"run": rd.name, "engine": meta["engine"], "layout": meta["layout"], "gpus": gpus,
         "concurrency": meta["concurrency"], "smoke": meta["smoke"], "duration_s": round(dur, 1),
         "requests": len(ok), "failed": len(prof) - len(ok),
         "context_overflow": (agg.get("context_overflow_count") or {}).get("avg"),
         "subagent_requests": sum(r["metadata"].get("source_kind") == "weka_subagent" for r in ok),
         "ttft_ms": _stat(agg, "time_to_first_token", "p50", "p90", "p99", "avg", "max"),
         "prompt_tokens": _stat(agg, "input_sequence_length", "p50", "p90", "avg", "max"),
         "requests_per_s": round(len(ok) / dur, 3) if dur else None,
         "prompt_tok_per_s_per_gpu": round(prompt / dur / gpus) if dur else None,
         "computed_tok_per_s_per_gpu": round((prompt - cached) / dur / gpus) if dur else None,
         "cache_hit": round(cached / prompt, 4) if prompt else None,
         "ready_s": meta.get("ready_s"), "client_rc": meta.get("client_rc"),
         "warmup_per_lane": _warmup_per_lane(meta["client_argv"]),
         "engine_build": _engine_build(meta)}
    (rd / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    return s


def _warmup_per_lane(cli: list[str]) -> int:
    """The client's warm-up requests per lane: --warmup-requests-per-lane, else the scenario's 10."""
    return int(cli[cli.index("--warmup-requests-per-lane") + 1]) if "--warmup-requests-per-lane" in cli else 10


def _engine_build(meta: dict) -> str:
    """What served the run: the image for container engines, else the engine's git commit."""
    v = meta.get("versions") or {}
    return v.get("ssengine_image") or v.get("vllm_image") or v.get("sglang_image") or v.get("vllm") or v.get("repo") or ""


def _load(d: Path) -> dict:
    return json.loads((Path(d) / "summary.json").read_text()) if (Path(d) / "summary.json").exists() else summarize(d)


def compare(dirs: list[Path]) -> str:
    """Runs side by side: one table per (concurrency, warm-up, smoke) group, engines as rows, best req/s first.
    Only runs measured the same way are compared; failed runs are listed under their group."""
    groups: dict[tuple, list[dict]] = {}
    for d in dirs:
        s = _load(d)
        if "error" in s:
            meta = json.loads((Path(d) / "run.json").read_text())
            s = {**s, "engine": meta["engine"], "layout": meta["layout"], "concurrency": meta["concurrency"],
                 "smoke": meta["smoke"], "warmup_per_lane": _warmup_per_lane(meta["client_argv"]),
                 "engine_build": _engine_build(meta)}
        s.setdefault("warmup_per_lane", 10)
        groups.setdefault((s["concurrency"], s["warmup_per_lane"], s["smoke"]), []).append(s)
    out = []
    for (c, w, smoke), ss in sorted(groups.items()):
        out += [f"### c{c}, warm-up {w}/lane{', smoke' if smoke else ''}", "",
                "| engine | layout | build | req/s | cache hit | TTFT p50 / p90 / p99 (s) | prompt tok/s/GPU | "
                "computed tok/s/GPU | requests | failed | run |", "|" + "---|" * 11]
        for s in sorted(ss, key=lambda s: -(s.get("requests_per_s") or -1)):
            if "error" in s:
                out.append(f"| {s['engine']} | {s['layout']} | {s['engine_build']} | {s['error']} |||||||{s['run']} |")
                continue
            t = s["ttft_ms"]
            ttft = " / ".join(f"{(t.get(k) or 0) / 1000:.2f}" for k in ("p50", "p90", "p99"))
            out.append(f"| {s['engine']} | {s['layout']} | {s.get('engine_build', '')} | {s['requests_per_s']} | "
                       f"{s['cache_hit']} | {ttft} | {s['prompt_tok_per_s_per_gpu']} | "
                       f"{s['computed_tok_per_s_per_gpu']} | {s['requests']} | {s['failed']} | {s['run']} |")
        out.append("")
    return "\n".join(out)


def table(dirs: list[Path]) -> str:
    rows = ["| run | engine | layout | GPUs | N | req/s | TTFT p50 / p90 / p99 (s) | prompt tok/s/GPU | computed tok/s/GPU "
            "| cache hit | failed |", "|" + "---|" * 11]
    for d in dirs:
        s = _load(d)
        if "error" in s:
            rows.append(f"| {s['run']} | {s['error']} |")
            continue
        t = s["ttft_ms"]
        ttft = " / ".join(f"{(t.get(k) or 0) / 1000:.2f}" for k in ("p50", "p90", "p99"))
        rows.append(f"| {s['run']} | {s['engine']} | {s['layout']} | {s['gpus']} | {s['concurrency']} | "
                    f"{s['requests_per_s']} | {ttft} | {s['prompt_tok_per_s_per_gpu']} | "
                    f"{s['computed_tok_per_s_per_gpu']} | {s['cache_hit']} | {s['failed']} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.agentx", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "argv"):
        s = sub.add_parser(name)
        s.add_argument("--engine", required=True, choices=("ssengine", "vllm", "sglang"))
        s.add_argument("--layout", required=True, help="tp<N>, pp<N>, dp<N>, ep joined by '-', e.g. dp4-ep, tp2-pp2")
        s.add_argument("--concurrency", type=int, required=True, help="live AgentX session trees")
        s.add_argument("--duration", type=int, help=f"measurement seconds (default {DURATION_S}; smoke "
                                                     f"{SMOKE_DURATION_S})")
        s.add_argument("--smoke", action="store_true", help=f"{SMOKE_ENTRIES} traces, short, --unsafe-override")
        s.add_argument("--entries", type=int, help="keep the first N eligible traces (--num-dataset-entries)")
        s.add_argument("--seed", type=int, default=SEED)
        s.add_argument("--ssengine-image", default=SSENGINE_IMAGE, help="run SSEngine from this image "
                       "(docker/Dockerfile, e.g. an aarch64 build) instead of --python")
        s.add_argument("--vllm-image", default=VLLM_IMAGE, help="run vLLM from this image (e.g. "
                       "docker.io/vllm/vllm-openai:v0.33.0) instead of the fork in .venv")
        s.add_argument("--warmup-per-lane", type=int, help="AgentX warm-up requests per lane after the primers "
                       "(scenario: 10; not a valid submission when set)")
        s.add_argument("--checkpoint", help="weights directory (default ssengine.config.CHECKPOINT)")
        s.add_argument("--port", type=int, help="server port (default job-derived)")
        s.add_argument("--server-arg", action="append", default=[], help="extra server argument (repeat)")
        s.add_argument("--python", default=sys.executable, help="interpreter of ssengine / vLLM")
        s.add_argument("--out", default=str(ROOT / "runs" / "agentx"))
        s.add_argument("--run-name")
        s.add_argument("--no-gpu-samples", action="store_true")
    sub.add_parser("summarize").add_argument("run_dir", type=Path)
    sub.add_parser("table").add_argument("run_dirs", type=Path, nargs="+")
    sub.add_parser("compare").add_argument("run_dirs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "argv":
        lay = parse_layout(a.layout)
        port = a.port or default_port()
        rd = Path(a.out) / "<run>"
        ckpt = a.checkpoint or "<checkpoint>"
        rtr = router_argv(a.engine, lay, port, rd)
        url = f"http://127.0.0.1:{ports(port)['router' if rtr else 'server']}"
        print("server:", shlex.join(server_argv(a.engine, lay, ckpt, port, rd, a.python, a.server_arg, a.vllm_image, a.ssengine_image)))
        if rtr:
            print("router:", shlex.join(rtr))
        print("client:", shlex.join(client_argv(url, ckpt, a.concurrency, a.duration or DURATION_S, rd / "aiperf",
                                                 a.smoke, a.entries, a.seed, a.warmup_per_lane)))
    elif a.cmd == "summarize":
        print(json.dumps(summarize(a.run_dir), indent=1))
    elif a.cmd == "compare":
        print(compare(a.run_dirs))
    else:
        print(table(a.run_dirs))


if __name__ == "__main__":
    main()
