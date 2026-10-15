"""Run handoff workloads against `python -m ppimpl serve` with req-frontend: one entry per run name.

  python bench/run.py one_8k saturated_8k [--fake] --time 00:20:00 [--partition priority]   submit one Slurm job
  python bench/run.py one_8k --fake --inline                                                 run here (no Slurm)

Names: one_<L>, saturated_<L> (4_iteration.md), L in 1k..128k. Each run gets a fresh server; the server and the
client run as concurrent steps of one job (port 20000 + job id % 10000). Everything goes to
measured/runs/<name>/ (with --fake: measured/runs/fake/<name>/): run.json, uptime.txt, server and client configs,
server logs (microbatches.jsonl, server_requests.jsonl, worker*.log), requests.jsonl, metrics.json
(handoff data/metrics.py), report.json (bench/report.py).
"""

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
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WS = Path("/raid/kanzhu/ServingStudio")
HANDOFF = Path(os.environ.get("PPIMPL_HANDOFF", WS / "wt-pp-main/doc/handoff/glm53_flash_pp4_b200_v1"))
REQ_FRONTEND = Path(os.environ.get("PPIMPL_REQ_FRONTEND", WS / "wt-pp-main/alignment/load_generator/req-frontend"))
CLIENT_BIN = Path(os.environ.get("PPIMPL_REQ_FRONTEND_BIN", ROOT / "build/req-frontend/release/session_runner"))
CHECKPOINT = Path("/raid/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348")
ENWIK9 = WS / "data/corpus/enwik9"
PYTHON = sys.executable


def parse_name(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(one|saturated)_(\d+)k", name)
    if not m:
        raise SystemExit(f"unknown run name {name!r}: one_<L> or saturated_<L>, L in 1k..128k")
    return m.group(1), int(m.group(2)) * 1024


def trace_for(kind: str, length: int) -> Path:
    return HANDOFF / "data/bringup" / f"{'one' if kind == 'one' else 'uni'}_{length // 1024}k.csv"


def server_config(kind: str, length: int, use_fake: bool, run_dir: Path, device: str = "cuda", **extra) -> dict:
    cfg = {"model": "fake" if use_fake else "glm53", "device": device, "run_dir": str(run_dir),
           # one request is one microbatch: the budget holds the whole prompt (4_iteration.md §2)
           "mb_tokens": max(8192, length) if kind == "one" else 8192}
    if not use_fake:
        cfg["checkpoint"] = str(CHECKPOINT)
    cfg.update(extra)
    return cfg


def client_config(trace: Path, port: int, run_dir: Path, *, text_file: Path = ENWIK9,
                  tokenizer: Path = CHECKPOINT / "tokenizer.json", token_pool_limit: int | None = None) -> dict:
    cfg = {
        "input": {"trace": str(trace), "format": "text-generation-independent"},
        "corpus": {"text_file": str(text_file), "tokenizer": str(tokenizer)},
        "server": {"backend": "vllm-tokens", "base_url": f"http://127.0.0.1:{port}", "model": "glm53-flash-pp4",
                   "temperature": 0},
        # no max_concurrency: the saturated run must not be capped (4_iteration.md §5)
        "replay": {"arrival_mode": "trace-timed", "context": {"max_model_len": 1048576}},
        "output": {"directory": str(run_dir)},
    }
    if token_pool_limit:
        cfg["corpus"]["token_pool_limit"] = token_pool_limit
    return cfg


def client_argv(client_yaml: Path) -> list[str]:
    """req-frontend's own lowering of the YAML (launcher.config), run with our build of its binary."""
    sys.path.insert(0, str(REQ_FRONTEND))
    try:
        from launcher.config import load_task_config
    finally:
        sys.path.pop(0)
    spec = load_task_config("run", client_yaml)
    return [str(CLIENT_BIN), *spec.arguments]


def ensure_client_bin():
    if CLIENT_BIN.exists():
        return
    target = CLIENT_BIN.parents[1]
    subprocess.run(["cargo", "build", "--release", "--features", "runtime", "--bin", "session_runner"],
                   cwd=REQ_FRONTEND, env={**os.environ, "CARGO_TARGET_DIR": str(target)}, check=True)


def git_rev(path: Path) -> str:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "."], capture_output=True,
                           text=True).stdout.strip()
    return r.stdout.strip() + ("+dirty" if dirty else "")


def wait_health(port: int, proc: subprocess.Popen, timeout_s: float) -> float:
    t = time.monotonic()
    while time.monotonic() - t < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before /health")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return time.monotonic() - t
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"/health not ready after {timeout_s} s")


def stop(proc: subprocess.Popen, timeout_s: float = 90):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def step(argv: list[str], gpus: int | None) -> list[str]:
    """Inside a Slurm job: one concurrent step (`srun --overlap`); else the command itself."""
    if not os.environ.get("SLURM_JOB_ID"):
        return argv
    return ["srun", "--overlap", "-n1", f"--gres={'gpu:%d' % gpus if gpus else 'none'}", *argv]


def execute(run_dir: Path, server_cfg: dict, client_cfg: dict, *, port: int, meta: dict,
            client_timeout_s: float = 1200, health_timeout_s: float = 1800, gpus: int | None = 4) -> dict:
    """One run: fresh server, req-frontend, metrics and report. Returns run.json's content."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("microbatches.jsonl", "server_requests.jsonl", "requests.jsonl"):
        (run_dir / f).unlink(missing_ok=True)
    server_cfg = {**server_cfg, "port": port, "run_dir": str(run_dir)}
    (run_dir / "server.yaml").write_text(yaml.safe_dump(server_cfg, sort_keys=False))
    (run_dir / "client.yaml").write_text(yaml.safe_dump(client_cfg, sort_keys=False))
    argv = client_argv(run_dir / "client.yaml")
    (run_dir / "client_command.txt").write_text(shlex.join(argv) + "\n")
    meta = {**meta, "port": port, "job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "ppimpl_commit": git_rev(ROOT), "req_frontend_commit": git_rev(REQ_FRONTEND)}
    uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    (run_dir / "uptime.txt").write_text(uptime + "\n")
    meta["uptime"] = uptime
    print(f"[{run_dir.name}] uptime: {uptime}", flush=True)

    with open(run_dir / "server.log", "w") as slog:
        server = subprocess.Popen(step([PYTHON, "-m", "ppimpl", "serve", "--config", str(run_dir / "server.yaml")],
                                       gpus), cwd=ROOT, stdout=slog, stderr=subprocess.STDOUT)
    try:
        meta["server_ready_s"] = round(wait_health(port, server, health_timeout_s), 1)
        print(f"[{run_dir.name}] server ready after {meta['server_ready_s']} s", flush=True)
        t = time.time()
        with open(run_dir / "client.log", "w") as clog:
            client = subprocess.run(step(argv, None), cwd=REQ_FRONTEND, stdout=clog, stderr=subprocess.STDOUT,
                                    timeout=client_timeout_s)
        meta["client_s"] = round(time.time() - t, 1)
        meta["client_returncode"] = client.returncode
    finally:
        stop(server)
        # uvicorn re-raises the SIGTERM it handled once it has shut down: -15 is a clean stop
        meta["server_returncode"] = 0 if server.returncode == -signal.SIGTERM else server.returncode
    pool = run_dir / "pool.json"
    if pool.exists():
        meta["pool"] = json.loads(pool.read_text())
    reqs = run_dir / "requests.jsonl"
    if reqs.exists() and reqs.stat().st_size:
        m = subprocess.run([PYTHON, str(HANDOFF / "data/metrics.py"), str(reqs)], capture_output=True, text=True)
        (run_dir / "metrics.json").write_text(m.stdout if m.returncode == 0 else m.stderr)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    rep = subprocess.run([PYTHON, str(ROOT / "bench/report.py"), str(run_dir)], cwd=ROOT, capture_output=True,
                         text=True)
    print(rep.stdout + rep.stderr, flush=True)
    return meta


def run_named(name: str, fake: bool, port: int, out_root: Path, args) -> dict:
    kind, length = parse_name(name)
    run_dir = out_root / name
    trace = trace_for(kind, length)
    extra = {}
    if fake and args.us_per_token is not None:
        extra["fake"] = {"us_per_token": args.us_per_token}
    if args.cpu:    # harness check without GPUs: gloo, small activations
        extra["fake"] = {**extra.get("fake", {}), "hidden_width": 64}
    return execute(run_dir, server_config(kind, length, fake, run_dir, device="cpu" if args.cpu else "cuda", **extra),
                   client_config(trace, port, run_dir), port=port, gpus=None if args.cpu else 4, meta={"name": name, "kind": kind, "prompt_tokens": length, "fake": fake,
                                    "trace": str(trace)},
                   client_timeout_s=args.client_timeout)


def submit(args) -> None:
    out = ROOT / "measured/runs" / ("fake" if args.fake else "")
    logs = out / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    inner = [PYTHON, str(Path(__file__).resolve()), *args.names, "--inline"]
    if args.fake:
        inner.append("--fake")
    if args.us_per_token is not None:
        inner += ["--us-per-token", str(args.us_per_token)]
    script = f"""#!/bin/bash
#SBATCH -J ppimpl-bench
#SBATCH --gres=gpu:4
#SBATCH --time={args.time}
#SBATCH -p {args.partition}
#SBATCH -o {logs}/%j.out
set -e
source {WS}/.env
cd {ROOT}
{shlex.join(inner)}
"""
    path = logs / f"job_{'_'.join(args.names)}.sbatch"[:200]
    path.write_text(script)
    subprocess.run(["sbatch", str(path)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--fake", action="store_true", help="FakeStageModel instead of the real stages")
    ap.add_argument("--inline", action="store_true", help="run here (inside a job, or without Slurm)")
    ap.add_argument("--time", help="Slurm --time (required to submit)")
    ap.add_argument("--partition", default="main")
    ap.add_argument("--port", type=int)
    ap.add_argument("--us-per-token", type=float, help="FakeStageModel stage time per token")
    ap.add_argument("--client-timeout", type=float, default=1200)
    ap.add_argument("--cpu", action="store_true", help="with --fake --inline: stages on CPU (gloo), to check the harness")
    args = ap.parse_args()
    for n in args.names:
        parse_name(n)
    ensure_client_bin()
    if not args.inline:
        if not args.time:
            raise SystemExit("--time is required to submit (run slurm-windows first)")
        return submit(args)
    job = os.environ.get("SLURM_JOB_ID")
    port = args.port or (20000 + int(job) % 10000 if job else 18500)
    if args.cpu and not (args.fake and args.inline):
        raise SystemExit("--cpu needs --fake --inline")
    out = ROOT / "measured/runs" / ("fake_cpu" if args.cpu else "fake" if args.fake else "")
    failed = []
    for i, name in enumerate(args.names):
        meta = run_named(name, args.fake, port + 10 * i, out, args)   # fresh ports per run
        if meta.get("client_returncode") != 0:
            failed.append(name)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
