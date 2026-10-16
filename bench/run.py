"""Run handoff workloads against `python -m ppimpl serve` with req-frontend: one entry per run name.

  python -m bench.run one_8k saturated_8k [--fake] --time 00:20:00 [--partition main]   submit one Slurm job
  python -m bench.run one_8k --fake --inline                                                 run here (no Slurm)
  ... --nsys [--nsys-mbs a:b|all]     the server under Nsight Systems -> trace.nsys-rep, trace.sqlite, trace.json

Names: one_<L>, saturated_<L> (4_iteration.md), L in 1k..128k. Each run gets a fresh server; the server and the
client run as concurrent steps of one job (port 20000 + job id % 120 * 100). Everything goes to
measured/runs/<name>/ (with --fake: measured/runs/fake/<name>/): run.json, uptime.txt, server and client configs,
server logs (microbatches.jsonl, server_requests.jsonl, worker*.log), requests.jsonl, metrics.json
(handoff data/metrics.py), report.json (bench/report.py).
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import yaml

from bench.common import ENWIK9, HANDOFF, ROOT
from ppimpl.runtime.cpus import cpu_list

WS = Path("/raid/kanzhu/ServingStudio")
REQ_FRONTEND = Path(os.environ.get("PPIMPL_REQ_FRONTEND", WS / "wt-pp-main/alignment/load_generator/req-frontend"))
CLIENT_BIN = Path(os.environ.get("PPIMPL_REQ_FRONTEND_BIN", ROOT / "build/req-frontend/release/session_runner"))
CHECKPOINT = Path("/raid/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348")
PYTHON = sys.executable
NSYS = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"


def parse_name(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(one|saturated)_(\d+)k", name)
    if not m:
        raise SystemExit(f"unknown run name {name!r}: one_<L> or saturated_<L>, L in 1k..128k")
    return m.group(1), int(m.group(2)) * 1024


# The queued plan forms this long before stage 0 frees: the real stage 0 spends 25-30 ms of host time launching an
# eager forward at <= 8k (model lane, M2), plus margin. Recorded in each run's server_config.yaml.
REAL_FORM_LEAD_MS = 35.0

# One request of 64k or more is one microbatch, and its activations outgrow the contract's 13.6 GB reserve beside the
# full KV pool (job 7117: one_64k needs ~37 GB, one_128k ran out of memory in warm-up). These runs need a few blocks of
# KV only, so they cap the pool (a setup difference in measured/issues.md); stage times do not depend on the pool size.
ONE_LONG_FROM = 65536
ONE_LONG_POOL_TOKENS = 2 * 1024 * 1024


def trace_for(kind: str, length: int) -> Path:
    return HANDOFF / "data/bringup" / f"{'one' if kind == 'one' else 'uni'}_{length // 1024}k.csv"


def server_config(kind: str, length: int, use_fake: bool, run_dir: Path, device: str = "cuda", **extra) -> dict:
    cfg = {"model": "fake" if use_fake else "glm53", "device": device, "run_dir": str(run_dir),
           # one request is one microbatch: the budget holds the whole prompt (4_iteration.md §2)
           "scheduler": {"mb_tokens": max(8192, length) if kind == "one" else 8192}}
    if not use_fake:
        cfg["checkpoint"] = str(CHECKPOINT)
        cfg["form_lead_ms"] = REAL_FORM_LEAD_MS
        if kind == "one" and length >= ONE_LONG_FROM:
            cfg["pool_tokens"] = ONE_LONG_POOL_TOKENS
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


def stop(proc: subprocess.Popen, port: int, timeout_s: float = 90):
    """Ask the server to stop (POST /shutdown) and wait; under nsys the report is written before it exits."""
    if proc.poll() is not None:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/shutdown", method="POST"), timeout=5)
    except OSError:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout_s)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def nsys_prefix(run_dir: Path, window: bool) -> list[str]:
    """Nsight Systems around the server; it follows the worker processes it spawns. With `window`, collection runs
    only between the workers' cudaProfilerStart/Stop (server config `profile_mbs`)."""
    argv = [NSYS, "profile", "-o", str(run_dir / "trace"), "--force-overwrite", "true", "-t", "cuda,nvtx",
            "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none"]
    if window:
        argv += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
    return argv


def nsys_window(kind: str, length: int, requests: int, spec: str | None) -> list[int] | None:
    """Microbatches to trace: `a:b`, `all`, or by default the whole one-request run and the middle 80% of a
    saturated run's expected microbatches (requests x L / 8192)."""
    if spec == "all" or (spec is None and kind == "one"):
        return None
    if spec:
        a, b = spec.split(":")
        return [int(a), int(b)]
    n = -(-requests * length // 8192)
    return [round(0.1 * n), round(0.9 * n)]


DIRECT = False   # --gpus: this process picked the GPUs (CUDA_VISIBLE_DEVICES) and its CPUs; no srun steps


def step(argv: list[str], gpus: int | None) -> list[str]:
    """Inside a Slurm job: one concurrent step (`srun --overlap`); else the command itself."""
    if not os.environ.get("SLURM_JOB_ID") or DIRECT:
        return argv
    return ["srun", "--overlap", "-n1", f"--gres={'gpu:%d' % gpus if gpus else 'none'}", *argv]


def execute(run_dir: Path, server_cfg: dict, client_cfg: dict, *, port: int, meta: dict,
            client_timeout_s: float = 1200, health_timeout_s: float = 1800, gpus: int | None = 4,
            nsys: bool = False) -> dict:
    """One run: fresh server, req-frontend, metrics and report. Returns run.json's content."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("run.json", "report.json", "pool.json", "metrics.json", "summary.json",   # a failed run leaves none
              "microbatches.jsonl", "server_requests.jsonl", "requests.jsonl", "trace.nsys-rep", "trace.sqlite",
              "trace.json", "token_check.json", "cpus.json", *(f"stage{s}.json" for s in range(8))):
        (run_dir / f).unlink(missing_ok=True)
    server_cfg = {**server_cfg, "port": port, "run_dir": str(run_dir)}
    (run_dir / "server.yaml").write_text(yaml.safe_dump(server_cfg, sort_keys=False))
    (run_dir / "client.yaml").write_text(yaml.safe_dump(client_cfg, sort_keys=False))
    argv = client_argv(run_dir / "client.yaml")
    (run_dir / "client_command.txt").write_text(shlex.join(argv) + "\n")
    meta = {**meta, "port": port, "job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "half": os.environ.get("PPIMPL_HALF"),
            "run_cpus": cpu_list(os.sched_getaffinity(0)),
            "ppimpl_commit": git_rev(ROOT), "req_frontend_commit": git_rev(REQ_FRONTEND)}
    uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    (run_dir / "uptime.txt").write_text(uptime + "\n")
    meta["uptime"] = uptime
    print(f"[{run_dir.name}] uptime: {uptime}", flush=True)

    serve = [PYTHON, "-m", "ppimpl", "serve", "--config", str(run_dir / "server.yaml")]
    if nsys:
        serve = nsys_prefix(run_dir, server_cfg.get("profile_mbs") is not None) + serve
        meta["nsys"] = {"command": shlex.join(serve), "profile_mbs": server_cfg.get("profile_mbs")}
    with open(run_dir / "server.log", "w") as slog:
        server = subprocess.Popen(step(serve, gpus), cwd=ROOT, stdout=slog, stderr=subprocess.STDOUT)
    try:
        meta["server_ready_s"] = round(wait_health(port, server, health_timeout_s), 1)
        print(f"[{run_dir.name}] server ready after {meta['server_ready_s']} s", flush=True)
        t = time.time()
        with open(run_dir / "client.log", "w") as clog:
            client = subprocess.run(step(argv, None), cwd=REQ_FRONTEND, stdout=clog, stderr=subprocess.STDOUT,
                                    timeout=client_timeout_s)
        meta["client_s"] = round(time.time() - t, 1)
        meta["client_returncode"] = client.returncode
        meta["uptime_after"] = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
        print(f"[{run_dir.name}] uptime after: {meta['uptime_after']}", flush=True)
    finally:
        stop(server, port, timeout_s=1800 if nsys else 90)
        # uvicorn re-raises the SIGTERM it handled once it has shut down: -15 (143 through nsys) is a clean stop
        meta["server_returncode"] = 0 if server.returncode in (-signal.SIGTERM, 128 + signal.SIGTERM) \
            else server.returncode
    if nsys and (run_dir / "trace.nsys-rep").exists():
        t = time.time()
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true", "-o",
                        str(run_dir / "trace.sqlite"), str(run_dir / "trace.nsys-rep")],
                       stdout=subprocess.DEVNULL, stderr=open(run_dir / "nsys_export.log", "w"), check=False)
        tr = subprocess.run([PYTHON, "-m", "bench.nsys_trace", str(run_dir)], cwd=ROOT, capture_output=True,
                            text=True)
        (run_dir / "trace.log").write_text(tr.stdout + tr.stderr)
        meta["nsys"]["export_and_parse_s"] = round(time.time() - t, 1)
    if (run_dir / "cpus.json").exists():
        meta["cpus"] = json.loads((run_dir / "cpus.json").read_text())
    pool = run_dir / "pool.json"
    if pool.exists():
        meta["pool"] = json.loads(pool.read_text())
    stages = sorted(run_dir.glob("stage[0-9].json"))
    if stages:
        meta["tactics"] = [json.loads(f.read_text())["tactics"] for f in stages]
    if server_cfg.get("model") != "fake" and (run_dir / "server_requests.jsonl").exists():
        tc = subprocess.run([PYTHON, "-m", "bench.check_tokens", str(run_dir)], cwd=ROOT, capture_output=True,
                            text=True)
        print(tc.stdout + tc.stderr, flush=True)
    reqs = run_dir / "requests.jsonl"
    if reqs.exists() and reqs.stat().st_size:
        m = subprocess.run([PYTHON, str(HANDOFF / "data/metrics.py"), str(reqs)], capture_output=True, text=True)
        (run_dir / "metrics.json").write_text(m.stdout if m.returncode == 0 else m.stderr)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    rep = subprocess.run([PYTHON, "-m", "bench.report", str(run_dir)], cwd=ROOT, capture_output=True,
                         text=True)
    print(rep.stdout + rep.stderr, flush=True)
    return meta


def run_named(name: str, fake: bool, port: int, out_root: Path, args) -> dict:
    kind, length = parse_name(name)
    run_dir = out_root / name
    trace = trace_for(kind, length)
    extra = {}
    if args.nsys:
        requests = len(trace.read_text().splitlines()) - 1
        window = nsys_window(kind, length, requests, args.nsys_mbs)
        if window is not None:
            extra["profile_mbs"] = window
    if fake and args.us_per_token is not None:
        extra["fake"] = {"us_per_token": args.us_per_token}
    if args.transport:
        extra["transport"] = args.transport
    if args.intake_procs is not None:
        extra["intake_procs"] = args.intake_procs
    if args.cpu:    # harness check without GPUs: gloo, small activations
        extra["fake"] = {**extra.get("fake", {}), "hidden_width": 64}
    return execute(run_dir, server_config(kind, length, fake, run_dir, device="cpu" if args.cpu else "cuda", **extra),
                   client_config(trace, port, run_dir), port=port, gpus=None if args.cpu else 4, meta={"name": name, "kind": kind, "prompt_tokens": length, "fake": fake,
                                    "trace": str(trace)},
                   client_timeout_s=args.client_timeout, nsys=args.nsys)


def submit(args) -> None:
    out = ROOT / "measured/runs" / ("fake" if args.fake else "")
    logs = out / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    inner = [PYTHON, "-m", "bench.run", *args.names, "--inline"]
    if args.fake:
        inner.append("--fake")
    if args.us_per_token is not None:
        inner += ["--us-per-token", str(args.us_per_token)]
    if args.nsys:
        inner.append("--nsys")
    if args.transport:
        inner += ["--transport", args.transport]
    if args.intake_procs is not None:
        inner += ["--intake-procs", str(args.intake_procs)]
    if args.nsys_mbs:
        inner += ["--nsys-mbs", args.nsys_mbs]
    if args.out_root:
        inner += ["--out-root", str(Path(args.out_root).resolve())]
    script = f"""#!/bin/bash
#SBATCH -J ppimpl-bench
#SBATCH --gres=gpu:4
#SBATCH --time={args.time}
#SBATCH -p {args.partition}
#SBATCH -o {logs}/%j.out
{f"#SBATCH -d {args.dependency}" if args.dependency else ""}
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
    ap.add_argument("--dependency", help="Slurm dependency, e.g. afterany:7034")
    ap.add_argument("--port", type=int)
    ap.add_argument("--us-per-token", type=float, help="FakeStageModel stage time per token")
    ap.add_argument("--transport", choices=["nccl", "ce"], help="RunConfig.transport (default: the config's)")
    ap.add_argument("--intake-procs", type=int, help="RunConfig.intake_procs (0: HTTP in the controller process)")
    ap.add_argument("--gpus", help="with --inline: CUDA_VISIBLE_DEVICES for the server, run without srun steps (two "
                                   "servers side by side in one 8-GPU job; start this under taskset for its CPUs)")
    ap.add_argument("--client-timeout", type=float, default=1200)
    ap.add_argument("--nsys", action="store_true", help="run the server under Nsight Systems (bench/nsys_trace.py parses it)")
    ap.add_argument("--nsys-mbs", help="microbatches to trace, a:b or all (default: one-request all, saturated the "
                    "middle 80%% of the expected microbatches)")
    ap.add_argument("--out-root", help="runs root (default measured/runs, or measured/runs/fake[_cpu])")
    ap.add_argument("--cpu", action="store_true", help="with --fake --inline: stages on CPU (gloo), to check the harness")
    args = ap.parse_args()
    for n in args.names:
        parse_name(n)
    ensure_client_bin()
    if not args.inline:
        if not args.time:
            raise SystemExit("--time is required to submit (run slurm-windows first)")
        return submit(args)
    if args.gpus:
        global DIRECT
        DIRECT = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    job = os.environ.get("SLURM_JOB_ID")
    # 100 ports per job (runs use port + 10 i), all below the ephemeral range (32768+): 7134 lost a bind to one there
    port = args.port or (20000 + int(job) % 120 * 100 if job else 18500)
    if args.cpu and not (args.fake and args.inline):
        raise SystemExit("--cpu needs --fake --inline")
    out = Path(args.out_root) if args.out_root else \
        ROOT / "measured/runs" / ("fake_cpu" if args.cpu else "fake" if args.fake else "")
    failed = {}
    for i, name in enumerate(args.names):   # a failed run does not stop the later ones; the exit status reports it
        try:
            meta = run_named(name, args.fake, port + 10 * i, out, args)   # fresh ports per run
        except Exception as e:
            traceback.print_exc()
            failed[name] = f"{type(e).__name__}: {e}"
            continue
        if why := run_failure(meta, out / name):
            failed[name] = why
    for name, why in failed.items():
        print(f"FAILED {name}: {why}", flush=True)
    if failed:
        raise SystemExit(1)


def run_failure(meta: dict, run_dir: Path) -> str | None:
    """Why a finished run counts as failed (the job's exit status), or None."""
    if meta.get("client_returncode") != 0:
        return f"client exited with {meta.get('client_returncode')}"
    tc = run_dir / "token_check.json"
    if tc.exists():
        s = json.loads(tc.read_text())["summary"]
        if s["no_token"] or not s["ok"]:
            return f"tokens: {s['pass']} pass, {s['fail']} fail, {s['no_token']} without a token"
    return None


if __name__ == "__main__":
    main()
