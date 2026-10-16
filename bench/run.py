"""Run handoff workloads against `python -m ppimpl serve` with req-frontend: one entry per run name.

  python -m bench.run one_8k saturated_8k [--fake] --time 00:20:00 [--partition main]   submit one Slurm job
  python -m bench.run one_8k --fake --inline                                                 run here (no Slurm)
  ... --nsys [--nsys-mbs a:b|all]     the server under Nsight Systems -> trace.nsys-rep, trace.sqlite, trace.json

Names: one_<L>, saturated_<L> (4_iteration.md), L in 1k..128k; prefix_bringup and prefix_n<N> (5_prefix_cache.md,
prefix cache on: `keep_sessions`); sched_check_<step>, force_check_<step> and sched_<step>_n<N> (6_scheduler.md, step
one of ppimpl.config.SCHEDULER_STEPS: plain, srpt8k, budget, even, good; prefix_n<N> is sched_plain_n<N>).
Session runs (prefix_n<N>, sched_<step>_n<N>) run the client under `timeout 22m`: 1_contract.md §2.2
says 20 min, but req-frontend first spends ~45 s tokenizing the corpus (measured, loaded host) and t = 0 is its first
submit; rounds answered after 20 min are not counted (data/metrics.py), so the extra 2 min change no number.
The cache on/off token check is bench/prefix_check.py. Each run gets a fresh server; the server and the
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

import msgspec
import yaml

from bench.common import ENWIK9, HANDOFF, REQ_FRONTEND, ROOT, free_port_block
from ppimpl.config import CHECKPOINT, SCHEDULER_STEPS
from ppimpl.runtime.cpus import cpu_list

CLIENT_BIN = Path(os.environ.get("PPIMPL_REQ_FRONTEND_BIN", ROOT / "build/req-frontend/release/session_runner"))
CHECKPOINT = Path(CHECKPOINT)
PYTHON = sys.executable
NSYS = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"


PREFIX_RUNS = ("prefix_bringup",)
CLIENT_TIMEOUT_S = 1200
SESSION_MINUTES = 20
SESSION_CLIENT_LIMIT = "22m"     # 20 min from the first submit + the client's corpus tokenization (~45 s) + margin


def parse_name(name: str) -> tuple[str, int]:
    """(kind, size): size is the prompt length (one, saturated), N (sessions) or 0."""
    if name in PREFIX_RUNS:
        return name.removeprefix("prefix_"), 0
    if m := re.fullmatch(r"prefix_n(\d+)", name):
        return "sessions", int(m.group(1))
    if (m := re.fullmatch(r"(sched_check|force_check)_(\w+)", name)) and m.group(2) in SCHEDULER_STEPS:
        return m.group(1), 0
    if (m := re.fullmatch(r"sched_(\w+)_n(\d+)", name)) and m.group(1) in SCHEDULER_STEPS:
        return "sessions", int(m.group(2))
    m = re.fullmatch(r"(one|saturated)_(\d+)k", name)
    if not m:
        raise SystemExit(f"unknown run name {name!r}: one_<L> or saturated_<L>, L in 1k..128k; "
                         f"{', '.join(PREFIX_RUNS)}; prefix_n<N>; sched_check_<step>, force_check_<step>, "
                         f"sched_<step>_n<N>, step in {', '.join(SCHEDULER_STEPS)}")
    return m.group(1), int(m.group(2)) * 1024


def sched_step(name: str) -> str:
    """The scheduler step of a run name (6_scheduler.md §2): plain unless the name gives one."""
    m = re.fullmatch(r"(?:sched_check|force_check)_(\w+)|sched_(\w+)_n\d+", name)
    return (m.group(1) or m.group(2)) if m else "plain"


def hold_for(trace: Path) -> int:
    """Requests of the trace at t = 0: the scheduling check queues them all before the first microbatch."""
    rows = trace.read_text().splitlines()
    col = rows[0].split(",").index("arrival_time")
    return sum(float(r.split(",")[col]) == 0 for r in rows[1:] if r.strip())


# The queued plan forms this long before stage 0 frees: stage 0's host time from a plan's forming to its forward
# returning with all GPU work enqueued, with CUDA graphs (smoke 7658, worker_host.jsonl): p50 / p90 / p99 7.2 / 10.1 /
# 41.5 ms (the p99 is 8,192-token microbatches with 56k-134k contexts), plus margin. Recorded in each run's
# server_config.yaml.
REAL_FORM_LEAD_MS = 45.0

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
                  tokenizer: Path = CHECKPOINT / "tokenizer.json", token_pool_limit: int | None = None,
                  fmt: str = "text-generation-independent", max_concurrency: int | None = None) -> dict:
    cfg = {
        "input": {"trace": str(trace), "format": fmt},
        "corpus": {"text_file": str(text_file), "tokenizer": str(tokenizer)},
        "server": {"backend": "vllm-tokens", "base_url": f"http://127.0.0.1:{port}", "model": "glm53-flash-pp4",
                   "temperature": 0},
        # no max_concurrency: the saturated run must not be capped (4_iteration.md §5)
        "replay": {"arrival_mode": "trace-timed", "context": {"max_model_len": 1048576}},
        "output": {"directory": str(run_dir)},
    }
    if token_pool_limit:
        cfg["corpus"]["token_pool_limit"] = token_pool_limit
    if max_concurrency:
        cfg["replay"]["max_concurrency"] = max_concurrency
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
            nsys: bool = False, client_limit: str | None = None) -> dict:
    """client_limit: run the client under `timeout <client_limit>` (session runs: 20m); its exit 124 is a normal end."""
    """One run: fresh server, req-frontend, metrics and report. Returns run.json's content."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("run.json", "report.json", "pool.json", "metrics.json", "summary.json",   # a failed run leaves none
              "cache.json", "microbatch_check.txt", "microbatches.jsonl", "server_requests.jsonl", "requests.jsonl",
              "trace.nsys-rep", "trace.sqlite",
              "trace.json", "token_check.json", "cpus.json", *(f"stage{s}.json" for s in range(8))):
        (run_dir / f).unlink(missing_ok=True)
    server_cfg = {**server_cfg, "port": port, "run_dir": str(run_dir)}
    (run_dir / "server.yaml").write_text(yaml.safe_dump(server_cfg, sort_keys=False))
    (run_dir / "client.yaml").write_text(yaml.safe_dump(client_cfg, sort_keys=False))
    argv = client_argv(run_dir / "client.yaml")
    if client_limit:
        argv = ["timeout", client_limit, *argv]
    (run_dir / "client_command.txt").write_text(shlex.join(argv) + "\n")
    meta = {**meta, "port": port, "job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
            client = subprocess.Popen(step(argv, None), cwd=REQ_FRONTEND, stdout=clog, stderr=subprocess.STDOUT)
            fault = watch(run_dir, server, client, client_timeout_s)
        meta["client_s"] = round(time.time() - t, 1)
        if fault:
            meta["fail_fast"] = fault
            print(f"[{run_dir.name}] FAIL FAST: {fault}", flush=True)
        meta["client_returncode"] = 0 if client_limit and client.returncode == 124 and not fault \
            else client.returncode
        meta["client_exit"] = client.returncode
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
    for name in ("pool", "cache"):
        f = run_dir / f"{name}.json"
        if f.exists():
            meta[name] = json.loads(f.read_text())
    stages = sorted(run_dir.glob("stage[0-9].json"))
    if stages:
        meta["tactics"] = [json.loads(f.read_text())["tactics"] for f in stages]
    if server_cfg.get("model") != "fake" and meta.get("kind") in ("one", "saturated") \
            and (run_dir / "server_requests.jsonl").exists():
        tc = subprocess.run([PYTHON, "-m", "bench.check_tokens", str(run_dir)], cwd=ROOT, capture_output=True,
                            text=True)
        print(tc.stdout + tc.stderr, flush=True)
    reqs = run_dir / "requests.jsonl"
    if reqs.exists() and reqs.stat().st_size:
        m = subprocess.run([PYTHON, str(HANDOFF / "data/metrics.py"), str(reqs)], capture_output=True, text=True)
        (run_dir / "metrics.json").write_text(m.stdout if m.returncode == 0 else m.stderr)
    mbs = run_dir / "microbatches.jsonl"
    if meta.get("kind") == "sessions" and mbs.exists() and mbs.stat().st_size:
        m = subprocess.run([PYTHON, str(HANDOFF / "data/microbatch_fit.py"), "check", str(mbs)], capture_output=True,
                           text=True)
        (run_dir / "microbatch_check.txt").write_text(m.stdout + m.stderr)
    jit = in_run_jit(run_dir)
    meta["in_run_jit"] = jit
    if mbs.exists():
        with open(run_dir / "microbatch_check.txt", "a") as f:
            f.write(f"in-run JIT compiles (must be 0): {jit['compiles']}"
                    + (f" by stage {jit['by_stage']}; " + "; ".join(jit['kernels']) if jit["compiles"] else "") + "\n")
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    kind = meta.get("kind")
    report = (["-m", "bench.report", str(run_dir)] if kind in ("one", "saturated")
              else ["-m", "bench.sched", "check", str(run_dir)] if kind in ("sched_check", "force_check")
              else ["-m", "bench.prefix", "run", str(run_dir)])
    rep = subprocess.run([PYTHON, *report], cwd=ROOT, capture_output=True, text=True)
    print(rep.stdout + rep.stderr, flush=True)
    return meta


FAULT_MARKS = ("Traceback", "controller: worker", "exited with")


def run_fault(run_dir: Path, server: subprocess.Popen, seen: dict) -> str | None:
    """A reason to stop the run now, or None: the server exited, a worker, intake or the controller logged a
    traceback (the server keeps answering, with errors, after a worker fails), or the client counts a failed step.
    `seen` keeps each log's read offset, so a call reads only what is new."""
    if server.poll() is not None:
        return f"server exited with {server.returncode}"
    for f in [run_dir / "server.log", *sorted(run_dir.glob("worker*.log")), *sorted(run_dir.glob("intake*.log"))]:
        try:
            with open(f, "rb") as fh:
                fh.seek(seen.get(f.name, 0))
                new = fh.read()
        except OSError:
            continue
        seen[f.name] = seen.get(f.name, 0) + len(new)
        text = new.decode(errors="replace")
        for mark in FAULT_MARKS:
            if mark in text:
                line = next(x for x in text.splitlines() if mark in x)
                return f"{f.name}: {line.strip()[:200]}"
    try:
        with open(run_dir / "client.log", "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            tail = fh.read().decode(errors="replace")
    except OSError:
        return None
    counts = re.findall(r"\bfailed=(\d+)", tail)
    if counts and int(counts[-1]) > 0:
        return f"client: failed={counts[-1]}"
    return None


def watch(run_dir: Path, server: subprocess.Popen, client: subprocess.Popen, timeout_s: float,
          every_s: float = 2.0) -> str | None:
    """Wait for the client; stop it at the first fault (`run_fault`) or after timeout_s. Returns the fault."""
    seen: dict = {}
    t = time.monotonic()
    fault = None
    while client.poll() is None:
        fault = run_fault(run_dir, server, seen) or (
            f"client still running after {timeout_s:.0f} s" if time.monotonic() - t > timeout_s else None)
        if fault:
            break
        time.sleep(every_s)
    else:
        fault = run_fault(run_dir, server, seen)
    if client.poll() is None:
        client.terminate()
        try:
            client.wait(30)
        except subprocess.TimeoutExpired:
            client.kill()
            client.wait()
    return fault


def compressed_bringup(src: Path, out: Path, sessions: list[int] | None, gap_s: float) -> Path:
    """The bring-up sessions `sessions` (all if None) with session k starting gap_s after session k-1, instead of
    every 30 s: the GPUs are not left idle for minutes (the cluster stops idle GPU jobs), and with gap_s above the
    longest session (512k + 8k: 4.4 s + 10 s tool wait + 0.3 s) sessions still never overlap. Rounds and tool waits
    are unchanged. req-frontend gives a session its corpus offset by its position in the file, so a subset gets other
    prompts than the full trace."""
    rows = src.read_text().splitlines()
    head = rows[0].split(",")
    at, sid = head.index("arrival_time_ms"), head.index("session_id")
    order = []
    for row in rows[1:]:
        k = int(row.split(",")[sid])
        if (sessions is None or k in sessions) and k not in order:
            order.append(k)
    lines = [rows[0]]
    for row in rows[1:]:
        f = row.split(",")
        if int(f[sid]) in order:
            f[at] = f"{gap_s * 1e3 * order.index(int(f[sid])):.6f}"
            lines.append(",".join(f))
    out.write_text("\n".join(lines) + "\n")
    return out


def run_named(name: str, fake: bool, port: int, out_root: Path, args) -> dict:
    kind, length = parse_name(name)
    run_dir = out_root / (args.run_name or name)
    if kind in ("one", "saturated"):
        return run_iteration(name, kind, length, fake, port, run_dir, args)
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = "text-generation-session-execution-v2"
    step = sched_step(name)
    hold = 0
    if kind in ("sched_check", "force_check"):   # 6_scheduler.md §1: independent prompts, no max_concurrency
        trace, ccfg, fmt = HANDOFF / "data/bringup" / f"{kind}.csv", {}, "text-generation-independent"
        if kind == "sched_check":
            hold = hold_for(trace)
    elif kind == "sessions":
        trace, ccfg = HANDOFF / "data/sessions_2ps.csv", {"max_concurrency": length}
    else:
        trace, ccfg = HANDOFF / "data/bringup/prefix_sessions.csv", {}   # no max_concurrency (5_prefix_cache.md §5)
        if args.gap_s:
            only = [int(x) for x in args.sessions.split(",")] if args.sessions else None
            trace = compressed_bringup(trace, run_dir / "bringup_sessions.csv", only, args.gap_s)
    if args.trace:
        trace = Path(args.trace).resolve()
    extra = {"keep_sessions": True, "scheduler": msgspec.to_builtins(SCHEDULER_STEPS[step])}
    if hold:
        extra["hold_requests"] = hold
    if fake and args.us_per_token is not None:
        extra["fake"] = {"us_per_token": args.us_per_token}
    if args.intake_procs is not None:
        extra["intake_procs"] = args.intake_procs
    if args.cpu:
        extra["fake"] = {**extra.get("fake", {}), "hidden_width": 64}
    if args.nsys and args.nsys_mbs and args.nsys_mbs != "all":
        extra["profile_mbs"] = [int(x) for x in args.nsys_mbs.split(":")]
    if args.pool_tokens:
        extra["pool_tokens"] = args.pool_tokens
    scfg = {"model": "fake" if fake else "glm53", "device": "cpu" if args.cpu else "cuda", "run_dir": str(run_dir),
            **extra}
    if not fake:
        scfg["checkpoint"] = str(CHECKPOINT)
        scfg["form_lead_ms"] = REAL_FORM_LEAD_MS
    limit = SESSION_CLIENT_LIMIT if kind == "sessions" else None
    return execute(run_dir, scfg, client_config(trace, port, run_dir, fmt=fmt, **ccfg,
                                                token_pool_limit=args.token_pool_limit),
                   port=port, gpus=None if args.cpu else 4,
                   meta={"name": name, "kind": kind, "max_concurrency": ccfg.get("max_concurrency"), "fake": fake,
                         "trace": str(trace), "scheduler_step": step},
                   client_timeout_s=max(args.client_timeout, (SESSION_MINUTES + 5) * 60), nsys=args.nsys,
                   client_limit=args.client_limit or limit)


def run_iteration(name: str, kind: str, length: int, fake: bool, port: int, run_dir: Path, args) -> dict:
    trace = trace_for(kind, length)
    extra = {}
    if args.nsys:
        requests = len(trace.read_text().splitlines()) - 1
        window = nsys_window(kind, length, requests, args.nsys_mbs)
        if window is not None:
            extra["profile_mbs"] = window
    if fake and args.us_per_token is not None:
        extra["fake"] = {"us_per_token": args.us_per_token}
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
    if args.intake_procs is not None:
        inner += ["--intake-procs", str(args.intake_procs)]
    if args.nsys_mbs:
        inner += ["--nsys-mbs", args.nsys_mbs]
    if args.client_timeout != CLIENT_TIMEOUT_S:
        inner += ["--client-timeout", str(args.client_timeout)]
    for opt in ("trace", "client_limit", "pool_tokens", "token_pool_limit", "gap_s", "sessions", "run_name"):
        if getattr(args, opt):
            inner += [f"--{opt.replace('_', '-')}", str(getattr(args, opt))]
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
source {ROOT}/bench/slurm/env.sh
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
    ap.add_argument("--intake-procs", type=int, help="RunConfig.intake_procs (0: HTTP in the controller process)")
    ap.add_argument("--gpus", help="with --inline: CUDA_VISIBLE_DEVICES for the server, run without srun steps (two "
                                   "servers side by side in one 8-GPU job; start this under taskset for its CPUs)")
    ap.add_argument("--client-timeout", type=float, default=CLIENT_TIMEOUT_S,
                    help="stop the client after this many s (prefix/session runs: at least 25 min); a run stopped "
                         "this way fails (fail_fast in run.json, exit 1)")
    ap.add_argument("--nsys", action="store_true", help="run the server under Nsight Systems (bench/nsys_trace.py parses it)")
    ap.add_argument("--nsys-mbs", help="microbatches to trace, a:b or all (default: one-request all, saturated the "
                    "middle 80%% of the expected microbatches)")
    ap.add_argument("--out-root", help="runs root (default measured/runs, or measured/runs/fake[_cpu])")
    ap.add_argument("--cpu", action="store_true", help="with --fake --inline: stages on CPU (gloo), to check the harness")
    ap.add_argument("--trace", help="prefix/session runs: another trace (harness checks)")
    ap.add_argument("--gap-s", type=float, help="prefix_bringup: sessions gap_s apart instead of 30 s (compressed_bringup)")
    ap.add_argument("--sessions", help="prefix_bringup with --gap-s: only these session ids (comma-separated)")
    ap.add_argument("--run-name", help="run directory name (default: the run name)")
    ap.add_argument("--client-limit", help="prefix/session runs: `timeout` for the client (default 22m for prefix_n<N>)")
    ap.add_argument("--pool-tokens", type=int, help="prefix/session runs: cap the KV pool (harness checks)")
    ap.add_argument("--token-pool-limit", type=int, help="req-frontend corpus.token_pool_limit (harness checks)")
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
    out = Path(args.out_root).resolve() if args.out_root else \
        ROOT / "measured/runs" / ("fake_cpu" if args.cpu else "fake" if args.fake else "")
    failed = {}
    for i, name in enumerate(args.names):   # a failed run does not stop the later ones; the exit status reports it
        try:
            meta = run_named(name, args.fake, free_port_block(port + 10 * i), out, args)   # fresh free ports per run
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


def in_run_jit(run_dir: Path) -> dict:
    """Kernels compiled after warm-up (ppimpl/runtime/worker.py FirstUse: jit_stage<S>.jsonl): count, by stage, and
    the distinct (microbatch, kernel) pairs of the first stage that compiled any."""
    rows = [json.loads(line) for f in sorted(run_dir.glob("jit_stage[0-9].jsonl")) for line in f.read_text().splitlines()
            if line.strip()]
    if not rows:     # runs from before jit_stage<S>.jsonl: the workers' log lines (None: no monitor in that run)
        pat = re.compile(r"stage (\d+) mb (-?\d+): first use after warm-up: triton (\S+) (-?[\d.]+) ms")
        logs = [f.read_text(errors="replace") for f in sorted(run_dir.glob("worker[0-9].log"))]
        if logs and not any("JIT monitor activated" in t for t in logs):
            return {"compiles": None, "by_stage": {}, "kernels": []}
        rows = [{"stage": int(m[1]), "mb": int(m[2]), "kernel": m[3], "ms": float(m[4])}
                for t in logs for m in pat.finditer(t)]
    by_stage: dict[int, int] = {}
    for r in rows:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    first = min(by_stage, default=None)
    return {"compiles": len(rows), "by_stage": by_stage,
            "kernels": [f"mb {r['mb']} {r['kernel']} {r['ms']} ms" for r in rows if r["stage"] == first]}


def run_failure(meta: dict, run_dir: Path) -> str | None:
    """Why a finished run counts as failed (the job's exit status), or None. Any kernel compiled after warm-up
    fails it: a compile stalls the whole pipeline (smoke 7658: 0.3-3.3 s each)."""
    if meta.get("fail_fast"):
        return f"stopped early: {meta['fail_fast']}"
    if (jit := meta.get("in_run_jit")) and jit["compiles"]:
        return f"in-run JIT: {jit['compiles']} compiles after warm-up, by stage {jit['by_stage']}"
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
