"""Shared end-to-end harness: start `python -m ssengine serve`, drive it over HTTP, check responses and the logs."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from ssengine.config import BLOCK_TOKENS, STATE_BLOCKS
from ssengine.model.fake import fake_token

ROOT = Path(__file__).resolve().parents[1]


class Server:
    def __init__(self, run_dir: Path, port: int, **cfg):
        self.run_dir, self.port = run_dir, port
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"port": port, "run_dir": str(run_dir), **cfg}
        (run_dir / "server.yaml").write_text(yaml.safe_dump(cfg))
        self.log = open(run_dir / "server.log", "w")
        env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        # its own session: the server, its workers and intake processes form one process group that kill_group()
        # ends, so no test leaves them behind (test_worker_failure_stops_server leaked servers holding its ports)
        self.proc = subprocess.Popen([sys.executable, "-m", "ssengine", "serve", "--config", str(run_dir / "server.yaml")],
                                     cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)

    def wait_ready(self, timeout_s=300):
        t = time.monotonic()
        while time.monotonic() - t < timeout_s:
            assert self.proc.poll() is None, (self.run_dir / "server.log").read_text()
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        raise TimeoutError("server not ready")

    def kill_group(self):
        """SIGKILL whatever is left of the server's process group (a finalizer: safe to call at any time, twice)."""
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if self.proc.poll() is None:
            self.proc.wait(10)
        self.log.close()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        self.kill_group()
        # uvicorn re-raises the SIGTERM it handled once it has shut down: -15 is a clean stop
        return 0 if self.proc.returncode == -signal.SIGTERM else self.proc.returncode


async def send_all(port: int, prompts: dict[str, int], stream_every: int = 2) -> dict[str, dict]:
    """POST every prompt at once (ids -> lengths); returns id -> {token, usage, ttft_s}."""
    limits = httpx.Limits(max_connections=len(prompts) + 10)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=600, limits=limits) as c:
        async def one(i, rid, n):
            body = {"token_ids": [(rid.__hash__() + k) % 1000 for k in range(n)],
                    "sampling_params": {"max_tokens": 1, "temperature": 0, "ignore_eos": True},
                    "stream": i % stream_every == 0}
            t = time.monotonic()
            r = await c.post("/inference/v1/generate", json=body, headers={"x-request-id": rid})
            assert r.status_code == 200, r.text
            if body["stream"]:
                events = [json.loads(line[6:]) for line in r.text.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                assert r.text.rstrip().endswith("data: [DONE]")
                tok = events[0]["choices"][0]["token_ids"]
                usage = events[-1]["usage"]
            else:
                j = r.json()
                tok, usage = j["choices"][0]["token_ids"], j["usage"]
            return rid, {"token": tok, "usage": usage, "ttft_s": time.monotonic() - t}
        res = await asyncio.gather(*(one(i, rid, n) for i, (rid, n) in enumerate(prompts.items())))
    return dict(res)


def check_responses(prompts: dict[str, int], res: dict[str, dict]):
    for rid, n in prompts.items():
        r = res[rid]
        assert r["token"] == [fake_token(rid, n)], (rid, r)
        assert r["usage"]["prompt_tokens"] == n and r["usage"]["completion_tokens"] == 1
        assert r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def check_microbatches(run_dir: Path, prompts: dict[str, int], *, stages=4, mb_tokens=8192, max_in_flight=4):
    lines = [json.loads(x) for x in (run_dir / "microbatches.jsonl").read_text().splitlines()]
    by_mb: dict[int, dict[int, dict]] = {}
    for x in lines:
        assert set(x) == {"stage", "index", "start_ms", "compute_ms", "requests"}
        assert x["compute_ms"] >= 0
        by_mb.setdefault(x["index"], {})[x["stage"]] = x
    assert sorted(by_mb) == list(range(len(by_mb)))
    seen: dict[str, list[tuple[int, int]]] = {}
    for idx in sorted(by_mb):
        mb = by_mb[idx]
        assert sorted(mb) == list(range(stages)), f"mb {idx}: stages {sorted(mb)}"
        reqs = mb[0]["requests"]
        assert all(mb[s]["requests"] == reqs for s in range(stages))
        assert sum(q["append"] for q in reqs) <= mb_tokens
        for s in range(1, stages):     # a stage starts after the previous one ends (it needs its activations)
            assert mb[s]["start_ms"] >= mb[s - 1]["start_ms"] + mb[s - 1]["compute_ms"] - 0.01, (idx, s)
        for q in reqs:
            seen.setdefault(q["id"], []).append((q["prefix"], q["append"]))
    assert set(seen) == set(prompts)
    for rid, chunks in seen.items():   # chunks in order, contiguous, covering the prompt
        pos = 0
        for prefix, append in chunks:
            assert prefix == pos, (rid, chunks)
            pos += append
        assert pos == prompts[rid], (rid, chunks)
    for s in range(stages):            # each stage runs microbatches in order, one at a time
        xs = [by_mb[i][s] for i in sorted(by_mb)]
        for a, b in zip(xs, xs[1:]):
            assert b["start_ms"] >= a["start_ms"] + a["compute_ms"] - 0.01, (s, a["index"], b["index"])
    # in flight <= 4: the [stage-0 start, last-stage end] intervals lie inside [dispatch, last stage done]
    ev = []
    for mb in by_mb.values():
        ev += [(mb[0]["start_ms"], 1), (mb[stages - 1]["start_ms"] + mb[stages - 1]["compute_ms"], -1)]
    cur = peak = 0
    for _, d in sorted(ev, key=lambda e: (e[0], e[1])):
        cur += d
        peak = max(peak, cur)
    assert peak <= max_in_flight          # the gate: stage 0 starts k after k-4 left the last stage
    # the dispatch rule forms ahead of reports: up to 2 more than the limit by the controller's view (plans.jsonl)
    plans = [json.loads(x) for x in (run_dir / "plans.jsonl").read_text().splitlines()]
    assert [p["index"] for p in plans] == sorted(by_mb)
    for p in plans:
        assert p["in_flight"] - p["exiting"] <= max_in_flight, p
        assert p["in_flight"] <= max_in_flight + 2, p
        assert p["sent_ms"] >= p["dispatch_ms"], p
    return by_mb, peak


def blocks_per_request(n: int) -> int:
    return -(-n // BLOCK_TOKENS) + STATE_BLOCKS
