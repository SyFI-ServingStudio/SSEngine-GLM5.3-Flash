"""The controller: spawns the stage workers, admits requests, forms microbatches, dispatches plans, completes requests.

It runs in the server process's event loop (ppimpl/server.py). Microbatches go to all workers at once, in mb_id order;
workers report `StageDone`; the last stage's report carries the tokens. Logs, in `run_dir` (times in ms from t0, the
first request's arrival at the server):
- `microbatches.jsonl`: one line per (microbatch, stage), 1_contract.md §3;
- `server_requests.jsonl`: one line per request: scheduler order (seq), arrival, first dispatch, token, response.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import zmq
import zmq.asyncio

from ppimpl.config import BLOCK_TOKENS, RunConfig, dump_config
from ppimpl.engine.kv import KvManager, kv_blocks_for, pool_size
from ppimpl.engine.scheduler import ChunkSpec, PlainScheduler, Request
from ppimpl.runtime.transport import (Chunk, Plan, Ready, Shutdown, StageDone, WorkerError, decode_to_controller,
                                      encode, for_stage)

READY_TIMEOUT_S = 1800   # a real stage loads its weights and captures graphs before Ready


class RequestError(Exception):
    """A request the engine cannot serve (HTTP 400)."""


class Controller:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.n = cfg.num_stages
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.procs: list[subprocess.Popen] = []
        self.sched: PlainScheduler | None = None
        self.reqs: dict[str, tuple[Request, asyncio.Future, dict]] = {}
        self.inflight_plans: dict[int, tuple[list[dict], int]] = {}   # mb_id -> (log requests, stages left)
        self.next_mb = 0
        self.next_seq = 0     # order in which requests entered the scheduler
        self.t0_ns: int | None = None
        self.ready = False
        self.failed: str | None = None
        self._tasks: list[asyncio.Task] = []

    # --- lifecycle ------------------------------------------------------------------------------------------------

    async def start(self):
        cfg = self.cfg
        cfg_path = self.run_dir / "server_config.yaml"
        cfg_path.write_text(dump_config(cfg))
        self.actx = zmq.asyncio.Context()
        self.reports = self.actx.socket(zmq.PULL)
        self.reports.bind(cfg.report_addr())
        self.ctx = zmq.Context()
        self.plans = []
        pkg_root = str(Path(__file__).resolve().parents[2])
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH")]))}
        for s in range(self.n):
            log = open(self.run_dir / f"worker{s}.log", "w")
            self.procs.append(subprocess.Popen(
                [sys.executable, "-m", "ppimpl", "worker", "--config", str(cfg_path), "--stage", str(s)],
                stdout=log, stderr=subprocess.STDOUT, env=env))
            sock = self.ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.connect(cfg.plan_addr(s))
            self.plans.append(sock)
        try:
            ready = await self._wait_ready()
        except BaseException:
            for p in self.procs:
                p.kill()
            raise
        num_blocks = pool_size([r.num_blocks for r in ready], cfg.pool_tokens)
        self.sched = PlainScheduler(KvManager(num_blocks), cfg.mb_tokens, cfg.max_in_flight, cfg.dispatch_ahead)
        self.mb_log = open(self.run_dir / "microbatches.jsonl", "w")
        self.req_log = open(self.run_dir / "server_requests.jsonl", "w")
        (self.run_dir / "pool.json").write_text(json.dumps(
            {"ready_blocks": [r.num_blocks for r in ready], "num_blocks": num_blocks,
             "pool_tokens": num_blocks * BLOCK_TOKENS}))
        self._tasks = [asyncio.create_task(self._report_loop()), asyncio.create_task(self._watchdog())]
        self.ready = True

    async def _wait_ready(self) -> list[Ready]:
        ready: dict[int, Ready] = {}
        deadline = time.monotonic() + READY_TIMEOUT_S
        while len(ready) < self.n:
            self._check_procs()
            if time.monotonic() > deadline:
                raise RuntimeError("workers not ready in time")
            if await self.reports.poll(1000):
                msg = decode_to_controller(await self.reports.recv())
                if isinstance(msg, WorkerError):
                    raise RuntimeError(f"worker {msg.stage} failed:\n{msg.error}")
                if isinstance(msg, Ready):
                    ready[msg.stage] = msg
        return [ready[s] for s in range(self.n)]

    def _check_procs(self):
        for s, p in enumerate(self.procs):
            if p.poll() is not None:
                raise RuntimeError(f"worker {s} exited with {p.returncode}; see {self.run_dir}/worker{s}.log")

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        for sock in self.plans:
            sock.send(encode(Shutdown()))
        for p in self.procs:
            try:
                await asyncio.to_thread(p.wait, 60)
            except subprocess.TimeoutExpired:
                p.kill()
        for sock in self.plans:
            sock.close(linger=0)
        self.reports.close(linger=0)
        for f in ("mb_log", "req_log"):
            if hasattr(self, f):
                getattr(self, f).close()

    async def _watchdog(self):
        while True:
            await asyncio.sleep(1.0)
            try:
                self._check_procs()
            except RuntimeError as e:
                self._fail(str(e))
                return
            self.mb_log.flush()
            self.req_log.flush()

    def _fail(self, why: str):
        self.failed = why
        print(f"controller: {why}", file=sys.stderr, flush=True)
        for _, fut, _ in self.reqs.values():
            if not fut.done():
                fut.set_exception(RuntimeError(why))
        os.kill(os.getpid(), 15)   # stop the server: a broken pipeline must not keep answering /health

    # --- requests -------------------------------------------------------------------------------------------------

    def _ms(self, ns: int) -> float:
        return (ns - self.t0_ns) / 1e6

    def submit(self, req_id: str, token_ids: list[int], arrival_ns: int) -> asyncio.Future:
        if self.failed:
            raise RequestError(self.failed)
        if not token_ids:
            raise RequestError("empty prompt")
        if not self.sched.kv.fits_ever(len(token_ids)):
            raise RequestError(f"prompt of {len(token_ids)} tokens does not fit the KV pool")
        if self.t0_ns is None:
            self.t0_ns = arrival_ns
            (self.run_dir / "t0.json").write_text(json.dumps({"t0_unix_ns": arrival_ns}) + "\n")
        base, k = req_id, 1
        while req_id in self.reqs:
            req_id, k = f"{base}#{k}", k + 1
        req = Request(req_id, len(token_ids), arrival_ns, token_ids)
        fut = asyncio.get_running_loop().create_future()
        self.reqs[req_id] = (req, fut, {"id": req_id, "seq": self.next_seq, "prompt_tokens": len(token_ids),
                                        "arrival_ms": self._ms(arrival_ns)})
        self.next_seq += 1
        self.sched.add(req)
        self._pump()
        return fut

    def _pump(self):
        """Dispatch microbatches while the scheduler forms them."""
        while (chunks := self.sched.next_microbatch()) is not None:
            self._dispatch(chunks)

    def _dispatch(self, specs: list[ChunkSpec]):
        mb_id, self.next_mb = self.next_mb, self.next_mb + 1
        now = time.time_ns()
        chunks = []
        for c in specs:
            r = c.req
            end = c.start + c.num_tokens
            chunks.append(Chunk(r.req_id, c.start, c.num_tokens, r.alloc.kv_blocks[:kv_blocks_for(end)],
                                r.alloc.state_blocks, c.is_last, r.token_ids[c.start:end]))
            info = self.reqs[r.req_id][2]
            info.setdefault("first_dispatch_ms", self._ms(now))
            info["chunks"] = info.get("chunks", 0) + 1
        plan = Plan(mb_id, chunks)
        for s in range(self.n):
            self.plans[s].send(encode(for_stage(plan, s)))
        self.inflight_plans[mb_id] = ([{"id": c.req.req_id, "prefix": c.start, "append": c.num_tokens}
                                       for c in specs], self.n)

    async def _report_loop(self):
        while True:
            msg = decode_to_controller(await self.reports.recv())
            if isinstance(msg, WorkerError):
                self._fail(f"worker {msg.stage} failed:\n{msg.error}")
                return
            if isinstance(msg, StageDone):
                self._on_stage_done(msg)

    def _on_stage_done(self, d: StageDone):
        reqs, left = self.inflight_plans[d.mb_id]
        self.mb_log.write(json.dumps({"stage": d.stage, "index": d.mb_id, "start_ms": round(self._ms(d.start_ns), 4),
                                      "compute_ms": round((d.end_ns - d.start_ns) / 1e6, 4),
                                      "requests": reqs}) + "\n")
        if left == 1:
            del self.inflight_plans[d.mb_id]
        else:
            self.inflight_plans[d.mb_id] = (reqs, left - 1)
        self.sched.stage_done(d.stage, self.n - 1)
        if d.tokens:
            now = time.time_ns()
            for req_id, tok in d.tokens:
                req, fut, info = self.reqs.pop(req_id)
                self.sched.finish(req)
                info["token_ms"] = self._ms(now)
                info["token"] = tok
                if not fut.done():
                    fut.set_result((tok, info))
        self._pump()

    def log_response(self, info: dict):
        info["response_ms"] = self._ms(time.time_ns())
        self.req_log.write(json.dumps(info) + "\n")
