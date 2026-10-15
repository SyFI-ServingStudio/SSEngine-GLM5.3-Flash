"""The controller: spawns the stage workers, admits requests, forms microbatches, dispatches plans, completes requests.

It runs in the server process's event loop (ppimpl/server.py). Microbatches go to all workers at once, in mb_id order;
workers report `StageDone`; the last stage's report carries the tokens. Logs, in `run_dir` (times in ms from t0, the
first request's arrival at the server):
- `microbatches.jsonl`: one line per (microbatch, stage), 1_contract.md §3;
- `server_requests.jsonl`: one line per request: scheduler order (seq), arrival, first dispatch, token, response;
- `plans.jsonl`: one line per microbatch at dispatch: tokens, dispatch time, whether it was queued behind a running
  stage-0 plan, and that plan's predicted stage-0 end.
When a microbatch forms is `DispatchRule`'s decision (ppimpl/runtime/dispatch.py); what it holds is the scheduler's.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zmq
import zmq.asyncio

from ppimpl.config import BLOCK_TOKENS, RunConfig, dump_config
from ppimpl.engine.kv import KvManager, kv_blocks_for, pool_size
from ppimpl.engine.scheduler import ChunkSpec, PlainScheduler, Request
from ppimpl.runtime.dispatch import NOW, WAIT, DispatchRule
from ppimpl.runtime.transport import (Chunk, Plan, Ready, Shutdown, StageDone, WorkerError, decode_to_controller,
                                      encode, for_stage)

READY_TIMEOUT_S = 1800   # a real stage loads its weights and captures graphs before Ready



def prompt_sha256(token_ids: list[int]) -> str:
    """The prompt's id in measured/reference.jsonl: sha256 of its token ids as little-endian int32."""
    return hashlib.sha256(np.asarray(token_ids, "<i4").tobytes()).hexdigest()

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
        self.rule = DispatchRule(cfg.max_in_flight, cfg.dispatch_ahead >= 1, cfg.form_lead_ms)
        # the scheduler's own cap is not the binding one: the rule admits the extra queued plan
        self.sched = PlainScheduler(KvManager(num_blocks), cfg.scheduler.mb_tokens, self.rule.cap, 1)
        self._timer: asyncio.TimerHandle | None = None
        self._timer_at = None
        self.mb_log = open(self.run_dir / "microbatches.jsonl", "w")
        self.plan_log = open(self.run_dir / "plans.jsonl", "w")
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
        if self._timer is not None:
            self._timer.cancel()
        for f in ("mb_log", "req_log", "plan_log"):
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
            self.plan_log.flush()

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
                                        "arrival_ms": self._ms(arrival_ns), "_prompt": token_ids})
        self.next_seq += 1
        self.sched.add(req)
        self._pump()
        return fut

    def _pump(self):
        """Form and dispatch microbatches while the dispatch rule allows and the scheduler has work."""
        while True:
            now = time.time_ns()
            verdict, at = self.rule.decide(self.sched.in_flight, now)
            if verdict == WAIT:
                self._arm(at, now)
                return
            if verdict != NOW:
                return
            chunks = self.sched.next_microbatch(now)
            if chunks is None:
                return
            self._dispatch(chunks, now)

    def _arm(self, at_ns: int, now_ns: int):
        if self._timer is not None:
            if self._timer_at == at_ns:
                return
            self._timer.cancel()
        self._timer_at = at_ns
        self._timer = asyncio.get_running_loop().call_later((at_ns - now_ns) / 1e9, self._on_timer)

    def _on_timer(self):
        self._timer = self._timer_at = None
        if not self.failed:
            self._pump()

    def _dispatch(self, specs: list[ChunkSpec], now: int):
        mb_id, self.next_mb = self.next_mb, self.next_mb + 1
        queued = len(self.rule.on_stage0) > 0
        pred = self.rule.predicted_end_ns() if queued else None
        tokens = sum(c.num_tokens for c in specs)
        self.rule.dispatched(mb_id, now, tokens)
        self.plan_log.write(json.dumps({"index": mb_id, "tokens": tokens, "dispatch_ms": round(self._ms(now), 4),
                                        "queued": queued, "in_flight": self.sched.in_flight,
                                        "predicted_s0_end_ms": round(self._ms(pred), 4) if pred else None}) + "\n")
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
        if d.stage == 0:
            self.rule.stage0_done(d.mb_id, d.start_ns, d.end_ns)
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
        info["prompt_sha256"] = prompt_sha256(info.pop("_prompt"))   # here, not in submit: intake bursts are the hot path
        self.req_log.write(json.dumps(info) + "\n")
