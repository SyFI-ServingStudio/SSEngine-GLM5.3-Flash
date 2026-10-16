"""The controller: spawns the stage workers, admits requests, forms microbatches, dispatches plans, completes requests.

Two threads. The HTTP server's event loop (ppimpl/serve/server.py) only admits: `submit` checks a request, queues it and
returns a future. The control thread owns everything else: it takes queued requests into the scheduler, receives the
workers' reports, forms and dispatches microbatches and resolves the futures (`call_soon_threadsafe`). So a burst of
thousands of requests cannot hold back a report or a dispatch: in one event loop the burst's handlers all ran before
the report socket was read again (~1 s for 2,048 requests, job 7056).

Microbatches go to all workers at once, in mb_id order; workers report `StageDone`; the last stage's report carries
the tokens. Logs, in `run_dir` (times in ms from t0, the first request's arrival at the server):
- `microbatches.jsonl`: one line per (microbatch, stage), 1_contract.md §3;
- `server_requests.jsonl`: one line per request: scheduler order (seq), arrival, first dispatch, token (when the
  control thread handled the last stage's report), response, prompt sha256; with the prefix cache, how it was served:
  `hit_kind` (none_first_round: no kept context for its key; full; superseded: the kept context is not a prefix of
  the prompt, dropped; evicted: its session's context was evicted earlier; repeat), `kept_tokens` (the kept context
  at admission, 0 if none), `common_prefix_tokens` (prompt and kept context, compared only on a miss),
  `evicted_context_tokens` (evicted), `evicted_sessions_at_admit` and `admit_wait_ms` (at the head, waiting for pool
  room);
- `plans.jsonl`: one line per microbatch at dispatch: tokens, dispatch time, whether it was queued behind a running
  stage-0 plan, and that plan's predicted stage-0 end;
- `cache.json` (rewritten every 10 s and at stop): the pool in blocks and tokens, the prefix cache's counters (hits,
  hit tokens, evicted sessions and tokens, preemptions: always 0), the kept sessions and the pool in use.
Prefix cache (`keep_sessions`, 5_prefix_cache.md §2): a finished request's blocks stay as its context; the next prompt
that starts with that whole context takes them over and computes only the rest (`engine/kv.py`, matched by content).
Its cached tokens go back in `usage.prompt_tokens_details.cached_tokens` and into `server_requests.jsonl`.
When a microbatch forms is `DispatchRule`'s decision (ppimpl/runtime/dispatch.py); what it holds is the scheduler's.
"""

import asyncio
import gc
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import msgspec
import numpy as np
import zmq

from ppimpl.config import BLOCK_TOKENS, RunConfig, dump_config
from ppimpl.engine.kv import KvManager, kv_blocks_for, pool_size
from ppimpl.engine.scheduler import ChunkSpec, PlainScheduler, Request
from ppimpl.runtime.cpus import CONTROLLER_CORES, cpu_list, pin_process, split_cores, threads
from ppimpl.runtime.dispatch import NOW, WAIT, DispatchRule
from ppimpl.runtime.transport import (Chunk, Plan, Ready, Shutdown, StageDone, WorkerError, decode_to_controller,
                                      encode, for_stage)
from ppimpl.serve.slab import SlabReader

READY_TIMEOUT_S = 1800   # a real stage loads its weights and captures graphs before Ready
WAKE_ADDR = "inproc://ppimpl-control-wake"
ADMIT_BATCH = 64   # requests the control thread takes in before it looks at the reports again
# PPIMPL_CONTROL_SPIN_S=S (intakes only): the control thread busy-polls (zmq poll without a timeout) until S seconds
# after the last message or open request. A blocked poll costs ~0.17 ms of wake-up per hop (64 B DEALER -> ROUTER over
# tcp, idle host: 525 us blocked, 360 us spinning), but end to end it was noise (CPU mock, 512k arrival -> dispatch
# 4.47 ms spinning, 4.50 blocked) and it holds a core on a contended host: off by default.
CONTROL_SPIN_S = float(os.environ.get("PPIMPL_CONTROL_SPIN_S", "0"))


class Remote(NamedTuple):
    """A request that came from an intake process (ppimpl/serve/intake.py): where its answer goes."""
    ident: bytes
    key: int


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
        self.t0_ns: int | None = None
        self.ready = False
        self.failed: str | None = None
        # event loop
        self.loop: asyncio.AbstractEventLoop | None = None
        self.live: set[str] = set()                           # ids of requests not answered yet (unique ids)
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()   # -> control thread: (id, tokens, arrival, future)
        self.wake_sent = False    # set by the event loop when it wakes the control thread, cleared by that thread
        self.thread: threading.Thread | None = None
        # control thread
        self.reqs: dict[str, tuple[Request, asyncio.Future, dict]] = {}
        self.inflight_plans: dict[int, tuple[list[dict], int]] = {}   # mb_id -> (log requests, stages left)
        self.next_mb = 0
        self.next_seq = 0                    # order in which requests entered the scheduler
        self.timer_at: int | None = None     # host ns at which the dispatch rule wants to be asked again
        # intake processes (cfg.intake_procs): all their messages go through the control thread
        self.intake = None
        # sockets made by start(); None/empty until then, so stop() after a failed start() closes what exists and
        # the start error is the one raised
        self.ctx = self.reports = self.wake_rx = self.wake_tx = None
        self.plans: list = []
        self.intake_procs: list[subprocess.Popen] = []
        self.intake_cpus: set[int] | None = None
        self.intakes_ready: set[bytes] = set()
        self.slabs = SlabReader()            # the intakes' shared-memory prompt slabs (ppimpl/serve/slab.py)
        self.slab_views: set[str] = set()    # requests admitted on a view of a slab, not yet copied out (_own)
        self.answered: dict[str, dict] = {}  # answered, response not yet sent: id -> log record
        self.pack, self.unpack = msgspec.msgpack.Encoder(), msgspec.msgpack.Decoder()

    # --- lifecycle ------------------------------------------------------------------------------------------------

    async def start(self):
        cfg = self.cfg
        cfg_path = self.run_dir / "server_config.yaml"
        cfg_path.write_text(dump_config(cfg))
        self.loop = asyncio.get_running_loop()
        self.ctx = zmq.Context()
        self.reports = self.ctx.socket(zmq.PULL)
        self.reports.bind(cfg.report_addr())
        self.wake_rx = self.ctx.socket(zmq.PULL)     # the control thread sleeps in a poll on reports + this
        self.wake_rx.bind(WAKE_ADDR)
        self.wake_tx = self.ctx.socket(zmq.PUSH)     # used by the event loop only
        self.wake_tx.setsockopt(zmq.SNDHWM, 0)
        self.wake_tx.connect(WAKE_ADDR)
        if cfg.intake_procs:
            self.intake = self.ctx.socket(zmq.ROUTER)
            self.intake.setsockopt(zmq.SNDHWM, 0)
            self.intake.setsockopt(zmq.RCVHWM, 0)
            self.intake.bind(cfg.intake_addr())
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
            ready = await asyncio.to_thread(self._wait_ready)
        except BaseException:
            for p in self.procs:
                p.kill()
            raise
        self._pin()
        num_blocks = pool_size([r.num_blocks for r in ready], cfg.pool_tokens)
        self.rule = DispatchRule(cfg.max_in_flight, cfg.dispatch_ahead >= 1, cfg.form_lead_ms)
        # the scheduler's own cap is not the binding one: the rule admits the extra queued plan
        self.sched = PlainScheduler(KvManager(num_blocks, keep_sessions=cfg.keep_sessions, content_keys=True),
                                    cfg.scheduler.mb_tokens, self.rule.cap, 1)
        self.mb_log = open(self.run_dir / "microbatches.jsonl", "w")
        self.plan_log = open(self.run_dir / "plans.jsonl", "w")
        self.req_log = open(self.run_dir / "server_requests.jsonl", "w")
        (self.run_dir / "pool.json").write_text(json.dumps(
            {"ready_blocks": [r.num_blocks for r in ready], "num_blocks": num_blocks,
             "pool_tokens": num_blocks * BLOCK_TOKENS, "keep_sessions": cfg.keep_sessions}))
        self._write_cache()
        # Leave start-up objects (torch, FastAPI, ...) out of later collections: a full one took ~50 ms with the GIL
        # held, stalling the control thread during a burst.
        gc.collect()
        gc.freeze()
        self.thread = threading.Thread(target=self._control, name="control", daemon=True)
        self.thread.start()
        self.ready = True
        for i in range(cfg.intake_procs):     # their /health answers 200 once all of them are connected
            log = open(self.run_dir / f"intake{i}.log", "w")
            p = subprocess.Popen([sys.executable, "-m", "ppimpl", "intake", "--config", str(cfg_path), "--index", str(i)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            if self.intake_cpus:
                os.sched_setaffinity(p.pid, self.intake_cpus)
            self.intake_procs.append(p)

    def _pin(self):
        """The controller takes the job's CPUs that the workers did not pin themselves to; cpus.json records both."""
        allowed = os.sched_getaffinity(0)
        workers = [os.sched_getaffinity(p.pid) for p in self.procs]
        rest = allowed - set().union(*workers)
        mine = rest
        if rest and rest != allowed:
            if self.cfg.intake_procs:
                mine, self.intake_cpus = split_cores(rest, CONTROLLER_CORES)
            pin_process(mine)
        info = {"job_cpus": cpu_list(allowed), "workers": [cpu_list(w) for w in workers], "controller": cpu_list(mine),
                "intakes": cpu_list(self.intake_cpus or mine),
                "threads": {"controller": threads(), "workers": [threads(p.pid) for p in self.procs]}}
        (self.run_dir / "cpus.json").write_text(json.dumps(info) + "\n")
        print("cpus: " + json.dumps(info), flush=True)

    def _wait_ready(self) -> list[Ready]:
        ready: dict[int, Ready] = {}
        deadline = time.monotonic() + READY_TIMEOUT_S
        while len(ready) < self.n:
            self._check_procs()
            if time.monotonic() > deadline:
                raise RuntimeError("workers not ready in time")
            if self.reports.poll(1000):
                msg = decode_to_controller(self.reports.recv())
                if isinstance(msg, WorkerError):
                    raise RuntimeError(f"worker {msg.stage} failed:\n{msg.error}")
                if isinstance(msg, Ready):
                    ready[msg.stage] = msg
        return [ready[s] for s in range(self.n)]

    def _check_procs(self):
        for s, p in enumerate(self.procs):
            if p.poll() is not None:
                raise RuntimeError(f"worker {s} exited with {p.returncode}; see {self.run_dir}/worker{s}.log")
        for i, p in enumerate(self.intake_procs):
            if p.poll() is not None:
                raise RuntimeError(f"intake {i} exited with {p.returncode}; see {self.run_dir}/intake{i}.log")

    async def stop(self):
        procs, self.intake_procs = self.intake_procs, []   # stopping them is not a failure (_check_procs)
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                await asyncio.to_thread(p.wait, 10)
            except subprocess.TimeoutExpired:
                p.kill()
        if self.thread is not None and self.thread.is_alive():
            self.inbox.put(None)
            self.wake_tx.send(b"")
            await asyncio.to_thread(self.thread.join, 10)
        for sock in self.plans:
            sock.send(encode(Shutdown()))
        for p in self.procs:
            try:
                await asyncio.to_thread(p.wait, 60)
            except subprocess.TimeoutExpired:
                p.kill()
        for sock in (*self.plans, self.reports, self.wake_rx, self.wake_tx, self.intake):
            if sock is not None:
                sock.close(linger=0)
        for f in ("mb_log", "req_log", "plan_log"):
            if hasattr(self, f):
                getattr(self, f).close()
        if self.sched is not None:
            self._write_cache()

    def _write_cache(self):
        """cache.json: the pool and the prefix cache's counters (5_prefix_cache.md §8)."""
        kv = self.sched.kv
        info = {"keep_sessions": kv.keep_sessions, "num_blocks": kv.num_blocks,
                "pool_tokens": kv.num_blocks * BLOCK_TOKENS, **kv.stats, "kept_sessions": len(kv.kept),
                "kept_tokens": sum(k.tokens for k in kv.kept.values()), "used_tokens": kv.used_tokens,
                "free_blocks": kv.pool.num_free, "time_ms": self._ms(time.time_ns()) if self.t0_ns else None}
        tmp = self.run_dir / "cache.json.tmp"
        tmp.write_text(json.dumps(info) + "\n")
        tmp.replace(self.run_dir / "cache.json")

    def _ms(self, ns: int) -> float:
        return (ns - self.t0_ns) / 1e6

    # --- event loop -----------------------------------------------------------------------------------------------

    def submit(self, req_id: str, token_ids: list[int], arrival_ns: int) -> asyncio.Future:
        """Check a request and queue it for the control thread; the future gets (token, info)."""
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
        while req_id in self.live:
            req_id, k = f"{base}#{k}", k + 1
        self.live.add(req_id)
        fut = self.loop.create_future()
        self.inbox.put((req_id, token_ids, arrival_ns, fut))
        if not self.wake_sent:    # one wake per batch the control thread takes, not one per request
            self.wake_sent = True
            self.wake_tx.send(b"")
        return fut

    def _answered(self, done: list[tuple[asyncio.Future, int, dict]]):
        for fut, tok, info in done:
            self.live.discard(info["id"])
            if not fut.done():
                fut.set_result((tok, info))

    def log_response(self, info: dict, sent_ns: int | None = None):
        """After the response went out."""
        info["response_ms"] = self._ms(sent_ns or time.time_ns())
        info["prompt_sha256"] = prompt_sha256(info.pop("_prompt"))   # here, not on intake: bursts are the hot path
        self.req_log.write(json.dumps(info) + "\n")

    # --- control thread -------------------------------------------------------------------------------------------

    def _control(self):
        poller = zmq.Poller()
        poller.register(self.reports, zmq.POLLIN)
        poller.register(self.wake_rx, zmq.POLLIN)
        if self.intake is not None:
            poller.register(self.intake, zmq.POLLIN)
        next_check = time.monotonic() + 1.0
        next_cache = time.monotonic() + 10.0
        spin = self.intake is not None and CONTROL_SPIN_S > 0
        busy_at = time.monotonic()     # the last message or open request
        try:
            more = False   # requests left in the inbox after this round's batch
            while True:
                if spin and self.reqs:
                    busy_at = time.monotonic()
                timeout_ms = 0 if more or (spin and time.monotonic() - busy_at < CONTROL_SPIN_S) else 1000
                if self.timer_at is not None:
                    timeout_ms = min(timeout_ms, max(0, math.ceil((self.timer_at - time.time_ns()) / 1e6)))
                if poller.poll(timeout_ms) and spin:
                    busy_at = time.monotonic()
                while self.reports.poll(0):          # reports first: they free in-flight slots and time stage 0
                    msg = decode_to_controller(self.reports.recv())
                    if isinstance(msg, WorkerError):
                        raise RuntimeError(f"worker {msg.stage} failed:\n{msg.error}")
                    if isinstance(msg, StageDone):
                        self._on_stage_done(msg)
                while self.wake_rx.poll(0):
                    self.wake_rx.recv()
                self.wake_sent = False
                more = False
                for _ in range(ADMIT_BATCH):   # bounded, so a burst of requests cannot hold back the reports
                    try:
                        item = self.inbox.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        return
                    self._admit(*item)
                else:
                    more = True
                if self.intake is not None:
                    for _ in range(ADMIT_BATCH):
                        try:
                            frames = self.intake.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self._from_intake(frames)
                    else:
                        more = True
                if self.timer_at is not None and time.time_ns() >= self.timer_at:
                    self.timer_at = None
                self._pump()
                for req_id in list(self.slab_views):
                    self._own(req_id)
                if time.monotonic() >= next_check:
                    next_check = time.monotonic() + 1.0
                    self._check_procs()
                    for f in (self.mb_log, self.plan_log, self.req_log):
                        f.flush()
                    if next_check >= next_cache:
                        next_cache = next_check + 10.0
                        self._write_cache()
        except Exception as e:
            self._fail(str(e) if isinstance(e, RuntimeError) else traceback.format_exc())

    def _fail(self, why: str):
        """Fail every open request and stop the server: a broken pipeline must not keep answering /health."""
        self.failed = why
        print(f"controller: {why}", file=sys.stderr, flush=True)
        futs = [fut for _, fut, _ in self.reqs.values()]
        for fut in futs:
            if isinstance(fut, Remote):
                self.intake.send_multipart([fut.ident, b"E", self.pack.encode((fut.key, why))])
        futs = [f for f in futs if not isinstance(f, Remote)]
        while True:
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                futs.append(item[3])
        self.loop.call_soon_threadsafe(_fail_all, futs, why)
        os.kill(os.getpid(), 15)

    def _from_intake(self, frames: list[bytes]):
        ident, kind = frames[0], frames[1]
        if kind == b"Q" or kind == b"M":
            key, req_id, arrival_ns, *stamps = self.unpack.decode(frames[2])
            ids = np.frombuffer(frames[3], np.int32) if kind == b"Q" else \
                self.slabs.view(*self.unpack.decode(frames[3]))
            recv_ns = time.time_ns()     # the ids are here
            why = self.failed or (None if self.sched.kv.fits_ever(len(ids))
                                  else f"prompt of {len(ids)} tokens does not fit the KV pool")
            if why:
                self.intake.send_multipart([ident, b"E", self.pack.encode((key, why))])
                return
            if self.t0_ns is None:
                self.t0_ns = arrival_ns
                (self.run_dir / "t0.json").write_text(json.dumps({"t0_unix_ns": arrival_ns}) + "\n")
            base, k = req_id, 1
            while req_id in self.live:
                req_id, k = f"{base}#{k}", k + 1
            self.live.add(req_id)
            if kind == b"M":
                self.slab_views.add(req_id)   # _own copies them after this round's dispatch
            self._admit(req_id, ids, arrival_ns, Remote(ident, key), (*stamps, recv_ns))
        elif kind == b"D":
            req_id, sent_ns = self.unpack.decode(frames[2])
            self.log_response(self.answered.pop(req_id), sent_ns)
        elif kind == b"R":
            if len(frames) > 2:
                self.slabs.attach(frames[2].decode())
            self.intakes_ready.add(ident)
            if len(self.intakes_ready) == self.cfg.intake_procs:
                for i in self.intakes_ready:
                    self.intake.send_multipart([i, b"G"])
        elif kind == b"S":   # /shutdown at an intake: the serve process stops on SIGTERM (ppimpl/serve/server.py)
            os.kill(os.getpid(), signal.SIGTERM)

    def _admit(self, req_id: str, token_ids, arrival_ns: int, fut: asyncio.Future, stamps: tuple = ()):
        """`stamps` (intake requests): host ns of body read, ids parsed, received here; logged as `host_ms`."""
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.asarray(token_ids, np.int32)
        req = Request(req_id, len(token_ids), arrival_ns, token_ids)
        info = {"id": req_id, "seq": self.next_seq, "prompt_tokens": len(token_ids),
                "arrival_ms": self._ms(arrival_ns), "_prompt": token_ids}
        self.reqs[req_id] = (req, fut, info)
        self.next_seq += 1
        try:
            self._admit_inner(req, info)
        finally:
            if len(stamps) == 3:     # phases of arrival -> admitted, ms from the arrival
                info["host_ms"] = {k: round((t - arrival_ns) / 1e6, 3) for k, t in
                                   zip(("body", "parsed", "received", "admitted"), (*stamps, time.time_ns()))}

    def _admit_inner(self, req: Request, info: dict):
        req_id = req.req_id
        token_ids = req.token_ids
        if self.sched.kv.keep_sessions:
            req.session = self.sched.kv.find_session(token_ids)
            tok = self.sched.kv.repeat_token(req)
            if tok is not None:      # the same prompt as a kept context: its token, nothing computed
                req.cached = req.num_tokens
                info["repeat"] = True
                info.update(hit_kind="repeat", kept_tokens=req.num_tokens, common_prefix_tokens=req.num_tokens,
                            evicted_sessions_at_admit=0, admit_wait_ms=0.0)
                done = self._reply(req, tok, time.time_ns())
                if done:
                    self.loop.call_soon_threadsafe(self._answered, done)
                return
        self.sched.add(req)

    def _pump(self):
        """Form and dispatch microbatches while the dispatch rule allows and the scheduler has work."""
        while True:
            now = time.time_ns()
            verdict, at = self.rule.decide(self.sched.in_flight, now)
            if verdict == WAIT:
                self.timer_at = at
                return
            if verdict != NOW:
                return
            chunks = self.sched.next_microbatch(now)
            if chunks is None:
                return
            self._dispatch(chunks, now)

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
            ids = r.token_ids[c.start:end]
            chunks.append(Chunk(r.req_id, c.start, c.num_tokens, r.alloc.kv_blocks[:kv_blocks_for(end)],
                                r.alloc.state_blocks, c.is_last, ids if isinstance(ids, list) else ids.tolist()))
            info = self.reqs[r.req_id][2]
            if "first_dispatch_ms" not in info:
                info["first_dispatch_ms"] = self._ms(now)
                info["cached"] = r.cached
                if r.admit_info is not None:
                    info.update(r.admit_info)
                info["admit_wait_ms"] = (round((r.admit_ns - r.room_wait_ns) / 1e6, 3)
                                         if r.room_wait_ns is not None and r.admit_ns is not None else 0.0)
            info["chunks"] = info.get("chunks", 0) + 1
        plan = Plan(mb_id, chunks)
        for s in range(self.n):
            self.plans[s].send(encode(for_stage(plan, s)))
        self.inflight_plans[mb_id] = ([{"id": c.req.req_id, "prefix": c.start, "append": c.num_tokens}
                                       for c in specs], self.n)

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
            done = []
            for req_id, tok in d.tokens:
                if req_id in self.slab_views:   # before its context is kept
                    self._own(req_id)
                req = self.reqs[req_id][0]
                req.output_token = tok
                self.sched.finish(req)
                done += self._reply(req, tok, now)
            if done:
                self.loop.call_soon_threadsafe(self._answered, done)

    def _own(self, req_id: str):
        """A request admitted on a view of an intake's slab: copy its ids out (the intake reuses the range once the
        request is answered; the scheduler's kept contexts and the response log keep them longer)."""
        self.slab_views.discard(req_id)
        entry = self.reqs.get(req_id)
        if entry is None:
            return
        req, _, info = entry
        req.token_ids = req.token_ids.copy()
        info["_prompt"] = req.token_ids

    def _reply(self, req: Request, tok: int, now: int) -> list:
        """Send the request's answer (intake) or return it for the event loop (in-process HTTP)."""
        if req.req_id in self.slab_views:
            self._own(req.req_id)
        _, fut, info = self.reqs.pop(req.req_id)
        info["token_ms"] = self._ms(now)
        info["token"] = tok
        info["cached"] = req.cached
        if isinstance(fut, Remote):
            self.live.discard(req.req_id)
            self.answered[req.req_id] = info
            self.intake.send_multipart([fut.ident, b"A", self.pack.encode((fut.key, req.req_id, tok, req.cached))])
            return []
        return [(fut, tok, info)]


def _fail_all(futs: list[asyncio.Future], why: str):
    for fut in futs:
        if not fut.done():
            fut.set_exception(RuntimeError(why))
