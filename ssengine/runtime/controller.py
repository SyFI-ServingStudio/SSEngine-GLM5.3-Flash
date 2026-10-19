"""The controller: spawns the stage workers, admits requests, forms microbatches, dispatches plans, completes requests.

Two threads. The HTTP server's event loop (ssengine/serve/server.py) only admits: `submit` checks a request, queues it and
returns a future. The control thread owns everything else: it takes queued requests into the scheduler, receives the
workers' reports, forms and dispatches microbatches and resolves the futures (`call_soon_threadsafe`). So a burst of
thousands of requests cannot hold back a report or a dispatch: in one event loop the burst's handlers all ran before
the report socket was read again (~1 s for 2,048 requests, job 7056).

Microbatches go to all workers at once, in mb_id order; workers report `StageDone`; the last stage's report carries
the tokens. Logs, in `run_dir` (times in ms from t0, the first request's arrival at the server):
- `microbatches.jsonl`: one line per (microbatch, stage), 1_contract.md §3;
- `worker_host.jsonl`: per (microbatch, stage), when the worker received the plan, called the stage forward and it
  returned (host clock): with microbatches.jsonl's GPU start and end, whether the stage waited for its host;
- `server_requests.jsonl`: one line per request: scheduler order (seq), arrival, first dispatch, token (when the
  control thread handled the last stage's report), response, prompt sha256; with the prefix cache, how it was served:
  `hit_kind` (none_first_round: no kept context for its key; full; superseded: the kept context is not a prefix of
  the prompt, dropped; evicted: its session's context was evicted earlier; repeat), `kept_tokens` (the kept context
  at admission, 0 if none), `common_prefix_tokens` (prompt and kept context, compared only on a miss),
  `evicted_context_tokens` (evicted), `evicted_sessions_at_admit` and `admit_wait_ms` (at the head, waiting for pool
  room);
- `plans.jsonl`: one line per microbatch: tokens, dispatch (forming) time, the time its plan was sent to the
  workers, whether stage 0 had a plan not reported done, the in-flight count with it (by reports), how many of
  those the rule projected to leave before it starts (`exiting`) and its projected stage-0 start;
- `cache.json` (rewritten every 10 s and at stop): the pool in blocks and tokens, the prefix cache's counters (hits,
  hit tokens, evicted sessions and tokens, preemptions: always 0), the kept sessions and the pool in use.
Prefix cache (`keep_sessions`, 5_prefix_cache.md §2): a finished request's blocks stay as its context; the next prompt
that starts with that whole context takes them over and computes only the rest (`engine/kv.py`, matched by content).
Its cached tokens go back in `usage.prompt_tokens_details.cached_tokens` and into `server_requests.jsonl`.
When a microbatch forms is `DispatchRule`'s decision (ssengine/runtime/dispatch.py); what it holds is the scheduler's.
Tiers (`tiers`, dev:research/TIERS.md): the KV pool's leader (`ssengine/kvpool`, in the scheduler) decides every byte
move; the control thread sends ops whose dependencies are done to all workers' tier sockets (`TierBatch`) and
completes an op when all of them reported it (`TierDone`): `tier_ops.jsonl`, one line per op (sent, done per stage,
bytes per GPU); `server_requests.jsonl` gains the tier, tokens read and read start/landing of a round served from a
tier, and `cache.json` the tiers' counters.
"""

import asyncio
import gc
import hashlib
import json
import math
import os
import queue
import re
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

from ssengine.config import BLOCK_TOKENS, CONTRACT_BLOCKS, PAGE_TOKENS, RunConfig, dump_config
from ssengine.engine.kv import KvManager, kv_pages_for, pool_size
from ssengine.engine.scheduler import ChunkSpec, Request, Scheduler
from ssengine.kvpool import BalancedKVPool
from ssengine.kvpool.manager import TierOp
from ssengine.kvpool.member import tier_move
from torch.cuda.nvtx import range as nvtx_range   # host ranges for Nsight Systems (benchmarks/run.py --nsys)

from ssengine.runtime.cpus import CONTROLLER_CORES, cpu_list, pin_process, split_cores, threads
from ssengine.runtime.dispatch import NOW, WAIT, DispatchRule
from ssengine.runtime.transport import (Chunk, Plan, Ready, Shutdown, StageDone, TierBatch, TierDone, WorkerError,
                                      decode_to_controller, encode, for_stage)
from ssengine.serve.slab import SlabReader

READY_TIMEOUT_S = 1800   # a real stage loads its weights and captures graphs before Ready
WAKE_ADDR = "inproc://ssengine-control-wake"
ADMIT_BATCH = 64   # requests the control thread takes in before it looks at the reports again
WATCHDOG_S = 30.0  # no stage or tier op done for this long while requests wait: log the scheduler's state


class Remote(NamedTuple):
    """A request that came from an intake process (ssengine/serve/intake.py): where its answer goes."""
    ident: bytes
    key: int


def prompt_sha256(token_ids: list[int]) -> str:
    """The prompt's id in benchmarks/data/reference.jsonl: sha256 of its token ids as little-endian int32."""
    return hashlib.sha256(np.asarray(token_ids, "<i4").tobytes()).hexdigest()


class RequestError(Exception):
    """A request the engine cannot serve (HTTP 400)."""


def id_order(req_id: str) -> tuple:
    """Natural order of request ids: sched_2 before sched_10."""
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", req_id))


class Controller:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.n = cfg.num_stages
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.procs: list[subprocess.Popen] = []
        self.shard_set = None                    # tiers: the SSD shard set this run holds (kvpool/shards.py)
        self.sched: Scheduler | None = None
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
        # RunConfig.hold_requests: requests held out of the scheduler until that many have arrived, then added in id
        # order (simultaneous submits reach us in a random order; the trace orders them by id)
        self.held: list[Request] | None = [] if cfg.hold_requests else None
        self.timer_at: int | None = None     # host ns at which the dispatch rule wants to be asked again
        self.gated: tuple | None = None      # (chunks, formed ns): a formed microbatch whose blocks an eviction's
                                             # write still copies from (`_write_gate`)
        # intake processes (cfg.intake_procs): all their messages go through the control thread
        self.intake = None
        # sockets made by start(); None/empty until then, so stop() after a failed start() closes what exists and
        # the start error is the one raised
        self.ctx = self.reports = self.wake_rx = self.wake_tx = None
        self.plans: list = []
        self.intake_procs: list[subprocess.Popen] = []
        self.intake_cpus: set[int] | None = None
        self.intakes_ready: set[bytes] = set()
        self.slabs = SlabReader()            # the intakes' shared-memory prompt slabs (ssengine/serve/slab.py)
        self.slab_views: set[str] = set()    # requests admitted on a view of a slab, not yet copied out (_own)
        self.answered: dict[str, dict] = {}  # answered, response not yet sent: id -> log record
        self.tier_socks: list = []           # tiers: PUSH to each worker's tier socket
        self.tier_log = None
        self.tier_sent: dict[int, tuple[TierOp, int, dict]] = {}   # op id -> (op, sent ns, stage -> TierDone)
        self.pack, self.unpack = msgspec.msgpack.Encoder(), msgspec.msgpack.Decoder()

    # --- lifecycle ------------------------------------------------------------------------------------------------

    async def start(self):
        tc = self.cfg.tiers
        if tc is not None and tc.ssd_gb and self.cfg.model != "fake":
            # the SSD tier: take a free shard set for the run (kvpool/shards.py); the workers open its shards
            from ssengine.kvpool.shards import ShardSet, shard_bytes
            base = tc.ssd_dir or os.environ.get("SSENGINE_SSD_DIR")
            if not base:
                raise ValueError("SSD tier: set tiers.ssd_dir or SSENGINE_SSD_DIR (the shard sets, dev:research/TIERS.md §5)")
            self.shard_set = ShardSet(base, self.n, shard_bytes(tc.ssd_gb))
            print(f"tiers: SSD shard set {self.shard_set.dir}", flush=True)
            self.cfg = msgspec.structs.replace(self.cfg, tiers=msgspec.structs.replace(tc, ssd_dir=self.shard_set.dir))
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
        # expandable segments: 32k-token microbatches of long contexts peak near the GPU's capacity, and fixed-size
        # segments left 7.2 GB reserved but unusable when stage 2 ran out of memory (even N=400, 7738 mb 4649)
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
               "PYTHONPATH": os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH")]))}
        for s in range(self.n):
            log = open(self.run_dir / f"worker{s}.log", "w")
            self.procs.append(subprocess.Popen(
                [sys.executable, "-m", "ssengine", "worker", "--config", str(cfg_path), "--stage", str(s)],
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
        if cfg.tiers is not None and cfg.model != "fake" and cfg.pool_tokens is None \
                and min(r.num_blocks for r in ready) < CONTRACT_BLOCKS:
            for p in self.procs:                 # the tiers' HBM (staging, IPC-shareable KV) must not shrink the pool
                p.kill()
            raise RuntimeError(f"tiers on: Ready blocks {[r.num_blocks for r in ready]} below the contract's "
                               f"{CONTRACT_BLOCKS}")
        num_blocks = pool_size([r.num_blocks for r in ready], cfg.pool_tokens)
        self.rule = DispatchRule(cfg.max_in_flight, cfg.form_lead_ms, self.n)
        pool = self._pool(ready) if cfg.tiers is not None else None
        # the scheduler's in-flight limits are the rule's (net of `exiting`, 2 ahead)
        self.sched = Scheduler(KvManager(num_blocks, keep_sessions=cfg.keep_sessions, content_keys=True),
                               cfg.scheduler, pool, *self.rule.sched_limits)
        self.mb_log = open(self.run_dir / "microbatches.jsonl", "w")
        self.host_log = open(self.run_dir / "worker_host.jsonl", "w")
        self.plan_log = open(self.run_dir / "plans.jsonl", "w")
        self.req_log = open(self.run_dir / "server_requests.jsonl", "w")
        self.tier_log = open(self.run_dir / "tier_ops.jsonl", "w") if pool is not None else None
        (self.run_dir / "pool.json").write_text(json.dumps(
            {"ready_blocks": [r.num_blocks for r in ready], "num_blocks": num_blocks,
             "pool_tokens": num_blocks * BLOCK_TOKENS, "page_tokens": PAGE_TOKENS, "keep_sessions": cfg.keep_sessions,
             "loading_area_pages": self.sched.kv.area, "loading_area_bytes": self.sched.kv.area_bytes,
             "dram_area_slots": pool.manager.tiers[0].area if pool is not None and pool.manager.tiers else 0}))
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
            p = subprocess.Popen([sys.executable, "-m", "ssengine", "intake", "--config", str(cfg_path), "--index", str(i)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            if self.intake_cpus:
                os.sched_setaffinity(p.pid, self.intake_cpus)
            self.intake_procs.append(p)

    def _pool(self, ready: list[Ready]) -> BalancedKVPool:
        """The pool's leader, in the workers' slot size (all equal: each moves a quarter of the same ranges), and a
        PUSH socket to each worker's tier thread."""
        sizes = {(r.slot_bytes, r.state_slots) for r in ready}
        if len(sizes) != 1 or not ready[0].slot_bytes:
            raise RuntimeError(f"workers report tier slots {sorted(sizes)}")
        tc = msgspec.structs.replace(self.cfg.tiers, slot_bytes=ready[0].slot_bytes, state_slots=ready[0].state_slots)
        for s in range(self.n):
            sock = self.ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.connect(self.cfg.tier_addr(s))
            self.tier_socks.append(sock)
        pool = BalancedKVPool(tc)
        pool.state_bytes = max(r.state_bytes for r in ready) or pool.state_bytes
        m = pool.manager
        info = {"slot_bytes": tc.slot_bytes, "state_slots": tc.state_slots,
                "slots": {x.name: x.num_slots for x in m.tiers}, "staging_slots": m.staging.num_slots,
                "write_ahead": {"hbm": tc.hbm_write_ahead, "dram": tc.dram_write_ahead}, "dram_writes": tc.dram_writes,
                "loading_area": tc.loading_area, "state_bytes": pool.state_bytes}
        (self.run_dir / "tiers.json").write_text(json.dumps(info) + "\n")
        print("tiers: " + json.dumps(info), flush=True)
        return pool

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
        for sock in (*self.plans, *self.tier_socks, self.reports, self.wake_rx, self.wake_tx, self.intake):
            if sock is not None:
                sock.close(linger=0)
        if self.shard_set is not None:
            self.shard_set.close()                     # after the workers exited: the set is free for another run
        for f in ("mb_log", "host_log", "req_log", "plan_log", "tier_log"):
            if getattr(self, f, None) is not None:
                getattr(self, f).close()
        if self.sched is not None:
            self._write_cache()

    def _write_cache(self):
        """cache.json: the pool and the prefix cache's counters (5_prefix_cache.md §8)."""
        kv = self.sched.kv
        info = {"keep_sessions": kv.keep_sessions, "num_blocks": kv.num_blocks,
                "pool_tokens": kv.num_blocks * BLOCK_TOKENS, **kv.stats, "kept_sessions": len(kv.kept),
                "kept_tokens": sum(k.tokens for k in kv.kept.values()), "used_tokens": kv.used_tokens,
                "free_pages": kv.pool.num_free, "free_whole_blocks": kv.pool.num_whole,
                "time_ms": self._ms(time.time_ns()) if self.t0_ns else None,
                "scheduler": dict(self.sched.stats),
                "loading_area": {"pages": kv.area, "held": kv.area_held, "reads": len(kv.reads)}}    # busy_waits: rounds that waited for a tier write (dev:research/TIERS.md §5)
        if self.sched.pool is not None:
            m = self.sched.pool.manager
            info["tiers"] = m.summary()
            info["write_back"] = self.sched.pool.write_back_stats()   # per level: ahead / at eviction / waits (§5)
            info["tier_ops_in_flight"] = len(m.in_flight)
            info["tier_ops_waiting"] = len(m.waiting)
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
        self.last_done = next_watch = time.monotonic()
        next_cache = time.monotonic() + 10.0
        try:
            more = False   # requests left in the inbox after this round's batch
            while True:
                timeout_ms = 0 if more else 1000
                if self.timer_at is not None:
                    timeout_ms = min(timeout_ms, max(0, math.ceil((self.timer_at - time.time_ns()) / 1e6)))
                poller.poll(timeout_ms)
                while self.reports.poll(0):          # reports first: they free in-flight slots and time stage 0
                    msg = decode_to_controller(self.reports.recv())
                    if isinstance(msg, WorkerError):
                        raise RuntimeError(f"worker {msg.stage} failed:\n{msg.error}")
                    if isinstance(msg, StageDone):
                        self._on_stage_done(msg)
                    elif isinstance(msg, TierDone):
                        self._on_tier_done(msg)
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
                with nvtx_range("pump"):
                    self._pump()
                if self.tier_socks:
                    with nvtx_range("load pass"):
                        self.sched.load(time.time_ns())      # staged loading: reads whose area has room now
                        self._tier_send()
                for req_id in list(self.slab_views):
                    self._own(req_id)
                if time.monotonic() >= next_check:
                    next_check = time.monotonic() + 1.0
                    self._check_procs()
                    if time.monotonic() - self.last_done >= WATCHDOG_S and time.monotonic() >= next_watch \
                            and self.sched.has_work():
                        next_watch = time.monotonic() + WATCHDOG_S
                        print(f"watchdog: no completion for {time.monotonic() - self.last_done:.0f} s: "
                              + json.dumps(self.sched.state()), flush=True)
                    for f in (self.mb_log, self.host_log, self.plan_log, self.req_log, self.tier_log):
                        if f is not None:
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
        elif kind == b"S":   # /shutdown at an intake: the serve process stops on SIGTERM (ssengine/serve/server.py)
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
        if self.held is not None:
            self.held.append(req)
        else:
            self.sched.add(req)

    def _pump(self):
        """Form and dispatch microbatches while the dispatch rule allows and the scheduler has work (none before
        `hold_requests` requests have arrived)."""
        if self.held is not None:
            if self.next_seq < self.cfg.hold_requests:
                return
            for req in sorted(self.held, key=lambda r: id_order(r.req_id)):
                self.sched.add(req)
            self.held = None
        if self.gated is not None:
            chunks, formed = self.gated
            if self._write_gate(chunks):
                return
            self.gated = None
            now = time.time_ns()
            st = self.sched.stats
            st["write_gate_ms"] = st.get("write_gate_ms", 0.0) + (now - formed) / 1e6
            self._dispatch(chunks, now)
        while True:
            now = time.time_ns()
            verdict, at = self.rule.decide(now)
            if verdict == WAIT:
                self.timer_at = at
                return
            if verdict != NOW:
                return
            chunks = self.sched.next_microbatch(now, self.rule.exiting)
            if chunks is None:
                return
            if self._write_gate(chunks):
                self.gated = (chunks, now)
                self.sched.stats["write_gates"] = self.sched.stats.get("write_gates", 0) + 1
                return
            self._dispatch(chunks, now)

    def _write_gate(self, chunks: list[ChunkSpec]) -> bool:
        """Whether a write at eviction (exclusive DRAM writes when HBM evicts, as the simulator) still copies from
        pages this microbatch writes: it is sent once those writes are done (`_on_tier_done` pumps)."""
        pool = self.sched.pool
        if pool is None or pool.hbm is None or not pool.hbm.at_eviction:
            return False
        for c in chunks:
            a = c.req.alloc
            end = c.start + c.num_tokens
            if pool.hbm_busy(a.kv_pages[c.start // PAGE_TOKENS:kv_pages_for(end)], a.state_blocks):
                return True
        return False

    def _dispatch(self, specs: list[ChunkSpec], now: int):
        mb_id, self.next_mb = self.next_mb, self.next_mb + 1
        queued = len(self.rule.on_stage0) > 0
        start = self.rule.projected["start"] if self.rule.projected else None
        log = {"index": mb_id, "tokens": sum(c.num_tokens for c in specs), "dispatch_ms": round(self._ms(now), 4),
               "queued": queued, "in_flight": self.sched.in_flight, "exiting": self.rule.exiting,
               "predicted_start_ms": round(self._ms(start), 4) if start else None}
        if self.sched.staged:                            # loading areas in use: HBM blocks, DRAM slots
            d = self.sched.pool.manager.dram_area
            kv = self.sched.kv                           # HBM: kept contexts and their blocks, running+held blocks
            log.update(area_held=kv.area_held, dram_area_used=None if d is None else d.area_used, kept=len(kv.kept),
                       kept_pages=kv.kept_charge, hard=kv.hard, evicted=kv.stats["evicted_sessions"])
        self.rule.dispatched(mb_id, now, log["tokens"])
        chunks = []
        for c in specs:
            r = c.req
            end = c.start + c.num_tokens
            ids = r.token_ids[c.start:end]
            chunks.append(Chunk(r.req_id, c.start, c.num_tokens, r.alloc.kv_pages[:kv_pages_for(end)],
                                r.alloc.state_blocks, c.is_last, ids if isinstance(ids, list) else ids.tolist()))
            info = self.reqs[r.req_id][2]
            if "first_dispatch_ms" not in info:
                info["first_dispatch_ms"] = self._ms(now)
                info["cached"] = r.cached
                if r.admit_info is not None:
                    info.update(r.admit_info)
                info["admit_wait_ms"] = (round((r.admit_ns - r.room_wait_ns) / 1e6, 3)
                                         if r.room_wait_ns is not None and r.admit_ns is not None else 0.0)
                if r.tier is not None:
                    info.update(tier=r.tier, tier_tokens=r.tier_tokens, hbm_tokens=r.hbm_tokens,
                                read_start_ms=round(self._ms(r.read_start_ns), 4),
                                read_land_ms=round(self._ms(r.read_land_ns), 4))
                    if r.fetch_land_ns is not None:          # staged SSD read: landed in DRAM's area, hop to HBM
                        info.update(fetch_land_ms=round(self._ms(r.fetch_land_ns), 4),
                                    hop_start_ms=round(self._ms(r.hop_start_ns), 4))
            info["chunks"] = info.get("chunks", 0) + 1
        plan = Plan(mb_id, chunks)
        for s in range(self.n):
            self.plans[s].send(encode(for_stage(plan, s)))
        log["sent_ms"] = round(self._ms(time.time_ns()), 4)
        self.plan_log.write(json.dumps(log) + "\n")
        self.inflight_plans[mb_id] = ([{"id": c.req.req_id, "prefix": c.start, "append": c.num_tokens}
                                       for c in specs], self.n)

    def _on_stage_done(self, d: StageDone):
        self.last_done = time.monotonic()
        reqs, left = self.inflight_plans[d.mb_id]
        self.mb_log.write(json.dumps({"stage": d.stage, "index": d.mb_id, "start_ms": round(self._ms(d.start_ns), 4),
                                      "compute_ms": round((d.end_ns - d.start_ns) / 1e6, 4),
                                      "requests": reqs}) + "\n")
        if d.host_ns is not None:
            plan, call, prep, ready, ret = (round(self._ms(t), 4) for t in d.host_ns)
            self.host_log.write(json.dumps({"stage": d.stage, "index": d.mb_id, "plan_ms": plan, "forward_ms": call,
                                            "prepared_ms": prep, "ready_ms": ready, "return_ms": ret}) + "\n")
        if left == 1:
            del self.inflight_plans[d.mb_id]
        else:
            self.inflight_plans[d.mb_id] = (reqs, left - 1)
        self.sched.stage_done(d.stage, self.n - 1)
        self.rule.stage_done(d.stage, d.mb_id, d.start_ns, d.end_ns, d.host_ns)
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

    def _tier_send(self):
        """Send the tier ops whose dependencies are done (reads, writes after `finish`, dependents of landed ops)."""
        ops = self.sched.pool.ready_ops()
        if not ops:
            return
        msg = encode(TierBatch([tier_move(o) for o in ops]))
        for sock in self.tier_socks:
            sock.send(msg)
        now = time.time_ns()
        for o in ops:
            self.tier_sent[o.op_id] = (o, now, {})

    def _on_tier_done(self, d: TierDone):
        self.last_done = time.monotonic()
        o, sent, done = self.tier_sent[d.op_id]
        done[d.stage] = d
        if len(done) < self.n:
            return
        del self.tier_sent[d.op_id]
        now = time.time_ns()
        self.sched.op_done(d.op_id, now)
        if self.gated is not None:
            self._pump()
        t0 = self.t0_ns or sent
        self.tier_log.write(json.dumps({
            "op": d.op_id, "kind": o.op, "tier": o.tier, "host_tier": o.host_tier, "entry": o.entry,
            "req": o.req_id, "tokens": [o.token_start, o.token_end], "slots": len(o.slots) + len(o.state_slots),
            "deps": o.deps, "sent_ms": round((sent - t0) / 1e6, 4), "landed_ms": round((now - t0) / 1e6, 4),
            "expected_land_ms": round((o.land_ns - t0) / 1e6, 4) if o.land_ns else None,
            "gpu_ms": [[round((x.start_ns - t0) / 1e6, 4), round((x.end_ns - t0) / 1e6, 4)]
                       for x in (done[s] for s in range(self.n))],
            "submit_ms": [round((done[s].submit_ns - done[s].start_ns) / 1e6, 4) for s in range(self.n)],
            "gpu_span_ms": [None if done[s].gpu_ns is None else round(done[s].gpu_ns / 1e6, 4) for s in range(self.n)],
            "mark_ms": [None if done[s].mark_ns is None else round((done[s].mark_ns - t0) / 1e6, 4)
                        for s in range(self.n)],
            "recv_ms": [None if done[s].recv_ns is None else round((done[s].recv_ns - t0) / 1e6, 4)
                        for s in range(self.n)],
            "bytes": [done[s].nbytes for s in range(self.n)]}) + "\n")

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
