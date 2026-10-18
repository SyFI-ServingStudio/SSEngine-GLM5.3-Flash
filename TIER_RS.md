# Worker tier execution in Rust (`TierConfig.engine = "rust"`)

Why: each stage worker's Python tier threads (`kvpool/member.py`, `mover.py`, `uring.py`) make thousands of ctypes
driver calls per second; every one releases and retakes the GIL, so stage 0's launch thread waited for the GIL
25-40% of its microbatch prepare under tiers (~13 of ~36 ms; HBM-only ~0). Here every per-op step of the worker's
side of the pool runs on Rust OS threads that never take the GIL. The Python worker keeps model prepare and launch;
the controller (`KvManager`, `BalancedKVPool`, `TierManager`, `WriteBack`, SlotGuard deps, write gates) is unchanged
and sees the same `TierMove` in and `TierDone` / `WorkerError` out on the same zmq sockets.

`engine = "python"` (default) is the existing code, untouched; `"rust"` is selected per run (`--tier engine=rust`).

## What stays in Python (start-up and close only)

`PoolMember.__init__` / `attach` as before: the DRAM arena (mmap, NUMA bind-with-check touch, `cudaHostRegister`),
the staging arena, `share_ranges` (CUDA IPC of every stage's KV), `BalancedPlacement`, the SSD shard file
(`open_direct`), the device ring (`torch.empty` after the KV pool) and the mover's four streams (torch). It then
builds one `tier_rs.Engine` and hands it raw values only:

| Handed over | From |
|---|---|
| stage, n, device ordinal, real / fake, slot bytes | config, `attach` |
| placement arrays (`kv_stage/dev/stride/len/host`, `st_stage/dev/stride/group/slot/off/len`) | `BalancedPlacement` (device addresses already valid in this process) |
| DRAM / staging arena base addresses | `HostTier.base` (slot i at base + i x slot bytes) |
| SSD fd, the host spans to register | `SsdTier` (file opened, no Python ring), the arenas |
| ring pointer, depth, chunk slots, hops | `torch.empty`, `TierConfig` |
| stream handles {read, write, read_nvl, write_nvl} | `torch.cuda.Stream.cuda_stream` |
| tier address (PULL bind), report address (PUSH connect), trace every / run dir, fake bandwidths | config, env |

Python keeps ownership of everything it allocated (arenas, ring tensor, streams, IPC handles, fd) and frees it in
`close()` after `Engine.close()` has joined every Rust thread. Rust owns its events (cuEventCreate), its anchor
streams, its io_uring and its zmq context. The GIL probe thread (`PPIMPL_TIER_TRACE`) stays in Python: it measures the
GIL, which is the point.

## Threads (one-to-one with member.py)

| Thread | Python | Rust |
|---|---|---|
| recv | zmq PULL, `decode_tier`, reads first | same (zmq crate, msgpack via rmpv), `recv_ns` carried in the op |
| submit-read / submit-write | `submit` -> `_move` -> `Mover.transfer_async` | same rules (below) |
| done-read / done-write | event sync, `cuEventElapsedTime`, anchor, `TierDone` PUSH | same; own PUSH socket each |
| trace | trace lines, pooled events back | same (serde_json lines, same fields) |
| ssd-read / ssd-write | `_ssd_op` | same |
| ssd-copy | `_ssd_read_copy`, landing order | same |
| uring | `UringFile._loop` (liburing via ctypes) | io-uring crate, same queues and depths |

Queues are crossbeam channels (lock-free MPMC); the end-of-run chain is the same as Python's (recv -> submit -> ssd
-> copy -> done -> trace), errors travel down it as `WorkerError(stage, "tier ... thread:\n...")`. Every Rust thread
makes the device's primary context current once (cuDevicePrimaryCtxRetain + cuCtxSetCurrent: the same context torch
uses, so torch's stream handles and the IPC pointers are valid).

Calls from Python into Rust (`Engine(...)`, `start`, `close`, test hooks) release the GIL (`py.allow_threads`). No
completion goes back to Python in a run: the done threads push `TierDone` to the controller themselves. Tests (no
controller) start the engine with `report=None`: completions then go to a channel that `Engine.poll_done(timeout)`
reads (GIL released while waiting), and `Engine.dispatch(batch_bytes)` hands a `TierBatch` to the same code the recv
thread runs.

## Rules carried over (checked by tests against the Python layer)

- **Placement** (`BalancedPlacement.pieces`, `local`): one block list for all stages or one per stage; KV pieces per
  block in range order, then the state pieces; own-stage pieces flagged local.
- **Mover, two hops** (5b9cec0): the op's slots of all segments cut into ring chunks of `ring_chunk_slots` together;
  per chunk, ring position = the direction's next (shared by every transfer of that direction); read: PCIe batch
  (peer runs host -> ring merged per slot by `_runs`, then own pieces host -> KV) on `read` after the chunk's
  second-hop event, recording its first-hop event; NVLink batch (ring -> peers) on `read_nvl` after the first-hop
  event, recording the second-hop event. Write: gather (peers -> ring) on `write_nvl`, then (ring runs, own pieces)
  -> host on `write`. An empty batch only waits and records. Direct: one batch of all pieces per transfer.
- **Direction lock**: a transfer without a gate holds its direction from the start mark to its last chunk (reads
  complete in submission order); a gated write takes it per chunk and, before each chunk, waits on the host for its
  previous chunk's first-hop event, then yields (4fa0a13).
- **Reads first**: DRAM reads are counted in at arrival (recv thread), SSD reads when their last slot landed; counted
  out by the read done thread; a non-urgent write's chunk waits for the reads counted in when it is due. Urgent
  writes (an HBM eviction waits for them: `WriteBack.make_room`, 1586fc8's evict-time writes) never wait.
- **Events**: a pool of 1,024 timing events (blocking sync) per mover, more made when empty and counted; per ring
  chunk a pair of blocking, non-timing events; a start mark and an end event per op; with the trace, two per batch;
  returned to the pool by the trace thread after the op's TierDone.
- **Timing**: `gpu_ns` = cuEventElapsedTime(mark, end); `mark_ns` through the done thread's anchor (an event on its
  own idle stream, synchronized, re-taken every 1 s); NOT_READY -> synchronize both, read again. Host clocks:
  `time_ns` = CLOCK_REALTIME, perf counters = CLOCK_MONOTONIC, CPU time = CLOCK_THREAD_CPUTIME_ID.
- **SSD**: one request per slot, cut at a 1 GiB fixed buffer's end; READ_FIXED / WRITE_FIXED on the registered file;
  8 reads / 4 writes in the kernel, read room filled first; eventfd read always armed. A read is copied to HBM in
  landing order (SSD-read copies use block_start 0, report the file bytes); fetch reported when its last slot is in
  DRAM; write = HBM -> host slots (gated unless urgent), host sync, then the file writes; demote = file writes only;
  writes and demotes reported when their last file write completed, without holding the SSD thread; the first failed
  I/O reports a WorkerError; the SSD thread's end waits (up to 60 s) for the I/O in flight.
- **Fake model** (no KV): each op done after (blocks + state slots) x slot bytes / configured bandwidth, one after
  another per direction.
- **Controller-side rules** (SlotGuard ("hbm", block) deps, write-at-evict ordering, urgency, placement of ops) are
  not in the worker in either engine: the worker receives ops whose deps are done.

Not carried over: the nvtx ranges around a DRAM op's submit (instrumentation only; nsys still shows the CUDA calls
per thread). Bytes are identical by construction (same pieces, same copies); tests compare the batch lists the two
movers issue for the same transfer, and bytes end to end.

## Build

`ppimpl/native/tier_rs` (Cargo + maturin, PyO3, module `tier_rs`): `just`-less, from the lane root:
`.venv/bin/maturin develop --release -m ppimpl/native/tier_rs/Cargo.toml` with `VIRTUAL_ENV=<pp-impl>/.venv`
(see `ppimpl/native/tier_rs/README` comment in Cargo.toml). libcuda is loaded at run time (dlopen), so the module
imports on CPU-only hosts; zmq is the vendored libzmq (no system libzmq here).

## What the Python engine puts on the GIL today (stage 0, minutes 10-20)

Rates from job 8356 (combo tree, `tier_ops.jsonl`: 32 DRAM reads, 33 DRAM writes, 20 SSD fetches, 21 demotes per s
per worker) with the call counts of the 78d94af mover per op; the nsys count of 8349 (27 s window, before the GIL
fixes) in brackets. Every line runs Python bytecode on a tier thread, so it needs the GIL whether or not the C call
inside releases it; "releases" marks the calls that also hand the GIL over and take it back.

| Tier-thread work | Thread | Per s | GIL |
|---|---|---|---|
| op submit: `placement.pieces`, `_transfer` numpy (segments, chunks, runs), per DRAM op 1.96 ms p50 | submit r/w | 65 ops, **180 ms of host time per s** | held throughout, except inside the batch calls |
| `cuMemcpyBatchAsync` (ctypes CFUNCTYPE) | submit, SSD, copy | ~800 [480] | releases |
| `cuEventRecord` (PyDLL since 14bf48b) | submit, done | ~1,100 [1,600] | held (was: releases) |
| `cuStreamWaitEvent` (PyDLL since 14bf48b) | submit | ~800 [480] | held (was: releases) |
| `cuEventElapsedTime` (driver, since ab0099a; torch `elapsed_time` before: 2 queries each) | done, trace | ~330 [2,000 queries] | held |
| event `synchronize` (torch) | done, SSD write, gated write | ~65 + one per gated chunk | releases (blocks) |
| `queue.Queue` put / get between recv, submit, SSD, copy, done, trace | all | ~400 | each get wakes a thread: GIL taken |
| zmq PULL poll (200 ms) / recv + msgspec decode | recv | 5 + ~50 | releases in poll / recv |
| zmq PUSH `TierDone` + msgspec encode | done | ~110 | releases in send |
| io_uring: per slot request (`Future`, piece list, `eventfd_write`), `submit_and_wait`, per completion `set_result` and the op's callback | SSD r/w, uring | ~600 slot requests | wait releases; the rest held |
| per-batch trace (1 op in 8): 2 extra records per batch, json line | submit, trace | ~50 records, ~8 lines | held |
| probe (trace on): wake every 5 ms | probe | 200 | taken each wake |

Measured effect: stage 0's main thread waited for the GIL 10.8 / 6.5 / 2.6 ms of 27.4 / 20.3 / 10.9 ms per prepare
(8349 windows), 13.4 of 35.7 ms in 8350 (HBM-only 0.08 of 8.0), in waits of 20-30 us p50: ping-pong with the threads
above, not the 5 ms switch interval. The PyDLL / fewer-records fixes (8358) remove the hand-overs of the record and
wait calls but not the ~180 ms/s of submit-thread bytecode and numpy nor the per-op queue and io_uring wake-ups.

With `engine = "rust"` none of these rows runs Python: the only tier-related Python left in the worker process is the
probe thread (with the trace on) and `Engine.close()` at exit.
