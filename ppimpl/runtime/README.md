# runtime: interfaces of phase A

The contract between the controller, the four stage workers and the stage model. The model lane (`ppimpl/model/`)
and the runtime lane (`ppimpl/runtime/`, `ppimpl/engine/`, `ppimpl/serve/`) build against this file in parallel. A
change to it goes through the orchestrator.

## Processes

- `python -m ppimpl serve --config <yaml>` starts one **controller** process (scheduler loop + KV allocator) and four
  **worker** processes, worker s on GPU s running stage s. HTTP is served by `intake_procs` **intake** processes on
  the same port (SO_REUSEPORT; default 4) that parse requests and pass them to the controller's control thread over
  zmq ROUTER/DEALER, answers coming back the same way; with `intake_procs: 0` the controller serves HTTP itself.
- Control: controller → each worker one zmq PUSH socket (plans, in order); workers → controller one zmq PULL socket
  (reports). Messages are msgspec structs (below), msgpack-encoded.
- Activations: worker s → worker s+1 by copy engine into a K-slot ring on the receiver (`CePipe`: CUDA IPC, ordered
  by cuStreamWaitValue32 / cuStreamWriteValue32 flags, no kernels; on CPU, gloo send/recv, `GlooPipe`). Start-up
  barriers and the IPC handle exchange use `torch.distributed` (gloo; world size 4, rank = stage). Every worker
  processes plans in `mb_id` order, so sends and receives match without tags.

## Block-id space

One block-id space for all four stages, owned by the controller (`engine/kv.py`). Block ids are the same on every
stage; each stage backs them with its own tensors.

- A block is 8,576 tokens of DSA KV, or one KDA layer's state (`1_contract.md` §1). Block 0 is the null block.
- `num_blocks` = the smallest of the four stages' reports (`Ready.num_blocks`), minus the null block. The pool is
  7,143 blocks (61,258,368 tokens) at the contract's HBM; `config.py` may cap it by token count.
- A request holds `ceil(prompt_tokens / 8576)` **KV blocks** (DSA, its token positions in order) and 6 **state
  blocks** (KDA): state block g backs KDA cache group g on every stage; a stage with fewer KDA groups ignores the
  extra ones. The last state block is scratch.
- Blocks are freed when the request finishes. With `keep_sessions` (phase B, `5_prefix_cache.md`) they stay as the
  request's kept context instead: the KV of its prompt and the KDA state after its last token. The next prompt that
  starts with that whole context takes over its blocks (same ids, so the state continues in place) and its first chunk
  has `start` = the cached tokens. Kept contexts are matched by content (`engine/kv.py`) and evicted least recently
  used first, a whole context at a time, when a running request needs room.

## Messages

```python
class Chunk(msgspec.Struct):
    req_id: str
    start: int                 # tokens of this request already in KV: cached (a prefix hit) + computed by earlier chunks
    num_tokens: int            # tokens computed in this chunk
    kv_blocks: list[int]       # DSA blocks covering [0, start + num_tokens)
    state_blocks: list[int]    # 6 KDA state blocks
    is_last: bool              # last chunk of the prompt: stage 3 returns this request's token
    token_ids: list[int] | None  # the chunk's prompt tokens; sent to stage 0 only, None elsewhere

class Plan(msgspec.Struct):
    mb_id: int                 # 0, 1, 2, ... in dispatch order
    chunks: list[Chunk]        # concatenated in this order: hidden rows [sum of earlier num_tokens, ...)

class Shutdown(msgspec.Struct): ...

# worker -> controller
class Ready(msgspec.Struct):
    stage: int
    num_blocks: int            # blocks this stage can back with its KV tensors

class StageDone(msgspec.Struct):
    stage: int
    mb_id: int
    start_ns: int              # GPU start of the stage's first kernel for this mb (CUDA event, host-clock aligned)
    end_ns: int                # GPU end of its last kernel, before the send
    tokens: list[tuple[str, int]] | None   # stage 3 only: (req_id, greedy token) per is_last chunk
    host_ns: tuple[int, int, int] | None   # host clock: plan received, prepare called, run returned
```

Clock: each worker aligns its CUDA events to `time.time_ns()` once at startup (record an event, synchronize, read the
host clock) so `start_ns`/`end_ns` of all stages share the host clock the controller uses for `t0`.

## Stage model (`ppimpl/model/stage.py`)

```python
class StageModel:
    def __init__(self, cfg: RunConfig, stage: int, device: torch.device): ...
        # builds the stage's layers (vLLM Glm5Next, PP rank = stage), loads only their weights,
        # allocates KV tensors, captures CUDA graphs if used
    num_blocks: int                       # for Ready
    hidden_width: int                     # 4 * 4096: elements per token sent between stages (BF16)
    def prepare(self, plan: Plan): ...    # host side: metadata, KDA plan, one H2D copy (no wait on queued compute)
    def run(self, batch, hidden_in: torch.Tensor | None) -> torch.Tensor:   # forward = run(prepare(plan), hidden_in)
        # stage 0: hidden_in is None, embeds plan's token_ids.
        # stages 0-2: returns [T, hidden_width] BF16 on device, T = sum(num_tokens).
        # stage 3: returns int64 [number of is_last chunks]: greedy tokens, in chunk order.
        # Enqueues GPU work only; no host sync inside.
```

- Stages 1–2 may return `hidden_in` itself (updated in place).
- The last stage's output is copied to the host by the worker's reporter thread after the end event; `forward` must
  not overwrite that tensor in a later call before the copy is done (allocate a fresh output per call).
- `Ready.num_blocks` counts the null block.
- CUDA graphs (`RunConfig.cuda_graphs`, default on): after warm-up the stage captures one graph per power-of-2
  bucket from 128 to 8,192 tokens, on a side stream of its own; `forward` pads a microbatch to the smallest bucket
  that holds it and replays the graph on the caller's current stream, and runs eagerly above 8,192. Stages 0–2 return
  a copy of the graph's output rows (the next replay overwrites the graph's buffer). The worker's info line reports
  `cuda_graphs` (buckets, graph memory, capture time). The KV pool is sized before capture; the graphs fit in the
  13.6 GB reserve (`measured/issues.md`, option (B)).
- Extra message `WorkerError(stage, error)`: a failing stage sends its traceback; the controller stops the server.
- Messages are msgspec structs in `ppimpl/runtime/transport.py`.

`ppimpl/model/fake.py` provides `FakeStageModel` with the same interface and no weights: a configurable per-token
GPU (or CPU) delay, deterministic outputs (token = hash of req_id and length). The runtime is developed and tested
against it.

## Worker loop

For each plan: `prepare` (while the GPU still runs earlier microbatches) → (stage > 0) receive `[T, hidden_width]`
from stage s-1 / (stage 0) the gate → record start event → `run` → record end event → (stage < 3) send to stage s+1 on a separate stream so the next plan's compute is not blocked by the send →
report `StageDone` once the end event has completed (a reporter thread, not a sync on the compute stream).

Limit on how far ahead a stage host can launch: CUDA holds at most ~1,024 launched-but-unfinished kernels per
context (measured, job 7091: every blocking launch had 1,025–1,030 kernels pending). After that the host blocks
inside the next launch until the GPU retires one (back-pressure, not a sync). A stage-0 forward is ~390 kernels at
≤ 8k tokens (1,245 at 128k), so a stage host runs at most ~2.6 microbatches ahead of its GPU. The thread that
launches should not also be the one that services messages or sends, or it stalls with the launch.

## Controller

- Scheduler (`engine/scheduler.py`; its docstring has the rules): what goes into the next microbatch. The steps of
  `6_scheduler.md` (`ppimpl/config.py` `SCHEDULER_STEPS`), each adding to the one before: plain (arrival order,
  8,192-token budget), srpt8k (SRPT order), budget (up to 32,768 tokens, scaled by the backlog), even (at most the
  backlog / 4), good (force-schedule after 60 s). A run config may set the budget larger (one-request runs: the
  whole prompt in one microbatch).
- Dispatch (`runtime/dispatch.py`; its docstring has the rule and why): when the next microbatch forms. At most 4 in
  flight, held on the GPU: stage 0's stream waits before microbatch k until the last stage has finished k-4 (a flag
  it writes, `transport.py`). The controller forms each plan ahead, `form_lead_ms` plus the predicted host time
  before its projected stage-0 start (stage ends reported, or predicted per stage), so neither the reports' latency
  nor the plan's delivery leaves stage 0 idle; the scheduler leaves the microbatches projected to exit by then out
  of the in-flight count and the backlog.
- Admission: a request enters the scheduler once its blocks are allocated; otherwise it waits in arrival order.
- A request completes when stage 3 reports its token: the HTTP response returns it, its blocks are freed.

## Server API

What req-frontend's backend sends (`alignment/load_generator/req-frontend`): token ids in, one output token id out,
`usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens` (the prefix-cache hit; 0 without
`keep_sessions`). Plus `GET /health`.

## Server logs (`1_contract.md` §3)

`<run dir>/microbatches.jsonl`, one line per (microbatch, stage), times in ms from `t0` = first request's arrival at
the server: `{"stage", "index", "start_ms", "compute_ms", "requests": [{"id", "prefix", "append"}]}`.
