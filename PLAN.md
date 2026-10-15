# pp-impl plan

A complete serving framework for GLM-5.3-Flash NVFP4 as a 4-stage pipeline on 4×B200: the design of handoff
`glm53_flash_pp4_b200_v1` (ServingStudioSim `pp-main`, frozen `7bb224a6`). It is built as one framework, evaluated
against the handoff's milestones in their order.

## 1. Reuse boundary

| Reused as is | Owned by pp-impl |
|---|---|
| vLLM fork `892da0822f`: `Glm5Next*` model and layer modules, weight loading, kernels (FlashInfer, DeepGEMM, DeepSelect, TileLang, causal-conv1d) and vLLM's attention-metadata builders (MLA/indexer, GDN for KDA) | process model and pipeline transport, KV page pool and allocator, exact prefix cache, scheduler, DRAM/SSD tiers, server |
| req-frontend (client), handoff `data/metrics.py`, `data/microbatch_fit.py`, handoff traces and targets | milestone runners and comparison tables |
| ServingStudioSim profiler (kernel rows, node check) | — |
| vLLM fork TP4+EP as the correctness reference | — |

The glm5next layers read vLLM's forward context, attention metadata and workspace manager. We keep that: each stage
process builds a `VllmConfig` for its PP rank, constructs the model with vLLM's PP layer skipping, allocates KV in the
tensor views vLLM's layers expect, builds metadata with vLLM's builder classes from **our** block tables and state
slots, and runs the forward inside `set_forward_context`. No vLLM engine, scheduler, model runner or KV manager runs.

Two gaps in the fork that we close with small overrides in `ppimpl/model/pp_patch.py`:
- PP is gated off for glm5next: the fork drops the mHC deferred state (`post`, `comb`) at the stage edge and has no
  `make_empty_intermediate_tensors`; weight loading also assumes every DSA layer is local. The sending stage applies
  its last layer's deferred `hc_post` and sends the full 4-stream residual `[T, 4 × 4096]` BF16 (32 KiB per token,
  the handoff's send); the receiving stage's first layer runs a standalone `hc_pre`. (A patch of this shape already
  runs vLLM TP4×PP2 on this node.) The targets price each stage's first boundary as the fused post+pre, which a
  32 KiB send cannot feed; ours costs one standalone `hc_post` on the sender and a standalone `hc_pre` instead of the
  fused kernel on the receiver: +0.03 ms (8k) to +0.32 ms (128k) per edge, < 0.06% of a stage. Recorded in issues.md.
- Prefill only, one output token: no decode path, no MTP, greedy sampling on stage 3.

**Kernels stay where they are.** No kernel is copied into pp-impl. `ppimpl/kernels.py` is the manifest of the
35 configurations of `2_kernels.md` §2: per (kind, backend, fixed configuration) the callable's import path, the
library version, and how the profiler row calls it. It serves three uses: the M1 identity check against the profiler
rows, one entry point for `bench/` and the correctness tests, and a startup check that a stage's traced GPU kernels
are exactly the manifest's (no silent fallback to another backend). Indexer top-k picks DeepSelect or vLLM
`topKPerRowPrefill` per shape, whichever row is faster (`2_kernels.md` §3), from a table in `kernels.py`. A kernel we
must change goes to `ppimpl/kernels_patched/` (empty until needed).

## 2. Process model

```
                HTTP (req-frontend, vllm-tokens: POST /inference/v1/generate)
                                   │
                    ┌──────────────▼──────────────┐
                    │ controller (stage-0 host)    │  scheduler · KV allocator · prefix cache · tier index
                    │ forms a microbatch when      │  ≤4 microbatches in flight
                    │ stage 0 is free              │
                    └──┬────────┬────────┬────────┬┘  plans (zmq)        ▲ tokens, stage timings, read landings
                 GPU0  ▼  GPU1  ▼  GPU2  ▼  GPU3  ▼                       │
               stage 0 ─NCCL─▶ stage 1 ─NCCL─▶ stage 2 ─NCCL─▶ stage 3 ───┘
               layers 0–11    12–22          23–33          34–44 + norm + lm_head
```

- The controller sends each microbatch plan to all four workers at once. Plans carry the chunks
  `(request, first token, tokens, cached prefix, state slot, block ids)`, which requests finish in this microbatch,
  and any tier copy orders.
- The workers pass activations stage to stage by NCCL point-to-point: the 4-stream residual plus mHC state, 32 KiB per
  token. Stage 3 returns greedy tokens.
- Stage compute runs back to back on each GPU with no host gaps. Planning and metadata building for microbatch k+1
  overlap with compute of microbatch k.
- Server logs `microbatches.jsonl` and `reads.jsonl` in the format of `1_contract.md` §3.

## 3. File structure

```
pp-impl/
  README.md              provenance, how to run
  PLAN.md                this file
  pyproject.toml         package `ppimpl`, `python -m ppimpl serve`
  ppimpl/
    config.py            design constants and run config: split [12,11,11,11], pool tokens, budgets, tiers
    kernels.py           kernel manifest (above); top-k per-shape choice
    kernels_patched/     only kernels we modify (empty)
    __main__.py          serve: spawn controller + 4 workers
    server.py            /inference/v1/generate (token ids in/out, usage.cached_tokens), /health
    model/
      stage.py           build one stage via vLLM (VllmConfig, PP rank), load only its layers' weights
      pp_patch.py        mHC post/comb across the stage edge; make_empty_intermediate_tensors
      kv_tensors.py      one page pool per stage: DSA FP8 KV pages and KDA state slots as vLLM tensor views
      metadata.py        our block tables / state slots → vLLM MLA, indexer, GDN metadata + forward context
    runtime/
      transport.py       NCCL p2p for activations; zmq control channel
      worker.py          stage loop: plan → (recv) → forward → (send) → report; executes tier copies
      controller.py      drives the scheduler, ≤4 in flight, dispatches plans, completes requests
    engine/
      kv.py              block pool accounting (8,576-token blocks, 6 state blocks per request), admission,
                         exact prefix cache, LRU eviction of whole sessions
      scheduler.py       plain scheduler, then steps 1–4: shortest first, load budget, even split, force-schedule
      tiers.py           DRAM/SSD index, per-tier read channel, read-wait bound, copy-up, SSD written on DRAM eviction
  bench/
    run.py               one entry per handoff workload: server + req-frontend in one Slurm job → measured/runs/<name>/
    configs/             req-frontend yamls for every handoff run
    layers.py            M2: one layer of each type through ppimpl.model, GPU time by nsys
    report.py            measured vs targets → measured/*.md
  tests/                 pytest. cpu: scheduler (vs sched_check_targets.json with a fake executor), kv, tiers,
                         plan/transport protocol, server API with a fake pipeline. gpu (marker): kernel
                         correctness vs FP32 (2_kernels.md §5), one stage vs the unsplit model, 4-stage
                         pipeline token vs reference, kernels.py manifest vs traced kernels
  measured/              handoff "What to record" outputs; issues.md
  third_party/vllm       (ignored) fork checkout at 892da0822f, build outputs copied from a same-HEAD tree
  .venv                  (ignored) shared env
```

## 4. Build phases and evaluation order

The framework is built in four phases. Each phase extends the same code. Its milestones are evaluated in the
handoff's order, and the next phase starts only when they pass or every miss is in `measured/issues.md` with a cause.

| Phase | Builds | Evaluates (handoff milestone) |
|---|---|---|
| A. Pipeline | env; `model/*`; `runtime/*`; `server.py`; `engine/kv.py` without reuse; plain scheduler | M1 kernels (`2_kernels.md`): call-path identity with the profiler rows, FP32 correctness, node check · M2 layers (`3_layers.md`) · M3 iteration (`4_iteration.md`): one request, saturated, output vs vLLM reference |
| B. Prefix cache | exact hits with the KDA state at the prompt end; kept sessions; LRU whole-session eviction; cached-token reporting | M4 (`5_prefix_cache.md`): prefix bring-up, sessions N=400/500 HBM only, microbatch fit |
| C. Scheduler | steps 1–4 (`6_scheduler.md` §2) | M5: scheduling check chunk for chunk, force check, sessions per step |
| D. Offloading | DRAM then SSD tiers, own-stage storage first (balanced storage gained nothing on the targets), SSD written on DRAM eviction | M6 (`7_offloading.md`): tier bring-up, sessions at full / Tiers / SSD-heavy memory = the final targets |

M1 does not re-time the 5,392 rows: they were measured on these B200s by the ServingStudioSim profiler through the
same callables. It checks that our engine calls the same callables at the same library versions, and re-measures,
with the profiler, only configurations where the callable or version differs.

## 5. Who does what

- **Orchestrator** (main session): this plan, interfaces, briefs, review of every diff, every benchmark run and its
  reading, `progress.md`. Writes no framework code.
- **Phase A, two implementers in parallel**, each in its own worktree under `lanes/`, after the interfaces in
  §2 are written down as `ppimpl/runtime/README.md`:
  - *model*: env, `model/*`, `bench/layers.py`;
  - *runtime*: `runtime/*`, `server.py`, `engine/kv.py` (no reuse yet), plain `engine/scheduler.py`,
    `bench/run.py`. It develops against a fake stage model with the same interface, then integrates.
- **measure**: `measured/setup.md`, kernel node check, call-path identity table; at M3 the vLLM TP4+EP reference.
- Phases B–D: one implementer each, on the integrated code.

## 6. Done means

The final targets of the README at the frozen design, measured by req-frontend on 4 B200: 20-minute sessions, N=500,
118.8k new tokens/s, TTFT p50 / p90 / p99 0.16 / 0.44 / 4.94 s at full memory, and the two cache-tier settings. Every
miss is in `measured/issues.md` with its cause.
