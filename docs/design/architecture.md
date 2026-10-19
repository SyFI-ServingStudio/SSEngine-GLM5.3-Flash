# Architecture

SSEngine serves GLM-5.3-Flash NVFP4 as a 4-stage pipeline on 4 B200 GPUs, with an exact prefix cache in HBM and
optional DRAM and SSD tiers behind it. It computes prefill and one output token: there is no decode path. This page
describes how the engine is built; [tiers.md](tiers.md) covers the KV tiers.

## Reuse boundary

| Reused as a library | Owned by SSEngine |
|---|---|
| The vLLM fork `892da0822f` ([serendipity-zk/vllm](https://github.com/serendipity-zk/vllm), branch `servingstudio-alignment`): the `Glm5Next*` model and layer modules, weight loading, the kernels (FlashInfer, DeepGEMM, DeepSelect, TileLang, causal-conv1d) and vLLM's attention-metadata builders (MLA and indexer, GDN for KDA) | process model and pipeline transport, KV page pool and allocator, exact prefix cache, scheduler, DRAM and SSD tiers, HTTP server |
| req-frontend (the benchmark client) and the ServingStudioSim handoff's traces, targets and metric code | the benchmark runners ([benchmarks/](../../benchmarks/)) |

The model's layers read vLLM's forward context, attention metadata and workspace manager, so each stage process:

1. builds a `VllmConfig` for its pipeline rank and constructs the model with vLLM's PP layer skipping;
2. allocates the KV cache in the tensor views vLLM's layers expect ([ssengine/model/kv_tensors.py](../../ssengine/model/kv_tensors.py));
3. builds attention metadata with vLLM's builder classes from **SSEngine's** block tables and state slots
   ([ssengine/model/metadata.py](../../ssengine/model/metadata.py));
4. runs the forward inside `set_forward_context`.

No vLLM engine, scheduler, model runner or KV manager runs. No kernel is copied:
[ssengine/kernels.py](../../ssengine/kernels.py) is a manifest of the kernel configurations the design uses (import path,
library version, how the profiler row calls it), checked against the kernels a stage actually launches.

The fork does not support pipeline parallelism for this model. [ssengine/model/pp_patch.py](../../ssengine/model/pp_patch.py)
closes the gap with small run-time overrides, each listed with its reason and the test that checks it. The main one:
the sending stage applies its last layer's deferred mHC `hc_post` and sends the full 4-stream residual
`[T, 4 × 4096]` in BF16 (32 KiB per token); the receiving stage's first layer runs a standalone `hc_pre`.

## Process model

```
                HTTP clients (OpenAI API, req-frontend tokens API)
                                   │
                         16 intake processes (ssengine/serve/)
                                   │ zmq
                    ┌──────────────▼──────────────┐
                    │ controller                   │  scheduler · KV allocator · prefix cache · tier index
                    │ forms a microbatch when      │  bounded microbatches in flight
                    │ stage 0 is about to be free  │
                    └──┬────────┬────────┬────────┬┘  plans (zmq)        ▲ tokens, stage timings, tier reports
                 GPU0  ▼  GPU1  ▼  GPU2  ▼  GPU3  ▼                       │
               stage 0 ──CE──▶ stage 1 ──CE──▶ stage 2 ──CE──▶ stage 3 ───┘
               layers 0–11    12–22          23–33          34–44 + norm + lm_head
```

- **Controller** ([ssengine/runtime/controller.py](../../ssengine/runtime/controller.py)). Drives the scheduler, sends each
  microbatch plan to all four workers at once, and completes requests. A plan carries the chunks (request, first
  token, tokens, cached prefix, state slot, block ids), the requests that finish in the microbatch, and any tier copy
  orders. [dispatch.py](../../ssengine/runtime/dispatch.py) decides when to form the next microbatch from the predicted
  end of stage 0.
- **Stage workers** ([ssengine/runtime/worker.py](../../ssengine/runtime/worker.py)). Loop plan → receive → forward →
  send → report. Activations go from stage to stage by a copy-engine copy into the next GPU's ring over CUDA IPC
  ([ssengine/runtime/README.md](../../ssengine/runtime/README.md) describes the transport). Planning and metadata for
  microbatch k+1 overlap the compute of microbatch k; microbatches up to 8,192 tokens replay CUDA graphs. Stage 3
  returns greedy tokens.
- **Intake** ([ssengine/serve/](../../ssengine/serve/)). Parses HTTP requests, renders chat templates and tokenizes in
  separate processes, and passes prompts to the controller through shared memory.

## Scheduling and the prefix cache

- **KV pool** ([ssengine/engine/kv.py](../../ssengine/engine/kv.py)). Blocks of 8,576 tokens (MLA and indexer pages of
  the DSA layers), plus KDA (linear-attention) state slots per request. HBM pages are allocated in 128-token pages
  inside blocks.
- **Exact prefix cache.** Sessions are kept whole and evicted least recently used first. A context resumes only where
  both its KV and its KDA state are saved.
- **Scheduler** ([ssengine/engine/scheduler.py](../../ssengine/engine/scheduler.py)). Microbatches of up to 32,768
  tokens, shortest remaining prefill first, with a load budget, an even split, and force-scheduling of any request
  that has waited 60 s. With tiers, landed tier reads are admitted first, in landing order (staged loading, see
  [tiers.md](tiers.md)). The served setting is the design setting of the evaluation ("st10",
  [ssengine/serve/launch.py](../../ssengine/serve/launch.py)); every field is in
  [ssengine/config.py](../../ssengine/config.py).

## Logs

Each run directory (`--run-dir`) holds `server.log`, `worker<k>.log`, `server_config.yaml`, `microbatches.jsonl` (one
line per microbatch and stage), `server_requests.jsonl`, `tier_ops.jsonl` with tiers, and `cache.json` (cache and
scheduler counters). [benchmarks/](../../benchmarks/) reads them.

## Provenance

The design was chosen and sized with the ServingStudio simulator, from its sim-to-real handoff
`glm53_flash_pp4_b200_v2` (ServingStudioSim `doc/handoff/`), whose milestone files are the specification.
[../sim_to_real/index.html](../sim_to_real/index.html) tells that story (Chinese, with an
[English version](../sim_to_real/en.html)). The original plan and the full design log of the tiers are kept on the
`dev` branch under `research/` (`research/PLAN.md`, `research/TIERS.md`); section numbers such as "TIERS.md §5" in
code comments refer to that log.
