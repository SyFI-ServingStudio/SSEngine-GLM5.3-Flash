"""One pipeline stage of GLM-5.3-Flash NVFP4, built from the vLLM fork's glm5next modules (runtime/README.md).

Construction:
1. VllmConfig from `EngineArgs` (text model only, FP8 KV, `max_num_batched_tokens` = the run's microbatch budget so a
   one-request run's whole prompt fits one call), vLLM process groups single-rank (`initialize_model_parallel(1, 1)`;
   the runtime's own 4-rank group, when present, becomes vLLM's external data-parallel dimension).
2. The model with only this stage's decoder layers (`pp_patch.stage_scope`), its weights read from the checkpoint
   files that hold them (`pp_patch.stage_weights`), `process_weights_after_loading`.
3. Kernel routing of the design (`pp_patch`): mega_mhc at every boundary, Dao conv + FlashInfer persistent KDA,
   per-shape indexer top-k.
4. vLLM's block size (8576) and the stage's KV tensors in vLLM's hybrid layout (`kv_tensors`), metadata builders
   (`metadata`).

Forward runs the layers inside vLLM's forward context; no host sync: plan -> metadata (host) -> GPU work.

CUDA graphs (`cuda_graphs`): a microbatch of up to the largest bucket (`GRAPH_BUCKETS`, powers of 2) is padded to its
bucket and replayed from the fork's breakable CUDA graph (`vllm.compilation.breakable_cudagraph`): one capture per
bucket of the whole stage forward, cut at the ops the fork marks as eager breaks (MLA attention, the kpool indexer,
our KDA prefill), which run eagerly on each replay with the microbatch's own metadata. The captured segments (mHC,
norms, GEMMs, MoE, quantization, MLA KV write) read only static buffers: positions, the g0 slot mapping (padded rows
-1), token ids or the received hidden rows. Larger microbatches run eagerly.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path

import torch

# the fork resolves its eager-break decorators at import: set before any vLLM model module is imported. Outside a
# capture the decorated ops run unchanged.
os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

from ppimpl import kernels  # noqa: E402
from ppimpl.model import kv_tensors, pp_patch  # noqa: E402
from ppimpl.model.metadata import StageMetadata  # noqa: E402

CHECKPOINT = "/raid/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348"
HIDDEN_WIDTH = 4 * 4096
# FlashInfer tactics, tuned once (bench/tune_tactics.py) and loaded at warm-up with tuning off, as vLLM loads its
# persisted configs: start-up skips ~150 s of MoE profiling and every stage, run and test uses the same tactics.
# The file's own metadata (FlashInfer/CUDA/cuBLAS/cuDNN versions, GPU) decides whether it is valid here.
TACTICS = Path(os.environ.get("PPIMPL_FI_TACTICS",
                              Path(__file__).resolve().parents[1] / "data" / "flashinfer_tactics_b200.json"))
RESERVE_GB = 13.6     # 1_contract.md §1: activations and workspace beside weights and KV (and the CUDA graph pool)
GRAPH_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)   # token counts captured as CUDA graphs


def graph_bucket(num_tokens: int, buckets) -> int | None:
    """The smallest bucket that holds the microbatch, or None (run eagerly)."""
    return next((b for b in sorted(buckets) if b >= num_tokens), None)
MAX_MODEL_LEN = 1_048_576


def build_vllm_config(checkpoint: str, max_num_batched_tokens: int, max_model_len: int = MAX_MODEL_LEN):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model=checkpoint, tensor_parallel_size=1, pipeline_parallel_size=1,
                      kv_cache_dtype="fp8_e4m3", enable_prefix_caching=False, language_model_only=True,
                      max_model_len=max_model_len, max_num_batched_tokens=max_num_batched_tokens,
                      max_num_seqs=max(64, max_num_batched_tokens // 128), enforce_eager=True,
                      distributed_executor_backend="external_launcher")
    vc = args.create_engine_config()
    # sparse MLA through TRTLLM's MQA kernel for every prefill (config 25), as the targets price it: the fork's dense
    # MHA prefill for prompts within the top-k (<= 2,048 pools) is slower (M2: DSA layer x1.11 at 2k)
    vc.attention_config.sparse_mla_force_mqa = True
    return vc


def init_vllm_distributed(device: torch.device) -> None:
    """vLLM's groups: one rank each. Joins the runtime's torch.distributed world if it exists (all stage workers
    construct their StageModel together, so the group creation is collective), else a one-process world."""
    import torch.distributed as dist
    from vllm.distributed import parallel_state as ps

    if ps._TP is not None:
        return
    if dist.is_initialized():
        ps.init_distributed_environment(world_size=dist.get_world_size(), rank=dist.get_rank(),
                                        distributed_init_method="env://", local_rank=device.index, backend="nccl")
    else:
        import socket
        for attempt in range(5):     # the free port can be taken between probe and listen (7114: EADDRINUSE)
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            try:
                ps.init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                                local_rank=device.index, backend="nccl")
                break
            except dist.DistNetworkError:
                if attempt == 4:
                    raise
    ps.initialize_model_parallel(1, 1)


def stage_settings(cfg) -> dict:
    """What StageModel reads of the run config (`ppimpl.config.RunConfig`; benches pass a namespace with the same
    fields). Read directly, so a missing field fails here instead of falling back silently."""
    return {"layer_split": tuple(cfg.layer_split), "checkpoint": cfg.checkpoint or CHECKPOINT,
            "mb_tokens": int(cfg.scheduler.mb_tokens), "pool_tokens": cfg.pool_tokens,
            "cuda_graphs": bool(cfg.cuda_graphs)}


def bench_config(mb_tokens: int, checkpoint: str | None = None, cuda_graphs: bool = False):
    """A run-config stand-in for the benches (same fields as RunConfig's that StageModel reads). Benches run eagerly
    unless they ask for CUDA graphs."""
    from types import SimpleNamespace

    return SimpleNamespace(layer_split=list(kernels.SPLIT), checkpoint=checkpoint, pool_tokens=None,
                           scheduler=SimpleNamespace(mb_tokens=mb_tokens), cuda_graphs=cuda_graphs)


NINJA = "/usr/bin/ninja"


def load_tactics(path: Path | None = None) -> dict:
    """Load the persisted FlashInfer tactics into the autotuner (lookup only, no profiling). Returns what a run
    records: source "file" with path and sha256, or source "missing"/"invalid" with the reason."""
    import hashlib

    from flashinfer.autotuner import AutoTuner

    path = Path(path or TACTICS)
    if not path.is_file():
        return {"source": "missing", "file": str(path), "reason": f"{path} does not exist"}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not AutoTuner.get().load_configs(str(path)):
        return {"source": "invalid", "file": str(path), "sha256": sha,
                "reason": f"{path} was tuned in another environment (FlashInfer/CUDA/GPU metadata)"}
    return {"source": "file", "file": str(path), "sha256": sha}


def check_ninja() -> bool:
    """FlashInfer JIT builds only with /usr/bin/ninja (1.11.1): another ninja on PATH hashes build commands differently
    and rebuilds every object of the shared ~/.cache/flashinfer (fused_moe_trtllm_sm100 ~290 s). Warns, never fails."""
    import shutil
    import warnings

    found = shutil.which("ninja")
    if found != NINJA:
        msg = f"ninja on PATH is {found}, expected {NINJA}: FlashInfer JIT will rebuild the shared cache"
        warnings.warn(msg, stacklevel=2)
        print(f"WARNING: {msg}", flush=True)
        return False
    return True


def json_dumps(x) -> str:
    import json
    return json.dumps(x)


class GraphEntry:
    """One captured bucket: its static buffers (`rows`: positions, g0 slot mapping, token ids; `hidden`: the received
    rows), the forward context's slot mapping (static views), the capture and its outputs (graph pool)."""

    def __init__(self, bucket: int, rows: torch.Tensor, hidden: torch.Tensor | None):
        self.bucket, self.rows, self.hidden = bucket, rows, hidden
        self.slots: dict = {}
        self.capture = None
        self.out: tuple = ()


class StageModel:
    def __init__(self, cfg, stage: int, device: torch.device, *, layers: range | None = None,
                 first: bool | None = None, last: bool | None = None, num_blocks: int | None = None,
                 max_model_len: int | None = None, warmup: bool = True,
                 host_trims: bool = True, cuda_graphs: bool | None = None):
        st = stage_settings(cfg)
        check_ninja()
        split = st["layer_split"]
        self.stage, self.device = stage, device
        self.layer_range = layers if layers is not None else kernels.stage_layers(stage, split)
        self.first = stage == 0 if first is None else first
        self.last = stage == len(split) - 1 if last is None else last
        self.hidden_width = HIDDEN_WIDTH
        torch.cuda.set_device(device)
        ckpt = st["checkpoint"]
        # one-request runs set mb_tokens >= the prompt, so the whole prompt is one call; at least 8192
        max_tokens = max(st["mb_tokens"], 8192)
        os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "SD")

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import set_default_torch_dtype
        from vllm.v1.worker.workspace import init_workspace_manager

        self.vc = vc = build_vllm_config(ckpt, max_tokens, max_model_len or MAX_MODEL_LEN)
        self._vc_ctx = set_current_vllm_config(vc)
        self._vc_ctx.__enter__()      # the stage process runs inside this config for good
        # as vLLM's WorkerBase: IR op priorities (rms_norm etc. -> their CUDA kernels) and torch-wrap state
        import vllm.ir
        vc.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(vc.compilation_config.ir_enable_torch_wrap)
        init_vllm_distributed(device)
        init_workspace_manager(device)

        with set_default_torch_dtype(vc.model_config.dtype), torch.device(device):
            with pp_patch.stage_scope(self.layer_range, self.first, self.last):
                model = initialize_model(vllm_config=vc)
        model.load_weights(pp_patch.stage_weights(Path(ckpt), self.layer_range, self.first, self.last))
        process_weights_after_loading(model, vc.model_config, device)
        self.model = model.eval()
        lm = model.language_model
        self.inner = lm.model
        self.layers = list(self.inner.layers[self.layer_range.start:self.layer_range.stop])
        self.embed = self.inner.embed_tokens if self.first else None
        self.norm = self.inner.norm if self.last else None
        self.lm_head = lm.lm_head if self.last else None

        if self.last and self.layers and self.layers[-1].layer_idx == self.layers[-1].num_hidden_layers - 1:
            # layer 44 defers its final mHC post + stream mean to output_head, which applies them to the rows whose
            # token is returned only (both are per token)
            self.layers[-1].num_hidden_layers += 1

        # design kernel routing
        pp_patch.route_mhc_boundaries(self.layers)
        pp_patch.route_hc_expand()
        for layer in self.layers:
            if layer.layer_kind == "kda":
                pp_patch.route_kda(layer.self_attn)
        pp_patch.install_topk_router()
        pp_patch.install_sparse_mla_split()
        # host trims (no kernel or argument changes): env cache, FlashInfer tactic memo / mm_fp4 skip_check, plain
        # parameters
        self.host_trims = self.install_host_trims() if host_trims else None

        # KV: vLLM's block size, then this stage's tensors
        current_platform.update_block_size_for_backend(vc)
        assert vc.cache_config.block_size == 8576, vc.cache_config.block_size
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

        attn_layers = get_layers_from_vllm_config(vc, AttentionLayerBase)
        local_spec = {n: l.get_kv_cache_spec(vc) for n, l in attn_layers.items()}
        per_block = kv_tensors.bytes_per_block(vc, local_spec)
        if num_blocks is None:
            free, _ = torch.cuda.mem_get_info(device)
            num_blocks = kv_tensors.blocks_that_fit(per_block, free, int(RESERVE_GB * 1e9),
                                                    st["pool_tokens"])
        self.kv = kv_tensors.allocate(vc, vc.compilation_config.static_forward_context, local_spec, num_blocks,
                                      device)
        self.num_blocks = num_blocks
        self.meta = StageMetadata(vc, self.kv, device)
        self.max_tokens = max_tokens
        self.graphs: dict[int, GraphEntry] = {}
        self.graph_info: dict = {"enabled": False}
        if warmup:
            self.warmup()
            if st["cuda_graphs"] if cuda_graphs is None else cuda_graphs:
                self.capture_graphs()

    def install_host_trims(self) -> dict:
        mods = [m for m in (*self.layers, self.embed, self.norm, self.lm_head) if m is not None]
        return pp_patch.install_host_trims(mods)

    def warmup(self, tune: str = "auto") -> None:
        """Before `Ready`: run every kernel once (first launches load modules) with FlashInfer's tactics settled.
        `tune`: "auto" loads `TACTICS` (tuning off) and tunes in this process, loudly, only when the file is missing
        or invalid here; "force" tunes in this process (bench/tune_tactics.py); "off" uses neither. Tuning is one
        autotuned pass at the microbatch budget, as vLLM's `kernel_warmup` does. Shapes: one request at the full
        budget (indexer and both top-k paths), then two requests, one continuing a cached prefix (conv and KDA initial
        state, chunk start > 0), then every power-of-2 length from 512 below the budget, a length not divisible by
        16, and the first layer's mHC pre at every split count. Uses blocks 1.. before any request holds them.
        `self.tactics` records the source."""
        from types import SimpleNamespace as NS

        B = self.kv.block_tokens
        T = min(self.max_tokens, (self.num_blocks - 1 - 6) * B)

        def chunk(rid, start, n, base):
            kv = list(range(base, base + -(-(start + n) // B)))
            return NS(req_id=rid, start=start, num_tokens=n, kv_blocks=kv, is_last=True,
                      state_blocks=list(range(base + len(kv), base + len(kv) + 6)), token_ids=[1] * n)

        def plans():
            yield NS(mb_id=-1, chunks=[chunk("w0", 0, T, 1)])
            yield NS(mb_id=-2, chunks=[chunk("w1", 0, 512, 1), chunk("w2", 1024, 1024, 1 + 1 + 6)])

        self.tactics = load_tactics() if tune == "auto" else {"source": tune}
        ctx = contextlib.nullcontext()
        if tune == "force" or (tune == "auto" and self.tactics["source"] != "file"):
            from flashinfer.autotuner import autotune as fi_autotune

            if tune == "auto":
                msg = (f"FlashInfer tactics not loaded ({self.tactics['reason']}): tuning in this process (~150 s, "
                       f"tactics may differ per process). Run bench/tune_tactics.py to write {TACTICS}.")
                warnings.warn(msg, stacklevel=2)
                print(f"WARNING: {msg}", flush=True)
                self.tactics["source"] = "tuned_in_process"
            ctx = fi_autotune(True)
        def run(plan):
            T_p = sum(c.num_tokens for c in plan.chunks)
            hidden = None if self.first else torch.zeros(T_p, self.hidden_width, dtype=torch.bfloat16,
                                                          device=self.device)
            self.forward(plan, hidden)

        with torch.inference_mode():
            for i, plan in enumerate(plans()):
                with ctx if i == 0 else contextlib.nullcontext():
                    run(plan)
                run(plan)
            # first-use costs keyed by shape (7126): every power-of-2 token count from 512 below the budget (MoE
            # tactic buckets, kernel modules first loaded per bucket; one-request runs at those exact lengths), and
            # one count not divisible by 16 (Triton specializes integer arguments on divisibility: 1000 tokens
            # compiled two builder/indexer kernels, 120 ms, on first use)
            p = 512
            while p < T:
                run(NS(mb_id=-3, chunks=[chunk("wb", 0, p, 1)]))
                p *= 2
            run(NS(mb_id=-4, chunks=[chunk("wo", 0, 1000, 1)]))
            self._warm_mhc_pre()
        torch.cuda.synchronize(self.device)
        pp_patch.clear_autotune_memo()     # tactics may have changed while tuning

    def _warm_mhc_pre(self) -> None:
        """The standalone mHC pre of the stage's first layer (layer 0 after the input head, or after the stage edge)
        runs a split-K DeepGEMM + TileLang pair whose split count is n_SMs // ceil(T / 64) below ~5k tokens, and each
        new count loads its own kernels on first use (7126: ~80 ms on the first microbatch of each new count, on
        every stage). Run it once per distinct split count up to the budget, on zeros."""
        from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split

        layer = self.layers[0]
        n, H = layer.n, layer.hidden_size
        first_T: dict[int, int] = {}
        for T in range(64, self.max_tokens + 1, 64):
            first_T.setdefault(compute_num_split(64, n * H, -(-T // 64)), T)
        x = torch.zeros(max(first_T.values()), n, H, dtype=torch.bfloat16, device=self.device)
        norm = layer.input_layernorm
        for T in sorted(first_T.values()):
            layer.hc_pre(x[:T], layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                         norm_weight=norm.weight.data, norm_eps=norm.variance_epsilon)
        self.mhc_pre_splits = sorted(first_T)

    # ------------------------------------------------------------------------------------------------ CUDA graphs

    def capture_graphs(self, buckets=GRAPH_BUCKETS) -> dict:
        """Capture the stage forward once per bucket (largest first, one shared memory pool) after warm-up: a dummy
        one-request plan of exactly the bucket, run once eagerly in PIECEWISE mode, then captured. Records what the
        graphs cost in `graph_info` (device memory taken, capture time)."""
        # Capture on a private side stream (the default stream cannot capture, and the caller's stream is not
        # ours to choose). `torch.cuda.stream` goes through vLLM's patched set_stream, so the eager breaks recorded
        # during capture see the same stream. Replays later run on whatever stream forward() is called on.
        caller = torch.cuda.current_stream(self.device)
        side = self._capture_stream = torch.cuda.Stream(self.device)
        side.wait_stream(caller)
        with torch.cuda.stream(side):
            info = self._capture_graphs(buckets)
        caller.wait_stream(side)
        torch.cuda.synchronize(self.device)
        return info

    def _capture_graphs(self, buckets) -> dict:
        import gc
        import time
        from types import SimpleNamespace as NS

        B = self.kv.block_tokens
        buckets = sorted((b for b in buckets if b <= min(self.max_tokens, B)), reverse=True)
        t0 = time.perf_counter()
        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        free0 = torch.cuda.mem_get_info(self.device)[0]
        pool = torch.cuda.graph_pool_handle()
        hidden = None if self.first else torch.zeros(buckets[0], self.hidden_width, dtype=torch.bfloat16,
                                                     device=self.device)
        with torch.inference_mode():
            for Tb in buckets:
                plan = NS(mb_id=-10, chunks=[NS(req_id="g", start=0, num_tokens=Tb, kv_blocks=[1], is_last=True,
                                                state_blocks=list(range(2, 8)), token_ids=[1] * Tb)])
                b = self.meta.build(plan, with_tokens=self.first, bucket=Tb)
                e = GraphEntry(Tb, torch.empty_like(b.graph_rows), None if hidden is None else hidden[:Tb])
                e.rows.copy_(b.graph_rows)
                e.slots = {name: e.rows[1] for name in b.slots}
                with self._graph_context(b, e):
                    self._graph_body(e)                       # eager once at this shape, PIECEWISE paths
                torch.cuda.synchronize(self.device)
                from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

                e.capture = BreakableCUDAGraphCapture(pool=pool)
                with self._graph_context(b, e), e.capture:
                    e.out = self._graph_body(e)
                torch.cuda.synchronize(self.device)
                self.graphs[Tb] = e
        free1 = torch.cuda.mem_get_info(self.device)[0]
        self.graph_info = {
            "enabled": True, "buckets": sorted(self.graphs), "capture_s": round(time.perf_counter() - t0, 1),
            "graph_memory_gib": round((free0 - free1) / 2**30, 3),
            "segments": {Tb: [e.capture.num_graphs, e.capture.num_eager_breaks] for Tb, e in self.graphs.items()},
            "free_after_gib": round(free1 / 2**30, 2)}
        print(f"stage {self.stage}: CUDA graphs {json_dumps(self.graph_info)}", flush=True)
        return self.graph_info

    def _graph_context(self, b, e):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context

        return set_forward_context(b.attn, self.vc, num_tokens=e.bucket, slot_mapping=e.slots,
                                   cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                                   batch_descriptor=BatchDescriptor(num_tokens=e.bucket))

    def _graph_body(self, e) -> tuple:
        """The stage forward on a graph's static buffers (what is captured, and its eager equivalent)."""
        x = self.embed(e.rows[2]) if self.first else pp_patch.edge_recv(e.hidden)
        residual = post = comb = None
        for layer in self.layers:
            x, residual, post, comb = layer(e.rows[0], x, residual, post, comb)
        if not self.last:
            return (pp_patch.edge_send(self.layers[-1], x, residual, post, comb),)
        return (x, residual, post, comb)

    def _run_graph(self, b, hidden_in, replay: bool = True, raw: bool = False):
        """Pad to the bucket, replay (or, `replay=False`, run the same body eagerly on the same static buffers: the
        exactness reference), then the output rows. A non-last stage returns a copy: the next replay of this bucket
        overwrites the graph's output while the send of this one may still be queued. `raw`: copies of the body's
        outputs' first T rows instead (tests)."""
        e = self.graphs[b.bucket]
        T = b.num_tokens
        e.rows.copy_(b.graph_rows)
        if not self.first:
            e.hidden[:T].copy_(hidden_in)
        with self._graph_context(b, e):
            if replay:
                e.capture.replay()
                out = e.out
            else:
                out = self._graph_body(e)
        if raw:
            return tuple(None if t is None else t[:T].clone() for t in out)
        if not self.last:
            return out[0][:T].clone()
        x, residual, post, comb = (None if t is None else t[:T] for t in out)
        return self.output_head(b, x, residual, post, comb)

    # ------------------------------------------------------------------------------------------------ forward

    def forward(self, plan, hidden_in: torch.Tensor | None) -> torch.Tensor:
        return self.run(self.prepare(plan), hidden_in)

    def prepare(self, plan):
        """Host side of one microbatch: metadata and its non-blocking copies (overlaps the previous compute)."""
        bucket = None
        if self.graphs:
            bucket = graph_bucket(sum(c.num_tokens for c in plan.chunks), self.graphs)
        return self.meta.build(plan, with_tokens=self.first, bucket=bucket)

    def run(self, b, hidden_in: torch.Tensor | None) -> torch.Tensor:
        from vllm.forward_context import set_forward_context

        if b.bucket is not None:
            return self._run_graph(b, hidden_in)
        with set_forward_context(b.attn, self.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
            x, residual, post, comb = self.run_layers(b, hidden_in)
            if not self.last:
                return pp_patch.edge_send(self.layers[-1], x, residual, post, comb)
            return self.output_head(b, x, residual, post, comb)

    def run_layers(self, b, hidden_in):
        if self.first:
            x = self.embed(b.token_ids)
        else:
            x = pp_patch.edge_recv(hidden_in)
        residual = post = comb = None
        for layer in self.layers:
            x, residual, post, comb = layer(b.positions, x, residual, post, comb)
        return x, residual, post, comb

    def output_head(self, b, x, residual, post, comb) -> torch.Tensor:
        return self.head_logits(b, x, residual, post, comb, fp32=False).argmax(dim=-1)

    def head_logits(self, b, x, residual, post, comb, fp32: bool = True) -> torch.Tensor:
        """The rows whose token is returned, then on those rows only: the last layer's deferred mHC post and the
        mean of the 4 streams (both per token), final norm, lm_head -> FP32 logits [rows, vocab]."""
        rows = b.last_rows
        if post is not None:
            x = self.layers[-1].hc_post(x.index_select(0, rows), residual.index_select(0, rows),
                                        post.index_select(0, rows), comb.index_select(0, rows)).mean(dim=1)
        else:
            x = x.index_select(0, rows)
        h = self.norm(x)
        logits = torch.nn.functional.linear(h, self.lm_head.weight)
        return logits.float() if fp32 else logits
