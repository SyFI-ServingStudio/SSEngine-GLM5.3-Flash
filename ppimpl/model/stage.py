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

CUDA graphs (`cuda_graphs`): a microbatch of up to the largest bucket (`GRAPH_BUCKETS`) is padded to its
bucket and replayed from the fork's breakable CUDA graph (`vllm.compilation.breakable_cudagraph`): one capture per
bucket of the whole stage forward, cut at the ops the fork marks as eager breaks (MLA attention, the kpool indexer,
our KDA prefill), which run eagerly on each replay with the microbatch's own metadata. The captured segments (mHC,
norms, GEMMs, MoE, quantization, MLA KV write) read only static buffers: positions, the g0 slot mapping (padded rows
-1), token ids or the received hidden rows. Larger microbatches run eagerly.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import warnings
from pathlib import Path

import torch

# the fork resolves its eager-break decorators at import: set before any vLLM model module is imported. Outside a
# capture the decorated ops run unchanged.
os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

from ppimpl import kernels  # noqa: E402
from ppimpl.config import CHECKPOINT  # noqa: E402
from ppimpl.model import kv_tensors, pp_patch  # noqa: E402
from ppimpl.model.metadata import StageMetadata  # noqa: E402

HIDDEN_WIDTH = 4 * 4096
# FlashInfer tactics, tuned once (bench/tune_tactics.py) and loaded at warm-up with tuning off, as vLLM loads its
# persisted configs: start-up skips ~150 s of MoE profiling and every stage, run and test uses the same tactics.
# The file's own metadata (FlashInfer/CUDA/cuBLAS/cuDNN versions, GPU) decides whether it is valid here.
TACTICS = Path(os.environ.get("PPIMPL_FI_TACTICS",
                              Path(__file__).resolve().parents[1] / "data" / "flashinfer_tactics_b200.json"))
RESERVE_GB = 13.6     # 1_contract.md §1: activations and workspace beside weights and KV (and the CUDA graph pool)
# token counts captured as CUDA graphs: 128, then every 256 to 8192 (33). A microbatch pads to its bucket, so
# coarse buckets cost compute: with powers of 2 a 3,000-token prompt ran as 4,096 (round 0 TTFT 114 vs 96.7 ms
# target, x1.18; job 7462), and on the N=400 smoke's stage-0 sizes (7569) the pad was 3.8% of the tokens (every 1024:
# 1.5%; every 256: 0.6%). The graph pool is shared and sized by the 8192 bucket, so more buckets cost little memory.
GRAPH_BUCKETS = (128,) + tuple(range(256, 8192 + 1, 256))
LONG_WARMUP = 65536    # context at which warm-up runs its long continuation chunks (StageModel.warmup)


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
    # enforce_eager turns vLLM's JIT kernel warm-up off (VllmConfig: "equivalent to --kernel_config.
    # enable_jit_warmup=False"), and with it every kernel's registration of its compile keys; StageModel compiles
    # them in warm-up (`_warm_jit_keys`), so keep it on (7693: 1 registration with it off)
    vc.kernel_config.enable_jit_warmup = True
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


def graph_capture(pool):
    """The fork's BreakableCUDAGraphCapture, except that each graph segment is replayed as soon as it is captured.
    An eager break runs during capture (it is not captured), on inputs the segments before it would have computed;
    without the replay those are whatever the shared graph pool last held there (another bucket's tensors, read as
    other dtypes: NaN keys, so NaN logits and out-of-range top-k rows: the illegal memory access with 12 buckets in one
    pool, 3,072 after 8,192-4,096, job 7559). With it, a capture is an ordinary forward of the dummy plan."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    class Capture(BreakableCUDAGraphCapture):
        def _end_segment(self) -> None:
            g = self._current_graph if self._capturing else None
            super()._end_segment()
            if g is not None:
                g.replay()

    return Capture(pool=pool)


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

        # vLLM's JIT kernels register their compile-key space while the model is built (as GPUModelRunner's init
        # does under JitWarmupRegistry.capture); `warmup` compiles every key (`_warm_jit_keys`)
        from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

        self.jit_registry = JitWarmupRegistry(vc)
        with set_default_torch_dtype(vc.model_config.dtype), torch.device(device), self.jit_registry.activate():
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
        pp_patch.install_kpool_unaligned()
        if pp_patch.FAST["kpool_expand"]:
            pp_patch.install_kpool_expand()
        if pp_patch.FAST["kda_split_heads"]:
            pp_patch.install_kda_split_heads()
        # the stage edge's hc_post outside the graphs (`_run_graph`); fixed here, before any capture
        self.edge_send_eager = bool(pp_patch.FAST["edge_send_eager"]) and not self.last
        pp_patch.install_decode_topk_canon()
        pp_patch.install_sparse_mla_split()
        pp_patch.install_indexer_split()
        pp_patch.install_moe_buckets()
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
            t0 = time.perf_counter()
            self.warmup()
            warmup_s = round(time.perf_counter() - t0, 1)
            if st["cuda_graphs"] if cuda_graphs is None else cuda_graphs:
                self.capture_graphs()
                self._regrow()
            self.graph_info["warmup_s"] = warmup_s
            self.graph_info["jit_warmup"] = self.jit_warmup

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
        16, one request alone continuing a cached prefix (start divisible by 16 and not; and past `LONG_WARMUP` when the
        pool and max_model_len hold it), the served shapes of `_warm_served_shapes` (decode rows, many requests
        continuing prefixes, unaligned pool starts), and the first layer's mHC pre at every split count. Uses blocks
        1.. before any request holds them.
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
            # one request alone continuing its cached prefix, as every round >= 1 is: the KDA state gather of one
            # slot loaded a torch module on the first such request (8k + 1k, 7432: 23 ms cuLibraryLoadData + 11.5
            # ms lazy kernel load); and the same at a start and length not divisible by 16 (P = 100,003)
            run(NS(mb_id=-5, chunks=[chunk("wc", 1024, 256, 1)]))
            run(NS(mb_id=-6, chunks=[chunk("wu", 1003, 253, 1)]))
            # past 64k of context (when the pool holds it): the sparse-MLA index remap at the long top-k width
            # (NUM_TOPK_TOKENS 2176, 180 ms Triton compile on the first chunk ending at 65,536, job 7505) and
            # topk_canon (the indexer is wider than k), at a row count divisible by 4 and not (its scratch rows are
            # 16-byte aligned or not: a 1,699-row chunk compiled three kernels, 1.2 s, job 7505)
            if (self.num_blocks - 1 - 6) * B >= LONG_WARMUP + 1003 + 253 \
                    and LONG_WARMUP + 1003 + 253 <= self.vc.model_config.max_model_len:
                run(NS(mb_id=-7, chunks=[chunk("wl", LONG_WARMUP, 256, 1)]))
                run(NS(mb_id=-8, chunks=[chunk("wv", LONG_WARMUP + 1003, 253, 1)]))
            for mb_id, specs in self._warm_served_shapes():
                run(NS(mb_id=mb_id, chunks=self._warm_chunks(specs)))
            self._warm_mhc_pre()
        self._warm_jit_keys()
        torch.cuda.synchronize(self.device)
        self._log_own_kernel_variants()
        pp_patch.clear_autotune_memo()     # tactics may have changed while tuning

    def _regrow(self) -> None:
        """Graph capture empties the allocator's cache (`_capture_graphs`), so the first served microbatches would
        allocate their eager-path buffers from the driver: 35 more device allocations per stage and +37 ms on the
        first 4x1k microbatch (sched_check_good 7738), +80 ms with expandable segments. Run the shapes that set the
        allocator's peak once more: several fresh prompts, the largest microbatch, and the largest one continuing
        long contexts."""
        from types import SimpleNamespace as NS

        T = self.max_tokens
        shapes = [[(0, 1024)] * 4, [(0, 1024)] * 16, [(0, T)]]
        long = [(LONG_WARMUP + 7 * i, T // 4) for i in range(4)]
        if (sum(-(-(st + n) // self.kv.block_tokens) + 6 for st, n in long) <= self.num_blocks - 1
                and LONG_WARMUP + T <= self.vc.model_config.max_model_len):
            shapes.append(long)
        with torch.inference_mode():
            for i, specs in enumerate(shapes):
                if sum(n for _, n in specs) > T:
                    continue
                hidden = None if self.first else torch.zeros(sum(n for _, n in specs), self.hidden_width,
                                                              dtype=torch.bfloat16, device=self.device)
                self.forward(NS(mb_id=-200 - i, chunks=self._warm_chunks(specs)), hidden)
        torch.cuda.synchronize(self.device)

    def _warm_chunks(self, specs: list[tuple[int, int]]) -> list:
        """Warm-up chunks (start, tokens), each continuing a cached prefix of `start` tokens, on blocks 1.. laid out
        one request after another (6 state blocks each)."""
        from types import SimpleNamespace as NS

        B, base, out = self.kv.block_tokens, 1, []
        for i, (start, n) in enumerate(specs):
            nb = -(-(start + n) // B)
            out.append(NS(req_id=f"ws{i}", start=start, num_tokens=n, kv_blocks=list(range(base, base + nb)),
                          is_last=True, state_blocks=list(range(base + nb, base + nb + 6)), token_ids=[1] * n))
            base += nb + 6
        return out

    def _warm_served_shapes(self):
        """The served microbatch shapes whose kernels the earlier warm-up shapes did not compile (smoke 7658: 11
        Triton compiles per stage during the run, 0.3-3.3 s pipeline stalls). Triton compiles a kernel per constexpr
        value and per specialization of its integer and pointer arguments (an integer divisible by 16 or not, a
        pointer 16-byte aligned or not), so each kernel has to run at every class the server meets:
        - rows past k pools (topk_canon kth / count / finish) at row counts 2 and 3 mod 4 (its scratch rows are
          16-byte aligned or not by the row count), and 6,246 rows (7658's microbatch 2);
        - a 1-token append after a cached prefix (the indexer's decode metadata, the kpool decode update, the split
          canonicalization) at every g0 table width (`StageMetadata.table_width`: powers of 2 of blocks, and the cap
          at max_model_len), which also covers the split canonicalization's piece counts;
        - 1, 2 and 3 such rows ahead of prefill requests that start inside a pool (the kpool insert with its write
          mask and tail seed, on slices offset by the decode rows);
        - 2 to 9 requests continuing cached prefixes, with and without one starting fresh (the indexer's prefill
          chunk metadata over request slices);
        - 3, 4, 8 and 16 fresh 1,024-token prompts (the shortest-first scheduler's first microbatches).
        Shapes the KV pool, the microbatch budget or max_model_len cannot hold are skipped. Yields
        (mb_id, [(start, tokens)])."""
        B = self.kv.block_tokens
        free = self.num_blocks - 1
        mml = self.vc.model_config.max_model_len

        def fits(specs):
            return (sum(-(-(st + n) // B) + 6 for st, n in specs) <= free
                    and sum(n for _, n in specs) <= self.max_tokens and all(st + n <= mml for st, n in specs))

        shapes = [[(8192, 254)], [(8192, 255)], [(8192, 6246)]]
        b = 1
        while True:                            # one 1-token append per table width, the last one at the cap
            P = b * B - 5
            if P + 1 > mml - 1:
                shapes.append([(mml - 2, 1)])
                break
            shapes.append([(P, 1)])
            b *= 2
        prefills = [(15290, 646), (36051, 1203), (20001, 300)]           # starts 2, 3, 1 mod 4
        for d in (1, 2, 3):
            shapes.append([(40001 + 7 * i, 1) for i in range(d)] + prefills)
        # the same with prefill tokens divisible by 16 (the tail seed at a slot slice offset by the decode rows and a
        # token count divisible by 16: budget N=500 7730 mb 16263, 1 decode row + 6,352 prefill tokens, 45 ms)
        for d in (1, 2, 3):
            shapes.append([(40001 + 7 * i, 1) for i in range(d)] + prefills[:2] + [(20001, 311)])
        for r in range(2, 10):
            reqs = [(20789 + 997 * i, 128 + 61 * i) for i in range(r)]
            shapes.append(reqs)
            shapes.append(reqs[:-1] + [(0, 300)])
        for r in (3, 4, 8, 16):                # several fresh prompts in one microbatch (6_scheduler.md's shortest-first
            shapes.append([(0, 1024)] * r)     # microbatches: 7708's first 4x1k / 8x1k one took +16-25 ms of host time)
        for i, specs in enumerate(shapes):
            if fits(specs):
                yield -100 - i, specs

    def _log_own_kernel_variants(self) -> None:
        """Print the compiled specializations of the Triton kernels vLLM's JitWarmupRegistry does not declare (ours and
        the fork's kpool ops), so a first use in a run can be matched against what the warm-up left compiled."""
        from triton.runtime.jit import JITFunction

        import ppimpl.model.topk_canon as tc
        from vllm.models.glm5next.nvidia.ops import kpool_compress as kops

        out = {}
        for mod in (kops, tc):
            for name, fn in vars(mod).items():
                fn = getattr(fn, "fn", fn) if not isinstance(fn, JITFunction) else fn
                if isinstance(fn, JITFunction):
                    caches = [c[0] for c in fn.device_caches.values()]
                    out[name] = sum(len(c) for c in caches)
        print(f"stage {self.stage}: own Triton kernel variants after warm-up {json.dumps(out)}", flush=True)

    def _warm_jit_keys(self) -> None:
        """Compile every compile key vLLM's JIT kernels declare for this model (`JitWarmupRegistry.warmup`, as vLLM's
        `kernel_warmup` does): each Triton integer argument at 1, divisible by 16 and neither, each pointer 16-byte
        aligned and not. Shapes alone miss classes the server meets (smoke 7683: BuildPrefillChunkMetadataKernel at
        an unaligned request slice with a query slice divisible by 16, mb 1095, 72-83 ms on every stage). Keys
        already compiled by the shapes above are cache hits."""
        t0 = time.perf_counter()
        self.jit_registry.warmup()
        self.jit_warmup = {"registrations": len(self.jit_registry), "s": round(time.perf_counter() - t0, 1)}
        print(f"stage {self.stage}: JIT key warm-up {self.jit_warmup}", flush=True)

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
                e.capture = graph_capture(pool)
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
        if not self.last and not self.edge_send_eager:
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
            if self.edge_send_eager:
                # the last layer's hc_post on the bucket's rows, eagerly into a new tensor: what the graph would
                # have computed (same kernel, same inputs), without the copy out of the graph's output buffer
                sent = pp_patch.edge_send(self.layers[-1], *out)
                return (sent[:T].clone(),) if raw else sent[:T]
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
        if b.last_rows.shape[0] == 0:   # no request ends in this microbatch (chunks of longer prompts): no head
            return torch.empty(0, dtype=torch.int64, device=x.device)
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
