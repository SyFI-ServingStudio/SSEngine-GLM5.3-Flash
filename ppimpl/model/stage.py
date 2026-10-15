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
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch

from ppimpl import kernels
from ppimpl.model import kv_tensors, pp_patch
from ppimpl.model.metadata import StageMetadata

CHECKPOINT = "/raid/hf/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/da920bb0b9f4a06727223a349e55468e38352348"
HIDDEN_WIDTH = 4 * 4096
RESERVE_GB = 13.6     # 1_contract.md §1: activations and workspace beside weights and KV
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
        world, rank, method = dist.get_world_size(), dist.get_rank(), None
    else:
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        world, rank, method = 1, 0, f"tcp://127.0.0.1:{port}"
    ps.init_distributed_environment(world_size=world, rank=rank, distributed_init_method=method or "env://",
                                    local_rank=device.index, backend="nccl")
    ps.initialize_model_parallel(1, 1)


def stage_settings(cfg) -> dict:
    """What StageModel reads of the run config (`ppimpl.config.RunConfig`; benches pass a namespace with the same
    fields). Read directly, so a missing field fails here instead of falling back silently."""
    return {"layer_split": tuple(cfg.layer_split), "checkpoint": cfg.checkpoint or CHECKPOINT,
            "mb_tokens": int(cfg.scheduler.mb_tokens), "pool_tokens": cfg.pool_tokens}


def bench_config(mb_tokens: int, checkpoint: str | None = None):
    """A run-config stand-in for the benches (same fields as RunConfig's that StageModel reads)."""
    from types import SimpleNamespace

    return SimpleNamespace(layer_split=list(kernels.SPLIT), checkpoint=checkpoint, pool_tokens=None,
                           scheduler=SimpleNamespace(mb_tokens=mb_tokens))


class StageModel:
    def __init__(self, cfg, stage: int, device: torch.device, *, layers: range | None = None,
                 first: bool | None = None, last: bool | None = None, num_blocks: int | None = None,
                 max_model_len: int | None = None, warmup: bool = True):
        st = stage_settings(cfg)
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
        if warmup:
            self.warmup()

    def warmup(self, autotune: bool = True) -> None:
        """Before `Ready`: run every kernel once (first launches load modules) and tune FlashInfer's kernels as vLLM's
        `kernel_warmup` does (one autotuned pass at the microbatch budget, then plain passes). Shapes: one request at
        the full budget (indexer and both top-k paths), then two requests, one continuing a cached prefix (conv and
        KDA initial state, chunk start > 0). Uses blocks 1.. before any request holds them."""
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

        ctx = contextlib.nullcontext()
        if autotune:
            try:
                from flashinfer.autotuner import autotune as fi_autotune
                ctx = fi_autotune(True)
            except ImportError:
                pass
        with torch.inference_mode():
            for i, plan in enumerate(plans()):
                T_p = sum(c.num_tokens for c in plan.chunks)
                hidden = None if self.first else torch.zeros(T_p, self.hidden_width, dtype=torch.bfloat16,
                                                              device=self.device)
                with ctx if i == 0 else contextlib.nullcontext():
                    self.forward(plan, hidden)
                self.forward(plan, hidden)
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------------------------------------ forward

    def forward(self, plan, hidden_in: torch.Tensor | None) -> torch.Tensor:
        return self.run(self.prepare(plan), hidden_in)

    def prepare(self, plan):
        """Host side of one microbatch: metadata and its non-blocking copies (overlaps the previous compute)."""
        return self.meta.build(plan, with_tokens=self.first)

    def run(self, b, hidden_in: torch.Tensor | None) -> torch.Tensor:
        from vllm.forward_context import set_forward_context

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
