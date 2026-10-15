"""Design constants of handoff glm53_flash_pp4_b200_v1 and the run config (`python -m ppimpl serve --config`)."""

from pathlib import Path

import msgspec
import yaml

# 1_contract.md §1 and README "Design".
BLOCK_TOKENS = 8576          # one block: 8,576 tokens of DSA KV, or one KDA layer's state
STATE_BLOCKS = 6             # KDA state blocks per request (5 cache groups + 1 scratch)
NULL_BLOCK = 0
HIDDEN_WIDTH = 4 * 4096      # 4-stream mHC residual, BF16: 32 KiB per token between stages
MB_TOKENS = 8192             # plain scheduler: microbatch budget and chunk size
MAX_IN_FLIGHT = 4
LAYER_SPLIT = [12, 11, 11, 11]


class FakeConfig(msgspec.Struct, forbid_unknown_fields=True):
    """FakeStageModel: no weights, a per-token delay, deterministic tokens."""
    us_per_token: float = 7.0         # stage time per token (57 ms per 8k microbatch ~ stage 0's target)
    ms_per_mb: float = 0.0            # fixed stage time per microbatch
    num_blocks: int = 7144            # reported in Ready (incl. the null block): 7,143 usable = 61,258,368 tokens
    hidden_width: int = HIDDEN_WIDTH  # smaller only for CPU tests


class RunConfig(msgspec.Struct, forbid_unknown_fields=True):
    model: str = "fake"               # "fake" (FakeStageModel) or "glm53" (ppimpl.model.stage.StageModel)
    checkpoint: str | None = None
    served_model_name: str = "glm53-flash-pp4"
    device: str = "cuda"              # "cuda": NCCL p2p, worker s on cuda:s; "cpu": gloo (tests)
    host: str = "127.0.0.1"
    port: int = 8000                  # HTTP
    control_port: int | None = None   # zmq + torch.distributed ports: control_port .. control_port+5 (default port+1)
    run_dir: str = "run"              # microbatches.jsonl, server logs
    layer_split: list[int] = msgspec.field(default_factory=lambda: list(LAYER_SPLIT))
    mb_tokens: int = MB_TOKENS        # microbatch budget; one-request runs set it >= the prompt
    max_in_flight: int = MAX_IN_FLIGHT
    dispatch_ahead: int = 1           # plans queued on stage 0 behind the running one (0: form when stage 0 is free)
    pool_tokens: int | None = None    # cap on the KV pool, in tokens (1_contract.md §2.3)
    fake: FakeConfig = msgspec.field(default_factory=FakeConfig)

    @property
    def num_stages(self) -> int:
        return len(self.layer_split)

    @property
    def base_port(self) -> int:
        return self.control_port if self.control_port is not None else self.port + 1

    # Port layout: base = controller PULL (reports), base+1+s = worker s PULL (plans), base+1+N = torch.distributed.
    def report_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.base_port}"

    def plan_addr(self, stage: int) -> str:
        return f"tcp://127.0.0.1:{self.base_port + 1 + stage}"

    def dist_init_method(self) -> str:
        return f"tcp://127.0.0.1:{self.base_port + 1 + self.num_stages}"


def load_config(path: str | Path, **overrides) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return msgspec.convert(raw, RunConfig)


def dump_config(cfg: RunConfig) -> str:
    return yaml.safe_dump(msgspec.to_builtins(cfg), sort_keys=False)
