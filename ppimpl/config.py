"""Design constants of handoff glm53_flash_pp4_b200_v2 (the same as v1's) and the run config
(`python -m ppimpl serve --config`)."""

import os
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

# The weights: Hugging Face nvidia/GLM-5.3-Flash-NVFP4 at this revision, in the hub cache under HF_HOME
# (`hf download nvidia/GLM-5.3-Flash-NVFP4 --revision <CHECKPOINT_REVISION>`). PPIMPL_CHECKPOINT names another
# directory.
CHECKPOINT_REPO = "nvidia/GLM-5.3-Flash-NVFP4"
CHECKPOINT_REVISION = "da920bb0b9f4a06727223a349e55468e38352348"
CHECKPOINT = os.environ.get("PPIMPL_CHECKPOINT") or str(
    Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    / f"models--{CHECKPOINT_REPO.replace('/', '--')}" / "snapshots" / CHECKPOINT_REVISION)


class SchedulerConfig(msgspec.Struct, forbid_unknown_fields=True):
    """The scheduler of 6_scheduler.md §2, one switch per step. `SCHEDULER_STEPS` names the handoff's settings."""
    order: str = "fifo"                   # "fifo": started first, then arrival order (step 0); "srpt": (b), (c)
    mb_tokens: int = MB_TOKENS            # largest microbatch (step 0-1: the fixed budget; 2-4: 32,768)
    load_budget: bool = False             # step 2: 4,096 tokens up to a 1M backlog, mb_tokens at 3M, linear between
    load_budget_low: int = 4096
    backlog_lo: int = 1_000_000
    backlog_hi: int = 3_000_000
    even_split: bool = False              # step 3: also at most ceil(B / depth), at least even_min
    even_min: int = 512
    depth: int = MAX_IN_FLIGHT
    force_after_ms: float | None = None   # step 4: a request waiting this long goes first


SCHEDULER_STEPS = {
    "plain": SchedulerConfig(),
    "srpt8k": SchedulerConfig(order="srpt"),
    "budget": SchedulerConfig(order="srpt", mb_tokens=32768, load_budget=True),
    "even": SchedulerConfig(order="srpt", mb_tokens=32768, load_budget=True, even_split=True),
    "good": SchedulerConfig(order="srpt", mb_tokens=32768, load_budget=True, even_split=True, force_after_ms=60_000.0),
}


class TierConfig(msgspec.Struct, forbid_unknown_fields=True):
    """DRAM and SSD behind the HBM prefix cache (7_offloading.md §2), per GPU. Own-stage storage: every GPU keeps its
    own stage's bytes, so the stage with the most bytes (stage 0) sets the size and the read time."""
    dram_gb: float | None = 150.0         # None: no DRAM tier
    dram_read_gb_s: float = 50.0
    ssd_gb: float | None = 256.0          # None: no SSD tier
    ssd_read_gb_s: float = 10.0
    ssd_writes: str = "dram-evictions"    # "dram-evictions": SSD is written only with what DRAM evicts; "every-context"
    bytes_per_token: int = 1635           # own stage (stage 0); balanced storage: 1,499
    state_bytes: int = 39_100_000         # one KDA state per context, own stage (stage 0); balanced: 36.9 MB
    max_read_wait_ms: float | None = 1000.0   # a read that would wait longer for its tier is passed over


class FakeConfig(msgspec.Struct, forbid_unknown_fields=True):
    """FakeStageModel: no weights, a per-token delay, deterministic tokens."""
    us_per_token: float = 7.0         # stage time per token (57 ms per 8k microbatch ~ stage 0's target)
    num_blocks: int = 7144            # reported in Ready (incl. the null block): 7,143 usable = 61,258,368 tokens
    hidden_width: int = HIDDEN_WIDTH  # smaller only for CPU tests


class RunConfig(msgspec.Struct, forbid_unknown_fields=True):
    model: str = "fake"               # "fake" (FakeStageModel) or "glm53" (ppimpl.model.stage.StageModel)
    checkpoint: str | None = None
    device: str = "cuda"              # "cuda": worker s on cuda:s, activations by copy engine (CePipe);
                                      # "cpu": gloo (tests)
    host: str = "127.0.0.1"
    port: int = 8000                  # HTTP
    control_port: int | None = None   # zmq + torch.distributed ports: control_port .. control_port+5 (default port+1)
    run_dir: str = "run"              # microbatches.jsonl, server logs
    layer_split: list[int] = msgspec.field(default_factory=lambda: list(LAYER_SPLIT))
    max_in_flight: int = MAX_IN_FLIGHT
    dispatch_ahead: int = 1           # 1: one plan may wait queued on stage 0, as the max_in_flight+1-th (0: form
                                      # only when stage 0 is free); see ppimpl/runtime/dispatch.py
    form_lead_ms: float = 3.0         # the queued plan forms this long before the running one's predicted stage-0 end
    pace_margin_ms: float | None = 2.0    # pace forming to the later stages (ppimpl/runtime/dispatch.py; None: off)
    pool_tokens: int | None = None    # cap on the KV pool, in tokens (1_contract.md §2.3)
    cuda_graphs: bool = True          # glm53: microbatches up to 8192 tokens replay a CUDA graph per power-of-2 bucket
                                      # (padded to it); larger ones run eagerly (ppimpl/model/stage.py)
    intake_procs: int = 16            # HTTP in N intake processes on the port (SO_REUSEPORT) that parse requests and
                                      # pass them on over zmq (ppimpl/serve/intake.py); 0: HTTP in the controller.
                                      # 16: first TTFT at 8k 0.25 s vs 0.42 s with 4 (measured/runs_intake)
    profile_mbs: list[int] | None = None  # [a, b): cudaProfilerStart at mb a, Stop after mb b-1 (bench/run.py --nsys)
    keep_sessions: bool = False       # phase B: exact prefix cache (kept sessions, LRU eviction of whole sessions)
    hold_requests: int = 0            # form no microbatch before this many requests have arrived (6_scheduler.md §6:
                                      # the scheduling check's 26 requests at t = 0 are all queued before the first)
    # mb_tokens: one-request runs set it >= the prompt
    scheduler: SchedulerConfig = msgspec.field(default_factory=SchedulerConfig)
    tiers: TierConfig | None = None   # phase D: DRAM / SSD behind HBM
    fake: FakeConfig = msgspec.field(default_factory=FakeConfig)

    @property
    def num_stages(self) -> int:
        return len(self.layer_split)

    @property
    def base_port(self) -> int:
        return self.control_port if self.control_port is not None else self.port + 1

    # Port layout: base = controller PULL (reports), base+1+s = worker s PULL (plans), base+1+N = torch.distributed,
    # base+2+N = controller ROUTER for intake processes.
    def report_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.base_port}"

    def plan_addr(self, stage: int) -> str:
        return f"tcp://127.0.0.1:{self.base_port + 1 + stage}"

    def dist_init_method(self) -> str:
        return f"tcp://127.0.0.1:{self.base_port + 1 + self.num_stages}"

    def intake_addr(self) -> str:   # controller ROUTER <- intake DEALERs (intake_procs > 0)
        return f"tcp://127.0.0.1:{self.base_port + 2 + self.num_stages}"


def load_config(path: str | Path, **overrides) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return msgspec.convert(raw, RunConfig)


def dump_config(cfg: RunConfig) -> str:
    return yaml.safe_dump(msgspec.to_builtins(cfg), sort_keys=False)
