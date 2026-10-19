"""Design constants of handoff glm53_flash_pp4_b200_v2 (the same as v1's) and the run config
(`python -m ppimpl serve --config`, or built from a model by `ppimpl serve <model>`: ppimpl/serve/launch.py)."""

import os
from pathlib import Path
from typing import Literal

import msgspec
import yaml

# 1_contract.md §1 and README "Design".
BLOCK_TOKENS = 8576          # one block: 8,576 tokens of DSA KV, or one KDA layer's state
CONTRACT_BLOCKS = 7144       # 1_contract.md §1: KV blocks per stage, incl. the null block
STATE_BLOCKS = 6             # KDA state blocks per request (5 cache groups + 1 scratch)
# The KV allocation unit (engine/kv.py): the smallest token range every KV kernel addresses through its own page table,
# lcm of FlashInfer sparse MLA's 64-token kernel block and the kpool indexer's storage page (DeepGEMM paged MQA: 32
# pooled entries x index_kpool 4 = 128 tokens). A block is BLOCK_PAGES pages; a KDA state still takes a whole block.
# Each stage derives it from its vLLM specs and checks it (`kv_tensors.page_tokens`).
PAGE_TOKENS = 128
BLOCK_PAGES = BLOCK_TOKENS // PAGE_TOKENS    # 67
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
    ssd_gb: float | None = 128.0          # None: no SSD tier. 128 GB per GPU (user, 2026-10-16); the handoff has 256
    ssd_read_gb_s: float = 10.0
    # Writes: the victim chain HBM -> DRAM -> SSD (user, 2026-10-17; TIERS.md §5): a tier takes only what the level
    # above evicts. Each level writes its least recently used entries ahead of their eviction once its occupancy
    # (HBM: running requests plus kept contexts not yet written; DRAM: entries not yet in SSD) passes its watermark
    # (kvpool/writeback.py); None: write only when evicting.
    hbm_write_ahead: float | None = 0.9
    dram_write_ahead: float | None = 0.9
    # "hbm-evictions": DRAM keeps a clean copy of what a read moves into HBM (inclusive after reads).
    # "hbm-evictions-exclusive" (simulator `dram_tier_writes`, d17027cf): a DRAM read releases the entry and HBM
    # writes a context when it evicts it (no write-ahead, as the simulator), so DRAM holds only what HBM evicted.
    # SSD stays inclusive.
    dram_writes: str = "hbm-evictions"
    # Staged loading (simulator `prefix_tier_loading: held`, `prefix_tier_loading_area`, ServingStudioSim pp-staged;
    # engine/scheduler.py): reads land in a loading area of this fraction of HBM's KV pool, which admission and kept
    # contexts never use, and SSD reads first in the same fraction of DRAM. A read holds its area from its start until
    # its round is admitted. None: off.
    loading_area: float | None = None
    bytes_per_token: int = 1635           # own stage (stage 0); balanced storage: 1,499
    state_bytes: int = 39_100_000         # one KDA state per context, own stage (stage 0); balanced: 36.9 MB
    max_read_wait_ms: float | None = 1000.0   # a read that would wait longer for its tier is passed over
    # Slot accounting (the server, lanes/m6 TIERS.md §3): None = bytes as above (replay). A slot is one GPU's share of
    # one KV block on all stages (balanced: 8,576 x 1,499 B, 4 KiB-aligned); a context's state takes state_slots.
    slot_bytes: int | None = None
    state_slots: int = 3
    staging_gb: float = 8.0               # host slots for every SSD read, and SSD writes without DRAM (never a hit)
    hops: str = "two"                     # "two": host <-> own HBM ring <-> the stages over NVLink; "direct": host <->
                                          # the stages' KV (peer writes from host: 31.7 vs 47.6 GB/s, TIERS.md §2.3)
    ring_depth: int = 2                   # two hops: chunks per direction's device ring (2: double buffering)
    ring_chunk_slots: int = 4             # slots per ring chunk (job 7961 sweep: 4 best or tied at 32k-512k). The ring
                                          # is 2 directions x depth x chunk x slot bytes, from the HBM left after the KV
    ssd_dir: str | None = None            # the SSD shard sets (kvpool/shards.py: set<k>/shard<q>.bin, kept across runs,
                                          # created once, contents never trusted); default $PPIMPL_SSD_DIR. The
                                          # workers get the set the leader took.
    ssd_backend: Literal["uring", "aio"] = "uring"   # the SSD tier's file I/O (kvpool/backends.py SSD_BACKENDS):
                                          # io_uring (uring.py) or Linux AIO (aio.py; Docker's default seccomp allows
                                          # it, not io_uring). The one named is set up or the start-up fails.


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
    control_port: int | None = None   # zmq, torch.distributed, tier ports: control_port .. +2+2N (default port+1)
    run_dir: str = "run"              # microbatches.jsonl, server logs
    layer_split: list[int] = msgspec.field(default_factory=lambda: list(LAYER_SPLIT))
    max_in_flight: int = MAX_IN_FLIGHT
    form_lead_ms: float = 3.0         # a plan forms this long before its predicted host time ahead of its projected
                                      # stage-0 start (ppimpl/runtime/dispatch.py)
    pool_tokens: int | None = None    # cap on the KV pool, in tokens (1_contract.md §2.3)
    cuda_graphs: bool = True          # glm53: microbatches up to 8192 tokens replay a CUDA graph per power-of-2 bucket
                                      # (padded to it); larger ones run eagerly (ppimpl/model/stage.py)
    intake_procs: int = 16            # HTTP in N intake processes on the port (SO_REUSEPORT) that parse requests and
                                      # pass them on over zmq (ppimpl/serve/intake.py); 0: HTTP in the controller.
                                      # 16: first TTFT at 8k 0.25 s vs 0.42 s with 4 (measured/runs_intake)
    profile_mbs: list[int] | None = None  # [a, b, ...]: profiler start at mb a, stop after b-1, per pair (run.py --nsys)
    keep_sessions: bool = False       # phase B: exact prefix cache (kept sessions, LRU eviction of whole sessions)
    hold_requests: int = 0            # form no microbatch before this many requests have arrived (6_scheduler.md §6:
                                      # the scheduling check's 26 requests at t = 0 are all queued before the first)
    # mb_tokens: one-request runs set it >= the prompt
    scheduler: SchedulerConfig = msgspec.field(default_factory=SchedulerConfig)
    tiers: TierConfig | None = None   # phase D: DRAM / SSD behind HBM
    # The OpenAI API (ppimpl/serve/openai.py; `ppimpl serve <model>` sets them, ppimpl/serve/launch.py): the model
    # name it answers to (None: "glm53"), and the longest prompt + output it takes (None: the model's, 1,048,576)
    served_model_name: str | None = None
    max_model_len: int | None = None
    fake: FakeConfig = msgspec.field(default_factory=FakeConfig)

    @property
    def num_stages(self) -> int:
        return len(self.layer_split)

    @property
    def base_port(self) -> int:
        return self.control_port if self.control_port is not None else self.port + 1

    # Port layout: base = controller PULL (reports), base+1+s = worker s PULL (plans), base+1+N = torch.distributed,
    # base+2+N = controller ROUTER for intake processes, base+3+N+s = worker s PULL (tier ops).
    def report_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.base_port}"

    def plan_addr(self, stage: int) -> str:
        return f"tcp://127.0.0.1:{self.base_port + 1 + stage}"

    def dist_init_method(self) -> str:
        return f"tcp://127.0.0.1:{self.base_port + 1 + self.num_stages}"

    def intake_addr(self) -> str:   # controller ROUTER <- intake DEALERs (intake_procs > 0)
        return f"tcp://127.0.0.1:{self.base_port + 2 + self.num_stages}"

    def tier_addr(self, stage: int) -> str:   # worker s PULL <- controller: tier ops (tiers on)
        return f"tcp://127.0.0.1:{self.base_port + 3 + self.num_stages + stage}"


def load_config(path: str | Path, **overrides) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return msgspec.convert(raw, RunConfig)


def dump_config(cfg: RunConfig) -> str:
    return yaml.safe_dump(msgspec.to_builtins(cfg), sort_keys=False)
