"""FakeStageModel: the StageModel interface (ppimpl/runtime/README.md) without weights.

Per microbatch it takes `ms_per_mb + us_per_token * T` of stage time: a `torch.cuda._sleep` spin on the GPU (calibrated
once at start-up), or `time.sleep` on CPU. Tokens are deterministic: `fake_token(req_id, prompt length)`. To check the
transport, stage 0 writes `row % 128` into column 0 of the hidden rows it sends, and stage 3 adds
`hidden[last row, 0] - last row % 128` (0 when every row arrived in place) to the token.
"""

import time
import zlib

import torch

from ppimpl.config import RunConfig

VOCAB = 151_552


def fake_token(req_id: str, prompt_tokens: int) -> int:
    return zlib.crc32(f"{req_id}:{prompt_tokens}".encode()) % VOCAB


class FakeStageModel:
    def __init__(self, cfg: RunConfig, stage: int, device: torch.device):
        self.cfg, self.stage, self.device = cfg, stage, device
        self.last = stage == cfg.num_stages - 1
        self.num_blocks = cfg.fake.num_blocks
        self.hidden_width = cfg.fake.hidden_width
        self.cuda = device.type == "cuda"
        # Like StageModel: a tensor made under inference_mode (its warm-up) used with a requires_grad parameter in
        # forward, which only works when the worker runs forward under inference_mode too (job 7095). CPU scalars.
        with torch.inference_mode():
            self._inference_tensor = torch.ones(1)
        self._param = torch.nn.Parameter(torch.ones(1))
        self.cycles_per_ms = self._calibrate() if self.cuda else 0.0
        if self.cuda:
            self._warm_up()

    def _warm_up(self):
        """Run every kernel forward uses once (the first launch of each loads its module: tens of ms), and let the
        caching allocator reserve the largest output once (a cudaMalloc inside a microbatch costs ms)."""
        torch.empty(self.cfg.scheduler.mb_tokens, self.hidden_width, dtype=torch.bfloat16, device=self.device)
        hidden = torch.empty(2, self.hidden_width, dtype=torch.bfloat16, device=self.device)
        hidden[:, 0] = torch.arange(2, device=self.device) % 128
        self._check([1], [0], hidden)
        torch.cuda.synchronize(self.device)

    def _calibrate(self) -> float:
        torch.cuda._sleep(1_000_000)  # warm up
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cycles = 50_000_000
        a.record()
        torch.cuda._sleep(cycles)
        b.record()
        b.synchronize()
        return cycles / a.elapsed_time(b)

    def stage_ms(self, num_tokens: int) -> float:
        return self.cfg.fake.ms_per_mb + self.cfg.fake.us_per_token * num_tokens / 1e3

    def forward(self, plan, hidden_in: torch.Tensor | None) -> torch.Tensor:
        T = sum(c.num_tokens for c in plan.chunks)
        self._inference_tensor * self._param   # raises "Inference tensors cannot be saved for backward" otherwise
        ms = self.stage_ms(T)
        if self.cuda:
            if ms > 0:
                torch.cuda._sleep(int(ms * self.cycles_per_ms))
        elif ms > 0:
            time.sleep(ms / 1e3)
        if self.last:
            return self._tokens(plan, hidden_in)
        if hidden_in is None:
            hidden = torch.empty(T, self.hidden_width, dtype=torch.bfloat16, device=self.device)
            hidden[:, 0] = torch.arange(T, device=self.device) % 128
            return hidden
        return hidden_in

    def _tokens(self, plan, hidden_in: torch.Tensor | None) -> torch.Tensor:
        rows, base, row = [], [], 0
        for c in plan.chunks:
            row += c.num_tokens
            if c.is_last:
                rows.append(row - 1)
                base.append(fake_token(c.req_id, c.start + c.num_tokens))
        if not self.cuda:
            out = torch.tensor(base, dtype=torch.int64)
            if hidden_in is not None and rows:
                idx = torch.tensor(rows, dtype=torch.int64)
                out = out + hidden_in[idx, 0].to(torch.int64) - idx % 128
            return out
        return self._check(rows, base, hidden_in)

    def _check(self, rows: list[int], base: list[int], hidden_in: torch.Tensor | None) -> torch.Tensor:
        # Pinned source: a pageable host-to-device copy blocks the host until the compute stream drains, so this
        # stage could not enqueue (and post the receive of) its next microbatch while the current one runs.
        host = torch.tensor([rows, base], dtype=torch.int64, pin_memory=True)
        idx, out = host.to(self.device, non_blocking=True)
        if hidden_in is not None and rows:
            out = out + hidden_in[idx, 0].to(torch.int64) - idx % 128
        return out
