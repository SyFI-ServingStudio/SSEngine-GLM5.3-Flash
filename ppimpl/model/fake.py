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
        self.cycles_per_ms = self._calibrate() if self.cuda else 0.0

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
        idx = torch.tensor(rows, dtype=torch.int64)
        out = torch.tensor(base, dtype=torch.int64).to(self.device, non_blocking=True)
        if hidden_in is not None and rows:
            check = hidden_in[idx.to(self.device, non_blocking=True), 0].to(torch.int64) - (idx % 128).to(self.device)
            out = out + check
        return out
