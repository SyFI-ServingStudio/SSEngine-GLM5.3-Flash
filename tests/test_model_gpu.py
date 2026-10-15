"""GPU tests of the stage model (run through Slurm: bench/slurm/chain.sbatch runs the same entry points).

(b) one stage vs the same layers unsplit (`edge` mode: with the fork's fused mHC op at the boundaries the split is
bitwise equal to unsplit; mega_mhc's ~4e-4 per-boundary difference is checked at the edge only, since NVFP4 layers
amplify it to ~0.3 over 23 layers whether or not the model is split), (c) the 4 chained stages vs vLLM TP4 on a 1k prompt, (d) the manifest
check on every traced stage. Each test runs `bench/chain.py` and reads its result file. (e) microbatches enqueued back
to back without a sync (the host several forwards ahead of the GPU, as in the pipeline) are bitwise equal to the same
microbatches run one at a time, at 1k-token requests (`bench/overlap.py`). (f) CUDA graph replay vs eager per bucket
(`bench/graphs.py`), eager repeated bitwise. (g) the Triton top-k canonicalization equals its torch reference and
does not depend on the kernel's set or order (`ppimpl/model/topk_canon.py`).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LANE = Path(__file__).resolve().parents[1]
JOB = os.environ.get("SLURM_JOB_ID", "local")
pytestmark = pytest.mark.gpu


def _run(mode: str, L: int = 1024) -> None:
    subprocess.run([sys.executable, str(LANE / "bench" / "chain.py"), mode, "--prompt-len", str(L)], check=True,
                   cwd=LANE)


def _read(name: str) -> dict:
    return json.loads((LANE / "measured" / "chain" / f"{name}_{JOB}.json").read_text())


def test_stage_split_matches_unsplit():
    _run("edge")
    r = _read("edge_1024")
    assert r["pass"], {k: v for k, v in r.items() if not isinstance(v, list)}


def test_chain_token_matches_vllm_and_manifest():
    _run("chain")
    _run("vllm")
    ours, ref = _read("chain_1024_rank3"), _read("vllm_1024")
    assert ours["finite"]
    assert ours["token"] == ref["token"], (ours["top5_ids"], ref["top5_ids"])
    for s in range(4):
        chk = _read(f"chain_1024_rank{s}")["trace_check"]
        assert chk["ok"], chk["summary"]


def test_unsynced_forwards_match_synced():
    subprocess.run([sys.executable, str(LANE / "bench" / "overlap.py"), "--stage", "1"], check=True, cwd=LANE)
    r = json.loads((LANE / "measured" / "host" / f"overlap_s1_{JOB}.json").read_text())
    assert r["pass"], r


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_graph_replay_matches_eager(stage):
    """(f) every bucket's replay bitwise equal to the same body run eagerly at the same padded T; without padding,
    equal to the eager forward (`bench/graphs.py`)."""
    out = LANE / "measured" / "graphs" / f"s{stage}_{JOB}.json"
    subprocess.run([sys.executable, str(LANE / "bench" / "graphs.py"), "--stage", str(stage), "--out", str(out)],
                   check=True, cwd=LANE)
    r = json.loads(out.read_text())
    keys = ("replay_eq_eager_padded", "eager_repeat_eq", "pad_independent")
    assert r["pass"], [c for c in r["cases"] if not all(c[k] for k in keys)]


@pytest.mark.parametrize("rows,width,two_requests", [(8192, 2048, False), (32768, 8192, False), (300, 700, True)])
def test_topk_canonical_matches_reference(rows, width, two_requests):
    """(g) ties at the k-th value (logits rounded to 0.1), rows of fewer and more keys than k, keys starting
    mid-row."""
    import torch

    from ppimpl.model.topk_canon import canonical_topk_torch, canonicalize

    dev, k = torch.device("cuda", 0), 512
    g = torch.Generator(device=dev).manual_seed(0)
    lg = torch.randn(rows, width, device=dev, generator=g).round(decimals=1)
    r = torch.arange(rows, device=dev)
    ks = (torch.where(r < rows // 2, 0, width // 3) if two_requests else torch.zeros_like(r)).to(torch.int32)
    ke = torch.minimum(ks + 1 + r // 4, torch.full_like(ks, width)).to(torch.int32)
    cols = torch.arange(width, device=dev)[None]
    masked = torch.where((cols >= ks[:, None]) & (cols < ke[:, None]), lg, float("-inf"))
    dst = masked.topk(k, dim=1).indices.to(torch.int32) - ks[:, None]
    dst[dst + ks[:, None] >= ke[:, None]] = -1
    ref = dst.clone()
    canonical_topk_torch(lg, ks, ke, ref, k)
    for start in (dst.clone(), dst.flip(1).contiguous()):
        canonicalize(lg, ks, ke, start, k)
        assert torch.equal(start, ref)
