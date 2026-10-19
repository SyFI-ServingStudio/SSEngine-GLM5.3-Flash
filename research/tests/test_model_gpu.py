"""GPU tests of the stage model (run through Slurm: research/bench/slurm/chain.sbatch runs the same entry points).

(b) one stage vs the same layers unsplit (`edge` mode: with the fork's fused mHC op at the boundaries the split is
bitwise equal to unsplit; mega_mhc's ~4e-4 per-boundary difference is checked at the edge only, since NVFP4 layers
amplify it to ~0.3 over 23 layers whether or not the model is split), (c) the 4 chained stages vs vLLM TP4 on a 1k prompt, (d) the manifest
check on every traced stage. Each test runs `research/bench/chain.py` and reads its result file. (e) microbatches enqueued back
to back without a sync (the host several forwards ahead of the GPU, as in the pipeline) are bitwise equal to the same
microbatches run one at a time, at 1k-token requests (`research/bench/overlap.py`). (f) CUDA graph replay vs eager per bucket
(`research/bench/graphs.py`), eager repeated bitwise. (g) the Triton top-k canonicalization equals its torch reference and
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
    subprocess.run([sys.executable, "-m", "research.bench.chain", mode, "--prompt-len", str(L)], check=True,
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
    subprocess.run([sys.executable, "-m", "research.bench.overlap", "--stage", "1"], check=True, cwd=LANE)
    r = json.loads((LANE / "measured" / "host" / f"overlap_s1_{JOB}.json").read_text())
    assert r["pass"], r


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_graph_replay_matches_eager(stage):
    """(f) every bucket's replay bitwise equal to the same body run eagerly at the same padded T; without padding,
    equal to the eager forward (`research/bench/graphs.py`)."""
    out = LANE / "measured" / "graphs" / f"s{stage}_{JOB}.json"
    subprocess.run([sys.executable, "-m", "research.bench.graphs", "--stage", str(stage), "--out", str(out)],
                   check=True, cwd=LANE)
    r = json.loads(out.read_text())
    keys = ("replay_eq_eager_padded", "eager_repeat_eq", "pad_independent")
    bad = [(c["bucket"], c["kind"], [k for k in keys if not c[k]]) for c in r["cases"] if not all(c[k] for k in keys)]
    assert r["pass"], f"(bucket, kind, failed keys): {bad}"


@pytest.mark.parametrize("rows,width,two_requests", [(8192, 2048, False), (32768, 8192, False), (300, 700, True),
                                                     (3, 131072, False)])
def test_topk_canonical_matches_reference(rows, width, two_requests):
    """(g) ties at the k-th value (logits rounded to 0.1), rows of fewer and more keys than k, keys starting
    mid-row."""
    import torch

    from ppimpl.model.topk_canon import canonical_topk_torch, canonicalize, canonicalize_split

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
    for fn in (canonicalize, canonicalize_split, lambda *a: canonicalize_split(*a, span=1024)):
        for start in (dst.clone(), dst.flip(1).contiguous()):
            fn(lg, ks, ke, start, k)
            assert torch.equal(start, ref)


@pytest.mark.parametrize("rows,width", [(300, 700), (3, 131072)])
def test_topk_canonical_ignores_out_of_row_indices(rows, width):
    """DeepSelect writes junk indices (0x3F3F3F3F, left-over values) for rows with NaN logits (warm-up on
    uninitialized KV, job 7670): canonicalization never reads past the row and returns indices within it."""
    import torch

    from ppimpl.model.topk_canon import canonicalize, canonicalize_split

    dev, k = torch.device("cuda", 0), 512
    g = torch.Generator(device=dev).manual_seed(1)
    lg = torch.randn(rows, width, device=dev, generator=g)
    lg[::7, ::5] = float("nan")
    ks = torch.zeros(rows, dtype=torch.int32, device=dev)
    ke = torch.minimum(1 + torch.arange(rows, device=dev) * 37, torch.tensor(width, device=dev)).to(torch.int32)
    junk = torch.randint(0, width, (rows, k), device=dev, generator=g, dtype=torch.int32)
    junk[:, ::3] = 0x3F3F3F3F
    for fn in (canonicalize, canonicalize_split):
        dst = junk.clone()
        fn(lg, ks, ke, dst, k)
        torch.cuda.synchronize()
        assert ((dst >= -1) & (dst < (ke - ks)[:, None])).all()


@pytest.mark.parametrize("path", ["prefill", "deep_select", "kernel"])
@pytest.mark.parametrize("rows,width", [(1, 32768), (3, 131072), (64, 8192), (128, 4096)])
def test_decode_topk_canonical(rows, width, path):
    """(g) the indexer's decode top-k (next_n = 1, seq_lens [B, 1]; cooperative up to 64 rows, persistent above)
    through `pp_patch.canonical_decode_topk`: ties at the k-th value (logits rounded to 0.1), row ends from half to
    the full width, a row of fewer keys than k; equal to the torch reference whatever order the kernel left
    (research/bench/topk_check.py: per backend and shape, whether the kernel's set itself is valid)."""
    import torch

    from ppimpl.model import pp_patch
    from ppimpl.model.topk_canon import canonical_topk_torch
    from vllm.model_executor.layers import indexer_topk
    from vllm.v1.worker.workspace import init_workspace_manager

    import vllm._custom_ops  # noqa: F401  registers torch.ops._C
    import vllm._deepselect_C  # noqa: F401  registers torch.ops.deep_select

    pp_patch._ORIG_DECODE_TOPK = pp_patch._ORIG_DECODE_TOPK or indexer_topk.get_indexer_topk
    pp_patch._ORIG_TOPK = pp_patch._ORIG_TOPK or torch.ops._C.top_k_per_row_prefill
    pp_patch.DECODE_TOPK_PATH = path
    dev, k = torch.device("cuda", 0), 512
    torch.cuda.set_device(dev)
    init_workspace_manager(dev)
    g = torch.Generator(device=dev).manual_seed(rows)
    lg = torch.randn(rows, width, device=dev, generator=g).round(decimals=1)
    seq = torch.linspace(width // 2, width, rows, device=dev).to(torch.int32).reshape(rows, 1)
    if rows > 1:
        seq[0] = k // 2                               # a row of fewer keys than k
    ends = seq.reshape(-1)
    cols = torch.arange(width, device=dev)[None]
    ref = torch.where(cols < ends[:, None], lg, float("-inf")).topk(k, dim=1).indices.to(torch.int32)
    ref[torch.arange(k, device=dev)[None] >= ends[:, None]] = -1
    canonical_topk_torch(lg, torch.zeros_like(ends), ends, ref, k)   # rewrites a valid top-k into the canonical one
    run = pp_patch.canonical_decode_topk("auto")
    for _ in range(3):
        dst = torch.full((rows, k), -7, dtype=torch.int32, device=dev)
        run(lg, seq, 1, dst, k, int(seq.max()))
        assert torch.equal(dst, ref)
