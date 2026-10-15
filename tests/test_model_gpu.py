"""GPU tests of the stage model (run through Slurm: bench/slurm/chain.sbatch runs the same entry points).

(b) one stage vs the same layers unsplit (`edge` mode: with the fork's fused mHC op at the boundaries the split is
bitwise equal to unsplit; mega_mhc's ~4e-4 per-boundary difference is checked at the edge only, since NVFP4 layers
amplify it to ~0.3 over 23 layers whether or not the model is split), (c) the 4 chained stages vs vLLM TP4 on a 1k prompt, (d) the manifest
check on every traced stage. Each test runs `bench/chain.py` and reads its result file.
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
