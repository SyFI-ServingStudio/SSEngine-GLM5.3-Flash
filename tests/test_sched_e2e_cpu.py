"""The scheduling check end to end on CPU (6_scheduler.md §6-§7): bench/run.py sched_check_<step> with FakeStageModel
over gloo and req-frontend; the server holds the 26 requests at t = 0 (RunConfig.hold_requests) and must form the
microbatches of data/sched_check_targets.json chunk for chunk. TTFT is not checked: the fake stages are not timed."""

import os
import subprocess
import sys

import pytest

from bench import run as bench_run
from bench.common import ROOT
from bench.sched import check_sched


@pytest.mark.parametrize("step", ["plain", "good"])
def test_sched_check_chunk_for_chunk(tmp_path, step):
    if not bench_run.CLIENT_BIN.exists():
        pytest.skip(f"req-frontend not built: {bench_run.CLIENT_BIN}")
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    name = f"sched_check_{step}"
    p = subprocess.run([sys.executable, "-m", "bench.run", name, "--fake", "--inline", "--cpu", "--us-per-token", "1",
                        "--out-root", str(tmp_path), "--client-timeout", "300"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    run_dir = tmp_path / name
    assert (run_dir / "microbatches.jsonl").exists(), p.stdout[-3000:] + p.stderr[-3000:]
    c = check_sched(run_dir, step)
    chunk_diffs = [d for d in c["diffs"] if not d.startswith("TTFT")]
    assert not chunk_diffs, chunk_diffs[:5]
    assert len(c["ttft_ms"]) == 28
