"""The frozen simulator runs behind the session targets, per round, so a shorter real run (a smoke) can be compared
with the same window of the target run.

  <python with pyarrow> -m research.bench.sim_rounds <sim run dir> [--out research/measured/sessions_sim/plain_hbm_c<N>.json.gz]

The pp-impl env has no parquet reader; run this once with one that has (e.g. the ServingStudioSim checkout's
`.venv/bin/python`, or `uv run` there). It reads raw/request_slo.parquet with the columns and the `completed`
filter of the handoff's data/metrics.py (`sim`), and writes them as arrays (arrival_ms, ttft_ms, new, prefix,
cached), with the run's session cap, the source path and the parquet's sha256. It checks that the full 20-minute
window of the export gives the run's line of data/session_targets.json (setting plain_hbm) exactly, so the export is
the run the targets came from.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.common import HANDOFF, MEASURED
from benchmarks.prefix import sim_arrays, window_metrics



def sim_rel(run: Path) -> str:
    """A simulator run directory as recorded: from its checkout's logs/ on (no host paths in tracked files)."""
    parts = str(run.resolve()).split("/logs/", 1)
    return "logs/" + parts[1] if len(parts) == 2 else str(run)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--setting", default="plain_hbm")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    import numpy as np
    import pyarrow.parquet as pq
    import yaml

    n = yaml.safe_load((a.run / "raw/run_config.yaml").read_text())["workload"]["max_concurrency"]
    pqf = a.run / "raw/request_slo.parquet"
    t = pq.read_table(pqf, columns=["arrival_time_ms", "completed", "ttft_ms", "declared_prefix_tokens",
                                    "fresh_prompt_tokens", "prefix_cache_hit_tokens"])
    c = t.column("completed").to_numpy(zero_copy_only=False)
    cols = {"arrival_ms": t.column("arrival_time_ms").to_numpy()[c].astype(float),
            "ttft_ms": t.column("ttft_ms").to_numpy(zero_copy_only=False)[c].astype(float),
            "new": t.column("fresh_prompt_tokens").to_numpy()[c].astype(np.int64),
            "prefix": t.column("declared_prefix_tokens").to_numpy()[c].astype(np.int64),
            "cached": np.nan_to_num(t.column("prefix_cache_hit_tokens").to_numpy(zero_copy_only=False)[c]
                                    .astype(float)).astype(np.int64)}
    full = window_metrics(sim_arrays({k: v.tolist() for k, v in cols.items()}), 20 * 60e3)
    target = next(r for r in json.loads((HANDOFF / "data/session_targets.json").read_text())
                  if r["setting"] == a.setting and r["max_sessions"] == n)
    for k in ("rounds", "rounds_per_s", "new_tokens_per_s", "computed_tokens_per_s", "prefix_hit_share"):
        assert abs(full[k] - target[k]) <= 1e-9 * max(1.0, abs(target[k])), (k, full[k], target[k])
    for p in ("p50", "p90", "p99"):
        assert abs(full["ttft_s"][p] - target["ttft_s"][p]) <= 1e-9, (p, full["ttft_s"][p], target["ttft_s"][p])
    out = a.out or MEASURED / "sessions_sim" / f"{a.setting}_c{n}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"setting": a.setting, "max_sessions": n, "source": sim_rel(a.run),
           "request_slo_sha256": hashlib.sha256(pqf.read_bytes()).hexdigest(),
           "full_window_equals_target": True, **{k: v.tolist() for k, v in cols.items()}}
    with gzip.open(out, "wt") as f:
        json.dump(doc, f)
    print(f"{out}: {len(cols['arrival_ms'])} rounds, N={n}; 20-minute window equals session_targets.json")


if __name__ == "__main__":
    main()
