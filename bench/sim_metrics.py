"""Session metrics of simulator runs beside the targets, e.g. the frozen simulator with a GPU-time multiplier
(measured/sessions_sim/mult106.json: `gpu_time_multiplier` 1.06, the sustained-load clock under the B200 power cap,
measured/issues.md).

  <python with pyarrow> -m bench.sim_metrics <setting>=<sim run dir> ... [--out measured/sessions_sim/mult106.json]

Each run's line is the handoff's data/metrics.py output (rounds/s, new, prompt and computed tokens/s, prefix hit share,
TTFT p50/p90/p99/max, rounds over 60 s) with its setting, session cap and GPU-time multiplier from the run's
run_config.yaml, and the run's directory relative to its ServingStudioSim checkout. Lines of other
(setting, N) pairs already in the file are kept. Needs pyarrow (the ServingStudioSim checkout's python), as
bench/sim_rounds.py does.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from bench.common import HANDOFF, MEASURED


def run_line(setting: str, run: Path) -> dict:
    text = next(f for f in (run / "raw/run_config.yaml", run / "run_config.yaml") if f.exists()).read_text()
    cfg = yaml.safe_load(text)
    mult = [float(x) for x in re.findall(r"gpu_time_multiplier:\s*([0-9.]+)", text)]
    out = subprocess.run([sys.executable, str(HANDOFF / "data/metrics.py"), str(run)], check=True,
                         capture_output=True, text=True)
    rel = str(run.resolve()).split("/logs/", 1)
    return {"setting": setting, "max_sessions": cfg["workload"]["max_concurrency"],
            "gpu_time_multiplier": max(mult) if mult else 1.0,
            "run": "logs/" + rel[1] if len(rel) == 2 else str(run), **json.loads(out.stdout)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="<setting>=<sim run dir>")
    ap.add_argument("--out", type=Path, default=MEASURED / "sessions_sim" / "mult106.json")
    a = ap.parse_args()
    lines = {(r["setting"], r["max_sessions"]): r for r in json.loads(a.out.read_text())} if a.out.exists() else {}
    for arg in a.runs:
        setting, run = arg.split("=", 1)
        r = run_line(setting, Path(run))
        lines[(r["setting"], r["max_sessions"])] = r
        print(json.dumps(r), flush=True)
    a.out.write_text(json.dumps(sorted(lines.values(), key=lambda r: (r["setting"], r["max_sessions"])), indent=1)
                     + "\n")


if __name__ == "__main__":
    main()
