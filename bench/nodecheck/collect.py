"""Collect the node check into measured/kernels_nodecheck.{jsonl,md}.

usage: python collect.py <specs_dir> <run_dir> [<run_dir> ...]
  specs_dir: sample.jsonl + calls.tsv + dbquery/ (read-only query of the current DB)
  run_dir:   one Slurm job's output: <kind>__<backend>.json from `run --fresh`, env.txt.
Later run dirs override earlier ones per (kind, backend) call that returned results.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TOL = 0.10


def load_call(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if "results" in data else None


def fmt_spec(spec):
    s = {k: v for k, v in spec.items() if k != "per_expert_batches"}
    if "per_expert_batches" in spec:
        s["per_expert_batches"] = "<288 counts>"
    return json.dumps(s, separators=(",", ":"))


def main(specs_dir, *run_dirs):
    specs_dir = Path(specs_dir)
    sample = [json.loads(line) for line in open(specs_dir / "sample.jsonl")]
    by_kb = defaultdict(list)
    for row in sample:
        by_kb[(row["kind"], row["backend"])].append(row)

    fresh, source = {}, {}
    for run_dir in run_dirs:
        for (kind, backend) in by_kb:
            data = load_call(Path(run_dir) / f"{kind}__{backend}.json")
            if data is None:
                continue
            fresh[(kind, backend)] = data["results"]
            source[(kind, backend)] = Path(run_dir).name
    out_rows = []
    for (kind, backend), rows in by_kb.items():
        dbq = load_call(specs_dir / "dbquery" / f"{kind}__{backend}.json")
        res = fresh.get((kind, backend))
        for i, row in enumerate(rows):
            r = res[i] if res else None
            fresh_ms = r["metrics"]["time_ms"] if r and r.get("status") == "ok" else None
            db_now = dbq["results"][i].get("metrics", {}).get("time_ms") if dbq else None
            out_rows.append({
                "kind": kind, "backend": backend, "config": row["config"][2], "rank": row["rank"], "config_rows": row["config_rows"],
                "spec": row["spec"], "db_time_ms": row["db_time_ms"], "fresh_ms": fresh_ms,
                "ratio": fresh_ms / row["db_time_ms"] if fresh_ms else None,
                "status": (r or {}).get("status", "not_run"),
                "error": (r or {}).get("error") or (r or {}).get("message"),
                "db_current_ms": db_now, "job": source.get((kind, backend)),
            })
    measured = Path(specs_dir).parent
    with open(measured / "kernels_nodecheck.jsonl", "w") as f:
        for o in out_rows:
            f.write(json.dumps(o) + "\n")

    configs = defaultdict(list)
    for o in out_rows:
        configs[(o["kind"], o["backend"], json.dumps(o["config"]))].append(o)
    lines = [
        "# Kernel node check (2_kernels.md §4)",
        "",
        "The ServingStudioSim profiler (`launcher kernel-profile run --fresh`, nothing written to any DB) re-measured on",
        "this node the smallest, median and largest row (by `db_time_ms`) of each of the 35 configurations in",
        "`data/kernel_specs.jsonl`. Ratio = fresh ÷ `db_time_ms`. Rows outside ±10% are marked **bold**.",
        "Per-spec data: `kernels_nodecheck.jsonl`; how it was run: the last section.",
        "",
        "| Kind / backend | Fixed configuration | Rows | Smallest: DB ms / ratio | Median: DB ms / ratio | Largest: DB ms / ratio | Outside ±10% |",
        "|---|---|---|---|---|---|---|",
    ]
    flagged, ratios = [], []
    for (kind, backend, cfg), os_ in sorted(configs.items()):
        cells, nflag = [], 0
        for o in sorted(os_, key=lambda o: ("smallest", "median", "largest").index(o["rank"])):
            if o["ratio"] is None:
                cells.append(f"{o['db_time_ms']:.4g} / {o['status']}")
                continue
            ratios.append(o["ratio"])
            out = abs(o["ratio"] - 1) > TOL
            nflag += out
            txt = f"{o['ratio']:.3f}"
            cells.append(f"{o['db_time_ms']:.4g} / " + (f"**{txt}**" if out else txt))
            if out:
                flagged.append(o)
        cfgtxt = ", ".join(f"{k} {v}" for k, v in json.loads(cfg).items()) or "—"
        rows = os_[0]["config_rows"] if "config_rows" in os_[0] else ""
        lines.append(f"| `{kind}` / `{backend}` | {cfgtxt} | {rows} | " + " | ".join(cells) + f" | {nflag} |")
    if ratios:
        lines += ["", f"{len(ratios)} of {len(out_rows)} specs measured; ratio median {statistics.median(ratios):.3f}, "
                      f"min {min(ratios):.3f}, max {max(ratios):.3f}; {len(flagged)} outside ±10%."]
    by_rank = defaultdict(list)
    for o in flagged:
        by_rank[o["rank"]].append((o["fresh_ms"] - o["db_time_ms"]) * 1000)
    if flagged:
        lines += ["", "Outside ±10% by rank: " + "; ".join(
            f"{rank} {len(by_rank[rank])} (fresh − DB {min(by_rank[rank]):+.2f} to {max(by_rank[rank]):+.2f} µs)"
            for rank in ("smallest", "median", "largest") if by_rank[rank]) + "."]
    lines += ["", "## Outside ±10%", ""]
    if not flagged:
        lines.append("None.")
    for o in flagged:
        cur = f", current DB {o['db_current_ms']:.4g} ms" if o["db_current_ms"] else ""
        lines.append(f"- `{o['kind']}` / `{o['backend']}` ({o['rank']}): DB {o['db_time_ms']:.4g} ms, fresh "
                     f"{o['fresh_ms']:.4g} ms, ratio {o['ratio']:.3f}{cur}. Spec `{fmt_spec(o['spec'])}`")
    notrun = [o for o in out_rows if o["ratio"] is None]
    if notrun:
        lines += ["", "## Not measured", ""]
        for o in notrun:
            lines.append(f"- `{o['kind']}` / `{o['backend']}` ({o['rank']}): {o['status']} {o['error'] or ''}")
    jobs = sorted({str(o["job"]) for o in out_rows if o["job"]})
    lines += ["", "## How it was run", "",
              f"Slurm jobs {', '.join(jobs)} (`bench/nodecheck/run_nodecheck.sbatch`; later jobs re-ran single calls after a",
              "spec fix), one profiler CLI call per (kind, backend), one GPU per lane:",
              "`uv run --no-sync python -m launcher kernel-profile run <kind> --backend <b> --specs <file> --fresh",
              "--gpu-name \"NVIDIA B200\" --db <scratch copy of profile.db> --json` in `wt-pp-main` (ServingStudioSim at the",
              "handoff commit), each backend in its registered profiler environment (most in the `vllm_env` container",
              "`vibesim-profiler-vllm:cu130-3f667d7e`), CUPTI timing with the profiler's cold-L2 clear. The samples come",
              "from `bench/nodecheck/sample_specs.py` (`measured/nodecheck_specs/`); DB text flags `clean_logits` and",
              "`return_valid_counts` are passed as booleans. `uptime`, GPUs and versions per job: `nodecheck_runs/<job>/env.txt`."]
    (measured / "kernels_nodecheck.md").write_text("\n".join(lines) + "\n")
    print(f"{len(out_rows)} rows, {len(ratios)} measured, {len(flagged)} outside ±10%")


if __name__ == "__main__":
    main(*sys.argv[1:])
