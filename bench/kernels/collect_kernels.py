"""Collect the pp-impl re-measure into measured/kernels.jsonl and measured/kernels.md (2_kernels.md §8).

usage: python collect_kernels.py <queue_dir> <measured_dir>
Reads done/<stem>.rows.jsonl (handoff rows), out/<stem>.json (CLI --fresh results, same order),
knames/<stem>/*.jsonl (worker hook: per-spec CUPTI kernel names and provenance notes).
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_queue import CONFIGS  # noqa: E402

TOL = 0.10


def canon(d):
    return json.dumps(d, sort_keys=True, default=list)


def hook_index(dirpath):
    """spec canon -> (kernel names, provenance note), from the worker hook logs."""
    names, notes = {}, {}
    for f in sorted(Path(dirpath).glob("*.jsonl")) if Path(dirpath).is_dir() else []:
        cur = None
        for line in open(f):
            ev = json.loads(line)
            if ev["event"] == "spec":
                cur = canon(ev["kwargs"])
                names.setdefault(cur, set())
            elif ev["event"] == "cupti" and cur is not None:
                names[cur].update(ev.get("names") or [])
            elif ev["event"] == "provenance":
                notes[canon(ev["kwargs"])] = ev.get("note")
    return names, notes


def main(queue, measured):
    queue, measured = Path(queue), Path(measured)
    rows_out = []
    for rows_file in sorted((queue / "done").glob("*.rows.jsonl")):
        stem = rows_file.name[: -len(".rows.jsonl")]
        rows = [json.loads(l) for l in open(rows_file)]
        try:
            res = json.loads((queue / "out" / f"{stem}.json").read_text())["results"]
        except (OSError, KeyError, json.JSONDecodeError):
            res = [None] * len(rows)
        names, notes = hook_index(queue / "knames" / stem)
        for row, r in zip(rows, res):
            spec = row["spec"]
            if isinstance(spec.get("per_expert_batches"), str):
                spec = {**spec, "per_expert_batches": json.loads(spec["per_expert_batches"])}
            ok = r is not None and r.get("status") == "ok"
            t = r["metrics"]["time_ms"] if ok else None
            key = canon(spec)
            rows_out.append({
                "config": row["config"], "kind": row["kind"], "backend": row["backend"], "spec": row["spec"],
                "db_time_ms": row["db_time_ms"], "real_time_ms": t,
                "ratio": t / row["db_time_ms"] if t else None,
                "kernels": sorted(names.get(key, [])) or None,
                "tactic": notes.get(key),
                "status": r.get("status") if r else "no_result",
                "error": (r or {}).get("error"),
            })
    rows_out.sort(key=lambda o: (o["config"], o["db_time_ms"]))
    with open(measured / "kernels.jsonl", "w") as f:
        for o in rows_out:
            f.write(json.dumps(o) + "\n")

    by = defaultdict(list)
    for o in rows_out:
        by[o["config"]].append(o)
    lines = [
        "# Kernel rows re-measured in pp-impl/.venv (2_kernels.md §8)",
        "",
        "Every row of the configurations below, measured by the ServingStudioSim profiler's own runner",
        "(`launcher kernel-profile run --fresh`, cold L2 as the rows, nothing written to any DB) with the worker",
        "executing in `pp-impl/.venv` (`bench/kernels/ppimpl_profile.py`). Ratio = real ÷ `db_time_ms`.",
        "How it was run: `kernels_run.md`. Per row: `kernels.jsonl`.",
        "",
        "| # | Kind / backend | Fixed configuration | Rows | Measured | Median ratio | Min ratio | Max ratio | Rows > 1.10 | Rows < 0.90 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    slow = []
    for c in sorted(by):
        os_ = by[c]
        kind, backend, fixed = CONFIGS[c]
        rs = [o["ratio"] for o in os_ if o["ratio"] is not None]
        hi = [o for o in os_ if o["ratio"] is not None and o["ratio"] > 1 + TOL]
        lo = [o for o in os_ if o["ratio"] is not None and o["ratio"] < 1 - TOL]
        fx = ", ".join(f"{k} {v}" for k, v in fixed.items()) or "—"
        if rs:
            lines.append(f"| {c} | `{kind}` / `{backend}` | {fx} | {len(os_)} | {len(rs)} | "
                         f"{statistics.median(rs):.3f} | {min(rs):.3f} | {max(rs):.3f} | {len(hi)} | {len(lo)} |")
        else:
            lines.append(f"| {c} | `{kind}` / `{backend}` | {fx} | {len(os_)} | 0 | — | — | — | — | — |")
        if hi:
            slow.append((c, kind, backend, hi))
    lines += ["", "## Rows above 1.10", ""]
    if not slow:
        lines.append("None.")
    for c, kind, backend, hi in slow:
        lines.append(f"**{c}. `{kind}` / `{backend}`**: {len(hi)} rows. Largest ratios:")
        for o in sorted(hi, key=lambda o: -o["ratio"])[:8]:
            s = {k: v for k, v in o["spec"].items() if k != "per_expert_batches"}
            lines.append(f"- {o['ratio']:.3f}: DB {o['db_time_ms']:.4g} ms, real {o['real_time_ms']:.4g} ms, `{json.dumps(s)}`")
        lines.append("")
    missing = [o for o in rows_out if o["ratio"] is None]
    if missing:
        lines += ["## Not measured", ""]
        for o in missing[:50]:
            lines.append(f"- {o['config']} `{o['kind']}` / `{o['backend']}`: {o['status']} {o['error'] or ''}"[:300])
    (measured / "kernels.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows_out)} rows, {sum(o['ratio'] is not None for o in rows_out)} measured")


if __name__ == "__main__":
    main(*sys.argv[1:])
