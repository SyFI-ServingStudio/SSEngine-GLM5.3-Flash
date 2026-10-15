"""Stratified node-check sample of the handoff's kernel rows (2_kernels.md §4).

For each of the 35 configurations (kind, backend, fixed configuration) of
data/kernel_specs.jsonl, take the smallest, median and largest row, ordered by
db_time_ms. Writes one JSONL of specs per (kind, backend) -- one profiler CLI
call each -- plus sample.jsonl with the full rows.

usage: python sample_specs.py <kernel_specs.jsonl> <out_dir>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Fields that fix a configuration inside a (kind, backend) that holds several
# (2_kernels.md §2): GEMM weight shapes and the quant hidden size / scale layout.
FIXED_FIELDS = {
    "single_gemm": ("n", "k", "dtype"),
    "gemm_fp32_output": ("n", "k", "input_dtype"),
    "nvfp4_quant": ("hidden_size", "scale_format"),
}


def cli_spec(spec):
    """The CLI takes tuple args as JSON lists; the export stores them as text."""
    out = dict(spec)
    if isinstance(out.get("per_expert_batches"), str):
        out["per_expert_batches"] = json.loads(out["per_expert_batches"])
    return out


def config_key(row):
    fixed = FIXED_FIELDS.get(row["kind"], ())
    return (row["kind"], row["backend"], tuple((f, row["spec"][f]) for f in fixed))


def main(src, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for line in open(src):
        row = json.loads(line)
        groups[config_key(row)].append(row)
    picked = []
    for key in sorted(groups, key=str):
        rows = sorted(groups[key], key=lambda r: r["db_time_ms"])
        n = len(rows)
        for label, i in (("smallest", 0), ("median", n // 2), ("largest", n - 1)):
            picked.append({"config": [key[0], key[1], dict(key[2])], "rank": label,
                           "config_rows": n, **rows[i]})
    by_kb = defaultdict(list)
    for p in picked:
        by_kb[(p["kind"], p["backend"])].append(p)
    with open(out / "sample.jsonl", "w") as f:
        for p in picked:
            f.write(json.dumps(p) + "\n")
    with open(out / "calls.tsv", "w") as f:
        for (kind, backend), ps in sorted(by_kb.items()):
            path = out / f"specs__{kind}__{backend}.jsonl"
            with open(path, "w") as g:
                for p in ps:
                    g.write(json.dumps(cli_spec(p["spec"])) + "\n")
            f.write(f"{kind}\t{backend}\t{path.name}\t{len(ps)}\n")
    print(f"{len(groups)} configurations, {len(picked)} specs, {len(by_kb)} calls")


if __name__ == "__main__":
    main(*sys.argv[1:])
