"""Markdown comparison table for the vLLM DP4/EP4 runs (each run's dpep_summary.json) next to the sim targets
(sim_targets_v3.txt from benchmarks.vllm_dpep_simwin). Windows: 0-20 = first token in [0, 20) min (handoff metrics.py);
10-20 = first token in [10, 20) min (the sim rows' window). HBM/CPU/compute shares are vLLM's prompt_tokens_by_source
(10-20 column: counter delta from the snapshot nearest 600 s to the end). full-miss = rounds > 0 with 0 cached tokens.

    python -m benchmarks.vllm_dpep_table runs/vllm_dpep [run ...]    # prints markdown
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def full_miss(run: Path) -> float:
    n = z = 0
    for line in (run / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["source"]["data"]["round_idx"] == 0 or str(r["outcome"].get("status")).upper() != "SUCCESS":
            continue
        n += 1
        z += not ((r["outcome"].get("server_usage") or {}).get("cached_prompt_tokens") or 0)
    return z / max(n, 1)


def vllm_rows(run: Path) -> list[list[str]]:
    s = json.loads((run / "dpep_summary.json").read_text())
    e, ins = s["engines"], s["inside_rounds"]
    late = e.get("late", {})
    out = []
    for win, m, src, imb in (("0-20", s["metrics_0_20"], e.get("run_source_share", {}),
                              e.get("run_request_imbalance_max_over_mean")),
                             ("10-20", s["metrics_first_token_10_20"], e.get("late_source_share", {}),
                              e.get("late_request_imbalance_max_over_mean"))):
        t = m["ttft_s"]
        rec = (m["computed_tokens_per_s"] - m["new_tokens_per_s"]) / max(m["new_tokens_per_s"], 1)
        out.append([run.name, win, f"{m['rounds_per_s']:.2f}", f"{m['new_tokens_per_s'] / 1e3:.1f}k",
                    f"{m['computed_tokens_per_s'] / 1e3:.1f}k", f"{t['p50']:.2f}/{t['p90']:.2f}/{t['p99']:.2f}",
                    str(m["rounds_ttft_over_60s"]), f"{m['prefix_hit_share']:.3f}",
                    f"{src.get('local_cache_hit', 0):.3f}/{src.get('external_kv_transfer', 0):.3f}", f"{rec:.2f}",
                    f"{imb:.2f}" if imb else "-"])
    out[1].append("/".join(str(round(d.get("requests", 0))) for d in late.values()))
    out[0].append(f"full-miss {full_miss(run):.3f}; failed {s['rounds_failed']}/{s['rounds_sent']}")
    out[1][-1] += f"; inside {ins['inside']['rounds']} rounds, {ins['inside']['hit_tokens']} hit"
    return out


def sim_rows(path: Path) -> list[list[str]]:
    out = []
    for line in path.read_text().splitlines():
        name, js = line.split(" ", 1)
        d = json.loads(js)
        for win in ("0_20", "10_20"):
            m = d[win]
            rec = (m["comp_tok_s"] - m["new_tok_s"]) / max(m["new_tok_s"], 1)
            out.append([f"sim {name}", win.replace("_", "-"), f"{m['rps']:.2f}", f"{m['new_tok_s'] / 1e3:.1f}k",
                        f"{m['comp_tok_s'] / 1e3:.1f}k", "/".join(f"{x:.2f}" for x in m["ttft"]), str(m["over60"]),
                        f"{m['hit']:.3f}", "-", f"{rec:.2f}", "-", ""])
    return out


HEAD = ["run", "window", "r/s", "new tok/s", "computed tok/s", "TTFT p50/p90/p99 s", ">60 s", "hit share",
        "HBM/CPU share", "recompute/new", "req max/mean", "notes"]


def main(root: Path, runs: list[str]) -> str:
    rows = [r for name in runs for r in vllm_rows(root / name)]
    if (root / "sim_targets_v3.txt").exists():
        rows += sim_rows(root / "sim_targets_v3.txt")
    lines = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    lines += ["| " + " | ".join(r + [""] * (len(HEAD) - len(r))) + " |" for r in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    runs = sys.argv[2:] or sorted(p.name for p in root.iterdir() if (p / "dpep_summary.json").exists()
                                  and not p.name.startswith(("smoke", "rsmoke")))
    print(main(root, runs))
