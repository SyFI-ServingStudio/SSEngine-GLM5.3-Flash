"""Summarize the correctness reference: one table row per request.

For each measured/runs/reference_one_<L>/ it joins
  proxy.jsonl     what req-frontend actually sent and got back (through the recording proxy),
  prompts.jsonl   the prompts rebuilt in Python from enwik9 (make_prompts.py),
  logprobs.jsonl  a second, non-streamed greedy pass with the top-5 logprobs,
and checks that the three agree on the prompt (sha256 of the int32 token ids) and on the output token.

usage: python summarize.py <runs_dir> > reference_table.md  (also writes <runs_dir>/../reference.jsonl)
"""

import json
import sys
from pathlib import Path

LENGTHS = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


def load(path, key):
    out = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            out[r[key].replace("independent_", "")] = r
    return out


def main():
    runs = Path(sys.argv[1])
    rows, bad = [], 0
    for L in LENGTHS:
        d = runs / f"reference_one_{L}"
        proxy = load(d / "proxy.jsonl", "request_id")
        prompts = load(d / "prompts.jsonl", "id")
        lps = load(d / "logprobs.jsonl", "request_id")
        for rid in sorted(prompts, key=lambda k: prompts[k]["ordinal"]):
            p, x, lp = prompts[rid], proxy.get(rid), lps.get(rid)
            top = (lp or {}).get("top_logprobs") or []
            row = {
                "length": L,
                "id": rid,
                "prompt_len": p["prompt_len"],
                "prompt_sha256": p["prompt_sha256"],
                "prompt_sent_matches": bool(x) and x["prompt_sha256"] == p["prompt_sha256"],
                "output_token_id": (x or {}).get("output_token_ids", [None])[0],
                "logprob_pass_token_id": (lp or {}).get("token_id"),
                "top5": [[t["token_id"], round(t["logprob"], 4)] for t in top[:5]],
            }
            row["margin"] = round(top[0]["logprob"] - top[1]["logprob"], 4) if len(top) > 1 else None
            row["agree"] = (row["prompt_sent_matches"] and row["output_token_id"] is not None
                            and row["output_token_id"] == row["logprob_pass_token_id"]
                            and bool(top) and top[0]["token_id"] == row["output_token_id"])
            bad += not row["agree"]
            rows.append(row)
    with open(runs.parent / "reference.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("| Length | Request | Prompt tokens | Prompt sha256 (first 16) | Output token | Top-2 logprobs | Margin | Checks |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        t2 = ", ".join(f"{t} ({lp})" for t, lp in r["top5"][:2])
        print(f"| {r['length']} | `{r['id']}` | {r['prompt_len']:,} | `{r['prompt_sha256'][:16]}` | "
              f"{r['output_token_id']} | {t2} | {r['margin']} | {'ok' if r['agree'] else '**mismatch**'} |")
    print(f"\n{len(rows)} requests, {bad} with a disagreement.", file=sys.stderr)


if __name__ == "__main__":
    main()
