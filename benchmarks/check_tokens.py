"""M3 output check: the engine's first tokens against the correctness reference (research/measured/reference.md).

Reads <run dir>/server_requests.jsonl, one line per request with at least `id` and `token` (ssengine/runtime/controller.py),
and benchmarks/data/reference.jsonl (research/bench/reference/summarize.py). Each request is matched to its reference prompt:
  - by `prompt_sha256` when the line has one (sha256 of the prompt's token ids as little-endian int32, as in reference.jsonl);
  - otherwise by request id: req-frontend's `independent_<row id>` with any `#k` dedup suffix dropped, and then the
    line's `prompt_tokens` must equal the reference prompt length.
A request passes if its token equals the reference token or, where the reference's top-1 logprob is tied
(`tied_top_tokens`), any of the tied tokens. Requests whose prompt is not in the reference are counted, not judged.
Near ties are reported, not judged: a reference whose top-1 margin (top-1 minus top-2 logprob) is below
--min-margin (3 nats). Batching changes the summation order of mHC split-K, and with NVFP4 activations that flips
near ties: vLLM TP4+EP itself returns 11490 for one_1k_3 (margin 2.0) when it batches 8 x 1k (jobs 7170-7174).
A request without a token always counts.

Writes <run dir>/token_check.json and prints a summary. Exit status: 0 if at least one request was judged and none
failed, 1 otherwise.

usage: python -m benchmarks.check_tokens <run dir> [--reference benchmarks/data/reference.jsonl] [--min-margin 3]
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_REFERENCE = Path(__file__).resolve().parent / "data" / "reference.jsonl"
MIN_MARGIN = 3.0   # nats


def ref_margin(ref: dict) -> float:
    """Top-1 minus top-2 logprob of the reference (0 when tied)."""
    if ref.get("margin") is not None:
        return float(ref["margin"])
    top = ref.get("top5") or []
    return float(top[0][1] - top[1][1]) if len(top) > 1 else float("inf")


def row_id(request_id: str) -> str:
    rid = re.sub(r"#\d+$", "", request_id)
    return rid[len("independent_"):] if rid.startswith("independent_") else rid


def check(run_dir: Path, reference: Path, min_margin: float = MIN_MARGIN) -> dict:
    refs = [json.loads(line) for line in open(reference)]
    by_sha = {r["prompt_sha256"]: r for r in refs}
    by_id = {r["id"]: r for r in refs}
    results, counts = [], {"pass": 0, "fail": 0, "no_reference": 0, "no_token": 0, "near_tie": 0}
    for line in open(run_dir / "server_requests.jsonl"):
        if not line.strip():
            continue
        req = json.loads(line)
        sha = req.get("prompt_sha256")
        ref = by_sha.get(sha) if sha else by_id.get(row_id(str(req.get("id", ""))))
        out = {"id": req.get("id"), "token": req.get("token"), "matched_by": "sha" if sha else "id"}
        if ref is None:
            out["verdict"] = "no_reference"
        elif not sha and req.get("prompt_tokens") is not None and req["prompt_tokens"] != ref["prompt_len"]:
            out.update(reference_id=ref["id"], verdict="no_reference",
                       why=f"prompt_tokens {req['prompt_tokens']} != reference {ref['prompt_len']}")
        elif req.get("token") is None:
            out.update(reference_id=ref["id"], verdict="no_token")
        elif ref_margin(ref) < min_margin:
            out.update(reference_id=ref["id"], reference_token=ref["logprob_pass_token_id"],
                       reference_margin=ref_margin(ref), matches=req["token"] == ref["logprob_pass_token_id"],
                       verdict="near_tie")
        else:
            ok = {ref["logprob_pass_token_id"], *(ref.get("tied_top_tokens") or [])}
            out.update(reference_id=ref["id"], reference_token=ref["logprob_pass_token_id"],
                       accepted=sorted(ok), verdict="pass" if req["token"] in ok else "fail")
            if out["verdict"] == "fail":
                out["reference_top5"] = ref.get("top5")
                rank = [t for t, _ in ref.get("top5") or []]
                out["engine_token_rank_in_reference_top5"] = rank.index(req["token"]) + 1 if req["token"] in rank else None
        counts[out["verdict"]] += 1
        results.append(out)
    summary = {"run_dir": str(run_dir), "reference": str(reference), "min_margin": min_margin, **counts,
               "judged": counts["pass"] + counts["fail"] + counts["no_token"],
               "ok": counts["fail"] == 0 and counts["no_token"] == 0 and counts["pass"] > 0}
    (run_dir / "token_check.json").write_text(json.dumps({"summary": summary, "requests": results}, indent=1) + "\n")
    return {"summary": summary, "requests": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN,
                    help="judge only references whose top-1 margin is at least this (nats)")
    a = ap.parse_args(argv)
    res = check(a.run_dir, a.reference, a.min_margin)
    s = res["summary"]
    print(f"{s['run_dir']}: {s['pass']} pass, {s['fail']} fail, {s['no_token']} without a token, "
          f"{s['near_tie']} near tie (margin < {s['min_margin']:g}, not judged), "
          f"{s['no_reference']} not in the reference -> {'OK' if s['ok'] else 'FAIL'}")
    for r in res["requests"]:
        if r["verdict"] in ("fail", "no_token"):
            print(f"  {r['verdict']}: {r['id']} token {r['token']}, reference {r.get('reference_token')} "
                  f"(top-5 {r.get('reference_top5')})")
        elif r["verdict"] == "near_tie" and not r["matches"]:
            print(f"  near tie, not judged: {r['id']} token {r['token']}, reference {r['reference_token']} "
                  f"(margin {r['reference_margin']:g})")
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
