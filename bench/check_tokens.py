"""M3 output check: the engine's first tokens against the correctness reference (measured/reference.md).

Reads <run dir>/server_requests.jsonl, one line per request with at least `id` and `token` (ppimpl/runtime/controller.py),
and measured/reference.jsonl (bench/reference/summarize.py). Each request is matched to its reference prompt:
  - by `prompt_sha256` when the line has one (sha256 of the prompt's token ids as little-endian int32, as in reference.jsonl);
  - otherwise by request id: req-frontend's `independent_<row id>` with any `#k` dedup suffix dropped, and then the
    line's `prompt_tokens` must equal the reference prompt length.
A request passes if its token equals the reference token or, where the reference's top-1 logprob is tied
(`tied_top_tokens`), any of the tied tokens. Requests whose prompt is not in the reference are counted, not judged.

Writes <run dir>/token_check.json and prints a summary. Exit status: 0 if at least one request was judged and none
failed, 1 otherwise.

usage: python check_tokens.py <run dir> [--reference measured/reference.jsonl]
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_REFERENCE = Path(__file__).resolve().parent.parent / "measured" / "reference.jsonl"


def row_id(request_id: str) -> str:
    rid = re.sub(r"#\d+$", "", request_id)
    return rid[len("independent_"):] if rid.startswith("independent_") else rid


def check(run_dir: Path, reference: Path) -> dict:
    refs = [json.loads(line) for line in open(reference)]
    by_sha = {r["prompt_sha256"]: r for r in refs}
    by_id = {r["id"]: r for r in refs}
    results, counts = [], {"pass": 0, "fail": 0, "no_reference": 0, "no_token": 0}
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
    summary = {"run_dir": str(run_dir), "reference": str(reference), **counts,
               "judged": counts["pass"] + counts["fail"] + counts["no_token"],
               "ok": counts["fail"] == 0 and counts["no_token"] == 0 and counts["pass"] > 0}
    (run_dir / "token_check.json").write_text(json.dumps({"summary": summary, "requests": results}, indent=1) + "\n")
    return {"summary": summary, "requests": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    a = ap.parse_args(argv)
    res = check(a.run_dir, a.reference)
    s = res["summary"]
    print(f"{s['run_dir']}: {s['pass']} pass, {s['fail']} fail, {s['no_token']} without a token, "
          f"{s['no_reference']} not in the reference -> {'OK' if s['ok'] else 'FAIL'}")
    for r in res["requests"]:
        if r["verdict"] in ("fail", "no_token"):
            print(f"  {r['verdict']}: {r['id']} token {r['token']}, reference {r.get('reference_token')} "
                  f"(top-5 {r.get('reference_top5')})")
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
