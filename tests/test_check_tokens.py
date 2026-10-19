"""CPU test of benchmarks/check_tokens.py on a stub run dir and a stub reference.

run: python -m pytest tests/test_check_tokens.py
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from benchmarks import check_tokens  # noqa: E402

REAL_REFERENCE = HERE.parent / "benchmarks" / "data" / "reference.jsonl"


def sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def ref_row(rid, ids, token, top5, tied=None):
    return {"id": rid, "prompt_len": len(ids), "prompt_sha256": sha(ids), "logprob_pass_token_id": token,
            "top5": top5, "tied_top_tokens": tied}


def write(d, refs, reqs):
    (d / "run").mkdir()
    with open(d / "ref.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in refs)
    with open(d / "run" / "server_requests.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in reqs)


def test_stub_run():
    a, b, c, d_ = [1, 2, 3], [4, 5, 6, 7], [8, 9], [10]
    refs = [ref_row("one_1k_0", a, 29, [[29, -0.1], [5, -3.0]]),
            ref_row("one_1k_1", b, 7, [[7, -0.7], [9, -0.7]], tied=[7, 9]),
            ref_row("one_1k_2", c, 11, [[11, -0.2], [12, -2.0]]),
            ref_row("one_1k_3", d_, 13, [[13, -0.1], [14, -4.0]])]
    reqs = [{"id": "independent_one_1k_0", "prompt_tokens": 3, "token": 29},            # pass, by id
            {"id": "x", "prompt_sha256": sha(b), "prompt_tokens": 4, "token": 9},      # pass, by sha, tied token
            {"id": "independent_one_1k_2#1", "prompt_tokens": 2, "token": 12},         # fail: second choice
            {"id": "independent_one_1k_3", "prompt_tokens": 5, "token": 13},           # length differs: not judged
            {"id": "independent_other_0", "prompt_tokens": 3, "token": 1},             # not in the reference
            {"id": "independent_one_1k_0", "prompt_tokens": 3, "token": None}]         # no token
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        write(t, refs, reqs)
        res = check_tokens.check(t / "run", t / "ref.jsonl", min_margin=0)   # judge every reference
        v = [r["verdict"] for r in res["requests"]]
        assert v == ["pass", "pass", "fail", "no_reference", "no_reference", "no_token"], v
        assert res["requests"][1]["matched_by"] == "sha"
        assert res["requests"][2]["engine_token_rank_in_reference_top5"] == 2
        assert not res["summary"]["ok"]
        assert (t / "run" / "token_check.json").exists()
        # The CLI exits 1 on a failure and 0 once only passing requests remain.
        cli = [sys.executable, "-m", "benchmarks.check_tokens", str(t / "run"),
               "--reference", str(t / "ref.jsonl"), "--min-margin", "0"]
        assert subprocess.run(cli, capture_output=True, cwd=HERE.parent).returncode == 1
        write_ok = [r for r in reqs[:2]]
        (t / "run" / "server_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in write_ok))
        assert subprocess.run(cli, capture_output=True, cwd=HERE.parent).returncode == 0


def test_real_reference_self_consistent():
    """A stub run that answers every reference prompt with its reference token passes; a wrong token fails."""
    refs = [json.loads(line) for line in open(REAL_REFERENCE)]
    assert len(refs) == 40
    with tempfile.TemporaryDirectory() as t:
        run = Path(t)
        reqs = [{"id": f"independent_{r['id']}", "prompt_tokens": r["prompt_len"], "token": r["logprob_pass_token_id"]}
                for r in refs]
        (run / "server_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
        s = check_tokens.check(run, REAL_REFERENCE, min_margin=0)["summary"]
        assert (s["pass"], s["fail"], s["ok"]) == (40, 0, True)
        s = check_tokens.check(run, REAL_REFERENCE)["summary"]
        assert s["pass"] + s["near_tie"] == 40 and s["fail"] == 0 and s["ok"]
        tie = next(r for r in refs if r.get("tied_top_tokens"))
        other = [x for x in tie["tied_top_tokens"] if x != tie["logprob_pass_token_id"]][0]
        reqs = [{"id": f"independent_{tie['id']}", "prompt_tokens": tie["prompt_len"], "token": other},
                {"id": f"independent_{refs[0]['id']}", "prompt_tokens": refs[0]["prompt_len"],
                 "token": refs[0]["logprob_pass_token_id"] + 1}]
        (run / "server_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
        v = [r["verdict"] for r in check_tokens.check(run, REAL_REFERENCE, min_margin=0)["requests"]]
        assert v == ["pass", "fail"], v
        v = [r["verdict"] for r in check_tokens.check(run, REAL_REFERENCE)["requests"]]
        assert v == ["near_tie", "fail"], v   # the tied reference is a near tie by default


if __name__ == "__main__":
    test_stub_run()
    test_real_reference_self_consistent()
    test_near_ties_not_judged()
    print("ok")


def test_near_ties_not_judged():
    """Default --min-margin 3: a reference below it is reported as a near tie, and its miss does not fail the run;
    a request without a token still fails it, whatever the margin."""
    a, b, c = [1, 2, 3], [4, 5], [6]
    refs = [ref_row("one_1k_0", a, 29, [[29, -0.1], [5, -5.0]]),       # margin 4.9: judged
            ref_row("one_1k_3", b, 2669, [[2669, -0.19], [11490, -2.19]]),  # margin 2.0: near tie
            ref_row("one_1k_4", c, 7, [[7, -0.1], [8, -1.0]])]           # margin 0.9: near tie
    reqs = [{"id": "x0", "prompt_sha256": sha(a), "token": 29},
            {"id": "x3", "prompt_sha256": sha(b), "token": 11490},
            {"id": "x4", "prompt_sha256": sha(c), "token": 7}]
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        write(t, refs, reqs)
        res = check_tokens.check(t / "run", t / "ref.jsonl")
        assert [r["verdict"] for r in res["requests"]] == ["pass", "near_tie", "near_tie"]
        assert [r.get("matches") for r in res["requests"][1:]] == [False, True]
        s = res["summary"]
        assert (s["pass"], s["fail"], s["near_tie"], s["judged"], s["ok"]) == (1, 0, 2, 1, True)
        assert check_tokens.main([str(t / "run"), "--reference", str(t / "ref.jsonl")]) == 0
        assert check_tokens.main([str(t / "run"), "--reference", str(t / "ref.jsonl"), "--min-margin", "0"]) == 1
        reqs[2]["token"] = None
        (t / "run" / "server_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
        s = check_tokens.check(t / "run", t / "ref.jsonl")["summary"]
        assert (s["no_token"], s["ok"]) == (1, False)
