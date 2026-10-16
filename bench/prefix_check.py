"""Round-1 tokens with the prefix cache against the same prompts computed without it (5_prefix_cache.md §5 "Check the
output"), through the real server.

  python -m bench.prefix_check [--out measured/runs_m4] [--fake --cpu] [--sessions 0,5,10] [--p-shift 1,2,3 --name N]

The bring-up sessions (data/bringup/prefix_sessions.csv), one after another, each round sent as soon as the previous
answer is back, on a fresh server with `keep_sessions` (prefix_check_on/) and on one without (prefix_check_off/). The
prompts are req-frontend's for the same trace: session k's tokens start at offset 9,973 k of the enwik9 token pool
(req-frontend src/executor/session.rs, src/tokens.rs; the pool as bench/reference/make_prompts.py builds it); round 1
is round 0's prompt, round 0's real token, then A - 1 new tokens. req-frontend itself cannot drive the run without the
cache: its prefix preflight refuses a server that reports no hits.

Without the cache, round 1 computes its whole prompt in the same 8k chunks that round 0 computed, plus the last chunk
[P, P + A) that the cached round computes alone: the microbatches are the same, so the tokens should be identical,
and a mismatch points at the saved KDA state or the KV of the cached tokens. Each server's server_requests.jsonl
holds every prompt's sha256 and token; bench/prefix.py compares them (also with the bring-up run's round 1).

--p-shift 1,2,3 adds 1, 2, 3, 1, ... tokens to the sessions' P in file order (the trace's P are multiples of 4): round
1 then resumes inside a 4-token indexer pool (pp_patch `install_kpool_unaligned`). Its results go to <name>_on/,
<name>_off/ and <name>.json.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import yaml

from bench.common import ENWIK9, HANDOFF, ROOT, free_port_block
from bench.run import CHECKPOINT, PYTHON, REAL_FORM_LEAD_MS, step, stop, wait_health

STRIDE = 9_973



def rel(path: Path) -> str:
    """A path under the repo root as written into tracked files: relative to the root (bench/margin.case_file)."""
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sessions_of(trace: Path) -> list[dict]:
    """Per session (file order = req-frontend's ordinal): P (round 0 input) and A - 1 (round 1 input)."""
    out = {}
    for r in csv.DictReader(open(trace)):
        out.setdefault(r["session_id"], {})[int(r["round_idx"])] = int(r["input_len"])
    return [{"session": int(s), "ordinal": i, "P": rounds[0], "new": rounds[1]} for i, (s, rounds) in enumerate(out.items())]


def token_pool(need: int) -> np.ndarray:
    from tokenizers import Tokenizer

    from bench.reference.make_prompts import build_pool
    pool = build_pool(ENWIK9, Tokenizer.from_file(str(Path(CHECKPOINT) / "tokenizer.json")), need)
    assert len(pool) >= need, "pool shorter than the trace needs"
    return pool


def sha256(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def post(c: httpx.Client, rid: str, ids: list[int]) -> dict:
    t = time.monotonic()
    r = c.post("/inference/v1/generate", headers={"x-request-id": rid},
               json={"token_ids": ids, "sampling_params": {"max_tokens": 1, "temperature": 0}, "stream": False})
    r.raise_for_status()
    j = r.json()
    return {"token": j["choices"][0]["token_ids"][0], "cached": j["usage"]["prompt_tokens_details"]["cached_tokens"],
            "ttft_ms": round((time.monotonic() - t) * 1e3, 2)}


def run_mode(run_dir: Path, keep: bool, sess: list[dict], pool: np.ndarray, port: int, fake: bool, cpu: bool) -> list:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"model": "fake" if fake else "glm53", "device": "cpu" if cpu else "cuda", "run_dir": str(run_dir),
           "port": port, "keep_sessions": keep, "scheduler": {"mb_tokens": 8192}}
    if cpu:
        cfg["fake"] = {"hidden_width": 64}
    if not fake:
        cfg.update(checkpoint=str(CHECKPOINT), form_lead_ms=REAL_FORM_LEAD_MS)
    (run_dir / "server.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    uptime = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    meta = {"name": run_dir.name, "keep_sessions": keep, "uptime": uptime, "job_id": os.environ.get("SLURM_JOB_ID")}
    print(f"[{run_dir.name}] uptime: {uptime}", flush=True)
    serve = [PYTHON, "-m", "ppimpl", "serve", "--config", str(run_dir / "server.yaml")]
    rows = []
    with open(run_dir / "server.log", "w") as slog:
        server = subprocess.Popen(step(serve, None if cpu else 4), cwd=ROOT, stdout=slog, stderr=subprocess.STDOUT)
    try:
        meta["server_ready_s"] = round(wait_health(port, server, 1800), 1)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600) as c:
            for s in sess:
                start = s["ordinal"] * STRIDE
                r0 = pool[start:start + s["P"]].tolist()
                a = post(c, f"session_{s['session']}_round_000000", r0)
                r1 = r0 + [a["token"]] + pool[start + s["P"]:start + s["P"] + s["new"]].tolist()
                b = post(c, f"session_{s['session']}_round_000001", r1)
                rows.append({"session": s["session"], "P": s["P"], "A": s["new"] + 1,
                             "round0": {**a, "sha256": sha256(r0)}, "round1": {**b, "sha256": sha256(r1)}})
                print(f"[{run_dir.name}] session {s['session']} P {s['P']} A {s['new'] + 1}: round 0 {a}, "
                      f"round 1 {b}", flush=True)
        meta["uptime_after"] = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    finally:
        stop(server, port)
    (run_dir / "check_rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "measured/runs_m4")
    ap.add_argument("--trace", type=Path, default=HANDOFF / "data/bringup/prefix_sessions.csv")
    ap.add_argument("--sessions", help="comma-separated session ids (default: all)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--p-shift", help="comma-separated tokens added to P, cycled over the sessions")
    ap.add_argument("--name", default="prefix_check")
    ap.add_argument("--modes", default="on,off",
                    help="which servers to run; a mode left out is read from <name>_<mode>/check_rounds.jsonl")
    a = ap.parse_args()
    sess = sessions_of(a.trace)
    if a.sessions:
        keep = {int(x) for x in a.sessions.split(",")}
        sess = [s for s in sess if s["session"] in keep]
    if a.p_shift:
        shifts = [int(x) for x in a.p_shift.split(",")]
        for i, s in enumerate(sess):
            s["P"] += shifts[i % len(shifts)]
    need = max(s["ordinal"] * STRIDE + s["P"] + s["new"] for s in sess)
    t = time.time()
    pool = token_pool(need)
    print(f"token pool: {len(pool)} tokens in {time.time() - t:.0f} s", flush=True)
    job = os.environ.get("SLURM_JOB_ID")
    port = a.port or (20000 + int(job) % 120 * 100 + 50 if job else 18550)
    res = {}
    modes = a.modes.split(",")
    for i, keep in enumerate((True, False)):
        mode_dir = a.out / f"{a.name}_{'on' if keep else 'off'}"
        if ("on" if keep else "off") not in modes:
            res[keep] = [json.loads(x) for x in (mode_dir / "check_rounds.jsonl").read_text().splitlines()]
            assert [r["P"] for r in res[keep]] == [s["P"] for s in sess], f"{mode_dir}: other sessions"
            continue
        res[keep] = run_mode(a.out / f"{a.name}_{'on' if keep else 'off'}", keep, sess, pool,
                             free_port_block(port + 10 * i), a.fake, a.cpu)
    on, off = res[True], res[False]
    lines = []
    for x, y in zip(on, off):
        lines.append({"session": x["session"], "P": x["P"], "A": x["A"], "hit": x["round1"]["cached"],
                      "hit_is_P": x["round1"]["cached"] == x["P"], "off_cached": y["round1"]["cached"],
                      "round0_same": x["round0"]["token"] == y["round0"]["token"],
                      "same_prompt": x["round1"]["sha256"] == y["round1"]["sha256"],
                      "token_on": x["round1"]["token"], "token_off": y["round1"]["token"],
                      "match": x["round1"]["token"] == y["round1"]["token"]})
    summary = {"sessions": len(lines), "p_shift": a.p_shift, "hit_is_P": sum(x["hit_is_P"] for x in lines),
               "round1_match": sum(x["match"] for x in lines), "same_prompt": sum(x["same_prompt"] for x in lines),
               "lines": lines}
    cases = margin_cases(sess, on, off, pool, a.out / f"{a.name}_margins")
    summary["margin_cases"] = str(a.out / f"{a.name}_margins" / "cases.json") if cases else None
    (a.out / f"{a.name}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"prefix check: {summary['sessions']} sessions, hit == P {summary['hit_is_P']}, round-1 token equal "
          f"{summary['round1_match']}, same round-1 prompt {summary['same_prompt']}; "
          f"{len(cases)} margin cases", flush=True)
    if summary["hit_is_P"] != summary["sessions"]:
        return 1
    return 2 if cases else 0      # 2: token differences left for the margin rule (bench/margin.py --judge)


def margin_cases(sess: list[dict], on: list[dict], off: list[dict], pool: np.ndarray, out: Path) -> list[dict]:
    """Every token that differs between the servers, as bench/margin.py cases with the tokens each side returned:
    a round 0 (no cache on either side, same chunks) as one case; a round 1 with the same prompt as two, the cached
    side's chunks (round 0's, then [P, P + A)) and the uncached side's. The margin rule passes a difference when, in
    every case, each side's token is among the top 2 and the top-2 margin is below 3 nats (bench/check_tokens.py)."""
    from bench.margin import server_schedule

    cases = []
    for s, x, y in zip(sess, on, off):
        st = s["ordinal"] * STRIDE
        r0 = pool[st:st + s["P"]].astype(np.int32)
        name = f"s{s['session']}"
        if x["round0"]["token"] != y["round0"]["token"]:
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"{name}_r0.npy", r0)
            cases.append({"name": f"{name}_r0", "file": rel(out / f"{name}_r0.npy"),
                          "schedule": server_schedule(s["P"]),
                          "expect_top2": [x["round0"]["token"], y["round0"]["token"]]})
        elif x["round1"]["token"] != y["round1"]["token"]:
            out.mkdir(parents=True, exist_ok=True)
            r1 = np.concatenate([r0, [x["round0"]["token"]], pool[st + s["P"]:st + s["P"] + s["new"]]]).astype(np.int32)
            assert sha256(r1) == x["round1"]["sha256"] == y["round1"]["sha256"]
            np.save(out / f"{name}_r1.npy", r1)
            cases.append({"name": f"{name}_r1_on", "file": rel(out / f"{name}_r1.npy"),
                          "schedule": server_schedule(s["P"]) + [[s["P"], len(r1) - s["P"], True]],
                          "expect_top2": [x["round1"]["token"]]})
            cases.append({"name": f"{name}_r1_off", "file": rel(out / f"{name}_r1.npy"),
                          "schedule": server_schedule(len(r1)), "expect_top2": [y["round1"]["token"]]})
    if cases:
        (out / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
