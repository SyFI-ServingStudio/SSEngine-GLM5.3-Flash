"""Write the extended prefix bring-up traces (req-frontend `text-generation-session-execution-v2`) beside this file.

The sessions have the shape of the handoff's prefix_sessions.csv (data/bringup/make_bringup.py, 5_prefix_cache.md
§1): round 0 is a prompt of P tokens; each later round, 10 s after the previous round's token, sends the previous
round's prompt, that token and A - 1 new tokens, so the previous prompt hits the cache and A tokens are computed.
Round k (k >= 1) hits P + (k - 1) * A tokens and computes A. Only one session is active at a time.

  prefix_sessions_ext.csv     two-round sessions. P in {2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k} plus the
                              unaligned P = 3,000 and 100,003 (resumes that start off a multiple of 4 and of a
                              block), A in {256, 1k, 8k}; 5 sessions per (P, A), 165 sessions, one every 30 s
                              (the longest, 512k + 8k, ends 14.7 s after it starts). It contains the handoff's 40
                              sessions' (P, A) cases: P in {8k, 32k, 128k, 512k} x A in {1k, 8k}.
  prefix_multiround_ext.csv   four-round sessions (rounds 0-3). P in {8k, 128k, 512k}, A in {256, 1k, 8k}; 5
                              sessions per (P, A), 45 sessions, one every 60 s (the longest, 512k + 3 x 8k, ends
                              35.3 s after it starts).

`arrival_time_ms` is the session's start, in ms from the start of replay; every round of a session carries it, and
the client sends round k when round k-1's token is back plus `tool_wait_after_ms` of round k-1.

Each file also comes as `*_w1s.csv`, with 1 s instead of 10 s between a round's token and the next round (the
handoff waits 10 s; the wait is only idle GPU time, and the GPU replay uses 1 s). Same slots.

The compressed versions (each session right after the previous one ends, plus 1 s) need the sessions' durations,
so prefix_targets_ext.py writes them from the simulator run; split_by_p.py cuts them into parts.

usage: python make_prefix_ext.py
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
K = 1024
TWO_ROUND = dict(prefixes=[2 * K, 3000, 4 * K, 8 * K, 16 * K, 32 * K, 64 * K, 100_003, 128 * K, 256 * K, 512 * K],
                 appends=[256, 1 * K, 8 * K], rounds=2, slot_ms=30_000)
MULTI_ROUND = dict(prefixes=[8 * K, 128 * K, 512 * K], appends=[256, 1 * K, 8 * K], rounds=4, slot_ms=60_000)
SESSIONS_PER_CASE = 5
HEADER = ["request_id", "session_id", "round_idx", "arrival_time_ms", "prefix_len", "input_len", "output_len",
          "tool_wait_after_ms"]


def session_rows(session: int, at_ms: float, prefix: int, append: int, rounds: int, wait_ms: int) -> list[list]:
    rows, prompt = [], prefix
    for k in range(rounds):
        wait = f"{wait_ms if k < rounds - 1 else 0:.6f}"
        if k == 0:
            rows.append([f"session_{session}_round_{k:06d}", session, k, f"{at_ms:.6f}", 0, prefix, 1, wait])
        else:
            rows.append([f"session_{session}_round_{k:06d}", session, k, f"{at_ms:.6f}", prompt + 1, append - 1, 1,
                         wait])
            prompt += append
    return rows


def write(name: str, prefixes: list[int], appends: list[int], rounds: int, slot_ms: int, wait_ms: int) -> None:
    with open(HERE / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        session = 0
        for prefix in prefixes:
            for append in appends:
                for _ in range(SESSIONS_PER_CASE):
                    w.writerows(session_rows(session, slot_ms * session, prefix, append, rounds, wait_ms))
                    session += 1
    print(HERE / name, session, "sessions")


if __name__ == "__main__":
    for suffix, wait_ms in (("", 10_000), ("_w1s", 1_000)):
        write(f"prefix_sessions_ext{suffix}.csv", **TWO_ROUND, wait_ms=wait_ms)
        write(f"prefix_multiround_ext{suffix}.csv", **MULTI_ROUND, wait_ms=wait_ms)
