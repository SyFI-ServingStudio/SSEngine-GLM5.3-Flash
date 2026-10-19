"""Cut a compressed trace into parts by round-0 prompt length P, each part to replay on its own fresh server.

Each part keeps its sessions' order and the gaps between them, with its first session at t = 0. Session ids stay
those of the full trace, so a part's sessions are found in it by id. Prints each part's wall time: its last session's
start plus that session's duration (its rounds' TTFTs from the targets json plus its tool waits).

usage: python split_by_p.py <compressed csv> <targets json> <P upper bound> [<P upper bound> ...]
  e.g. split_by_p.py prefix_sessions_ext_w1s_compressed.csv prefix_targets_ext_w1s.json 8192 65536 262144 524288
  writes <stem>_p1.csv ... next to the input: part i holds the sessions with bound[i-1] < P <= bound[i].
"""
import csv, json, sys
from pathlib import Path

src, targets, bounds = Path(sys.argv[1]), Path(sys.argv[2]), [int(x) for x in sys.argv[3:]]
rows = list(csv.DictReader(open(src)))
header = list(rows[0].keys())
sessions: dict[int, list[dict]] = {}
for r in rows:
    sessions.setdefault(int(r["session_id"]), []).append(r)
order = sorted(sessions, key=lambda s: float(sessions[s][0]["arrival_time_ms"]))
ttfts = {(t["prefix_tokens"], t["computed_tokens"], t.get("rounds", 2)):
         [t[f"round{k}"]["ttft_ms"] for k in range(t.get("rounds", 2))] for t in json.loads(targets.read_text())}


def duration_ms(s: int) -> float:
    r = sessions[s]
    key = (int(r[0]["input_len"]), int(r[1]["input_len"]) + 1, len(r))
    return sum(ttfts[key]) + sum(float(x["tool_wait_after_ms"]) for x in r[:-1])


assert max(int(sessions[s][0]["input_len"]) for s in order) <= bounds[-1], "the last bound must cover every P"
lo = 0
for n, hi in enumerate(bounds, 1):
    part = [s for s in order if lo < int(sessions[s][0]["input_len"]) <= hi]
    assert part, (lo, hi)
    t0 = float(sessions[part[0]][0]["arrival_time_ms"])
    out = src.with_name(f"{src.stem}_p{n}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in part:
            for r in sessions[s]:
                w.writerow([f"{float(r[h]) - t0:.6f}" if h == "arrival_time_ms" else r[h] for h in header])
    last = part[-1]
    end = float(sessions[last][0]["arrival_time_ms"]) - t0 + duration_ms(last)
    ps = sorted({int(sessions[s][0]["input_len"]) for s in part})
    print(f"{out.name}: P {ps[0]}..{ps[-1]}, {len(part)} sessions, ends at {end / 1e3:.1f} s")
    lo = hi
