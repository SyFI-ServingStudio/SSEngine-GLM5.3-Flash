"""Per-microbatch, per-stage GPU numbers from an Nsight Systems capture of `python -m ppimpl serve`.

  python bench/trace.py <run dir>        reads <run dir>/trace.sqlite (`nsys export --type sqlite`), writes trace.json

Attribution: each worker wraps a plan in the NVTX range `stage <s> mb <id>` (ppimpl/runtime/worker.py). A CUDA API
call made inside that range on the worker's thread (kernel launch, graph launch, NCCL's launch) carries a correlation
id; the kernels with that id (graph-node kernels share their graph launch's id) belong to that (stage, microbatch).

Per (microbatch, stage), times in ms, by the 4_iteration.md §6 GPU-trace definitions:
- `start_ms` / `end_ms`: first compute-kernel start / last compute-kernel end (NCCL kernels excluded), from the first
  traced stage-0 kernel; `compute_ms` = end - start; `busy_ms`: union of the kernels; `gaps_ms` = compute - busy;
- `nccl_recv_ms` / `nccl_send_ms`: durations of the NCCL kernels launched before / after the compute launches;
- per microbatch: `sends` (stage s last kernel end -> stage s+1 first kernel start) and `gpu_span_ms`.
Per stage and kernel name: count and time over the microbatches complete on every stage, with the manifest's
configuration and layer positions (`ppimpl.kernels.check_trace`) when the manifest is importable.
"""

import bisect
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANGE = re.compile(r"^stage (\d+) mb (\d+)$")
THREAD_SPACE = 1 << 24          # Nsight global thread ids keep the OS thread id in the low 24 bits


def owning_pid(global_tid: int) -> int:
    return global_tid - global_tid % THREAD_SPACE


def merged_ns(intervals: list[tuple[int, int]]) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def is_nccl(name: str) -> bool:
    return "nccl" in name.lower()


def load(db: Path):
    """-> {(stage, mb): [(start, end, short name, demangled name, launch ns)]}, unattributed kernel count."""
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if t not in tables:
            raise SystemExit(f"{db}: no {t} table (was the capture taken with -t cuda,nvtx on a GPU run?)")
    strings = dict(con.execute("SELECT id, value FROM StringIds")) if "StringIds" in tables else {}

    ranges: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)   # globalTid -> (start, end, stage, mb)
    nvtx_cols = {r[1] for r in con.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in nvtx_cols else "text"
    for label, start, end, tid in con.execute(f"SELECT {text}, start, end, globalTid FROM NVTX_EVENTS "
                                              "WHERE end IS NOT NULL"):
        m = RANGE.match(label or "")
        if m:
            ranges[tid].append((start, end, int(m.group(1)), int(m.group(2))))
    starts = {}
    for tid in ranges:
        ranges[tid].sort()
        starts[tid] = [r[0] for r in ranges[tid]]

    corr: dict[tuple[int, int], tuple[int, int, int]] = {}                    # (pid, correlation) -> (stage, mb, launch)
    for start, tid, cid in con.execute("SELECT start, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        rs = ranges.get(tid)
        if not rs:
            continue
        i = bisect.bisect_right(starts[tid], start) - 1
        if i >= 0 and start <= rs[i][1]:
            corr[(owning_pid(tid), cid)] = (rs[i][2], rs[i][3], start)

    kcols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    short = "shortName" if "shortName" in kcols else "demangledName"
    out: dict[tuple[int, int], list] = defaultdict(list)
    unattributed = 0
    for start, end, pid, cid, sid, did in con.execute(
            f"SELECT start, end, globalPid, correlationId, {short}, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        hit = corr.get((pid, cid))
        if hit is None:
            unattributed += 1
            continue
        stage, mb, launch = hit
        out[(stage, mb)].append((start, end, strings.get(sid, str(sid)), strings.get(did, str(did)), launch))
    return out, unattributed


def stage_numbers(kernels: list) -> dict:
    comp = [k for k in kernels if not is_nccl(k[2])]
    nccl = [k for k in kernels if is_nccl(k[2])]
    if not comp:
        return {}
    first_launch = min(k[4] for k in comp)
    start, end = min(k[0] for k in comp), max(k[1] for k in comp)
    busy = merged_ns([(k[0], k[1]) for k in comp])
    recv = [k for k in nccl if k[4] < first_launch]
    send = [k for k in nccl if k[4] >= first_launch]
    return {"start_ns": start, "end_ns": end, "compute_ms": (end - start) / 1e6, "busy_ms": busy / 1e6,
            "gaps_ms": (end - start - busy) / 1e6, "kernels": len(comp),
            "nccl_recv_ms": sum(k[1] - k[0] for k in recv) / 1e6 if recv else None,
            "nccl_send_ms": sum(k[1] - k[0] for k in send) / 1e6 if send else None}


def manifest_class(stage: int, demangled: str) -> dict:
    try:
        from ppimpl import kernels as manifest
    except ImportError:
        return {}
    tc = manifest.check_trace([demangled], stage)
    if tc.matched:
        cfgs = [manifest.by_config(c) for c in sorted(tc.matched)]
        return {"class": "manifest", "configs": [c.config for c in cfgs],
                "positions": sorted({f"{lt}.{p}" for c in cfgs for lt in c.layer_types for p in c.positions})}
    if tc.forbidden:
        return {"class": "forbidden", "configs": [c for c, _ in tc.forbidden]}
    return {"class": "glue" if tc.glue else "unknown"}


def analyze(db: Path) -> dict:
    per, unattributed = load(db)
    nstages = 1 + max((s for s, _ in per), default=-1)
    mbs = sorted({mb for _, mb in per})
    complete = [mb for mb in mbs if all(per.get((s, mb)) and stage_numbers(per[(s, mb)]) for s in range(nstages))]
    origin = min((stage_numbers(per[(0, mb)])["start_ns"] for mb in complete), default=0)
    rel = lambda ns: (ns - origin) / 1e6  # noqa: E731
    out_mbs = {}
    for mb in complete:
        st = []
        for s in range(nstages):
            n = stage_numbers(per[(s, mb)])
            st.append({"stage": s, "start_ms": rel(n.pop("start_ns")), "end_ms": rel(n.pop("end_ns")), **n})
        out_mbs[str(mb)] = {
            "stages": st,
            "sends": [{"after_stage": s, "ms": st[s + 1]["start_ms"] - st[s]["end_ms"]} for s in range(nstages - 1)],
            "gpu_span_ms": st[-1]["end_ms"] - st[0]["start_ms"]}
    s0 = [out_mbs[str(mb)]["stages"][0] for mb in complete]
    s0_gaps = [b["start_ms"] - a["end_ms"] for a, b, ma, mb_ in zip(s0, s0[1:], complete, complete[1:])
               if mb_ == ma + 1]

    agg: dict[tuple[int, str], dict] = {}
    for mb in complete:
        for s in range(nstages):
            for k in per[(s, mb)]:
                a = agg.setdefault((s, k[2]), {"stage": s, "name": k[2], "demangled": k[3][:300], "count": 0,
                                               "total_ms": 0.0})
                a["count"] += 1
                a["total_ms"] += (k[1] - k[0]) / 1e6
    kernels = []
    for a in sorted(agg.values(), key=lambda a: (a["stage"], -a["total_ms"])):
        a["per_mb_ms"] = a["total_ms"] / max(len(complete), 1)
        a["mean_ms"] = a["total_ms"] / a["count"]
        a.update(manifest_class(a["stage"], a["demangled"]))
        kernels.append(a)
    return {"source": str(db), "stages": nstages, "microbatches": len(complete),
            "partial_microbatches": len(mbs) - len(complete), "unattributed_kernels": unattributed,
            "mbs": out_mbs, "stage0_gap_ms": s0_gaps, "kernels": kernels}


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1])
    sys.path.insert(0, str(ROOT))
    res = analyze(run_dir / "trace.sqlite")
    (run_dir / "trace.json").write_text(json.dumps(res) + "\n")
    print(f"{run_dir.name}: {res['microbatches']} microbatches on all {res['stages']} stages "
          f"({res['partial_microbatches']} partial), {res['unattributed_kernels']} kernels outside the mb ranges")


if __name__ == "__main__":
    main()
