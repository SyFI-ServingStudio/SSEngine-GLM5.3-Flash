"""Align nvidia-smi samples (research/bench/slurm/study.sbatch SAMPLE=1: gpu_samples_<job>.csv) with a run's microbatches.jsonl.

  python research/bench/clock_align.py <gpu_samples.csv> <run dir>

Stage s ran on the s-th GPU of run.json's `gpus` (CUDA_VISIBLE_DEVICES). microbatches.jsonl times are ms from the
run's t0 (t0.json, unix ns); nvidia-smi timestamps are local wall time. For each (stage, microbatch): the mean SM
clock, power and temperature of the samples inside its compute interval and the throttle reasons seen there. Prints
the first microbatches, deciles, and the fit of compute_ms against 1 / SM clock.
"""

import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REASONS = {0x1: "idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
           0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display"}


def reasons(mask: int) -> str:
    return "+".join(n for b, n in REASONS.items() if mask & b) or "none"


def load_samples(path: Path):
    out = defaultdict(list)   # gpu index -> [(unix ns, sm MHz, power W, temp C, reasons mask)]
    with open(path) as f:
        rows = csv.reader(f)
        head = [h.strip() for h in next(rows)]
        names = [h.split(" ")[0] for h in head]

        def find(*keys):   # nvidia-smi renames: clocks.sm -> clocks.current.sm, throttle -> event reasons
            return next(i for i, n in enumerate(names) if n in keys)
        col = {"timestamp": find("timestamp"), "index": find("index"), "clocks.sm": find("clocks.sm", "clocks.current.sm"),
               "power.draw": find("power.draw"), "temperature.gpu": find("temperature.gpu"),
               "clocks_throttle_reasons.active": find("clocks_throttle_reasons.active", "clocks_event_reasons.active")}
        for r in rows:
            if len(r) < len(head):
                continue
            r = [x.strip() for x in r]
            try:
                t = datetime.strptime(r[col["timestamp"]], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out[int(r[col["index"]])].append((int(t * 1e9), float(r[col["clocks.sm"]]),
                                                  float(r[col["power.draw"]]), float(r[col["temperature.gpu"]]),
                                                  int(r[col["clocks_throttle_reasons.active"]], 16)))
            except (ValueError, KeyError):
                continue
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    samples = load_samples(Path(sys.argv[1]))
    run = Path(sys.argv[2])
    t0 = json.loads((run / "t0.json").read_text())["t0_unix_ns"]
    gpus = [int(g) for g in json.loads((run / "run.json").read_text())["gpus"].split(",")]
    idx = sorted(samples)
    if not set(gpus) <= set(idx):          # nvidia-smi renumbered inside the allocation: by order
        gpus = idx[:len(gpus)]
    mbs = [json.loads(line) for line in open(run / "microbatches.jsonl")]
    rows = defaultdict(dict)
    for m in mbs:
        s, i = m["stage"], m["index"]
        a = t0 + int(m["start_ms"] * 1e6)
        b = a + int(m["compute_ms"] * 1e6)
        ss = [x for x in samples[gpus[s]] if a <= x[0] <= b]
        if not ss:
            continue
        rows[i][s] = {"ms": m["compute_ms"], "sm": st.mean(x[1] for x in ss), "w": st.mean(x[2] for x in ss),
                      "temp": st.mean(x[3] for x in ss), "reasons": Counter(reasons(x[4]) for x in ss)}
    n = max(rows) + 1
    print(f"{run}: {n} microbatches, stages on GPUs {gpus}")
    print("first microbatches: compute ms / SM MHz / W per stage")
    for i in range(min(12, n)):
        print(f"  {i:4d} " + "  ".join(f"{r['ms']:5.1f}/{r['sm']:4.0f}/{r['w']:4.0f}" if (r := rows[i].get(s)) else
                                       " " * 15 for s in range(len(gpus))))
    dec = max(n // 10, 1)
    for s in range(len(gpus)):
        print(f"stage {s} deciles (ms, SM MHz, W, C):")
        for d in range(10):
            rs = [rows[i][s] for i in range(d * dec, (d + 1) * dec) if s in rows[i]]
            if rs:
                print(f"  {d}: {st.median(r['ms'] for r in rs):6.2f} {st.median(r['sm'] for r in rs):6.0f} "
                      f"{st.median(r['w'] for r in rs):6.0f} {st.median(r['temp'] for r in rs):5.1f}")
        allr = [rows[i][s] for i in rows if s in rows[i]]
        c = Counter()
        for r in allr:
            c.update(r["reasons"])
        print(f"  throttle reasons in samples: {dict(c.most_common())}")
        x = [1 / r["sm"] for r in allr]
        y = [r["ms"] for r in allr]
        if len(set(x)) > 2:
            fit = st.linear_regression(x, y)
            print(f"  compute_ms = {fit.slope:.0f} / MHz {fit.intercept:+.2f}  (r={st.correlation(x, y):.3f}): "
                  f"at 1965 MHz {fit.slope / 1965 + fit.intercept:.2f} ms")


if __name__ == "__main__":
    main()
