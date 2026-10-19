"""Tune FlashInfer's tactics once and persist them (`ppimpl.model.stage.TACTICS`), or verify a persisted file.

    python -m benchmarks.tune_tactics tune   [--out PATH] --routing RUN_DIR [--routing ...] | --routing-json PATH  # 1 GPU
    python -m benchmarks.tune_tactics verify [--lengths 1024,8192,131072]  # fresh process: load, Ready time, cache misses
    python -m benchmarks.tune_tactics merge --base FILE --ab A.json [--ab ...] --routing-name real --out PATH  # CPU

`tune` builds stage 0 with the output head (every FlashInfer op the four stages use: KDA-dense and MoE layers'
NVFP4 GEMMs, the trtllm NVFP4 MoE, the TRTLLM sparse MLA), budget 131,072 tokens (the largest microbatch any run
config uses, so every token bucket is profiled), runs the warm-up with tuning forced and saves the autotuner's
configs. The fused MoE is tuned at the serving buckets (`ppimpl.model.moe_tuning`, installed by every stage) on
served routing: `--routing` reads the experts_stage<S>.jsonl of runs made with PPIMPL_LOG_EXPERTS=1 and folds each
routed-row count's histograms; `--routing-json` takes {rows: histogram} or a single histogram list (then every
bucket scales it). The histograms each bucket was tuned on go to <out>.routing.json. The file carries FlashInfer's metadata (FlashInfer/CUDA/cuBLAS/cuDNN versions, GPU); a stage loads it only
where they match, otherwise it tunes in-process and warns.

`verify` builds the same stage in a new process with `warmup(tune="auto")`, reports the warm-up time and, over one
forward per length, every autotuner lookup that missed the file (op, input shapes): none is the pass condition.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

from benchmarks import common as lb

BUDGET = 131072


def build(warmup: str):
    import torch
    from ppimpl.model.stage import StageModel, bench_config

    t0 = time.time()
    model = StageModel(bench_config(BUDGET), 0, torch.device("cuda", 0), warmup=False, last=True,
                       num_blocks=lb.NUM_BLOCKS, max_model_len=BUDGET + 8576)
    built = time.time() - t0
    t0 = time.time()
    model.warmup(tune=warmup)
    return model, built, time.time() - t0


def load_profile(runs: list[Path], json_path: Path | None) -> dict[int, list[int]]:
    from ppimpl.model import moe_tuning

    if runs:
        return moe_tuning.routing_profile(runs)
    d = json.loads(json_path.read_text())
    if isinstance(d, list):
        return {sum(d) // moe_tuning.TOP_K: d}
    return {int(k): v for k, v in d.items() if not k.startswith("_")}


def tune(out: Path, profile: dict[int, list[int]], sources: list[str]) -> None:
    from flashinfer.autotuner import AutoTuner
    from ppimpl.model import moe_tuning

    if not profile:
        raise SystemExit("tune: no routing histograms (--routing / --routing-json)")
    print(f"[tune] routing profile: row counts {sorted(profile)[:8]}..{sorted(profile)[-3:]} "
          f"({len(profile)}) from {sources}", flush=True)
    with moe_tuning.real_routing(profile) as used:
        model, built, warm = build("force")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()            # a fresh file: no merge with entries tuned elsewhere
    AutoTuner.get().save_configs(str(out))
    data = json.loads(out.read_text())
    moe = sorted(int(k.split("((")[1].split(",")[0]) for k in data if "trtllm_fp4_block_scale_moe" in k)
    side = out.with_suffix(".routing.json")
    side.write_text(json.dumps({"sources": sources, "moe_buckets": moe,
                                "tuned_on": {str(k): {"max": max(h), "active": sum(1 for x in h if x), "hist": h}
                                             for k, h in sorted(used.items())}}))
    print(f"[tune] built {built:.0f} s, tuned warm-up {warm:.0f} s; wrote {out} ({len(data)} keys, "
          f"{len(moe)} MoE buckets, routing at {len(used)} token counts -> {side}; metadata {data.get('_metadata')})",
          flush=True)
    if not used:
        raise SystemExit("tune: the MoE was not tuned on the given routing (initializer never called)")


OP_MOE = "flashinfer::trtllm_fp4_block_scale_moe"


def _moe_key_tokens(key: str) -> int:
    return int(key.split("((")[1].split(",")[0])


def merge(base: Path, abs_: list[Path], routing: str, out: Path) -> dict:
    """`base` with an MoE entry at every serving bucket (`moe_tuning.serving_buckets`): a bucket the base lacks gets
    the tactic the base served it with (FlashInfer's hybrid bucket at or above it), then every bucket timed in the
    research.bench.moe_tactics reports `abs_` at routing `routing` takes the sustained-best tactic where it is faster than that
    one. Every other key is kept as is. Returns what changed."""
    from flashinfer.fused_moe.utils import map_to_hybrid_bucket
    from ppimpl.model import moe_tuning

    data = json.loads(base.read_text())
    moe = {_moe_key_tokens(k): k for k in data if OP_MOE in k}
    top = max(moe)
    template = moe[top]
    key_for = lambda t: template.replace(f"({top}, ", f"({t}, ")       # noqa: E731
    added, changed = {}, {}
    for b in moe_tuning.serving_buckets(top):
        if b not in moe:
            src = moe.get(map_to_hybrid_bucket(b, top)) or moe[min(x for x in moe if x >= b)]
            data[key_for(b)] = json.loads(json.dumps(data[src]))
            moe[b] = key_for(b)
            added[b] = data[src][1]
    for p in abs_:
        for c in json.loads(Path(p).read_text())["cases"]:
            if c["routing"] != routing or c["tokens"] not in moe:
                continue
            times = {tuple(t[0]): t[1] for t in c["tactics"]}
            cur = tuple(data[moe[c["tokens"]]][1])
            best = min(times, key=times.get)
            if cur in times and times[best] < times[cur]:
                data[moe[c["tokens"]]][1] = list(best)
                changed[c["tokens"]] = {"from": list(cur), "to": list(best), "from_ms": round(times[cur], 4),
                                        "to_ms": round(times[best], 4)}
    out.write_text(json.dumps(data, indent=2))
    res = {"base": str(base), "out": str(out), "added": added, "changed": changed}
    print(f"[merge] {json.dumps(res)}", flush=True)
    return res


def verify(lengths: list[int]) -> None:
    import torch
    from flashinfer.autotuner import AutoTuner
    from ppimpl.model.stage import CHECKPOINT

    misses: collections.Counter = collections.Counter()
    orig = AutoTuner.search_cache

    def search_cache(self, custom_op, runners, input_shapes, tuning_config, inputs=None, **kw):
        out = orig(self, custom_op, runners, input_shapes, tuning_config, inputs=inputs, **kw)
        if not out[0]:
            misses[(custom_op, str(input_shapes)[:200])] += 1
        return out

    AutoTuner.search_cache = search_cache
    model, built, warm = build("auto")
    print(f"[verify] tactics {model.tactics}; built {built:.0f} s, warm-up {warm:.0f} s; misses during warm-up "
          f"{sum(misses.values())}", flush=True)
    warm_misses = dict(misses)
    misses.clear()
    tokens = lb.enwik9_tokens(CHECKPOINT, max(lengths))
    with torch.inference_mode():
        for L in lengths:
            model.forward(lb.one_request_plan(tokens[:L]), None)
            torch.cuda.synchronize()
    res = {"tactics": model.tactics, "build_s": round(built, 1), "warmup_s": round(warm, 1),
           "warmup_misses": {f"{k[0]} {k[1]}": v for k, v in warm_misses.items()},
           "forward_misses": {f"{k[0]} {k[1]}": v for k, v in misses.items()}, "lengths": lengths}
    res["pass"] = model.tactics.get("source") == "file" and not warm_misses and not misses
    out = lb.MEASURED / "host" / f"tactics_verify_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[verify] {json.dumps(res)[:3000]}", flush=True)
    print(f"[verify] wrote {out}", flush=True)


def main() -> None:
    from ppimpl.model.stage import TACTICS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tune", "verify", "merge"])
    ap.add_argument("--base", type=Path)
    ap.add_argument("--ab", type=Path, action="append", default=[])
    ap.add_argument("--routing-name", default="real")
    ap.add_argument("--out", type=Path, default=TACTICS)
    ap.add_argument("--lengths", default="1024,2048,4096,4352,6144,8192,131072")
    ap.add_argument("--routing", type=Path, action="append", default=[], help="run dir with experts_stage*.jsonl")
    ap.add_argument("--routing-json", type=Path)
    a = ap.parse_args()
    if a.mode == "merge":
        merge(a.base, a.ab, a.routing_name, a.out)
    elif a.mode == "tune":
        tune(a.out, load_profile(a.routing, a.routing_json), [str(p) for p in a.routing or [a.routing_json]])
    else:
        verify([int(x) for x in a.lengths.split(",")])


if __name__ == "__main__":
    main()
