"""Tune FlashInfer's tactics once and persist them (`ppimpl.model.stage.TACTICS`), or verify a persisted file.

    python bench/tune_tactics.py tune   [--out PATH]                 # 1 GPU, ~5 min
    python bench/tune_tactics.py verify [--lengths 1024,8192,131072]  # fresh process: load, Ready time, cache misses

`tune` builds stage 0 with the output head (every FlashInfer op the four stages use: KDA-dense and MoE layers'
NVFP4 GEMMs, the trtllm NVFP4 MoE, the TRTLLM sparse MLA), budget 131,072 tokens (the largest microbatch any run
config uses, so every token bucket is profiled), runs the warm-up with tuning forced and saves the autotuner's
configs. The file carries FlashInfer's metadata (FlashInfer/CUDA/cuBLAS/cuDNN versions, GPU); a stage loads it only
where they match, otherwise it tunes in-process and warns.

`verify` builds the same stage in a new process with `warmup(tune="auto")`, reports the warm-up time and, over one
forward per length, every autotuner lookup that missed the file (op, input shapes): none is the pass condition.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers as lb  # noqa: E402  (also puts the lane on sys.path)

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


def tune(out: Path) -> None:
    from flashinfer.autotuner import AutoTuner

    model, built, warm = build("force")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()            # a fresh file: no merge with entries tuned elsewhere
    AutoTuner.get().save_configs(str(out))
    data = json.loads(out.read_text())
    print(f"[tune] built {built:.0f} s, tuned warm-up {warm:.0f} s; wrote {out} ({len(data)} keys, "
          f"metadata {data.get('_metadata')})", flush=True)


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
    ap.add_argument("mode", choices=["tune", "verify"])
    ap.add_argument("--out", type=Path, default=TACTICS)
    ap.add_argument("--lengths", default="1024,2048,4096,8192,131072")
    a = ap.parse_args()
    if a.mode == "tune":
        tune(a.out)
    else:
        verify([int(x) for x in a.lengths.split(",")])


if __name__ == "__main__":
    main()
