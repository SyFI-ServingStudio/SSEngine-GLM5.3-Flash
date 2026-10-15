"""vLLM TP4+EP reference for uni_1k_3 alone and inside the 7140 / 7162 8 x 1k microbatches (4 GPUs).

    python bench/vllm_batch.py --prompts measured/batchflip/prompts.npz --out JSON

Same engine flags as `bench/chain.py vllm` (eager, fp8 KV, no prefix caching) with max_num_batched_tokens 16384, so
one `generate` call over 8 x 1024-token prompts is scheduled as one 8,192-token step. Prints and records uni_1k_3's
top-5 first-token logprobs for: alone (T = 1024), b114 (7140 mb 114 order), b206 (7162 mb 206 order), and alone again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batchflip import BATCHES, TARGET  # noqa: E402


def main() -> None:
    import numpy as np
    from vllm import LLM, SamplingParams

    from ppimpl.model.stage import CHECKPOINT

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = {k: [int(t) for t in v] for k, v in np.load(a.prompts).items()}
    llm = LLM(model=CHECKPOINT, tensor_parallel_size=4, enable_expert_parallel=True, kv_cache_dtype="fp8_e4m3",
              language_model_only=True, max_model_len=8192 + 64, enable_prefix_caching=False, enforce_eager=True,
              max_num_batched_tokens=16384, max_num_seqs=64)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=5)

    def run(names):
        outs = llm.generate([{"prompt_token_ids": prompts[n]} for n in names], sp, use_tqdm=False)
        o = outs[names.index(TARGET)].outputs[0]
        top = sorted(o.logprobs[0].items(), key=lambda kv: -kv[1].logprob)[:5]
        return {"token": o.token_ids[0], "top5": [[int(t), round(v.logprob, 4)] for t, v in top]}

    res = {"alone": run([TARGET])}
    for name in ("b114", "b206"):
        res[name] = run([f"uni_1k_{k}" for k in BATCHES[name]])
    res["alone_again"] = run([TARGET])
    for k, v in res.items():
        print(f"[vllm_batch] {k}: {v}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
