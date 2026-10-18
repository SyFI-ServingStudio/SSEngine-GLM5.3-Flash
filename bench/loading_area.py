"""The staged-loading area's inputs (TierConfig.loading_read_gb_s, loading_prefill_tok_s; ppimpl/engine/scheduler.py)
from a measured run of the same setting: Little's law, area = the tiers' read rate into HBM x one batch period
(scheduler mb_tokens / prefill tokens/s).

- read rate: every read into HBM in tier_ops.jsonl (one per round served from a tier; a staged SSD read's hop, not
  its fetch), at the HBM blocks it holds (its context's KV blocks plus the state's blocks) x slot bytes, per GPU,
  over the run's span of tier ops;
- prefill tokens/s: metrics.json computed_tokens_per_s.

    .venv/bin/python -m bench.loading_area measured/runs_m6/<run>   # prints the --tier flags and the area
"""

import argparse
import json
import math
from pathlib import Path

import yaml

from ppimpl.config import STATE_BLOCKS
from ppimpl.engine.kv import kv_blocks_for


def inputs(run: Path) -> dict:
    ops = [json.loads(x) for x in (run / "tier_ops.jsonl").read_text().splitlines()]
    slot = json.loads((run / "tiers.json").read_text())["slot_bytes"]
    reads = [o for o in ops if o["kind"] == "read" and o["req"] is not None]
    span_s = (max(o["landed_ms"] for o in ops) - min(o["sent_ms"] for o in ops)) / 1e3
    held = sum(kv_blocks_for(o["tokens"][1]) + STATE_BLOCKS for o in reads) * slot
    cfg = yaml.safe_load((run / "server_config.yaml").read_text())
    mb = cfg["scheduler"]["mb_tokens"]
    tok_s = json.loads((run / "metrics.json").read_text())["computed_tokens_per_s"]
    blocks = json.loads((run / "pool.json").read_text())["num_blocks"]
    gb_s = held / span_s / 1e9
    area = math.ceil(gb_s * 1e9 * mb / tok_s / slot)
    return {"loading_read_gb_s": round(gb_s, 4), "loading_prefill_tok_s": round(tok_s), "reads": len(reads),
            "span_s": round(span_s, 1), "batch_period_s": round(mb / tok_s, 4), "area_blocks": area,
            "hbm_blocks": blocks, "area_fraction": round(area / blocks, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    x = inputs(ap.parse_args().run)
    print(json.dumps(x))
    print(f"--tier loading_read_gb_s={x['loading_read_gb_s']} --tier loading_prefill_tok_s={x['loading_prefill_tok_s']}")


if __name__ == "__main__":
    main()
