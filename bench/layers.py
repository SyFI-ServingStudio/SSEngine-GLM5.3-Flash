"""M2 per-layer alignment (handoff `3_layers.md`): one layer of each type, plus the input and output heads, at
L = 1k ... 128k on one GPU, through the serving code path (`ppimpl.model.stage.StageModel`).

The bench model is stage 0 (layers 0-11: `first_kda_dense`, 2 x `kda_dense`, 3 x `dsa_moe`, 6 x `kda_moe`, the
`input_head`) built with the final norm and `lm_head` (`output_head`), so one forward runs every layer type at once
and the stage-0 sum is measured directly as well. One request, L fresh enwik9 tokens, the whole prompt in one call;
DSA layers write their KV, KDA layers start from a zero state (§4).

Routing (§4): `forced` makes every MoE layer's experts receive exactly `moe_per_expert_batches` (the gate GEMM runs,
its logits are replaced by +-30 FP32 logits whose top-8 under the layer's correction bias are those of
`exact_topk_ids`); `router` runs the real router on the same text.

Timing (§4): torch.profiler (CUPTI). Each forward is preceded by a GPU sleep longer than the forward's host launch
time, so its kernels run back to back as in serving with a launch queue ahead of the GPU; a third pass without the
sleep measures the host gaps of plain eager launching (`gaps.*` fields). A layer's time is the GPU span from its
first kernel start to its last kernel end; per-position times add the kernels the rules below assign. Every kernel
is attributed to the module whose forward launched it (record_function scopes on every submodule; the launch's CUPTI
correlation id links kernel and scope).

    python bench/layers.py --lengths 1024,8192 [--runs 10] [--routing forced,router] [--no-autotune]
    python bench/layers.py --summarize          # measured/layers.jsonl -> measured/layers.md

Outputs: `measured/layers.jsonl` (one line per (prompt length, layer type, routing), §7, appended; the summary
takes the latest line per key), `measured/layers_raw/L<L>_<routing>.json` (every kernel of one run with its scope
and position, for the rules), `measured/layers.md`.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

LANE = Path(__file__).resolve().parents[1]
PPIMPL = LANE.parents[1] if LANE.parent.name == "lanes" else LANE
SIM = PPIMPL / "third_party" / "sim_7bb224a6"
for p in (str(LANE), str(SIM)):
    if p not in sys.path:
        sys.path.insert(0, p) if p == str(LANE) else sys.path.append(p)

HANDOFF = Path(os.environ.get("PPIMPL_HANDOFF",
                              "/raid/kanzhu/ServingStudio/wt-pp-main/doc/handoff/glm53_flash_pp4_b200_v1"))
ENWIK9 = Path(os.environ.get("PPIMPL_ENWIK9", "/raid/kanzhu/ServingStudio/data/corpus/enwik9"))
MEASURED = LANE / "measured"
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
TYPES = ("input_head", "first_kda_dense", "kda_dense", "kda_moe", "dsa_moe", "output_head")
MOE_TYPES = ("kda_moe", "dsa_moe")
STAGE_COMPOSITION = {   # 3_layers.md §2
    0: {"input_head": 1, "first_kda_dense": 1, "kda_dense": 2, "dsa_moe": 3, "kda_moe": 6},
    1: {"dsa_moe": 2, "kda_moe": 9},
    2: {"dsa_moe": 3, "kda_moe": 8},
    3: {"dsa_moe": 3, "kda_moe": 8, "output_head": 1},
}
README_STAGE_MS = {     # handoff README, one request, whole prompt in one microbatch
    1024: (12.3, 13.0, 13.1, 13.3), 2048: (18.0, 18.3, 18.4, 18.6), 4096: (30.9, 30.1, 30.6, 30.9),
    8192: (57.1, 54.1, 55.3, 55.7), 16384: (111.7, 103.9, 106.8, 107.4), 32768: (220.4, 203.1, 209.0, 209.9),
    65536: (431.2, 395.5, 406.9, 408.4), 131072: (871.5, 792.6, 821.2, 823.9),
}
KV_BLOCKS = 16           # ceil(131072 / 8576)
NUM_BLOCKS = 1 + KV_BLOCKS + 6


# ================================================================ kernel -> position rules

GEMM = r"(?i)^(?!.*(splitKreduce|flash_fwd)).*(nvjet|gemm|cutlass|xmma|gemv)"
SPLITK_REDUCE = re.compile(r"splitKreduce")
QUANT = r"(?i)cvt_fp16_to_fp4|fp4_quant|scaled_fp4|quantize_with_block_size|fp4_kernel"

# (position, scope regex, kernel regex, group, nth). The scope is the module path below the unit (layer / head), each
# element `<attribute>#<call ordinal within the unit>`, e.g. `self_attn#0/indexer#0/wq_b#0`. First match wins;
# position None = no target position (copies, casts). A rule with a group matches only the nth kernel of that group
# (the group's kernels are those matching its scope and kernel regex) in the unit.
KDA_RULES = [
    ("kda.in_proj", r"in_proj_qkvbfg_a", None),
    ("kda.f_b_proj", r"f_b_proj", None),
    ("kda.g_b_proj", r"g_b_proj", None),
    ("kda.short_conv_prefill", r"_forward", r"causal_conv1d"),
    ("kda.state_gather", r"_forward", r"(?i)index_select|indexSelect|FillFunctor"),
    ("kda.state_scatter", r"_forward", r"(?i)index_copy|indexFunc|index_put|index_elementwise"),
    (None, r"_forward", r"(?i)copy"),
    ("kda.chunk_prefill", r"_forward", None),
    ("kda.gated_norm", r"o_norm", None),
    ("kda.o_proj", r"o_proj", None),
]
DENSE_RULES = [
    ("dense_ffn.gate_up_input_quant", r"^mlp#\d+/gate_up_proj", QUANT),
    ("dense_ffn.gate_up", r"^mlp#\d+/gate_up_proj", None),
    ("dense_ffn.act_and_mul", r"^mlp#\d+/act_fn", None),
    ("dense_ffn.down_input_quant", r"^mlp#\d+/down_proj", QUANT),
    ("dense_ffn.down", r"^mlp#\d+/down_proj", None),
]
MOE_RULES = [
    ("moe.shared_expert.gate_up", r"shared_experts#\d+/gate_up_proj", None),
    ("moe.shared_expert.act_and_mul", r"shared_experts#\d+/act_fn", None),
    ("moe.shared_expert.down", r"shared_experts#\d+/down_proj", None),
    ("moe.router.gate", r"(^|/)gate#\d+$", None),
    ("moe.routed_rank0.input_quant", r"^mlp#", QUANT),
    ("moe.routed_rank0.fused_moe", r"^mlp#",
     r"(?i)moe|routing|permute|finalize|bmm_|trtllm|expert|topk|gemm|activation"),
    # remaining MoE kernels: input_glue before the first fused_moe kernel, combine_glue after (`_moe_glue`)
]
DSA_RULES = [
    ("dsa.fused_qkv_a", r"fused_qkv_a_proj", None),
    ("dsa.q_kv_a_norm", r"(q_a_layernorm|kv_a_layernorm)", None),
    ("dsa.q_b_proj", r"q_b_proj", None),
    ("dsa.indexer.wq_b", r"indexer#\d+/wq_b", None),
    ("dsa.indexer.wk_weights", r"indexer#\d+/wk_weights_proj", None),
    ("dsa.indexer.head_weights", r"indexer#\d+$", GEMM, "idx_gemm", 0),
    ("dsa.indexer.kpool_gate_score", r"indexer#\d+$", GEMM, "idx_gemm", 1),
    ("dsa.indexer.q_fwht_quant", r"indexer#", r"(?i)fwht"),
    ("dsa.indexer.weight_scale", r"indexer#", r"(?i)weight_scale|fused_mul_unsqueeze"),
    ("dsa.indexer.k_norm", r"indexer#", r"(?i)k_norm|layer_?norm"),
    ("dsa.indexer.mqa_logits_prefill", r"indexer#", r"(?i)mqa_logits|mqa"),
    ("dsa.indexer.topk_prefill", r"indexer#", r"(?i)top_?k|deep_?select|radix|select"),
    ("dsa.indexer.expand_pools", r"indexer#", r"(?i)expand"),
    ("dsa.indexer.prefill_gather", r"indexer#", r"(?i)gather"),
    ("dsa.indexer.kpool_tail_seed", r"indexer#", r"(?i)tail"),
    ("dsa.indexer.kpool_prefill_write", r"indexer#", r"(?i)kpool|compress|cache|write"),
    ("dsa.o_proj", r"o_proj", None),
    ("dsa.sparse_mla.index_remap", r"^self_attn#", r"(?i)remap|req_index|global_index|physical"),
    ("dsa.sparse_mla.prefill", r"^self_attn#", r"(?i)fmha|trtllm|sparse_?mla|mla_?sparse|flash_fwd"),
    ("dsa.sparse_mla.mla_cache_append", r"^self_attn#", r"(?i)concat_and_cache|cache_mla|reshape_and_cache"),
    ("dsa.q_kv_a_norm", r"mla_attn#\d+$", r"^Kernel$"),
    ("dsa.q_absorb", r"mla_attn#\d+$", GEMM, "mla_bmm", 0),
    ("dsa.v_up", r"mla_attn#\d+$", GEMM, "mla_bmm", 1),
    ("dsa.q_fp8_quant", r"mla_attn#", r"(?i)quant|fp8|scaled"),
]
MHC_FIRST = [("attn_mhc_pre", r"^mhc_pre_op", None), ("ffn_mhc_post_pre", r"^mhc_fused_post_pre_op#0", None)]
MHC = [("attn_mhc_post_pre", r"^mhc_fused_post_pre_op#0", None),
       ("ffn_mhc_post_pre", r"^mhc_fused_post_pre_op#1", None),
       ("stage_edge.hc_pre", r"^mhc_pre_op", None)]
RULES = {
    "first_kda_dense": MHC_FIRST + KDA_RULES + DENSE_RULES,
    "kda_dense": MHC + KDA_RULES + DENSE_RULES,
    "kda_moe": MHC + KDA_RULES + MOE_RULES,
    "dsa_moe": MHC + DSA_RULES + MOE_RULES,
    "input_head": [("embedding", r"^embed", None), ("hc_expand", r"^hc_expand", None)],
    "output_head": [("final_mhc_post", r"mhc_post_op", None), ("final_norm", r"(^|/)norm#", None),
                    ("lm_head", r"", GEMM), ("hc_contract_mean", r"", r"MeanOps")],
}
_rx = {}


def _re(p):
    if p not in _rx:
        _rx[p] = re.compile(p)
    return _rx[p]


def assign_positions(layer_type: str, kernels: list[dict]) -> None:
    """Set k["position"] (`pp.<type>.<pos>` or None) on a unit's kernels, in start order."""
    rules = [r + (None, None) if len(r) == 3 else r for r in RULES[layer_type]]
    counters: dict = collections.Counter()
    groups = {r[3]: (r[1], r[2]) for r in rules if r[3]}
    prefix = "pp." if layer_type in ("input_head", "output_head") else f"pp.{layer_type}."
    for k in kernels:
        nth = {}
        for g, (sp, kp) in groups.items():
            if _re(sp).search(k["scope"]) and _re(kp).search(k["name"]):
                nth[g] = counters[g]
                counters[g] += 1
        k["position"] = None
        k["rule"] = False
        for pos, sp, kp, g, n in rules:
            if g is not None and nth.get(g) != n:
                continue
            if _re(sp).search(k["scope"]) and (kp is None or _re(kp).search(k["name"])):
                k["position"] = None if pos is None else prefix + pos
                k["rule"] = True
                break
    prev = None
    for k in kernels:
        if SPLITK_REDUCE.search(k["name"]) and prev is not None:
            k["position"] = prev
        prev = k["position"]
    if layer_type in MOE_TYPES:
        _moe_glue(layer_type, kernels)


def _moe_glue(layer_type, kernels):
    fused = [k["ts"] for k in kernels if k["position"] == f"pp.{layer_type}.moe.routed_rank0.fused_moe"]
    first = min(fused) if fused else math.inf
    for k in kernels:
        if not k["rule"] and k["scope"].startswith("mlp#"):
            k["position"] = f"pp.{layer_type}.moe.{'input_glue' if k['ts'] < first else 'combine_glue'}"


# ================================================================ record_function scopes


class Scopes:
    """record_function around module forwards. `on` toggles module scopes; units (layers, heads) and runs always."""
    on = True

    @staticmethod
    def wrap(obj, attr: str, name: str, unit: bool = False):
        import torch

        orig = getattr(obj, attr)

        def call(*a, **kw):
            if not (unit or Scopes.on):
                return orig(*a, **kw)
            with torch.profiler.record_function(name):
                return orig(*a, **kw)

        setattr(obj, attr, call)


def install_scopes(model) -> None:
    import vllm.models.glm5next.common.model as glm_model

    for layer in model.layers:
        for name, mod in layer.named_modules():
            if name:
                Scopes.wrap(mod, "forward", name.rsplit(".", 1)[-1])
        if layer.layer_kind == "kda":
            Scopes.wrap(layer.self_attn, "_forward", "_forward")
        Scopes.wrap(layer, "forward", f"L{layer.layer_idx}", unit=True)
    if model.embed is not None:
        Scopes.wrap(model.embed, "forward", "embed", unit=True)
    if model.norm is not None:
        Scopes.wrap(model.norm, "forward", "norm")
    Scopes.wrap(model, "output_head", "output_head", unit=True)
    Scopes.wrap(glm_model, "hc_expand", "hc_expand", unit=True)


# ================================================================ trace -> units


GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
UNIT = re.compile(r"^(L\d+|embed|hc_expand|output_head)$")
RUN = re.compile(r"^run(\d+)$")


def parse_trace(path: Path) -> dict[int, list[dict]]:
    """run index -> its GPU events: name, ts/end (us), stream, unit (`L3`, `embed`, ...), scope below the unit."""
    events = json.loads(path.read_text())["traceEvents"]
    anns = collections.defaultdict(list)       # tid -> [(ts, end, name, id)]
    launches = {}                               # correlation -> (tid, ts)
    gpu = []
    for e in events:
        cat, ph = e.get("cat"), e.get("ph")
        if ph != "X":
            continue
        if cat == "user_annotation":
            anns[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
        elif cat in LAUNCH_CATS and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = (e["tid"], e["ts"])
        elif cat in GPU_CATS:
            gpu.append(e)
    # per launch: the annotation stack (outermost first) open at its timestamp, with per-unit call ordinals
    stacks = {}
    by_tid = collections.defaultdict(list)
    for corr, (tid, ts) in launches.items():
        by_tid[tid].append((ts, corr))
    for tid, launch_list in by_tid.items():
        marks = []
        for i, (ts, end, name) in enumerate(anns.get(tid, [])):
            marks.append((ts, 0, -(end - ts), i))
            marks.append((end, 2, 0, i))
        for ts, corr in launch_list:
            marks.append((ts, 1, 0, corr))
        marks.sort()
        tid_anns = anns.get(tid, [])
        open_: list = []            # [(name, ordinal within the enclosing unit instance)]
        open_ids: list = []
        counters: dict = collections.Counter()
        for _, kind, _, ref in marks:
            if kind == 0:
                name = tid_anns[ref][2]
                unit_ref = next((open_ids[j] for j in range(len(open_) - 1, -1, -1) if UNIT.match(open_[j][0])), None)
                key = (unit_ref, name)
                open_.append((name, counters[key]))
                counters[key] += 1
                open_ids.append(ref)
            elif kind == 2:
                if ref in open_ids:
                    j = len(open_ids) - 1 - open_ids[::-1].index(ref)
                    del open_[j], open_ids[j]
            else:
                stacks[ref] = list(open_)
    runs: dict[int, list[dict]] = collections.defaultdict(list)
    for e in gpu:
        corr = e.get("args", {}).get("correlation")
        stack = stacks.get(corr)
        if stack is None:
            continue
        run = next((int(RUN.match(n).group(1)) for n, _ in stack if RUN.match(n)), None)
        if run is None:
            continue
        unit_pos = max((j for j, (n, _) in enumerate(stack) if UNIT.match(n)), default=-1)
        unit = stack[unit_pos][0] if unit_pos >= 0 else None
        scope = "/".join(f"{n}#{o}" for n, o in stack[unit_pos + 1:])
        runs[run].append({"name": e["name"], "ts": e["ts"], "end": e["ts"] + e["dur"], "dur": e["dur"],
                          "stream": e.get("args", {}).get("stream"), "unit": unit, "scope": scope})
    for evs in runs.values():
        evs.sort(key=lambda k: k["ts"])
    return runs


def _union_us(kernels) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for k in sorted(kernels, key=lambda k: k["ts"]):
        if cur_e is None or k["ts"] > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = k["ts"], k["end"]
        else:
            cur_e = max(cur_e, k["end"])
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def unit_type(unit: str) -> str:
    from ppimpl import kernels as km

    if unit in ("embed", "hc_expand"):
        return "input_head"
    if unit == "output_head":
        return "output_head"
    return km.layer_type(int(unit[1:]))


def summarize_run(kernels: list[dict]) -> dict:
    """One run: per unit instance (layer index or head) its span, busy time, kernels with positions."""
    units = collections.defaultdict(list)
    for k in kernels:
        if k["unit"] is None:
            continue
        key = "input_head" if k["unit"] in ("embed", "hc_expand") else k["unit"]
        if key == "input_head" and not k["scope"].startswith(k["unit"]):
            k["scope"] = f"{k['unit']}#0" + (f"/{k['scope']}" if k["scope"] else "")
        units[key].append(k)
    out = {}
    for key, ks in units.items():
        lt = unit_type(key if key != "input_head" else "embed")
        assign_positions(lt, ks)
        start, end = min(k["ts"] for k in ks), max(k["end"] for k in ks)
        out[key] = {"type": lt, "span_us": end - start, "busy_us": _union_us(ks), "kernels": ks}
    all_k = [k for k in kernels]
    out["_forward"] = {"span_us": max(k["end"] for k in all_k) - min(k["ts"] for k in all_k),
                       "busy_us": _union_us(all_k)}
    return out


# ================================================================ inputs


def load_targets() -> dict:
    """(L, layer type) -> target line of layer_targets.jsonl (stage 0 lines; output_head from stage 3)."""
    out = {}
    for line in (HANDOFF / "data" / "layer_targets.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.setdefault((r["prompt_tokens"], r["layer_type"]), r)
    return out


def enwik9_tokens(checkpoint: str, n: int) -> list[int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
    nbytes = 8 * n
    while True:
        with open(ENWIK9, "rb") as f:
            text = f.read(nbytes).decode("utf-8", errors="ignore")
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) >= n:
            return ids[:n]
        nbytes *= 2


def one_request_plan(tokens: list[int]):
    L = len(tokens)
    n = -(-L // 8576)
    chunk = SimpleNamespace(req_id="r0", start=0, num_tokens=L, kv_blocks=list(range(1, 1 + n)),
                            state_blocks=list(range(1 + KV_BLOCKS, 1 + KV_BLOCKS + 6)), is_last=True,
                            token_ids=tokens)
    return SimpleNamespace(mb_id=0, chunks=[chunk])


class Routing:
    """Source of the MoE gates' forced logits; None = real router."""
    logits = None

    @classmethod
    def get(cls):
        return cls.logits


def check_forced(model, logits, counts) -> int:
    """Experts each token selects under DeepSeek-V3 routing with each MoE layer's correction bias; returns the
    number of layers whose per-expert counts differ from `counts`."""
    import torch

    bad = 0
    want = torch.tensor(counts, device=logits.device)
    scores = torch.sigmoid(logits)
    for layer in model.layers:
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        bias = getattr(gate, "e_score_correction_bias", None)
        if bias is None:
            continue
        ids = (scores + bias.float()).topk(8, dim=-1).indices
        got = torch.bincount(ids.flatten(), minlength=len(counts))
        bad += int(not torch.equal(got, want))
    return bad


# ================================================================ measurement


def sleep_cycles_per_ms() -> float:
    import torch

    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cycles = 50_000_000
    a.record()
    torch.cuda._sleep(cycles)
    b.record()
    b.synchronize()
    return cycles / a.elapsed_time(b)


def host_syncs(model, plan) -> list[dict]:
    """Synchronizing CUDA calls in one forward (prepare + run), with the innermost non-torch frame of each:
    torch's sync debug mode warns on every .item()/.cpu()/pageable copy/stream sync it issues."""
    import traceback as tb
    import warnings

    import torch

    found: dict = {}

    def show(message, category, filename, lineno, file=None, line=None):
        frames = [f for f in tb.extract_stack()[:-1] if "/torch/" not in f.filename and "warnings" not in f.filename]
        where = " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames[-3:][::-1])
        key = (str(message)[:120], where)
        found[key] = found.get(key, 0) + 1

    torch.cuda.synchronize()
    old = warnings.showwarning
    warnings.showwarning = show
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show
            torch.cuda.set_sync_debug_mode("warn")
            try:
                model.forward(plan, None)
            finally:
                torch.cuda.set_sync_debug_mode("default")
    finally:
        warnings.showwarning = old
    torch.cuda.synchronize()
    return [{"message": m, "where": w, "count": n} for (m, w), n in found.items()]


def profile_runs(model, plan, runs: int, lead_ms: float, cycles_per_ms: float, trace_dir: Path, tag: str):
    """`runs` forwards under the profiler, each after a GPU sleep of `lead_ms` (0: none). Returns parsed runs."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for r in range(runs):
            b = model.prepare(plan)
            if lead_ms > 0:
                torch.cuda._sleep(int(lead_ms * cycles_per_ms))
            with record_function(f"run{r}"):
                out = model.run(b, None)
            torch.cuda.synchronize()
    path = trace_dir / f"{tag}.json"
    prof.export_chrome_trace(str(path))
    parsed = parse_trace(path)
    path.unlink()
    return parsed, out


def aggregate(L: int, routing: str, parsed: dict, targets: dict, extra: dict) -> list[dict]:
    """§7 lines: per layer type, mean span over its instances and runs; per position, mean kernel time."""
    per_run = [summarize_run(parsed[r]) for r in sorted(parsed)]
    lines = []
    for lt in TYPES:
        spans, busy, inst = [], [], collections.defaultdict(list)
        pos_ms = collections.defaultdict(list)
        pos_names = collections.defaultdict(collections.Counter)
        for run in per_run:
            units = [(u, d) for u, d in run.items() if not u.startswith("_") and d["type"] == lt]
            for u, d in units:
                spans.append(d["span_us"] / 1e3)
                busy.append(d["busy_us"] / 1e3)
                inst[u].append(d["span_us"] / 1e3)
                acc = collections.defaultdict(float)
                for k in d["kernels"]:
                    acc[k["position"]] += k["dur"] / 1e3
                    pos_names[k["position"]][k["name"][:120]] += 1
                for p, v in acc.items():
                    pos_ms[p].append(v)
        if not spans:
            continue
        n_inst = len(inst)
        tgt = targets.get((L, lt))
        target_ms = tgt["layer_ms"] if tgt else None
        real = statistics.mean(spans)
        kernels = []
        target_pos = [(k["position"], k["total_ms"]) for k in tgt["kernels"]] if tgt else []
        seen = set()
        for p, t in target_pos + [(p, None) for p in pos_ms if p not in {q for q, _ in target_pos}]:
            if p in seen:
                continue
            seen.add(p)
            vals = pos_ms.get(p, [])
            # a position that appears in only some instances counts 0 in the others
            real_p = sum(vals) / (len(spans)) if vals else 0.0
            kernels.append({"position": p, "gpu_kernels": sorted(pos_names[p]), "real_ms": round(real_p, 5),
                            "target_ms": None if t is None else round(t, 5)})
        layers = sorted(int(u[1:]) for u in inst if u.startswith("L"))
        lines.append({
            "prompt_tokens": L, "layer_type": lt, "layer": layers[0] if layers else None, "layers": layers,
            "routing": routing, "real_ms": round(real, 4), "target_ms": None if target_ms is None else round(target_ms, 4),
            "ratio": None if not target_ms else round(real / target_ms, 4), "runs": len(spans) // max(n_inst, 1),
            "busy_ms": round(statistics.mean(busy), 4),
            "std_ms": round(statistics.pstdev(spans), 4),
            "per_layer_ms": {u: round(statistics.mean(v), 4) for u, v in sorted(inst.items())},
            "kernels": kernels, **extra})
    fwd = [run["_forward"]["span_us"] / 1e3 for run in per_run]
    lines.append({"prompt_tokens": L, "layer_type": "stage0_forward", "routing": routing,
                  "real_ms": round(statistics.mean(fwd), 4), "target_ms": README_STAGE_MS[L][0],
                  "ratio": round(statistics.mean(fwd) / README_STAGE_MS[L][0], 4), "runs": len(fwd),
                  "busy_ms": round(statistics.mean(run["_forward"]["busy_us"] / 1e3 for run in per_run), 4),
                  "note": "stage-0 layers + input head + output head (the bench model), one GPU", **extra})
    return lines


def host_gaps(parsed: dict) -> dict:
    per_run = [summarize_run(parsed[r]) for r in sorted(parsed)]
    fwd_span = statistics.mean(r["_forward"]["span_us"] for r in per_run) / 1e3
    fwd_busy = statistics.mean(r["_forward"]["busy_us"] for r in per_run) / 1e3
    worst = collections.defaultdict(list)
    for r in per_run:
        for u, d in r.items():
            if not u.startswith("_"):
                worst[d["type"]].append((d["span_us"] - d["busy_us"]) / 1e3)
    return {"eager_forward_ms": round(fwd_span, 4), "eager_busy_ms": round(fwd_busy, 4),
            "eager_gap_ms": round(fwd_span - fwd_busy, 4),
            "eager_gap_by_type_ms": {t: round(statistics.mean(v), 4) for t, v in worst.items()}}


def raw_dump(L: int, routing: str, parsed: dict) -> None:
    run = summarize_run(parsed[min(parsed)])
    out = {u: {"type": d.get("type"), "span_us": d["span_us"], "busy_us": d["busy_us"],
               "kernels": [{"name": k["name"], "scope": k["scope"], "position": k["position"],
                            "dur_us": round(k["dur"], 3), "stream": k["stream"]} for k in d.get("kernels", [])]}
           for u, d in run.items()}
    path = MEASURED / "layers_raw" / f"L{L}_{routing}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


def trace_check(parsed: dict) -> dict:
    from ppimpl import kernels as km

    run = parsed[min(parsed)]
    names = [k["name"] for k in run if k["unit"] not in (None, "output_head")]
    chk = km.check_trace(names, 0, optional=frozenset({19, 20, 21}))
    return {"ok": chk.ok, "summary": chk.summary(), "matched": {c: sorted(set(v)) for c, v in chk.matched.items()}}


def bench(args) -> None:
    import torch
    from ppimpl.model import pp_patch
    from ppimpl.model.stage import CHECKPOINT, StageModel, bench_config

    targets = load_targets()
    lengths = [int(x) for x in args.lengths.split(",")]
    routings = args.routing.split(",")
    dev = torch.device("cuda", 0)
    t0 = time.time()
    cfg = bench_config(max(lengths))
    model = StageModel(cfg, 0, dev, warmup=False, last=True, num_blocks=NUM_BLOCKS,
                       max_model_len=max(max(lengths), 8192) + 8576)
    print(f"[layers] model built in {time.time() - t0:.0f} s; layers {model.layer_range}; "
          f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    for layer in model.layers:
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is not None and hasattr(gate, "e_score_correction_bias"):
            pp_patch.force_routing(gate, Routing.get)
    install_scopes(model)
    cpm = sleep_cycles_per_ms()
    tokens_all = enwik9_tokens(CHECKPOINT, max(lengths))
    out_path = MEASURED / "layers_parts" / f"{args.tag or os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(tempfile.mkdtemp(prefix="ppm-", dir=os.environ.get("TMPDIR")))
    if args.autotune:
        from flashinfer.autotuner import autotune
    for L in lengths:
        tL = time.time()
        plan = one_request_plan(tokens_all[:L])
        counts = targets[(L, "dsa_moe")]["moe_per_expert_batches"]
        forced = pp_patch.forced_routing_logits(counts, L, 8, dev)
        bad = check_forced(model, forced, counts)
        extra_common = {"forced_layers_off": bad, "autotune": bool(args.autotune)}
        try:
            for routing in routings:
                Routing.logits = forced if routing == "forced" else None
                Scopes.on = True
                with torch.inference_mode():
                    ctx = autotune(True) if args.autotune else contextlib.nullcontext()
                    with ctx:
                        model.forward(plan, None)
                    torch.cuda.synchronize()
                    launch = []
                    for _ in range(2):
                        torch.cuda.synchronize()
                        torch.cuda._sleep(int(1000 * cpm))   # keep the GPU busy so host launch time is measured
                        h0 = time.perf_counter()
                        model.forward(plan, None)
                        launch.append((time.perf_counter() - h0) * 1e3)
                        torch.cuda.synchronize()
                    lead = 1.5 * max(launch) + 5
                    stats0 = torch.cuda.memory_stats()
                    torch.cuda.reset_peak_memory_stats()
                    syncs = host_syncs(model, plan)
                    stats1 = torch.cuda.memory_stats()
                    mem = {"alloc_retries": stats1.get("num_alloc_retries", 0) - stats0.get("num_alloc_retries", 0),
                           "cuda_mallocs": stats1.get("num_device_alloc", 0) - stats0.get("num_device_alloc", 0),
                           "cuda_frees": stats1.get("num_device_free", 0) - stats0.get("num_device_free", 0),
                           "peak_gib": round(stats1.get("allocated_bytes.all.peak", 0) / 2**30, 1)}
                    parsed, tok = profile_runs(model, plan, args.runs, lead, cpm, trace_dir, f"L{L}_{routing}")
                    Scopes.on = False
                    gap_parsed, _ = profile_runs(model, plan, 3, 0, cpm, trace_dir, f"L{L}_{routing}_eager")
                    Scopes.on = True
                extra = dict(extra_common, host_launch_ms=round(max(launch), 2), lead_ms=round(lead, 1),
                             host_syncs=syncs, memory=mem,
                             gaps=host_gaps(gap_parsed), token=int(tok[0]))
                lines = aggregate(L, routing, parsed, targets, extra)
                if routing == routings[0]:
                    extra["trace_check"] = trace_check(parsed)
                    lines[-1]["trace_check"] = extra["trace_check"]
                with out_path.open("a") as f:
                    for line in lines:
                        f.write(json.dumps(line) + "\n")
                raw_dump(L, routing, parsed)
                summary = " ".join(f"{x['layer_type']}={x['real_ms']:.3f}({x['ratio']})" for x in lines)
                print(f"[layers] L={L} {routing}: {summary}; launch {max(launch):.1f} ms; "
                      f"eager gap {extra['gaps']['eager_gap_ms']} ms; host syncs {len(syncs)}; memory {mem}", flush=True)
                for sync in syncs:
                    print(f"[layers]   sync x{sync['count']}: {sync['message'][:80]} @ {sync['where']}", flush=True)
        except Exception as e:  # noqa: BLE001 - record and go on to the next length
            traceback.print_exc()
            with out_path.open("a") as f:
                f.write(json.dumps({"prompt_tokens": L, "layer_type": "error", "routing": ",".join(routings),
                                    "error": f"{type(e).__name__}: {e}"[:2000]}) + "\n")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(f"[layers] L={L} done in {time.time() - tL:.0f} s", flush=True)
    summarize()


# ================================================================ summary


def _latest(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["prompt_tokens"], r["layer_type"], r["routing"])] = r
    return out


def _fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def merge_parts() -> Path:
    """measured/layers_parts/*.jsonl (one per job, oldest first) -> measured/layers.jsonl."""
    parts = sorted((MEASURED / "layers_parts").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    path = MEASURED / "layers.jsonl"
    if parts:
        path.write_text("".join(p.read_text() for p in parts))
    return path


def summarize() -> None:
    rows = _latest(merge_parts())
    targets = load_targets()
    Ls = sorted({k[0] for k in rows})
    head = "| Layer type | " + " | ".join(f"{L // 1024}k" for L in Ls) + " |"
    sep = "|---|" + "---|" * len(Ls)
    md = ["# M2: per-layer alignment (3_layers.md)", "",
          "Generated by `bench/layers.py --summarize` from `measured/layers.jsonl`. Bench model: stage 0 (layers 0-11) "
          "with the output head, one request of L enwik9 tokens in one call, one B200. Layer time = GPU span of the "
          "layer's kernels (CUPTI), mean over its instances in stage 0 and the runs, launch queue ahead of the GPU.",
          ""]
    for routing in ("forced", "router"):
        if not any(k[2] == routing for k in rows):
            continue
        md += [f"## Ratio real / target, {routing} routing", "", head, sep]
        for lt in TYPES + ("stage0_forward",):
            cells = []
            for L in Ls:
                r = rows.get((L, lt, routing))
                cells.append("–" if r is None or r.get("ratio") is None else
                             (f"**{r['ratio']:.3f}**" if r["ratio"] > 1.10 else f"{r['ratio']:.3f}"))
            md.append(f"| `{lt}` | " + " | ".join(cells) + " |")
        md += ["", f"Real ms ({routing}):", "", head, sep]
        for lt in TYPES + ("stage0_forward",):
            md.append(f"| `{lt}` | " + " | ".join(_fmt((rows.get((L, lt, routing)) or {}).get("real_ms")) for L in Ls)
                      + " |")
        md.append("")
    if any(k[2] == "router" for k in rows):
        md += ["## Forced / router time, MoE layers", "", head, sep]
        for lt in MOE_TYPES:
            cells = []
            for L in Ls:
                f, r = rows.get((L, lt, "forced")), rows.get((L, lt, "router"))
                cells.append("–" if not (f and r) else f"{f['real_ms'] / r['real_ms']:.3f}")
            md.append(f"| `{lt}` | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Stage sums (forced routing) vs README", "",
           "Measured layer times added by §2; stage 0 also measured directly (`stage0_forward`, includes the output "
           "head).", "", "| Prompt | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 0 direct |", "|---|---|---|---|---|---|"]
    for L in Ls:
        cells = []
        for s, comp in STAGE_COMPOSITION.items():
            parts = [rows.get((L, lt, "forced")) for lt in comp]
            if any(p is None for p in parts):
                cells.append("–")
                continue
            tot = sum(n * rows[(L, lt, "forced")]["real_ms"] for lt, n in comp.items())
            tgt = README_STAGE_MS[L][s]
            ratio = tot / tgt
            cells.append(f"{tot:.1f} / {tgt} ({'**' if ratio > 1.10 else ''}{ratio:.3f}{'**' if ratio > 1.10 else ''})")
        d = rows.get((L, "stage0_forward", "forced"))
        cells.append("–" if d is None else f"{d['real_ms']:.1f}")
        md.append(f"| {L // 1024}k | " + " | ".join(cells) + " |")
    md += ["", "## Host gaps of plain eager launching (no queue lead, no module scopes)", "",
           "| Prompt | routing | forward span ms | GPU busy ms | gap ms | host launch ms |", "|---|---|---|---|---|---|"]
    for (L, lt, routing), r in sorted(rows.items()):
        if lt == "stage0_forward" and r.get("gaps"):
            g = r["gaps"]
            md.append(f"| {L // 1024}k | {routing} | {g['eager_forward_ms']} | {g['eager_busy_ms']} | "
                      f"{g['eager_gap_ms']} | {r.get('host_launch_ms')} |")
    md += ["", "## Misses (ratio > 1.10): positions that grew most", ""]
    misses = 0
    for (L, lt, routing), r in sorted(rows.items()):
        if lt not in TYPES or r.get("ratio") is None or r["ratio"] <= 1.10:
            continue
        misses += 1
        grown = sorted((k for k in r["kernels"] if k["target_ms"] is not None),
                       key=lambda k: -(k["real_ms"] - k["target_ms"]))[:4]
        null = next((k for k in r["kernels"] if k["position"] is None), None)
        desc = "; ".join(f"`{k['position']}` {k['real_ms']:.3f} vs {k['target_ms']:.3f}" for k in grown)
        if null:
            desc += f"; unmapped kernels {null['real_ms']:.3f}"
        md.append(f"- {L // 1024}k `{lt}` {routing}: {r['real_ms']:.3f} vs {r['target_ms']:.3f} ms "
                  f"(x{r['ratio']:.3f}): {desc}")
    if not misses:
        md.append("None.")
    md += ["", "## Per-position real vs target (forced), 8k, 64k and 128k", ""]
    for L in (8192, 65536, 131072):
        for lt in TYPES:
            r = rows.get((L, lt, "forced"))
            if r is None:
                continue
            md += [f"### {L // 1024}k `{lt}`: {r['real_ms']:.3f} vs {_fmt(r['target_ms'])} ms", "",
                   "| Position | real ms | target ms | GPU kernels |", "|---|---|---|---|"]
            for k in r["kernels"]:
                names = ", ".join(f"`{n[:60]}`" for n in k["gpu_kernels"][:3])
                md.append(f"| `{k['position']}` | {k['real_ms']:.4f} | {_fmt(k['target_ms'], 4)} | {names} |")
            md.append("")
    errors = [r for k, r in rows.items() if k[1] == "error"]
    if errors:
        md += ["## Errors", ""] + [f"- {r['prompt_tokens']}: {r['error'][:300]}" for r in errors] + [""]
    (MEASURED / "layers.md").write_text("\n".join(md) + "\n")
    print(f"[layers] wrote {MEASURED / 'layers.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default=",".join(str(L) for L in LENGTHS))
    ap.add_argument("--routing", default="forced,router")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--no-autotune", dest="autotune", action="store_false")
    ap.add_argument("--tag", default=None, help="output part name (default: the Slurm job id)")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        bench(args)


if __name__ == "__main__":
    main()
