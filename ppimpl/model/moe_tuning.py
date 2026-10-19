"""The trtllm NVFP4 MoE's tactic buckets and the routing its tactics are tuned on.

Buckets. FlashInfer tunes the fused MoE at its "hybrid" token buckets (powers of 2 to 256, steps of 256 to 2,048,
steps of 512 to 4,096, then powers of 2) and serves every token count with the tactic of the bucket at or above it,
so 4,097-8,191-token microbatches ran 8,192's tactic. `install_buckets` (process-wide, `pp_patch` table) makes the
MoE buckets the serving shapes instead: powers of 2 to 128, then every CUDA-graph bucket (`GRAPH_BUCKETS`: steps of
256 to 8,192), then powers of 2 to the tuning bound. A token count still maps to the bucket at or above it. The
persisted tactic file holds one entry per bucket, so a file and the process reading it must use the same buckets:
`TACTICS` is tuned and read with these installed.

Routing. FlashInfer's autotuner times every tactic on inputs it makes itself: router logits `torch.rand` (U[0,1)),
the layer's real correction bias, so its routing is whatever sigmoid(U[0,1)) + bias picks (232 of 288 experts, at
most 319 tokens per expert at 4,096 tokens on layer 3) rather than what served tokens route to. `real_routing`
replaces that initializer while tuning (benchmarks/tune_tactics.py) with logits whose top 8 gives a served histogram
scaled to the profiled token count: the histogram profile of `routing_profile`, measured from microbatches the
engine routed (PPIMPL_LOG_EXPERTS, ppimpl/model/expert_log.py).
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import json
from pathlib import Path

TOP_K = 8
NUM_EXPERTS = 288


def serving_buckets(max_num_tokens: int, min_num_tokens: int = 1) -> tuple[int, ...]:
    from ppimpl.model.stage import GRAPH_BUCKETS
    out, m = [], max(1, min_num_tokens)
    while m <= min(128, max_num_tokens):
        out.append(m)
        m *= 2
    out += [b for b in GRAPH_BUCKETS if min_num_tokens <= b <= max_num_tokens]
    m = 1 << (max(GRAPH_BUCKETS)).bit_length()
    while m <= max_num_tokens:
        out.append(m)
        m *= 2
    if out[-1] != max_num_tokens:
        out.append(max_num_tokens)
    return tuple(sorted(set(out)))


def map_to_serving_bucket(x: int, max_num_tokens: int) -> int:
    """The smallest bucket >= x, clamped to [1, max_num_tokens]."""
    if x <= 0:
        return 1
    b = serving_buckets(max_num_tokens)
    return b[min(bisect.bisect_left(b, x), len(b) - 1)]


@functools.cache
def serving_bucket_mapper(max_num_tokens: int):
    # stable per bound: FlashInfer hashes the mapper into an lru_cache key
    return functools.partial(map_to_serving_bucket, max_num_tokens=max_num_tokens)


_ORIG = None


def install_buckets() -> None:
    """Process-wide: the MoE tuning config (`flashinfer.fused_moe.shared.tuning.make_moe_tuning_config`, which reads
    these module attributes on every call) buckets token counts at `serving_buckets`."""
    global _ORIG
    from flashinfer.fused_moe.shared import tuning as t

    if _ORIG is not None:
        return
    _ORIG = (t.get_hybrid_num_tokens_buckets, t.make_hybrid_bucket_mapper)
    t.get_hybrid_num_tokens_buckets = functools.cache(serving_buckets)
    t.make_hybrid_bucket_mapper = serving_bucket_mapper


# ------------------------------------------------------------------------------------------- routing for tuning

def fold(layer_counts: list[list[int]]) -> list[int]:
    """Histograms of one token count, each sorted descending and averaged position-wise (the simulator's fold);
    the remainder goes to the largest fractional parts."""
    d = len(layer_counts)
    sums = [sum(col) for col in zip(*(sorted(c, reverse=True) for c in layer_counts))]
    folded = [s // d for s in sums]
    for i in sorted(range(len(sums)), key=lambda i: (-(sums[i] % d), i))[:sum(layer_counts[0]) - sum(folded)]:
        folded[i] += 1
    return folded


def scale_hist(h: list[int], tokens: int, top_k: int = TOP_K) -> list[int]:
    """`h` rescaled to tokens * top_k routed rows (largest remainders round up), at most `tokens` per expert."""
    total, s = tokens * top_k, sum(h)
    raw = [x * total / s for x in h]
    out = [min(int(r), tokens) for r in raw]
    for i in sorted(range(len(h)), key=lambda i: (-(raw[i] - int(raw[i])), i)):
        if sum(out) >= total:
            break
        if out[i] < tokens:
            out[i] += 1
    i = 0
    while sum(out) < total:            # only when capping at `tokens` left rows over
        if out[i] < tokens:
            out[i] += 1
        i = (i + 1) % len(out)
    return out


def routing_profile(runs: list[Path], top_k: int = TOP_K) -> dict[int, list[int]]:
    """Routed rows -> folded histogram over every (microbatch, MoE layer) of that row count in the runs'
    experts_stage<S>.jsonl (all stages)."""
    by_rows: dict[int, list[list[int]]] = {}
    for run in runs:
        for f in sorted(Path(run).glob("experts_stage*.jsonl")):
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                for c in rec["counts"]:
                    by_rows.setdefault(sum(c) // top_k, []).append(c)
    return {rows: fold(hs) for rows, hs in sorted(by_rows.items())}


def profile_for(profile: dict[int, list[int]], tokens: int, min_samples_rows: int = 512) -> list[int]:
    """Histogram for `tokens`: the measured row count nearest to it (ties: the larger), among counts of at least
    `min_samples_rows` rows when the profile has any (small batches are mostly decode and pad rows), scaled to
    `tokens`."""
    keys = [k for k in profile if k >= min_samples_rows] or list(profile)
    near = min(keys, key=lambda k: (abs(k - tokens), -k))
    return scale_hist(profile[near], tokens)


def forced_logits(hist: list[int], tokens: int, dtype, device):
    """[tokens, experts] logits whose top-k under sigmoid + any correction bias spread below 1 is exactly `hist`
    (expert e gets hist[e] rows): selected logits saturate the sigmoid at 1, the rest at 0."""
    import torch

    k = sum(hist) // tokens
    # row-major fill: expert e takes the next hist[e] slots of the (tokens x k) grid in column order, so every
    # token gets k distinct experts while hist[e] <= tokens
    slots = torch.repeat_interleave(torch.arange(len(hist)), torch.tensor(hist))
    ids = slots.view(k, tokens).t()                      # column j holds a run of experts, each at most once per row
    logits = torch.full((tokens, len(hist)), -30.0)
    logits.scatter_(1, ids, 30.0)
    return logits.to(device=device, dtype=dtype)


@contextlib.contextmanager
def real_routing(profile: dict[int, list[int]]):
    """While tuning: the MoE tuning config's router-logit initializer gives `profile`'s routing at each profiled
    token count (`forced_logits`, `profile_for`)."""
    from flashinfer.fused_moe.shared import tuning as t

    used: dict[int, list[int]] = {}

    def init(shapes, dtype, device):
        tokens, experts = shapes[0], shapes[-1]
        h = profile_for(profile, tokens)
        assert len(h) == experts, (len(h), experts)
        used[tokens] = h
        return forced_logits(h, tokens, dtype, device)

    orig = t.autotuner_initializer_rand
    t.autotuner_initializer_rand = init
    try:
        yield used
    finally:
        t.autotuner_initializer_rand = orig
