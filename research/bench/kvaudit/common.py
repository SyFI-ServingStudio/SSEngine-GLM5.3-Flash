"""KV-cache audit: what is stored after prefill, vLLM TP4+EP (the reference) against pp-impl PP4, layer by layer.

    # vLLM side (4 GPUs): one in-process engine, the cases' prompts one after another, dump after each
    python -m research.bench.kvaudit.vllm_side --length 8192 --out DIR
    # our side (4 GPUs): StageModel per stage over the CE pipe, the cases' plans, dump after each
    python -m research.bench.kvaudit.ours --length 8192 --out DIR
    # compare (CPU or 1 GPU): every pair of `cases(L)`, per layer and tensor -> DIR/compare.json
    python -m research.bench.kvaudit.compare --length 8192 --out DIR [--device cuda]
    # tables: research/measured/kvaudit.md from one or more compare.json
    python -m research.bench.kvaudit.report DIR...

Every cache tensor either side keeps per request, in one canonical form (both sides read them through the same
vLLM layer objects, `static_forward_context[name].kv_cache`; the block size differs: ours 8576 tokens, vLLM's
TP4 hybrid alignment picks its own):

| Layer kind | Tensor | Canonical form | vLLM TP4 |
|---|---|---|---|
| DSA (3, 7, ..., 43) | MLA latent `self_attn.attn` | uint8 [T, 512]: fp8_e4m3 latent (no-rope MLA: qk_rope_head_dim 0), x `_k_scale` | replicated |
| DSA | indexer K `self_attn.indexer.k_cache` | per complete pool (T // 4): fp8 [128] + fp32 scale, gathered with vLLM's own `cp_gather_indexer_k_quant_cache` | replicated |
| DSA | kpool tail `self_attn.indexer.tail_cache` | bf16 [2 (K, gate), 4 (pos % 4), 128]: the request's 1 block; slots < T % 4 are the open pool | replicated |
| KDA (the other 34) | conv state | bf16 [3, 3 x 8192] (window, q|k|v channels) | channels sharded per q/k/v: reassembled |
| KDA | recurrent state | fp32 [64, 128, 128] | heads sharded: concatenated in head order |

Replicated tensors: vLLM rank 0 writes them, ranks 1-3 send a sha256 of their canonical bytes (`replicas` in
meta.json). Also per case: the full-vocabulary log-probabilities of the first output token (vLLM: `logprobs=-1`;
ours: log_softmax of stage 3's fp32 logits) and the indexer's top-k rows for the last `TOPK_ROWS` positions of every
DSA layer (a forward hook on each `Indexer` module).

Dump layout: DIR/<case>/layer<ii>.pt (dict of CPU tensors), logprobs.pt, topk.pt, meta.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from benchmarks.common import MEASURED

PREFIX = "language_model.model.layers"
NUM_LAYERS = 45
DSA_LAYERS = tuple(range(3, 45, 4))
KDA_LAYERS = tuple(i for i in range(NUM_LAYERS) if i not in DSA_LAYERS)
KPOOL = 4
TOPK_ROWS = 8
FILLER = 1000
CHUNK = 8192
OUT = MEASURED / "kvaudit"
DUMPS = MEASURED.parent / "tmp" / "kvaudit"          # big dumps, deleted after compare (git-ignored tmp/)


# ------------------------------------------------------------------------------------------------- cases


@dataclass
class OursCase:
    name: str
    rounds: list[list[int]]          # per round, its chunk sizes; round r starts where round r-1 ended
    tokens: int = 0

    def __post_init__(self):
        self.tokens = sum(sum(r) for r in self.rounds)


@dataclass
class VllmCase:
    name: str
    tokens: int
    filler_first: bool = False       # a 1,000-token request scheduled before it: other batch, other chunk starts


@dataclass
class Cases:
    length: int
    vllm: list[VllmCase] = field(default_factory=list)
    ours: list[OursCase] = field(default_factory=list)
    pairs: list[tuple[str, str, str]] = field(default_factory=list)   # (x, reference, kind)

    def max_tokens(self) -> int:
        return max([c.tokens for c in self.vllm] + [c.tokens for c in self.ours])


def chunks(n: int, size: int = CHUNK, merge_below: int = 512) -> list[int]:
    """`n` tokens in chunks of `size`; a remainder below `merge_below` joins the previous chunk (so a round never ends
    in a decode-class chunk unless asked)."""
    out = [size] * (n // size)
    rem = n - size * len(out)
    if rem:
        if out and rem < merge_below:
            out[-1] += rem
        else:
            out.append(rem)
    return out


def cases(L: int) -> Cases:
    """The audit at prompt length L.

    vLLM: v_full_<N> is the prompt [0, N) alone; v_filler_<L> the same prompt with the filler request first.
    Ours: o_whole_<L> (one microbatch; up to 128k), o_c8192_<L> / o_c8001_<L> (chunks of 8192 / 8001, 3001 at 8k: the
    latter start chunks inside kpool pools), o_full_<N> (chunks of 8192), o_prefix_<P>_<A> (round 0 = P tokens in chunks of
    8192, round 1 = A tokens resuming at P, one microbatch: a prefix hit; A <= 256 is a decode-class row).
    """
    c = Cases(L)
    P, A = L + 3, 1021                     # prefill-class prefix hit, resume start inside a pool (P % 4 = 3)
    dec = {8192: [(3000, 256), (3000, 3)], 131072: [(100003, 256), (100003, 3)]}.get(L, [])
    c.vllm = [VllmCase(f"v_full_{L}", L), VllmCase(f"v_filler_{L}", L, filler_first=True),
              VllmCase(f"v_full_{P + A}", P + A)] + [VllmCase(f"v_full_{p + a}", p + a) for p, a in dec]
    if L <= 131072:
        c.ours.append(OursCase(f"o_whole_{L}", [[L]]))
    if L > CHUNK:
        c.ours.append(OursCase(f"o_c8192_{L}", [chunks(L)]))
    odd = 8001 if L > 2 * CHUNK else 3001   # chunk starts inside pools (8001 % 4 = 1); 3001 at 8k (8001 would merge)
    c.ours.append(OursCase(f"o_c{odd}_{L}", [chunks(L, odd)]))
    c.ours.append(OursCase(f"o_full_{P + A}", [chunks(P + A)]))
    c.ours.append(OursCase(f"o_prefix_{P}_{A}", [chunks(P), [A]]))
    for p, a in dec:
        c.ours.append(OursCase(f"o_full_{p + a}", [chunks(p + a)]))
        c.ours.append(OursCase(f"o_prefix_{p}_{a}", [chunks(p), [a]]))
    # noise for the kpool tail: the noise pairs above end on a pool boundary (no open pool); these end inside one
    tails = [p + a for p, a in dec if (p + a) % KPOOL]
    for n in tails:
        odd = 8001 if n > 2 * CHUNK else (n // 2) | 1
        c.vllm.append(VllmCase(f"v_filler_{n}", n, filler_first=True))
        c.ours.append(OursCase(f"o_c{odd}_{n}", [chunks(n, odd)]))

    base = f"o_whole_{L}" if L <= 131072 else f"o_c8192_{L}"
    vref = f"v_full_{L}"
    c.pairs.append((f"v_filler_{L}", vref, "noise_vllm"))
    for o in c.ours:
        if o.name.startswith(("o_whole", "o_c")) and o.tokens == L:
            c.pairs.append((o.name, vref, "audit"))
            if o.name != base:
                c.pairs.append((o.name, base, "noise_ours"))
    for n in tails:
        odd = 8001 if n > 2 * CHUNK else (n // 2) | 1
        c.pairs += [(f"v_filler_{n}", f"v_full_{n}", "noise_vllm"), (f"o_c{odd}_{n}", f"o_full_{n}", "noise_ours")]
    for p, a in [(P, A)] + dec:
        full_o, full_v, pre = f"o_full_{p + a}", f"v_full_{p + a}", f"o_prefix_{p}_{a}"
        c.pairs += [(full_o, full_v, "audit"), (pre, full_v, "audit_prefix"), (pre, full_o, "prefix_vs_ours_full")]
    return c


def prompt(n: int) -> list[int]:
    """The first n tokens of enwik9 (benchmarks/common.py), as every pp-impl check uses."""
    from benchmarks.common import enwik9_tokens
    from ssengine.config import CHECKPOINT

    return enwik9_tokens(CHECKPOINT, n)


def filler(n_main: int) -> list[int]:
    """1,000 tokens from later in enwik9 than any case's prompt."""
    return prompt(n_main + 2 * FILLER)[n_main + FILLER:n_main + 2 * FILLER]


# ------------------------------------------------------------------------------------------------- extraction


def kv_of(layer):
    kv = layer.kv_cache
    if isinstance(kv, (list, tuple)) and len(kv) == 1:
        kv = kv[0]
    return kv


def mla_tokens(kv: torch.Tensor, blocks: list[int], T: int, block_tokens: int) -> torch.Tensor:
    """uint8 [T, entry]: token t's MLA cache entry, from logical block blocks[t // block_tokens] (kernel blocks of
    kv.shape[1] tokens: kernel block b * (block_tokens / kv.shape[1]) + j, as both metadata builders map them)."""
    kb_tokens = kv.shape[1]
    assert block_tokens % kb_tokens == 0, (block_tokens, kv.shape)
    kpb = block_tokens // kb_tokens
    dev = kv.device
    t = torch.arange(T, device=dev)
    b = torch.as_tensor(blocks, device=dev, dtype=torch.int64)[t // block_tokens]
    w = t % block_tokens
    rows = kv[b * kpb + w // kb_tokens, w % kb_tokens]
    return rows.reshape(T, -1).view(torch.uint8)


def indexer_table(blocks: list[int], npools: int, block_tokens: int, cache_block: int) -> list[int]:
    """Gather table of the indexer K cache in its own pages of `cache_block` pools, as the kpool indexer builder
    derives it (block_table[:, ::factor] // factor): logical block b -> pages b * ppb .. b * ppb + ppb - 1."""
    per_block = block_tokens // KPOOL
    assert per_block % cache_block == 0, (block_tokens, cache_block)
    ppb = per_block // cache_block
    out = []
    for i in range(-(-npools // cache_block)):
        p0 = i * cache_block
        out.append(blocks[p0 // per_block] * ppb + (p0 % per_block) // cache_block)
    return out


def indexer_pools(kv: torch.Tensor, blocks: list[int], T: int, block_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp8 bytes uint8 [T // 4, 128], fp32 scale [T // 4]) of every complete pool."""
    from vllm import _custom_ops as ops

    npools = T // KPOOL
    dev = kv.device
    table = torch.tensor([indexer_table(blocks, npools, block_tokens, kv.shape[1])], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, npools], dtype=torch.int32, device=dev)
    k = torch.empty(npools, 128, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty(npools, 4, dtype=torch.uint8, device=dev)
    if npools:
        ops.cp_gather_indexer_k_quant_cache(kv, k, s, table, cu)
    return k.view(torch.uint8), s.view(torch.float32).reshape(npools)


def dsa_layer(ctx: dict, i: int, g0_blocks: list[int], tail_block: int, T: int, block_tokens: int) -> dict:
    mla_l = ctx[f"{PREFIX}.{i}.self_attn.attn"]
    mla = kv_of(mla_l)
    k, s = indexer_pools(kv_of(ctx[f"{PREFIX}.{i}.self_attn.indexer.k_cache"]), g0_blocks, T, block_tokens)
    tail = kv_of(ctx[f"{PREFIX}.{i}.self_attn.indexer.tail_cache"])[tail_block]
    ks = getattr(mla_l, "_k_scale", None)
    return {"mla": mla_tokens(mla, g0_blocks, T, block_tokens).cpu(), "mla_dtype": str(mla.dtype),
            "mla_scale": float(ks) if ks is not None else 1.0,
            "idx_k": k.cpu(), "idx_s": s.cpu(), "tail": tail.clone().cpu(), "tail_shape": list(tail.shape),
            "tail_valid": T % KPOOL}


def conv_canonical(conv: torch.Tensor, width: int = 3) -> torch.Tensor:
    """One request's conv state as [width, channels] (SD layout; DS stores [channels, width])."""
    return conv if conv.shape[0] == width else conv.t()


def kda_layer(ctx: dict, i: int, state_block: int) -> dict:
    conv, rec = kv_of(ctx[f"{PREFIX}.{i}.self_attn"])
    return {"conv": conv_canonical(conv[state_block]).clone().cpu(), "rec": rec[state_block].clone().cpu()}


def merge_kda_shards(shards: list[dict]) -> dict:
    """vLLM TP shards (rank order) -> our layout: recurrent heads concatenated; conv channels per q/k/v section."""
    rec = torch.cat([s["rec"] for s in shards], dim=0)
    convs = [s["conv"] for s in shards]
    c = convs[0].shape[1] // 3
    conv = torch.cat([torch.cat([x[:, j * c:(j + 1) * c] for x in convs], dim=1) for j in range(3)], dim=1)
    return {"conv": conv, "rec": rec}


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def dsa_fingerprint(d: dict) -> dict:
    return {k: sha(d[k]) for k in ("mla", "idx_k", "idx_s", "tail")}


# ------------------------------------------------------------------------------------------------- top-k hook

TOPK_CAPTURE: dict = {"lo": None, "hi": None, "rows": {}}


def install_topk_hooks(model) -> int:
    """Forward hook on every `Indexer` module: rows whose position is in [lo, hi) -> TOPK_CAPTURE["rows"][layer]
    (positions, the top-k index row). The hook reads the buffer the indexer returns, right after it ran."""
    import re

    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Indexer":
            continue
        layer = int(re.search(r"layers\.(\d+)\.", name).group(1))

        def hook(mod, args, kwargs, out, layer=layer):
            lo, hi = TOPK_CAPTURE["lo"], TOPK_CAPTURE["hi"]
            if lo is None:
                return
            pos = kwargs.get("positions") if "positions" in kwargs else args[2]
            sel = ((pos >= lo) & (pos < hi)).nonzero().flatten()
            if sel.numel():
                TOPK_CAPTURE["rows"].setdefault(layer, []).append((pos[sel].cpu(), out[sel].cpu()))

        mod.register_forward_hook(hook, with_kwargs=True)
        n += 1
    return n


def topk_start(T: int) -> None:
    TOPK_CAPTURE.update(lo=T - TOPK_ROWS, hi=T, rows={})


def topk_take() -> dict:
    rows = TOPK_CAPTURE["rows"]
    TOPK_CAPTURE.update(lo=None, hi=None, rows={})
    out = {}
    for layer, parts in rows.items():
        pos = torch.cat([p for p, _ in parts])
        idx = torch.cat([x for _, x in parts])
        out[layer] = (pos, idx)
    return out


def write_meta(d: Path, meta: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
