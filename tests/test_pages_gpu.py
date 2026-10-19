"""GPU test of the KV page layout (one GPU, through Slurm: pytest -m gpu tests/test_pages_gpu.py).

The controller hands out KV in 128-token pages (`config.PAGE_TOKENS`, engine/kv.py), anywhere in the pool, not in
whole 8,576-token blocks. The g0 kernels (FlashInfer sparse MLA through 64-token kernel blocks, the kpool indexer's
DeepGEMM paged MQA through its 128-token storage pages, the K insert and the indexer's decode path) take them through
their page tables (`StageMetadata.build`). Here: one stage model runs the same microbatches (two requests: fresh,
continuing across a block boundary, starting inside a kpool pool, decode-class appends) once on whole blocks in order
(the layout before pages) and once on pages scattered over shared blocks; every output row is bitwise equal.
"""

import random

import pytest

from ppimpl.config import BLOCK_PAGES
from ppimpl.engine.kv import block_pages, kv_pages_for

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("stage", [0, 3])
def test_scattered_pages_match_whole_blocks(stage):
    from types import SimpleNamespace as NS

    import torch

    from ppimpl.model.stage import StageModel, bench_config

    dev = torch.device("cuda", 0)
    B = 8576
    model = StageModel(bench_config(8192, cuda_graphs=False), stage, dev, num_blocks=64, max_model_len=4 * B)
    toks = {r: [(13 * i + 7 * len(r)) % 50000 + 1 for i in range(2 * B)] for r in ("a", "b")}
    # (request, start, tokens) per microbatch: a crosses the 8,576 boundary; b resumes at 1,501 (inside a pool of 4)
    mbs = [[("a", 0, 6000), ("b", 0, 1501)],
           [("a", 6000, 6000), ("b", 1501, 2001)],
           [("a", 12000, 100), ("b", 3502, 255)],             # decode-class rows (<= 256 tokens)
           [("b", 3757, 4000), ("a", 12100, 1)]]
    need = {r: kv_pages_for(max(s + n for x in mbs for q, s, n in x if q == r)) for r in ("a", "b")}
    states = {"a": list(range(50, 56)), "b": list(range(56, 62))}
    whole = {"a": block_pages([1, 2])[:need["a"]], "b": block_pages([3])[:need["b"]]}
    pool = block_pages(range(10, 40))
    random.Random(7).shuffle(pool)
    scattered = {"a": pool[:need["a"]], "b": pool[need["a"]:need["a"] + need["b"]]}
    assert len({p // BLOCK_PAGES for p in scattered["a"]} & {p // BLOCK_PAGES for p in scattered["b"]}) > 0

    def run(pages):
        g = torch.Generator(device=dev).manual_seed(5)
        outs = []
        with torch.inference_mode():
            for i, mb in enumerate(mbs):
                chunks = [NS(req_id=r, start=s, num_tokens=n, kv_pages=pages[r], is_last=True,
                             state_blocks=states[r], token_ids=toks[r][s:s + n]) for r, s, n in mb]
                T = sum(n for _, _, n in mb)
                h = None if model.first else (torch.randn(T, model.hidden_width, generator=g, device=dev) * 0.5).to(
                    torch.bfloat16)
                outs.append(model.forward(NS(mb_id=i, chunks=chunks), h).clone())
        torch.cuda.synchronize(dev)
        return outs

    ref = run(whole)
    got = run(scattered)
    assert all(bool(torch.isfinite(o.float()).all()) for o in ref)
    for i, (x, y) in enumerate(zip(ref, got)):
        assert x.shape == y.shape and torch.equal(x, y), (i, (x.float() - y.float()).abs().max().item())
