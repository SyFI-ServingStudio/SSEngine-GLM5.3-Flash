"""GPU tests of the stage-edge send and the indexer's cast (one GPU, through Slurm: pytest -m gpu
tests/test_glue_gpu.py).

The stage edge's hc_post runs after the graph replay, into a new tensor (`StageModel._run_graph`; job 7919 proved it
bitwise equal to the hc_post captured in the graph and copied out). Here: one stage model, graphs captured for two
buckets, a sequence of microbatches (fresh, continuing a cached prefix, padded, two requests) replayed and run eagerly
on the same buffers (`replay=False`, the exactness reference): every sent row bitwise equal.
"""

import pytest

from ppimpl.engine.kv import block_pages

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("stage", [0, 2])
def test_edge_send_after_replay_matches_eager(stage):
    from types import SimpleNamespace as NS

    import torch

    from ppimpl.model.stage import StageModel, bench_config

    dev = torch.device("cuda", 0)
    B = 8576
    model = StageModel(bench_config(8192, cuda_graphs=False), stage, dev, num_blocks=64, max_model_len=8192 + B)
    assert not model.last

    def chunk(rid, start, n, kv):
        return NS(req_id=rid, start=start, num_tokens=n, kv_pages=block_pages([kv]), is_last=True,
                  state_blocks=list(range(kv + 1, kv + 7)), token_ids=[(7 * i + 11) % 50000 + 1 for i in range(n)])

    plans = [NS(mb_id=0, chunks=[chunk("a", 0, 4096, 1)]),                         # fresh, exactly the bucket
             NS(mb_id=1, chunks=[chunk("a", 4096, 3001, 1), chunk("b", 0, 999, 8)]),  # continuation + fresh, padded
             NS(mb_id=2, chunks=[chunk("c", 0, 1024, 15)]),                          # another bucket
             NS(mb_id=3, chunks=[chunk("b", 999, 700, 8)])]                          # continuation, padded 1024

    def run_all(replay: bool):
        g = torch.Generator(device=dev).manual_seed(3)
        outs = []
        with torch.inference_mode():
            for p in plans:
                T = sum(c.num_tokens for c in p.chunks)
                h = None if model.first else (torch.randn(T, model.hidden_width, generator=g, device=dev) * 0.5).to(
                    torch.bfloat16)
                b = model.prepare(p)
                assert b.bucket is not None
                outs.append(model._run_graph(b, h, replay=replay).clone())
        torch.cuda.synchronize(dev)
        return outs

    model.capture_graphs(buckets=[1024, 4096])
    old = run_all(replay=False)
    new = run_all(replay=True)
    assert all(bool(torch.isfinite(o.float()).all()) for o in old)
    for i, (x, y) in enumerate(zip(old, new)):
        assert x.shape == y.shape and torch.equal(x, y), (i, (x.float() - y.float()).abs().max().item())


def test_indexer_float_matches_torch_cast():
    """pp_patch `install_indexer_float` (glue_fast.to_float for the indexer's `hidden_states.float()`): on stage 2
    (3 DSA layers), the same microbatches eagerly with the fork's forward and with the rewritten one give the same top-k
    indices from every indexer call and the same sent rows, bitwise; then graphs captured with it on replay the
    first microbatch (exactly the bucket) to the same rows."""
    from types import SimpleNamespace as NS

    import torch
    from vllm.models.glm5next.common.attention import Indexer

    from ppimpl.model import glue_fast
    from ppimpl.model.stage import StageModel, bench_config

    dev = torch.device("cuda", 0)
    B = 8576
    x = torch.randn(4096, 4096, device=dev).to(torch.bfloat16)
    assert torch.equal(glue_fast.to_float(x), x.float())
    assert torch.equal(glue_fast.to_float(x[:4095]), x[:4095].float())
    model = StageModel(bench_config(8192, cuda_graphs=False), 2, dev, num_blocks=64, max_model_len=8192 + B)
    indexers = [m for m in model.model.modules() if isinstance(m, Indexer)]
    assert len(indexers) == 3
    seen = []
    for m in indexers:
        m.register_forward_hook(lambda mod, args, out: seen.append(out.clone()))

    def chunk(rid, start, n, kv):
        return NS(req_id=rid, start=start, num_tokens=n, kv_pages=block_pages([kv]), is_last=True,
                  state_blocks=list(range(kv + 1, kv + 7)), token_ids=None)

    plans = [NS(mb_id=0, chunks=[chunk("a", 0, 4096, 1)]),
             NS(mb_id=1, chunks=[chunk("a", 4096, 3001, 1), chunk("b", 0, 999, 8)]),
             NS(mb_id=2, chunks=[chunk("b", 999, 2500, 8)])]

    def run_all():
        g = torch.Generator(device=dev).manual_seed(5)
        seen.clear()
        outs = []
        with torch.inference_mode():
            for p in plans:
                T = sum(c.num_tokens for c in p.chunks)
                h = (torch.randn(T, model.hidden_width, generator=g, device=dev) * 0.5).to(torch.bfloat16)
                outs.append(model.forward(p, h).clone())
        torch.cuda.synchronize(dev)
        return outs, list(seen)

    glue_fast.uninstall_indexer_float()
    old, old_idx = run_all()
    glue_fast.install_indexer_float()
    try:
        new, new_idx = run_all()
        assert len(old_idx) == len(new_idx) == 3 * len(plans)
        for i, (a, b) in enumerate(zip(old_idx, new_idx)):   # the microbatch's rows (later rows: other calls' data)
            T = sum(c.num_tokens for c in plans[i // 3].chunks)
            assert torch.equal(a[:T], b[:T]), ("indexer", i)
        for i, (a, b) in enumerate(zip(old, new)):
            assert torch.equal(a, b), ("sent rows", i)
        model.capture_graphs(buckets=[4096])
        with torch.inference_mode():
            g = torch.Generator(device=dev).manual_seed(5)
            h = (torch.randn(4096, model.hidden_width, generator=g, device=dev) * 0.5).to(torch.bfloat16)
            b = model.prepare(plans[0])
            assert b.bucket == 4096
            rep = model._run_graph(b, h).clone()
        torch.cuda.synchronize(dev)
        assert torch.equal(rep, old[0])
    finally:
        glue_fast.install_indexer_float()   # as the engine runs
