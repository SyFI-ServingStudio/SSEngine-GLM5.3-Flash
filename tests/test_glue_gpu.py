"""GPU test of pp_patch FAST["edge_send_eager"] (one GPU, through Slurm: pytest -m gpu tests/test_glue_gpu.py).

The stage edge's hc_post run after the graph replay, into a new tensor, must send the same bits as the graph that
computes it and whose output is then copied (`StageModel._run_graph`). One stage model, graphs captured for two
buckets with the switch off, a sequence of microbatches run (fresh, continuing a cached prefix, padded, two
requests), then recaptured with the switch on and the same sequence rerun: every sent row bitwise equal.
"""

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("stage", [0, 2])
def test_edge_send_eager_matches_graph_copy(stage):
    from types import SimpleNamespace as NS

    import torch

    from ppimpl.model.stage import StageModel, bench_config

    dev = torch.device("cuda", 0)
    B = 8576
    model = StageModel(bench_config(8192, cuda_graphs=False), stage, dev, num_blocks=64, max_model_len=8192 + B)
    assert not model.last

    def chunk(rid, start, n, kv):
        return NS(req_id=rid, start=start, num_tokens=n, kv_blocks=[kv], is_last=True,
                  state_blocks=list(range(kv + 1, kv + 7)), token_ids=[(7 * i + 11) % 50000 + 1 for i in range(n)])

    plans = [NS(mb_id=0, chunks=[chunk("a", 0, 4096, 1)]),                         # fresh, exactly the bucket
             NS(mb_id=1, chunks=[chunk("a", 4096, 3001, 1), chunk("b", 0, 999, 8)]),  # continuation + fresh, padded
             NS(mb_id=2, chunks=[chunk("c", 0, 1024, 15)]),                          # another bucket
             NS(mb_id=3, chunks=[chunk("b", 999, 700, 8)])]                          # continuation, padded 1024

    def run_all():
        g = torch.Generator(device=dev).manual_seed(3)
        outs = []
        with torch.inference_mode():
            for p in plans:
                T = sum(c.num_tokens for c in p.chunks)
                h = None if model.first else (torch.randn(T, model.hidden_width, generator=g, device=dev) * 0.5).to(
                    torch.bfloat16)
                b = model.prepare(p)
                assert b.bucket is not None
                outs.append(model._run_graph(b, h).clone())
        torch.cuda.synchronize(dev)
        return outs

    model.edge_send_eager = False
    model.capture_graphs(buckets=[1024, 4096])
    old = run_all()
    model.graphs = {}
    model.edge_send_eager = True
    model.capture_graphs(buckets=[1024, 4096])
    new = run_all()
    assert all(bool(torch.isfinite(o.float()).all()) for o in old)
    for i, (x, y) in enumerate(zip(old, new)):
        assert x.shape == y.shape and torch.equal(x, y), (i, (x.float() - y.float()).abs().max().item())
