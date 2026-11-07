"""Copy-on-resume on 4 GPUs (docs/design/kept_states.md, k2): real StageModels joined by the serving pipe, compact
KDA states bound over the KV tensors (`kv_tensors.bind_states`).

Round 0 computes P tokens (P off KDA's 64-token grid, chunked) into KV blocks KX and state S1. Round 1a continues the
context: its chunk carries `state_from = S1` and computes on its own state S2. S1 is unchanged by it. Round 1b, a
prompt that diverges after P, resumes from S1 again (k2's inner state) into S3, over the same KV pages. Each round 1 is
compared with a context computed without any cache reuse or copy: [0, P) then the same round-1 tokens in place, on its
own blocks and state. The logits, every state piece and every KV byte are bitwise equal. Run through Slurm:

    sbatch -p <partition> --gres=gpu:4 --time=00:20:00 -o runs/logs/%j.out benchmarks/jobs/gtest.sbatch kept_states
"""

import json
from types import SimpleNamespace as NS

import pytest

from conftest import free_port_block
from ssengine.engine.kv import block_pages

pytestmark = pytest.mark.gpu

P, A, CHUNK, B = 20_003, 1_000, 8_100, 8576
N_KV = -(-(P + A) // B)


def _plans(name, toks, start, sizes, kv, st, mb0, state_from=None):
    plans, s = [], start
    for j, n in enumerate(sizes):
        plans.append(NS(mb_id=mb0 + len(plans), chunks=[NS(
            req_id=name, start=s, num_tokens=n, kv_pages=block_pages(kv), state_blocks=st, is_last=j == len(sizes) - 1,
            token_ids=toks[s:s + n], state_from=state_from if j == 0 else None)]))
        s += n
    return plans


def _bytes(ranges, dev, kv, st, kind):
    """This stage's bytes of a context: KV rows of blocks kv, or each state piece at its run's block."""
    import torch

    from ssengine.runtime.transport import _DevMem

    out = []
    for r in ranges:
        if r.kind != kind:
            continue
        for b in (kv if kind == "kv" else [st[r.group]]):
            out.append(torch.as_tensor(_DevMem(r.base + b * r.stride, r.nbytes), device=dev))
    return out


def _equal(xs, ys) -> bool:
    import torch

    return len(xs) == len(ys) > 0 and all(torch.equal(x, y) for x, y in zip(xs, ys))


def _worker(rank: int, port: int, out_dir: str):
    import torch

    from ssengine.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig
    from ssengine.model.stage import StageModel, bench_config
    from ssengine.runtime.transport import open_pipe

    from test_kvpool_bytes_gpu import _run

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    pipe = open_pipe(RunConfig(port=port, scheduler=SchedulerConfig(mb_tokens=CHUNK)), rank, dev, HIDDEN_WIDTH)
    m = StageModel(bench_config(CHUNK), rank, dev, num_blocks=1 + 3 * N_KV + 5 * 6, max_model_len=P + A + B,
                   cuda_graphs=False)
    c, R = m.state_geometry
    blocks = iter(range(1, m.num_blocks))
    KX, KR, KQ = ([next(blocks) for _ in range(N_KV)] for _ in range(3))
    S1, S2, S3, SR, SQ = ([next(blocks) for _ in range(R)] for _ in range(5))
    mine = m.tier_ranges()
    res = {"rank": rank, "geometry": [c, R], "pieces": sum(r.kind == "state" for r in mine)}
    g = torch.Generator().manual_seed(11)
    toks = torch.randint(1, 150_000, (P,), generator=g).tolist()
    ta = toks + torch.randint(1, 150_000, (A,), generator=g).tolist()
    tb = toks + torch.randint(1, 150_000, (A,), generator=g).tolist()
    sizes = [CHUNK] * (P // CHUNK) + [P % CHUNK]
    with torch.inference_mode():
        for v in _bytes(mine, dev, [], S2, "state") + _bytes(mine, dev, [], S3, "state"):
            v.random_(0, 256)                      # the working states start as garbage: the copy must cover them
        _run(m, pipe, rank, _plans("x0", toks, 0, sizes, KX, S1, 0))
        s1 = [v.clone() for v in _bytes(mine, dev, [], S1, "state")]
        la = _run(m, pipe, rank, _plans("x1a", ta, P, [A], KX, S2, 100, state_from=S1))
        res["inner_kept"] = _equal(s1, _bytes(mine, dev, [], S1, "state"))
        kv_a = [v.clone() for v in _bytes(mine, dev, KX, S2, "kv")]
        st_a = [v.clone() for v in _bytes(mine, dev, [], S2, "state")]
        lb = _run(m, pipe, rank, _plans("x1b", tb, P, [A], KX, S3, 200, state_from=S1))
        # references: no reuse, no copy (round 1 in place on round 0's state)
        _run(m, pipe, rank, _plans("ra0", ta, 0, sizes, KR, SR, 300))
        ra = _run(m, pipe, rank, _plans("ra1", ta, P, [A], KR, SR, 400))
        _run(m, pipe, rank, _plans("rb0", tb, 0, sizes, KQ, SQ, 500))
        rb = _run(m, pipe, rank, _plans("rb1", tb, P, [A], KQ, SQ, 600))
        res["a_state_equal"] = _equal(st_a, _bytes(mine, dev, [], SR, "state"))
        res["a_kv_equal"] = _equal(kv_a, _bytes(mine, dev, KR, SR, "kv"))
        res["b_state_equal"] = _equal(_bytes(mine, dev, [], S3, "state"), _bytes(mine, dev, [], SQ, "state"))
        res["b_kv_equal"] = _equal(_bytes(mine, dev, KX, S3, "kv"), _bytes(mine, dev, KQ, SQ, "kv"))
        res["b_differs_from_a"] = not _equal(_bytes(mine, dev, [], S3, "state"), st_a)
    if rank == 3:
        res["a_logits_equal"] = bool(torch.equal(la, ra))
        res["b_logits_equal"] = bool(torch.equal(lb, rb))
        res["max_abs_diff"] = [float((la.float() - ra.float()).abs().max()), float((lb.float() - rb.float()).abs().max())]
        res["tokens"] = [int(la.argmax()), int(ra.argmax()), int(lb.argmax()), int(rb.argmax())]
    with open(f"{out_dir}/rank{rank}.json", "w") as f:
        json.dump(res, f)
    pipe.close()


def test_resume_from_a_kept_state_equals_a_fresh_prefill(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    import torch.multiprocessing as mp

    mp.start_processes(_worker, args=(free_port_block(), str(tmp_path)), nprocs=4, start_method="spawn")
    res = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(4)]
    print(json.dumps(res))
    assert len({tuple(r["geometry"]) for r in res}) == 1
    for r in res:
        assert r["pieces"] > 0 and r["inner_kept"] and r["b_differs_from_a"], r
        assert r["a_state_equal"] and r["a_kv_equal"] and r["b_state_equal"] and r["b_kv_equal"], r
    assert res[3]["a_logits_equal"] and res[3]["b_logits_equal"], res[3]
