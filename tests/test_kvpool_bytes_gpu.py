"""KV pool bytes on 4 GPUs (lanes/m6 TIERS.md §8): real StageModels joined by the serving pipe, a PoolMember per
worker (ppimpl/kvpool/member.py); the ranges come from the real StageKV (layout.from_stage_kv).

Round 0 (P = 40,000 tokens: a partial last block, chunk starts off KDA's 64-token grid) on blocks K0. Every worker
writes its quarter of all four stages' KV and state to its DRAM arena (peer stages' KV over NVLink). Fresh blocks
K1 are filled with garbage and the context is read back into them: every byte of every stage equals K0's. A partial
read (blocks 2.. and the state, as when HBM holds the first blocks) into K2 likewise. Through SSD: the DRAM copy
is demoted to the SSD file (O_DIRECT) and read back through the staging arena into K3. Then round 1 (A = 1,000) on K0
(an HBM hit) and on K1 (after the tier read): the logits are bitwise equal. The workers run with expandable segments,
as the server's do (the KV is allocated IPC-shareable regardless). Run through Slurm:

    sbatch -p main --gres=gpu:4 --time=00:20:00 -o measured/logs/%j.out bench/slurm/gtest.sbatch kvpool_bytes
"""

import json
from types import SimpleNamespace as NS

import pytest

from conftest import free_port_block

pytestmark = pytest.mark.gpu

P, A, CHUNK, B = 40_000, 1_000, 8_100, 8576
N_KV, N_ST = -(-(P + A) // B), 6


def _plans(name, toks, start, sizes, kv, st, mb0):
    plans, s = [], start
    for j, n in enumerate(sizes):
        plans.append(NS(mb_id=mb0 + len(plans), chunks=[NS(req_id=name, start=s, num_tokens=n, kv_blocks=kv,
                                                             state_blocks=st, is_last=j == len(sizes) - 1,
                                                             token_ids=toks[s:s + n])]))
        s += n
    return plans


def _run(m, pipe, rank, plans):
    """The plans through the pipeline (as bench/kvaudit/ours.py); rank 3 returns the last chunk's logits."""
    import torch
    from vllm.forward_context import set_forward_context

    from ppimpl.config import HIDDEN_WIDTH

    logits = None
    for plan in plans:
        n = sum(c.num_tokens for c in plan.chunks)
        hidden = pipe.recv(n, HIDDEN_WIDTH) if rank > 0 else None
        if rank < 3:
            out = m.forward(plan, hidden)
            done = torch.cuda.Event()
            done.record()
            if hidden is not None:
                pipe.release()
            pipe.send(out, done)
        else:
            b = m.prepare(plan)
            with set_forward_context(b.attn, m.vc, num_tokens=b.num_tokens, slot_mapping=b.slots):
                x, r, po, co = m.run_layers(b, hidden)
                if b.last_rows.shape[0]:
                    logits = m.head_logits(b, x, r, po, co)[0]
            pipe.release()
    torch.cuda.synchronize()
    return logits


def _views(ranges, dev, kv, st, b_lo=0):
    """This stage's bytes of a context: each range's rows of blocks kv[b_lo:] / its state block."""
    import torch

    from ppimpl.runtime.transport import _DevMem

    out = []
    for r in ranges:
        for b in (kv[b_lo:] if r.kind == "kv" else [st[r.group]]):
            out.append(torch.as_tensor(_DevMem(r.base + b * r.stride, r.nbytes), device=dev))
    return out


def _worker(rank: int, port: int, out_dir: str):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # as the controller starts its workers

    import torch
    import torch.distributed as dist
    import zmq

    from ppimpl.config import HIDDEN_WIDTH, RunConfig, SchedulerConfig, TierConfig
    from ppimpl.model.stage import StageModel, bench_config
    from ppimpl.kvpool.member import PoolMember
    from ppimpl.runtime.transport import TierMove, open_pipe

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    pipe = open_pipe(RunConfig(port=port, scheduler=SchedulerConfig(mb_tokens=CHUNK)), rank, dev, HIDDEN_WIDTH)
    tc = TierConfig(dram_gb=2.0, ssd_gb=1.0, staging_gb=0.5, ssd_dir=f"{out_dir}/set0")   # the shard set, made below
    eng = PoolMember(NS(tiers=tc, num_stages=4, model="glm53", run_dir=out_dir), rank, dev, zmq.Context())
    m = StageModel(bench_config(CHUNK), rank, dev, num_blocks=1 + 4 * (N_KV + N_ST), max_model_len=P + A + B,
                   cuda_graphs=False, ipc_kv=True)
    mine = m.tier_ranges()
    slot_bytes, state_slots = eng.attach(mine)
    res = {"rank": rank, "slot_bytes": slot_bytes, "state_slots": state_slots,
           "register_s": eng.dram.register_s, "touch_s": eng.dram.touch_s}
    blocks = iter(range(1, m.num_blocks))
    K0, S0, K1, S1, K2, S2, K3, S3 = ([next(blocks) for _ in range(k)] for k in (N_KV, N_ST) * 4)
    g = torch.Generator().manual_seed(7)
    toks = torch.randint(1, 150_000, (P + A,), generator=g).tolist()
    sizes = [CHUNK] * (P // CHUNK) + [P % CHUNK]
    with torch.inference_mode():
        logits0 = _run(m, pipe, rank, _plans("r0", toks, 0, sizes, K0, S0, 0))
        t0 = [int(logits0.argmax()) if rank == 3 else 0]
        dist.broadcast_object_list(t0, src=3)
        toks[P] = t0[0]
        n0 = -(-P // B)
        slots, sslots = list(range(n0)), list(range(n0, n0 + state_slots))

        def move(op_id, op, kv, st, b_lo, tier="dram", host=None):
            if host is None:
                mv = TierMove(op_id, op, tier, None, b_lo, n0, 0, [kv], [st], slots, sslots, [], [])
            else:                           # SSD: file slots = the DRAM ones' ids; host side DRAM or staging
                mv = TierMove(op_id, op, tier, host, b_lo, n0, 0, [kv], [st], slots, sslots, slots, sslots)
            eng.submit(mv)
            if tier == "ssd":
                eng._ssd_op(*eng.ssd_q[eng.direction(mv)].get())
            _, ev, _, nbytes, _, _ = eng.q[eng.direction(mv)].get()
            if ev is not None:
                ev.synchronize()
            torch.cuda.synchronize()
            dist.barrier()
            return nbytes

        res["write_bytes"] = move(0, "write", K0, S0, 0)
        want = [v.clone() for v in _views(mine, dev, K0[:n0], S0)]
        for kv, st in ((K1, S1), (K2, S2), (K3, S3)):
            for v in _views(mine, dev, kv[:n0], st):
                v.random_(0, 256)
        torch.cuda.synchronize()
        dist.barrier()
        res["read_bytes"] = move(1, "read", K1, S1, 0)
        got = _views(mine, dev, K1[:n0], S1)
        res["full_read_equal"] = all(torch.equal(a, b) for a, b in zip(want, got))
        res["partial_read_bytes"] = move(2, "read", K2, S2, 2)
        part = _views(mine, dev, K2[:n0], S2, b_lo=2)
        want2 = _views(mine, dev, K0[:n0], S0, b_lo=2)
        res["partial_read_equal"] = all(torch.equal(a, b) for a, b in zip(want2, part))
        res["partial_left_garbage"] = not torch.equal(_views(mine, dev, K2[:1], S2)[0], want[0])
        res["ssd_direct"] = eng.ssd.direct
        res["demote_bytes"] = move(3, "demote", [], [], 0, "ssd", "dram")
        res["ssd_read_bytes"] = move(4, "read", K3, S3, 0, "ssd", "staging")
        res["ssd_read_equal"] = all(torch.equal(a, b) for a, b in zip(want, _views(mine, dev, K3[:n0], S3)))
        la = _run(m, pipe, rank, _plans("r1_hbm", toks, P, [A], K0, S0, 100))
        lb = _run(m, pipe, rank, _plans("r1_tier", toks, P, [A], K1, S1, 200))
    if rank == 3:
        res["round1_logits_equal"] = bool(torch.equal(la, lb))
        res["round1_tokens"] = [int(la.argmax()), int(lb.argmax())]
        res["round1_max_abs_diff"] = float((la.float() - lb.float()).abs().max())
    with open(f"{out_dir}/rank{rank}.json", "w") as f:
        json.dump(res, f)
    eng.close()
    pipe.close()


def test_tier_round_trip_is_bitwise(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    import torch.multiprocessing as mp

    from ppimpl.kvpool.shards import create
    create(str(tmp_path), 1, 4, 1.0)
    mp.start_processes(_worker, args=(free_port_block(), str(tmp_path)), nprocs=4, start_method="spawn")
    res = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(4)]
    print(json.dumps(res))
    assert len({(r["slot_bytes"], r["state_slots"]) for r in res}) == 1
    for r in res:
        assert r["full_read_equal"] and r["partial_read_equal"] and r["partial_left_garbage"], r
        assert r["write_bytes"] == r["read_bytes"] > r["partial_read_bytes"] > 0
        assert r["ssd_read_equal"] and r["ssd_direct"] and r["demote_bytes"] == r["ssd_read_bytes"] > 0, r
    assert res[3]["round1_logits_equal"], res[3]
