"""Activation pipes on 2-4 GPUs: CePipe (copy engine into the receiver's ring) delivers the same bytes as NCCL Pipe,
across ring wrap-around, with a slow last stage so the sender has to wait for free slots. Run through Slurm."""

import time

import pytest

from conftest import free_port_block

pytestmark = pytest.mark.gpu

ROWS, WIDTH, MAX_IN_FLIGHT = 4096, 4096, 2                    # K = 3 slots
SIZES = [4096, 1, 777, 4096, 2048, 3, 4000, 1234, 4096, 17]  # K + 7 microbatches of varying size


def sent(i: int, n: int, device):
    import torch
    g = torch.Generator(device=device).manual_seed(1000 + i)
    return torch.randint(-32768, 32767, (n, WIDTH), dtype=torch.int16, device=device, generator=g).view(torch.bfloat16)


def _stage(stage: int, n: int, transport: str, port: int, out_path: str):
    import torch
    from ppimpl.runtime.transport import CePipe, Pipe

    device = torch.device("cuda", stage)
    torch.cuda.set_device(device)
    init = f"tcp://127.0.0.1:{port}"
    pipe = (CePipe(stage, n, device, init, MAX_IN_FLIGHT, ROWS, WIDTH) if transport == "ce"
            else Pipe(stage, n, device, init))
    got = []
    t0 = time.perf_counter()
    for i, rows in enumerate(SIZES):
        h = sent(i, rows, device) if stage == 0 else pipe.recv(rows, WIDTH)
        out = h.clone()
        if stage == n - 1:
            got.append(out)
            torch.cuda._sleep(20_000_000)          # slow consumer: the ring fills, the sender waits on free[]
        end = torch.cuda.Event()
        end.record()
        if stage > 0:
            pipe.release()
        if stage < n - 1:
            pipe.send(out, end)
    torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) * 1e3
    pipe.close()
    if stage == n - 1:
        torch.save({"got": [g.view(torch.int16).cpu() for g in got], "ms": ms}, out_path)


def _run(transport: str, n: int, tmp_path):
    import torch.multiprocessing as mp
    out = str(tmp_path / f"{transport}.pt")
    mp.start_processes(_stage, args=(n, transport, free_port_block(2), out), nprocs=n, start_method="spawn")
    import torch
    return torch.load(out)


def test_ce_matches_nccl(tmp_path):
    torch = pytest.importorskip("torch")
    n = min(4, torch.cuda.device_count())
    if n < 2:
        pytest.skip("needs 2 GPUs")
    ce, nccl = _run("ce", n, tmp_path), _run("nccl", n, tmp_path)
    for i, rows in enumerate(SIZES):
        want = sent(i, rows, torch.device("cuda", 0)).view(torch.int16).cpu()
        assert torch.equal(ce["got"][i], want), f"ce microbatch {i}"
        assert torch.equal(nccl["got"][i], want), f"nccl microbatch {i}"
    print(f"{n} stages: ce {ce['ms']:.1f} ms, nccl {nccl['ms']:.1f} ms")
