"""Activation pipe on 2-4 GPUs: CePipe (copy engine into the receiver's ring) delivers exactly the bytes stage 0 sent,
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


def _stage(stage: int, n: int, port: int, out_path: str):
    import torch
    from ppimpl.runtime.transport import CePipe

    device = torch.device("cuda", stage)
    torch.cuda.set_device(device)
    pipe = CePipe(stage, n, device, f"tcp://127.0.0.1:{port}", MAX_IN_FLIGHT, ROWS, WIDTH)
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


def test_ce_delivers_the_sent_bytes(tmp_path):
    torch = pytest.importorskip("torch")
    n = min(4, torch.cuda.device_count())
    if n < 2:
        pytest.skip("needs 2 GPUs")
    import torch.multiprocessing as mp
    out = str(tmp_path / "ce.pt")
    mp.start_processes(_stage, args=(n, free_port_block(2), out), nprocs=n, start_method="spawn")
    ce = torch.load(out)
    for i, rows in enumerate(SIZES):
        want = sent(i, rows, torch.device("cuda", 0)).view(torch.int16).cpu()
        assert torch.equal(ce["got"][i], want), f"microbatch {i}"
    print(f"{n} stages: ce {ce['ms']:.1f} ms")
