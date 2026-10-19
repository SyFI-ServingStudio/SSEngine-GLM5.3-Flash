"""Activation pipe on 2-4 GPUs: CePipe (copy engine into the receiver's ring) delivers exactly the bytes stage 0 sent,
across ring wrap-around, with a slow last stage so the sender has to wait for free slots; the in-flight gate holds
stage 0 to the last stage's exits. Run through Slurm."""

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
    from ssengine.runtime.transport import CePipe

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


GATE_MBS, GATE_SLOW_MS = 12, 20


def _gate_stage(stage: int, n: int, port: int, out_path: str):
    """Stage 0 runs fast, the last stage 20 ms per microbatch: with the gate (2 in flight) stage 0 can finish its
    microbatch k only after the last stage finished k-2."""
    import torch
    from ssengine.runtime.transport import CePipe

    device = torch.device("cuda", stage)
    torch.cuda.set_device(device)
    pipe = CePipe(stage, n, device, f"tcp://127.0.0.1:{port}", MAX_IN_FLIGHT, ROWS, WIDTH)
    t0 = time.monotonic()
    ends = []
    for i in range(GATE_MBS):
        h = sent(i, 64, device) if stage == 0 else pipe.recv(64, WIDTH)
        if stage == 0:
            pipe.gate(i + 1)
        out = h.clone()
        torch.cuda._sleep(GATE_SLOW_MS * 2_000_000 if stage == n - 1 else 1_000)   # ~1 ns per cycle at ~2 GHz
        end = torch.cuda.Event()
        end.record()
        if stage == n - 1:
            pipe.exited(i + 1)
        if stage > 0:
            pipe.release()
        if stage < n - 1:
            pipe.send(out, end)
        ends.append(end)
    marks = []
    for e in ends:
        e.synchronize()
        marks.append((time.monotonic() - t0) * 1e3)
    pipe.close(GATE_MBS)
    torch.save(marks, f"{out_path}.{stage}")


def test_gate_holds_stage0_to_the_exits(tmp_path):
    torch = pytest.importorskip("torch")
    n = min(4, torch.cuda.device_count())
    if n < 2:
        pytest.skip("needs 2 GPUs")
    import torch.multiprocessing as mp
    out = str(tmp_path / "gate")
    mp.start_processes(_gate_stage, args=(n, free_port_block(2), out), nprocs=n, start_method="spawn")
    first, last = torch.load(f"{out}.0"), torch.load(f"{out}.{n - 1}")
    step = (last[-1] - last[0]) / (GATE_MBS - 1)
    assert step > 0.8 * GATE_SLOW_MS
    for k in range(MAX_IN_FLIGHT, GATE_MBS):          # stage 0's k ends after the last stage's k - 2 (host-observed)
        assert first[k] >= last[k - MAX_IN_FLIGHT] - 2.0, (k, first, last)
    assert first[-1] > (GATE_MBS - 1 - MAX_IN_FLIGHT) * 0.8 * GATE_SLOW_MS
