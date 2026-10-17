import random
import socket

import pytest


def free_port_block(n: int = 12) -> int:
    """A base port with n free consecutive ports (HTTP, zmq, torch.distributed, tier ops), below the ephemeral range so
    client connections in TIME_WAIT never hold them, and clear of the 63xxx service ports."""
    for _ in range(200):
        base = random.randrange(18000, 30000)
        ok = True
        for p in range(base, base + n):
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
                    break
        if ok:
            return base
    raise RuntimeError("no free port block")


def pytest_configure(config):
    from bench.common import HANDOFF

    if not HANDOFF.is_dir():   # most CPU tests read the handoff's traces and targets
        raise pytest.UsageError(f"handoff not found at {HANDOFF}: set PPIMPL_SIM or run env/make_env.sh (README)")
