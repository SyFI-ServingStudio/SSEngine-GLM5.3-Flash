import random
import socket


def free_port_block(n: int = 8) -> int:
    """A base port with n free consecutive ports (HTTP, zmq, torch.distributed), below the ephemeral range so
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
