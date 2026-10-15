import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))


def free_port_block(n: int = 8) -> int:
    """A base port with n free consecutive ports (HTTP, zmq, torch.distributed)."""
    for _ in range(200):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        if base + n > 65000 or (base + n > 63000 and base < 63200):   # keep clear of the service ports
            continue
        ok = True
        for p in range(base, base + n):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
                    break
        if ok:
            return base
    raise RuntimeError("no free port block")
