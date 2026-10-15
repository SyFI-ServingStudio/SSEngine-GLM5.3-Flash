"""python -m ppimpl serve --config <yaml> [--port P] [--run-dir D]   controller + HTTP server (spawns the workers)
python -m ppimpl worker --config <yaml> --stage S                    one stage worker (spawned by serve)
python -m ppimpl intake --config <yaml> --index I                    one HTTP intake process (spawned by serve)"""

import argparse
import os

# Before numpy/torch load: one thread per CPU pool in every process here (controller, intakes, workers). The host side
# only parses requests and launches kernels; OpenBLAS alone started 64 idle threads in the controller.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
    os.environ.setdefault(_v, "1")

from ppimpl.config import load_config


def main():
    ap = argparse.ArgumentParser(prog="python -m ppimpl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--config", required=True)
    s.add_argument("--port", type=int)
    s.add_argument("--run-dir")
    w = sub.add_parser("worker")
    w.add_argument("--config", required=True)
    w.add_argument("--stage", type=int, required=True)
    i = sub.add_parser("intake")
    i.add_argument("--config", required=True)
    i.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "serve":
        from ppimpl.server import serve
        serve(load_config(a.config, port=a.port, run_dir=a.run_dir))
    elif a.cmd == "intake":
        from ppimpl.intake import run_intake
        run_intake(load_config(a.config), a.index)
    else:
        from ppimpl.runtime.worker import run_worker
        run_worker(load_config(a.config), a.stage)


if __name__ == "__main__":
    main()
