"""python -m ppimpl serve --config <yaml> [--port P] [--run-dir D]   controller + HTTP server (spawns the workers)
python -m ppimpl worker --config <yaml> --stage S                    one stage worker (spawned by serve)"""

import argparse

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
    a = ap.parse_args()
    if a.cmd == "serve":
        from ppimpl.server import serve
        serve(load_config(a.config, port=a.port, run_dir=a.run_dir))
    else:
        from ppimpl.runtime.worker import run_worker
        run_worker(load_config(a.config), a.stage)


if __name__ == "__main__":
    main()
