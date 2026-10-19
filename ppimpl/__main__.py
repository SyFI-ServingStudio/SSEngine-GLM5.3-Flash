"""ppimpl serve [<model>] [options]                                     the server, vLLM style (ppimpl/serve/launch.py);
                                                                     <model> defaults to nvidia/GLM-5.3-Flash-NVFP4
python -m ppimpl serve --config <yaml> [--port P] [--run-dir D]   controller + HTTP server (spawns the workers)
python -m ppimpl worker --config <yaml> --stage S                    one stage worker (spawned by serve)
python -m ppimpl intake --config <yaml> --index I                    one HTTP intake process (spawned by serve)"""

import argparse
import os

# Before numpy/torch load: one thread per CPU pool in every process here (controller, intakes, workers). The host side
# only parses requests and launches kernels; OpenBLAS alone started 64 idle threads in the controller.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
    os.environ.setdefault(_v, "1")

from ppimpl.config import CHECKPOINT_REPO, load_config


def main():
    ap = argparse.ArgumentParser(prog="ppimpl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="the OpenAI-compatible server (GLM-5.3-Flash NVFP4 on 4 GPUs)")
    from ppimpl.serve.launch import add_arguments
    add_arguments(s)
    w = sub.add_parser("worker")
    w.add_argument("--config", required=True)
    w.add_argument("--stage", type=int, required=True)
    i = sub.add_parser("intake")
    i.add_argument("--config", required=True)
    i.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "serve":
        if a.config and not a.model:
            if any(v is not None for v in (a.revision, a.dram_gb, a.ssd_gb, a.ssd_dir, a.ssd_backend)):
                s.error("--revision and the tier options need a model; with --config alone, set them in the YAML")
            cfg = load_config(a.config, port=a.port, run_dir=a.run_dir, host=a.host,
                              served_model_name=a.served_model_name, max_model_len=a.max_model_len)
        else:
            from ppimpl.serve.launch import build_config
            a.model = a.model or CHECKPOINT_REPO   # the one model served
            cfg = build_config(a)
        from ppimpl.serve.server import serve
        serve(cfg)
    elif a.cmd == "intake":
        from ppimpl.serve.intake import run_intake
        run_intake(load_config(a.config), a.index)
    else:
        from ppimpl.runtime.worker import run_worker
        run_worker(load_config(a.config), a.stage)


if __name__ == "__main__":
    main()
