"""Run the ServingStudioSim profiler CLI with every worker executing in pp-impl/.venv.

Run from the ServingStudioSim checkout (its project venv is the controller):
  uv run --no-sync python <this> run <kind> --backend <b> --specs F --fresh --json ...

Every profiling env a configuration of 2_kernels.md can name (the vllm_env container,
vllm_upstream_fork_env, flashinfer_kda_env, causal_conv1d_env) is re-registered as a
host env on pp-impl/.venv's interpreter, with the fork checkout on PYTHONPATH, torch/lib first on LD_LIBRARY_PATH and inherited site-packages dropped.
Nothing in the profiler is edited; with --fresh nothing is written to any DB.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
PPI = _ROOT.parents[1] if _ROOT.parent.name == "lanes" else _ROOT   # the tree with .venv and third_party/
VENV_PY = PPI / ".venv" / "bin" / "python"
TORCH_LIB = PPI / ".venv" / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
PATHS = [PPI / "third_party" / "vllm"]
HOOK = Path(__file__).resolve().parent / "hook"
# The profiler package is the ServingStudioSim checkout itself (not installed into its venv);
# running this file as a script puts only its own directory on sys.path.
SIM = Path(os.environ.get("PPIMPL_SIM") or PPI / "third_party" / "ServingStudioSim")
# FlashInfer JITs through whichever `ninja` is first on PATH. `uv run` from the SIM checkout puts
# its venv's ninja 1.13 first, while every process started as `.venv/bin/python` (engine, vLLM,
# the model lane's benches) gets /usr/bin/ninja 1.11.1. The two hash build commands differently,
# so each switch rebuilds every object in the shared JIT cache (~5 min for fused_moe_trtllm_sm100).
# Pin the workers to /usr/bin/ninja through a directory that holds only that symlink.
NINJA_DIR = Path(__file__).resolve().parent / "ninja-system"
ENV_NAMES = ("vllm_env", "vllm_upstream_fork_env", "flashinfer_kda_env", "causal_conv1d_env")


def main():
    os.environ["PATH"] = f"{NINJA_DIR}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(SIM))
    from profiling.cli import main as cli_main
    from profiling.exec.env import register_profile_env

    extra = ([HOOK] if os.environ.get("PPM_KNAMES_DIR") else []) + PATHS
    for name in ENV_NAMES:
        register_profile_env(name, VENV_PY, additional_python_paths=extra,
                             additional_library_paths=(TORCH_LIB,), isolated_site_packages=True)
    if os.environ.get("PPM_CPU_DRYRUN"):
        _cpu_dryrun()
    return cli_main(sys.argv[1:])


def _cpu_dryrun():
    """Exercise the whole CLI path without a GPU: spec parsing, chunking, worker launch in
    pp-impl/.venv, spec coercion and runner load in the worker. The pool is told GPU 0 is
    idle, and the worker gets CUDA_VISIBLE_DEVICES="" so it touches no device; every spec
    then ends in the runner's device error, which is the expected outcome."""
    import profiling.exec.local as local

    local.find_idle_gpus = lambda *args, **kwargs: [0]
    original = local._host_worker_command

    def host_worker_command(*args, **kwargs):
        cmd, env = original(*args, **kwargs)
        env["CUDA_VISIBLE_DEVICES"] = ""
        return cmd, env

    local._host_worker_command = host_worker_command


if __name__ == "__main__":
    sys.exit(main())
