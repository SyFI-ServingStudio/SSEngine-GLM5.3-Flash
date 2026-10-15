"""Run the ServingStudioSim profiler CLI with every worker executing in pp-impl/.venv.

Run from the ServingStudioSim checkout (its project venv is the controller):
  uv run --no-sync python <this> run <kind> --backend <b> --specs F --fresh --json ...

Every profiling env a configuration of 2_kernels.md can name (the vllm_env container,
vllm_upstream_fork_env, flashinfer_kda_env, causal_conv1d_env) is re-registered as a
host env on pp-impl/.venv's interpreter, with the fork checkout and Dao causal-conv1d
on PYTHONPATH, torch/lib first on LD_LIBRARY_PATH and inherited site-packages dropped.
Nothing in the profiler is edited; with --fresh nothing is written to any DB.
"""

import os
import sys
from pathlib import Path

PPI = Path("/raid/kanzhu/ServingStudio/pp-impl")
VENV_PY = PPI / ".venv" / "bin" / "python"
TORCH_LIB = PPI / ".venv" / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
PATHS = [PPI / "third_party" / "vllm", PPI / "third_party" / "causal_conv1d_try" / "pkg"]
HOOK = Path(__file__).resolve().parent / "hook"
# The profiler package is the ServingStudioSim checkout itself (not installed into its venv);
# running this file as a script puts only its own directory on sys.path.
SIM = Path(os.environ.get("PPM_SIM_ROOT", "/raid/kanzhu/ServingStudio/wt-pp-main"))
ENV_NAMES = ("vllm_env", "vllm_upstream_fork_env", "flashinfer_kda_env", "causal_conv1d_env")


def main():
    sys.path.insert(0, str(SIM))
    from profiling.cli import main as cli_main
    from profiling.exec.env import register_profile_env

    extra = ([HOOK] if os.environ.get("PPM_KNAMES_DIR") else []) + PATHS
    for name in ENV_NAMES:
        register_profile_env(name, VENV_PY, additional_python_paths=extra,
                             additional_library_paths=(TORCH_LIB,), isolated_site_packages=True)
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
