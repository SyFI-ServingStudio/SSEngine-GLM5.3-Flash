"""CPU check of the re-measure worker environment (no GPU): for each (kind, backend) of a queue,
start the worker interpreter exactly as the profiler would (its own host-worker command
builder, with ppimpl_profile.py's env overrides), and in it import the runner, load the list
runner and coerce the first spec. CUDA is hidden, so only import/env errors show.

usage (from the ServingStudioSim checkout):
  uv run --no-sync python <this> <queue_dir>/todo
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppimpl_profile  # noqa: E402

SNIPPET = r"""
import json, sys
from profiling.db.registry import find_kernel_profiler_spec
from profiling.db.batch import coerce_args, args_to_spec
kind, backend, spec = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
ps = find_kernel_profiler_spec(kind, backend)
runner = ps.load_list_runner()
mod = ps.runner_ref.load(ps.runner_module)
kw = args_to_spec(coerce_args(ps.args_schema, spec))
import importlib, torch
print("OK", kind, backend, getattr(mod, "__module__", mod), "torch", torch.__version__)
"""


def main(todo):
    sys.path.insert(0, str(ppimpl_profile.SIM))
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.exec.env import register_profile_env, resolve_profile_env
    from profiling.exec.local import _host_worker_command

    for name in ppimpl_profile.ENV_NAMES:
        register_profile_env(name, ppimpl_profile.VENV_PY,
                             additional_python_paths=[ppimpl_profile.HOOK] + ppimpl_profile.PATHS,
                             additional_library_paths=(ppimpl_profile.TORCH_LIB,),
                             isolated_site_packages=True)
    seen = set()
    for f in sorted(Path(todo).glob("*.jsonl")):
        if f.name.endswith(".rows.jsonl"):
            continue
        _, kind, backend, _ = f.stem.split("__")
        if (kind, backend) in seen:
            continue
        seen.add((kind, backend))
        spec = json.loads(f.read_text().splitlines()[0])
        ps = find_kernel_profiler_spec(kind, backend)
        env_obj = resolve_profile_env(ps.subprocess_env)
        with tempfile.TemporaryDirectory() as tmp:
            cmd, env = _host_worker_command(env_obj, [0], Path(tmp) / "in.json", Path(tmp) / "out.json",
                                            worker_env=dict(ps.worker_env))
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PPM_KNAMES_DIR"] = tempfile.mkdtemp()
        r = subprocess.run([cmd[0], "-c", SNIPPET, kind, backend, json.dumps(spec)], env=env,
                           capture_output=True, text=True, timeout=600)
        tail = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else r.stderr.strip()[-1500:]
        print(f"[{ps.subprocess_env} -> {cmd[0]}] rc={r.returncode} {tail}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
