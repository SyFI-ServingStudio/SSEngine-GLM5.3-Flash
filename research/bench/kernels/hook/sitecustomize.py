"""Worker-side recorder for the pp-impl re-measure (loaded only from research/bench/kernels/hook on PYTHONPATH).

When PPM_KNAMES_DIR is set, the profiler worker appends to <dir>/<pid>.jsonl:
  {"event": "spec", "seq": i, "kwargs": {...}}            before each spec (batched runners)
  {"event": "cupti", "names": [...], "mean_ms": x}         each CUPTI capture (matched kernel names)
  {"event": "provenance", "kwargs": {...}, "note": "..."}  the row-provenance note (e.g. KDA tactic)
It changes no timing path: wrappers call through and only write a line around the call.
"""

import json
import os

_DIR = os.environ.get("PPM_KNAMES_DIR")


def _install():
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, f"{os.getpid()}.jsonl")

    def write(record):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    import profiling.runners.batched as batched_mod

    original_batched = batched_mod.batched

    def batched(single_fn):
        inner = original_batched(single_fn)

        def run(kwargs_list):
            results = []
            for i, kwargs in enumerate(kwargs_list):
                write({"event": "spec", "seq": i, "fn": single_fn.__name__, "kwargs": kwargs})
                results.extend(inner([kwargs]))
            return results

        return run

    batched_mod.batched = batched

    import profiling.profilers.cupti_kernel_profiler as cupti

    def wrap(name):
        fn = getattr(cupti, name, None)
        if fn is None:
            return

        def wrapper(*args, **kwargs):
            summary = fn(*args, **kwargs)
            write({"event": "cupti", "fn": name,
                   "names": sorted(getattr(summary, "matched_kernel_names", []) or []),
                   "mean_ms": getattr(summary, "mean_ms", None)})
            return summary

        setattr(cupti, name, wrapper)

    for name in ("profile_kernel", "profile_kernel_until_converged", "profile_kernel_for_duration"):
        wrap(name)

    import profiling.db.registry as registry

    original_prov = registry.KernelProfilerSpec.load_row_provenance

    def load_row_provenance(self):
        fn = original_prov(self)
        if fn is None:
            return None

        def logged(**kwargs):
            note = fn(**kwargs)
            write({"event": "provenance", "kwargs": kwargs, "note": note})
            return note

        return logged

    registry.KernelProfilerSpec.load_row_provenance = load_row_provenance


if _DIR:
    try:
        _install()
    except Exception as exc:  # never break the worker
        import sys

        print(f"ppm sitecustomize: not installed: {exc!r}", file=sys.stderr)
