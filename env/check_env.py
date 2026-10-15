"""Import every kernel library pp-impl uses and print its version (CPU only; no kernel launches).

    .venv/bin/python env/check_env.py [--markdown]
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import subprocess
import sys
from pathlib import Path


def version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "-"


def check(name: str, fn) -> tuple[str, str, bool]:
    try:
        return name, str(fn()), True
    except Exception as exc:  # report, keep going
        return name, f"FAILED: {exc!r}"[:200], False


def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short=10", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def rows() -> list[tuple[str, str, bool]]:
    import torch

    out = [
        check("python", lambda: sys.version.split()[0]),
        check("torch", lambda: f"{torch.__version__} (CUDA {torch.version.cuda})"),
        check("cuBLAS (nvidia-cublas)", lambda: version("nvidia-cublas") if version("nvidia-cublas") != "-" else version("nvidia-cublas-cu13")),
    ]

    def vllm_info():
        import vllm

        path = Path(vllm.__file__).resolve().parents[1]
        return f"{vllm.__version__} at {path} (HEAD {git_head(path)})"

    def deep_gemm():
        from vllm.utils.deep_gemm import _import_deep_gemm

        dg = _import_deep_gemm()
        return f"{getattr(dg, '__version__', '?')} at {Path(dg.__file__).parent}"

    def ext(module: str, ns: str, op: str):
        importlib.import_module(module)
        import torch as t

        getattr(getattr(t.ops, ns), op)
        return f"torch.ops.{ns}.{op} registered"

    def causal_conv1d():
        import causal_conv1d
        import causal_conv1d_cuda  # noqa: F401

        return f"{version('causal-conv1d')} at {Path(causal_conv1d.__file__).parent}"

    out += [
        check("vllm (fork)", vllm_info),
        check("flashinfer-python", lambda: importlib.import_module("flashinfer").__version__),
        check("nvidia-cutlass-dsl", lambda: version("nvidia-cutlass-dsl")),
        check("triton", lambda: importlib.import_module("triton").__version__),
        check("tilelang", lambda: importlib.import_module("tilelang").__version__),
        check("DeepGEMM (vendored)", deep_gemm),
        check("DeepSelect", lambda: ext("vllm._deepselect_C", "deep_select", "topk")),
        check("FlashKDA", lambda: ext("vllm._flashkda_C", "_flashkda_C", "fwd")),
        check("vLLM _C", lambda: ext("vllm._custom_ops", "_C", "top_k_per_row_prefill")),
        check("causal-conv1d (Dao; optional, profiler row 4 only)", causal_conv1d),
    ]
    return out


def main() -> int:
    table = rows()
    if "--markdown" in sys.argv:
        print("| Library | Version / status |\n|---|---|")
        for name, value, _ in table:
            print(f"| {name} | {value} |")
    else:
        for name, value, ok in table:
            print(f"{'ok ' if ok else 'BAD'} {name}: {value}")
    required_ok = all(ok for name, _, ok in table if "optional" not in name)
    return 0 if required_ok else 1


if __name__ == "__main__":
    sys.exit(main())
