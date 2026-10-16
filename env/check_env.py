"""Import every kernel library pp-impl uses, check its version against ENV.md, and check that the custom ops and the
FlashInfer KDA backend the manifest names are registered (CPU only; no kernel launches). Exits non-zero on any miss.

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


def check(name: str, fn, want: str | None = None) -> tuple[str, str, bool]:
    """`want`: the ENV.md version, which must appear in the reported value."""
    try:
        value = str(fn())
    except Exception as exc:  # report, keep going
        return name, f"FAILED: {exc!r}"[:200], False
    if want is not None and want not in value:
        return name, f"{value} (want {want})", False
    return name, value, True


def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short=10", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def rows() -> list[tuple[str, str, bool]]:
    import torch

    out = [
        check("python", lambda: sys.version.split()[0], "3.12."),
        check("torch", lambda: f"{torch.__version__} (CUDA {torch.version.cuda})", "2.13.0+cu130"),
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

    def kda_persistent():
        import typing

        from flashinfer.kda import recurrent_kda

        backends = typing.get_args(typing.get_type_hints(recurrent_kda)["backend"])
        if "cute-dsl-persistent" not in backends:
            raise RuntimeError(f"backends {backends}")
        return "recurrent_kda backend available"

    out += [
        check("vllm (fork)", vllm_info, "HEAD 892da0822f"),
        check("flashinfer-python", lambda: importlib.import_module("flashinfer").__version__, "0.7.2.dev20261008"),
        check("nvidia-cutlass-dsl", lambda: version("nvidia-cutlass-dsl"), "4.8.0"),
        check("triton", lambda: importlib.import_module("triton").__version__, "3.7.1"),
        check("tilelang", lambda: importlib.import_module("tilelang").__version__, "0.1.12"),
        check("DeepGEMM (vendored)", deep_gemm, "2.8.0"),
        check("DeepSelect", lambda: ext("vllm._deepselect_C", "deep_select", "topk")),
        check("FlashKDA", lambda: ext("vllm._flashkda_C", "_flashkda_C", "fwd")),
        check("vLLM _C", lambda: ext("vllm._custom_ops", "_C", "top_k_per_row_prefill")),
        check("causal-conv1d (Dao; KDA short conv)", causal_conv1d, "1.7.0"),
        check("FlashInfer KDA cute-dsl-persistent", kda_persistent),
        check("ninja (FlashInfer JIT)", ninja),
    ]
    return out


NINJA = "/usr/bin/ninja"


def ninja() -> str:
    """FlashInfer JIT builds only with /usr/bin/ninja 1.11.1: processes sharing ~/.cache/flashinfer with another ninja
    hash build commands differently and rebuild every object (fused_moe_trtllm_sm100 ~290 s). Warns, never fails."""
    import shutil

    found = shutil.which("ninja")
    if found != NINJA:
        msg = f"WARNING: ninja on PATH is {found}, expected {NINJA} (shared FlashInfer JIT cache would rebuild)"
        print(msg, file=sys.stderr)
        return msg
    return found


def main() -> int:
    table = rows()
    if "--markdown" in sys.argv:
        print("| Library | Version / status |\n|---|---|")
        for name, value, _ in table:
            print(f"| {name} | {value} |")
    else:
        for name, value, ok in table:
            print(f"{'ok ' if ok else 'BAD'} {name}: {value}")
    required_ok = all(ok for _, _, ok in table)
    return 0 if required_ok else 1


if __name__ == "__main__":
    sys.exit(main())
