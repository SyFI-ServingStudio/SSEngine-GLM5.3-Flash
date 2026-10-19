"""Build, without a GPU, the FlashInfer JIT modules and download the trtllm-gen cubins that serving uses on B200
(env/make_env.sh step `jit`; the Docker image bakes them, docker/Dockerfile).

    python env/bake_jit.py [--jobs N]

The modules are FlashInfer's own JIT specs, built by its own ninja path for sm_100a (FLASHINFER_CUDA_ARCH_LIST), so
they land where and as a serving process on a B200 would build them (~/.cache/flashinfer/<version>/100a/cached_ops;
FLASHINFER_WORKSPACE_BASE moves it) and that process's ninja finds nothing to do. Use the ninja and CUDA_HOME the
server uses (/usr/bin/ninja, ENV.md): the build command is part of ninja's up-to-date check.

The cubins are the kernels FlashInfer fetches from NVIDIA's artifactory at run time, one per kernel the first time it
runs (FLASHINFER_CUBIN_DIR, default ~/.cache/flashinfer/cubins). Rather than the ones one run happened to pick, every
kernel of the families GLM-5.3-Flash NVFP4 dispatches to, so other batch shapes do not download either:
  - the MoE batched GEMMs (fused_moe_trtllm_sm100): NVFP4 x NVFP4 FC1 with SwiGLU and NVFP4 output, and
    NVFP4 x NVFP4 FC2 with BF16 output;
  - sparse-MLA decode (trtllm_batch_decode_mla): FP8 KV, BF16 output, head dim 512, token-sparse.
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

# sm_100a only (B200): the workspace directory is named after it (100a) and the nvcc flags derive from it
os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "10.0a")

from flashinfer.artifacts import ArtifactPath, CheckSumHash  # noqa: E402
from flashinfer.jit import env as jit_env  # noqa: E402
from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module  # noqa: E402
from flashinfer.jit.cubin_loader import get_artifact  # noqa: E402
from flashinfer.jit.fp4_quantization import gen_fp4_quantization_sm100_module  # noqa: E402
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module  # noqa: E402

CUBIN_FAMILIES = {
    ArtifactPath.TRTLLM_GEN_BMM: (CheckSumHash.TRTLLM_GEN_BMM, re.compile(
        r"^Bmm_(E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_.*_swiGlu_|Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16(tokFp32)?_)"
        r".*_sm100f\.cubin$")),
    ArtifactPath.TRTLLM_GEN_FMHA: (CheckSumHash.TRTLLM_GEN_FMHA, re.compile(
        r"^fmhaSm100fKernel_QkvE4m3OBfloat16H512.*TokenSparse.*\.cubin$")),
}


def cubins() -> list[tuple[str, str]]:
    """(artifact path, sha256) of every cubin of CUBIN_FAMILIES, from each directory's checksums.txt."""
    out = []
    for path, (checksum_sha, pattern) in CUBIN_FAMILIES.items():
        listing = get_artifact(path + "checksums.txt", checksum_sha)
        assert listing, f"no checksums.txt for {path}"
        for line in listing.decode().splitlines():
            sha, name = line.split()
            if pattern.match(name):
                out.append((path + name, sha))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=16, help="parallel cubin downloads")
    args = ap.parse_args()
    # the specs fetch their own headers (flashinferMetaInfo.h, the bmm export headers) into the cubin directory
    specs = [gen_fp4_quantization_sm100_module(), gen_trtllm_gen_fmha_module(),
             gen_trtllm_gen_fused_moe_sm100_module()]
    print(f"bake_jit: building {[s.name for s in specs]} into {jit_env.FLASHINFER_JIT_DIR}", flush=True)
    # one ninja per module, in its own directory, as a server's build_and_load runs it: ninja's logs (.ninja_log,
    # .ninja_deps) are per directory, and a module built from elsewhere would be rebuilt at the first start
    with ThreadPoolExecutor(len(specs)) as pool:
        list(pool.map(lambda s: s.build(need_lock=True), specs))
    for s in specs:
        assert s.jit_library_path.exists(), s.jit_library_path
    todo = cubins()
    with ThreadPoolExecutor(args.jobs) as pool:
        got = list(pool.map(lambda c: len(get_artifact(*c)), todo))
    missing = [c[0] for c, n in zip(todo, got) if not n]
    if missing:
        sys.exit(f"bake_jit: {len(missing)} cubins failed: {missing[:5]}")
    print(f"bake_jit: {len(todo)} cubins ({sum(got) / 2**20:.0f} MiB) in {jit_env.FLASHINFER_CUBIN_DIR}", flush=True)


if __name__ == "__main__":
    main()
