"""`ppimpl serve <model>`: the run config from a model, like `vllm serve <model>` (README "Serve").

<model> is a Hugging Face repo id or a local checkpoint directory. Only the model this engine is built for is served:
GLM-5.3-Flash NVFP4 as 4 pipeline stages (`check_model`); anything else is refused before a weight is read or
downloaded. The config is the design setting of the M6 evaluation (measured/eval_m6/summary.md, "st10"): the
scheduler of 6_scheduler.md step "good", the exact prefix cache (kept sessions) and, when a DRAM or SSD tier is given,
exclusive DRAM writes and a 10% loading area (TIERS.md); without one, HBM only. A `--config` YAML is applied on top of
that, the command-line options last.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import msgspec
import yaml

from ppimpl.config import (CHECKPOINT_REPO, CHECKPOINT_REVISION, HIDDEN_WIDTH, LAYER_SPLIT, SCHEDULER_STEPS,
                           RunConfig)

# The engine's model length: the stage models' vLLM config (ppimpl/model/stage.py MAX_MODEL_LEN), = the checkpoint's
# max_position_embeddings. A prompt must also fit the KV pool (controller).
MAX_MODEL_LEN = 1_048_576
ARCHITECTURE = "Glm5NextForConditionalGeneration"
# config.json text_config values the stage code, the layer split and the tuned tactics are built for
TEXT_CONFIG = {"num_hidden_layers": sum(LAYER_SPLIT), "hidden_size": HIDDEN_WIDTH // 4, "hc_mult": 4,
               "num_attention_heads": 64, "n_routed_experts": 288, "num_experts_per_tok": 8,
               "moe_intermediate_size": 2048, "vocab_size": 154880}
DESIGN_TIERS = {"dram_writes": "hbm-evictions-exclusive", "loading_area": 0.1}


class UnsupportedModel(SystemExit):
    pass


def check_model(config: dict, name: str) -> None:
    """Refuse every model but GLM-5.3-Flash NVFP4 (config.json, with its quantization config)."""
    archs = config.get("architectures") or []
    text = config.get("text_config") or {}
    quant = config.get("quantization_config") or {}
    why = []
    if ARCHITECTURE not in archs:
        why.append(f"architecture {', '.join(archs) or 'none'} (want {ARCHITECTURE})")
    else:
        why += [f"text_config.{k} = {text.get(k)} (want {v})" for k, v in TEXT_CONFIG.items() if text.get(k) != v]
        if quant.get("quant_algo") != "NVFP4":
            why.append(f"quantization {quant.get('quant_algo') or quant.get('quant_method') or 'none'} (want NVFP4)")
    if why:
        raise UnsupportedModel(
            f"unsupported model {name}: " + "; ".join(why) + f". ppimpl serves only {CHECKPOINT_REPO} "
            f"(GLM-5.3-Flash NVFP4, {ARCHITECTURE}) as a {len(LAYER_SPLIT)}-stage pipeline on {len(LAYER_SPLIT)} GPUs")


def resolve_model(model: str, revision: str | None = None) -> str:
    """The checkpoint directory of `model`: a local directory, or a Hugging Face repo id fetched into the HF cache
    (HF_HOME) as vLLM does. config.json is fetched and checked first, so an unsupported model downloads nothing more.
    The default revision of nvidia/GLM-5.3-Flash-NVFP4 is the one this repo is measured on."""
    path = Path(model).expanduser()
    if path.is_dir():
        cfg = path / "config.json"
        if not cfg.is_file():
            raise UnsupportedModel(f"{model}: no config.json in this directory")
        check_model(json.loads(cfg.read_text()), model)
        return str(path.resolve())
    if model.startswith((".", "/", "~")) or model.count("/") != 1:
        raise UnsupportedModel(f"{model}: neither a local directory nor a Hugging Face repo id")
    from huggingface_hub import hf_hub_download, snapshot_download

    if revision is None and model == CHECKPOINT_REPO:
        revision = CHECKPOINT_REVISION
    check_model(json.loads(Path(hf_hub_download(model, "config.json", revision=revision)).read_text()), model)
    print(f"ppimpl: {model} at revision {revision or 'main'} into the HF cache (~191 GB the first time)", flush=True)
    return snapshot_download(model, revision=revision)


def design_config(checkpoint: str) -> dict:
    """The st10 design setting without tiers (`build_config` adds them)."""
    return {"model": "glm53", "checkpoint": checkpoint, "keep_sessions": True,
            "scheduler": msgspec.to_builtins(SCHEDULER_STEPS["good"])}


def merge(base: dict, over: dict) -> dict:
    """`over` on top of `base`, nested dicts key by key."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


SECCOMP_FIX = """In Docker, run with the image's seccomp profile (Docker's default plus the three io_uring syscalls):
  docker compose -f docker/compose.yaml up            (from a checkout: it sets the profile), or
  docker run --rm <image> cat /opt/ppimpl/docker/seccomp.json > ppimpl-seccomp.json
  docker run --security-opt seccomp=ppimpl-seccomp.json ...
On a host: sysctl kernel.io_uring_disabled must be 0 (or 1 with this user in kernel.io_uring_group)."""


MEMLOCK_FIX = """In Docker: --ulimit memlock=-1 (docker/compose.yaml sets it). On a host: memlock unlimited in
/etc/security/limits.conf (`ulimit -l` shows the limit)."""


def check_io_uring(dram_gb: float | None) -> None:
    """The SSD tier reads and writes through io_uring only: stop before loading anything when it is denied, or when
    the locked-memory limit is below the DRAM tier the ring registers (io_uring counts registered buffers against
    RLIMIT_MEMLOCK, per user: all stages together)."""
    import resource

    from ppimpl.kvpool.uring import probe

    err = probe()
    if err:
        raise SystemExit(f"the SSD tier needs io_uring, and io_uring_setup failed here: "
                         f"{errno.errorcode.get(err, err)} ({os.strerror(err)}).\n{SECCOMP_FIX}")
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    need = len(LAYER_SPLIT) * (dram_gb or 0) * 1e9
    if soft != resource.RLIM_INFINITY and soft < need:
        raise SystemExit(f"the SSD tier registers the DRAM tier ({need / 1e9:.0f} GB on {len(LAYER_SPLIT)} stages) "
                         f"with io_uring, which counts it as locked memory; the limit here is {soft / 2**20:.0f} MiB."
                         f"\n{MEMLOCK_FIX}")


def build_config(a) -> RunConfig:
    """The run config of `ppimpl serve <model>` (argparse namespace of ppimpl/__main__.py)."""
    checkpoint = resolve_model(a.model, a.revision)
    raw = design_config(checkpoint)
    tiers = a.dram_gb is not None or a.ssd_gb is not None
    if tiers:
        raw["tiers"] = {"dram_gb": a.dram_gb or None, "ssd_gb": a.ssd_gb or None, **DESIGN_TIERS}
    if a.config:
        raw = merge(raw, yaml.safe_load(Path(a.config).read_text()) or {})
    if a.ssd_dir:
        if not raw.get("tiers"):
            raise SystemExit("--ssd-dir without an SSD tier: give --ssd-gb")
        raw["tiers"]["ssd_dir"] = a.ssd_dir
    if raw.get("tiers") and raw["tiers"].get("ssd_gb") and not (raw["tiers"].get("ssd_dir")
                                                                or os.environ.get("PPIMPL_SSD_DIR")):
        raise SystemExit("the SSD tier needs --ssd-dir (or PPIMPL_SSD_DIR): a directory with a shard set, made once by "
                         "`python -m ppimpl.kvpool.shards create --dir DIR --gb <ssd-gb>` (TIERS.md §5)")
    if raw.get("tiers") and raw["tiers"].get("ssd_gb"):
        check_io_uring(raw["tiers"].get("dram_gb"))
    if a.max_model_len is not None and not 1 < a.max_model_len <= MAX_MODEL_LEN:
        raise SystemExit(f"--max-model-len {a.max_model_len}: at most {MAX_MODEL_LEN} (the model's)")
    raw.update({k: v for k, v in (("host", a.host), ("port", a.port), ("run_dir", a.run_dir),
                                  ("max_model_len", a.max_model_len)) if v is not None})
    raw["served_model_name"] = a.served_model_name or raw.get("served_model_name") or a.model
    raw.setdefault("host", "0.0.0.0")
    return msgspec.convert(raw, RunConfig)


def add_arguments(s) -> None:
    """The `serve` options (ppimpl/__main__.py)."""
    s.add_argument("model", nargs="?", help="Hugging Face repo id or local checkpoint directory (only "
                   f"{CHECKPOINT_REPO}, the default); with --config and no model, the YAML is the whole run config")
    s.add_argument("--config", help="run config YAML (ppimpl/config.py RunConfig); with a model, applied on top of the "
                   "design setting")
    s.add_argument("--host", help="HTTP address (with a model: default 0.0.0.0)")
    s.add_argument("--port", type=int, help="HTTP port (default 8000); the engine also uses the next 11 ports on "
                   "127.0.0.1")
    s.add_argument("--run-dir", help="logs and per-run files (default ./run)")
    s.add_argument("--revision", help=f"Hugging Face revision (default for {CHECKPOINT_REPO}: {CHECKPOINT_REVISION[:10]})")
    s.add_argument("--served-model-name", help="the model name of the OpenAI API (default: <model> as given)")
    s.add_argument("--max-model-len", type=int, help=f"longest prompt + output (default and maximum {MAX_MODEL_LEN})")
    s.add_argument("--dram-gb", type=float, help="DRAM tier per GPU in GB (the evaluation's full setting: 50)")
    s.add_argument("--ssd-gb", type=float, help="SSD tier per GPU in GB (128); needs a shard set in --ssd-dir")
    s.add_argument("--ssd-dir", help="the SSD tier's shard sets (default $PPIMPL_SSD_DIR)")
