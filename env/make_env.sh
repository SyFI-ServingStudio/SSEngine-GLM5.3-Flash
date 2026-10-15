#!/bin/bash
# Recreate pp-impl's shared environment (see ENV.md). Run as the normal user from anywhere; never as root.
#   env/make_env.sh <pp-impl root> <built fork tree at 892da0822f>
# The fork tree must be a built checkout of the same HEAD with its own .venv (uv); nothing in it is modified.
set -euo pipefail
ROOT=$(realpath "${1:?pp-impl root}")
REF=$(realpath "${2:?built fork tree}")
HERE=$(dirname "$(realpath "$0")")
COMMIT=892da0822f
[ "$(id -u)" != 0 ] || { echo "do not run as root"; exit 1; }
[ "$(git -C "$REF" rev-parse --short=10 HEAD)" = "$COMMIT" ] || { echo "$REF is not at $COMMIT"; exit 1; }

# 1. Fork checkout: a detached worktree of the same repository, plus the ignored build outputs of the built tree.
if [ ! -e "$ROOT/third_party/vllm/.git" ]; then
  git -C "$REF" worktree add --detach "$ROOT/third_party/vllm" "$COMMIT"
fi
rsync -a --files-from="$HERE/build_outputs.txt" "$REF/" "$ROOT/third_party/vllm/"

# 2. Venv: the reference venv's pinned packages (offline, from the uv cache), minus the editable vllm.
uv venv --python /usr/bin/python3.12 "$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" --offline -r "$HERE/requirements.lock"

# 3. Editable vllm: copy the reference venv's finder and dist-info and point them at our checkout.
SP_REF="$REF/.venv/lib/python3.12/site-packages"
SP="$ROOT/.venv/lib/python3.12/site-packages"
N=vllm-0.1.dev21920+g892da0822.precompiled
cp "$SP_REF/__editable__.$N.pth" "$SP/"
cp "$SP_REF/__editable___vllm_0_1_dev21920_g892da0822_precompiled_finder.py" "$SP/"
cp -r "$SP_REF/$N.dist-info" "$SP/"
sed -i "s#$REF#$ROOT/third_party/vllm#g" \
  "$SP/__editable___vllm_0_1_dev21920_g892da0822_precompiled_finder.py" \
  "$SP/$N.dist-info/direct_url.json" "$SP/$N.dist-info/RECORD"
rm -f "${SP:?}/${N:?}.dist-info/uv_cache.json" "${SP:?}/${N:?}.dist-info/uv_build.json"

# 4. Profiler runners (input builders and references), frozen at the handoff's commit.
SIM=/raid/kanzhu/ServingStudio/wt-pp-main
mkdir -p "$ROOT/third_party/sim_7bb224a6"
git -C "$SIM" archive 7bb224a6 profiling | tar -x -C "$ROOT/third_party/sim_7bb224a6"

"$ROOT/.venv/bin/python" "$HERE/check_env.py"
