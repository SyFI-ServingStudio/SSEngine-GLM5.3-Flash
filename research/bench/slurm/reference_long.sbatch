#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J pp-reference-long
# Correctness reference for the prefix bring-up traces: vLLM fork TP4 + EP (the serve command of reference.sbatch,
# job 7029), prefix caching off, so every round is a full prefill. One server; the session traces TRACES
# (space-separated, e.g. the research/bench/targets_ext/*_w1s_compressed_p*.csv parts) are replayed one after another by
# req-frontend through the recording proxy, which asks for the first token's top-5 logprobs on every request and
# saves every prompt (and, prefix caching being off, reports a hit for req-frontend's preflight probe only). Requests whose response carried no logprobs get a second, one-at-a-time pass
# (ref_logprobs.py). Writes OUT (default research/measured/runs_lc/reference_long): per trace <name>/proxy.jsonl and
# client/, and reference.jsonl over all of them (rows in benchmarks/data/reference.jsonl's format, matched by
# prompt_sha256: `python -m benchmarks.check_tokens <our run> --reference OUT/reference.jsonl`). Without TRACES: the
# handoff sessions SESSIONS (default 30,35) compressed GAP_S apart (benchmarks/run.py compressed_bringup). Usage:
#   TRACES="research/bench/targets_ext/prefix_sessions_ext_w1s_compressed_p4.csv" OUT=research/measured/runs_lc/ref_p4 \
#     sbatch -p priority --time=00:30:00 -o research/measured/runs_lc/slurm/%j.out research/bench/slurm/reference_long.sbatch
set -u
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}/benchmarks/jobs/env.sh"
eval "$($PY -c "
from benchmarks.common import ENWIK9, HANDOFF, REQ_FRONTEND
from ppimpl.config import CHECKPOINT
print(f'RF={REQ_FRONTEND} HANDOFF={HANDOFF} SNAP={CHECKPOINT} ENWIK9={ENWIK9}')")"
OUT=$ROOT/${OUT:-measured/runs_lc/reference_long}
SESSIONS=${SESSIONS:-30,35}
GAP_S=${GAP_S:-45}
mkdir -p "$OUT"
PORT=$((20000 + SLURM_JOB_ID % 120 * 100 + 50)); PROXY_PORT=$((PORT + 1))
MODEL=glm53-flash-nvfp4
# KDA prefill: Triton for token references (the user's rule). The fork's pinned FlashKDA (b59532f, before upstream
# 17a037d) rounds the KDA state to bf16 every 16 tokens: wrong above ~24k of context. Triton needs
# max_num_batched_tokens < 262144 (16384 is vLLM's B200 default, stated here). KDA_PREFILL=flashkda for a timing run.
KDA_PREFILL=${KDA_PREFILL:-triton}
{
  echo "job=$SLURM_JOB_ID cvd=$CUDA_VISIBLE_DEVICES start=$(date -Is) port=$PORT proxy=$PROXY_PORT traces=${TRACES:-sessions $SESSIONS} kda_prefill=$KDA_PREFILL"
  echo "uptime: $(uptime)"
  echo "fork=$(git -C $MAIN/third_party/vllm rev-parse HEAD) req-frontend=$(git -C $RF rev-parse HEAD)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  ss -ltn | grep -E ":($PORT|$PROXY_PORT)\b" && echo "PORT IN USE"
} > "$OUT/env.txt" 2>&1
grep -q "PORT IN USE" "$OUT/env.txt" && { echo "port conflict"; exit 1; }
[ -n "${TRACES:-}" ] || $PY -c "
from pathlib import Path
from benchmarks.run import compressed_bringup
compressed_bringup(Path('$HANDOFF/data/bringup/prefix_sessions.csv'), Path('$OUT/sessions.csv'),
                   [int(x) for x in '$SESSIONS'.split(',')], $GAP_S)" || exit 1

SERVE=(serve "$SNAP" --served-model-name "$MODEL" --tensor-parallel-size 4 --enable-expert-parallel
  --kv-cache-dtype fp8_e4m3 --max-model-len 1048576 --no-enable-prefix-caching --language-model-only
  --tokens-only --enable-prompt-tokens-details --kda-prefill-backend "$KDA_PREFILL" --max-num-batched-tokens 16384
  --host 127.0.0.1 --port "$PORT")
echo "$PY -m vllm.entrypoints.cli.main ${SERVE[*]}" > "$OUT/serve_cmd.txt"
srun --overlap --gres=gpu:4 -n1 $PY -m vllm.entrypoints.cli.main "${SERVE[@]}" > "$OUT/server.log" 2>&1 &
SERVER=$!
t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server died; tail:"; tail -40 "$OUT/server.log"; exit 1; }
  (( $(date +%s) - t0 > 1200 )) && { echo "server not ready after 20 min"; kill $SERVER; exit 1; }
  sleep 10
done
echo "server ready after $(( $(date +%s) - t0 )) s at $(date -Is)"

if [ -z "${TRACES:-}" ]; then TRACES=$OUT/sessions.csv; fi
for T in $TRACES; do
  T=$(realpath "$T"); NAME=$(basename "$T" .csv); D=$OUT/$NAME
  mkdir -p "$D" "$TMPDIR/prompts/$NAME"
  $PY research/bench/reference/record_proxy.py --listen-port $PROXY_PORT --upstream http://127.0.0.1:$PORT --logprobs 5 --preflight-hit \
    --log "$D/proxy.jsonl" --save-prompts "$TMPDIR/prompts/$NAME" > "$D/proxy.log" 2>&1 &
  PROXY=$!
  sleep 3
  cat > "$D/run.yaml" <<YAML
input:
  trace: $T
  format: text-generation-session-execution-v2
corpus:
  text_file: $ENWIK9
  tokenizer: $SNAP/tokenizer.json
server:
  backend: vllm-tokens
  base_url: http://127.0.0.1:$PROXY_PORT
  model: $MODEL
  temperature: 0
replay:
  arrival_mode: trace-timed
  context:
    max_model_len: 1048576
output:
  directory: $D/client
YAML
  echo "$NAME: uptime before client: $(uptime)"
  (cd $RF && uv run --no-sync python -m launcher run "$D/run.yaml" > "$D/launcher.log" 2>&1)
  echo "$NAME: client rc=$? at $(date -Is)"
  grep -q '"status":"FAILED"' "$D"/client/requests.jsonl 2>/dev/null && echo "$NAME: CLIENT FAILED REQUESTS"
  kill $PROXY 2>/dev/null; wait $PROXY 2>/dev/null
  rm -f "$TMPDIR"/prompts/$NAME/*preflight*.npy
  # the second pass, only for prompts whose response had no logprobs
  mkdir -p "$TMPDIR/missing/$NAME"
  $PY - "$D/proxy.jsonl" "$TMPDIR/prompts/$NAME" "$TMPDIR/missing/$NAME" <<'PY'
import json, os, sys
for r in map(json.loads, open(sys.argv[1])):
    src = os.path.join(sys.argv[2], f"{r['request_id']}.npy")
    if "top_logprobs" not in r and os.path.exists(src):
        os.link(src, os.path.join(sys.argv[3], f"{r['request_id']}.npy"))
PY
  if ls "$TMPDIR/missing/$NAME"/*.npy > /dev/null 2>&1; then
    echo "$NAME: $(ls "$TMPDIR/missing/$NAME" | wc -l) prompts without logprobs: second pass"
    $PY research/bench/reference/ref_logprobs.py --base-url http://127.0.0.1:$PORT --prompts "$TMPDIR/missing/$NAME" \
      --out "$D/logprobs.jsonl" --k 5 --model "$MODEL" > "$D/logprobs.log" 2>&1
    echo "$NAME: logprobs rc=$? at $(date -Is)"
  fi
  rm -rf "$TMPDIR/prompts/$NAME" "$TMPDIR/missing/$NAME"
done
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for d in sorted(p.parent for p in out.glob("*/proxy.jsonl")):
    second = {r["prompt_sha256"]: r for r in map(json.loads, open(d / "logprobs.jsonl"))} \
        if (d / "logprobs.jsonl").exists() else {}
    for p in map(json.loads, open(d / "proxy.jsonl")):
        if "preflight" in str(p["request_id"]):
            continue
        top = p.get("top_logprobs")
        if top is None and p["prompt_sha256"] in second:
            top = [[t["token_id"], t["logprob"]] for t in second[p["prompt_sha256"]].get("top_logprobs", [])]
        top = top or []
        tied = [t for t, lp in top if lp == top[0][1]]
        out_tok = (p.get("output_token_ids") or [None])[0]
        rows.append({"id": f"{d.name}/{p['request_id']}", "prompt_len": p["prompt_len"],
                     "prompt_sha256": p["prompt_sha256"], "output_token_id": out_tok,
                     "logprob_pass_token_id": top[0][0] if top else out_tok, "top5": top,
                     "margin": top[0][1] - top[1][1] if len(top) > 1 else None,
                     "tied_top_tokens": tied if len(tied) > 1 else None})
(out / "reference.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
bad = [x["id"] for x in rows if not x["top5"] or x["output_token_id"] != x["logprob_pass_token_id"]]
print(f"{len(rows)} reference rows; without logprobs or output != top-1: {len(bad)} {bad[:10]}")
PY
echo "done $(date -Is); uptime: $(uptime)"
