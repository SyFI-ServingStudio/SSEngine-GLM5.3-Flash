#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH -J pp-reference
# Correctness reference (1_contract.md §4) for the §2.1 one-request prompts:
# vLLM fork TP4 + EP serving GLM-5.3-Flash NVFP4, driven by req-frontend (vllm-tokens)
# on data/bringup/one_<L>.csv through a recording proxy, then a logprobs pass. Usage, from the repo root:
#   sbatch -p main -o measured/logs/%x-%j.out bench/slurm/reference.sbatch
set -u
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
# the same paths the Python side uses (bench/common.py, ppimpl/config.py)
eval "$($PY -c "
from bench.common import ENWIK9, HANDOFF, REQ_FRONTEND
from ppimpl.config import CHECKPOINT
print(f'RF={REQ_FRONTEND} HANDOFF={HANDOFF} SNAP={CHECKPOINT} ENWIK9={ENWIK9}')")"
RUNS=$ROOT/measured/runs
LOGS=$RUNS/reference_logs/$SLURM_JOB_ID
mkdir -p "$LOGS"
PORT=$((20000 + SLURM_JOB_ID % 10000)); PROXY_PORT=$((PORT + 1))
MODEL=glm53-flash-nvfp4
{
  echo "job=$SLURM_JOB_ID cvd=$CUDA_VISIBLE_DEVICES start=$(date -Is) port=$PORT proxy=$PROXY_PORT"
  echo "uptime: $(uptime)"
  echo "fork=$(git -C $MAIN/third_party/vllm rev-parse HEAD) req-frontend=$(git -C $RF rev-parse HEAD) dirty=$(git -C $RF status --porcelain . | wc -l)"
  $PY -c "import torch, flashinfer, vllm; print('torch', torch.__version__, torch.version.cuda, 'flashinfer', flashinfer.__version__, 'vllm', vllm.__version__)"
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu,driver_version --format=csv
  nvidia-smi topo -m
  ss -ltn | grep -E ":($PORT|$PROXY_PORT)\b" && echo "PORT IN USE"
} > "$LOGS/env.txt" 2>&1
grep -q "PORT IN USE" "$LOGS/env.txt" && { echo "port conflict"; exit 1; }

SERVE=(serve "$SNAP" --served-model-name "$MODEL" --tensor-parallel-size 4 --enable-expert-parallel
  --kv-cache-dtype fp8_e4m3 --max-model-len 1048576 --no-enable-prefix-caching --language-model-only
  --tokens-only --enable-prompt-tokens-details --host 127.0.0.1 --port "$PORT")
echo "$PY -m vllm.entrypoints.cli.main ${SERVE[*]}" > "$LOGS/serve_cmd.txt"
srun --overlap --gres=gpu:4 -n1 $PY -m vllm.entrypoints.cli.main "${SERVE[@]}" > "$LOGS/server.log" 2>&1 &
SERVER=$!
t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server died; tail:"; tail -40 "$LOGS/server.log"; exit 1; }
  (( $(date +%s) - t0 > 2400 )) && { echo "server not ready after 40 min"; kill $SERVER; exit 1; }
  sleep 10
done
echo "server ready after $(( $(date +%s) - t0 )) s at $(date -Is)"

$PY bench/reference/record_proxy.py --listen-port $PROXY_PORT --upstream http://127.0.0.1:$PORT \
  --log "$LOGS/proxy_all.jsonl" > "$LOGS/proxy.log" 2>&1 &
PROXY=$!
sleep 3

for L in 1k 2k 4k 8k 16k 32k 64k 128k; do
  OUT=$RUNS/reference_one_$L
  mkdir -p "$OUT"
  cat > "$OUT/run.yaml" <<YAML
input:
  trace: $HANDOFF/data/bringup/one_$L.csv
  format: text-generation-independent
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
  directory: $OUT/client
YAML
  echo "uptime before one_$L: $(uptime)" > "$OUT/uptime.txt"
  n0=$(wc -l < "$LOGS/proxy_all.jsonl" 2>/dev/null || echo 0)
  (cd $RF && uv run --no-sync python -m launcher run "$OUT/run.yaml" > "$OUT/launcher.log" 2>&1)
  echo "one_$L rc=$? at $(date -Is)"
  tail -n +$((n0 + 1)) "$LOGS/proxy_all.jsonl" > "$OUT/proxy.jsonl"
  # The exact prompts, rebuilt from the corpus and checked against what was sent.
  $PY bench/reference/make_prompts.py "$HANDOFF/data/bringup/one_$L.csv" \
    "$ENWIK9" "$SNAP/tokenizer.json" --out "$TMPDIR/prompts_$L" > "$OUT/prompts.jsonl"
  $PY bench/reference/ref_logprobs.py --base-url http://127.0.0.1:$PORT --prompts "$TMPDIR/prompts_$L" \
    --out "$OUT/logprobs.jsonl" --k 5 --model "$MODEL" > "$OUT/logprobs.log" 2>&1
  echo "one_$L logprobs rc=$? at $(date -Is)"
done

kill $PROXY 2>/dev/null
kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
echo "done $(date -Is); uptime: $(uptime)"
sleep 5
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
