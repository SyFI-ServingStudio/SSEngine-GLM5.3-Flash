#!/bin/bash
# M4 (5_prefix_cache.md) runs on one 4-GPU server, into OUT (default measured/runs_m4). Usage, from the repo root:
#   mkdir -p measured/runs_m4/slurm
#   sbatch -p priority -J ppimpl --time=00:12:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch check
#   sbatch -p priority -J ppimpl --time=00:30:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch bringup
#   sbatch -p priority -J ppimpl --time=00:20:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch preflight
#   sbatch -p priority -J ppimpl --time=00:28:00 -o measured/runs_m4/slurm/%j.out bench/slurm/m4.sbatch prefix_n400
# preflight (before any long run), stopping at the first failure: kvcheck at 32,768 tokens (8,192-token chunks must be
# bitwise; 8,000-token chunks leave a 768-token chunk of another mHC split-K count: the batch-numerics baseline), then
# at 40,000 tokens with chunk starts inside 4-token indexer pools and off KDA's 32-token grid (8,001 and 8,190: within
# 2x the baseline per stage, KL within 4x, top-1 equal), prefix_check with P + 1/2/3
# (prefix_check_unaligned), then a smoke of prefix_n400's first ~2 min (client limit 3m incl. ~45 s tokenization:
# prefix_smoke_n400) that must end with failed=0. bench/run.py stops any run at the first fault (server or worker
# traceback, server exit, client failed > 0). margin: bench/margin.py on CASES (top-2 margins, repeats; MARGIN_ARGS=--trace-topk), after the GPU tests GTEST (-k) if set. SKIP_KVCHECK=1 and CHECK_MODES=off resume a preflight whose kvcheck and
# cache-on half passed (the cache-on rounds are read from prefix_check_unaligned_on/); SKIP_CHECK=1 skips the prefix
# check too. Token differences between the cache-on and cache-off servers go to the margin rule (bench/margin.py
# --judge: each side's token in the top 2, margin < 3 nats).
# SAMPLE=1: nvidia-smi samples (SM clock, power, temperature, throttle reasons) every 200 ms into
# OUT/slurm/gpu_samples_<job>.csv. preflight also runs GTEST (-k) first, as margin does.
# resume_nsys: bench/margin.py --nvtx on CASES (absolute paths, space-separated; default cases_resume_cost.json: P
# 100,000 / 100,003 / 100,004 resumed with A 256 / 1,024) at each commit of REVS (default "worktree") under Nsight
# Systems; bench/mb_kernels.py per stage and kernel (compare runs: python -m bench.mb_kernels --compare); GTEST as in margin.
# check: bench/prefix_check.py (round-1 tokens, cache on vs off, ~6 min). bringup: prefix_bringup (20 min) under
# Nsight Systems, traced from microbatch NSYS_MBS (default 564:631: the last round-0 chunk and round 1 of session
# 34, P 512k A 1k, and all of session 35, P 512k A 8k; microbatch 0 is req-frontend's preflight). prefix_n<N>: the
# 20-minute session run at N. BRINGUP_ARGS adds bench/run.py options to bringup, e.g. "--gap-s 17 --sessions 30,31
# --run-name prefix_bringup_rest" (sessions 17 s apart: the cluster stops GPU jobs idle for 10 min). Then `python -m bench.prefix summarize measured/runs_m4 --out measured`.
#SBATCH -J ppimpl
#SBATCH --gres=gpu:4
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
R=$ROOT/${OUT:-measured/runs_m4}
mkdir -p "$R/slurm"
G=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
P=$((20000 + ${SLURM_JOB_ID:-0} % 120 * 100))
WHAT=${1:?check, preflight, margin, resume_nsys, bringup or prefix_n<N>}
echo "== $(date -Is) gpus $G port $P: $WHAT"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader
if [ -n "${SAMPLE:-}" ]; then   # GPU clocks, power, temperature and throttle reasons every 200 ms, for the whole job
  nvidia-smi -i "$G" --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active \
    --format=csv,nounits -lms 200 > "$R/slurm/gpu_samples_${SLURM_JOB_ID:-local}.csv" 2>&1 &
  SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null; rm -rf "$TMPDIR"' EXIT   # env.sh's trap too
fi
case $WHAT in
  check) $PY -m bench.prefix_check --out "$R" --port "$P"; rc=$? ;;
  preflight)
    rc=0; [ -n "${GTEST:-}" ] && { $PY -m pytest tests -m gpu -k "$GTEST" -v; rc=$?; }   # GPU tests first
    if (( rc == 0 )) && [ -z "${SKIP_KVCHECK:-}" ]; then
      $PY -m bench.kvcheck --prompt-len 32768 --chunk 8000 --chunk-b 8192 --exact chunked_b,whole_filler; rc=$?
      (( rc == 0 )) && { $PY -m bench.kvcheck --prompt-len 40000 --chunk 8001 --chunk-b 8190 --noise-len 32768; rc=$?; }
    fi
    (( rc == 0 )) && [ -z "${SKIP_CHECK:-}" ] && { $PY -m bench.prefix_check --out "$R" --port "$P" --p-shift 1,2,3 --name prefix_check_unaligned \
                         --modes "${CHECK_MODES:-on,off}"; rc=$?
                       if (( rc == 2 )); then   # tokens that differ: the margin rule decides
                         $PY -m bench.margin "$R/prefix_check_unaligned_margins/cases.json" --judge; rc=$?
                       fi; }
    (( rc == 0 )) && { $PY -m bench.run prefix_n400 --inline --gpus "$G" --port "$P" --out-root "$R" --client-limit 3m \
                         --run-name prefix_smoke_n400; rc=$?; }
    ;;
  margin) rc=0; [ -n "${GTEST:-}" ] && { $PY -m pytest tests -m gpu -k "$GTEST" -v; rc=$?; }   # GPU tests first
    (( rc == 0 )) && $PY -m bench.margin "${CASES:-measured/margins/cases_unaligned.json}" --repeats "${REPEATS:-3}" ${MARGIN_ARGS:-}; rc=$? ;;
  resume_nsys)   # per-kernel cost of the last chunks of CASES, at each of REVS (a commit: its ppimpl/ and bench/ from
                 # git archive; "worktree": this tree), under Nsight Systems; bench/mb_kernels.py per stage and kernel
    rc=0; [ -n "${GTEST:-}" ] && { $PY -m pytest tests -m gpu -k "$GTEST" -v; rc=$?; }   # GPU tests first
    (( rc == 0 )) && for rev in ${REVS:-worktree}; do
      D=$R/resume_nsys_${SLURM_JOB_ID:-local}/$rev; mkdir -p "$D"; src=$ROOT
      if [ "$rev" != worktree ]; then
        src=$TMPDIR/rev_$rev; mkdir -p "$src"; git -C "$ROOT" archive "$rev" ppimpl bench | tar -x -C "$src" || { rc=1; break; }
      fi
      echo "== resume_nsys $rev ($(git -C "$ROOT" log --oneline -1 "${rev/worktree/HEAD}")) from $src"
      for cases in ${CASES:-$ROOT/measured/margins/cases_resume_cost.json}; do
        n=$(basename "$cases" .json)
        (cd "$src" && /usr/local/cuda/bin/nsys profile -o "$D/$n" --force-overwrite true -t cuda,nvtx \
           --cuda-graph-trace=node --sample=none --cpuctxsw=none $PY -m bench.margin "$cases" --repeats "${REPEATS:-4}" \
           --nvtx --out "$D/$n.json") || { rc=1; break 2; }
        /usr/local/cuda/bin/nsys export --type sqlite --force-overwrite true -o "$D/$n.sqlite" "$D/$n.nsys-rep" \
          > "$D/$n.export.log" 2>&1 || { rc=1; break 2; }
        $PY -m bench.mb_kernels "$D/$n.sqlite" "$D/$n.mbs.json" --base p100000 --out "$D/$n.kernels.json" > "$D/$n.kernels.txt" \
          || { rc=1; break 2; }
        rm -f "$D/$n.nsys-rep"
      done
    done
    ;;
  bringup) $PY -m bench.run prefix_bringup --inline --nsys --nsys-mbs "${NSYS_MBS:-564:631}" --gpus "$G" --port "$P" \
             --out-root "$R" ${BRINGUP_ARGS:-}; rc=$? ;;
  prefix_n*) $PY -m bench.run "$WHAT" --inline --gpus "$G" --port "$P" --out-root "$R"; rc=$? ;;
  *) echo "unknown: $WHAT"; exit 2 ;;
esac
uptime
echo "== done $(date -Is) rc=$rc"
exit $rc
