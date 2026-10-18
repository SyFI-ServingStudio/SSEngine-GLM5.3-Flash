#!/bin/bash
# m6.sbatch plus py-spy on the stage-0 worker over workload minutes PYSPY_FROM..+PYSPY_MIN (default 8..18), from
# outside the process: one all-threads sampler with idle frames and one GIL-holder sampler, both --nonblocking (py-spy
# has no native stacks with --nonblocking, and a blocking sampler would stall the worker). Raw collapsed stacks in the
# run dir: pyspy_s0_all.txt, pyspy_s0_gil.txt. Same usage as m6.sbatch.
#SBATCH --gres=gpu:4
set -uo pipefail
{
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/bench/slurm/env.sh"
R=$ROOT/${OUT:-measured/runs_m6}
RUN=$1; for ((i = 1; i <= $#; i++)); do [ "${!i}" = --run-name ] && { j=$((i + 1)); RUN=${!j}; }; done
D=$R/$RUN
SPY=${PYSPY:-$HOME/.local/bin/py-spy}
FROM=${PYSPY_FROM:-8}; MIN=${PYSPY_MIN:-10}
(
  until [ -f "$D/client.log" ]; do sleep 2; done
  sleep $((FROM * 60))
  PID=$(pgrep -f "ppimpl worker --config $D/.* --stage 0\$" | head -1)
  [ -z "$PID" ] && { echo "== py-spy: no stage-0 worker for $D"; exit 0; }
  echo "== $(date -Is) py-spy on stage-0 worker pid $PID for $MIN min"
  "$SPY" record --nonblocking --threads --idle --rate 100 --duration $((MIN * 60)) --format raw \
    -o "$D/pyspy_s0_all.txt" --pid "$PID" > "$D/pyspy_s0_all.log" 2>&1 &
  "$SPY" record --nonblocking --threads --gil --rate 100 --duration $((MIN * 60)) --format raw \
    -o "$D/pyspy_s0_gil.txt" --pid "$PID" > "$D/pyspy_s0_gil.log" 2>&1 &
  wait
  echo "== $(date -Is) py-spy done"
) &
W=$!
bash "$ROOT/bench/slurm/m6.sbatch" "$@"
rc=$?
pkill -P $W 2>/dev/null; kill $W 2>/dev/null; wait
exit $rc
}
