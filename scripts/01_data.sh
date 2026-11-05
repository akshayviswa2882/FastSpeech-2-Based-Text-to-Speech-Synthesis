#!/usr/bin/env bash
# Data stage for both languages, run in parallel: download/filter/split -> MFA -> features.
# Expect ~1-2 h on 8+ vCPUs (Telugu MFA trains an acoustic model from scratch).
#   tmux new -s data 'bash scripts/01_data.sh 2>&1 | tee logs/data.log'
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/opt/micromamba}
export MFA_CMD=${MFA_CMD:-"micromamba run -n aligner mfa"}
: "${HF_TOKEN:?export HF_TOKEN=hf_... first (Svarah is gated)}"
NPROC=$(nproc); JOBS=$(( NPROC > 3 ? NPROC / 2 : 1 ))

run_lang () {
  local L=$1
  python $L/preprocess.py --stage prepare            > logs/${L}_prepare.log  2>&1
  python $L/preprocess.py --stage align --jobs $JOBS > logs/${L}_align.log    2>&1
  python $L/preprocess.py --stage features --jobs $JOBS > logs/${L}_features.log 2>&1
  echo "[$L] data ready"
}
run_lang English & PID_EN=$!
run_lang Telugu  & PID_TE=$!
wait $PID_EN; wait $PID_TE

for L in English Telugu; do
  echo "== $L"; tail -n 25 logs/${L}_features.log
done
echo "Next: bash scripts/02_train.sh"
