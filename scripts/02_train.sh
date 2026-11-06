#!/usr/bin/env bash
# Train both models concurrently on one GPU (~35M params each, fits easily in 24 GB).
# Survives SSH disconnects; safe to re-run (resumes from the latest checkpoint).
#   bash scripts/02_train.sh ; tail -f logs/train_English.out
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
nohup python English/train.py > logs/train_English.out 2>&1 &
echo "English training PID $!"
nohup python Telugu/train.py  > logs/train_Telugu.out  2>&1 &
echo "Telugu training PID $!"
echo "Monitor: tail -f logs/train_*.out | grep -E 'step|val'   or   tensorboard --logdir logs --port 6006"
