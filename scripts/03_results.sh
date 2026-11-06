#!/usr/bin/env bash
# After training: test-sentence audio + variance-adaptor plots, loss curves, ASR evaluation,
# then slim checkpoints for the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
for L in English Telugu; do
  python $L/inference.py
done
python scripts/plot_losses.py
python scripts/evaluate_asr.py English --n_val 60 | tee logs/asr_English.log
python scripts/evaluate_asr.py Telugu  --n_val 60 | tee logs/asr_Telugu.log
python scripts/export_checkpoints.py
echo "Results in Results/. Commit with: git lfs track '*.pth.tar' && git add -A"
