#!/usr/bin/env bash
# Copy trained artefacts into this repo and push to GitHub (works in Git Bash on Windows).
#
#   bash scripts/assemble_and_push.sh <artifacts_dir> https://github.com/<user>/<repo>.git
#
# <artifacts_dir> is either the training workspace itself or the extracted fs2_artifacts.tar.gz
# (same layout: <Lang>/checkpoints, Results/<Lang>, logs/<Lang>, data/<Lang>/preprocessed).
# Needs GH_TOKEN (classic PAT with `repo` scope) in the environment. Copies:
#   <Lang>/checkpoints/{fastspeech2_<lang>.pth.tar, stats.json, speakers.json, inference_meta.json}
#   Results/<Lang>/ (test audio, variance-adaptor plots, loss curves, asr_eval.json)
#   logs/<Lang>/{train,val}_log.csv           -> Results/<Lang>/logs/
#   data/<Lang>/preprocessed/*report.json     -> Results/<Lang>/data/
set -euo pipefail
SRC=${1:?usage: assemble_and_push.sh <training_workspace> <github_https_url>}
URL=${2:?usage: assemble_and_push.sh <training_workspace> <github_https_url>}
: "${GH_TOKEN:?export GH_TOKEN=ghp_... (classic token with repo scope)}"
cd "$(dirname "$0")/.."

command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1 || { echo "Install Git LFS first (https://git-lfs.com)"; exit 1; }
for L in English Telugu; do
  l=$(echo "$L" | tr 'A-Z' 'a-z')
  mkdir -p "$L/checkpoints" "Results/$L/logs" "Results/$L/data"
  cp "$SRC/$L/checkpoints/fastspeech2_$l.pth.tar" "$SRC/$L/checkpoints/stats.json" \
     "$SRC/$L/checkpoints/speakers.json" "$SRC/$L/checkpoints/inference_meta.json" "$L/checkpoints/"
  cp -r "$SRC/Results/$L/." "Results/$L/"
  cp "$SRC/logs/$L/train_log.csv" "$SRC/logs/$L/val_log.csv" "Results/$L/logs/" 2>/dev/null || true
  cp "$SRC/data/$L/preprocessed/data_report.json" "$SRC/data/$L/preprocessed/preprocess_report.json" \
     "Results/$L/data/" 2>/dev/null || true
  echo "[$L] copied: $(du -sh "$L/checkpoints" | cut -f1) checkpoints, $(find "Results/$L" -name '*.wav' | wc -l) wav, $(find "Results/$L" -name '*.png' | wc -l) png"
done


git lfs install --local >/dev/null
git lfs track "*.pth.tar" "*.pth.tar.zip" >/dev/null
git add -A
git -c user.name="${GIT_NAME:-Akshay}" -c user.email="${GIT_EMAIL:-akshayviswa9876@gmail.com}" \
    commit -qm "Add trained checkpoints (step 100k), test-sentence audio, variance-adaptor plots, loss curves, ASR evaluation" || true
git lfs ls-files | sed 's/^/LFS: /'

AUTH_URL=$(echo "$URL" | sed "s#https://#https://x-access-token:${GH_TOKEN}@#")
git push "$AUTH_URL" main:main || {
  echo "Push rejected (remote not empty?). If the GitHub repo was created with a README, rerun with FORCE=1."
  [ "${FORCE:-0}" = "1" ] && git push --force "$AUTH_URL" main:main
}
git remote remove origin 2>/dev/null || true
git remote add origin "$URL"
echo "Done: $URL"
