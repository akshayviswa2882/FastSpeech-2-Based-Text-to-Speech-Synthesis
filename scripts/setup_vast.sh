#!/usr/bin/env bash
# One-time setup on a fresh GPU box (tested on vast.ai PyTorch images, Ubuntu 22.04).
#   bash scripts/setup_vast.sh
set -euo pipefail
cd "$(dirname "$0")/.."

apt-get update -qq && apt-get install -y -qq git-lfs tmux bzip2 curl libsndfile1 >/dev/null || true
pip install -q -r requirements.txt

python - <<'PY'
import nltk
for p in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"]:
    nltk.download(p, quiet=True)
PY

# Montreal Forced Aligner in its own env (needs kaldi / openfst binaries from conda-forge)
if ! command -v micromamba >/dev/null 2>&1; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba
fi
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/opt/micromamba}
if ! micromamba env list | grep -q "^ *aligner "; then
  micromamba create -y -q -n aligner -c conda-forge "montreal-forced-aligner>=3.0" python=3.11
fi
micromamba run -n aligner mfa version

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo
echo "Setup done. Next:"
echo "  1. Accept the Svarah terms at https://huggingface.co/datasets/ai4bharat/Svarah (and IndicTTS_Telugu if asked)"
echo "  2. export HF_TOKEN=hf_xxx"
echo "  3. bash scripts/01_data.sh"
