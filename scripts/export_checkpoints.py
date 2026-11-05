"""Export slim, submission-ready checkpoints (model weights only, no optimizer state).

For each language: <Lang>/checkpoints/best.pth.tar (or latest step) ->
  <Lang>/checkpoints/fastspeech2_<lang>.pth.tar   (~140 MB, tracked with Git LFS)
  <Lang>/checkpoints/stats.json, speakers.json, inference_meta.json
so `python <Lang>/inference.py` works on a fresh clone without the preprocessed data.
Training checkpoints (with optimizer state) are moved to <Lang>/checkpoints/full/ (git-ignored).
"""

import json
import os
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fs2.synthesis import resolve_checkpoint, resolve_speaker  # noqa: E402
from fs2.utils.model import load_config  # noqa: E402

for lang in sys.argv[1:] or ["English", "Telugu"]:
    p, m, t, cfg = load_config(os.path.join(ROOT, lang, "config.yaml"))
    ckpt_dir, pre = t["path"]["ckpt_path"], p["path"]["preprocessed_path"]
    src = resolve_checkpoint(ckpt_dir)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    dst = os.path.join(ckpt_dir, f"fastspeech2_{lang.lower()}.pth.tar")
    torch.save({"model": ck["model"], "step": ck.get("step"), "source": os.path.basename(src),
                "val_postnet_mel_loss": ck.get("val_postnet_mel_loss")}, dst)
    for f in ("stats.json", "speakers.json"):
        shutil.copy(os.path.join(pre, f), os.path.join(ckpt_dir, f))
    spk, _ = resolve_speaker(p, cfg.get("inference", {}).get("speaker"))
    with open(os.path.join(ckpt_dir, "inference_meta.json"), "w") as f:
        json.dump({"default_speaker": spk, "source_checkpoint": os.path.basename(src), "step": ck.get("step")}, f, indent=1)
    full = os.path.join(ckpt_dir, "full")
    os.makedirs(full, exist_ok=True)
    for fn in os.listdir(ckpt_dir):
        if fn.endswith(".pth.tar") and not fn.startswith("fastspeech2_"):
            shutil.move(os.path.join(ckpt_dir, fn), os.path.join(full, fn))
    print(f"{lang}: {src} (step {ck.get('step')}) -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB), default speaker {spk}")
