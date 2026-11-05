"""Write Results/<Lang>/training_loss.png, validation_loss.png and train_vs_val.png from CSV logs.

  python scripts/plot_losses.py            # both languages
  python scripts/plot_losses.py English
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fs2.plots import plot_losses, plot_train_vs_val  # noqa: E402
from fs2.utils.model import load_config  # noqa: E402


def main(langs):
    for lang in langs:
        _, _, t, _ = load_config(os.path.join(ROOT, lang, "config.yaml"))
        log, res = t["path"]["log_path"], t["path"]["result_path"]
        os.makedirs(res, exist_ok=True)
        tr, va = os.path.join(log, "train_log.csv"), os.path.join(log, "val_log.csv")
        ok1 = os.path.exists(tr) and plot_losses(tr, f"{lang} - training loss (100-step averages)",
                                                os.path.join(res, "training_loss.png"), smooth=10)
        ok2 = os.path.exists(va) and plot_losses(va, f"{lang} - validation loss (full val set every 1k steps)",
                                                os.path.join(res, "validation_loss.png"))
        ok3 = ok1 and ok2 and plot_train_vs_val(tr, va, f"{lang} - train vs validation",
                                                os.path.join(res, "train_vs_val.png"))
        print(f"{lang}: training_loss={bool(ok1)} validation_loss={bool(ok2)} train_vs_val={bool(ok3)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["English", "Telugu"])
