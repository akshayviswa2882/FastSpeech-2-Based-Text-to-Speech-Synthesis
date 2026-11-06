"""Telugu inference: synthesise the assignment test sentences (or custom text) and plot
the variance-adaptor predictions (duration / pitch / energy).

  python Telugu/inference.py                          # all test sentences, checkpoint from config
  python Telugu/inference.py --speaker <name>         # pick a training speaker (see speakers.json)
  python Telugu/inference.py --text "..." --id custom # one custom sentence
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import frontend  # noqa: E402
from fs2.common import read_test_sentences  # noqa: E402
from fs2.synthesis import Synthesizer, resolve_checkpoint, resolve_speaker, use_exported_stats_if_needed  # noqa: E402
from fs2.utils.model import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (default: inference.checkpoint in config)")
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--id", default="custom")
    ap.add_argument("--out", default=None, help="default: train.path.result_path from config")
    args = ap.parse_args()

    p, m, t, cfg = load_config(args.config)
    ckpt = resolve_checkpoint(t["path"]["ckpt_path"], args.ckpt, cfg.get("inference", {}).get("checkpoint", "latest"))
    use_exported_stats_if_needed(p, ckpt)
    speaker, speaker_id = resolve_speaker(p, args.speaker or cfg.get("inference", {}).get("speaker"))
    print(f"checkpoint: {ckpt}\nspeaker: {speaker} (id {speaker_id})")
    items = [(args.id, args.text)] if args.text else read_test_sentences("telugu")
    Synthesizer((p, m, t), ckpt).run(items, frontend, speaker_id, args.out or t["path"]["result_path"],
                                     speaker_name=speaker)


if __name__ == "__main__":
    main()
