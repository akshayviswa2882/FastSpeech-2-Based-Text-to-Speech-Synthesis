"""Train the English FastSpeech 2 model.

  python English/train.py                 # resumes automatically from the latest checkpoint
  python English/train.py --no_resume     # start from scratch
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from fs2.trainer import train  # noqa: E402
from fs2.utils.model import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--total_step", type=int, default=None, help="override train.step.total_step")
    args = ap.parse_args()
    p, m, t, _ = load_config(args.config)
    if args.total_step:
        t["step"]["total_step"] = args.total_step
    train((p, m, t), resume=not args.no_resume)


if __name__ == "__main__":
    main()
