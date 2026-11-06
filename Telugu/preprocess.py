"""Telugu preprocessing: IndicTTS Telugu -> 5 h train / 1 h val -> MFA (trained) -> features.

IndicTTS Telugu has only ~8.7 h in total, split over one male (4.47 h) and one
female (4.27 h) speaker, so a single speaker cannot supply 5 h + 1 h. Both
speakers are used with a speaker embedding, with a balanced 2.5 h + 0.5 h each.

Usage:
  python Telugu/preprocess.py --stage prepare
  python Telugu/preprocess.py --stage align       # trains an MFA acoustic model on the corpus
  python Telugu/preprocess.py --stage features
"""

import argparse
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import frontend  # noqa: E402
from fs2 import pipeline  # noqa: E402
from fs2.common import read_test_sentences  # noqa: E402
from fs2.utils.model import load_config  # noqa: E402


def _gender_of(meta):
    for k in ("gender", "speaker", "speaker_id"):
        if k in meta and meta[k] is not None:
            v = str(meta[k]).lower()
            if v.startswith("f") or "fem" in v:
                return "te_female"
            if v.startswith("m") or "male" in v:
                return "te_male"
            return "te_" + v
    name = os.path.basename(meta.get("source_path", "")).lower()
    if "fem" in name:
        return "te_female"
    if "male" in name:
        return "te_male"
    return "te_unknown"


def load_rows_hf(src, token):
    from datasets import Audio, load_dataset
    import soundfile as sf

    ds = load_dataset(src["hf_name"], split=src["hf_split"], token=token)
    acol = "audio" if "audio" in ds.column_names else ds.column_names[0]
    tcol = "text" if "text" in ds.column_names else [c for c in ds.column_names if c != acol][0]
    print(f"IndicTTS columns: {ds.column_names}, rows: {len(ds)}")
    ds = ds.cast_column(acol, Audio(decode=False))

    def rows():
        for i, r in enumerate(ds):
            a = r[acol]
            wav, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            meta = {k: v for k, v in r.items() if k not in (acol, tcol)}
            meta["source_path"] = a.get("path") or f"{i}.wav"
            yield {"id": i, "wav": wav, "sr": sr, "text": r[tcol] or "", "meta": meta}

    return rows(), lambda row: _gender_of(row["meta"])


def load_rows_local(local_dir):
    """Original IIT-M release: <dir>/<speaker>/wav/*.wav and <dir>/<speaker>/txt.done.data."""
    import soundfile as sf
    entry = re.compile(r'\(\s*(\S+)\s+"(.*)"\s*\)')

    def rows():
        i = 0
        for spk in sorted(os.listdir(local_dir)):
            sdir = os.path.join(local_dir, spk)
            txt = os.path.join(sdir, "txt.done.data")
            if not os.path.isfile(txt):
                continue
            with open(txt, encoding="utf-8") as f:
                for line in f:
                    m = entry.search(line)
                    if not m:
                        continue
                    path = os.path.join(sdir, "wav", m.group(1) + ".wav")
                    if not os.path.exists(path):
                        continue
                    wav, sr = sf.read(path, dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    yield {"id": i, "wav": wav, "sr": sr, "text": m.group(2),
                           "meta": {"speaker": spk, "source_path": path}}
                    i += 1

    return rows(), lambda row: _gender_of(row["meta"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--stage", choices=["prepare", "align", "features", "all"], default="all")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()

    preprocess_config, _, _, _ = load_config(args.config)
    src = preprocess_config["source"]

    if args.stage in ("prepare", "all"):
        if src.get("local_dir"):
            rows, speaker_of = load_rows_local(src["local_dir"])
        else:
            rows, speaker_of = load_rows_hf(src, args.hf_token)
        tests = [t for _, t in read_test_sentences("telugu")]
        pipeline.prepare(preprocess_config, rows, frontend, speaker_of, test_sentences=tests)
    if args.stage == "align":
        a = preprocess_config["alignment"]
        pipeline.align(preprocess_config, a["mode"], a.get("pretrained_model"), args.jobs,
                       a.get("beam", 10), a.get("retry_beam", 40), a.get("mfa_config"))
    if args.stage in ("features", "all"):
        if args.stage == "all" and not os.path.isdir(preprocess_config["path"]["textgrid_path"]):
            print("No TextGrids yet - run `--stage align` in the aligner env, then `--stage features`.")
            return
        pipeline.features(preprocess_config, args.jobs)


if __name__ == "__main__":
    main()
