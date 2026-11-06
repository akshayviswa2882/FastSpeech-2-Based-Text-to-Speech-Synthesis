"""English preprocessing: Svarah -> 5 h train / 1 h val -> MFA alignment -> features.

Usage (from anywhere):
  python English/preprocess.py --stage prepare     # download + filter + subsample + split + lexicon
  python English/preprocess.py --stage align       # MFA with pretrained english_us_arpa (aligner env)
  python English/preprocess.py --stage features    # mel / pitch / energy / duration
  python English/preprocess.py --stage all         # prepare + features (run align in between)
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import frontend  # noqa: E402
from fs2 import pipeline  # noqa: E402
from fs2.common import read_test_sentences  # noqa: E402
from fs2.utils.model import load_config  # noqa: E402

SPEAKER_META = ["gender", "age-group", "primary_language", "native_place_state",
                "native_place_district", "highest_qualification", "job_category", "occupation_domain"]
N_SPEAKERS_SVARAH = 117


def _choose_speaker_key(columns, paths):
    """Svarah on HF has no explicit speaker column (the GitHub csv does). Pick the most plausible
    speaker key: explicit id column > filename prefix > tuple of per-speaker metadata.
    Chosen as the candidate whose number of distinct values is closest to the 117 speakers."""
    for c in ("speaker_id", "speaker", "client_id"):
        if c in columns:
            return ("column", c)
    cands = {}
    prefixes = [os.path.basename(p).split("_")[0] for p in paths]
    if prefixes and len(set(prefixes)) > 1:
        cands[("prefix", None)] = len(set(prefixes))
    meta_cols = [c for c in SPEAKER_META if c in columns]
    if meta_cols:
        cands[("meta", tuple(meta_cols))] = None
    return cands


def load_rows_hf(src, token):
    from datasets import Audio, load_dataset
    import io
    import soundfile as sf

    ds = load_dataset(src["hf_name"], split=src["hf_split"], token=token)
    acol = "audio_filepath" if "audio_filepath" in ds.column_names else "audio"
    ds = ds.cast_column(acol, Audio(decode=False))
    print(f"Svarah columns: {ds.column_names}, rows: {len(ds)}")
    meta_ds = ds.remove_columns([acol])
    paths = [a.get("path") or "" for a in ds[acol]] if "speaker_id" not in ds.column_names else []
    key = _choose_speaker_key(ds.column_names, paths)
    if isinstance(key, dict):  # decide between prefix and metadata tuple
        options = {}
        for k in key:
            if k[0] == "prefix":
                options[k] = key[k]
            else:
                tuples = {tuple(r[c] for c in k[1]) for r in meta_ds}
                options[k] = len(tuples)
        key = min(options, key=lambda k: abs(options[k] - N_SPEAKERS_SVARAH))
        print(f"Speaker key candidates (distinct values): {options} -> using {key}")

    def speaker_of(row):
        kind, val = key
        if kind == "column":
            return row["meta"][val]
        if kind == "prefix":
            return "spk" + os.path.basename(row["meta"]["source_path"]).split("_")[0]
        key_str = "|".join(str(row["meta"].get(c)) for c in val)
        return "spk" + hashlib.md5(key_str.encode()).hexdigest()[:8]

    def rows():
        for i, r in enumerate(ds):
            a = r[acol]
            wav, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            meta = {k: v for k, v in r.items() if k not in (acol, "text")}
            meta["source_path"] = a.get("path") or f"{i}.wav"
            yield {"id": i, "wav": wav, "sr": sr, "text": r["text"] or "", "meta": meta}

    return rows(), speaker_of


def load_rows_local(manifest):
    import soundfile as sf
    base = os.path.dirname(os.path.abspath(manifest))

    def rows():
        with open(manifest, encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                path = r["audio_filepath"] if os.path.isabs(r["audio_filepath"]) else os.path.join(base, r["audio_filepath"])
                wav, sr = sf.read(path, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                yield {"id": i, "wav": wav, "sr": sr, "text": r.get("text", ""),
                       "meta": {**r, "source_path": path}}

    def speaker_of(row):
        return row["meta"].get("speaker_id") or "spk" + os.path.basename(row["meta"]["source_path"]).split("_")[0]

    return rows(), speaker_of


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
        if src.get("local_manifest"):
            rows, speaker_of = load_rows_local(src["local_manifest"])
        else:
            rows, speaker_of = load_rows_hf(src, args.hf_token)
        tests = [t for _, t in read_test_sentences("english")]
        pipeline.prepare(preprocess_config, rows, frontend, speaker_of, test_sentences=tests)
    if args.stage == "align":
        a = preprocess_config["alignment"]
        pipeline.align(preprocess_config, a["mode"], a.get("pretrained_model"), args.jobs,
                       a.get("beam", 10), a.get("retry_beam", 40))
    if args.stage in ("features", "all"):
        if args.stage == "all" and not os.path.isdir(preprocess_config["path"]["textgrid_path"]):
            print("No TextGrids yet - run `--stage align` in the aligner env, then `--stage features`.")
            return
        pipeline.features(preprocess_config, args.jobs)


if __name__ == "__main__":
    main()
