"""Language-agnostic preprocessing driver used by English/preprocess.py and Telugu/preprocess.py.

Stages
  prepare  : load corpus -> normalise text -> filter -> quality score -> stage 22.05 kHz wavs
             -> speaker-stratified 5h/1h split -> .lab files, lexicon.txt, split.json, data_report.json
  align    : Montreal Forced Aligner (pretrained english_us_arpa for English; trained from
             scratch on the corpus itself for Telugu, as no pretrained Telugu model exists)
  features : mel / F0 / energy / phoneme durations + normalisation stats (fs2.preprocessor)
"""

import json
import os
import re
import shutil
import subprocess
from collections import Counter

from fs2.data_prep import finalize_corpus, quality_metrics, select_and_split, stage_utterance


def _safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_") or "spk"


def prepare(preprocess_config, rows, frontend, speaker_fn, test_sentences=()):
    p = preprocess_config["path"]
    sel = preprocess_config["selection"]
    sr_out = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    raw_path, meta_path = p["raw_path"], p["preprocessed_path"]
    if os.path.exists(raw_path):
        shutil.rmtree(raw_path)
    os.makedirs(meta_path, exist_ok=True)

    forbidden = [frontend.normalize_text(t) for t in test_sentences]
    counts = Counter()
    utts = []
    for row in rows:
        counts["seen"] += 1
        wav, sr = row["wav"], row["sr"]
        dur = len(wav) / sr
        if not (sel["min_duration"] <= dur <= sel["max_duration"]):
            counts["drop_duration"] += 1
            continue
        if hasattr(frontend, "is_supported") and not frontend.is_supported(row["text"]):
            counts["drop_unsupported_script"] += 1
            continue
        norm = frontend.normalize_text(row["text"])
        words = frontend.words_for_alignment(norm)
        if len(words) < sel["min_words"]:
            counts["drop_short_text"] += 1
            continue
        cps = len("".join(words)) / dur
        if not (sel["min_chars_per_sec"] <= cps <= sel["max_chars_per_sec"]):
            counts["drop_speaking_rate"] += 1
            continue
        q = quality_metrics(wav, sr)
        if q["clip_ratio"] > 0.001:
            counts["drop_clipping"] += 1
            continue
        if q["snr_db"] < sel.get("min_snr_db", 0):
            counts["drop_low_snr"] += 1
            continue
        speaker = _safe(speaker_fn(row))
        uid = f"{speaker}_{int(row['id']):05d}"
        dur_out = stage_utterance(raw_path, speaker, uid, wav, sr, sr_out)
        utts.append({"uid": uid, "speaker": speaker, "dur": dur_out, "norm_text": norm,
                     "lab_words": words, "snr_db": q["snr_db"]})
        counts["kept_after_filters"] += 1

    print(f"Filter counts: {dict(counts)}")
    train, val, report = select_and_split(
        utts, sel["train_hours"], sel["val_hours"], seed=sel.get("seed", 1234),
        speaker_cap_frac=sel.get("speaker_cap_frac", 0.5), forbidden_texts=forbidden,
    )
    chosen = set(train) | set(val)
    vocab = sorted({w for u in utts if u["uid"] in chosen for w in u["lab_words"]})
    lexicon = {w: frontend.lexicon_entries(w) for w in vocab}
    lexicon = {w: v for w, v in lexicon.items() if v and all(v)}
    lex_path = finalize_corpus(raw_path, meta_path, utts, train, val, lexicon)

    report["filter_counts"] = dict(counts)
    report["vocabulary_size"] = len(lexicon)
    report["mean_snr_db_selected"] = sum(u["snr_db"] for u in utts if u["uid"] in chosen) / max(len(chosen), 1)
    report["mean_snr_db_rejected"] = sum(u["snr_db"] for u in utts if u["uid"] not in chosen) / max(len(utts) - len(chosen), 1)
    with open(os.path.join(meta_path, "data_report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "per_speaker_minutes"}, indent=1))
    print(f"Lexicon: {lex_path} ({len(lexicon)} words)")


def available_cpus():
    """CPUs actually granted to this container (cgroup quota), not the host's core count."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def align(preprocess_config, mode, pretrained_model=None, num_jobs=None, beam=10, retry_beam=40,
          mfa_config=None):
    """mode: 'pretrained' (mfa align with a downloaded model) or 'train' (mfa train from scratch).

    beam / retry_beam: a wide beam rescues hard utterances when aligning with a good pretrained
    model, but when *training* from scratch the few utterances that fail the first pass are
    re-decoded with the retry beam in every iteration, which can turn a 1-hour run into a day.
    Keep MFA's defaults (10 / 40) for training."""
    p = preprocess_config["path"]
    mfa_cmd = os.environ.get("MFA_CMD", "mfa").split()
    if shutil.which(mfa_cmd[0]) is None:
        raise SystemExit("MFA not found. Run scripts/setup_vast.sh, then either activate the env "
                         "(`micromamba activate aligner`) or export MFA_CMD='micromamba run -n aligner mfa'.")
    raw, tg = p["raw_path"], p["textgrid_path"]
    lex = os.path.join(p["preprocessed_path"], "lexicon.txt")
    jobs = str(num_jobs or max(1, available_cpus() - 1))
    # MFA names its working directory after the corpus folder; both languages use a folder
    # called "raw", so give each language its own MFA root to avoid clobbering a parallel run.
    env = dict(os.environ)
    env.setdefault("MFA_ROOT_DIR", os.path.join(os.path.dirname(p["raw_path"]), "mfa_work"))
    if os.path.exists(tg):
        shutil.rmtree(tg)
    common = ["-j", jobs, "--clean", "--overwrite", "--beam", str(beam), "--retry_beam", str(retry_beam)]
    if mode == "pretrained":
        subprocess.run(mfa_cmd + ["model", "download", "acoustic", pretrained_model], check=False, env=env)
        cmd = mfa_cmd + ["align", raw, lex, pretrained_model, tg] + common
    else:
        model_out = os.path.join(p["preprocessed_path"], "mfa_acoustic_model.zip")
        cmd = mfa_cmd + ["train", raw, lex, model_out, "--output_directory", tg] + common
        if mfa_config:
            cmd += ["--config_path", mfa_config]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    n = sum(f.endswith(".TextGrid") for _, _, fs in os.walk(tg) for f in fs)
    print(f"TextGrids written: {n}")


def features(preprocess_config, num_workers=None):
    from fs2.preprocessor import Preprocessor
    Preprocessor(preprocess_config, num_workers=num_workers).build_from_path()
