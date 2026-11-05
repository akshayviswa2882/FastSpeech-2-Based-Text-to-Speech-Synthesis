"""CPU end-to-end smoke test on synthetic audio (no dataset / GPU / MFA needed).

Exercises every stage with the real code paths: prepare (filters, quality score,
speaker-stratified split, lexicon), feature extraction from MFA-style TextGrids
(including MFA>=2 empty silence intervals), a few training steps with
validation + checkpointing, inference with plots, and loss plots.

  python scripts/smoke_test.py english
  python scripts/smoke_test.py telugu
"""

import copy
import os
import shutil
import sys
import tempfile

import numpy as np
import tgt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fake_rows(sentences, n_speakers=3, per_speaker=12, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    i = 0
    for s in range(n_speakers):
        f0 = 110 + 60 * s
        for k in range(per_speaker):
            text = sentences[(s * per_speaker + k) % len(sentences)]
            dur = 1.5 + 0.06 * len(text.split()) + rng.uniform(0, 1.0)
            t = np.arange(int(dur * sr)) / sr
            vib = f0 * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
            phase = 2 * np.pi * np.cumsum(vib) / sr
            env = (np.sin(2 * np.pi * 4 * t) > -0.3).astype(float)
            wav = sum(np.sin(h * phase) / h for h in range(1, 6)) * env * 0.3
            wav = wav + 0.003 * rng.standard_normal(len(wav))
            wav[: int(0.2 * sr)] *= 0.01
            wav[-int(0.2 * sr):] *= 0.01
            yield {"id": i, "wav": wav.astype(np.float32), "sr": sr, "text": text,
                   "meta": {"gender": "female" if s % 2 else "male", "spk": f"S{s}"}}
            i += 1


def fake_textgrids(raw_path, tg_path, lexicon_path):
    lex = {}
    with open(lexicon_path, encoding="utf-8") as f:
        for line in f:
            w, p = line.rstrip("\n").split("\t")
            lex.setdefault(w, p.split())
    import soundfile as sf
    for spk in os.listdir(raw_path):
        for fn in os.listdir(os.path.join(raw_path, spk)):
            if not fn.endswith(".lab"):
                continue
            base = fn[:-4]
            words = open(os.path.join(raw_path, spk, fn), encoding="utf-8").read().split()
            info = sf.info(os.path.join(raw_path, spk, base + ".wav"))
            total = info.frames / info.samplerate
            phones = []
            for j, w in enumerate(words):
                phones += lex[w]
                if j == len(words) // 2:
                    phones.append("")  # MFA>=2 style empty silence interval mid-utterance
            start, end = 0.2, total - 0.2
            step = (end - start) / len(phones)
            tier = tgt.IntervalTier(0, total, "phones")
            tier.add_interval(tgt.Interval(0, start, ""))
            for k, p in enumerate(phones):
                tier.add_interval(tgt.Interval(start + k * step, start + (k + 1) * step, p))
            tier.add_interval(tgt.Interval(end, total, ""))
            tg = tgt.TextGrid()
            tg.add_tier(tier)
            os.makedirs(os.path.join(tg_path, spk), exist_ok=True)
            tgt.io.write_to_file(tg, os.path.join(tg_path, spk, base + ".TextGrid"), format="long")


def main(language):
    lang_dir = os.path.join(ROOT, language.capitalize())
    sys.path[:0] = [ROOT, lang_dir]
    import frontend
    import yaml
    from fs2 import pipeline
    from fs2.common import read_test_sentences
    from fs2.plots import plot_losses
    from fs2.synthesis import Synthesizer, resolve_checkpoint, resolve_speaker
    from fs2.trainer import train
    from fs2.utils.model import load_config

    work = tempfile.mkdtemp(prefix=f"fs2_smoke_{language}_")
    cfg_path = os.path.join(lang_dir, "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["preprocess"]["path"] = {k: os.path.join(work, k) for k in ("raw_path", "textgrid_path", "preprocessed_path")}
    cfg["preprocess"]["selection"].update(train_hours=0.015, val_hours=0.004, min_snr_db=0.0, speaker_cap_frac=0.5,
                                          min_chars_per_sec=1.0, max_chars_per_sec=60.0)
    cfg["train"]["path"] = {k: os.path.join(work, k) for k in ("ckpt_path", "log_path", "result_path")}
    cfg["train"]["optimizer"].update(batch_size=2, group_size=2, warm_up_step=10)
    cfg["train"]["step"].update(total_step=24, log_step=4, val_step=8, save_step=12)
    cfg["train"]["num_workers"] = 0
    tmp_cfg = os.path.join(work, "config.yaml")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    p, m, t, full = load_config(tmp_cfg)

    if language == "english":
        sents = ["The quick brown fox jumps over the lazy dog.", "She sells sea shells by the sea shore.",
                 "We will conduct the meeting at 9:30 AM on 3 March 2025.", "The orchestra's conduct was good.",
                 "Speech synthesis needs clean data.", "Rhythm and melody change over time."]
    else:
        sents = ["తాడుపై అతని పట్టు చాలా బలంగా ఉంది.", "ఆమె పాట పాడింది.", "ఈ రోజు 12వ తేదీ.",
                 "పులి అడవిలో నడిచింది.", "మేము 2025 లో కలిశాము.", "సమావేశం 10:30 కి మొదలవుతుంది."]

    print("== prepare");  pipeline.prepare(p, fake_rows(sents), frontend, lambda r: r["meta"]["spk"],
                                         test_sentences=[x for _, x in read_test_sentences(language)])
    print("== fake align"); fake_textgrids(p["path"]["raw_path"], p["path"]["textgrid_path"],
                                          os.path.join(p["path"]["preprocessed_path"], "lexicon.txt"))
    print("== features"); pipeline.features(p, num_workers=2)
    print("== train");    train((p, m, t), resume=True)
    print("== resume");   t["step"]["total_step"] = 28; train((p, m, t), resume=True)
    print("== inference")
    ckpt = resolve_checkpoint(t["path"]["ckpt_path"])
    spk, sid = resolve_speaker(p, "auto")
    Synthesizer((p, m, t), ckpt).run(read_test_sentences(language), frontend, sid, t["path"]["result_path"], spk)
    assert plot_losses(os.path.join(t["path"]["log_path"], "train_log.csv"), "train", os.path.join(work, "tl.png"))
    assert plot_losses(os.path.join(t["path"]["log_path"], "val_log.csv"), "val", os.path.join(work, "vl.png"))
    print(f"SMOKE TEST OK ({language}) -> {work}")
    return work


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "english")
