"""Objective intelligibility evaluation with ASR (CER / WER).

For each language:
  * test sentences   : synthesised audio -> ASR -> compare with the normalised input text
  * N val utterances : ground-truth recording vs. FastSpeech 2 synthesis of the same text with
                       the same speaker -> the gap measures what the TTS model loses in
                       intelligibility (ASR errors on the recordings themselves form the floor)

Default ASR models (override with --asr_model):
  English: openai/whisper-large-v3
  Telugu : vasista22/whisper-telugu-large-v2 (Whisper fine-tuned on Telugu; plain Whisper is weak on Telugu)

  python scripts/evaluate_asr.py English --n_val 60
  python scripts/evaluate_asr.py Telugu  --n_val 60
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ASR = {"english": "openai/whisper-large-v3", "telugu": "vasista22/whisper-telugu-large-v2"}


def norm_for_scoring(text, frontend):
    text = frontend.normalize_text(text)  # same number/time expansion on both sides
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("language", choices=["English", "Telugu"])
    ap.add_argument("--asr_model", default=None)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lang_dir = os.path.join(ROOT, args.language)
    sys.path[:0] = [ROOT, lang_dir]
    import frontend
    import jiwer
    import librosa
    import torch
    from transformers import pipeline

    from fs2.common import read_test_sentences
    from fs2.dataset import read_metadata
    from fs2.synthesis import Synthesizer, resolve_checkpoint, resolve_speaker
    from fs2.utils.model import load_config

    p, m, t, cfg = load_config(os.path.join(lang_dir, "config.yaml"))
    lang = p["language"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_name = args.asr_model or DEFAULT_ASR[lang]
    asr = pipeline("automatic-speech-recognition", model=asr_name, device=device,
                   torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    gen = {"task": "transcribe", "language": "en" if lang == "english" else "te"}

    def transcribe(wav, sr):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.dtype != np.float32 or np.abs(wav).max() > 1.5:
            wav = wav / 32768.0
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return asr({"raw": wav16, "sampling_rate": 16000}, generate_kwargs=gen)["text"]

    synth = Synthesizer((p, m, t), resolve_checkpoint(t["path"]["ckpt_path"], args.ckpt,
                                                     cfg.get("inference", {}).get("checkpoint", "latest")))
    spk_name, spk_id = resolve_speaker(p, cfg.get("inference", {}).get("speaker"))
    with open(os.path.join(p["path"]["preprocessed_path"], "speakers.json")) as f:
        speakers = json.load(f)

    rows = []
    # 1) test sentences
    for sid, text in read_test_sentences(lang):
        phones, _, _ = frontend.text_to_phones(text)
        wav, *_ = synth.synthesize(sid, phones, spk_id)
        hyp = transcribe(wav, synth.sr)
        rows.append({"set": "test", "id": sid, "speaker": spk_name, "ref": norm_for_scoring(text, frontend),
                     "hyp_synth": norm_for_scoring(hyp, frontend), "hyp_gt": None})

    # 2) validation utterances: recording vs synthesis
    names, spks, _, raws = read_metadata(os.path.join(p["path"]["preprocessed_path"], "val.txt"))
    idx = list(range(len(names)))
    random.Random(args.seed).shuffle(idx)
    for i in idx[: args.n_val]:
        gt, sr = librosa.load(os.path.join(p["path"]["raw_path"], spks[i], f"{names[i]}.wav"), sr=None)
        phones, _, _ = frontend.text_to_phones(raws[i], normalized=True)
        wav, *_ = synth.synthesize(names[i], phones, speakers[spks[i]])
        rows.append({"set": "val", "id": names[i], "speaker": spks[i], "ref": norm_for_scoring(raws[i], frontend),
                     "hyp_gt": norm_for_scoring(transcribe(gt, sr), frontend),
                     "hyp_synth": norm_for_scoring(transcribe(wav, synth.sr), frontend)})

    def score(refs, hyps):
        pairs = [(r, h if h else "<empty>") for r, h in zip(refs, hyps) if r]
        if not pairs:
            return None
        r, h = zip(*pairs)
        return {"wer": round(jiwer.wer(list(r), list(h)) * 100, 2), "cer": round(jiwer.cer(list(r), list(h)) * 100, 2)}

    test = [r for r in rows if r["set"] == "test"]
    val = [r for r in rows if r["set"] == "val"]
    for r in test:
        r.update({f"{k}_synth": v for k, v in score([r["ref"]], [r["hyp_synth"]]).items()})
    summary = {
        "asr_model": asr_name,
        "speaker_for_test_sentences": spk_name,
        "test_sentences_synth": score([r["ref"] for r in test], [r["hyp_synth"] for r in test]),
        "val_ground_truth": score([r["ref"] for r in val], [r["hyp_gt"] for r in val]),
        "val_synth": score([r["ref"] for r in val], [r["hyp_synth"] for r in val]),
        "n_val": len(val),
    }
    out_dir = t["path"]["result_path"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "asr_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "utterances": rows}, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
    for r in test:
        print(f"{r['id']}: CER {r['cer_synth']}%\n   ref: {r['ref']}\n   hyp: {r['hyp_synth']}")


if __name__ == "__main__":
    main()
