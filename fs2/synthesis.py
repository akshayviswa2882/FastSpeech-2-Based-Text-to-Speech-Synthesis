"""Inference: text -> phones -> FastSpeech 2 -> HiFi-GAN, plus variance-adaptor plots.

For every sentence it writes
  <result>/test_generated_audios/<ID>.wav
  <result>/test_generated_audios/<ID>_predictions.json   (phones, durations, F0 Hz, energy)
  <result>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png
and a frontend.json with the normalised text + phone sequence of each sentence.
"""

import glob
import json
import os
import re

import numpy as np
import torch
from scipy.io import wavfile

from fs2.dataset import TextBatch
from fs2.plots import C_ENERGY, C_PITCH, plot_contour, plot_duration
from fs2.utils.model import get_model, get_vocoder, vocoder_infer
from fs2.utils.tools import to_device


def resolve_checkpoint(ckpt_dir, ckpt=None):
    if ckpt:
        return ckpt
    best = os.path.join(ckpt_dir, "best.pth.tar")
    if os.path.exists(best):
        return best
    steps = [(int(m.group(1)), p) for p in glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
             if (m := re.search(r"(\d+)\.pth\.tar$", p))]
    if steps:
        return max(steps)[1]
    exported = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth.tar")))
    if exported:
        return exported[0]
    raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")


def use_exported_stats_if_needed(preprocess_config, ckpt_path):
    """If the preprocessed data folder is absent (fresh clone), read stats.json / speakers.json
    from the folder of the exported checkpoint (scripts/export_checkpoints.py puts them there)."""
    pre = preprocess_config["path"]["preprocessed_path"]
    if not os.path.exists(os.path.join(pre, "stats.json")):
        preprocess_config["path"]["preprocessed_path"] = os.path.dirname(os.path.abspath(ckpt_path))


def resolve_speaker(preprocess_config, speaker):
    pre = preprocess_config["path"]["preprocessed_path"]
    with open(os.path.join(pre, "speakers.json")) as f:
        speakers = json.load(f)
    meta_file = os.path.join(pre, "inference_meta.json")
    if speaker in (None, "auto") and not os.path.exists(os.path.join(pre, "train.txt")) and os.path.exists(meta_file):
        with open(meta_file) as f:
            speaker = json.load(f)["default_speaker"]
    if speaker in (None, "auto"):
        mins = {}
        with open(os.path.join(pre, "train.txt"), encoding="utf-8") as f:
            for line in f:
                base, spk = line.split("|", 2)[:2]
                n = np.load(os.path.join(pre, "duration", f"{spk}-duration-{base}.npy")).sum()
                mins[spk] = mins.get(spk, 0) + n
        speaker = max(mins, key=mins.get)
    if speaker not in speakers:
        raise KeyError(f"Unknown speaker {speaker}; choose from {sorted(speakers)}")
    return speaker, speakers[speaker]


class Synthesizer:
    def __init__(self, configs, ckpt_path, device=None):
        self.preprocess_config, self.model_config, self.train_config = configs
        self.language = self.preprocess_config["language"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(ckpt_path, configs, self.device, train=False)
        self.vocoder = get_vocoder(self.model_config, self.device)
        pre = self.preprocess_config["path"]["preprocessed_path"]
        with open(os.path.join(pre, "stats.json")) as f:
            stats = json.load(f)
        self.pitch_mean, self.pitch_std = stats["pitch"][2:]
        self.energy_mean, self.energy_std = stats["energy"][2:]
        p = self.preprocess_config["preprocessing"]
        self.sr = p["audio"]["sampling_rate"]
        self.hop = p["stft"]["hop_length"]
        self.frame_ms = self.hop / self.sr * 1000

    @torch.no_grad()
    def synthesize(self, sid, phones, speaker_id, p_control=1.0, e_control=1.0, d_control=1.0):
        batch = to_device(TextBatch.build([(sid, speaker_id, phones, "")], self.language), self.device)
        out = self.model(*(batch[2:]), p_control=p_control, e_control=e_control, d_control=d_control)
        mel_len = int(out[9][0].item())
        mel = out[1][0, :mel_len].transpose(0, 1)  # postnet mel [n_mels, T]
        wav = vocoder_infer(mel.unsqueeze(0), self.vocoder, self.preprocess_config,
                            lengths=[mel_len * self.hop])[0]
        n = len(phones)
        pitch = out[2][0, :n].cpu().numpy() * self.pitch_std + self.pitch_mean
        energy = out[3][0, :n].cpu().numpy() * self.energy_std + self.energy_mean
        dur = out[5][0, :n].cpu().numpy().astype(int)
        return wav, mel.cpu().numpy(), dur, pitch, energy

    def run(self, items, frontend, speaker_id, result_dir, speaker_name=""):
        """items: list of (sid, text)."""
        audio_dir = os.path.join(result_dir, "test_generated_audios")
        plot_dirs = {k: os.path.join(result_dir, "variance_adaptor_plots", k) for k in ("duration", "pitch", "energy")}
        for d in [audio_dir, *plot_dirs.values()]:
            os.makedirs(d, exist_ok=True)

        summary = []
        for sid, text in items:
            phones, spans, norm = frontend.text_to_phones(text)
            wav, mel, dur, pitch, energy = self.synthesize(sid, phones, speaker_id)
            wavfile.write(os.path.join(audio_dir, f"{sid}.wav"), self.sr, wav)
            rec = {
                "id": sid, "text": text, "normalized": norm, "speaker": speaker_name,
                "phones": phones, "words": [{"word": w, "start": s, "end": e} for w, s, e in spans],
                "duration_frames": dur.tolist(),
                "duration_ms": [round(float(d * self.frame_ms), 1) for d in dur],
                "pitch_hz": [round(float(x), 2) for x in pitch],
                "energy": [round(float(x), 3) for x in energy],
                "total_seconds": round(float(dur.sum() * self.frame_ms / 1000), 3),
            }
            with open(os.path.join(audio_dir, f"{sid}_predictions.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)

            plot_duration(phones, dur, spans, self.frame_ms, f"{sid} - predicted phoneme durations",
                          os.path.join(plot_dirs["duration"], f"{sid}.png"))
            plot_contour(pitch, phones, dur, spans, mel, self.frame_ms, "predicted F0 (Hz)", C_PITCH,
                         f"{sid} - predicted pitch (phoneme-level F0)",
                         os.path.join(plot_dirs["pitch"], f"{sid}.png"))
            plot_contour(energy, phones, dur, spans, mel, self.frame_ms, "predicted energy (L2 of |STFT|)",
                         C_ENERGY, f"{sid} - predicted energy",
                         os.path.join(plot_dirs["energy"], f"{sid}.png"))
            print(f"[{sid}] {rec['total_seconds']:.2f}s  {norm}\n      {' '.join(phones)}")
            summary.append({k: rec[k] for k in ("id", "text", "normalized", "phones", "total_seconds")})

        with open(os.path.join(audio_dir, "frontend.json"), "w", encoding="utf-8") as f:
            json.dump({"speaker": speaker_name, "sentences": summary}, f, indent=1, ensure_ascii=False)
        return summary
