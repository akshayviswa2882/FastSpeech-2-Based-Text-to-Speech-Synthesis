"""Feature extraction from MFA alignments (adapted from ming024 preprocessor/preprocessor.py).

Input layout (produced by <Language>/preprocess.py --stage prepare):
    raw_path/<speaker>/<basename>.wav   (mono, 22.05 kHz)
    raw_path/<speaker>/<basename>.lab   (normalised transcript)
    preprocessed_path/split.json        {"train": ["spk/base", ...], "val": [...]}
    textgrid_path/<speaker>/<basename>.TextGrid  (MFA output)

Output (preprocessed_path): mel/ pitch/ energy/ duration/ *.npy, train.txt,
val.txt, speakers.json, stats.json.

Changes vs. reference:
  * Train/val split is decided upstream (speaker-stratified, duration-targeted)
    and read from split.json instead of a random `val_size` cut.
  * Pitch/energy normalisation statistics are fitted on the TRAIN split only
    (the reference fits on everything, leaking validation statistics).
  * MFA >= 2 writes silences as *empty* intervals; tgt drops empty intervals by
    default, which silently shifts every duration after a pause. We read with
    include_empty_intervals=True and map empty labels to "sp".
  * Utterances containing "spn" (aligner could not model a word) or phones
    outside the language's symbol table are dropped and counted.
  * Multiprocessing for feature extraction; CPU-only STFT (no hard .cuda()).
"""

import json
import os
from functools import partial
from multiprocessing import Pool

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from fs2.audio import MelExtractor
from fs2.text import get_symbols

SIL_PHONES = {"sil", "sp", "spn", ""}


def get_alignment(tier, sampling_rate, hop_length):
    phones, durations = [], []
    start_time = end_time = 0.0
    end_idx = 0
    for t in tier.intervals:
        s, e, p = t.start_time, t.end_time, t.text.strip()
        if p in ("", "sil"):
            p = "sp"
        if not phones and p in SIL_PHONES | {"sp"}:
            if p != "spn":
                continue  # trim leading silence
        if not phones:
            start_time = s
        phones.append(p)
        if p not in ("sp",):
            end_time = e
            end_idx = len(phones)
        durations.append(int(np.round(e * sampling_rate / hop_length) - np.round(s * sampling_rate / hop_length)))
    return phones[:end_idx], durations[:end_idx], start_time, end_time


def _phone_tier(textgrid):
    for tier in textgrid.tiers:
        if tier.name.lower().endswith("phones"):
            return tier
    raise ValueError("No phones tier")


def process_utterance(item, cfg):
    speaker, basename = item
    raw_path, tg_root, out_dir = cfg["raw_path"], cfg["textgrid_path"], cfg["out_dir"]
    sr, hop = cfg["sampling_rate"], cfg["hop_length"]
    tg_path = os.path.join(tg_root, speaker, f"{basename}.TextGrid")
    if not os.path.exists(tg_path):
        return ("missing_textgrid", item)

    try:
        tg = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        phones, duration, start, end = get_alignment(_phone_tier(tg), sr, hop)
    except Exception as exc:  # noqa: BLE001
        return ("bad_textgrid", item, str(exc))
    if start >= end or not phones:
        return ("empty_alignment", item)
    if "spn" in phones:
        return ("spn", item)
    bad = [p for p in phones if p not in cfg["symbols"]]
    if bad:
        return ("unknown_phone", item, " ".join(sorted(set(bad))))

    wav, _ = librosa.load(os.path.join(raw_path, speaker, f"{basename}.wav"), sr=sr)
    wav = wav[int(sr * start): int(sr * end)].astype(np.float32)
    with open(os.path.join(raw_path, speaker, f"{basename}.lab"), encoding="utf-8") as f:
        raw_text = f.readline().strip()

    # F0 (WORLD dio + stonemask), one value per mel frame
    pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop / sr * 1000)
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)

    mel, energy = cfg["extractor"](wav)
    total = int(sum(duration))
    if mel.shape[1] < total or len(pitch) < total:
        return ("length_mismatch", item)
    mel, energy, pitch = mel[:, :total], energy[:total], pitch[:total]
    if np.sum(pitch != 0) <= 1:
        return ("unvoiced", item)

    if cfg["pitch_phoneme"]:
        nz = np.where(pitch != 0)[0]
        pitch = interp1d(nz, pitch[nz], fill_value=(pitch[nz[0]], pitch[nz[-1]]), bounds_error=False)(
            np.arange(len(pitch))
        )
        pitch = _phoneme_average(pitch, duration)
    if cfg["energy_phoneme"]:
        energy = _phoneme_average(energy, duration)

    np.save(os.path.join(out_dir, "duration", f"{speaker}-duration-{basename}.npy"), np.array(duration))
    np.save(os.path.join(out_dir, "pitch", f"{speaker}-pitch-{basename}.npy"), pitch)
    np.save(os.path.join(out_dir, "energy", f"{speaker}-energy-{basename}.npy"), energy)
    np.save(os.path.join(out_dir, "mel", f"{speaker}-mel-{basename}.npy"), mel.T)

    line = "|".join([basename, speaker, " ".join(phones), raw_text])
    return ("ok", item, line, _remove_outlier(pitch), _remove_outlier(energy), mel.shape[1])


def _phoneme_average(values, duration):
    out = np.zeros(len(duration), dtype=np.float64)
    pos = 0
    for i, d in enumerate(duration):
        out[i] = np.mean(values[pos: pos + d]) if d > 0 else 0.0
        pos += d
    return out


def _remove_outlier(values):
    values = np.asarray(values)
    p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
    lower, upper = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
    return values[np.logical_and(values > lower, values < upper)]


class Preprocessor:
    def __init__(self, config, num_workers=None):
        self.config = config
        p = config["preprocessing"]
        self.raw_path = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.textgrid_path = config["path"]["textgrid_path"]
        self.sampling_rate = p["audio"]["sampling_rate"]
        self.hop_length = p["stft"]["hop_length"]
        self.pitch_normalization = p["pitch"]["normalization"]
        self.energy_normalization = p["energy"]["normalization"]
        from fs2.pipeline import available_cpus
        self.num_workers = num_workers or max(1, available_cpus() - 1)
        self.worker_cfg = {
            "raw_path": self.raw_path,
            "textgrid_path": self.textgrid_path,
            "out_dir": self.out_dir,
            "sampling_rate": self.sampling_rate,
            "hop_length": self.hop_length,
            "pitch_phoneme": p["pitch"]["feature"] == "phoneme_level",
            "energy_phoneme": p["energy"]["feature"] == "phoneme_level",
            "symbols": set(get_symbols(config["language"])),
            "extractor": MelExtractor.from_config(config),
        }

    def build_from_path(self):
        for d in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(self.out_dir, d), exist_ok=True)
        with open(os.path.join(self.out_dir, "split.json")) as f:
            split = json.load(f)
        items = {k: [tuple(x.split("/", 1)) for x in v] for k, v in split.items()}
        all_items = items["train"] + items["val"]
        train_set = set(items["train"])

        speakers = sorted({s for s, _ in all_items})
        speaker_map = {s: i for i, s in enumerate(speakers)}

        print(f"Extracting features for {len(all_items)} utterances with {self.num_workers} workers ...")
        fn = partial(process_utterance, cfg=self.worker_cfg)
        results = []
        with Pool(self.num_workers) as pool:
            for r in tqdm(pool.imap_unordered(fn, all_items, chunksize=8), total=len(all_items)):
                results.append(r)

        pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
        lines = {"train": [], "val": []}
        frames = {"train": 0, "val": 0}
        dropped = {}
        for r in results:
            if r[0] != "ok":
                dropped.setdefault(r[0], []).append("/".join(r[1]) + ("" if len(r) < 3 else f" ({r[2]})"))
                continue
            _, item, line, pitch, energy, n = r
            part = "train" if item in train_set else "val"
            lines[part].append(line)
            frames[part] += n
            if part == "train":  # fit normalisation on train only
                if len(pitch):
                    pitch_scaler.partial_fit(pitch.reshape(-1, 1))
                if len(energy):
                    energy_scaler.partial_fit(energy.reshape(-1, 1))

        pitch_mean, pitch_std = (pitch_scaler.mean_[0], pitch_scaler.scale_[0]) if self.pitch_normalization else (0.0, 1.0)
        energy_mean, energy_std = (energy_scaler.mean_[0], energy_scaler.scale_[0]) if self.energy_normalization else (0.0, 1.0)

        kept = {l.split("|")[1] + "-{}-" + l.split("|")[0] for part in lines for l in lines[part]}
        pitch_min, pitch_max = self._normalize("pitch", kept, pitch_mean, pitch_std)
        energy_min, energy_max = self._normalize("energy", kept, energy_mean, energy_std)

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speaker_map, f, indent=1)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump({
                "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
                "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
            }, f, indent=1)
        for part in ("train", "val"):
            with open(os.path.join(self.out_dir, f"{part}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(lines[part])) + "\n")

        hours = {k: v * self.hop_length / self.sampling_rate / 3600 for k, v in frames.items()}
        report = {
            "kept_utterances": {k: len(v) for k, v in lines.items()},
            "hours_after_trimming": hours,
            "dropped": {k: len(v) for k, v in dropped.items()},
            "dropped_examples": {k: v[:20] for k, v in dropped.items()},
            "n_speakers": len(speaker_map),
        }
        with open(os.path.join(self.out_dir, "preprocess_report.json"), "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        print(json.dumps(report, indent=1, ensure_ascii=False)[:3000])
        return report

    def _normalize(self, kind, kept, mean, std):
        lo, hi = np.finfo(np.float64).max, np.finfo(np.float64).min
        d = os.path.join(self.out_dir, kind)
        for key in kept:
            path = os.path.join(d, key.format(kind) + ".npy")
            values = (np.load(path) - mean) / std
            np.save(path, values)
            lo, hi = min(lo, values.min()), max(hi, values.max())
        return lo, hi
