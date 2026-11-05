"""Shared corpus preparation: loading, quality filtering, subsampling, split, MFA corpus.

Selection strategy (identical code for both languages, parameters per config):
  1. Hard filters: duration in [min_dur, max_dur]; non-empty normalised
     transcript with >= min_words words; speaking rate (characters/second of
     normalised text) within [min_cps, max_cps] to drop truncated or
     mis-transcribed clips; clipping ratio < 0.1%.
  2. Quality score per utterance: crude SNR proxy = 95th - 10th percentile of
     frame RMS (dB). Crowd-sourced Svarah clips vary a lot in recording
     conditions; preferring cleaner clips matters more for TTS than for ASR.
  3. Speaker-stratified budget: every speaker gets a share of the 5h + 1h
     budget proportional to how much usable audio it has (capped so that no
     speaker dominates), the highest-quality clips of each speaker are taken,
     and each speaker's selection is split 5:1 into train/val. Hence every
     validation speaker is seen in training (needed for speaker embeddings) but
     no clip is in both sets.
  4. Transcript disjointness (best effort): if read-speech prompts repeat
     across speakers, a validation clip whose normalised text also occurs in
     train is swapped for an unused clip of the same speaker with unseen text,
     so val loss does not partly measure memorised text. If no such clip
     exists the clip stays in val (audio is still disjoint) and is counted.
  5. Any clip whose text matches a test sentence is excluded from both sets.
"""

import io
import json
import os
import random
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ----------------------------------------------------------------------------- loading

def _pick(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def iter_hf_dataset(name, split=None, token=None, audio_candidates=("audio", "audio_filepath"),
                    text_candidates=("text", "transcript", "sentence"), cache_dir=None):
    """Yield dicts {id, wav(np.float32), sr, text, meta} from a HF dataset without torchcodec."""
    from datasets import Audio, load_dataset

    ds = load_dataset(name, split=split, token=token, cache_dir=cache_dir)
    if isinstance(ds, dict) or hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        ds = ds[list(ds.keys())[0]]
    acol = _pick(ds.column_names, audio_candidates)
    tcol = _pick(ds.column_names, text_candidates)
    if acol is None or tcol is None:
        raise ValueError(f"Could not find audio/text columns in {ds.column_names}")
    print(f"[{name}] columns: {ds.column_names}; audio={acol} text={tcol}; rows={len(ds)}")
    ds = ds.cast_column(acol, Audio(decode=False))
    for i, row in enumerate(ds):
        a = row[acol]
        if a.get("bytes"):
            wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
        else:
            wav, sr = sf.read(a["path"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        meta = {k: v for k, v in row.items() if k not in (acol, tcol)}
        meta["source_path"] = os.path.basename(a.get("path") or f"{i}.wav")
        yield {"id": i, "wav": wav, "sr": sr, "text": row[tcol] or "", "meta": meta}


# ----------------------------------------------------------------------------- quality

def quality_metrics(wav, sr):
    hop = int(0.010 * sr)
    win = int(0.025 * sr)
    if len(wav) < win * 2:
        return {"snr_db": 0.0, "clip_ratio": 1.0}
    frames = librosa.util.frame(np.pad(wav, (0, win)), frame_length=win, hop_length=hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    return {
        "snr_db": float(np.percentile(db, 95) - np.percentile(db, 10)),
        "clip_ratio": float(np.mean(np.abs(wav) > 0.999)),
    }


# ----------------------------------------------------------------------------- selection

def select_and_split(utts, train_hours, val_hours, seed=1234, speaker_cap_frac=0.5,
                     forbidden_texts=()):
    """utts: list of dicts with keys uid, speaker, dur, norm_text, snr_db.

    Returns (train_uids, val_uids, report)."""
    rng = random.Random(seed)
    forbidden = {t.lower() for t in forbidden_texts}
    utts = [u for u in utts if u["norm_text"].lower() not in forbidden]

    by_spk = defaultdict(list)
    for u in utts:
        by_spk[u["speaker"]].append(u)
    for v in by_spk.values():
        v.sort(key=lambda u: -u["snr_db"])  # best quality first

    total_budget = (train_hours + val_hours) * 3600
    avail = {s: sum(u["dur"] for u in v) for s, v in by_spk.items()}
    total_avail = sum(avail.values())
    if total_avail < total_budget:
        print(f"WARNING: only {total_avail / 3600:.2f}h usable audio < requested {total_budget / 3600:.2f}h")

    # proportional share, capped, remainder redistributed
    cap = speaker_cap_frac * total_budget
    share = {s: min(avail[s], cap, total_budget * avail[s] / total_avail) for s in by_spk}
    for _ in range(10):
        left = total_budget - sum(share.values())
        room = {s: min(avail[s], cap) - share[s] for s in by_spk if min(avail[s], cap) - share[s] > 1}
        if left < 1 or not room:
            break
        tot_room = sum(room.values())
        for s, r in room.items():
            share[s] += min(r, left * r / tot_room)

    val_frac = val_hours / (train_hours + val_hours)
    train, val = [], []
    for s, v in sorted(by_spk.items()):
        chosen, acc = [], 0.0
        for u in v:
            if acc >= share[s]:
                break
            chosen.append(u)
            acc += u["dur"]
        rng.shuffle(chosen)
        n_val_target = val_frac * acc
        acc_v = 0.0
        for u in chosen:
            if acc_v < n_val_target and len(chosen) > 1:
                val.append(u)
                acc_v += u["dur"]
            else:
                train.append(u)

    # transcript disjointness: move val items whose text occurs in train
    train_texts = defaultdict(int)
    for u in train:
        train_texts[u["norm_text"].lower()] += 1
    clean_val, leaked, kept_overlap = [], [], []
    for u in val:
        (leaked if train_texts[u["norm_text"].lower()] else clean_val).append(u)
    # replace leaked val clips by unused clips with novel text from the same speaker
    used = {u["uid"] for u in train + val}
    val_texts = {u["norm_text"].lower() for u in clean_val}
    replaced = 0
    for lk in leaked:
        for cand in by_spk[lk["speaker"]]:
            t = cand["norm_text"].lower()
            if cand["uid"] not in used and not train_texts[t] and t not in val_texts:
                clean_val.append(cand)
                used.add(cand["uid"])
                val_texts.add(t)
                replaced += 1
                break
        else:
            kept_overlap.append(lk)  # no replacement: keep in val (audio is still disjoint)
    val = clean_val + kept_overlap

    report = {
        "n_speakers_total": len(by_spk),
        "usable_hours": total_avail / 3600,
        "train_hours": sum(u["dur"] for u in train) / 3600,
        "val_hours": sum(u["dur"] for u in val) / 3600,
        "train_utts": len(train),
        "val_utts": len(val),
        "train_speakers": len({u["speaker"] for u in train}),
        "val_speakers": len({u["speaker"] for u in val}),
        "val_text_overlap_replaced": replaced,
        "val_text_overlap_kept_no_replacement": len(kept_overlap),
        "audio_overlap": len({u["uid"] for u in train} & {u["uid"] for u in val}),
        "text_overlap_after_fix": len({u["norm_text"].lower() for u in train} & {u["norm_text"].lower() for u in val}),
        "per_speaker_minutes": {
            s: {"train": round(sum(u["dur"] for u in train if u["speaker"] == s) / 60, 2),
                "val": round(sum(u["dur"] for u in val if u["speaker"] == s) / 60, 2)}
            for s in sorted(by_spk)
        },
    }
    return [u["uid"] for u in train], [u["uid"] for u in val], report


# ----------------------------------------------------------------------------- corpus writing

def stage_utterance(raw_path, speaker, uid, wav, sr, sr_out=22050):
    """Resample to sr_out, peak-normalise, write raw_path/<speaker>/<uid>.wav. Returns duration (s)."""
    d = os.path.join(raw_path, speaker)
    os.makedirs(d, exist_ok=True)
    if sr != sr_out:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
    wav = (wav / (np.max(np.abs(wav)) + 1e-9) * 0.95).astype(np.float32)
    sf.write(os.path.join(d, f"{uid}.wav"), wav, sr_out, subtype="PCM_16")
    return len(wav) / sr_out


def finalize_corpus(raw_path, meta_path, utts, train_uids, val_uids, lexicon):
    """Write .lab for selected clips, delete unselected staged wavs, write split.json + lexicon.txt
    into meta_path (kept outside the MFA corpus folder so MFA does not parse them).

    utts: list of dicts with uid, speaker, lab_words."""
    keep = set(train_uids) | set(val_uids)
    by_uid = {u["uid"]: u for u in utts}
    for u in utts:
        wav = os.path.join(raw_path, u["speaker"], f"{u['uid']}.wav")
        if u["uid"] in keep:
            with open(wav[:-4] + ".lab", "w", encoding="utf-8") as f:
                f.write(" ".join(u["lab_words"]))
        elif os.path.exists(wav):
            os.remove(wav)
    for d in os.listdir(raw_path):
        full = os.path.join(raw_path, d)
        if os.path.isdir(full) and not os.listdir(full):
            os.rmdir(full)
    os.makedirs(meta_path, exist_ok=True)
    with open(os.path.join(meta_path, "split.json"), "w") as f:
        json.dump({"train": [f"{by_uid[u]['speaker']}/{u}" for u in sorted(train_uids)],
                   "val": [f"{by_uid[u]['speaker']}/{u}" for u in sorted(val_uids)]}, f, indent=0)
    lex_path = os.path.join(meta_path, "lexicon.txt")
    with open(lex_path, "w", encoding="utf-8") as f:
        for w in sorted(lexicon):
            for pron in lexicon[w]:
                f.write(f"{w}\t{' '.join(pron)}\n")
    return lex_path
