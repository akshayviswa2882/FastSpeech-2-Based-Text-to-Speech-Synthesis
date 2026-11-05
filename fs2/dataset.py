"""Training / validation dataset (adapted from ming024 dataset.py).

Metadata lines: basename|speaker|phones (space separated)|raw_text
Batching keeps the reference trick: the DataLoader yields a "group" of
batch_size * group_size utterances, which collate_fn sorts by length and splits
into group_size batches with similar lengths (less padding).
"""

import json
import os

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from fs2.text import phones_to_sequence
from fs2.utils.tools import pad_1D, pad_2D


def read_metadata(path):
    names, speakers, texts, raw = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            n, s, t, r = line.split("|", 3)
            names.append(n)
            speakers.append(s)
            texts.append(t)
            raw.append(r)
    return names, speakers, texts, raw


class Dataset(TorchDataset):
    def __init__(self, filename, preprocess_config, train_config, sort=False, drop_last=False):
        self.language = preprocess_config["language"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.basename, self.speaker, self.text, self.raw_text = read_metadata(
            os.path.join(self.preprocessed_path, filename)
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def _load(self, kind, speaker, basename):
        return np.load(os.path.join(self.preprocessed_path, kind, f"{speaker}-{kind}-{basename}.npy"))

    def __getitem__(self, idx):
        basename, speaker = self.basename[idx], self.speaker[idx]
        return {
            "id": basename,
            "speaker": self.speaker_map[speaker],
            "text": np.array(phones_to_sequence(self.text[idx], self.language)),
            "raw_text": self.raw_text[idx],
            "mel": self._load("mel", speaker, basename),
            "pitch": self._load("pitch", speaker, basename),
            "energy": self._load("energy", speaker, basename),
            "duration": self._load("duration", speaker, basename),
        }

    def reprocess(self, data, idxs):
        texts = [data[i]["text"] for i in idxs]
        mels = [data[i]["mel"] for i in idxs]
        text_lens = np.array([t.shape[0] for t in texts])
        mel_lens = np.array([m.shape[0] for m in mels])
        return (
            [data[i]["id"] for i in idxs],
            [data[i]["raw_text"] for i in idxs],
            np.array([data[i]["speaker"] for i in idxs]),
            pad_1D(texts),
            text_lens,
            int(max(text_lens)),
            pad_2D(mels),
            mel_lens,
            int(max(mel_lens)),
            pad_1D([data[i]["pitch"] for i in idxs]),
            pad_1D([data[i]["energy"] for i in idxs]),
            pad_1D([data[i]["duration"] for i in idxs]),
        )

    def collate_fn(self, data):
        n = len(data)
        if self.sort:
            idx_arr = np.argsort(-np.array([d["text"].shape[0] for d in data]))
        else:
            idx_arr = np.arange(n)
        cut = n - (n % self.batch_size)
        tail = idx_arr[cut:]
        groups = idx_arr[:cut].reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            groups.append(tail.tolist())
        return [self.reprocess(data, g) for g in groups]


class TextBatch:
    """Builds a 6-tuple inference batch from already-phonemised utterances."""

    @staticmethod
    def build(items, language):
        """items: list of (id, speaker_id, phones(list[str]), raw_text)."""
        seqs = [np.array(phones_to_sequence(p, language)) for _, _, p, _ in items]
        lens = np.array([len(s) for s in seqs])
        return (
            [i[0] for i in items],
            [i[3] for i in items],
            np.array([i[1] for i in items]),
            pad_1D(seqs),
            lens,
            int(max(lens)),
        )
