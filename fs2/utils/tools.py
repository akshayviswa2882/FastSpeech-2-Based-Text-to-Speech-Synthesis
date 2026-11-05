"""Tensor / padding helpers (adapted from ming024 utils/tools.py; device-agnostic)."""

import numpy as np
import torch
import torch.nn.functional as F


def to_device(data, device):
    """Move a collated batch (12-tuple for training, 6-tuple for inference) to device."""
    if len(data) == 12:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, mels, mel_lens,
         max_mel_len, pitches, energies, durations) = data
        return (
            ids, raw_texts,
            torch.from_numpy(speakers).long().to(device),
            torch.from_numpy(texts).long().to(device),
            torch.from_numpy(src_lens).long().to(device),
            max_src_len,
            torch.from_numpy(mels).float().to(device),
            torch.from_numpy(mel_lens).long().to(device),
            max_mel_len,
            torch.from_numpy(pitches).float().to(device),
            torch.from_numpy(energies).float().to(device),
            torch.from_numpy(durations).long().to(device),
        )
    if len(data) == 6:
        ids, raw_texts, speakers, texts, src_lens, max_src_len = data
        return (
            ids, raw_texts,
            torch.from_numpy(speakers).long().to(device),
            torch.from_numpy(texts).long().to(device),
            torch.from_numpy(src_lens).long().to(device),
            max_src_len,
        )
    raise ValueError(f"Unexpected batch of length {len(data)}")


def get_mask_from_lengths(lengths, max_len=None):
    """True where padded."""
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return ids >= lengths.unsqueeze(1).expand(-1, max_len)


def expand(values, durations):
    out = []
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def pad_1D(inputs, PAD=0):
    max_len = max(len(x) for x in inputs)
    return np.stack([
        np.pad(x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD) for x in inputs
    ])


def pad_2D(inputs, maxlen=None):
    max_len = maxlen or max(np.shape(x)[0] for x in inputs)
    return np.stack([
        np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
        for x in inputs
    ])


def pad(input_ele, mel_max_length=None):
    max_len = mel_max_length or max(t.size(0) for t in input_ele)
    out = []
    for t in input_ele:
        if t.dim() == 1:
            out.append(F.pad(t, (0, max_len - t.size(0)), "constant", 0.0))
        else:
            out.append(F.pad(t, (0, 0, 0, max_len - t.size(0)), "constant", 0.0))
    return torch.stack(out)
