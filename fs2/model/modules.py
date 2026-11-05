"""Variance adaptor (duration / pitch / energy predictors + length regulator).

Adapted from ming024/FastSpeech2 model/modules.py. Changes vs. reference:
  * BUGFIX: energy predictor used the *pitch* control factor (p_control) instead
    of e_control, so energy control at inference was silently ignored.
  * Device is taken from input tensors instead of a module-level global, so the
    model runs on CPU/GPU without edits.
  * Length regulator is vectorised with repeat_interleave (no python loop over
    phonemes) - same output, noticeably faster training.
"""

import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from fs2.utils.tools import get_mask_from_lengths


class VarianceAdaptor(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super().__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
        pitch_min, pitch_max = stats["pitch"][:2]
        energy_min, energy_max = stats["energy"][:2]

        def bins(lo, hi, mode):
            if mode == "log":
                return torch.exp(torch.linspace(np.log(lo), np.log(hi), n_bins - 1))
            return torch.linspace(lo, hi, n_bins - 1)

        self.pitch_bins = nn.Parameter(bins(pitch_min, pitch_max, pitch_quantization), requires_grad=False)
        self.energy_bins = nn.Parameter(bins(energy_min, energy_max, energy_quantization), requires_grad=False)

        hidden = model_config["transformer"]["encoder_hidden"]
        self.pitch_embedding = nn.Embedding(n_bins, hidden)
        self.energy_embedding = nn.Embedding(n_bins, hidden)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding

    def forward(self, x, src_mask, mel_mask=None, max_len=None, pitch_target=None,
                energy_target=None, duration_target=None, p_control=1.0, e_control=1.0,
                d_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction = energy_prediction = None

        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(log_duration_prediction) - 1) * d_control, min=0
            )
            duration_rounded = duration_rounded.masked_fill(src_mask, 0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)
            x = x + energy_embedding

        return (x, pitch_prediction, energy_prediction, log_duration_prediction,
                duration_rounded, mel_len, mel_mask)


class LengthRegulator(nn.Module):
    """Repeat each phoneme hidden vector `duration` times and pad to max_len."""

    def forward(self, x, duration, max_len=None):
        duration = duration.long().clamp(min=0)
        mel_len = duration.sum(dim=1)
        out_len = int(max_len) if max_len is not None else int(mel_len.max().item())
        out_len = max(out_len, 1)
        out = x.new_zeros(x.size(0), out_len, x.size(2))
        for b in range(x.size(0)):
            expanded = torch.repeat_interleave(x[b], duration[b], dim=0)[:out_len]
            out[b, : expanded.size(0)] = expanded
        return out, mel_len


class VariancePredictor(nn.Module):
    """Two 1-D conv layers + linear projection (duration, pitch and energy)."""

    def __init__(self, model_config):
        super().__init__()
        input_size = model_config["transformer"]["encoder_hidden"]
        filter_size = model_config["variance_predictor"]["filter_size"]
        kernel = model_config["variance_predictor"]["kernel_size"]
        dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(input_size, filter_size, kernel_size=kernel, padding=(kernel - 1) // 2)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(filter_size)),
            ("dropout_1", nn.Dropout(dropout)),
            ("conv1d_2", Conv(filter_size, filter_size, kernel_size=kernel, padding=(kernel - 1) // 2)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(filter_size)),
            ("dropout_2", nn.Dropout(dropout)),
        ]))
        self.linear_layer = nn.Linear(filter_size, 1)

    def forward(self, encoder_output, mask):
        out = self.linear_layer(self.conv_layer(encoder_output)).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        return self.conv(x.contiguous().transpose(1, 2)).contiguous().transpose(1, 2)
