"""Config loading, model / vocoder construction (adapted from ming024 utils/model.py)."""

import json
import os
import zipfile

import numpy as np
import torch
import yaml

from fs2 import hifigan
from fs2.model import FastSpeech2, ScheduledOptim

HIFIGAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hifigan")


def load_config(config_path):
    """Load a single per-language config.yaml and return (preprocess, model, train) dicts.

    Relative paths inside the config are resolved against the config file's folder,
    so scripts can be launched from anywhere.
    """
    config_path = os.path.abspath(config_path)
    base = os.path.dirname(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def resolve(d):
        for k, v in d.items():
            if isinstance(v, str) and not os.path.isabs(v):
                d[k] = os.path.normpath(os.path.join(base, v))

    resolve(cfg["preprocess"]["path"])
    al = cfg["preprocess"].get("alignment", {})
    if al.get("mfa_config") and not os.path.isabs(al["mfa_config"]):
        al["mfa_config"] = os.path.normpath(os.path.join(base, al["mfa_config"]))
    resolve(cfg["train"]["path"])
    language = cfg["language"]
    for section in ("preprocess", "model", "train"):
        cfg[section]["language"] = language
    return cfg["preprocess"], cfg["model"], cfg["train"], cfg


def get_model(ckpt_path, configs, device, train=False):
    preprocess_config, model_config, train_config = configs
    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = None
    restore_step = 0
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        restore_step = ckpt.get("step", 0)

    if train:
        optim = ScheduledOptim(model, train_config, model_config, restore_step)
        if ckpt is not None and "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, optim, restore_step

    model.eval()
    model.requires_grad_(False)
    return model


def get_param_num(model):
    return sum(p.numel() for p in model.parameters())


def get_vocoder(model_config, device):
    speaker = model_config["vocoder"]["speaker"]
    with open(os.path.join(HIFIGAN_DIR, "config.json")) as f:
        config = hifigan.AttrDict(json.load(f))
    vocoder = hifigan.Generator(config)
    ckpt_file = os.path.join(HIFIGAN_DIR, f"generator_{speaker}.pth.tar")
    if not os.path.exists(ckpt_file):
        with zipfile.ZipFile(ckpt_file + ".zip") as z:
            z.extractall(HIFIGAN_DIR)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder.to(device)


@torch.no_grad()
def vocoder_infer(mels, vocoder, preprocess_config, lengths=None):
    """mels: [B, n_mels, T] -> list of int16 numpy waveforms."""
    wavs = vocoder(mels).squeeze(1)
    wavs = (wavs.cpu().numpy() * preprocess_config["preprocessing"]["audio"]["max_wav_value"]).astype("int16")
    wavs = [w for w in wavs]
    if lengths is not None:
        wavs = [w[: int(l)] for w, l in zip(wavs, lengths)]
    return wavs
