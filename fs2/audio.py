"""Mel-spectrogram / energy extraction.

Re-implementation of the reference TacotronSTFT (audio/stft.py in ming024) with
torch.stft. The reference hard-codes `.cuda()` inside the STFT and uses
positional librosa calls that were removed in librosa >= 0.10. The maths is
unchanged so mels stay compatible with the pretrained HiFi-GAN vocoder:
  - Hann window, reflect padding (center=True)
  - magnitude spectrogram -> Slaney mel filterbank (librosa default)
  - log(clamp(mel, 1e-5))
  - energy = L2 norm of the magnitude spectrum per frame
"""

import librosa
import numpy as np
import torch


class MelExtractor:
    def __init__(self, sampling_rate, filter_length, hop_length, win_length,
                 n_mel_channels, mel_fmin, mel_fmax):
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()
        self.window = torch.hann_window(win_length)

    @classmethod
    def from_config(cls, preprocess_config):
        p = preprocess_config["preprocessing"]
        return cls(
            p["audio"]["sampling_rate"], p["stft"]["filter_length"], p["stft"]["hop_length"],
            p["stft"]["win_length"], p["mel"]["n_mel_channels"], p["mel"]["mel_fmin"],
            p["mel"]["mel_fmax"],
        )

    @torch.no_grad()
    def __call__(self, wav):
        """wav: float np.ndarray in [-1, 1]. Returns (mel [n_mels, T], energy [T])."""
        y = torch.clamp(torch.from_numpy(np.asarray(wav, dtype=np.float32)), -1, 1)
        spec = torch.stft(
            y, n_fft=self.filter_length, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, center=True,
            pad_mode="reflect", return_complex=True,
        )
        magnitude = spec.abs()
        mel = torch.matmul(self.mel_basis, magnitude)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        energy = torch.norm(magnitude, dim=0)
        return mel.numpy().astype(np.float32), energy.numpy().astype(np.float32)
