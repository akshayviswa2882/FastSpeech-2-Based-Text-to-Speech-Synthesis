"""Variance-adaptor and loss-curve plots."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

C_DUR, C_PITCH, C_ENERGY = "#3b6ea8", "#c0504d", "#7a5195"


def _word_boundaries(ax, spans, boundaries_t, ymax, label_y):
    for word, s, e in spans:
        x0, x1 = boundaries_t[s], boundaries_t[e]
        ax.axvline(x0, color="0.75", lw=0.6, ls="--", zorder=0)
        ax.text((x0 + x1) / 2, label_y, word, ha="center", va="bottom", fontsize=7, color="0.25",
                rotation=0 if len(word) < 12 else 20, clip_on=False)
    if spans:
        ax.axvline(boundaries_t[spans[-1][2]], color="0.75", lw=0.6, ls="--", zorder=0)


def plot_duration(phones, dur_frames, spans, frame_ms, title, path):
    n = len(phones)
    ms = np.asarray(dur_frames, dtype=float) * frame_ms
    fig, ax = plt.subplots(figsize=(max(10, 0.2 * n + 2), 3.6))
    colors = ["0.8" if p in ("sp", "sil") else C_DUR for p in phones]
    ax.bar(np.arange(n), ms, color=colors, width=0.8)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("predicted duration (ms)")
    ymax = max(ms.max() * 1.25, 50)
    ax.set_ylim(0, ymax)
    edges = np.arange(n + 1) - 0.5
    _word_boundaries(ax, spans, edges, ymax, ymax * 0.9)
    ax.set_title(f"{title}   (total {ms.sum() / 1000:.2f} s, {n} phones)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contour(values, phones, dur_frames, spans, mel, frame_ms, ylabel, color, title, path):
    """Phoneme-level prediction drawn as a step contour over predicted time, above the mel."""
    dur = np.asarray(dur_frames, dtype=float)
    bounds = np.concatenate([[0], np.cumsum(dur)]) * frame_ms / 1000.0
    n = len(phones)
    fig, (ax_m, ax) = plt.subplots(2, 1, figsize=(max(10, 0.2 * n + 2), 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    T = mel.shape[1]
    ax_m.imshow(mel, origin="lower", aspect="auto", extent=[0, T * frame_ms / 1000, 0, mel.shape[0]],
                cmap="magma")
    ax_m.set_ylabel("mel bin")
    ax_m.set_title(title, loc="left")

    vals = np.asarray(values, dtype=float)
    for i in range(n):
        if dur[i] <= 0:
            continue
        silent = phones[i] in ("sp", "sil")
        ax.hlines(vals[i], bounds[i], bounds[i + 1], color="0.6" if silent else color, lw=2.2)
        if i + 1 < n and dur[i + 1] > 0:
            ax.vlines(bounds[i + 1], vals[i], vals[i + 1], color=color, lw=0.6, alpha=0.5)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    pad = (hi - lo) * 0.25 + 1e-6
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (s)")
    mids = (bounds[:-1] + bounds[1:]) / 2
    for i, p in enumerate(phones):
        if dur[i] > 0:
            ax.text(mids[i], lo - pad * 0.9, p, ha="center", va="bottom", fontsize=6, rotation=90, color="0.35")
    _word_boundaries(ax, spans, bounds, hi + pad * 1.6, hi + pad * 0.6)
    ax.set_xlim(0, bounds[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None


def plot_losses(csv_path, title, path, smooth=1):
    d = _read_csv(csv_path)
    if d is None:
        return False
    names = ["total_loss", "mel_loss", "postnet_mel_loss", "pitch_loss", "energy_loss", "duration_loss"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, name in zip(axes.flat, names):
        y = d[name]
        if smooth > 1 and len(y) > smooth:
            k = np.ones(smooth) / smooth
            ys = np.convolve(y, k, mode="valid")
            ax.plot(d["step"], y, color="0.8", lw=0.8)
            ax.plot(d["step"][smooth - 1:], ys, color=C_DUR, lw=1.4)
        else:
            ax.plot(d["step"], y, color=C_DUR, lw=1.4, marker="o" if len(y) < 40 else None, ms=3)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_train_vs_val(train_csv, val_csv, title, path):
    t, v = _read_csv(train_csv), _read_csv(val_csv)
    if t is None or v is None:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name in zip(axes, ["postnet_mel_loss", "duration_loss", "pitch_loss"]):
        ax.plot(t["step"], t[name], color=C_DUR, lw=1, alpha=0.8, label="train")
        ax.plot(v["step"], v[name], color=C_PITCH, lw=1.6, label="val")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
