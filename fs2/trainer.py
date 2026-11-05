"""Training / validation loop (adapted from ming024 train.py + evaluate.py).

Changes vs. reference:
  * One config.yaml per language; checkpoints store the step so `--resume`
    picks up the latest checkpoint automatically (useful on preemptible GPUs).
  * Losses are also written to CSV (train_log.csv / val_log.csv) so the loss
    curves required by the assignment can be plotted without TensorBoard.
  * Best checkpoint by validation mel-postnet loss is kept as best.pth.tar.
  * nn.DataParallel removed (single GPU); TF32 enabled; DataLoader workers.
"""

import csv
import glob
import os
import re
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fs2.dataset import Dataset
from fs2.model import FastSpeech2Loss
from fs2.utils.model import get_model, get_param_num
from fs2.utils.tools import to_device

LOSS_NAMES = ["total_loss", "mel_loss", "postnet_mel_loss", "pitch_loss", "energy_loss", "duration_loss"]


def latest_checkpoint(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
    steps = [(int(m.group(1)), p) for p in ckpts if (m := re.search(r"(\d+)\.pth\.tar$", p))]
    return max(steps)[1] if steps else None


@torch.no_grad()
def evaluate(model, configs, device, loader=None, loss_fn=None):
    preprocess_config, model_config, train_config = configs
    if loader is None:
        ds = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
        loader = DataLoader(ds, batch_size=train_config["optimizer"]["batch_size"], shuffle=False,
                            collate_fn=ds.collate_fn, num_workers=2)
    loss_fn = loss_fn or FastSpeech2Loss(preprocess_config, model_config).to(device)
    was_training = model.training
    model.eval()
    sums, n = [0.0] * 6, 0
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            output = model(*(batch[2:]))
            losses = loss_fn(batch, output)
            bs = len(batch[0])
            for i, l in enumerate(losses):
                sums[i] += l.item() * bs
            n += bs
    if was_training:
        model.train()
    return [s / max(n, 1) for s in sums]


def _append_csv(path, step, losses):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step"] + LOSS_NAMES)
        w.writerow([step] + [f"{l:.6f}" for l in losses])


def train(configs, resume=True, device=None):
    preprocess_config, model_config, train_config = configs
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(train_config.get("seed", 1234))

    paths = train_config["path"]
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = train_config["optimizer"].get("group_size", 4)
    assert batch_size * group_size < len(dataset), "dataset too small for batch_size*group_size"
    loader = DataLoader(dataset, batch_size=batch_size * group_size, shuffle=True,
                        collate_fn=dataset.collate_fn, num_workers=train_config.get("num_workers", 4),
                        persistent_workers=train_config.get("num_workers", 4) > 0)
    val_ds = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_ds.collate_fn,
                            num_workers=2)

    ckpt = latest_checkpoint(paths["ckpt_path"]) if resume else None
    model, optimizer, step = get_model(ckpt, configs, device, train=True)
    if ckpt:
        print(f"Resumed from {ckpt} (step {step})")
    print(f"Train utterances: {len(dataset)}, val utterances: {len(val_ds)}")
    print(f"Number of FastSpeech2 parameters: {get_param_num(model):,}")
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(paths["log_path"])
    except Exception:  # tensorboard optional
        pass

    s = train_config["step"]
    total_step, log_step, val_step, save_step = s["total_step"], s["log_step"], s["val_step"], s["save_step"]
    grad_acc = train_config["optimizer"]["grad_acc_step"]
    clip = train_config["optimizer"]["grad_clip_thresh"]
    train_csv = os.path.join(paths["log_path"], "train_log.csv")
    val_csv = os.path.join(paths["log_path"], "val_log.csv")
    best_path = os.path.join(paths["ckpt_path"], "best.pth.tar")
    best_val = float("inf")
    if os.path.exists(best_path):
        best_val = torch.load(best_path, map_location="cpu", weights_only=False).get("val_postnet_mel_loss", best_val)

    def save(path, extra=None):
        payload = {"model": model.state_dict(), "optimizer": optimizer._optimizer.state_dict(), "step": step}
        payload.update(extra or {})
        torch.save(payload, path)

    bar = tqdm(total=total_step, initial=step, desc="Training")
    running = [0.0] * 6
    n_running = 0
    t0 = time.time()
    model.train()
    while step < total_step:
        for batchs in loader:
            for batch in batchs:
                if step >= total_step:
                    break
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = loss_fn(batch, output)
                (losses[0] / grad_acc).backward()
                running = [r + l.item() for r, l in zip(running, losses)]
                n_running += 1

                if (step + 1) % grad_acc == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                step += 1
                bar.update(1)

                if step % log_step == 0:
                    avg = [r / n_running for r in running]
                    _append_csv(train_csv, step, avg)
                    if writer:
                        for name, v in zip(LOSS_NAMES, avg):
                            writer.add_scalar(f"train/{name}", v, step)
                    lr = optimizer._optimizer.param_groups[0]["lr"]
                    bar.write(f"step {step} | " + " ".join(f"{n[:-5]}={v:.4f}" for n, v in zip(LOSS_NAMES, avg))
                              + f" | lr={lr:.2e} | {log_step / (time.time() - t0):.1f} it/s")
                    running, n_running, t0 = [0.0] * 6, 0, time.time()

                if step % val_step == 0 and len(val_ds) > 0:
                    val = evaluate(model, configs, device, val_loader, loss_fn)
                    _append_csv(val_csv, step, val)
                    if writer:
                        for name, v in zip(LOSS_NAMES, val):
                            writer.add_scalar(f"val/{name}", v, step)
                    bar.write(f"[val] step {step} | " + " ".join(f"{n[:-5]}={v:.4f}" for n, v in zip(LOSS_NAMES, val)))
                    if val[2] < best_val:
                        best_val = val[2]
                        save(best_path, {"val_postnet_mel_loss": best_val})
                        bar.write(f"[val] new best postnet mel loss {best_val:.4f} -> best.pth.tar")

                if step % save_step == 0 or step == total_step:
                    save(os.path.join(paths["ckpt_path"], f"{step}.pth.tar"))
    bar.close()
    print("Training finished.")
