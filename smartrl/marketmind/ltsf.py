"""E3 (SPEC §15): the MarketMind backbone on the long-term forecasting benchmark ETTh1 / ETTm1.

Backbone = channel-independent patches + ProbSparse encoder with distilling (factor 5), without the sentiment stream
and without the market embedding, with a PatchTST-style flatten head. Standard splits of Zhou et al. (2021):
12/4/4 months (ETTm: 4 rows per hour); data standardised with the training-split mean and standard deviation;
MSE and MAE on the standardised scale over all channels and every test window (stride 1).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import RAW
from ..sentiflow.train import set_seed
from .informer import encoder_out_len, informer_encoder
from .model import make_patches, n_patches

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ett(name: str, seq_len: int) -> dict[str, np.ndarray]:
    df = pd.read_csv(RAW / "ett" / f"{name}.csv")
    x = df.drop(columns=["date"]).to_numpy(np.float32)
    per = 4 if name.startswith("ETTm") else 1
    m = 30 * 24 * per
    b1 = [0, 12 * m - seq_len, 12 * m + 4 * m - seq_len]
    b2 = [12 * m, 12 * m + 4 * m, 12 * m + 8 * m]
    mu, sd = x[b1[0]:b2[0]].mean(0), x[b1[0]:b2[0]].std(0)
    z = (x - mu) / sd
    return {s: z[a:b] for s, a, b in zip(("train", "val", "test"), b1, b2)}


class Windows(torch.utils.data.Dataset):
    def __init__(self, z: np.ndarray, L: int, H: int):
        self.z, self.L, self.H = torch.from_numpy(z), L, H

    def __len__(self):
        return len(self.z) - self.L - self.H + 1

    def __getitem__(self, i):
        return self.z[i:i + self.L], self.z[i + self.L:i + self.L + self.H]


class Backbone(nn.Module):
    def __init__(self, n_ch: int, L: int, H: int, patch: int, stride: int, d_model: int, heads: int, d_ff: int,
                 layers: int, dropout: float, factor: int = 5):
        super().__init__()
        self.patch, self.stride, self.n_ch = patch, stride, n_ch
        self.N = n_patches(L, patch, stride)
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(self.N, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.drop_in = nn.Dropout(dropout)
        self.encoder = informer_encoder(d_model, heads, d_ff, layers, factor, dropout, distil=True, prob=True)
        # PatchTST flatten head (its ETT configuration uses no head dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(encoder_out_len(self.N, layers, True) * d_model, H))

    def forward(self, x):                                   # x [B, L, C] -> [B, H, C]
        B = x.shape[0]
        e = self.drop_in(self.embed(make_patches(x, self.patch, self.stride)) + self.pos)    # [B, C, N, d]
        z = self.encoder(e.reshape(B * self.n_ch, self.N, -1))
        return self.head(z).reshape(B, self.n_ch, -1).transpose(1, 2)


def _loader(ds, bs, shuffle, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g, num_workers=0, pin_memory=True)


@torch.no_grad()
def _evaluate(model, dl) -> tuple[float, float]:
    model.eval()
    se = ae = n = 0.0
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            p = model(x)
        d = p.float() - y
        se += float((d ** 2).sum())
        ae += float(d.abs().sum())
        n += d.numel()
    return se / n, ae / n


def run(name: str, H: int, seed: int, cfg: dict, max_epochs: int | None = None, log=print) -> dict:
    L = cfg["lookback"]
    data = load_ett(name, L)
    set_seed(seed)
    model = Backbone(data["train"].shape[1], L, H, cfg["patch"], cfg["stride"], cfg["d_model"], cfg["heads"],
                     cfg["d_ff"], cfg["layers"], cfg["dropout"]).to(DEV)
    tc = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    tr = _loader(Windows(data["train"], L, H), tc["batch"], True, seed)
    va = _loader(Windows(data["val"], L, H), 512, False)
    te = _loader(Windows(data["test"], L, H), 512, False)
    best, best_v, bad, hist, t0 = None, math.inf, 0, [], time.time()
    for ep in range(max_epochs or tc["max_epochs"]):
        model.train()
        for x, y in tr:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                loss = nn.functional.mse_loss(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        v_mse, _ = _evaluate(model, va)
        hist.append(v_mse)
        if v_mse < best_v:
            best_v, best, bad = v_mse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best)
    mse, mae = _evaluate(model, te)
    out = {"dataset": name, "horizon": H, "seed": seed, "mse": mse, "mae": mae, "val_mse": best_v,
           "epochs": len(hist), "best_epoch": int(np.argmin(hist)), "seconds": time.time() - t0}
    log(f"{name} H{H} seed {seed}: test MSE {mse:.4f} MAE {mae:.4f} ({out['epochs']} epochs, {out['seconds']:.0f}s)")
    return out
