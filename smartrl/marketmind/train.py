"""Training loop shared by MarketMind and the neural comparison models (SPEC §4.5): AdamW lr 1e-3, weight decay
1e-4, batch 128, up to 60 epochs, early stopping (patience 8) on the last 10 % of the training window, gradient
clipping 1.0, fp16 autocast on the GPU. Rows whose 20-day target is unknown never enter training (see data.py)."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from ..sentiflow.train import set_seed
from .data import WindowSet
from .model import quantile_vol_loss

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(n: int, bs: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def _tensors(ws: WindowSet, idx: np.ndarray):
    t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a[idx])).to(DEV, dt, non_blocking=True)
    return t(ws.xp), t(ws.xs), t(ws.mk, torch.long), t(ws.y_ret), t(ws.y_vol)


@torch.no_grad()
def evaluate_loss(model, ws: WindowSet, idx: np.ndarray, cfg: dict) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in _batches(len(idx), 1024, None):
        xp, xs, mk, yr, yv = _tensors(ws, idx[b])
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            pred = model(xp, xs, mk)
        loss = quantile_vol_loss({k: v.float() for k, v in pred.items()}, yr, yv, cfg["vol_loss_weight"])
        tot += float(loss) * len(b)
        n += len(b)
    return tot / max(n, 1)


def fit(model: torch.nn.Module, ws: WindowSet, cfg: dict, seed: int, log=None) -> tuple[torch.nn.Module, dict]:
    """cfg = frozen()['marketmind'] (train block inside)."""
    set_seed(seed)
    tc = cfg["train"]
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    tr_idx, va_idx = np.where(~ws.is_val)[0], np.where(ws.is_val)[0]
    rng = np.random.default_rng(seed)
    best, best_loss, bad, hist = None, math.inf, 0, []
    bs = tc["batch"]
    t0 = time.time()
    for ep in range(tc["max_epochs"]):
        model.train()
        tot = 0.0
        for b in _batches(len(tr_idx), bs, rng):
            xp, xs, mk, yr, yv = _tensors(ws, tr_idx[b])
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                pred = model(xp, xs, mk)
            loss = quantile_vol_loss({k: v.float() for k, v in pred.items()}, yr, yv, cfg["vol_loss_weight"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(b)
        vl = evaluate_loss(model, ws, va_idx, cfg)
        hist.append({"epoch": ep, "train_loss": tot / len(tr_idx), "val_loss": vl})
        if vl < best_loss:
            best_loss, best, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best)
    info = {"epochs": len(hist), "best_epoch": int(np.argmin([h["val_loss"] for h in hist])), "best_val_loss": best_loss,
            "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "seconds": time.time() - t0, "history": hist}
    if log:
        log(f"  fit: {info['epochs']} epochs, best {info['best_epoch']}, val {best_loss:.4f}, {info['seconds']:.0f}s")
    return model.eval(), info


@torch.no_grad()
def predict(model, ws: WindowSet) -> tuple[np.ndarray, np.ndarray]:
    """Scaled outputs: q [N, 3, 3] (h × τ) and logRV [N, 2]."""
    model.eval()
    Q, V = [], []
    for b in _batches(len(ws.mk), 2048, None):
        xp, xs, mk, _, _ = _tensors(ws, b)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            pred = model(xp, xs, mk)
        Q.append(pred["q"].float().cpu().numpy())
        V.append(pred["logrv"].float().cpu().numpy())
    return np.concatenate(Q), np.concatenate(V)
