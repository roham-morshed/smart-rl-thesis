"""Training of the E13 direction models: batch 256, up to 100 epochs, early stopping on validation AUC
(patience 10); optimiser and loss = SENTIN's E13 choice (SGD lr 1e-3 + MSE on the sigmoid output, as published,
or Adam lr 1e-3 + binary cross-entropy). CNN, LSTM and GRU use the same recipe, so every neural model gets the
setting chosen for the base model."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ..sentiflow.train import set_seed

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (score > threshold).astype(int)
    return {"acc": float((pred == y).mean()), "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "auc": auc(y, score), "prauc": float(average_precision_score(y, score))}


def fit(model, X_tr, y_tr, X_va, y_va, choice: str, seed: int, cfg: dict) -> tuple[torch.nn.Module, dict]:
    set_seed(seed)
    tc = cfg["sentin"]["train"]
    model = model.to(DEV)
    opt = (torch.optim.SGD(model.parameters(), lr=1e-3) if choice == "sgd_mse"
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    Xt, yt = torch.from_numpy(X_tr).to(DEV), torch.from_numpy(y_tr.astype(np.float32)).to(DEV)
    rng = np.random.default_rng(seed)
    best, best_auc, bad, hist = None, -np.inf, 0, []
    for ep in range(tc["max_epochs"]):
        model.train()
        order = torch.from_numpy(rng.permutation(len(yt))).to(DEV)
        for i in range(0, len(yt), tc["batch"]):
            sel = order[i:i + tc["batch"]]
            logit = model(Xt[sel])
            loss = (F.mse_loss(torch.sigmoid(logit), yt[sel]) if choice == "sgd_mse"
                    else F.binary_cross_entropy_with_logits(logit, yt[sel]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        a = auc(y_va, predict(model, X_va))
        hist.append(a)
        if np.isfinite(a) and a > best_auc:
            best_auc, best, bad = a, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    if best is not None:
        model.load_state_dict(best)
    return model.eval(), {"epochs": len(hist), "best_val_auc": best_auc, "best_epoch": int(np.nanargmax(hist)) if hist else -1}


@torch.no_grad()
def predict(model, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(DEV))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
