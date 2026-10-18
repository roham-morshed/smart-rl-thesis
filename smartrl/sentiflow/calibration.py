"""Temperature scaling and calibration metrics (SPEC §3.6)."""
from __future__ import annotations

import numpy as np
import torch


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """T minimising the NLL of softmax(log p / T) on (p, y), by L-BFGS on log T."""
    logp = torch.tensor(np.log(np.clip(p, 1e-12, 1.0)), dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logp / torch.exp(log_t), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_t).item())


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 15) -> dict:
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                     "acc": float((pred[m] == y[m]).mean()) if m.any() else None,
                     "conf": float(conf[m].mean()) if m.any() else None})
    return {"bins": rows}


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean over sentences of Σ_k (p_k − 1[y = k])²."""
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(axis=1).mean())
