"""Non-neural references of SPEC §4.6: NAIVE quantiles, EWMA (RiskMetrics, λ = 0.94) and HAR-RV (Corsi 2009).

Only daily closes are available, so the daily realised variance in HAR is r1²; the HAR regression is fitted by
OLS on the training window, per market and horizon, with the target RV_h = mean of r1² over the next h days and
regressors [1, r_t², mean of r² over the last 5 days, over the last 22 days].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketPanel, targets
from .model import HORIZONS, VOL_HORIZONS


def naive_quantiles(mp: MarketPanel, train_end: pd.Timestamp, idx: np.ndarray) -> np.ndarray:
    """q50 = 0, q10/q90 = empirical 10/90 % quantiles of r_h over the training window -> [len(idx), 3, 3]."""
    y_ret, _ = targets(mp)
    last = int(np.where(mp.dates <= train_end)[0].max())
    out = np.zeros((len(idx), len(HORIZONS), 3))
    for k, h in enumerate(HORIZONS):
        r = y_ret[: last - h + 1, k]
        r = r[np.isfinite(r)]
        out[:, k, 0], out[:, k, 2] = np.quantile(r, 0.10), np.quantile(r, 0.90)
    return out


def ewma_var(r1: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """σ²_{t+1|t} after observing r_t (seeded with the variance of the first 20 returns)."""
    r = np.nan_to_num(r1, nan=0.0)
    out = np.empty(len(r))
    v = np.var(r[1:21]) if len(r) > 21 else np.var(r)
    for t in range(len(r)):
        v = lam * v + (1 - lam) * r[t] ** 2
        out[t] = v
    return out


def _har_x(r1: np.ndarray) -> np.ndarray:
    r2 = pd.Series(np.nan_to_num(r1, nan=0.0) ** 2)
    return np.column_stack([np.ones(len(r2)), r2, r2.rolling(5).mean(), r2.rolling(22).mean()])


def har_forecast(mp: MarketPanel, train_end: pd.Timestamp, idx: np.ndarray) -> np.ndarray:
    """Forecast of RV_h for h in (5, 20) at rows idx -> [len(idx), 2] (floored at a tiny positive value)."""
    _, y_rv = targets(mp)
    X = _har_x(mp.r1)
    last = int(np.where(mp.dates <= train_end)[0].max())
    out = np.zeros((len(idx), len(VOL_HORIZONS)))
    for j, h in enumerate(VOL_HORIZONS):
        rows = np.arange(22, last - h + 1)
        ok = np.isfinite(X[rows]).all(1) & np.isfinite(y_rv[rows, j])
        beta, *_ = np.linalg.lstsq(X[rows][ok], y_rv[rows, j][ok], rcond=None)
        out[:, j] = np.maximum(X[idx] @ beta, 1e-10)
    return out
