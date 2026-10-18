"""Sentiment Trading Module (SPEC §7) and the FinBERT baseline B5 (STM-1 on Branch-B sentiment).

All statistics (μ, σ, logistic fit, in-sample percentiles) come from the fold's training rows; the exposure of day
t uses only the daily features of day t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

STM2_FEATURES = ["Se_loc", "Se_glob", "disp_loc", "cov", "log_n_loc"]


def blended_sentiment(sent: pd.DataFrame, source: str = "SF") -> np.ndarray:
    """S̃_t = cov_t · Se_loc + (1 − cov_t) · Se_glob (Branch-B columns when source = 'B')."""
    sfx = "" if source == "SF" else "_B"
    cov = sent["cov"].to_numpy(float)
    return cov * sent[f"Se_loc{sfx}"].to_numpy(float) + (1 - cov) * sent[f"Se_glob{sfx}"].to_numpy(float)


def stm1(sent: pd.DataFrame, train: slice, kappa: float, z0: float, source: str = "SF") -> np.ndarray:
    """e_SA = 1 − clip(κ · max(0, −z_t − z0), 0, 1), z_t = (S̃_t − μ)/σ with μ, σ from the training rows."""
    s = blended_sentiment(sent, source)
    mu, sd = np.nanmean(s[train]), np.nanstd(s[train], ddof=1)
    z = (s - mu) / (sd if sd > 0 else 1.0)
    return 1.0 - np.clip(kappa * np.maximum(0.0, -z - z0), 0.0, 1.0)


def stm2(sent: pd.DataFrame, X_scaled: pd.DataFrame, ret_tr: np.ndarray, train: slice, C: float = 1.0,
         horizon: int = 5, pct: tuple[int, int] = (20, 80)) -> np.ndarray:
    """Logistic regression (L2, C = 1) on the fold-scaled [Se_loc, Se_glob, disp_loc, cov, log1p(n_loc)] for
    P(r_{t→t+h} > 0); e_SA = clip((p − p20)/(p80 − p20), 0, 1) with in-sample percentiles. Labels are formed only
    where the whole h-day window lies inside the training rows."""
    T = len(ret_tr)
    lr = np.log1p(np.nan_to_num(ret_tr))
    cs = np.concatenate([[0.0], np.cumsum(lr)])
    fwd = np.full(T, np.nan)
    fwd[:T - horizon] = cs[horizon + 1:T + 1] - cs[1:T - horizon + 1]          # Σ_{i=1..h} log(1 + R^tr_{t+i})
    tr_idx = np.arange(train.start, train.stop - horizon)
    X = X_scaled[STM2_FEATURES].to_numpy(float)
    model = LogisticRegression(C=C, penalty="l2", max_iter=1000)
    model.fit(X[tr_idx], (fwd[tr_idx] > 0).astype(int))
    p = model.predict_proba(X)[:, 1]
    p20, p80 = np.percentile(p[tr_idx], pct)
    return np.clip((p - p20) / max(p80 - p20, 1e-9), 0.0, 1.0)


def confidence(sent: pd.DataFrame) -> np.ndarray:
    """C_SA = c̄_t · cov_t (0 when the market has no news that day)."""
    has = (sent["n_loc"].fillna(0) + sent["n_glob"].fillna(0)) > 0
    return np.where(has, sent["cbar"].fillna(0).to_numpy(float) * sent["cov"].to_numpy(float), 0.0)
