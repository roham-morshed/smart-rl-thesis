"""Rows, splits and scaling of the base paper's direction task (SPEC §15) and of the SENTIN refits (§9.1, §4.7).

Inputs per row: open, high, low, close, volume (0 after scaling where volume is disabled or missing) and a daily
sentiment index S (last column). Label y_t = 1[C_{t+1} > C_t] on the row sequence.
- PUB_R / PUB_C: calendar-day rows, non-trading days forward-filled, news by UTC calendar date, scaling on all rows.
- CAUSAL: trading days, news by session (SPEC §2.3), scaling on the training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import frozen
from ..data import news, panel
from ..sentiflow import daily, scoring

INPUTS = ["open", "high", "low", "close", "volume"]


def _utc_day_index(market: str, kind: str) -> pd.Series:
    """Daily index by UTC calendar date over the market's local headlines: 'A' = mean uncalibrated Branch-A
    polarity (SENTIN), 'SF' = confidence-weighted calibrated SentiFlow polarity (SENTIN_SF)."""
    local = frozen()["markets"][market]["news_tag"]
    items = pd.read_parquet(news.ITEMS, columns=["id", "ts_utc", f"tag_{local}"])
    items = items[items[f"tag_{local}"]]
    sc = pd.read_parquet(scoring.SCORES, columns=["id", "s", "c", "a_s"]).set_index("id")
    j = items.join(sc, on="id")
    j["day"] = pd.to_datetime(j["ts_utc"], utc=True).dt.tz_convert(None).dt.normalize()
    if kind == "A":
        return j.groupby("day")["a_s"].mean()
    j["cs"] = j["c"] * j["s"]
    g = j.groupby("day")[["cs", "c"]].sum()
    return g["cs"] / g["c"]


def rows(market: str, protocol: str, kind: str, start: str, end: str) -> pd.DataFrame:
    """DataFrame indexed by date with INPUTS, S, y (label of the row; NaN on the last row)."""
    px = pd.read_parquet(panel.processed_path(market)).set_index("date")
    px = px.loc[start:end, INPUTS].copy()
    if market in frozen()["features"].get("volume_disabled", []):
        px["volume"] = np.nan
    px.loc[px["volume"] <= 0, "volume"] = np.nan
    if protocol == "CAUSAL":
        d = pd.read_parquet(daily.daily_path(market)).set_index("date") if daily.daily_path(market).exists() else None
        col = "sentin_S" if kind == "A" else "S_loc"
        px["S"] = d[col].reindex(px.index).fillna(0.0) if d is not None else 0.0
    else:
        cal = pd.date_range(px.index.min(), px.index.max(), freq="D")
        px = px.reindex(cal).ffill()
        s = _utc_day_index(market, kind) if scoring.SCORES.exists() else pd.Series(dtype=float)
        px["S"] = s.reindex(cal).fillna(0.0).to_numpy()
    c = px["close"].to_numpy()
    y = np.full(len(c), np.nan)
    y[:-1] = (c[1:] > c[:-1]).astype(float)
    px["y"] = y
    px.index.name = "date"
    return px


def split(n_rows: int, lookback: int, protocol: str, cfg: dict) -> dict[str, np.ndarray]:
    """Sample t = window ending at row t (t >= lookback−1, label known). Returns index arrays for train/val/test."""
    s = cfg["e13"]["split"]
    idx = np.arange(lookback - 1, n_rows - 1)
    if protocol == "PUB_R":
        rng = np.random.default_rng(s["random_seed"])
        perm = rng.permutation(idx)
        n_te = int(round(s["test"] * len(perm)))
        te, tr_all = np.sort(perm[:n_te]), perm[n_te:]
        n_va = int(round(s["early_stop_share_of_train"] * len(tr_all)))
        return {"train": np.sort(tr_all[n_va:]), "val": np.sort(tr_all[:n_va]), "test": te}
    n_tr_all = int(round(s["train"] * len(idx)))
    tr_all, te = idx[:n_tr_all], idx[n_tr_all:]
    n_va = int(round(s["early_stop_share_of_train"] * len(tr_all)))
    return {"train": tr_all[:-n_va], "val": tr_all[-n_va:], "test": te}


def scaled_matrix(r: pd.DataFrame, fit_rows: np.ndarray) -> np.ndarray:
    """q01/q99 min-max with the clip of §2.2, fitted on `fit_rows`; masked inputs become 0."""
    cols = INPUTS + ["S"]
    params = panel.fit_minmax(r.iloc[fit_rows], cols)
    return panel.apply_minmax(r[cols], params, fill=0.0).to_numpy(np.float32)


def windows(X: np.ndarray, idx: np.ndarray, lookback: int) -> np.ndarray:
    w = idx[:, None] + np.arange(-lookback + 1, 1)[None, :]
    return X[w]
