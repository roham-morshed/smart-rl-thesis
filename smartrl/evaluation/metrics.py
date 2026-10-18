"""Trading metrics of SPEC §11 and the COMBINED equal-weight portfolio."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..env.portfolio import RF_DAILY

METRICS = ["cum_return", "ann_return", "ann_vol", "sharpe", "sortino", "mdd", "calmar", "turnover", "win_rate"]


def max_drawdown(r: np.ndarray) -> float:
    w = np.cumprod(1.0 + np.asarray(r, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return float((w / peak - 1.0).min()) if len(w) else float("nan")


def sharpe(r: np.ndarray, rf: float = RF_DAILY) -> float:
    x = np.asarray(r, float) - rf
    sd = x.std(ddof=1) if len(x) > 1 else float("nan")
    return float(x.mean() / sd * math.sqrt(252)) if sd and sd > 0 else float("nan")


def trading_metrics(r: np.ndarray, held: np.ndarray | None = None, e0: float = 1.0, rf: float = RF_DAILY) -> dict:
    """r: daily net returns; held: exposure held during each return day (e_{t-1} for the return of day t).
    turnover = mean |Δe| per day (starting from e0); win rate = share of invested days with R > r_f."""
    r = np.asarray(r, float)
    T = len(r)
    if T == 0:
        return {k: float("nan") for k in METRICS}
    growth = float(np.prod(1.0 + r))
    ann = growth ** (252.0 / T) - 1.0
    x = r - rf
    down = math.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    mdd = max_drawdown(r)
    out = {"cum_return": growth - 1.0, "ann_return": ann, "ann_vol": float(r.std(ddof=1) * math.sqrt(252)),
           "sharpe": sharpe(r, rf), "sortino": float(x.mean() / down * math.sqrt(252)) if down > 0 else float("nan"),
           "mdd": mdd, "calmar": float(ann / abs(mdd)) if mdd < 0 else float("nan"),
           "turnover": float("nan"), "win_rate": float("nan")}
    if held is not None:
        h = np.asarray(held, float)
        out["turnover"] = float(np.mean(np.abs(np.diff(np.concatenate([[e0], h])))))
        inv = h > 0
        out["win_rate"] = float((r[inv] > rf).mean()) if inv.any() else float("nan")
    return out


def combined(sleeves: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight portfolio of the market sleeves on the union calendar; a sleeve earns 0 on its market's
    holidays; rebalanced daily to 1/4 with no extra cost (SPEC §11)."""
    df = pd.concat(sleeves, axis=1).sort_index()
    return df.fillna(0.0).mean(axis=1)


def regime_labels(index_close: pd.Series, lookback: int = 126, threshold: float = 0.05) -> pd.Series:
    """bull / bear / sideways from the trailing 126-day index return (causal)."""
    tr = index_close / index_close.shift(lookback) - 1.0
    lab = pd.Series("sideways", index=index_close.index)
    lab[tr > threshold] = "bull"
    lab[tr < -threshold] = "bear"
    lab[tr.isna()] = np.nan
    return lab
