"""MarketMind inputs and targets (SPEC §4.1, §4.4, §4.7).

For a refit whose training window ends at `train_end`:
- per market, the 13 price features (§2.2) and the 6 sentiment features (§3.9) are min-max scaled with the 1 % / 99 %
  quantiles of that market's rows dated <= train_end (cov uses the median daily n_loc of the same rows);
- a training sample at day t needs 96 days of valid inputs and its 20-day target window inside the training window
  (index t + 20 <= last training day), so no target reaches into the predicted year;
- early stopping uses the samples on the last 10 % of the training dates;
- return targets are divided by the training standard deviation of r_h of each market; volatility targets are
  log RV_h of r1 / sd(r1) (the same scaling written in logs). Predictions are converted back to raw units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import MARKETS, frozen
from ..data import panel as P
from ..sentiflow import daily as D
from .model import HORIZONS, VOL_HORIZONS

SENT_BASE = ["Se_loc", "Se_glob", "log_n_loc", "log_n_glob", "disp_loc"]   # + cov, computed per window


@dataclass
class MarketPanel:
    market: str
    mid: int
    dates: pd.DatetimeIndex
    price: np.ndarray          # [T, 13] raw features (NaN = not available / masked)
    sent: np.ndarray           # [T, 5] raw sentiment features (without cov)
    n_loc: np.ndarray          # [T]
    r1: np.ndarray             # [T] log returns of the index
    close: np.ndarray          # [T]
    has_sentiment: bool


def load_panels(markets=MARKETS) -> dict[str, MarketPanel]:
    out = {}
    for mid, m in enumerate(markets):
        px = pd.read_parquet(P.processed_path(m))
        dates = pd.DatetimeIndex(px["date"])
        sp = D.daily_path(m)
        if sp.exists():
            s = pd.read_parquet(sp).set_index("date").reindex(dates)
            sent, n_loc, has = s[SENT_BASE].to_numpy(float), s["n_loc"].fillna(0).to_numpy(float), True
        else:
            sent, n_loc, has = np.zeros((len(dates), len(SENT_BASE))), np.zeros(len(dates)), False
        out[m] = MarketPanel(m, mid, dates, px[P.PRICE_FEATURES].to_numpy(float), sent, n_loc,
                             px["r1"].to_numpy(float), px["close"].to_numpy(float), has)
    return out


def targets(mp: MarketPanel) -> tuple[np.ndarray, np.ndarray]:
    """y_ret[t, k] = log(C_{t+h_k}/C_t); y_rv[t, j] = mean of r1^2 over t+1..t+h_j (NaN beyond the data)."""
    T = len(mp.close)
    lc = np.log(mp.close)
    y_ret = np.full((T, len(HORIZONS)), np.nan)
    for k, h in enumerate(HORIZONS):
        y_ret[:T - h, k] = lc[h:] - lc[:T - h]
    r2 = np.nan_to_num(mp.r1, nan=0.0) ** 2
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    y_rv = np.full((T, len(VOL_HORIZONS)), np.nan)
    for j, h in enumerate(VOL_HORIZONS):
        idx = np.arange(T - h)
        y_rv[idx, j] = (cs[idx + h + 1] - cs[idx + 1]) / h
    return y_ret, y_rv


@dataclass
class WindowSet:
    xp: np.ndarray        # [N, L, Cp] float32
    xs: np.ndarray        # [N, L, Cs] float32
    mk: np.ndarray        # [N] market id
    y_ret: np.ndarray     # [N, 3] scaled (NaN when unknown)
    y_vol: np.ndarray     # [N, 2] scaled
    market: np.ndarray    # [N] market key
    date: np.ndarray      # [N] datetime64
    is_val: np.ndarray    # [N] bool (early-stopping rows), only for training sets


@dataclass
class Scaling:
    price: dict
    sent: dict
    cov_median: float
    sd_ret: np.ndarray    # [3]
    sd_r1: float


def _first_valid(price: np.ndarray, use_cols: list[int]) -> int:
    ok = np.isfinite(price[:, use_cols]).all(axis=1)
    return int(np.argmax(ok)) if ok.any() else len(price)


def build_refit(panels: dict[str, MarketPanel], train_end: str | dict, predict: tuple[str, str] | dict | None,
                lookback: int = 96, early_tail: float = 0.10) -> tuple[WindowSet, WindowSet | None, dict[str, Scaling]]:
    """Training windows (dated <= train_end) and prediction windows for days in `predict` (inclusive).
    `train_end` and `predict` may be dicts keyed by market (E13 MM_DIR: each market's own 80 % cut)."""
    vol_disabled = set(frozen()["features"].get("volume_disabled", []))
    lo_c, hi_c = frozen()["features"]["clip"]
    tr_parts, pr_parts, scales = [], [], {}
    for m, mp in panels.items():
        te = pd.Timestamp(train_end[m] if isinstance(train_end, dict) else train_end)
        pr_rng = predict[m] if isinstance(predict, dict) else predict
        price_cols = [i for i, f in enumerate(P.PRICE_FEATURES) if f != "vchg"]
        fv = _first_valid(mp.price, price_cols)
        in_train = (mp.dates <= te) & (np.arange(len(mp.dates)) >= fv)
        price_df = pd.DataFrame(mp.price, columns=P.PRICE_FEATURES)
        if m in vol_disabled:
            price_df["vchg"] = np.nan
        sent_df = pd.DataFrame(mp.sent, columns=SENT_BASE)
        sent_df["cov"] = D.coverage(pd.Series(mp.n_loc), pd.Series(in_train)).to_numpy()
        cov_med = float(np.median(mp.n_loc[in_train])) if in_train.any() else 0.0
        ps = P.fit_minmax(price_df[in_train], P.PRICE_FEATURES)
        ss = P.fit_minmax(sent_df[in_train], SENT_BASE + ["cov"])
        X_p = P.apply_minmax(price_df, ps).to_numpy(np.float32)
        X_s = P.apply_minmax(sent_df, ss).to_numpy(np.float32) if mp.has_sentiment else np.zeros((len(sent_df), 6), np.float32)
        y_ret, y_rv = targets(mp)
        last_tr = int(np.where(mp.dates <= te)[0].max())
        tr_idx = np.arange(max(fv + lookback - 1, lookback - 1), last_tr - max(HORIZONS) + 1)
        sd_ret = np.nanstd(y_ret[tr_idx], axis=0, ddof=1)
        sd_r1 = float(np.nanstd(mp.r1[fv:last_tr + 1], ddof=1))
        scales[m] = Scaling(ps, ss, cov_med, sd_ret, sd_r1)
        yr = y_ret / sd_ret
        yv = np.log(np.maximum(y_rv, 1e-12)) - 2 * np.log(sd_r1)
        yv[~np.isfinite(y_rv)] = np.nan

        def take(idx):
            w = idx[:, None] + np.arange(-lookback + 1, 1)[None, :]
            return (X_p[w], X_s[w], np.full(len(idx), mp.mid), yr[idx], yv[idx], np.full(len(idx), m),
                    mp.dates.values[idx])
        tr_parts.append(take(tr_idx))
        if pr_rng is not None:
            a, b = pd.Timestamp(pr_rng[0]), pd.Timestamp(pr_rng[1])
            pi = np.where((mp.dates >= a) & (mp.dates <= b))[0]
            pi = pi[pi >= fv + lookback - 1]
            if len(pi):
                pr_parts.append(take(pi))

    def cat(parts, with_val):
        xp, xs, mk, yr, yv, mkt, dt = (np.concatenate([p[i] for p in parts]) for i in range(7))
        if with_val:
            ud = np.unique(dt)
            cut = ud[int(np.floor((1 - early_tail) * len(ud)))]
            is_val = dt >= cut
        else:
            is_val = np.zeros(len(dt), bool)
        return WindowSet(xp, xs, mk, yr.astype(np.float32), yv.astype(np.float32), mkt, dt, is_val)
    tr = cat(tr_parts, True)
    pr = cat(pr_parts, False) if pr_parts else None
    return tr, pr, scales


def to_raw(market: np.ndarray, q_scaled: np.ndarray, logrv_scaled: np.ndarray, scales: dict[str, Scaling]):
    """Undo the target scaling: q [N, 3h, 3τ] in log-return units, logRV [N, 2] of raw returns."""
    sd = np.stack([scales[m].sd_ret for m in market])                  # [N, 3]
    sd1 = np.array([scales[m].sd_r1 for m in market])
    return q_scaled * sd[:, :, None], logrv_scaled + 2 * np.log(sd1)[:, None]
