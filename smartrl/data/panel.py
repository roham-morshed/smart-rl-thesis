"""Data Processing Module, prices (SPEC §2.1–2.2): per-market calendars, total-return series, causal features.

Every feature of day t is a function of bars dated <= t only (rolling windows, recursive EWMs). Scaling to the
fold's training quantiles happens later (`fit_minmax` / `apply_minmax`), never here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import PROCESSED, frozen
from . import dividends, prices

PRICE_FEATURES = ["r1", "r5", "r20", "r60", "rv5", "rv20", "range", "rsi14", "macd", "gap50", "gap200", "dd252", "vchg"]
RF_DAILY = (1.0 + frozen()["risk_free_annual"]) ** (1 / 252) - 1


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI: first average = simple mean of the first n changes, then recursive smoothing (causal)."""
    d = close.diff().to_numpy()
    gain, loss = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(d), np.nan)
    if len(d) <= n:
        return pd.Series(out, index=close.index)
    ag, al = np.nanmean(gain[1:n + 1]), np.nanmean(loss[1:n + 1])
    for t in range(n, len(d)):
        if t > n:
            ag = (ag * (n - 1) + gain[t]) / n
            al = (al * (n - 1) + loss[t]) / n
        out[t] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return pd.Series(out, index=close.index)


def price_features(bars: pd.DataFrame, use_volume: bool) -> pd.DataFrame:
    """bars: date-indexed open/high/low/close/volume of one market's index, on that market's calendar."""
    c, h, l = bars["close"], bars["high"], bars["low"]
    lc = np.log(c)
    f = pd.DataFrame(index=bars.index)
    f["r1"] = lc.diff()
    for k in (5, 20, 60):
        f[f"r{k}"] = lc - lc.shift(k)
    for k in (5, 20):
        f[f"rv{k}"] = np.sqrt((f["r1"] ** 2).rolling(k, min_periods=k).mean()) * math.sqrt(252)
    f["range"] = (h - l) / c
    f["rsi14"] = wilder_rsi(c, 14) / 100.0
    line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd"] = (line - line.ewm(span=9, adjust=False).mean()) / c
    for k in (50, 200):
        f[f"gap{k}"] = c / c.rolling(k, min_periods=k).mean() - 1.0
    f["dd252"] = c / c.rolling(252, min_periods=252).max() - 1.0
    if use_volume:
        v = bars["volume"].where(bars["volume"] > 0)          # zero volume = missing (masked, never imputed)
        f["vchg"] = np.log(v / v.rolling(20, min_periods=20).mean())
    else:
        f["vchg"] = np.nan
    return f


def _ttm_yield_isf(dates: pd.DatetimeIndex) -> pd.Series:
    """Trailing-12-month distribution yield of ISF on `dates` (distributions in GBP, ISF.L closes in pence)."""
    dist = dividends.isf_distributions().set_index("ex_date")["gbp_per_share"] * 100.0
    px = prices.fetch("ISF.L").set_index("date")["close"]
    daily = pd.Series(0.0, index=pd.date_range(min(dist.index.min(), px.index.min()), dates.max(), freq="D"))
    daily = daily.add(dist.groupby(level=0).sum(), fill_value=0.0)
    ttm = daily.rolling("365D").sum()
    px_asof = px.reindex(dates, method="ffill")
    y = ttm.reindex(dates) / px_asof
    return y.where(dates >= px.index.min())


def dividend_yield(market: str, dates: pd.DatetimeIndex) -> tuple[pd.Series, str]:
    fb = frozen()["dividend_fallback"]
    if market == "SP500":
        return pd.Series(np.nan, index=dates), "^SP500TR (total-return index)"
    if market == "FTSE100":
        return _ttm_yield_isf(dates), "ISF trailing-12-month distribution yield (iShares distribution history)"
    if market == "NIFTY50":
        d = dividends.nifty50_daily_yield().set_index("date")["dy"]
        return d.reindex(dates, method="ffill"), "NSE Indices daily dividend yield of NIFTY 50"
    if market == "SSE":
        if isinstance(fb["SSE"], dict) and isinstance(fb["SSE"].get("values"), dict):
            vals = fb["SSE"]["values"]
            return pd.Series([vals.get(d.year, np.nan) for d in dates], index=dates), str(fb["SSE"]["source"])
        return pd.Series(np.nan, index=dates), "none: price return only (no cited SSE Composite dividend yield)"
    raise KeyError(market)


def build_market(market: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    cfg = frozen()
    m = cfg["markets"][market]
    raw = prices.fetch(m["index"]).copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
    n_no_close = int(raw["close"].isna().sum())
    bars = raw.dropna(subset=["close"]).set_index("date").sort_index()
    for col in ("open", "high", "low"):          # a missing high/low/open is replaced by the close of the same bar
        bars[col] = bars[col].where(bars[col] > 0, bars["close"])
    use_volume = market not in cfg["features"].get("volume_disabled", [])
    feats = price_features(bars, use_volume)
    out = bars[["open", "high", "low", "close", "volume"]].join(feats)
    out["r_price"] = out["close"].pct_change()
    dy, dy_src = dividend_yield(market, out.index)
    out["dy"] = dy
    if m["total_return"]:
        tr = prices.fetch(m["total_return"]).set_index("date")["close"]
        tr.index = pd.to_datetime(tr.index)
        tr = tr.reindex(out.index)
        out["r_tr"] = tr / tr.shift(1) - 1.0
        n_tr_missing = int(tr.isna().sum())
    else:
        out["r_tr"] = out["r_price"] + out["dy"].fillna(0.0) / 252.0
        if market != "SSE":                       # no yield known yet (before the proxy's first price): no TR
            out.loc[out["dy"].isna(), "r_tr"] = np.nan
        n_tr_missing = int(out["dy"].isna().sum())
    flagged = out.index[(out["r1"].abs() > 0.20)]
    out.insert(0, "market", market)
    out = out.reset_index().rename(columns={"index": "date"})
    stats = {
        "start": str(out["date"].min().date()), "end": str(out["date"].max().date()), "n_days": int(len(out)),
        "dividend_source": dy_src, "n_flagged_returns": int(len(flagged)),
        "flagged_dates": [str(d.date()) for d in flagged], "n_rows_without_close": n_no_close,
        "volume_used": use_volume, "n_days_tr_unavailable": n_tr_missing,
        "share_high_eq_low": float((out["high"] == out["low"]).mean()),
    }
    return out, stats


def first_common_date(end: str) -> str:
    cfg = frozen()
    firsts = [pd.to_datetime(prices.fetch(m["index"])["date"]).min() for m in cfg["markets"].values()]
    return str(max(firsts).date())


def processed_path(market: str):
    return PROCESSED / f"prices_{market}.parquet"


# ---------------------------------------------------------------- fold scaling (SPEC §2.2)
def fit_minmax(train: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    """q01 and q99 of each column on the fold's training rows only."""
    lo_q, hi_q = frozen()["features"]["minmax_quantiles"]
    out = {}
    for c in cols:
        x = train[c].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        out[c] = (float(np.quantile(x, lo_q)), float(np.quantile(x, hi_q))) if len(x) else (0.0, 1.0)
    return out


def apply_minmax(df: pd.DataFrame, params: dict[str, tuple[float, float]], fill: float = 0.0) -> pd.DataFrame:
    """x' = (x − q01)/(q99 − q01), clipped; masked (NaN) inputs become `fill` after scaling."""
    lo_c, hi_c = frozen()["features"]["clip"]
    out = df.copy()
    for c, (lo, hi) in params.items():
        den = hi - lo if hi > lo else 1.0
        out[c] = ((df[c] - lo) / den).clip(lo_c, hi_c).fillna(fill)
    return out
