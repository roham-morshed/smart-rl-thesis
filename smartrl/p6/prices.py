"""E14 price rows (protocol §2.1).

- Daily OHLCV from Yahoo (`auto_adjust=False`, already stored under data/raw/prices) from 2003-01-01; NIFTY 50 before
  2007-09-17 from the official NSE Indices history (data/raw/nifty50_2003_2008.csv), Yahoo from 2007-09-17.
- Calendar rows: every calendar date, non-trading dates take all values of the previous trading day (the paper).
- Flags: trading_row (the date is a session) and informative_row (the next calendar date is a session).
- Price features of SPEC §2.2 on the protocol's row grid (calendar rows or trading days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ROOT, frozen
from ..data import panel, prices
from .cfg import p6

COLS = ["open", "high", "low", "close", "volume"]


def nifty_splice() -> tuple[pd.DataFrame, dict]:
    """NSE history before the Yahoo start, Yahoo from it; returns bars and the splice check."""
    c = p6()["e14"]["nifty"]
    nse = pd.read_csv(ROOT / c["source_csv"], parse_dates=["date"]).set_index("date").sort_index()
    y = prices.fetch(frozen()["markets"]["NIFTY50"]["index"]).copy()
    y["date"] = pd.to_datetime(y["date"])
    y = y.set_index("date").sort_index()
    cut = pd.Timestamp(c["yahoo_from"])
    common = nse.index.intersection(y.index)
    common = common[common >= cut][: c["check_days"]]
    rel = (nse.loc[common, "close"] / y.loc[common, "close"] - 1.0).abs()
    check = {"n_days": int(len(common)), "max_rel_diff": float(rel.max()), "ok": bool(rel.max() <= c["max_rel_diff"])}
    early = nse[nse.index < cut][["open", "high", "low", "close"]].assign(volume=np.nan)
    late = y[y.index >= cut][COLS]
    return pd.concat([early, late]).sort_index(), check


def trading_bars(market: str) -> tuple[pd.DataFrame, dict]:
    """Trading-day OHLCV from history_start to the window end; volume NaN where masked or missing."""
    e = p6()["e14"]
    start, end = e["history_start"], e["window"][1]
    info: dict = {}
    if market == "NIFTY50":
        bars, info["nifty_splice"] = nifty_splice()
    else:
        raw = prices.fetch(frozen()["markets"][market]["index"]).copy()
        raw["date"] = pd.to_datetime(raw["date"])
        bars = raw.set_index("date").sort_index()[COLS]
    bars = bars[(bars.index >= start) & (bars.index <= end)].dropna(subset=["close"]).copy()
    for col in ("open", "high", "low"):
        bars[col] = bars[col].where(bars[col] > 0, bars["close"])
    bars["volume"] = bars["volume"].where(bars["volume"] > 0)
    if market in frozen()["features"].get("volume_disabled", []) or market == "NIFTY50":
        bars["volume"] = np.nan                              # Nifty volume masked for all dates (protocol §2.1)
    return bars, info


def calendar_rows(bars: pd.DataFrame) -> pd.DataFrame:
    """Every calendar date from the first trading date to the window end, filled with the previous trading day."""
    end = pd.Timestamp(p6()["e14"]["window"][1])
    cal = pd.date_range(bars.index.min(), end, freq="D")
    out = bars.reindex(cal).ffill()
    out["trading_row"] = cal.isin(bars.index)
    nxt = np.append(out["trading_row"].to_numpy()[1:], False)
    out["informative_row"] = nxt
    out.index.name = "date"
    return out


def trading_rows(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out["trading_row"] = True
    out["informative_row"] = True
    out.index.name = "date"
    return out


def grid(market: str, protocol: str) -> pd.DataFrame:
    """OHLCV, flags and SPEC §2.2 price features on the protocol's row grid."""
    bars, _ = trading_bars(market)
    g = calendar_rows(bars) if protocol.startswith("PUB") else trading_rows(bars)
    feats = panel.price_features(g[COLS], use_volume=g["volume"].notna().any())
    return g.join(feats)


def labels(close: np.ndarray) -> dict[str, np.ndarray]:
    """GE: 1[C_{t+1} >= C_t]; GT: 1[C_{t+1} > C_t]; NaN on the last row."""
    out = {}
    for name, op in (("GE", np.greater_equal), ("GT", np.greater)):
        y = np.full(len(close), np.nan)
        y[:-1] = op(close[1:], close[:-1]).astype(float)
        out[name] = y
    return out


def ghost_features(g: pd.DataFrame) -> pd.DataFrame:
    """For every row t, the SPEC §2.2 features of a row t+1 whose close is replaced by C_t (protocol §2.4, LA);
    open, high, low and volume are those of t+1. Row t of the result describes that ghost row."""
    c = g["close"].to_numpy(float)
    lc = np.log(c)
    h1, l1 = g["high"].shift(-1).to_numpy(float), g["low"].shift(-1).to_numpy(float)
    v = g["volume"].to_numpy(float)
    v1 = g["volume"].shift(-1).to_numpy(float)
    n = len(c)
    out = pd.DataFrame(index=g.index)
    out["r1"] = 0.0
    for k in (5, 20, 60):                                    # log(C_ghost / C_{t+1-k}) = log(C_t / C_{t+1-k})
        r = np.full(n, np.nan)
        r[k - 1:] = lc[k - 1:] - lc[: n - k + 1]
        out[f"r{k}"] = r
    r1 = g["r1"].to_numpy(float)
    for k in (5, 20):                                        # the ghost's own r1 is 0
        s = pd.Series(r1 ** 2).rolling(k - 1, min_periods=k - 1).sum().to_numpy()
        out[f"rv{k}"] = np.sqrt(s / k) * np.sqrt(252)
    out["range"] = (h1 - l1) / c
    out["rsi14"] = g["rsi14"].to_numpy(float)                # a zero change leaves Wilder's ratio unchanged
    cs = pd.Series(c, index=g.index)
    e12, e26 = cs.ewm(span=12, adjust=False).mean(), cs.ewm(span=26, adjust=False).mean()
    a12, a26, a9 = 2 / 13, 2 / 27, 2 / 10
    e12g, e26g = e12 + a12 * (cs - e12), e26 + a26 * (cs - e26)
    line, line_g = e12 - e26, e12g - e26g
    sig = line.ewm(span=9, adjust=False).mean()
    sig_g = sig + a9 * (line_g - sig)
    out["macd"] = ((line_g - sig_g) / cs).to_numpy()
    for k in (50, 200):                                      # window of the ghost: C_{t-k+2..t} and C_t again
        s = cs.rolling(k - 1, min_periods=k - 1).sum()
        out[f"gap{k}"] = (cs / ((s + cs) / k) - 1.0).to_numpy()
    out["dd252"] = (cs / cs.rolling(251, min_periods=251).max() - 1.0).to_numpy()
    if np.isfinite(v).any():
        vs = pd.Series(v).rolling(19, min_periods=19).sum().to_numpy()
        out["vchg"] = np.log(v1 / ((vs + v1) / 20.0))
    else:
        out["vchg"] = np.nan
    return out[panel.PRICE_FEATURES]
