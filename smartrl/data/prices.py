"""Raw daily prices from Yahoo Finance (SPEC §2.1): download once, store under data/raw/prices."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import RAW

PRICE_DIR = RAW / "prices"
COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]


def raw_path(ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace(".", "_")
    return PRICE_DIR / f"{safe}.parquet"


def download(ticker: str, start: str = "1990-01-01", end_inclusive: str = "2026-09-30",
             retries: int = 5) -> pd.DataFrame:
    """Daily bars with auto_adjust=False plus dividend and split actions; dates are local exchange dates."""
    import yfinance as yf
    end_excl = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_err: Exception | None = None
    for k in range(retries):
        try:
            h = yf.Ticker(ticker).history(start=start, end=end_excl, auto_adjust=False, actions=True,
                                          raise_errors=True)
            if h is None or h.empty:
                raise RuntimeError(f"empty history for {ticker}")
            idx = h.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            df = pd.DataFrame({
                "date": pd.DatetimeIndex(idx).normalize(),
                "open": h["Open"].to_numpy(), "high": h["High"].to_numpy(), "low": h["Low"].to_numpy(),
                "close": h["Close"].to_numpy(), "adj_close": h.get("Adj Close", h["Close"]).to_numpy(),
                "volume": h["Volume"].to_numpy(),
                "dividends": h["Dividends"].to_numpy() if "Dividends" in h else 0.0,
                "splits": h["Stock Splits"].to_numpy() if "Stock Splits" in h else 0.0,
            })
            df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
            return df[COLUMNS]
        except Exception as e:  # network errors and rate limits: back off and retry
            last_err = e
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"yfinance download failed for {ticker}: {type(last_err).__name__}: {last_err}")


def fetch(ticker: str, refresh: bool = False, **kw) -> pd.DataFrame:
    p = raw_path(ticker)
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    df = download(ticker, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df
