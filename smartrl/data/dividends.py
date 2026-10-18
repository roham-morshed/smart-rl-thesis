"""Dividend-yield sources for the total-return construction of SPEC §2.1.

`nifty50_daily_yield` reads the official daily dividend yield of the Nifty 50 published by NSE Indices Ltd
(niftyindices.com, Reports → Historical Data → P/E, P/B & Div. Yield). The site serves at most one year per
request, so the history is fetched year by year and stored under data/raw/dividends.
"""
from __future__ import annotations

import time

import pandas as pd

from ..config import RAW

DIV_DIR = RAW / "dividends"
NIFTY_PAGE = "https://www.niftyindices.com/reports/historical-data"
NIFTY_API = "https://www.niftyindices.com/BackPage/getpepbHistoricaldataDBtoString"


def nifty50_daily_yield(first_year: int = 2007, end: str = "2026-09-30", refresh: bool = False) -> pd.DataFrame:
    path = DIV_DIR / "nifty50_pe_pb_dy.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from curl_cffi import requests
    s = requests.Session(impersonate="chrome")
    s.get(NIFTY_PAGE, timeout=60)
    rows = []
    end_ts = pd.Timestamp(end)
    for y in range(first_year, end_ts.year + 1):
        a = pd.Timestamp(f"{y}-01-01")
        b = min(pd.Timestamp(f"{y}-12-31"), end_ts)
        fmt = lambda d: d.strftime("%d-%b-%Y")
        cinfo = f"{{'name':'NIFTY 50','startDate':'{fmt(a)}','endDate':'{fmt(b)}','indexName':'NIFTY 50'}}"
        for k in range(4):
            r = s.post(NIFTY_API, json={"cinfo": cinfo}, timeout=60,
                       headers={"Referer": NIFTY_PAGE, "X-Requested-With": "XMLHttpRequest"})
            if r.status_code == 200 and r.text.startswith("["):
                rows += r.json()
                break
            time.sleep(3 * (k + 1))
        else:
            raise RuntimeError(f"niftyindices request failed for {y}: HTTP {r.status_code} {r.text[:200]}")
        time.sleep(1.0)
    df = pd.DataFrame(rows)
    df = pd.DataFrame({"date": pd.to_datetime(df["DATE"], format="%d %b %Y"),
                       "pe": pd.to_numeric(df["pe"], errors="coerce"),
                       "pb": pd.to_numeric(df["pb"], errors="coerce"),
                       "dy": pd.to_numeric(df["divYield"], errors="coerce") / 100.0})
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    DIV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
