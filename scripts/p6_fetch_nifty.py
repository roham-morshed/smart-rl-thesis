"""P6 E14 (protocol §2.1): official NIFTY 50 price index history (OHLC) from NSE Indices, 2003-01-01 → 2008-03-31.

Source: niftyindices.com → Historical Data → NIFTY 50 (the page's own data request, /BackPage/getHistoricaldatatabletoString).
Writes data/raw/nifty50_2003_2008.csv (date, open, high, low, close) and a JSON note with the request details.

    python scripts/p6_fetch_nifty.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.config import RAW  # noqa: E402

BASE = "https://www.niftyindices.com"
OUT = RAW / "nifty50_2003_2008.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
CHUNKS = [("01-Jan-2003", "31-Dec-2003"), ("01-Jan-2004", "31-Dec-2004"), ("01-Jan-2005", "31-Dec-2005"),
          ("01-Jan-2006", "31-Dec-2006"), ("01-Jan-2007", "31-Dec-2007"), ("01-Jan-2008", "31-Mar-2008")]


def main() -> None:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    s.get(f"{BASE}/reports/historical-data", timeout=60)
    rows = []
    for start, end in CHUNKS:
        cinfo = f"{{'name':'NIFTY 50','startDate':'{start}','endDate':'{end}','indexName':'NIFTY 50'}}"
        r = s.post(f"{BASE}/BackPage/getHistoricaldatatabletoString", json={"cinfo": cinfo}, timeout=120,
                   headers={"Referer": f"{BASE}/reports/historical-data", "Origin": BASE,
                            "X-Requested-With": "XMLHttpRequest"})
        r.raise_for_status()
        data = json.loads(r.text)
        for x in data:
            rows.append({"date": pd.to_datetime(x["HistoricalDate"], format="%d %b %Y"),
                         "open": float(x["OPEN"]), "high": float(x["HIGH"]), "low": float(x["LOW"]),
                         "close": float(x["CLOSE"])})
        print(f"{start} .. {end}: {len(data)} rows", flush=True)
        time.sleep(1.0)
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    note = {"source": "NSE Indices (niftyindices.com), Historical Data, NIFTY 50 price index (OHLC)",
            "endpoint": f"{BASE}/BackPage/getHistoricaldatatabletoString", "chunks": CHUNKS,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(df), "first": str(df["date"].min().date()), "last": str(df["date"].max().date())}
    OUT.with_suffix(".json").write_text(json.dumps(note, indent=1), encoding="utf-8")
    print(note, flush=True)


if __name__ == "__main__":
    main()
