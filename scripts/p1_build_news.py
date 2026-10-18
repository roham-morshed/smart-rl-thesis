"""P1 (E0, news): build news_items.parquet and news_aligned_<MKT>.parquet from the approved subsets, and the
data.news.* keys of results.json. Needs the downloaded subsets (scripts/download_all.py)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, RESULTS, frozen  # noqa: E402
from smartrl.data import news, panel  # noqa: E402
from smartrl.runs import run_record  # noqa: E402

STATS_WINDOW = ("2010-01-01", "2025-12-31")     # policy start .. end of the news date range (SPEC §1, §2.3)


def coverage(market: str, items: pd.DataFrame) -> dict:
    cfg = frozen()
    local = cfg["markets"][market]["news_tag"]
    cal = pd.DatetimeIndex(pd.read_parquet(panel.processed_path(market), columns=["date"])["date"])
    al = news.align_market(items, market, cal)
    al.to_parquet(news.aligned_path(market), index=False)
    cnt = al.groupby(["trading_date", "tag"]).size().unstack(fill_value=0).reindex(cal, fill_value=0)
    for col in ("GLOBAL", local):
        if col not in cnt:
            cnt[col] = 0
    w = cnt.loc[STATS_WINDOW[0]:STATS_WINDOW[1]]
    by_year = (cnt[local] > 0).groupby(cnt.index.year).mean()
    return {"mean_daily_local": float(w[local].mean()), "share_days_with_local": float((w[local] > 0).mean()),
            "mean_daily_global": float(w["GLOBAL"].mean()),
            "share_days_with_local_by_year": {str(y): float(v) for y, v in by_year.items()},
            "mean_daily_local_by_year": {str(y): float(v) for y, v in cnt[local].groupby(cnt.index.year).mean().items()},
            "n_aligned_rows": int(len(al)), "n_items_not_aligned": int(len(items) - al.loc[al["tag"] == "GLOBAL"].shape[0])}


def main() -> None:
    cfg = frozen()
    subsets = cfg["news"]["subsets_keep"]
    out: dict = {"stats_window": STATS_WINDOW, "subsets": {}}
    with run_record("P1", "build_news", None, "cpu", {"subsets": subsets}) as run:
        for s in subsets:
            t0 = time.time()
            out["subsets"][s] = news.extract_subset(s)
            out["subsets"][s]["seconds"] = time.time() - t0
            print(s, json.dumps(out["subsets"][s], default=str)[:400], flush=True)
        out["dedup"] = news.dedupe_and_tag(subsets)
        print(json.dumps(out["dedup"], indent=1, default=str), flush=True)
        items = pd.read_parquet(news.ITEMS, columns=["id", "ts_utc"] + [f"tag_{t}" for t in ("US", "UK", "IN", "CN")])
        out["markets"] = {m: coverage(m, items) for m in MARKETS}
        run.set(outputs=["news_items.parquet"] + [news.aligned_path(m).name for m in MARKETS])
    d = out["dedup"]
    keys = [("data.news.n_raw", sum(v["n_raw"] for v in out["subsets"].values())),
            ("data.news.n_kept", d["n_kept"]), ("data.news.n_duplicates", d["n_duplicates"])]
    for s, v in out["subsets"].items():
        k = d["kept_by_source"].get(s, {"n_kept": 0, "first_date": None, "last_date": None})
        keys += [(["data", "news", "source", s, "n_raw"], v["n_raw"]), (["data", "news", "source", s, "n_kept"], k["n_kept"]),
                 (["data", "news", "source", s, "first_date"], k["first_date"]),
                 (["data", "news", "source", s, "last_date"], k["last_date"])]
    for t, n in d["tags"].items():
        keys.append((f"data.news.tag.{t}.n", n))
    for m, c in out["markets"].items():
        for k in ("mean_daily_local", "share_days_with_local", "mean_daily_global"):
            keys.append((f"data.news.market.{m}.{k}", c[k]))
        for y, v in c["share_days_with_local_by_year"].items():
            keys.append((["data", "news", "market", m, "share_days_with_local_by_year", y], v))
    keys.append(("p1.news", out))
    notes = [f"data.news.market.*: means and shares over trading days {STATS_WINDOW[0]} .. {STATS_WINDOW[1]} "
             "(policy start to the end of the news date range); by-year shares cover every panel year."]
    R.update(keys, notes)
    (RESULTS / "p1_news.json").write_text(json.dumps(R._clean(out), indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
