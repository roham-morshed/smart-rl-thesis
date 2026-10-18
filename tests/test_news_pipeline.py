"""End-to-end check of the news pipeline (SPEC §2.3) on synthetic shards with the multisource schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.data import news  # noqa: E402


def _row(date, text, **extra):
    return {"date": date, "text": text, "extra_fields": json.dumps(extra)}


def _shard(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "PARTS", tmp_path / "parts")
    monkeypatch.setattr(news, "ITEMS", tmp_path / "items.parquet")
    fn = _shard(tmp_path / "fn.parquet", [
        _row("2019-03-08T20:59:00Z", "Apple <b>beats</b> estimates on iPhone sales\n\nbody", time_precision="minute"),
        _row("2019-03-08T22:00:00Z", "APPLE beats estimates, on iPhone sales!\n\nbody", time_precision="minute"),   # dup
        _row("2019-03-09T00:00:00Z", "FTSE 100 falls as sterling slides http://x.co/a", time_precision="day"),
        _row("2019-03-10T00:00:00Z", "Too short", time_precision="day"),
        _row("2006-12-31T10:00:00Z", "Out of range item about markets today", time_precision="minute"),
    ])
    nyt = _shard(tmp_path / "nyt.parquet", [
        _row("2019-03-11T15:00:00Z", "A summary that is not the headline\n\nFed holds rates steady again\n\nlead",
             text_type="abstract+headline+lead", news_desk="Business", section_name="Business Day"),
        _row("2019-03-11T15:00:00Z", "Summary\n\nTheater review of a new play tonight\n\nlead",
             text_type="abstract+headline+lead", news_desk="Culture", section_name="Arts"),
        _row("2019-03-12T03:00:00Z", "India's Sensex hits record as rupee firms\n\nx",
             text_type="headline+lead", url="https://www.nytimes.com/2019/03/12/business/india.html"),
        _row("2019-03-12T04:00:00Z", "SSE plc shares rise after Shanghai rally\n\nx",
             text_type="headline+lead", news_desk="Business"),
        _row("2019-03-12T05:00:00Z", "U.K. inflation surprises the Bank of England\n\nx",
             text_type="headline+lead", news_desk="Financial Desk"),
    ])
    s1 = news.extract_subset("fnspid_news", [fn])
    s2 = news.extract_subset("nyt_headlines_2010_2021", [nyt])
    assert s1["n_raw"] == 4 and s1["n_short_dropped"] == 1
    assert s2["n_nonbusiness_dropped"] == 1
    parts = pd.read_parquet(tmp_path / "parts" / "nyt_headlines_2010_2021.parquet")
    assert "Fed holds rates steady again" in set(parts["headline"])
    p1 = pd.read_parquet(tmp_path / "parts" / "fnspid_news.parquet")
    assert "Apple beats estimates on iPhone sales" in set(p1["headline"])
    day = p1[p1["headline"].str.startswith("FTSE")].iloc[0]
    assert pd.Timestamp(day["ts_utc"]).tz_convert("UTC").strftime("%H:%M:%S") == "23:59:59" and day["day_only"]
    assert "http" not in day["headline"]
    rep = news.dedupe_and_tag(["fnspid_news", "nyt_headlines_2010_2021"])
    assert rep["n_duplicates"] == 1
    items = pd.read_parquet(tmp_path / "items.parquet").set_index("headline")
    assert items.loc["FTSE 100 falls as sterling slides", "tag_UK"]
    assert items.loc["Fed holds rates steady again", "tag_US"]
    assert items.loc["Apple beats estimates on iPhone sales", "tag_US"]          # auto-tag subset
    assert items.loc["India's Sensex hits record as rupee firms", "tag_IN"]
    assert not items.loc["India's Sensex hits record as rupee firms", "tag_US"]
    assert items.loc["U.K. inflation surprises the Bank of England", "tag_UK"]
    sse = items.loc["SSE plc shares rise after Shanghai rally"]
    assert sse["tag_CN"]                                                          # via "Shanghai", not "SSE"
    cal = pd.DatetimeIndex(pd.bdate_range("2019-03-01", "2019-03-29"))
    al = news.align_market(items.reset_index(), "SP500", cal)
    assert set(al["tag"]) == {"GLOBAL", "US"}
    first = items.reset_index().set_index("id").loc[al["id"]]
    assert (news.utc_seconds(first["ts_utc"].to_numpy()) < news.close_utc("SP500", pd.DatetimeIndex(al["trading_date"]))).all()
