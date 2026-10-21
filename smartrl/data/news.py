"""Data Processing Module, news (SPEC §2.3): headline extraction, cleaning, deduplication, market tags, sessions.

Pipeline (scripts/p1_build_news.py):
1. `extract_subset`: per approved subset, read the parquet shards with DuckDB, keep 2007-01-01..2025-12-31, take
   the headline, clean it, apply the NYT business filter, set ts_utc (day-only rows -> 23:59:59 UTC).
2. `dedupe`: key = normalised headline + UTC date over all subsets; earliest timestamp wins.
3. `tag`: GLOBAL for every item; US / UK / IN / CN by subset or keyword (word boundaries; words case-insensitive,
   acronyms case-sensitive).
4. `align`: per market, first trading date d with ts_utc < close_utc(d).
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import MARKETS, PROCESSED, frozen
from . import hf

NEWS_REPO = "Brianferrell787/financial-news-multisource"
PARTS = PROCESSED / "news_parts"
ITEMS = PROCESSED / "news_items.parquet"

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_WS_RE = re.compile(r"\s+")
_NONALNUM_RE = re.compile(r"[^0-9a-z\s]+")
BUSINESS_RE = re.compile(r"business|financial|dealbook|your money|your-money|economy|sundaybusiness|markets", re.I)
URL_BUSINESS_RE = re.compile(r"nytimes\.com/(?:\d{4}/\d{2}/\d{2}/)?(?:business|your-money|economy|dealbook)/", re.I)


# ---------------------------------------------------------------- text helpers
def dedup_key(headline: str) -> str:
    """SPEC §2.3: lower case, non-alphanumerics removed, whitespace collapsed."""
    return _WS_RE.sub(" ", _NONALNUM_RE.sub("", str(headline).lower())).strip()


def clean_headline(text: str, max_tokens: int) -> str:
    s = html.unescape(_TAG_RE.sub(" ", str(text)))
    s = _URL_RE.sub(" ", s)
    toks = _WS_RE.sub(" ", s).strip().split(" ")
    return " ".join(toks[:max_tokens]) if toks != [""] else ""


def headline_from_text(text: str, text_type: str | None) -> str:
    """First line of `text` (SPEC §2.3). When the subset documents another field order (e.g. NYT 2010-2021:
    'abstract+headline+lead', fields separated by blank lines), the headline/title field is taken instead."""
    text = str(text or "")
    tt = [p.strip() for p in str(text_type or "").split("+")]
    first = next((i for i, p in enumerate(tt) if p in ("headline", "title")), 0)
    if first > 0:
        fields = [f for f in re.split(r"\n\s*\n", text) if f.strip()]
        if len(fields) > first:
            return fields[first].split("\n", 1)[0]
    return text.split("\n", 1)[0]


# ---------------------------------------------------------------- 1. extraction
def subset_files(subset: str) -> list[Path]:
    """Complete local files of a subset (data/raw/news_multisource/<subset>/, written by scripts/download_all.py or
    by a manual download); every file of manifest.json must be present with its exact size."""
    import json
    from ..config import RAW
    d = RAW / "news_multisource"
    man = [f for f in json.loads((d / "manifest.json").read_text(encoding="utf-8")) if f["subset"] == subset]
    out = []
    for f in man:
        p = d / subset / Path(f["path"]).name
        if not p.exists() or p.stat().st_size != f["size"]:
            raise FileNotFoundError(f"{p} missing or incomplete; run scripts/download_all.py")
        out.append(p)
    return out


def extract_subset(subset: str, files: list[Path] | None = None) -> dict:
    """Write data/processed/news_parts/<subset>.parquet (source, ts_utc, day_only, headline) and return counts."""
    import duckdb
    cfg = frozen()["news"]
    files = files if files is not None else subset_files(subset)
    lo, hi = cfg["date_range"]
    business = subset in cfg.get("business_only_subsets", [])
    con = duckdb.connect()
    flist = "[" + ",".join("'" + str(p).replace("\\", "/") + "'" for p in files) + "]"
    n_total = con.execute(f"SELECT count(*) FROM read_parquet({flist})").fetchone()[0]
    q = f"""
        SELECT CAST(date AS VARCHAR) AS date, left(text, 2000) AS text,
               json_extract_string(extra_fields, '$.time_precision') AS tp,
               json_extract_string(extra_fields, '$.text_type') AS tt,
               json_extract_string(extra_fields, '$.news_desk') AS desk,
               json_extract_string(extra_fields, '$.section_name') AS sec,
               coalesce(json_extract_string(extra_fields, '$.url'), json_extract_string(extra_fields, '$.web_url')) AS url
        FROM read_parquet({flist})
        WHERE CAST(date AS VARCHAR) >= '{lo}' AND CAST(date AS VARCHAR) < '{int(hi[:4]) + 1}-01-01'
    """
    rel = con.execute(q)
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARTS.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("source", pa.string()), ("ts_utc", pa.timestamp("us", tz="UTC")), ("day_only", pa.bool_()),
                        ("headline", pa.string())])
    writer = pq.ParquetWriter(PARTS / f"{subset}.parquet", schema)
    n_raw = n_short = n_nonbusiness = n_out = n_day = n_ny_midnight = 0
    desks: dict = {}
    max_tok, min_tok = cfg["max_tokens"], cfg["min_tokens"]
    day_time = time.fromisoformat(cfg["day_only_time_utc"])
    while True:
        chunk = rel.fetch_df_chunk(50)          # 50 vectors of 2048 rows
        if chunk is None or len(chunk) == 0:
            break
        n_raw += len(chunk)
        if business:
            is_bus = (chunk["desk"].fillna("").str.contains(BUSINESS_RE) | chunk["sec"].fillna("").str.contains(BUSINESS_RE)
                      | chunk["url"].fillna("").str.contains(URL_BUSINESS_RE))
            for k, v in chunk.loc[is_bus, "desk"].fillna(chunk.loc[is_bus, "sec"]).fillna("(url)").value_counts().items():
                desks[k] = desks.get(k, 0) + int(v)
            n_nonbusiness += int((~is_bus).sum())
            chunk = chunk[is_bus]
        heads = [clean_headline(headline_from_text(x, tt), max_tok) for x, tt in zip(chunk["text"], chunk["tt"])]
        ntok = np.fromiter((len(h.split(" ")) if h else 0 for h in heads), dtype=np.int32, count=len(heads))
        keep = ntok >= min_tok
        n_short += int((~keep).sum())
        ts = pd.to_datetime(chunk["date"], utc=True, format="mixed")
        midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
        # a timestamp at exactly midnight in New York is a date stamp too (NYT print dates): conservative day-only
        ny = ts.dt.tz_convert("America/New_York")
        ny_mid = (ny.dt.hour == 0) & (ny.dt.minute == 0) & (ny.dt.second == 0)
        day_only = (chunk["tp"].fillna("") == "day").to_numpy() | midnight.to_numpy() | ny_mid.to_numpy()
        n_ny_midnight += int((ny_mid & ~midnight).sum())
        # day-only items take 23:59:59 UTC of their (UTC or New York) calendar date
        cal_date = pd.Series(np.where(ny_mid & ~midnight, ny.dt.date, ts.dt.date), index=ts.index)
        day_ts = pd.to_datetime(cal_date.astype(str)).dt.tz_localize("UTC") + pd.Timedelta(
            hours=day_time.hour, minutes=day_time.minute, seconds=day_time.second)
        ts = ts.where(~day_only, day_ts)
        part = pd.DataFrame({"source": subset, "ts_utc": ts.dt.as_unit("us"), "day_only": day_only,
                             "headline": np.asarray(heads, dtype=object)})[keep]
        if len(part):
            writer.write_table(pa.Table.from_pandas(part, schema=schema, preserve_index=False))
            n_out += len(part)
            n_day += int(part["day_only"].sum())
    writer.close()
    return {"n_rows_subset": int(n_total), "n_raw": int(n_raw), "n_short_dropped": n_short,
            "n_nonbusiness_dropped": n_nonbusiness, "n_after_clean": int(n_out),
            "share_day_only": n_day / n_out if n_out else None, "n_new_york_midnight_day_only": n_ny_midnight,
            "business_desks_kept": dict(sorted(desks.items(), key=lambda kv: -kv[1])[:40]) if business else None}


# ---------------------------------------------------------------- 2. deduplication (+ 3. tags)
def _boundary_regex(terms: list[str]) -> str:
    """(?:^|\\W)(t1|t2|...)(?:\\W|$) — a word-boundary match that also works for terms ending in '.' (U.K.)."""
    esc = [re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", t) for t in terms]
    return r"(?:^|\W)(?:" + "|".join(esc) + r")(?:\W|$)"


def tag_patterns() -> dict[str, tuple[str | None, str | None]]:
    kw = frozen()["news_keywords"]
    return {tag: (_boundary_regex(v["words"]) if v.get("words") else None,
                  _boundary_regex(v["acronyms"]) if v.get("acronyms") else None) for tag, v in kw.items()}


def dedupe_and_tag(subsets: list[str]) -> dict:
    """Global dedup over all parts, then tags; writes news_items.parquet (id, source, ts_utc, headline, tags...)."""
    import duckdb
    files = ",".join("'" + str(PARTS / f"{s}.parquet").replace("\\", "/") + "'" for s in subsets)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
        CREATE TABLE items AS
        SELECT source, ts_utc, day_only, headline,
               trim(regexp_replace(regexp_replace(lower(headline), '[^0-9a-z\\s]+', '', 'g'), '\\s+', ' ', 'g')) AS k,
               CAST(ts_utc AS DATE) AS d
        FROM read_parquet([{files}])
    """)
    n_before = con.execute("SELECT count(*) FROM items").fetchone()[0]
    con.execute("""
        CREATE TABLE ranked AS
        SELECT *, row_number() OVER (PARTITION BY k, d ORDER BY ts_utc, source) AS rn FROM items
    """)
    dup_by_source = dict(con.execute("SELECT source, count(*) FROM ranked WHERE rn > 1 GROUP BY source").fetchall())
    us_subsets = frozen()["news_keywords"]["US"].get("auto_tag_subsets", [])
    pats = tag_patterns()
    cols = []
    for tag, (words, acr) in pats.items():
        conds = []
        if words:
            conds.append(f"regexp_matches(headline, '{words.replace(chr(39), chr(39) * 2)}', 'i')")
        if acr:
            conds.append(f"regexp_matches(headline, '{acr.replace(chr(39), chr(39) * 2)}')")
        if tag == "US" and us_subsets:
            conds.append("source IN (" + ",".join(f"'{s}'" for s in us_subsets) + ")")
        cols.append(f"({' OR '.join(conds) if conds else 'false'}) AS tag_{tag}")
    con.execute(f"""
        CREATE TABLE kept AS
        SELECT row_number() OVER (ORDER BY ts_utc, source, k) - 1 AS id, source, ts_utc, day_only, headline,
               {', '.join(cols)}
        FROM ranked WHERE rn = 1
    """)
    con.execute(f"COPY kept TO '{str(ITEMS).replace(chr(92), '/')}' (FORMAT PARQUET)")
    n_after = con.execute("SELECT count(*) FROM kept").fetchone()[0]
    by_src = con.execute("SELECT source, count(*), min(ts_utc), max(ts_utc) FROM kept GROUP BY source").fetchall()
    tags = {t: con.execute(f"SELECT count(*) FROM kept WHERE tag_{t}").fetchone()[0] for t in pats}
    return {"n_before_dedup": int(n_before), "n_kept": int(n_after), "n_duplicates": int(n_before - n_after),
            "duplicates_by_source": {k: int(v) for k, v in dup_by_source.items()},
            "kept_by_source": {s: {"n_kept": int(n), "first_date": str(a)[:10], "last_date": str(b)[:10]}
                               for s, n, a, b in by_src},
            "tags": {"GLOBAL": int(n_after), **{t: int(v) for t, v in tags.items()}}}


# ---------------------------------------------------------------- 4. sessions
def close_utc(market: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Official close of `market` on each local trading date, as UTC (zoneinfo, DST-aware)."""
    m = frozen()["markets"][market]
    tz = ZoneInfo(m["tz"])
    hh, mm = (int(x) for x in m["close"].split(":"))
    return np.array([datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz).timestamp() for d in dates])


def assign_sessions(ts_utc: pd.Series, market: str, calendar: pd.DatetimeIndex) -> pd.Series:
    """First trading date d of `market` with ts_utc < close_utc(d); NaT when the item is after the last close or
    earlier than one day before the first close of the calendar."""
    closes = close_utc(market, calendar)
    t = utc_seconds(ts_utc)
    pos = np.searchsorted(closes, t, side="right")
    valid = (pos < len(calendar)) & (t >= closes[0] - 86400 * 1)
    vals = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
    vals[valid] = calendar.to_numpy().astype("datetime64[ns]")[pos[valid]]
    return pd.Series(vals, index=ts_utc.index)


def utc_seconds(ts) -> np.ndarray:
    """Seconds since the epoch, independent of the datetime resolution pandas chose."""
    s = pd.to_datetime(pd.Series(ts), utc=True)
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def align_market(items: pd.DataFrame, market: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """news_aligned_<MKT>: one row per (item, tag) with tag = the market's local tag or GLOBAL."""
    local = frozen()["markets"][market]["news_tag"]
    d = assign_sessions(items["ts_utc"], market, calendar)
    ok = d.notna()
    base = pd.DataFrame({"id": items.loc[ok, "id"].to_numpy(), "trading_date": d[ok].to_numpy()})
    loc_mask = items.loc[ok, f"tag_{local}"].to_numpy()
    return pd.concat([base.assign(tag="GLOBAL"), base[loc_mask].assign(tag=local)], ignore_index=True)


def aligned_path(market: str) -> Path:
    return PROCESSED / f"news_aligned_{market}.parquet"
