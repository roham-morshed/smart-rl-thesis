"""E14 news and the three daily sentiment indices (protocol §2.2).

- The multisource corpus is read from 2005-01-01: items of 2005-2006 are extracted, de-duplicated and tagged with the
  P1 code into data/processed/p6 (the P1 files from 2007 on are not changed). Item keys: uid = id (P1 items) and
  uid = 10^9 + id (2005-2006 items).
- 2005-2006 items are scored with the frozen SentiFlow seed 0 (the corpus scoring of P2), giving s, c (SentiFlow)
  and a_s (Branch A, E13's SENTIN index).
- A_PAPER scores every market-tagged headline of 2005-01-01 .. 2022-03-31 (seed-averaged): s_paper = p_pos - p_neg.
- Daily values per market: S_PAPER = mean s_paper and S_E13 = mean a_s over the local headlines of the row's date,
  S_SF = sum(c*s)/sum(c) (SentiFlow's calibrated S_loc); 0 without local news. Date = UTC calendar date (PUB
  protocols) or the session of SPEC §2.3 (CAUSAL). The SPEC §3.9 features for MarketMind come from the same rows.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import MARKETS, frozen
from ..data import news as N
from ..sentiflow import daily as D
from ..sentiflow import scoring
from .cfg import P6_DIR, p6

PARTS = P6_DIR / "news_parts"
ITEMS_NEW = P6_DIR / "news_items_2005_2006.parquet"
SCORES_NEW = P6_DIR / "news_scores_2005_2006.parquet"
APAPER_SCORES = P6_DIR / "a_paper_scores.parquet"
OFFSET = 1_000_000_000
TAG_COLS = ["tag_US", "tag_UK", "tag_IN", "tag_CN"]


def extract(log=print) -> dict:
    out = {}
    for s in frozen()["news"]["subsets_keep"]:
        out[s] = N.extract_subset(s, date_range=["2005-01-01", "2006-12-31"], parts_dir=PARTS)
        log(f"{s}: {out[s]['n_after_clean']:,} rows 2005-2006")
    out["_dedup"] = N.dedupe_and_tag(frozen()["news"]["subsets_keep"], parts_dir=PARTS, out=ITEMS_NEW)
    return out


def score_new(log=print) -> dict:
    items = pd.read_parquet(ITEMS_NEW, columns=["id", "headline"])
    res = scoring.score_corpus(items["headline"], [0], 1e9, log=log)
    df = pd.DataFrame({"id": items["id"].to_numpy(), **res["columns"]})
    df.to_parquet(SCORES_NEW, index=False)
    return {k: v for k, v in res.items() if k != "columns"}


def items_frame(cols: list[str]) -> pd.DataFrame:
    """P1 items (2007+) and P6 items (2005-2006) up to the E14 window end, with uid."""
    end = pd.Timestamp(p6()["e14"]["window"][1]) + pd.Timedelta(days=1)
    parts = []
    for path, off in ((ITEMS_NEW, OFFSET), (N.ITEMS, 0)):
        d = pd.read_parquet(path, columns=["id", "ts_utc"] + [c for c in cols if c not in ("id", "ts_utc")])
        d = d[d["ts_utc"] < end.tz_localize("UTC")]
        d["uid"] = d["id"].to_numpy() + off
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def sf_scores() -> pd.DataFrame:
    """SentiFlow seed-0 scores (s, c, b_s, b_c, a_s) indexed by uid."""
    cols = ["id", "s", "c", "b_s", "b_c", "a_s"]
    a = pd.read_parquet(SCORES_NEW, columns=cols)
    a["uid"] = a["id"] + OFFSET
    b = pd.read_parquet(scoring.SCORES, columns=cols)
    b["uid"] = b["id"]
    return pd.concat([a, b], ignore_index=True).drop(columns="id").set_index("uid")


def score_a_paper(wv, log=print) -> dict:
    """Seed-averaged A_PAPER polarity for every market-tagged headline of the window (seed 0 alone if > max hours)."""
    from . import a_paper as A
    c = p6()["e14"]
    it = items_frame(["headline"] + TAG_COLS)
    it = it[it[TAG_COLS].any(axis=1) & (it["ts_utc"] >= pd.Timestamp(c["news_start"], tz="UTC"))]
    texts = it["headline"].astype(str).tolist()
    acc = np.zeros((len(texts), 3))
    used, t_all = [], time.time()
    for k, s in enumerate(c["seeds"]):
        t0 = time.time()
        model, vocab = A.load(s, wv)
        for i in range(0, len(texts), 500_000):
            acc[i:i + 500_000] += A.predict(model, vocab, texts[i:i + 500_000])
        used.append(s)
        log(f"A_PAPER seed {s} scored {len(texts):,} headlines in {(time.time() - t0) / 60:.1f} min")
        if k == 0 and (time.time() - t0) * len(c["seeds"]) / 3600 > c["a_paper"]["max_scoring_hours"]:
            log("three-seed scoring would exceed the limit: seed 0 only")
            break
    p = acc / len(used)
    pd.DataFrame({"uid": it["uid"].to_numpy(), "s_paper": (p[:, 2] - p[:, 0]).astype(np.float32)}).to_parquet(
        APAPER_SCORES, index=False)
    return {"n_headlines": len(texts), "seeds_used": used, "hours": (time.time() - t_all) / 3600}


def counts_by_year() -> dict:
    """Local and global headline counts per market and UTC year, 2005-2022 (e14.news)."""
    it = items_frame(TAG_COLS)
    it = it[it["ts_utc"] >= pd.Timestamp(p6()["e14"]["news_start"], tz="UTC")]
    yr = it["ts_utc"].dt.year
    out = {}
    glob = yr.value_counts().sort_index()
    for m in MARKETS:
        tag = frozen()["markets"][m]["news_tag"]
        loc = yr[it[f"tag_{tag}"]].value_counts().sort_index()
        for y in range(2005, 2023):
            out[(m, y)] = {"local": int(loc.get(y, 0)), "global": int(glob.get(y, 0))}
    return out


def daily_tables(log=print) -> None:
    """Per market and grid ('cal' = UTC calendar date, 'ses' = session): S_PAPER, S_E13, S_SF and SPEC §3.9 features."""
    from . import prices as PR
    it = items_frame(TAG_COLS)
    it = it[it["ts_utc"] >= pd.Timestamp(p6()["e14"]["news_start"], tz="UTC")].reset_index(drop=True)
    sc = sf_scores()
    ap = pd.read_parquet(APAPER_SCORES).set_index("uid")["s_paper"]
    scores = sc.join(ap, how="left")
    utc_day = it["ts_utc"].dt.tz_convert(None).dt.normalize()
    for m in MARKETS:
        local = frozen()["markets"][m]["news_tag"]
        bars, _ = PR.trading_bars(m)
        for grid in ("cal", "ses"):
            g = PR.calendar_rows(bars) if grid == "cal" else PR.trading_rows(bars)
            cal = pd.DatetimeIndex(g.index)
            if grid == "cal":
                d = utc_day
            else:
                d = N.assign_sessions(it["ts_utc"], m, cal)
            ok = d.notna().to_numpy()
            base = pd.DataFrame({"id": it.loc[ok, "uid"].to_numpy(), "trading_date": d[ok].to_numpy()})
            loc_mask = it.loc[ok, f"tag_{local}"].to_numpy()
            aligned = pd.concat([base.assign(tag="GLOBAL"), base[loc_mask].assign(tag=local)], ignore_index=True)
            feats = D.build(m, aligned, scores, cal).set_index("date")
            j = base[loc_mask].join(scores[["s_paper", "a_s", "s", "c"]], on="id")
            j["cs"] = j["c"] * j["s"]
            gr = j.groupby("trading_date").agg(sp=("s_paper", "mean"), se=("a_s", "mean"), cs=("cs", "sum"), c=("c", "sum"))
            out = pd.DataFrame(index=cal)
            out["S_PAPER"] = gr["sp"].reindex(cal).fillna(0.0)
            out["S_E13"] = gr["se"].reindex(cal).fillna(0.0)
            out["S_SF"] = (gr["cs"] / gr["c"]).reindex(cal).fillna(0.0)
            out = out.join(feats[["n_loc", "n_glob", "S_loc", "S_glob", "disp_loc", "Se_loc", "Se_glob", "log_n_loc",
                                  "log_n_glob"]])
            out.index.name = "date"
            out.reset_index().to_parquet(daily_path(m, grid), index=False)
            log(f"{m} {grid}: {len(out)} rows, days with local news {(out['n_loc'] > 0).mean():.3f}")


def daily_path(market: str, grid: str):
    return P6_DIR / f"daily_{grid}_{market}.parquet"
