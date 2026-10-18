"""Daily sentiment features per market (SPEC §3.9) and the SENTIN daily index (§9.1).

Stored per market on its own calendar (data/processed/sentiment_daily_<MKT>.parquet):
n_loc, n_glob, S_loc, S_glob, disp_loc, Se_loc, Se_glob, cbar (from calibrated SentiFlow);
the same S/Se columns from calibrated Branch B with suffix _B (baseline B5); sentin_S = mean of the uncalibrated
Branch-A polarity over the local headlines (0 when there is none).
`cov` depends on the fold (median daily n_loc over the fold's training dates) and is computed by `coverage`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PROCESSED, frozen

FEATURES = ["Se_loc", "Se_glob", "log_n_loc", "log_n_glob", "disp_loc", "cov"]


def daily_path(market: str):
    return PROCESSED / f"sentiment_daily_{market}.parquet"


def _ewma_carry(S: np.ndarray, has: np.ndarray, half_life: float) -> np.ndarray:
    """Exponentially weighted S with a half-life in trading days; on days without news the value is carried."""
    lam = 2.0 ** (-1.0 / half_life)
    out = np.zeros(len(S))
    prev = 0.0
    for t in range(len(S)):
        if has[t]:
            prev = lam * prev + (1 - lam) * S[t]
        out[t] = prev
    return out


def build(market: str, aligned: pd.DataFrame, scores: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """aligned: id, trading_date, tag; scores: indexed by id with s, c, b_s, b_c, a_s."""
    local = frozen()["markets"][market]["news_tag"]
    hl = frozen()["news"]["ewma_half_life_days"]
    j = aligned.join(scores[["s", "c", "b_s", "b_c", "a_s"]], on="id")
    j["cs"], j["b_cs"] = j["c"] * j["s"], j["b_c"] * j["b_s"]
    g = j.groupby(["tag", "trading_date"]).agg(n=("s", "size"), cs=("cs", "sum"), c=("c", "sum"), sd=("s", "std"),
                                               cmean=("c", "mean"), b_cs=("b_cs", "sum"), b_c=("b_c", "sum"),
                                               a_mean=("a_s", "mean"))
    out = pd.DataFrame(index=calendar)
    for tag, sfx in ((local, "loc"), ("GLOBAL", "glob")):
        t = g.xs(tag, level="tag").reindex(calendar) if tag in g.index.get_level_values("tag") else pd.DataFrame(index=calendar)
        n = t.get("n", pd.Series(0, index=calendar)).fillna(0).astype(int)
        out[f"n_{sfx}"] = n
        out[f"S_{sfx}"] = (t["cs"] / t["c"]).where(n > 0, 0.0).fillna(0.0) if "cs" in t else 0.0
        out[f"S_{sfx}_B"] = (t["b_cs"] / t["b_c"]).where(n > 0, 0.0).fillna(0.0) if "b_cs" in t else 0.0
        out[f"cmean_{sfx}"] = t["cmean"].fillna(0.0) if "cmean" in t else 0.0
        if sfx == "loc":
            out["disp_loc"] = t["sd"].where(n >= 2, 0.0).fillna(0.0) if "sd" in t else 0.0
            out["sentin_S"] = t["a_mean"].where(n > 0, 0.0).fillna(0.0) if "a_mean" in t else 0.0
        for src in ("", "_B"):
            out[f"Se_{sfx}{src}"] = _ewma_carry(out[f"S_{sfx}{src}"].to_numpy(), (n > 0).to_numpy(), hl)
    out["cbar"] = np.where(out["n_loc"] > 0, out["cmean_loc"], np.where(out["n_glob"] > 0, out["cmean_glob"], 0.0))
    out["log_n_loc"], out["log_n_glob"] = np.log1p(out["n_loc"]), np.log1p(out["n_glob"])
    out.index.name = "date"
    return out.reset_index()


def coverage(n_loc: pd.Series, train_mask: pd.Series) -> pd.Series:
    """cov = min(1, n_loc / median daily n_loc over the fold's training dates) (SPEC §3.9)."""
    med = float(np.median(n_loc[train_mask])) if train_mask.any() else 0.0
    if med <= 0:
        return (n_loc > 0).astype(float)
    return np.minimum(1.0, n_loc / med)
