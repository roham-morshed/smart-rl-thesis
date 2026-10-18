"""Folds of SPEC §1 and the observation features of the policies (SPEC §5).

Row 0 of every fold is the first trading day of 2010 (policies never see earlier days). Splits are inclusive index
ranges on that market's trading days:
- dev: 2010-01-01 .. 2019-12-31 split 70 / 15 / 15 (train / val / devtest);
- conf<Y>: train 2010-01-01 .. (Y−2)-12-31, val = year Y−1, test = year Y;
- forward: train .. 2023-12-31, val 2024, test 2025-01-01 .. last day <= raw_end.
Every read goes through smartrl.data.access with purpose "policy", so test-period rows (>= 2020) need the G4 lock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PROCESSED, frozen
from ..sentiflow import daily as D
from . import access
from . import panel as P

MM_COLS = [f"q{t}_h{h}" for h in (1, 5, 20) for t in (10, 50, 90)] + ["logRV5", "logRV20"]
SENT_COLS = ["Se_loc", "Se_glob", "log_n_loc", "log_n_glob", "disp_loc", "cov"]


@dataclass
class Fold:
    name: str
    market: str
    dates: pd.DatetimeIndex
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    @property
    def end(self) -> int:
        return self.test[1]


def fold_names() -> list[str]:
    return ["dev"] + [f"conf{y}" for y in frozen()["dates"]["confirmation_test_years"]] + ["forward"]


def make_fold(name: str, market: str) -> Fold:
    cfg = frozen()["dates"]
    cal = pd.DatetimeIndex(pd.read_parquet(P.processed_path(market), columns=["date"])["date"])
    cal = cal[cal >= pd.Timestamp(cfg["policy_start"])]
    if name == "dev":
        d = cal[cal <= pd.Timestamp(cfg["dev"]["end"])]
        n = len(d)
        a, b = int(round(cfg["dev"]["split"][0] * n)), int(round((cfg["dev"]["split"][0] + cfg["dev"]["split"][1]) * n))
        return Fold(name, market, d, (0, a - 1), (a, b - 1), (b, n - 1))
    if name == "forward":
        f = cfg["forward"]
        d = cal[cal <= pd.Timestamp(f["test_end_max"])]
        tr_end, val_year, ts = pd.Timestamp(f["train_end"]), f["val_year"], pd.Timestamp(f["test_start"])
    else:
        y = int(name[4:])
        d = cal[cal <= pd.Timestamp(f"{y}-12-31")]
        tr_end, val_year, ts = pd.Timestamp(f"{y - 2}-12-31"), y - 1, pd.Timestamp(f"{y}-01-01")
    i_tr = int(np.where(d <= tr_end)[0].max())
    i_v0 = int(np.where(d.year == val_year)[0].min())
    i_v1 = int(np.where(d.year == val_year)[0].max())
    i_t0 = int(np.where(d >= ts)[0].min())
    return Fold(name, market, d, (0, i_tr), (i_v0, i_v1), (i_t0, len(d) - 1))


def _check_lock(fold: Fold, upto: int | None = None) -> None:
    last = fold.dates[fold.end if upto is None else upto]
    access.check([last], "policy")


def forecast_frame(variant: str) -> pd.DataFrame:
    f = pd.read_parquet(PROCESSED / f"forecast_{variant}.parquet")
    f["date"] = pd.to_datetime(f["date"])
    return f


def fold_arrays(fold: Fold, sentiment: bool, mm: str | None, sentiment_source: str = "SF") -> dict:
    """Scaled observation features [T, F], total returns, index close and the raw daily sentiment table of the fold.
    `mm`: None, 'MM' or 'MM_NOSENT'. Scaling (q01/q99 min-max, clip) and cov use the fold's training rows only."""
    _check_lock(fold)
    m = fold.market
    px = pd.read_parquet(P.processed_path(m)).set_index("date").reindex(fold.dates)
    tr = slice(fold.train[0], fold.train[1] + 1)
    feats = px[P.PRICE_FEATURES].copy()
    if m in frozen()["features"].get("volume_disabled", []):
        feats["vchg"] = np.nan
    cols = list(P.PRICE_FEATURES)
    sent = None
    if D.daily_path(m).exists():
        sent = pd.read_parquet(D.daily_path(m)).set_index("date").reindex(fold.dates)
        train_mask = pd.Series(False, index=fold.dates)
        train_mask.iloc[tr] = True
        sent["cov"] = D.coverage(sent["n_loc"].fillna(0), train_mask)
    if sentiment:
        if sent is None:
            raise FileNotFoundError(f"sentiment features of {m} missing (P2)")
        src = sent if sentiment_source == "SF" else sent.assign(Se_loc=sent["Se_loc_B"], Se_glob=sent["Se_glob_B"])
        feats[SENT_COLS] = src[SENT_COLS].to_numpy()
        cols += SENT_COLS
    if mm:
        f = forecast_frame(mm)
        f = f[f["market"] == m].drop_duplicates("date", keep="last").set_index("date").reindex(fold.dates)
        feats[MM_COLS] = f[MM_COLS].to_numpy()
        cols += MM_COLS
    params = P.fit_minmax(feats.iloc[tr], cols)
    X = P.apply_minmax(feats[cols], params, fill=0.0).to_numpy(np.float32)
    return {"X": X, "cols": cols, "ret_tr": px["r_tr"].to_numpy(float), "close": px["close"].to_numpy(float),
            "sent": sent, "dates": fold.dates}
