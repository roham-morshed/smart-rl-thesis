"""E15: trading in the base paper's test window (protocol §3).

One split per market (no refits inside the test window): train 2010-01-01 .. 2017-10-17, validation 2017-10-18 ..
2018-10-17, test 2018-10-18 .. 2022-03-31. The P5 code runs unchanged on the fold name "e15": `patch()` adds the fold
and the E15 forecast files to smartrl.data.folds inside every E15 process.

Forecast features: the existing annual refits for dates before 2018-10-18, and for the test window one fit on data
before 2018-10-18 (MM, LSTM_PRICE; seed 0). BASE: SENTIN (E13 recipe) fitted once on trading days before 2018-10-18,
its last year for early stopping. BASE_PAPER: SENTIN_PAPER (Table 3 recipe, S_PAPER index) fitted once on the same
trading days, no validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import MARKETS, frozen
from .cfg import P6_DIR, p6

FORECASTS = ("MM", "LSTM_PRICE", "SENTIN", "SENTIN_PAPER")


def forecast_path(variant: str):
    return P6_DIR / f"e15_forecast_{variant}.parquet"


def e15_fold(market: str):
    from ..data import folds as F
    from ..data import panel as P
    c = p6()["e15"]
    cal = pd.DatetimeIndex(pd.read_parquet(P.processed_path(market), columns=["date"])["date"])
    d = cal[(cal >= pd.Timestamp(c["train"][0])) & (cal <= pd.Timestamp(c["test"][1]))]
    i_tr = int(np.where(d <= pd.Timestamp(c["train"][1]))[0].max())
    i_v0 = int(np.where(d >= pd.Timestamp(c["val"][0]))[0].min())
    i_v1 = int(np.where(d <= pd.Timestamp(c["val"][1]))[0].max())
    i_t0 = int(np.where(d >= pd.Timestamp(c["test"][0]))[0].min())
    return F.Fold("e15", market, d, (0, i_tr), (i_v0, i_v1), (i_t0, len(d) - 1))


def e15_forecast(variant: str) -> pd.DataFrame:
    """Existing annual refits before the test window + the single E15 fit inside it (MM, LSTM_PRICE); the E15 fits
    alone for SENTIN and SENTIN_PAPER; MM_NOSENT (no E15 method uses it) = the annual refits before the window."""
    from ..data import folds as F
    t0 = pd.Timestamp(p6()["e15"]["test"][0])
    if variant in ("SENTIN", "SENTIN_PAPER"):
        f = pd.read_parquet(forecast_path(variant))
    else:
        old = F._orig_forecast_frame(variant)
        old = old[old["date"] < t0]
        parts = [old]
        if forecast_path(variant).exists():
            parts.append(pd.read_parquet(forecast_path(variant)))
        f = pd.concat(parts, ignore_index=True)
    f["date"] = pd.to_datetime(f["date"])
    return f


def patch() -> None:
    from ..data import folds as F
    if getattr(F, "_e15_patched", False):
        return
    F._orig_make_fold, F._orig_forecast_frame = F.make_fold, F.forecast_frame
    F.make_fold = lambda name, market: e15_fold(market) if name == "e15" else F._orig_make_fold(name, market)
    F.forecast_frame = e15_forecast
    F._e15_patched = True


# ---------------------------------------------------------------- forecasts
def fit_forecasts(log=print) -> None:
    import sys
    from ..config import ROOT
    sys.path.insert(0, str(ROOT / "scripts"))
    import p3_marketmind as PM3
    from ..marketmind import data as MD
    from .cfg import p6_run
    c = p6()["e15"]
    mm = frozen()["marketmind"]
    te, pred = c["val"][1], (c["test"][0], c["test"][1])
    panels = MD.load_panels()
    cache: dict = {}
    for v in ("MM", "LSTM_PRICE"):
        if forecast_path(v).exists():
            continue
        with p6_run(f"E15_forecast_{v}", 0, "cuda", {"train_end": te}):
            df = PM3.run_one(v, 0, panels, te, pred, mm, cache, log)
        df.drop(columns=["variant", "seed", "epochs", "best_epoch"], errors="ignore").to_parquet(forecast_path(v), index=False)
    if not forecast_path("SENTIN").exists():
        _fit_base(log)
    if not forecast_path("SENTIN_PAPER").exists():
        from . import news as NW
        if all(NW.daily_path(m, "ses").exists() for m in MARKETS):
            _fit_base_paper(log)
        else:
            log("BASE_PAPER waits for the E14 daily indices (python scripts/p6_e14.py --stage daily)")


def _sentin_rows_split(dates: pd.DatetimeIndex, L: int):
    c = p6()["e15"]
    last_tr = int(np.where(dates <= pd.Timestamp(c["val"][1]))[0].max())
    idx = np.arange(L - 1, last_tr)                         # the label of t uses C_{t+1} <= the training end
    pi = np.where((dates >= pd.Timestamp(c["test"][0])) & (dates <= pd.Timestamp(c["test"][1])))[0]
    return last_tr, idx, pi


def _fit_base(log) -> None:
    from ..data import panel
    from ..e13 import data as E13D
    from ..e13 import models as EM
    from ..e13 import train as ET
    from .cfg import p6_run
    cfg = frozen()
    L = cfg["sentin"]["lookback"]
    start = panel.first_common_date(cfg["dates"]["raw_end"])
    out = []
    for m in MARKETS:
        r = E13D.rows(m, "CAUSAL", "A", start, p6()["e15"]["test"][1])
        dates = r.index
        last_tr, idx, pi = _sentin_rows_split(dates, L)
        va = idx[dates[idx] >= pd.Timestamp(p6()["e15"]["val"][0])]        # its last year for early stopping
        tr = idx[dates[idx] < pd.Timestamp(p6()["e15"]["val"][0])]
        X = E13D.scaled_matrix(r, np.arange(0, last_tr + 1))
        y = r["y"].to_numpy()
        with p6_run(f"E15_BASE_SENTIN_{m}", 0, str(ET.DEV)):
            model, info = ET.fit(EM.build("SENTIN", cfg), E13D.windows(X, tr, L), y[tr].astype(int),
                                 E13D.windows(X, va, L), y[va].astype(int), cfg["sentin"]["optimiser_choice"], 0, cfg)
        out.append(pd.DataFrame({"market": m, "date": dates[pi], "p_up": ET.predict(model, E13D.windows(X, pi, L))}))
        log(f"BASE {m}: {info}")
    pd.concat(out, ignore_index=True).to_parquet(forecast_path("SENTIN"), index=False)


def _fit_base_paper(log) -> None:
    from ..data import panel
    from ..e13 import data as E13D
    from . import e14data as ED
    from . import models as PM
    from . import news as NW
    from . import prices as PR
    from .cfg import p6_run
    from .e14run import DEV, fit_sentin_paper, predict_prob
    from ..sentiflow.train import set_seed
    cfg = frozen()
    L = p6()["e14"]["lookback"]
    start = pd.Timestamp(panel.first_common_date(cfg["dates"]["raw_end"]))
    out = []
    for m in MARKETS:
        g = PR.grid(m, "CAUSAL")
        d = pd.read_parquet(NW.daily_path(m, "ses")).set_index("date").reindex(g.index)
        g = g[(g.index >= start) & (g.index <= pd.Timestamp(p6()["e15"]["test"][1]))]
        df = g[ED.INPUTS].copy()
        df["S"] = d["S_PAPER"].reindex(g.index).fillna(0.0)
        dates = df.index
        c = df["close"].to_numpy(float)
        y = np.full(len(c), np.nan)
        y[:-1] = (c[1:] > c[:-1]).astype(float)
        last_tr, idx, pi = _sentin_rows_split(dates, L)
        fit = np.zeros(len(df), bool)
        fit[: last_tr + 1] = True
        X = ED.scale(df, ED.INPUTS + ["S"], fit, robust=False)
        set_seed(0)
        with p6_run(f"E15_BASE_PAPER_{m}", 0, str(DEV)):
            net, info = fit_sentin_paper(PM.build("SENTIN_PAPER", cfg, p6()), E13D.windows(X, idx, L), y[idx].astype(int), 0)
        out.append(pd.DataFrame({"market": m, "date": dates[pi], "p_up": predict_prob(net, E13D.windows(X, pi, L))}))
        log(f"BASE_PAPER {m}: {info}")
    pd.concat(out, ignore_index=True).to_parquet(forecast_path("SENTIN_PAPER"), index=False)


# ---------------------------------------------------------------- training (P5 code, fold "e15")
E15_ARMS = ("SMARTRL", "B6", "B7")


def train_job(arm: str, market: str, seed: int) -> dict:
    patch()
    from ..evaluation import confirm as C
    return C.train_job("e15", arm, market, seed)


def jobs(sac: bool) -> list[tuple[str, str, int]]:
    from ..evaluation import confirm as C
    arms = ["B7"] if sac else ["SMARTRL", "B6"]
    return [(a, m, s) for a in arms for m in MARKETS for s in range(C.ARMS[a]["seeds"])]
