"""E14 rows, labels, splits, scaling and windows (protocol §2.1-2.4).

Settings: PUB_C_GE, PUB_R_GE, CAUSAL, PUB_C_GT, PUB_R_GT, PUB_C_GE_LA, PUB_R_GE_LA, PUB_C_GE_ROBUST.
- PUB_C / PUB_R: calendar rows, UTC-date news, scaling fitted on all rows of the window; PUB_C train = labelled
  rows up to 2018-10-17, test = 2018-10-18 .. 2022-03-30; PUB_R = one permutation of the 6298 dates (seed 2026),
  first 5038 train, last 1260 test.
- CAUSAL: trading days, session news, scaling fitted on the training rows, first 80 % of the samples train; label GT.
- Validation slices (only for recipes that use one): last 10 % of the training rows (PUB_C, CAUSAL) or a random 10 %
  of them (PUB_R, seed 2026).
- LA: the window for the label of row t ends at a ghost row t+1 with close = C_t (open, high, low, volume and
  sentiment of t+1 visible). ROBUST: median/IQR scaling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import MARKETS, frozen
from ..data import panel
from ..sentiflow import daily as D
from . import news as NW
from . import prices as PR
from .cfg import p6

INPUTS = ["open", "high", "low", "close", "volume"]
SENT_BASE = ["Se_loc", "Se_glob", "log_n_loc", "log_n_glob", "disp_loc"]
INDEX = {"SVM_P": "S_PAPER", "CNN_P": "S_PAPER", "LSTM_P": "S_PAPER", "GRU_P": "S_PAPER", "SENTIN_PAPER": "S_PAPER",
         "SENTIN_E13": "S_E13", "SENTIN_SF_PAPER": "S_SF", "SENTIN_SF_E13": "S_SF"}


def parse(setting: str) -> dict:
    base = "CAUSAL" if setting.startswith("CAUSAL") else setting[:5]
    label = "GT" if base == "CAUSAL" else setting[6:8]
    return {"base": base, "label": label, "la": setting.endswith("_LA"), "robust": setting.endswith("_ROBUST"),
            "grid": "ses" if base == "CAUSAL" else "cal"}


@dataclass
class MarketRows:
    market: str
    frame: pd.DataFrame          # grid rows: OHLCV, flags, price features, indices, sentiment features
    pos: np.ndarray              # grid positions of the labelled samples
    y: np.ndarray                # labels of the samples
    informative: np.ndarray      # informative_row of the samples
    train: np.ndarray            # indices into pos
    val: np.ndarray              # subset of train (empty for PUB recipes without validation)
    test: np.ndarray
    fit_mask: np.ndarray         # grid rows used to fit the scaling


def _val_slice(train: np.ndarray, base: str) -> np.ndarray:
    e = p6()["e14"]
    n_va = int(round(e["val_share"] * len(train)))
    if base == "PUB_R":
        rng = np.random.default_rng(e["random_seed"])
        return np.sort(rng.choice(train, n_va, replace=False))
    return train[-n_va:]


def market_rows(market: str, setting: str) -> MarketRows:
    e = p6()["e14"]
    s = parse(setting)
    a, b = pd.Timestamp(e["window"][0]), pd.Timestamp(e["window"][1])
    g = PR.grid(market, s["base"])
    d = pd.read_parquet(NW.daily_path(market, s["grid"])).set_index("date").reindex(g.index)
    for c in ("S_PAPER", "S_E13", "S_SF", "n_loc", "n_glob", "S_loc", "S_glob", "disp_loc", "Se_loc", "Se_glob",
              "log_n_loc", "log_n_glob"):
        d[c] = d[c].fillna(0.0)
    f = g.join(d)
    dates = f.index
    lab = PR.labels(f["close"].to_numpy(float))[s["label"]]
    if s["base"] == "CAUSAL":
        pos = np.where((dates >= a) & (dates <= b) & np.isfinite(lab))[0]
        n_tr = int(round(e["causal_train_share"] * len(pos)))
        train, test = np.arange(n_tr), np.arange(n_tr, len(pos))
        fit_mask = (dates >= a) & (np.arange(len(dates)) <= pos[train[-1]])
    else:
        pos = np.where((dates >= a) & (dates < b))[0]
        assert len(pos) == e["expected"]["labelled_rows"], (market, len(pos))
        if s["base"] == "PUB_C":
            cut = pd.Timestamp(e["pub_c"]["train_end"])
            train, test = np.where(dates[pos] <= cut)[0], np.where(dates[pos] > cut)[0]
        else:
            perm = np.random.default_rng(e["random_seed"]).permutation(len(pos))
            n_tr = e["expected"]["pub_train"]
            train, test = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        assert len(train) == e["expected"]["pub_train"] and len(test) == e["expected"]["pub_test"]
        fit_mask = (dates >= a) & (dates <= b)
    val = _val_slice(train, s["base"])
    return MarketRows(market, f, pos, lab[pos].astype(int), f["informative_row"].to_numpy(bool)[pos], train, val, test,
                      fit_mask)


def _robust_params(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        x = df[c].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x):
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            out[c] = (float(med), float(q3 - q1) if q3 > q1 else 1.0)
        else:
            out[c] = (0.0, 1.0)
    return out


def scale(df: pd.DataFrame, cols: list[str], fit_mask: np.ndarray, robust: bool) -> np.ndarray:
    """q01/q99 min-max with the SPEC clip (masked inputs -> 0), or median/IQR without clipping."""
    if robust:
        params = _robust_params(df[fit_mask], cols)
        out = pd.DataFrame({c: ((df[c] - m) / iqr).fillna(0.0) for c, (m, iqr) in params.items()})
        return out[cols].to_numpy(np.float32)
    params = panel.fit_minmax(df[fit_mask], cols)
    return panel.apply_minmax(df[cols], params, fill=0.0).to_numpy(np.float32)


def ohlcv_windows(mr: MarketRows, index: str, setting: str) -> tuple[np.ndarray, np.ndarray]:
    """Scaled [OHLCV, S] windows of all samples [N, L, 6]; the SVM uses the last `svm.window` rows."""
    s = parse(setting)
    L = p6()["e14"]["lookback"]
    df = mr.frame[INPUTS].copy()
    df["S"] = mr.frame[index]
    X = scale(df, INPUTS + ["S"], mr.fit_mask, s["robust"])
    if s["la"]:
        w = mr.pos[:, None] + np.arange(-L + 2, 1)[None, :]
        ghost = X[mr.pos + 1].copy()
        ghost[:, INPUTS.index("close")] = X[mr.pos, INPUTS.index("close")]
        return np.concatenate([X[w], ghost[:, None, :]], axis=1), X
    w = mr.pos[:, None] + np.arange(-L + 1, 1)[None, :]
    return X[w], X


def mm_arrays(mr: MarketRows, setting: str) -> tuple[np.ndarray, np.ndarray]:
    """MarketMind inputs of all samples: price [N, L, 13] and sentiment [N, L, 6] (SPEC §2.2, §3.9) on the grid."""
    s = parse(setting)
    L = p6()["e14"]["lookback"]
    f = mr.frame
    pf = f[panel.PRICE_FEATURES].copy()
    if f["volume"].isna().all():
        pf["vchg"] = np.nan
    sf = f[SENT_BASE].copy()
    sf["cov"] = D.coverage(f["n_loc"].reset_index(drop=True), pd.Series(mr.fit_mask)).to_numpy()
    Xp = scale(pf, panel.PRICE_FEATURES, mr.fit_mask, s["robust"])
    Xs = scale(sf, SENT_BASE + ["cov"], mr.fit_mask, s["robust"])
    if s["la"]:
        gf = PR.ghost_features(f)
        if f["volume"].isna().all():
            gf["vchg"] = np.nan
        if s["robust"]:
            raise NotImplementedError("LA and ROBUST are not combined in the run matrix")
        params = panel.fit_minmax(pf[mr.fit_mask], panel.PRICE_FEATURES)
        G = panel.apply_minmax(gf, params, fill=0.0).to_numpy(np.float32)
        w = mr.pos[:, None] + np.arange(-L + 2, 1)[None, :]
        xp = np.concatenate([Xp[w], G[mr.pos][:, None, :]], axis=1)
        xs = np.concatenate([Xs[w], Xs[mr.pos + 1][:, None, :]], axis=1)
        return xp, xs
    w = mr.pos[:, None] + np.arange(-L + 1, 1)[None, :]
    return Xp[w], Xs[w]


def mm_targets(mr: MarketRows) -> tuple[np.ndarray, np.ndarray]:
    """MarketMind targets of the samples on the grid: log returns over 1/5/20 rows and log RV over 5/20 rows."""
    from ..marketmind.model import HORIZONS, VOL_HORIZONS
    c = mr.frame["close"].to_numpy(float)
    r1 = np.nan_to_num(mr.frame["r1"].to_numpy(float))
    T = len(c)
    lc = np.log(c)
    y_ret = np.full((T, len(HORIZONS)), np.nan)
    for k, h in enumerate(HORIZONS):
        y_ret[:T - h, k] = lc[h:] - lc[:T - h]
    cs = np.concatenate([[0.0], np.cumsum(r1 ** 2)])
    y_rv = np.full((T, len(VOL_HORIZONS)), np.nan)
    for j, h in enumerate(VOL_HORIZONS):
        i = np.arange(T - h)
        y_rv[i, j] = (cs[i + h + 1] - cs[i + 1]) / h
    return y_ret[mr.pos], y_rv[mr.pos]


def all_markets(setting: str) -> dict[str, MarketRows]:
    return {m: market_rows(m, setting) for m in MARKETS}


def market_id(m: str) -> int:
    return list(MARKETS).index(m)


__all__ = ["parse", "market_rows", "ohlcv_windows", "mm_arrays", "mm_targets", "all_markets", "INDEX", "market_id",
           "frozen"]
