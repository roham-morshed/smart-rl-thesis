"""E15 evaluation (protocol §3.3): paths of every method on the test window, trading metrics (full window and
2020-01-01 .. 2022-03-31), direction metrics of the decisions, SMART-RL versus BASE / BASE_PAPER / B1_BH tests, and the
P5 overlap without retraining."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LOGS, MARKETS, frozen
from .cfg import P6_DIR, p6
from .e15 import E15_ARMS, patch

METHODS = ["SMARTRL", "B1_BH", "B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC", "B8_STATIC",
           "BASE", "BASE_PAPER"]
RETURNS = P6_DIR / "e15_returns.parquet"


def evaluate_market(market: str, cost_bps: float) -> list[pd.DataFrame]:
    """Test-window paths of every E15 method for one market and cost (execution at the close of t, SPEC §5)."""
    patch()
    import torch
    from ..baselines import rules
    from ..data import folds as F
    from ..env.evaluate import path_from_targets
    from ..evaluation import confirm as C
    torch.set_num_threads(1)
    inp = C.fold_inputs("e15", market)
    f, c = inp["fold"], C.env_cfg(cost_bps)
    n = len(f.dates)
    a, b = f.test
    path = lambda tgt: path_from_targets(tgt, inp["ret_tr"], a, b, c.cost, c.band, c.initial_exposure, 0)
    out = {}
    ens = {arm: C.ensemble_paths(inp, C.ARMS[arm]["state"], C.ensemble("e15", arm, market), c, 0) for arm in E15_ARMS}
    P = ens["SMARTRL"]
    e_rl = C.full_series(n, f.val[0], P["val"]["e_rl"]) + C.full_series(n, a, P["test"]["e_rl"])
    c_rl = C.full_series(n, f.val[0], P["val"]["c_rl"]) + C.full_series(n, a, P["test"]["c_rl"])
    sm, _ = C._aws(inp, e_rl, c_rl, None, c, 0)
    out["SMARTRL"] = {"r": sm["r"], "held": sm["held"]}
    out["B8_STATIC"] = path(rules.b8_static(e_rl, inp["e_sa"], frozen()["baselines"]["B8_STATIC"]["w_rl"]))
    out["B6_PPOPRICE"] = {k: ens["B6"]["test"][k] for k in ("r", "held")}
    out["B7_SAC"] = {k: ens["B7"]["test"][k] for k in ("r", "held")}
    close = inp["close"]
    out["B1_BH"] = path(rules.b1_buy_and_hold(n))
    out["B2_SMA"] = path(np.nan_to_num(rules.b2_sma(close), nan=1.0))
    out["B3_TSMOM"] = path(np.nan_to_num(rules.b3_tsmom(close), nan=1.0))
    lp = F.forecast_frame("LSTM_PRICE")
    lp = lp[lp["market"] == market].drop_duplicates("date", keep="last").set_index("date").reindex(f.dates)
    out["B4_LSTM"] = path(rules.b4_lstm(lp["q50_h5"].to_numpy()))
    out["B5_FINBERT"] = path(inp["e_b5"])
    thr = frozen()["base"]["trading_rule"]["p_up"]
    for key, var in (("BASE", "SENTIN"), ("BASE_PAPER", "SENTIN_PAPER")):
        sn = F.forecast_frame(var)
        sn = sn[sn["market"] == market].drop_duplicates("date", keep="last").set_index("date").reindex(f.dates)
        out[key] = path(rules.base_sentin(sn["p_up"].to_numpy(), thr))
    dates = f.dates[a:b + 1]
    return [pd.DataFrame({"cost_bps": cost_bps, "market": market, "method": k, "date": dates, "r": v["r"],
                          "held": v["held"]}) for k, v in out.items()]


def evaluate_all(n_jobs: int = 4) -> pd.DataFrame:
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(evaluate_market)(m, cb) for m in MARKETS
                                                    for cb in p6()["e15"]["costs_bps"])
    df = pd.concat([x for r in res for x in r], ignore_index=True)
    df.to_parquet(RETURNS, index=False)
    return df


def block(df: pd.DataFrame, method: str, lo=None, hi=None) -> dict:
    """Per-market and COMBINED metrics (SPEC §11); in a sub-window the turnover starts from the exposure held just
    before it."""
    from ..evaluation.metrics import combined, trading_metrics
    out, sl = {}, {}
    for m in MARKETS:
        s = df[(df["method"] == method) & (df["market"] == m)].sort_values("date").set_index("date")
        e0 = 1.0
        if lo is not None:
            before = s[s.index < pd.Timestamp(lo)]
            e0 = float(before["held"].iloc[-1]) if len(before) else 1.0
            s = s[(s.index >= pd.Timestamp(lo)) & (s.index <= pd.Timestamp(hi))]
        sl[m] = s
        out[m] = trading_metrics(s["r"].to_numpy(), s["held"].to_numpy(), e0=e0)
    comb = combined({m: s["r"] for m, s in sl.items()})
    held = pd.concat({m: s["held"] for m, s in sl.items()}, axis=1).fillna(0).mean(axis=1).reindex(comb.index)
    cm = trading_metrics(comb.to_numpy(), held.to_numpy(), e0=1.0)
    cm["turnover"] = float(np.mean([out[m]["turnover"] for m in MARKETS]))
    out["COMBINED"] = cm
    return out


def direction(df: pd.DataFrame) -> list:
    """Decisions as next-day direction calls: prediction 1[e_t > 0.5], score e_t (exposure decided at the close of t),
    label 1[C_{t+1} > C_t] on trading days."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from ..data import panel as P
    items, mean = [], {}
    for m in MARKETS:
        px = pd.read_parquet(P.processed_path(m)).set_index("date")["close"]
        for meth in ("SMARTRL", "BASE", "BASE_PAPER"):
            s = df[(df["method"] == meth) & (df["market"] == m)].sort_values("date").set_index("date")
            e_t = s["held"].shift(-1).iloc[:-1]
            cl = px.reindex(s.index)
            y = (cl.shift(-1) > cl).astype(int).iloc[:-1]
            sc, yy = e_t.to_numpy(float), y.to_numpy(int)
            pr = (sc > 0.5).astype(int)
            v = {"acc": float((pr == yy).mean()), "f1": float(f1_score(yy, pr, zero_division=0)),
                 "auc": float(roc_auc_score(yy, sc)) if len(np.unique(sc)) > 1 else 0.5,
                 "prauc": float(average_precision_score(yy, sc)), "up_share": float(yy.mean())}
            for k, x in v.items():
                items.append((f"e15.dir.{m}.{meth}.{k}", x))
                mean.setdefault((meth, k), []).append(x)
    items += [(f"e15.dir.MEAN.{meth}.{k}", float(np.mean(v))) for (meth, k), v in mean.items()]
    return items


def tests(df: pd.DataFrame) -> list:
    from ..data.panel import RF_DAILY
    from ..evaluation.metrics import combined
    from ..stats import trading as ST
    base = df[df["cost_bps"] == p6()["e15"]["costs_bps"][0]]
    sl = lambda meth: combined({m: base[(base["method"] == meth) & (base["market"] == m)].sort_values("date")
                                .set_index("date")["r"] for m in MARKETS})
    X = pd.DataFrame({k: sl(k) for k in ("SMARTRL", "BASE", "BASE_PAPER", "B1_BH")}).dropna() - RF_DAILY
    st = frozen()["stats"]
    items = []
    for k in ("BASE", "BASE_PAPER", "B1_BH"):
        h = ST.lw_hac(X["SMARTRL"].to_numpy(), X[k].to_numpy())
        bt = ST.lw_bootstrap(X["SMARTRL"].to_numpy(), X[k].to_numpy(), st["bootstrap"]["block"], st["bootstrap"]["reps"])
        items += [(f"e15.tests.{k}.diff", h["diff"]), (f"e15.tests.{k}.p_hac", h["p"]), (f"e15.tests.{k}.p_boot", bt["p"])]
        d = ST.stationary_bootstrap_diff((X["SMARTRL"] + RF_DAILY).to_numpy(), (X[k] + RF_DAILY).to_numpy(),
                                         st["stationary_bootstrap"]["mean_block"], st["stationary_bootstrap"]["reps"])
        for q in ("est", "lo", "hi"):
            items += [(f"e15.tests.{k}.mdd_diff.{q}", d["mdd"][q]), (f"e15.tests.{k}.cr_diff.{q}", d["cr"][q])]
    return items


def p5overlap() -> list:
    """The same trading metrics on 2020-01-01 .. 2022-03-31 from the existing P5 daily returns (no retraining)."""
    c = p6()["e15"]["sub"]
    df = pd.read_parquet(LOGS / "returns_base.parquet")
    df = df[df["fold"].isin(["conf2020", "conf2021", "conf2022"])].copy()
    df["date"] = pd.to_datetime(df["date"])
    items = []
    for meth in [m for m in METHODS if m != "BASE_PAPER"]:
        for mk, mv in block(df, meth, c[0], c[1]).items():
            items += [(f"e15.p5overlap.{mk}.{meth}.{k}", v) for k, v in mv.items()]
    return items


def summarise(df: pd.DataFrame) -> list:
    c = p6()["e15"]
    items = []
    base = df[df["cost_bps"] == c["costs_bps"][0]]
    hi = df[df["cost_bps"] == c["costs_bps"][1]]
    for meth in METHODS:
        for mk, mv in block(base, meth).items():
            items += [(f"e15.full.{mk}.{meth}.{k}", v) for k, v in mv.items()]
        for mk, mv in block(base, meth, c["sub"][0], c["sub"][1]).items():
            items += [(f"e15.sub2020.{mk}.{meth}.{k}", v) for k, v in mv.items()]
        items += [(f"e15.cost20.COMBINED.{meth}.{k}", v) for k, v in block(hi, meth)["COMBINED"].items()]
    d = sorted(base["date"].unique())
    items += [("e15.window.first_test_day", str(pd.Timestamp(d[0]).date())),
              ("e15.window.last_test_day", str(pd.Timestamp(d[-1]).date())), ("e15.window.n_days", int(len(d)))]
    return items + direction(base) + tests(df) + p5overlap()



# ---------------------------------------------------------------- outputs T23, T24, F18
def outputs() -> None:
    import matplotlib.pyplot as plt
    from .. import results as R
    from ..config import TABLES
    from ..evaluation.metrics import combined
    from ..figures import style
    t = R.load()
    g = lambda k: (lambda v: None if v is R.MISSING else v)(R.lookup(t, k))
    mets = ["sharpe", "sortino", "ann_return", "ann_vol", "mdd", "calmar", "cum_return", "turnover", "win_rate"]
    rows = []
    for win in ("full", "sub2020"):
        for mk in ["COMBINED", *MARKETS]:
            for meth in METHODS:
                rows.append({"window": win, "market": mk, "method": meth,
                             **{k: g(f"e15.{win}.{mk}.{meth}.{k}") for k in mets}})
    for meth in METHODS:
        rows.append({"window": "full_cost20", "market": "COMBINED", "method": meth,
                     **{k: g(f"e15.cost20.COMBINED.{meth}.{k}") for k in mets}})
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "T23.csv", index=False)
    rows = [{"market": mk, "method": meth, **{k: g(f"e15.dir.{mk}.{meth}.{k}") for k in ("acc", "f1", "auc", "prauc", "up_share")}}
            for mk in [*MARKETS, "MEAN"] for meth in ("SMARTRL", "BASE", "BASE_PAPER")]
    pd.DataFrame(rows).to_csv(TABLES / "T24.csv", index=False)
    df = pd.read_parquet(RETURNS)
    df = df[df["cost_bps"] == p6()["e15"]["costs_bps"][0]]
    others = [m for m in METHODS if m not in ("SMARTRL", "BASE", "BASE_PAPER", "B1_BH")]
    best = max(others, key=lambda m: g(f"e15.full.COMBINED.{m}.sharpe") or -9)
    show = ["SMARTRL", "BASE", "BASE_PAPER", "B1_BH", best]
    fig, (a1, a2) = style.figure("F18", 2, 1, height_cm=14, sharex=True)
    for meth in show:
        r = combined({m: df[(df["method"] == meth) & (df["market"] == m)].sort_values("date").set_index("date")["r"]
                      for m in MARKETS})
        w = (1 + r).cumprod()
        lab = {"BASE": "SenT-In (E13 recipe)", "BASE_PAPER": "SenT-In (paper recipe)"}.get(meth) or style.method(meth)
        lw = 1.6 if meth == "SMARTRL" else 0.9
        a1.plot(w.index, w.to_numpy(), lw=lw, label=lab)
        a2.plot(w.index, (w / w.cummax() - 1).to_numpy(), lw=lw)
    a1.set_ylabel("Growth of 1 (COMBINED)", fontsize=7)
    a2.set_ylabel("Drawdown", fontsize=7)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.legend(frameon=False, fontsize=6, ncol=3)
    style.save(fig, "F18")
    plt.close(fig)
    print("T23, T24, F18 written; best other method:", best, flush=True)
