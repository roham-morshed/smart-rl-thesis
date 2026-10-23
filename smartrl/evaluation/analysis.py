"""P5 analysis (E5-E12): every trading, test, H2, H5, H6, AWS, robustness, forward and E13-overlap key of
RESULTS_SCHEMA from the daily paths written by scripts/p5_confirm.py --stage evaluate.

Conventions written to the notes: COMBINED turnover is the mean of the four sleeves' turnovers and its win rate uses
days on which the average sleeve exposure is positive; regimes label the return of day t with the regime known at
the close of t−1; the daily decision flip rate is the share of days whose Buy/Hold/Sell label differs from the
previous day's.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .. import results as R
from ..config import LOGS, MARKETS, PROCESSED, RESULTS, frozen
from ..data import folds, panel
from ..env.evaluate import path_from_targets
from ..env.portfolio import RF_DAILY
from ..runs import compute_totals
from ..sentiflow import daily as D
from ..stats import trading as ST
from ..stats.classification import macro_f1, mcnemar_exact, paired_bootstrap_f1_diff
from . import confirm as C
from .metrics import METRICS, combined, max_drawdown, regime_labels, sharpe, trading_metrics

METHODS = ["SMARTRL", "B1_BH", "B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC", "B8_STATIC",
           "BASE", "A1_NOSENT", "A2_NOMM", "A3_NOAWS", "A4_NOCONF", "A5_NOCURR"]
COMPARED = ["B1_BH", "B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC", "B8_STATIC", "BASE"]
H1_SET = ["B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC"]
FORWARD_METHODS = ["SMARTRL"] + COMPARED


def load(variant: str) -> pd.DataFrame:
    df = pd.read_parquet(LOGS / f"returns_{variant}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def sleeve(df: pd.DataFrame, method: str, market: str) -> pd.DataFrame:
    return df[(df["method"] == method) & (df["market"] == market)].sort_values("date").set_index("date")


def metrics_block(df: pd.DataFrame, method: str) -> dict:
    out = {}
    sl = {m: sleeve(df, method, m) for m in MARKETS}
    for m, s in sl.items():
        if len(s):
            out[m] = trading_metrics(s["r"].to_numpy(), s["held"].to_numpy(), e0=1.0)
    present = {m: s for m, s in sl.items() if len(s)}
    if present:
        comb = combined({m: s["r"] for m, s in present.items()})
        held = pd.concat({m: s["held"] for m, s in present.items()}, axis=1).fillna(0).mean(axis=1).reindex(comb.index)
        cm = trading_metrics(comb.to_numpy(), held.to_numpy(), e0=1.0)
        cm["turnover"] = float(np.mean([out[m]["turnover"] for m in present]))
        out["COMBINED"] = cm
    return out


def combined_excess(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    cols = {}
    for meth in methods:
        cols[meth] = combined({m: sleeve(df, meth, m)["r"] for m in MARKETS})
    return pd.DataFrame(cols).dropna() - RF_DAILY


# ---------------------------------------------------------------- E5, E6, E9
def trading_and_tests(base: pd.DataFrame, items: list) -> dict:
    conf = base[base["fold"] != "forward"]
    for meth in METHODS:
        for mkt, mv in metrics_block(conf, meth).items():
            items += [(f"trading.{mkt}.{meth}.{k}", v) for k, v in mv.items()]
        for y in range(2020, 2025):
            for mkt, mv in metrics_block(conf[conf["date"].dt.year == y], meth).items():
                items += [(f"trading_year.{y}.{mkt}.{meth}.{k}", mv[k]) for k in ("sharpe", "ann_return", "mdd")]
    X = combined_excess(conf, METHODS)
    lw = {}
    for meth in METHODS[1:]:
        h = ST.lw_hac(X["SMARTRL"].to_numpy(), X[meth].to_numpy())
        bt = ST.lw_bootstrap(X["SMARTRL"].to_numpy(), X[meth].to_numpy(), frozen()["stats"]["bootstrap"]["block"],
                             frozen()["stats"]["bootstrap"]["reps"])
        lw[meth] = {"diff": h["diff"], "p_hac": h["p"], "p_boot": bt["p"]}
        items += [(f"tests.lw.{meth}.{k}", v) for k, v in lw[meth].items()]
    holm = ST.holm({m: lw[m]["p_hac"] for m in H1_SET})
    items += [(f"tests.h1.p_holm.{m}", v) for m, v in holm.items()]
    sb = frozen()["stats"]["stationary_bootstrap"]
    for m in H1_SET:
        d = ST.stationary_bootstrap_diff((X["SMARTRL"] + RF_DAILY).to_numpy(), (X[m] + RF_DAILY).to_numpy(),
                                         sb["mean_block"], sb["reps"])
        for k in ("est", "lo", "hi"):
            items += [(f"tests.h1.mdd_diff.{m}.{k}", d["mdd"][k]), (f"tests.h1.cr_diff.{m}.{k}", d["cr"][k])]
    ps = [lw[m]["p_hac"] for m in COMPARED]
    items += [("tests.h7.iut_all_significant", bool(all(p < 0.05 for p in ps))), ("tests.h7.max_p", float(max(ps)))]
    st = frozen()["stats"]
    items += [(f"tests.spa.{k}", v) for k, v in ST.spa(X["SMARTRL"].to_numpy(), X[COMPARED].to_numpy(),
                                                     st["spa"]["reps"], st["spa"]["block"]).items()]
    names = ["SMARTRL"] + COMPARED
    items.append(("tests.mcs.included", ST.mcs(-X[names].to_numpy(), names, st["mcs"]["size"], st["mcs"]["reps"],
                                               st["mcs"]["block"])))
    blocks = []
    for m in MARKETS:
        for y in range(2020, 2025):
            sub = conf[(conf["market"] == m) & (conf["date"].dt.year == y)]
            blocks.append([sharpe(sleeve(sub, n, m)["r"].to_numpy()) for n in names])
    fn = ST.friedman_nemenyi(np.array(blocks), names)
    items += [("tests.friedman.chi2", fn["chi2"]), ("tests.friedman.p", fn["p"]), ("tests.nemenyi.cd", fn["cd"])]
    items += [(f"tests.nemenyi.rank.{k}", v) for k, v in fn["rank"].items()]
    trials = [json.loads(l) for l in (RESULTS / "dev_trials.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sr_var = float(np.var([t["val_sharpe"] for t in trials], ddof=1))
    dsr = ST.deflated_sharpe((X["SMARTRL"] + RF_DAILY).to_numpy(), len(trials), sr_var, RF_DAILY)
    items += [(f"tests.dsr.{k}", dsr[k]) for k in ("sr_annual", "dsr", "n_trials", "sr_var")]
    series = {}
    for i, t in enumerate(trials):
        s = pd.concat([pd.Series(t["combined_val"]["r"], index=pd.to_datetime(t["combined_val"]["dates"])),
                       pd.Series(t["combined_devtest"]["r"], index=pd.to_datetime(t["combined_devtest"]["dates"]))])
        series[i] = s[~s.index.duplicated()]
    M = pd.DataFrame(series).sort_index().fillna(0.0).to_numpy()
    items.append(("tests.pbo.pbo", ST.pbo_cscv(M, frozen()["stats"]["pbo_blocks"])["pbo"]))
    return {"friedman": fn, "blocks": blocks}


# ---------------------------------------------------------------- E10 (H2)
def ground_truth(market: str) -> pd.Series:
    """Up / Down / Flat (2 / 0 / 1) of the forward 5-day excess total return against τ_t = 0.5 × the standard
    deviation of the 5-day excess returns known at t (windows ending by t, the previous 252 days)."""
    cfg = frozen()["decision_labels"]
    px = pd.read_parquet(panel.processed_path(market)).set_index("date")
    x1 = px["r_tr"] - RF_DAILY
    h = cfg["horizon"]
    fwd = x1[::-1].rolling(h).sum()[::-1].shift(-1)                   # Σ_{i=1..5} x_{t+i}
    past = x1.rolling(h).sum()                                        # 5-day sums ending at t
    tau = cfg["tau_multiplier"] * past.rolling(cfg["tau_lookback"], min_periods=cfg["tau_lookback"] // 2).std()
    y = pd.Series(1, index=px.index)
    y[fwd > tau] = 2
    y[fwd < -tau] = 0
    y[fwd.isna() | tau.isna()] = -1
    return y


def fold_cov(market: str) -> pd.Series:
    """cov of each test day under its own fold's training median (SPEC §3.9)."""
    parts = []
    s = pd.read_parquet(D.daily_path(market)).set_index("date")
    for name in folds.fold_names()[1:-1]:
        f = folds.make_fold(name, market)
        n = s["n_loc"].reindex(f.dates).fillna(0)
        mask = pd.Series(False, index=f.dates)
        mask.iloc[f.train[0]:f.train[1] + 1] = True
        parts.append(D.coverage(n, mask).iloc[f.test[0]:f.test[1] + 1])
    return pd.concat(parts)


def h2(base: pd.DataFrame, items: list) -> None:
    conf = base[base["fold"] != "forward"]
    pooled = {"y": [], "a": [], "b": [], "cov": []}
    label = {1: 2, -1: 0, 0: 1}                                       # Buy → Up, Sell → Down, Hold → Flat
    for m in MARKETS:
        gt = ground_truth(m)
        cov = fold_cov(m)
        sa, sb = sleeve(conf, "SMARTRL", m), sleeve(conf, "A1_NOSENT", m)
        d = pd.DataFrame({"a": sa["decision"], "b": sb["decision"], "y": gt.reindex(sa.index), "cov": cov.reindex(sa.index)})
        d = d[d["y"] >= 0].dropna()
        pa, pb = d["a"].map(label).to_numpy(int), d["b"].map(label).to_numpy(int)
        y = d["y"].to_numpy(int)
        for meth, p, raw in (("SMARTRL", pa, d["a"]), ("A1_NOSENT", pb, d["b"])):
            act = raw.to_numpy() != 0
            hit = float((p[act] == y[act]).mean()) if act.any() else float("nan")
            items += [(f"h2.{m}.{meth}.acc", float((p == y).mean())), (f"h2.{m}.{meth}.f1", macro_f1(y, p)),
                      (f"h2.{m}.{meth}.hit", hit)]
        bt = paired_bootstrap_f1_diff(y, [pa], [pb], reps=frozen()["stats"]["bootstrap"]["reps"])
        items += [(f"h2.{m}.mcnemar_p", mcnemar_exact(pa == y, pb == y)["p"]), (f"h2.{m}.f1_diff", bt["est"]),
                  (f"h2.{m}.f1_diff_lo", bt["lo"]), (f"h2.{m}.f1_diff_hi", bt["hi"])]
        for k, v in (("y", y), ("a", pa), ("b", pb), ("cov", d["cov"].to_numpy())):
            pooled[k].append(v)
        pooled.setdefault("raw_a", []).append(d["a"].to_numpy())
        pooled.setdefault("raw_b", []).append(d["b"].to_numpy())
    y, pa, pb, cov = (np.concatenate(pooled[k]) for k in ("y", "a", "b", "cov"))
    ra, rb = np.concatenate(pooled["raw_a"]), np.concatenate(pooled["raw_b"])
    for meth, p, raw in (("SMARTRL", pa, ra), ("A1_NOSENT", pb, rb)):
        act = raw != 0
        items += [(f"h2.POOLED.{meth}.acc", float((p == y).mean())), (f"h2.POOLED.{meth}.f1", macro_f1(y, p)),
                  (f"h2.POOLED.{meth}.hit", float((p[act] == y[act]).mean()) if act.any() else float("nan"))]
        for grp, mask in (("high", cov >= 0.5), ("low", cov < 0.5)):
            items += [(f"h2.coverage.{grp}.{meth}.acc", float((p[mask] == y[mask]).mean())),
                      (f"h2.coverage.{grp}.{meth}.f1", macro_f1(y[mask], p[mask]))]
    for grp, mask in (("high", cov >= 0.5), ("low", cov < 0.5)):          # descriptive; the p-value is exploratory
        items += [(f"h2.coverage.{grp}.n", int(mask.sum())),
                  (f"h2.coverage.{grp}.mcnemar_p_exploratory", mcnemar_exact(pa[mask] == y[mask], pb[mask] == y[mask])["p"])]
    bt = paired_bootstrap_f1_diff(y, [pa], [pb], reps=frozen()["stats"]["bootstrap"]["reps"])
    items += [("h2.POOLED.mcnemar_p", mcnemar_exact(pa == y, pb == y)["p"]), ("h2.POOLED.f1_diff", bt["est"]),
              ("h2.POOLED.f1_diff_lo", bt["lo"]), ("h2.POOLED.f1_diff_hi", bt["hi"])]


# ---------------------------------------------------------------- E8 (H5)
def _walk(eps: np.ndarray, s: float) -> np.ndarray:
    """Reflecting random walk on [0, 1] starting at 1 with steps s·ε."""
    e = np.empty(len(eps))
    x = 1.0
    for t, z in enumerate(eps):
        x += s * z
        x = -x if x < 0 else (2.0 - x if x > 1 else x)
        x = min(max(x, 0.0), 1.0)
        e[t] = x
    return e


def _placebo(ret: np.ndarray, eps: np.ndarray, s: float, c) -> tuple[np.ndarray, float]:
    R_, TO = np.zeros(eps.shape), np.zeros(len(eps))
    for k in range(len(eps)):
        p = path_from_targets(_walk(eps[k], s), ret, 0, len(ret) - 1, c.cost, c.band, c.initial_exposure)
        R_[k] = p["r"]
        TO[k] = np.mean(np.abs(np.diff(np.concatenate([[c.initial_exposure], p["held"]]))))
    return R_, float(TO.mean())


def placebo_paths(ret: np.ndarray, target_turnover: float, n: int, rng: np.random.Generator, c) -> np.ndarray:
    """Daily returns (n × T) of random exposure paths over the days of `ret` (index total returns), traded with the
    band and costs; the step size is set by bisection so that the mean |Δe| matches the PPO ensemble's."""
    eps = rng.normal(size=(n, len(ret)))
    lo, hi = 1e-4, 2.0
    for _ in range(16):
        mid = math.sqrt(lo * hi)
        _, to = _placebo(ret, eps[:50], mid, c)
        lo, hi = (mid, hi) if to < target_turnover else (lo, mid)
    return _placebo(ret, eps, math.sqrt(lo * hi), c)[0]


def h5(base: pd.DataFrame, items: list) -> None:
    conf = base[base["fold"] != "forward"]
    c = C.env_cfg()
    n_paths = frozen()["stats"]["placebo_paths"]
    for m in MARKETS:
        ppo, bh = sleeve(conf, "A3_NOAWS", m), sleeve(conf, "B1_BH", m)
        tr = pd.read_parquet(panel.processed_path(m)).set_index("date")["r_tr"]
        rng = np.random.default_rng(2026 + MARKETS.index(m))
        sims = []
        for y in range(2020, 2025):
            s = ppo[ppo.index.year == y]
            turnover = float(np.mean(np.abs(np.diff(np.concatenate([[c.initial_exposure], s["held"].to_numpy()])))))
            sims.append(placebo_paths(np.nan_to_num(tr.reindex(s.index).to_numpy()), max(turnover, 1e-6), n_paths, rng, c))
        placebo_sh = np.array([sharpe(r) for r in np.hstack(sims)])
        ppo_sh = sharpe(ppo["r"].to_numpy())
        items += [(f"h5.{m}.ppo_sharpe", ppo_sh), (f"h5.{m}.placebo_mean", float(placebo_sh.mean())),
                  (f"h5.{m}.placebo_p95", float(np.quantile(placebo_sh, 0.95))),
                  (f"h5.{m}.p", float((placebo_sh >= ppo_sh).mean())),
                  (f"h5.{m}.ppo_mdd", max_drawdown(ppo["r"].to_numpy())), (f"h5.{m}.bh_mdd", max_drawdown(bh["r"].to_numpy()))]


# ---------------------------------------------------------------- E7, E11 (H6)
def h6(base: pd.DataFrame, items: list) -> None:
    conf = base[base["fold"] != "forward"]
    budget = frozen()["ppo"]["total_timesteps"]
    arms = {"CURR": "SMARTRL", "PLAIN": "A5", "BC": "BC", "IMIT": "IMIT"}
    units = {k: {} for k in arms}
    for key, arm in arms.items():
        for name in folds.fold_names()[1:-1]:
            for m in MARKETS:
                thr = stm_val_sharpe(name, m)
                for s in range(3):
                    info = json.loads((C.policy_dir(name, arm, m, s) / "info.json").read_text(encoding="utf-8"))
                    cur = info["curve"]
                    vs = np.array([p["val_sharpe"] for p in cur], float)
                    steps = np.array([p["steps"] for p in cur])
                    reach = steps[np.nan_to_num(vs, nan=-np.inf) >= thr]
                    early = [p["val_mdd"] for p in cur if p["steps"] <= 0.2 * budget]
                    units[key][(name, m, s)] = {"aulc": float(np.nanmean(vs)),
                                                "steps_to_thr": float(reach.min()) if len(reach) else float(budget),
                                                "early_mdd": float(np.mean(early))}
    for key in arms:
        for met in ("aulc", "steps_to_thr", "early_mdd"):
            v = np.array([u[met] for u in units[key].values()])
            items += [(f"h6.{key}.{met}", float(v.mean())), (f"h6.{key}.{met}_sd", float(v.std(ddof=1)))]
        ens = combined({m: sleeve(conf, f"H6_{key}", m)["r"] for m in MARKETS})
        seeds = [sharpe(combined({m: sleeve(conf, f"H6_{key}_s{s}", m)["r"] for m in MARKETS}).to_numpy()) for s in range(3)]
        items += [(f"h6.{key}.test_sharpe", sharpe(ens.to_numpy())), (f"h6.{key}.test_sharpe_seed_sd", float(np.std(seeds, ddof=1)))]
    for pair in ("CURR_vs_PLAIN", "BC_vs_PLAIN", "IMIT_vs_PLAIN", "CURR_vs_IMIT"):
        a, b = pair.split("_vs_")
        ks = sorted(units[a])
        raw = {}
        for met in ("aulc", "steps_to_thr", "early_mdd"):
            x = np.array([units[a][k][met] for k in ks])
            y = np.array([units[b][k][met] for k in ks])
            r = stats.wilcoxon(x, y, zero_method="wilcox") if np.any(x != y) else None
            raw[met] = (float(r.statistic) if r else 0.0, float(r.pvalue) if r else 1.0)
        adj = ST.holm({k: v[1] for k, v in raw.items()})
        for met, (stat, p) in raw.items():
            items += [(f"h6.tests.{met}.{pair}.stat", stat), (f"h6.tests.{met}.{pair}.p", p),
                      (f"h6.tests.{met}.{pair}.p_holm", adj[met])]
    to, fl = {"SMARTRL": [], "A3_NOAWS": []}, {"SMARTRL": [], "A3_NOAWS": []}
    for m in MARKETS:
        for y in range(2020, 2025):
            for meth in to:
                s = sleeve(conf, meth, m)
                s = s[s.index.year == y]
                to[meth].append(float(np.mean(np.abs(np.diff(np.concatenate([[1.0], s["held"].to_numpy()]))))))
                dcs = s["decision"].to_numpy()
                fl[meth].append(float(np.mean(dcs[1:] != dcs[:-1])))
    for meth in to:
        items += [(f"h6.stability.{meth}.turnover", float(np.mean(to[meth]))), (f"h6.stability.{meth}.flip_rate", float(np.mean(fl[meth])))]
    items += [("h6.stability.tests.turnover_p", float(stats.wilcoxon(to["SMARTRL"], to["A3_NOAWS"]).pvalue)),
              ("h6.stability.tests.flip_p", float(stats.wilcoxon(fl["SMARTRL"], fl["A3_NOAWS"]).pvalue))]


def stm_val_sharpe(fold_name: str, market: str) -> float:
    inp = C.fold_inputs(fold_name, market)
    c = C.env_cfg()
    p = path_from_targets(inp["e_sa"], inp["ret_tr"], *inp["fold"].val, c.cost, c.band, c.initial_exposure)
    return sharpe(p["r"])


# ---------------------------------------------------------------- AWS analysis
def aws_analysis(items: list) -> None:
    allw = []
    for m in MARKETS:
        logs = [pd.read_parquet(LOGS / f"aws_daily_{m}_{y}.parquet") for y in range(2020, 2025)]
        w = pd.concat(logs)
        w["date"] = pd.to_datetime(w["date"])
        items += [(f"aws.{m}.w_rl_mean", float(w["w_rl"].mean())), (f"aws.{m}.w_rl_sd", float(w["w_rl"].std(ddof=1))),
                  (f"aws.{m}.regret", float(np.mean([l["regret"].iloc[0] for l in logs])))]
        close = pd.read_parquet(panel.processed_path(m)).set_index("date")["close"]
        reg = regime_labels(close, frozen()["regimes"]["lookback"], frozen()["regimes"]["threshold"])
        w["regime"] = reg.reindex(w["date"]).to_numpy()
        w["cov"] = fold_cov(m).reindex(w["date"]).to_numpy()
        allw.append(w)
    W = pd.concat(allw)
    for r in ("bull", "bear", "sideways"):
        items.append((f"aws.regime.{r}.w_rl_mean", float(W.loc[W["regime"] == r, "w_rl"].mean())))
    items += [("aws.coverage.high.w_rl_mean", float(W.loc[W["cov"] >= 0.5, "w_rl"].mean())),
              ("aws.coverage.low.w_rl_mean", float(W.loc[W["cov"] < 0.5, "w_rl"].mean()))]


# ---------------------------------------------------------------- E12
def robustness(base: pd.DataFrame, items: list) -> None:
    conf = base[base["fold"] != "forward"]
    for bps, var in ((0, "cost0"), (5, "cost5"), (10, "base"), (20, "cost20")):
        df = conf if var == "base" else load(var)
        for meth in METHODS:
            items.append((f"robust.cost.{bps}.{meth}.sharpe", metrics_block(df, meth)["COMBINED"]["sharpe"]))
    dl = load("delay")
    for meth in METHODS:
        mv = metrics_block(dl, meth)["COMBINED"]
        items += [(f"robust.delay.{meth}.sharpe", mv["sharpe"]), (f"robust.delay.{meth}.ann_return", mv["ann_return"])]
    regs = {}
    for m in MARKETS:
        close = pd.read_parquet(panel.processed_path(m)).set_index("date")["close"]
        regs[m] = regime_labels(close, frozen()["regimes"]["lookback"], frozen()["regimes"]["threshold"]).shift(1)
    for meth in METHODS:
        rows = []
        for m in MARKETS:
            s = sleeve(conf, meth, m)
            rows.append(pd.DataFrame({"r": s["r"].to_numpy(), "regime": regs[m].reindex(s.index).to_numpy()}))
        d = pd.concat(rows)
        for r in ("bull", "bear", "sideways"):
            x = d.loc[d["regime"] == r, "r"].to_numpy()
            ann = float(np.prod(1 + x) ** (252 / len(x)) - 1) if len(x) else float("nan")
            items += [(f"robust.regime.{r}.{meth}.sharpe", sharpe(x)), (f"robust.regime.{r}.{meth}.ann_return", ann),
                      (f"robust.regime.{r}.{meth}.days", int(len(x)))]


def forward(base: pd.DataFrame, items: list) -> None:
    fw = base[base["fold"] == "forward"]
    for meth in FORWARD_METHODS:
        for mkt, mv in metrics_block(fw, meth).items():
            items += [(f"forward.{mkt}.{meth}.{k}", v) for k, v in mv.items()]
    days = combined({m: sleeve(fw, "B1_BH", m)["r"] for m in MARKETS}).index
    items += [("forward.period.start", str(days.min().date())), ("forward.period.end", str(days.max().date())),
              ("forward.period.n_days", int(len(days)))]
    for m in MARKETS:
        s = pd.read_parquet(D.daily_path(m)).set_index("date")["n_loc"]
        idx = sleeve(fw, "B1_BH", m).index
        items.append((f"forward.news_days_share.{m}", float((s.reindex(idx).fillna(0) > 0).mean())))


def e13_overlap(base: pd.DataFrame, items: list) -> None:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    lo, hi = (pd.Timestamp(x) for x in frozen()["base"]["overlap_window"])
    conf = base[base["fold"] != "forward"]
    e13 = pd.read_parquet(PROCESSED / "e13_predictions.parquet")
    e13 = e13[(e13["protocol"] == "CAUSAL") & (e13["model"] == "SENTIN")]
    for m in MARKETS:
        c = pd.read_parquet(panel.processed_path(m)).set_index("date")["close"]
        y = (c.shift(-1) > c).astype(int)
        for meth in ("SMARTRL", "BASE"):
            s = sleeve(conf, meth, m)
            e_t = s["held"].groupby(s.index.year).shift(-1)           # exposure decided at the close of t
            d = pd.DataFrame({"score": e_t, "y": y.reindex(s.index)}).dropna()
            d = d[(d.index >= lo) & (d.index <= hi)]
            sc, yy = d["score"].to_numpy(), d["y"].to_numpy(int)
            pr = (sc > 0.5).astype(int)
            items += [(f"e13.overlap.{m}.{meth}.acc", float((pr == yy).mean())),
                      (f"e13.overlap.{m}.{meth}.f1", float(f1_score(yy, pr, zero_division=0))),
                      (f"e13.overlap.{m}.{meth}.auc", float(roc_auc_score(yy, sc)) if len(np.unique(sc)) > 1 else 0.5),
                      (f"e13.overlap.{m}.{meth}.prauc", float(average_precision_score(yy, sc))),
                      (f"e13.overlap.{m}.{meth}.up_share", float(yy.mean()))]
        g = e13[e13["market"] == m].groupby("date")["score"].mean()
        g.index = pd.to_datetime(g.index)
        g = g[(g.index >= lo) & (g.index <= hi)]
        yy = y.reindex(g.index).to_numpy(int)
        pr = (g.to_numpy() > 0.5).astype(int)
        items += [(f"e13.overlap.{m}.SENTIN_CAUSAL.acc", float((pr == yy).mean())),
                  (f"e13.overlap.{m}.SENTIN_CAUSAL.f1", float(f1_score(yy, pr, zero_division=0))),
                  (f"e13.overlap.{m}.SENTIN_CAUSAL.auc", float(roc_auc_score(yy, g.to_numpy()))),
                  (f"e13.overlap.{m}.SENTIN_CAUSAL.prauc", float(average_precision_score(yy, g.to_numpy()))),
                  (f"e13.overlap.{m}.SENTIN_CAUSAL.up_share", float(yy.mean()))]


NOTES = [
    "trading.COMBINED: equal-weight daily portfolio of the four sleeves (0 on a market's holidays); turnover = mean of the "
    "sleeves' turnovers; win_rate on days with positive average sleeve exposure. Each test year starts at e = 1 (folds).",
    "robust.regime: the return of day t carries the regime known at the close of t−1 (trailing 126-day index return).",
    "h6.stability.flip_rate: share of days whose Buy/Hold/Sell label differs from the previous day's label.",
    "h5: placebo exposure paths are reflecting random walks on [0, 1] from e = 1 with Gaussian steps whose size is set so "
    "that the mean |Δe| after the band equals the PPO ensemble's (per market-year); p = share of placebo Sharpe ≥ PPO Sharpe "
    "over 2020-2024.",
    "e13.overlap: SMARTRL and BASE score = exposure decided at the close of t (prediction 1[e_t > 0.5]); SENTIN_CAUSAL = "
    "seed-averaged P(up) of the E13 CAUSAL test predictions; window 2020-01-01 .. 2022-03-31.",
]


def subperiod(base: pd.DataFrame, items: list) -> None:
    """S1 (pre-registered 2026-10-21, frozen.yaml `supplementary.subperiod`): the main E5/E9 statistics on the
    test years with complete news coverage, under the prefix supp.sub2020_2023."""
    sp = frozen()["supplementary"]["subperiod"]
    pre = sp["keys_prefix"]
    lo, hi = (pd.Timestamp(x) for x in sp["confirmation"])
    conf = base[(base["fold"] != "forward") & (base["date"] >= lo) & (base["date"] <= hi)]
    for meth in METHODS:
        for mkt, mv in metrics_block(conf, meth).items():
            items += [(f"{pre}.trading.{mkt}.{meth}.{k}", v) for k, v in mv.items()]
    X = combined_excess(conf, METHODS)
    st = frozen()["stats"]
    lw = {}
    for meth in METHODS[1:]:
        h = ST.lw_hac(X["SMARTRL"].to_numpy(), X[meth].to_numpy())
        bt = ST.lw_bootstrap(X["SMARTRL"].to_numpy(), X[meth].to_numpy(), st["bootstrap"]["block"], st["bootstrap"]["reps"])
        lw[meth] = {"diff": h["diff"], "p_hac": h["p"], "p_boot": bt["p"]}
        items += [(f"{pre}.tests.lw.{meth}.{k}", v) for k, v in lw[meth].items()]
    items += [(f"{pre}.tests.h1.p_holm.{m}", v) for m, v in ST.holm({m: lw[m]["p_hac"] for m in H1_SET}).items()]
    ps = [lw[m]["p_hac"] for m in COMPARED]
    items += [(f"{pre}.tests.h7.iut_all_significant", bool(all(p < 0.05 for p in ps))), (f"{pre}.tests.h7.max_p", float(max(ps)))]
    trials = [json.loads(l) for l in (RESULTS / "dev_trials.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dsr = ST.deflated_sharpe((X["SMARTRL"] + RF_DAILY).to_numpy(), len(trials),
                             float(np.var([t["val_sharpe"] for t in trials], ddof=1)), RF_DAILY)
    items += [(f"{pre}.tests.dsr.{k}", dsr[k]) for k in ("sr_annual", "dsr", "n_trials", "sr_var")]


def run_all() -> None:
    base = load("base")
    items: list = []
    trading_and_tests(base, items)
    subperiod(base, items)
    h2(base, items)
    h5(base, items)
    h6(base, items)
    aws_analysis(items)
    robustness(base, items)
    forward(base, items)
    e13_overlap(base, items)
    items += [(f"compute.{k}", v) for k, v in compute_totals().items()]
    R.update(items, NOTES)
    print(f"{len(items)} keys written", flush=True)
