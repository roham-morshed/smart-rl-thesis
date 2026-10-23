"""Tables (results/tables/<ID>.csv) and figures (results/figures/<ID>.png|pdf) of RESULTS_SCHEMA, built only from
results.json and the detail files written next to it. A table or figure whose inputs do not exist yet is skipped."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import results as R
from ..config import MARKETS, RESULTS, TABLES
from ..figures import style

DS = ["phrasebank50", "phrasebank_all", "fiqa", "twitter", "pooled"]
SM = ["LEX", "A", "A_REV", "B", "SF", "PROSUS"]
FM = ["MM", "LSTM", "GRU", "INFORMER", "PATCHTST", "DLINEAR", "NAIVE"]
METHODS = ["SMARTRL", "B1_BH", "B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC", "B8_STATIC", "BASE"]
ABL = ["SMARTRL", "A1_NOSENT", "A2_NOMM", "A3_NOAWS", "A4_NOCONF", "A5_NOCURR"]


def _save(df: pd.DataFrame, tid: str) -> str:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{tid}.csv", index=False)
    return tid


def _get(tree, key, default=None):
    v = R.lookup(tree, key)
    return default if v is R.MISSING else v


def _detail(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- tables
def t03(res):
    if "sentiflow" not in res:
        return None
    rows = []
    for ds in DS:
        for sm in SM:
            m = _get(res, f"sentiflow.{ds}.{sm}")
            if isinstance(m, dict):
                rows.append({"dataset": ds, "model": sm, **m})
    return _save(pd.DataFrame(rows), "T03")


def t04(res):
    if _get(res, "sentiflow.calibration") is None:
        return None
    rows = [{"block": "E1-b", "dataset": ds, "model": sm, **_get(res, f"sentiflow.e1b.{ds}.{sm}", {})}
            for ds in ("phrasebank_all", "phrasebank50") for sm in ("B", "SF")]
    for blk, sm in (("calibration", "SF"), ("calibration_b", "B")):
        rows.append({"block": "calibration (pooled test)", "dataset": "pooled", "model": sm, **_get(res, f"sentiflow.{blk}", {})})
    return _save(pd.DataFrame(rows), "T04")


def t05(res):
    if _get(res, "marketmind.h20") is None:
        return None
    rows = []
    for h in (1, 5, 20):
        for fm in FM:
            m = _get(res, f"marketmind.h{h}.{fm}")
            if isinstance(m, dict):
                rows.append({"horizon": h, "model": fm, **m, "dm_stat_vs_MM": _get(res, f"marketmind.dm.h{h}.{fm}.stat"),
                             "dm_p_vs_MM": _get(res, f"marketmind.dm.h{h}.{fm}.p")})
    for h in (5, 20):
        for fm in FM[:-1] + ["EWMA", "HAR"]:
            m = _get(res, f"marketmind.vol.h{h}.{fm}")
            if isinstance(m, dict):
                rows.append({"horizon": f"vol h{h}", "model": fm, **m})
    return _save(pd.DataFrame(rows), "T05")


def t06(res):
    if "ltsf" not in res:
        return None
    rows = [{"dataset": ds, "horizon": int(h[1:]), **v} for ds, hs in res["ltsf"].items() for h, v in hs.items()]
    return _save(pd.DataFrame(rows), "T06")


def t07(res):
    if "dev" not in res:
        return None
    rows = []
    for fam in ("ppo", "imit", "stm", "b5", "aws"):
        for cfg, v in (_get(res, f"dev.{fam}") or {}).items():
            rows.append({"family": fam, "config": cfg, **v})
    for k, v in (_get(res, "dev.selected") or {}).items():
        rows.append({"family": "selected", "config": k, "value": v})
    return _save(pd.DataFrame(rows), "T07")


def t20(res):
    if "e13" not in res:
        return None
    rows = []
    for proto in ("PUB_R", "PUB_C", "CAUSAL"):
        for m in list(MARKETS) + ["MEAN"]:
            block = _get(res, f"e13.{proto}.{m}") or {}
            meta = {k: v for k, v in block.items() if not isinstance(v, dict)}
            for model, v in block.items():
                if isinstance(v, dict):
                    rows.append({"protocol": proto, "market": m, "model": model, **v, **meta})
    for m, block in (_get(res, "e13.overlap") or {}).items():
        for model, v in block.items():
            rows.append({"protocol": "overlap 2020-01..2022-03", "market": m, "model": model, **v})
    return _save(pd.DataFrame(rows), "T20")


# ---------------------------------------------------------------- figures
def f02(res):
    d = _detail("p2_e1_detail.json")
    if not d:
        return None
    rel = d["detail"]["reliability"]["calibration"]
    fig, ax = style.figure("F02")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label=style.label("Perfect calibration", "کالیبراسیون کامل (Perfect)"))
    for key, lab, mk in (("before", ("Before temperature scaling", "پیش از مقیاس‌دهی دمایی (Before)"), "o"),
                         ("after", ("After temperature scaling", "پس از مقیاس‌دهی دمایی (After)"), "s")):
        b = [x for x in rel[key]["bins"] if x["n"] > 0]
        ax.plot([x["conf"] for x in b], [x["acc"] for x in b], marker=mk, ms=4, lw=1.5, label=style.label(*lab))
    ax.set_xlabel(style.label("Confidence (max probability)", "اطمینان (Confidence)"))
    ax.set_ylabel(style.label("Accuracy", "دقت (Accuracy)"))
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    return style.save(fig, "F02")


def f03(res):
    d = _detail("p2_e1_detail.json")
    if not d:
        return None
    hist = d["detail"]["alpha_hist"]
    fig, axes = style.figure("F03", 1, len(hist), sharey=True)
    edges = np.linspace(0, 1, 21)
    for ax, (ds, h) in zip(np.atleast_1d(axes), hist.items()):
        h = np.array(h, float)
        ax.bar(edges[:-1], h / h.sum(), width=0.05, align="edge", color="C0")
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("α")
    np.atleast_1d(axes)[0].set_ylabel(style.label("Share of sentences", "سهم جمله‌ها (Share)"))
    return style.save(fig, "F03")


def f04(res):
    from ..sentiflow import daily as D
    if not all(D.daily_path(m).exists() for m in MARKETS):
        return None
    fig, axes = style.figure("F04", 4, 1, sharex=True, height_cm=14)
    for ax, m in zip(axes, MARKETS):
        d = pd.read_parquet(D.daily_path(m)).set_index("date").loc["2010":"2025"]
        ax.plot(d.index, d["Se_glob"], lw=0.6, color="0.55", label=style.label("Global", "سراسری (Global)"))
        ax.plot(d.index, d["Se_loc"], lw=0.7, color="C0", label=style.label("Local", "محلی (Local)"))
        ax.set_ylabel(style.market(m), fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left")
    return style.save(fig, "F04")


def _bars(fig_id, res, key_fmt, ylabel, ref=None):
    fig, ax = style.figure(fig_id)
    w = 0.11
    for j, fm in enumerate(FM):
        vals = [_get(res, key_fmt.format(h=h, fm=fm)) for h in (1, 5, 20)]
        ax.bar(np.arange(3) + (j - 3) * w, [np.nan if v is None else v for v in vals], width=w, label=fm)
    if ref is not None:
        ax.axhline(ref, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(3), ["h = 1", "h = 5", "h = 20"])
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=4, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    return style.save(fig, fig_id)


def f05(res):
    if _get(res, "marketmind.h20") is None:
        return None
    return _bars("F05", res, "marketmind.h{h}.{fm}.pinball", style.label("Pinball loss (test 2020-2024)", "زیان پین‌بال (Pinball)"))


def f06(res):
    if _get(res, "marketmind.h20") is None:
        return None
    return _bars("F06", res, "marketmind.h{h}.{fm}.cov80", style.label("Coverage of [q10, q90]", "پوشش بازه (Coverage)"), ref=0.8)


def f15(res):
    if "e13" not in res:
        return None
    fig, axes = style.figure("F15", 1, 2, sharey=True)
    protos = ["PUB_R", "PUB_C", "CAUSAL"]
    for ax, met, lab in ((axes[0], "acc", "Accuracy"), (axes[1], "auc", "AUC-ROC")):
        for j, p in enumerate(protos):
            vals = [_get(res, f"e13.{p}.{m}.SENTIN.{met}") for m in MARKETS]
            ax.bar(np.arange(4) + (j - 1) * 0.27, [np.nan if v is None else v for v in vals], width=0.27, label=p)
            if met == "acc":
                ups = [_get(res, f"e13.{p}.{m}.up_share") for m in MARKETS]
                ax.scatter(np.arange(4) + (j - 1) * 0.27, [np.nan if v is None else v for v in ups], marker="_", s=120,
                           color="k", zorder=3)
        ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
        ax.set_xticks(range(4), [style.market(m) if m != "SSE" else "SSE" for m in MARKETS], fontsize=7)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    return style.save(fig, "F15")


def f16(res):
    """E13x: accuracy and AUC of SENTIN (mean of the four markets) along the leakage-removal chain."""
    if "e13x" not in res or "e13" not in res:
        return None
    chain = [("PUB_R", "e13"), ("PUB_C", "e13"), ("X_TRD_UTC", "e13x"), ("X_TRD_SES", "e13x"), ("CAUSAL", "e13")]
    labels = ["Published\n(random split)", "Chronological\nsplit", "Trading days\n(no forward-fill)",
              "Session-aligned\nnews", "Causal\n(train scaling)"]
    acc = [_get(res, f"{src}.{p}.MEAN.SENTIN.acc") for p, src in chain]
    auc = [_get(res, f"{src}.{p}.MEAN.SENTIN.auc") for p, src in chain]
    ups = [np.mean([_get(res, f"{src}.{p}.{m}.up_share") for m in MARKETS]) for p, src in chain]
    maj = [max(u, 1 - u) for u in ups]
    fig, ax = style.figure("F16")
    x = np.arange(len(chain))
    ax.plot(x, acc, marker="o", lw=1.8, label=style.label("SenT-In accuracy", "دقت (Accuracy)"))
    ax.plot(x, auc, marker="s", lw=1.8, label=style.label("SenT-In AUC-ROC", "AUC-ROC"))
    ax.plot(x, maj, ls="--", color="0.4", lw=1.2, label=style.label("Majority-class accuracy", "دقت دسته‌ی غالب (Majority)"))
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    pub = 0.8731
    ax.axhline(pub, color="C3", lw=1.0, ls="-.", label=style.label("Published accuracy (mean of 4 indices)",
                                                                    "دقت منتشرشده (Published)"))
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel(style.label("Mean over the four indices", "میانگین چهار شاخص (Mean)"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    return style.save(fig, "F16")


# ---------------------------------------------------------------- P5 tables
from ..config import LOGS  # noqa: E402

ALL_METHODS = METHODS + ABL[1:]
METRIC_COLS = ["cum_return", "ann_return", "ann_vol", "sharpe", "sortino", "mdd", "calmar", "turnover", "win_rate"]


def _rows_methods(res, prefix, methods, mkts=("COMBINED",), cols=METRIC_COLS):
    rows = []
    for mkt in mkts:
        for m in methods:
            v = _get(res, f"{prefix}.{mkt}.{m}")
            if isinstance(v, dict):
                rows.append({"market": mkt, "method": m, **{c: v.get(c) for c in cols}})
    return rows


def t08(res):
    rows = _rows_methods(res, "trading", METHODS)
    return _save(pd.DataFrame(rows), "T08") if rows else None


def t09(res):
    rows = _rows_methods(res, "trading", METHODS, MARKETS + ("COMBINED",), ["sharpe", "mdd", "ann_return"])
    return _save(pd.DataFrame(rows), "T09") if rows else None


def t10(res):
    rows = []
    for y, by in (_get(res, "trading_year") or {}).items():
        for mkt, bm in by.items():
            for m, v in bm.items():
                rows.append({"year": y, "market": mkt, "method": m, **v})
    return _save(pd.DataFrame(rows), "T10") if rows else None


def t11(res):
    rows = _rows_methods(res, "trading", ABL)
    for r in rows:
        if r["method"] != "SMARTRL":
            r.update({f"lw_{k}": _get(res, f"tests.lw.{r['method']}.{k}") for k in ("diff", "p_hac", "p_boot")})
    return _save(pd.DataFrame(rows), "T11") if rows else None


def t12(res):
    if "h6" not in res:
        return None
    rows = [{"arm": a, **{k: v for k, v in res["h6"][a].items()}} for a in ("CURR", "PLAIN", "BC", "IMIT") if a in res["h6"]]
    for met, pairs in res["h6"].get("tests", {}).items():
        for pair, v in pairs.items():
            rows.append({"arm": f"test {met} {pair}", **v})
    for m, v in res["h6"].get("stability", {}).items():
        rows.append({"arm": f"stability {m}", **v})
    return _save(pd.DataFrame(rows), "T12")


def t13(res):
    if "h5" not in res:
        return None
    return _save(pd.DataFrame([{"market": m, **v} for m, v in res["h5"].items()]), "T13")


def t14(res):
    if "tests" not in res:
        return None
    t = res["tests"]
    rows = [{"test": "LW Sharpe difference", "method": m, **v} for m, v in t.get("lw", {}).items()]
    rows += [{"test": "H1 Holm p", "method": m, "p": v} for m, v in t.get("h1", {}).get("p_holm", {}).items()]
    for kind in ("mdd_diff", "cr_diff"):
        rows += [{"test": f"H1 {kind}", "method": m, **v} for m, v in t.get("h1", {}).get(kind, {}).items()]
    rows.append({"test": "H7 IUT", "value": t.get("h7", {}).get("iut_all_significant"), "p": t.get("h7", {}).get("max_p")})
    rows.append({"test": "SPA", **t.get("spa", {})})
    rows.append({"test": "MCS 90%", "value": ", ".join(t.get("mcs", {}).get("included", []))})
    rows.append({"test": "Friedman", **t.get("friedman", {})})
    rows.append({"test": "Nemenyi CD", "value": t.get("nemenyi", {}).get("cd")})
    rows += [{"test": "Nemenyi mean rank", "method": m, "value": v} for m, v in t.get("nemenyi", {}).get("rank", {}).items()]
    rows.append({"test": "DSR", **t.get("dsr", {})})
    rows.append({"test": "PBO", **t.get("pbo", {})})
    return _save(pd.DataFrame(rows), "T14")


def t15(res):
    if "h2" not in res:
        return None
    rows = []
    for k, v in res["h2"].items():
        if k == "coverage":
            for grp, gv in v.items():
                rows += [{"block": f"coverage {grp}", "method": m, **mv} for m, mv in gv.items()]
        else:
            rows += [{"block": k, "method": m, **mv} for m, mv in v.items() if isinstance(mv, dict)]
            rows.append({"block": k, "method": "test", **{a: b for a, b in v.items() if not isinstance(b, dict)}})
    return _save(pd.DataFrame(rows), "T15")


def t16(res):
    if "aws" not in res:
        return None
    rows = [{"block": "market", "key": m, **v} for m, v in res["aws"].items() if m in MARKETS]
    rows += [{"block": "regime", "key": r, **v} for r, v in res["aws"].get("regime", {}).items()]
    rows += [{"block": "coverage", "key": c, **v} for c, v in res["aws"].get("coverage", {}).items()]
    rows += [{"block": "stability", "key": k, **v} for k, v in _get(res, "h6.stability", {}).items()]
    return _save(pd.DataFrame(rows), "T16")


def t17(res):
    if "robust" not in res:
        return None
    rows = [{"variant": f"cost {c} bps", "method": m, **v} for c, b in res["robust"]["cost"].items() for m, v in b.items()]
    rows += [{"variant": "execution t+1", "method": m, **v} for m, v in res["robust"]["delay"].items()]
    return _save(pd.DataFrame(rows), "T17")


def t18(res):
    if "robust" not in res:
        return None
    rows = [{"regime": r, "method": m, **v} for r, b in res["robust"]["regime"].items() for m, v in b.items()]
    return _save(pd.DataFrame(rows), "T18")


def t19(res):
    if "forward" not in res:
        return None
    rows = _rows_methods(res, "forward", METHODS, MARKETS + ("COMBINED",))
    meta = {**res["forward"].get("period", {}), **{f"news_share_{m}": v for m, v in res["forward"].get("news_days_share", {}).items()}}
    rows.append({"market": "period", "method": json.dumps(meta)})
    return _save(pd.DataFrame(rows), "T19")


def ts1(res):
    """Supplementary S1 (pre-registered): 2020-2023 sub-period."""
    sub = _get(res, "supp.sub2020_2023")
    if not isinstance(sub, dict):
        return None
    rows = _rows_methods(sub, "trading", ALL_METHODS)
    for r in rows:
        if r["method"] != "SMARTRL":
            r.update({f"lw_{k}": _get(sub, f"tests.lw.{r['method']}.{k}") for k in ("diff", "p_hac", "p_boot")})
    return _save(pd.DataFrame(rows), "TS1")


# ---------------------------------------------------------------- P5 figures
def _returns(variant="base"):
    p = LOGS / f"returns_{variant}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _series(df, method, market):
    s = df[(df["method"] == method) & (df["market"] == market)].sort_values("date").set_index("date")["r"]
    return s


def _comb(df, method):
    from .metrics import combined
    return combined({m: _series(df, method, m) for m in MARKETS})


def _best_other(res, prefix="trading"):
    cands = [m for m in METHODS if m not in ("SMARTRL", "B1_BH", "BASE")]
    vals = {m: _get(res, f"{prefix}.COMBINED.{m}.sharpe") for m in cands}
    vals = {m: v for m, v in vals.items() if v is not None}
    return max(vals, key=vals.get) if vals else "B2_SMA"


def _date_axis(ax):
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", labelsize=6)


def _wealth_plot(ax, series: dict, log_scale=False):
    for name, s in series.items():
        ax.plot(s.index, np.cumprod(1 + s.to_numpy()), lw=1.3 if name != "SMARTRL" else 1.9, label=style.method(name))
    ax.grid(alpha=0.3)
    _date_axis(ax)


def f07(res):
    df = _returns()
    if df is None:
        return None
    conf = df[df["fold"] != "forward"]
    other = _best_other(res)
    fig, axes = style.figure("F07", 2, 2, height_cm=12)
    for ax, m in zip(axes.ravel(), MARKETS):
        _wealth_plot(ax, {k: _series(conf, k, m) for k in ("SMARTRL", "B1_BH", "BASE", other)})
        ax.set_title(style.market(m), fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=6)
    return style.save(fig, "F07")


def f08(res):
    df = _returns()
    if df is None:
        return None
    conf = df[df["fold"] != "forward"]
    other = _best_other(res)
    fig, axes = style.figure("F08", 2, 2, height_cm=12, sharey=True)
    for ax, m in zip(axes.ravel(), MARKETS):
        for k in ("SMARTRL", "B1_BH", "BASE", other):
            w = np.cumprod(1 + _series(conf, k, m).to_numpy())
            ax.plot(_series(conf, k, m).index, w / np.maximum.accumulate(w) - 1, lw=1.2, label=style.method(k))
        ax.set_title(style.market(m), fontsize=9)
        ax.grid(alpha=0.3)
        _date_axis(ax)
        style.ticks(ax, "y", 0, percent=True)
    axes[0, 0].legend(frameon=False, fontsize=6)
    return style.save(fig, "F08")


def f09(res):
    df = _returns()
    if df is None:
        return None
    conf = df[df["fold"] != "forward"]
    other = _best_other(res)
    fig, ax = style.figure("F09")
    _wealth_plot(ax, {k: _comb(conf, k) for k in ("SMARTRL", "B1_BH", other, "BASE")})
    ax.set_ylabel(style.label("Growth of 1 (COMBINED, 2020-2024)", "رشد یک واحد سرمایه (Wealth)"))
    ax.legend(frameon=False, fontsize=8)
    return style.save(fig, "F09")


def f10(res):
    from . import confirm as C
    arms = {"CURR": "SMARTRL", "PLAIN": "A5", "BC": "BC", "IMIT": "IMIT"}
    curves = {}
    for key, arm in arms.items():
        rows = []
        for fold in [f"conf{y}" for y in range(2020, 2025)]:
            for m in MARKETS:
                for s in range(3):
                    p = C.policy_dir(fold, arm, m, s) / "info.json"
                    if p.exists():
                        cur = json.loads(p.read_text(encoding="utf-8"))["curve"]
                        rows.append(pd.Series([c["val_sharpe"] for c in cur], index=[c["steps"] for c in cur]))
        if rows:
            curves[key] = pd.concat(rows, axis=1)
    if not curves:
        return None
    fig, ax = style.figure("F10")
    for key, c in curves.items():
        mu, sd = c.mean(axis=1), c.std(axis=1)
        ax.plot(mu.index / 1000, mu, lw=1.6, label=style.method(key))
        ax.fill_between(mu.index / 1000, mu - sd / np.sqrt(c.shape[1]), mu + sd / np.sqrt(c.shape[1]), alpha=0.2)
    ax.set_xlabel(style.label("Training steps (thousands)", "گام‌های آموزش (هزار)"))
    ax.set_ylabel(style.label("Validation Sharpe (mean ± s.e.)", "شارپ اعتبارسنجی (Sharpe)"))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    return style.save(fig, "F10")


def f11(res):
    rank = _get(res, "tests.nemenyi.rank")
    cd = _get(res, "tests.nemenyi.cd")
    if not rank or cd is None:
        return None
    items = sorted(rank.items(), key=lambda kv: kv[1])
    fig, ax = style.figure("F11")
    ax.scatter([v for _, v in items], range(len(items)), color="C0", zorder=3)
    for i, (k, v) in enumerate(items):
        ax.text(v + 0.05, i, style.method(k), va="center", fontsize=7)
    best = items[0][1]
    ax.axvspan(best, best + cd, color="C1", alpha=0.15, label=style.label(f"Critical difference = {cd:.2f}",
                                                                            "اختلاف بحرانی (CD)"))
    ax.set_yticks([])
    ax.set_xlabel(style.label("Mean rank over 20 market-years (1 = best Sharpe)", "میانگین رتبه (Rank)"))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    return style.save(fig, "F11")


def f12(res):
    files = sorted(LOGS.glob("aws_daily_*_20[0-9][0-9].parquet"))
    if not files:
        return None
    from ..data import panel
    from .metrics import regime_labels
    fig, axes = style.figure("F12", 4, 1, sharex=True)
    colors = {"bull": "#d9f0d3", "bear": "#f4cccc", "sideways": "#f2f2f2"}
    for ax, m in zip(axes, MARKETS):
        w = pd.concat([pd.read_parquet(f) for f in files if f.stem.startswith(f"aws_daily_{m}_")])
        w["date"] = pd.to_datetime(w["date"])
        w = w.sort_values("date").set_index("date")
        close = pd.read_parquet(panel.processed_path(m)).set_index("date")["close"]
        reg = regime_labels(close).reindex(w.index)
        for r, col in colors.items():
            mask = (reg == r).to_numpy()
            ax.fill_between(w.index, 0, 1, where=mask, color=col, step="mid", lw=0)
        ax.plot(w.index, w["w_rl"], lw=0.8, color="C0")
        ax.set_ylim(0, 1)
        ax.set_ylabel(style.market(m) if m != "SSE" else "SSE", fontsize=7)
        _date_axis(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = {"bull": ("Bull", "صعودی"), "bear": ("Bear", "نزولی"), "sideways": ("Sideways", "خنثی")}
    handles = [Line2D([], [], color="C0", lw=1.2, label=style.label("w_RL", "w_RL"))]
    handles += [Patch(color=col, label=style.label(*names[r])) for r, col in colors.items()]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=7)
    return style.save(fig, "F12")


def f13(res):
    cost = _get(res, "robust.cost")
    if not cost:
        return None
    fig, ax = style.figure("F13")
    cs = sorted(int(c) for c in cost)
    for m in METHODS:
        ys = [_get(res, f"robust.cost.{c}.{m}.sharpe") for c in cs]
        ax.plot(cs, ys, marker="o", ms=3, lw=1.8 if m == "SMARTRL" else 1.0, label=style.method(m))
    ax.set_xlabel(style.label("Trading cost (bps per unit turnover)", "هزینه‌ی معامله (bps)"))
    ax.set_ylabel(style.label("COMBINED Sharpe", "نسبت شارپ (Sharpe)"))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    return style.save(fig, "F13")


def f14(res):
    df = _returns()
    if df is None or (df["fold"] == "forward").sum() == 0:
        return None
    fw = df[df["fold"] == "forward"]
    other = _best_other(res, "forward")
    fig = None
    fig, axes = style.figure("F14", 2, 3)
    ax0 = plt_ax = axes[0, 0]
    _wealth_plot(ax0, {k: _comb(fw, k) for k in ("SMARTRL", "B1_BH", other, "BASE")})
    ax0.set_title(style.market("COMBINED"), fontsize=8)
    for ax, m in zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]], MARKETS):
        _wealth_plot(ax, {k: _series(fw, k, m) for k in ("SMARTRL", "B1_BH", other, "BASE")})
        ax.set_title(style.market(m), fontsize=8)
    axes[1, 2].axis("off")
    h, l = ax0.get_legend_handles_labels()
    axes[1, 2].legend(h, l, frameon=False, fontsize=7, loc="center")
    for ax in axes.ravel()[:5]:
        ax.tick_params(labelsize=6)
    return style.save(fig, "F14")


BUILDERS_P5 = {"T08": t08, "T09": t09, "T10": t10, "T11": t11, "T12": t12, "T13": t13, "T14": t14, "T15": t15,
               "T16": t16, "T17": t17, "T18": t18, "T19": t19, "TS1": ts1, "F07": f07, "F08": f08, "F09": f09,
               "F10": f10, "F11": f11, "F12": f12, "F13": f13, "F14": f14}

BUILDERS = {**BUILDERS_P5, "F16": f16, "T03": t03, "T04": t04, "T05": t05, "T06": t06, "T07": t07, "T20": t20,
            "F02": f02, "F03": f03, "F04": f04, "F05": f05, "F06": f06, "F15": f15}


def build(ids: list[str] | None = None) -> dict:
    res = R.load()
    done = {}
    for k, fn in BUILDERS.items():
        if ids and k not in ids:
            continue
        done[k] = bool(fn(res))
    return done
