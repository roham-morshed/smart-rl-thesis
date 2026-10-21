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
        for m in MARKETS + ["MEAN"]:
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


BUILDERS = {"T03": t03, "T04": t04, "T05": t05, "T06": t06, "T07": t07, "T20": t20,
            "F02": f02, "F03": f03, "F04": f04, "F05": f05, "F06": f06, "F15": f15}


def build(ids: list[str] | None = None) -> dict:
    res = R.load()
    done = {}
    for k, fn in BUILDERS.items():
        if ids and k not in ids:
            continue
        done[k] = bool(fn(res))
    return done
