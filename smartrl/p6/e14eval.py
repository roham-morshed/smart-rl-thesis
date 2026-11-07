"""E14 metrics (protocol §2.6), pre-registered tests and claim rules (§2.7), tables T21, T22, T25 and figures F17, F19.

Metrics are computed per seed and averaged (sd over seeds); tests use seed-averaged scores of all test rows.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .. import results as R
from ..config import MARKETS, RESULTS, TABLES
from . import e14data as ED
from . import stats as ST
from .cfg import P6_DIR, p6, p6_run
from .e14run import POOLED, PRED

PAPER_ORDER = ["SP500", "FTSE100", "SSE", "NIFTY50"]


def settings_models() -> list[tuple[str, list[str]]]:
    e = p6()["e14"]
    out = []
    for grp in ("full", "gt", "la", "robust"):
        out += [(s, e["models"][grp]) for s in e["settings"][grp]]
    return out


def labels_for(setting: str) -> pd.DataFrame:
    rows = []
    for m in MARKETS:
        mr = ED.market_rows(m, setting)
        rows.append(pd.DataFrame({"market": m, "date": mr.frame.index[mr.pos[mr.test]], "y": mr.y[mr.test],
                                  "informative": mr.informative[mr.test]}))
    return pd.concat(rows, ignore_index=True)


def load_preds(setting: str, model: str) -> pd.DataFrame | None:
    """All seeds of one model; MM_DIR in a GT setting = its GE fit scored against the GT labels."""
    src = setting.replace("_GT", "_GE") if (model == "MM_DIR" and setting.endswith("_GT")) else setting
    files = sorted((PRED / src / model).glob("*.parquet"))
    if not files:
        return None
    P = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if src != setting:
        lab = labels_for(setting)
        P = P.drop(columns=["y", "informative"]).merge(lab, on=["market", "date"], how="inner")
    return P


def predict_label(model: str, label: str, score: np.ndarray) -> np.ndarray:
    if model == "SVM_P":
        return (score > 0).astype(int)
    if model == "MM_DIR":
        return (score >= 0).astype(int) if label == "GE" else (score > 0).astype(int)
    return (score > p6()["e14"]["threshold"]).astype(int)


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    two = len(np.unique(y)) == 2
    return {"acc": float((pred == y).mean()), "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, score)) if two else float("nan"),
            "prauc": float(average_precision_score(y, score)) if y.sum() > 0 else float("nan")}


# ---------------------------------------------------------------- stage eval
def stage_eval(args) -> None:
    items, notes = [], []
    with p6_run("E14_eval"):
        for setting, models in settings_models():
            label = ED.parse(setting)["label"]
            per_mkt = {}
            for model in models:
                P = load_preds(setting, model)
                if P is None:
                    notes.append(f"missing predictions: {setting} {model}")
                    continue
                for m in MARKETS:
                    g = P[P["market"] == m]
                    rows, inf = [], []
                    for _, h in g.groupby("seed"):
                        h = h.sort_values("date")
                        y, sc = h["y"].to_numpy(int), h["score"].to_numpy(float)
                        pr = predict_label(model, label, sc)
                        rows.append(metrics(y, sc, pr))
                        k = h["informative"].to_numpy(bool)
                        inf.append(metrics(y[k], sc[k], pr[k]))
                    for met in ("acc", "f1", "auc", "prauc"):
                        v = np.array([r[met] for r in rows])
                        items.append((f"e14.{setting}.{m}.{model}.{met}", float(np.nanmean(v))))
                        if len(v) > 1:
                            items.append((f"e14.{setting}.{m}.{model}.{met}_sd", float(np.nanstd(v, ddof=1))))
                        items.append((f"e14.{setting}.{m}.{model}.inf.{met}", float(np.nanmean([r[met] for r in inf]))))
                        per_mkt.setdefault((model, met), []).append(float(np.nanmean(v)))
                        per_mkt.setdefault((model, "inf." + met), []).append(float(np.nanmean([r[met] for r in inf])))
            for (model, met), v in per_mkt.items():
                items.append((f"e14.{setting}.MEAN.{model}.{met}", float(np.mean(v))))
            for m in MARKETS:
                mr = ED.market_rows(m, setting)
                d = mr.frame.index[mr.pos]
                items += [(f"e14.{setting}.{m}.n_train", int(len(mr.train))), (f"e14.{setting}.{m}.n_test", int(len(mr.test))),
                          (f"e14.{setting}.{m}.test_start", str(d[mr.test].min().date())),
                          (f"e14.{setting}.{m}.test_end", str(d[mr.test].max().date())),
                          (f"e14.{setting}.{m}.pos_share", float(mr.y[mr.test].mean())),
                          (f"e14.{setting}.{m}.informative_share", float(mr.informative[mr.test].mean()))]
            print(setting, "evaluated", flush=True)
        for m in MARKETS:                                         # label evidence (no training)
            for lab in ("GE", "GT"):
                for base in ("PUB_C", "PUB_R"):
                    mr = ED.market_rows(m, f"{base}_{lab}")
                    y = mr.y[mr.test]
                    p = float(y.mean())
                    items += [(f"e14.labels.{m}.{lab}.{base}.pos_share", p), (f"e14.labels.{m}.{lab}.{base}.allpos_acc", p),
                              (f"e14.labels.{m}.{lab}.{base}.allpos_f1", 2 * p / (1 + p)),
                              (f"e14.labels.{m}.{lab}.{base}.allpos_prauc", float(average_precision_score(y, np.ones(len(y)))))]
    R.update(items, ["e14.<SET>.<MKT>.<MODEL>: test metrics, mean (and sd) over seeds 0-2 (SVM deterministic); .inf = "
                     "the same metrics on the informative_row test rows (the next calendar date is a session). "
                     "Thresholds: P(up) > 0.5; SVM decision > 0; MM_DIR q50 >= 0 (GE) or > 0 (GT). MM_DIR in the GT "
                     "settings is its GE fit scored against the GT labels (protocol run matrix).",
                     "e14.labels: always-positive rule on the PUB test rows (diagnostic, no training)."] + notes)
    print(f"{len(items)} keys", notes, flush=True)


# ---------------------------------------------------------------- stage tests
def seed_avg(setting: str, model: str) -> pd.DataFrame:
    P = load_preds(setting, model)
    return P.groupby(["market", "date"], as_index=False).agg(score=("score", "mean"), y=("y", "first"),
                                                             informative=("informative", "first"))


def compare(setting: str, a: str, b: str, informative_only: bool = False) -> dict:
    t = p6()["e14"]["tests"]
    label = ED.parse(setting)["label"]
    A, B = seed_avg(setting, a), seed_avg(setting, b)
    J = A.merge(B, on=["market", "date"], suffixes=("_a", "_b"))
    assert (J["y_a"] == J["y_b"]).all()
    groups, delong = [], {}
    for m in MARKETS:
        g = J[J["market"] == m].sort_values("date")
        if informative_only:
            g = g[g["informative_a"].astype(bool)]
        y = g["y_a"].to_numpy(int)
        sa, sb = g["score_a"].to_numpy(float), g["score_b"].to_numpy(float)
        groups.append({"y": y, "s_a": sa, "s_b": sb, "p_a": predict_label(a, label, sa), "p_b": predict_label(b, label, sb)})
        delong[m] = ST.delong(y, sa, sb)["p"]
    bs = ST.paired_bootstrap(groups, t["reps"], t["mean_block"], t["seed"])
    ca = np.concatenate([g["p_a"] == g["y"] for g in groups])
    cb = np.concatenate([g["p_b"] == g["y"] for g in groups])
    return {"boot": bs, "mcnemar_p": ST.mcnemar_exact(ca, cb)["p"], "delong": delong,
            "delong_holm": ST.holm(delong), "n_rows": int(sum(len(g["y"]) for g in groups))}


def stage_tests(args) -> None:
    from joblib import Parallel, delayed
    e = p6()["e14"]
    prim = [("PUB_C_GE", a, b) for a, b in e["primary"]] + [("PUB_R_GE", a, b) for a, b in e["primary"]]
    plan = [("primary", s, a, b, False) for s, a, b in prim]
    plan += [("v1", s, a, b, True) for s, a, b in prim]
    plan += [("v2", s, a, b, False) for s in ("PUB_C_GE", "PUB_R_GE") for a, b in e["v2"]]
    plan += [("v3", s, a, b, False) for s in ("PUB_C_GT", "PUB_R_GT") for a, b in e["primary"]]
    plan += [("v4", "CAUSAL", a, b, False) for a, b in e["v4"]]
    with p6_run("E14_tests", None, "cpu", {"reps": e["tests"]["reps"], "n_comparisons": len(plan)}):
        res = Parallel(n_jobs=getattr(args, "jobs", 4), backend="loky", verbose=5)(
            delayed(compare)(s, a, b, inf) for _, s, a, b, inf in plan)
    out = {(fam, s, a, b): r for (fam, s, a, b, _), r in zip(plan, res)}
    items = []
    fam = {k: v for k, v in out.items() if k[0] == "primary"}
    holm = {k: ST.holm({kk: vv["boot"][k]["p"] for kk, vv in fam.items()}) for k in ("auc", "acc", "f1", "prauc")}
    mc_holm = ST.holm({kk: vv["mcnemar_p"] for kk, vv in fam.items()})
    for (f, s, a, b), r in out.items():
        cmp = f"{a}_vs_{b}"
        bs = r["boot"]
        if f == "v1":
            items += [(f"e14.tests_inf.{s}.{cmp}.{k}", bs["auc"][kk]) for k, kk in
                      (("auc_diff", "diff"), ("lo", "lo"), ("hi", "hi"), ("p", "p"))]
            continue
        pre = f"e14.tests.{s}.{cmp}"
        items += [(f"{pre}.auc_diff", bs["auc"]["diff"]), (f"{pre}.lo", bs["auc"]["lo"]), (f"{pre}.hi", bs["auc"]["hi"]),
                  (f"{pre}.p", bs["auc"]["p"]), (f"{pre}.acc_diff", bs["acc"]["diff"]), (f"{pre}.acc_p", bs["acc"]["p"]),
                  (f"{pre}.mcnemar_p", r["mcnemar_p"]), (f"{pre}.f1_diff", bs["f1"]["diff"]), (f"{pre}.f1_p", bs["f1"]["p"]),
                  (f"{pre}.prauc_diff", bs["prauc"]["diff"]), (f"{pre}.prauc_p", bs["prauc"]["p"]),
                  (f"{pre}.n_rows", r["n_rows"])]
        if f == "primary":
            k = (f, s, a, b)
            items += [(f"{pre}.p_holm", holm["auc"][k]), (f"{pre}.acc_p_holm", holm["acc"][k]),
                      (f"{pre}.mcnemar_p_holm", mc_holm[k]), (f"{pre}.f1_p_holm", holm["f1"][k]),
                      (f"{pre}.prauc_p_holm", holm["prauc"][k])]
        for m in MARKETS:
            items += [(f"{pre}.delong.{m}.p", r["delong"][m]), (f"{pre}.delong.{m}.p_holm", r["delong_holm"][m])]
    claims = claim_sentences(out)
    items += [(f"e14.claims.{k}", v) for k, v in claims.items()]
    (P6_DIR / "e14_tests.json").write_text(json.dumps({"|".join(k): v for k, v in out.items()}, indent=1, default=float),
                                           encoding="utf-8")
    R.update(items, ["e14.tests: protocol §2.7. Statistic = mean over the four indices of the metric difference "
                     "(ours - SENTIN_PAPER unless named), seed-averaged scores of all test rows; one-sided p (H1: ours "
                     "better) from a paired stationary block bootstrap stratified by index (test rows sorted by date, "
                     "mean block 10, 10 000 resamples, seed 2026, centred); lo/hi = two-sided 95 % percentile interval; "
                     "p_holm = Holm over the four primary tests (secondary metrics: Holm within each metric); mcnemar_p "
                     "= exact McNemar on the threshold predictions pooled over the indices (two-sided); delong = "
                     "two-sided DeLong per index with Holm over the four indices. tests_inf = V1 (informative rows). "
                     "V2 (vs SENTIN_E13), V3 (GT settings) and V4 (CAUSAL) are stored under e14.tests without p_holm.",
                     "e14.claims: sentences produced by the claim rules of protocol §2.7, written verbatim into "
                     "reports/P6_report.md."])
    for k, v in claims.items():
        print(k, ":", v, flush=True)


def claim_sentences(out: dict) -> dict:
    e = p6()["e14"]
    alpha = e["tests"]["alpha"]
    reading = {"PUB_C_GE": "chronological", "PUB_R_GE": "random"}
    fam = {k: v for k, v in out.items() if k[0] == "primary"}
    holm = ST.holm({kk: vv["boot"]["auc"]["p"] for kk, vv in fam.items()})
    res = {}
    for a, b in e["primary"]:
        ok = [s for s in ("PUB_C_GE", "PUB_R_GE") if holm[("primary", s, a, b)] < alpha]
        if not ok:
            res[a] = (f"{a} is not significantly better than SenT-In under the paper's conditions (Holm-adjusted "
                      f"one-sided p {holm[('primary', 'PUB_C_GE', a, b)]:.3f} in PUB_C_GE and "
                      f"{holm[('primary', 'PUB_R_GE', a, b)]:.3f} in PUB_R_GE).")
            continue
        head = (f"{a} is significantly better than SenT-In under the paper's conditions" if len(ok) == 2 else
                f"{a} is significantly better than SenT-In under the paper's conditions in the {reading[ok[0]]} "
                f"reading of the split ({ok[0]})")
        parts = []
        for s in ok:
            v1 = out[("v1", s, a, b)]["boot"]["auc"]["p"]
            parts.append(f"in {s} the advantage is not explained by the filled calendar rows (V1 p {v1:.3f})" if v1 < alpha
                         else f"in {s} the advantage comes from rows whose label is fixed by the filling (V1 p {v1:.3f})")
            v2_key = ("v2", s, "MM_CLS" if a == "MM_CLS" else "SENTIN_SF_E13", "SENTIN_E13")
            v2 = out[v2_key]["boot"]["auc"]["p"]
            if v2 < alpha:
                parts.append(f"in {s} it holds also against the stronger reproduction of SenT-In (V2 p {v2:.3f})")
        res[a] = head + "; " + "; ".join(parts) + "."
    return res


# ---------------------------------------------------------------- stage outputs (T21, T22, T25, F17, F19)
MODEL_LABEL = {"SVM_P": "SVM", "CNN_P": "CNN", "LSTM_P": "LSTM", "GRU_P": "GRU", "SENTIN_PAPER": "SenT-In (paper recipe)",
               "SENTIN_E13": "SenT-In (E13 recipe)", "SENTIN_SF_PAPER": "SenT-In + SentiFlow (paper recipe)",
               "SENTIN_SF_E13": "SenT-In + SentiFlow (E13 recipe)", "MM_CLS": "MarketMind classifier",
               "MM_DIR": "MarketMind q50"}


def _g(t, key):
    v = R.lookup(t, key)
    return None if v is R.MISSING else v


def auc_intervals(reps: int = 2000) -> list:
    """95 % stationary-bootstrap intervals (mean block 10) of each model's AUC per index, seed-averaged scores (F17)."""
    t = p6()["e14"]["tests"]
    items = []
    rng = np.random.default_rng(t["seed"])
    for setting in ("PUB_C_GE", "PUB_R_GE", "CAUSAL"):
        for model in p6()["e14"]["models"]["full"]:
            A = seed_avg(setting, model)
            for m in MARKETS:
                g = A[A["market"] == m].sort_values("date")
                y, s = g["y"].to_numpy(int), g["score"].to_numpy(float)
                bs = []
                for _ in range(reps):
                    i = ST.stationary_indices(len(y), t["mean_block"], rng)
                    bs.append(ST.auc_fast(y[i], s[i]))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                items += [(f"e14.ci.{setting}.{m}.{model}.auc_lo", float(lo)), (f"e14.ci.{setting}.{m}.{model}.auc_hi", float(hi))]
    return items


def auc_points() -> list:
    """AUC of the seed-averaged scores per setting, index and model: the point estimate that belongs to the e14.ci
    interval (the e14.<SET> keys hold the mean of the per-seed AUCs, a different estimator)."""
    items = []
    for setting in ("PUB_C_GE", "PUB_R_GE", "CAUSAL"):
        for model in p6()["e14"]["models"]["full"]:
            A = seed_avg(setting, model)
            for m in MARKETS:
                g = A[A["market"] == m]
                items.append((f"e14.ci.{setting}.{m}.{model}.auc", ST.auc_fast(g["y"].to_numpy(int), g["score"].to_numpy(float))))
    return items


def stage_outputs(args) -> None:
    import matplotlib.pyplot as plt
    from ..figures import style
    if not _g(R.load(), "e14.ci.PUB_C_GE.SP500.MM_CLS.auc_lo"):
        R.update(auc_intervals(), ["e14.ci: 95 % stationary-bootstrap percentile intervals (mean block 10, 2000 "
                                   "resamples, seed 2026) of each model's AUC per index on seed-averaged scores; "
                                   "drawn in figure F17, not used by any test."])
    if _g(R.load(), "e14.ci.PUB_C_GE.SP500.MM_CLS.auc") is None:
        R.update(auc_points(), ["e14.ci.<SET>.<MKT>.<MODEL>.auc: AUC of the seed-averaged scores, the point estimate of "
                                "the e14.ci interval and of the tests (e14.<SET>.<MKT>.<MODEL>.auc is the mean of the "
                                "per-seed AUCs); F17 draws this point with its interval."])
    t = R.load()
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for setting, models in settings_models():
        for m in MARKETS:
            for model in models:
                key = f"e14.{setting}.{m}.{model}"
                if _g(t, key + ".auc") is None:
                    continue
                r = {"setting": setting, "index": m, "model": model, "pos_share": _g(t, f"e14.{setting}.{m}.pos_share")}
                for k in ("acc", "f1", "auc", "prauc"):
                    r[k], r[k + "_sd"] = _g(t, f"{key}.{k}"), _g(t, f"{key}.{k}_sd")
                r["inf_auc"] = _g(t, f"{key}.inf.auc")
                rows.append(r)
    pd.DataFrame(rows).to_csv(TABLES / "T21.csv", index=False)
    rows = []
    for setting in ("PUB_C_GE", "PUB_R_GE"):
        for m in PAPER_ORDER:
            for model in p6()["e14"]["models"]["full"]:
                key = f"e14.{setting}.{m}.{model}"
                rows.append({"setting": setting, "index": m, "model": MODEL_LABEL[model],
                             "Accuracy": _g(t, key + ".acc"), "F1-Score": _g(t, key + ".f1"),
                             "AUC-ROC": _g(t, key + ".auc"), "PR-AUC": _g(t, key + ".prauc")})
    pd.DataFrame(rows).to_csv(TABLES / "T22.csv", index=False)
    rows = []
    e = p6()["e14"]
    fams = ([("primary", s, a, b) for s in ("PUB_C_GE", "PUB_R_GE") for a, b in e["primary"]] +
            [("V1 informative rows", s, a, b) for s in ("PUB_C_GE", "PUB_R_GE") for a, b in e["primary"]] +
            [("V2 stronger base", s, a, b) for s in ("PUB_C_GE", "PUB_R_GE") for a, b in e["v2"]] +
            [("V3 label GT", s, a, b) for s in ("PUB_C_GT", "PUB_R_GT") for a, b in e["primary"]] +
            [("V4 CAUSAL", "CAUSAL", a, b) for a, b in e["v4"]])
    for fam, s, a, b in fams:
        pre = (f"e14.tests_inf.{s}.{a}_vs_{b}" if fam.startswith("V1") else f"e14.tests.{s}.{a}_vs_{b}")
        r = {"family": fam, "setting": s, "comparison": f"{a} vs {b}"}
        for k in ("auc_diff", "lo", "hi", "p", "p_holm", "acc_diff", "mcnemar_p", "f1_diff", "f1_p", "prauc_diff", "prauc_p"):
            r[k] = _g(t, f"{pre}.{k}")
        rows.append(r)
    for k in ("MM_CLS", "SENTIN_SF_PAPER"):
        rows.append({"family": "claim", "setting": "", "comparison": k, "claim": _g(t, f"e14.claims.{k}")})
    pd.DataFrame(rows).to_csv(TABLES / "T25.csv", index=False)
    # F17: AUC per index and model with 95 % intervals
    models = e["models"]["full"]
    fig, axes = style.figure("F17", 3, 1, height_cm=20, sharex=True)
    for ax, setting in zip(axes, ("PUB_C_GE", "PUB_R_GE", "CAUSAL")):
        w = 0.8 / len(MARKETS)
        for j, m in enumerate(PAPER_ORDER):
            x = np.arange(len(models)) + (j - 1.5) * w
            v = np.array([_g(t, f"e14.ci.{setting}.{m}.{mo}.auc") or np.nan for mo in models], float)
            lo = np.array([_g(t, f"e14.ci.{setting}.{m}.{mo}.auc_lo") or np.nan for mo in models], float)
            hi = np.array([_g(t, f"e14.ci.{setting}.{m}.{mo}.auc_hi") or np.nan for mo in models], float)
            err = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
            ax.errorbar(x, v, yerr=err, fmt="o", ms=3, capsize=2, lw=0.8, label=style.market(m))
        ax.axhline(0.5, color="grey", lw=0.6, ls="--")
        ax.set_ylabel(f"AUC-ROC ({setting})", fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(np.arange(len(models)))
    axes[-1].set_xticklabels([MODEL_LABEL[mo] for mo in models], rotation=35, ha="right", fontsize=6)
    axes[0].legend(frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    style.save(fig, "F17")
    plt.close(fig)
    # F19: accuracy with and without the look-ahead, paper SenT-In accuracies as markers
    la_models = e["models"]["la"]
    fig, axes = style.figure("F19", 2, 1, height_cm=14, sharex=True)
    for ax, base in zip(axes, ("PUB_C", "PUB_R")):
        w = 0.8 / len(la_models)
        for j, mo in enumerate(la_models):
            x = np.arange(len(PAPER_ORDER)) + (j - (len(la_models) - 1) / 2) * w
            a0 = [_g(t, f"e14.{base}_GE.{m}.{mo}.acc") or np.nan for m in PAPER_ORDER]
            a1 = [_g(t, f"e14.{base}_GE_LA.{m}.{mo}.acc") or np.nan for m in PAPER_ORDER]
            ax.bar(x, a1, w * 0.9, color=f"C{j}", alpha=0.35)
            ax.bar(x, a0, w * 0.9, color=f"C{j}", label=MODEL_LABEL[mo])
        pa = [e["paper_sentin_acc"][m] for m in PAPER_ORDER]
        ax.scatter(np.arange(len(PAPER_ORDER)), pa, marker="*", s=60, color="black", zorder=5,
                   label="SenT-In, published accuracy")
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel(f"Accuracy ({base}_GE; light = with LA)", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[-1].set_xticks(np.arange(len(PAPER_ORDER)))
    axes[-1].set_xticklabels([style.market(m) for m in PAPER_ORDER], fontsize=7)
    axes[0].legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    style.save(fig, "F19")
    plt.close(fig)
    print("T21, T22, T25, F17, F19 written", flush=True)
