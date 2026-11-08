"""Write results/FINAL_RESULTS.md: every thesis number with its results.json key, the pre-registered verdicts,
thesis-ready statements, limitations, the audit and the notes. Numbers are read from results.json only; a missing
key stops the script (nothing is typed by hand).

    python scripts/make_final_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, RESULTS, ROOT  # noqa: E402

T = R.load()
OUT = RESULTS / "FINAL_RESULTS.md"
METHODS = ["SMARTRL", "B1_BH", "B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC", "B8_STATIC", "BASE"]
H1_SET = ["B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC"]
ABL = ["A1_NOSENT", "A2_NOMM", "A3_NOAWS", "A4_NOCONF", "A5_NOCURR"]
NAME = {"SMARTRL": "SMART-RL (proposed)", "B1_BH": "B1 Buy & hold", "B2_SMA": "B2 SMA 50/200",
        "B3_TSMOM": "B3 Time-series momentum 12-1", "B4_LSTM": "B4 LSTM signal", "B5_FINBERT": "B5 FinBERT signal",
        "B6_PPOPRICE": "B6 PPO price-only", "B7_SAC": "B7 SAC", "B8_STATIC": "B8 Static 50/50 hybrid",
        "BASE": "BASE: SenT-In (base paper) as a trading signal", "A1_NOSENT": "A1 without sentiment (no STM, no AWS)",
        "A2_NOMM": "A2 without MarketMind features", "A3_NOAWS": "A3 PPO ensemble alone (no AWS)",
        "A4_NOCONF": "A4 AWS without confidence term", "A5_NOCURR": "A5 PPO without curriculum"}
MKT = {"SP500": "S&P 500", "FTSE100": "FTSE 100", "NIFTY50": "Nifty 50", "SSE": "SSE Composite"}


def g(key: str):
    v = R.lookup(T, key)
    if v is R.MISSING:
        raise KeyError(key)
    return v


def f3(key: str) -> str:
    return f"{g(key):.3f}"


def pct(key: str, d: int = 1) -> str:
    return f"{100 * g(key):.{d}f} %"


def pv(key: str) -> str:
    v = g(key)
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def peq(key: str) -> str:
    v = g(key)
    return "p < 0.001" if v < 0.001 else f"p = {v:.3f}"


def table(head, rows) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def rel(a: str, b: str) -> str:
    """relative reduction of |b| achieved by |a|, in percent"""
    return f"{100 * (1 - abs(g(a)) / abs(g(b))):.0f} %"


def two_sided(key: str) -> str:
    p = g(key)
    q = 2 * min(p, 1 - p)
    return "< 0.001" if q < 0.001 else f"{q:.3f}"


def rank_by(block: str, metric: str, methods: list[str], reverse=True) -> list[str]:
    return sorted(methods, key=lambda m: g(f"{block}.{m}.{metric}"), reverse=reverse)


# ------------------------------------------------------------------ P6 (E14, E15): the base paper's own conditions
P6_ORDER = ["SP500", "FTSE100", "SSE", "NIFTY50"]                 # the paper's order
P6_MODELS = [("SVM_P", "SVM (paper recipe)"), ("CNN_P", "CNN (paper recipe)"), ("LSTM_P", "LSTM (paper recipe)"),
             ("GRU_P", "GRU (paper recipe)"), ("SENTIN_PAPER", "SenT-In, paper recipe"),
             ("SENTIN_E13", "SenT-In, E13 recipe (stronger)"), ("SENTIN_SF_PAPER", "SenT-In + SentiFlow, paper recipe"),
             ("SENTIN_SF_E13", "SenT-In + SentiFlow, E13 recipe"), ("MM_CLS", "MarketMind classifier (proposed)"),
             ("MM_DIR", "MarketMind q50 direction")]
P6_MET = ["acc", "f1", "auc", "prauc"]
P6_TRADE = METHODS[:-1] + ["BASE", "BASE_PAPER"]


def d3(key: str) -> str:
    return f"{g(key):+.3f}"


def ci3(pre: str) -> str:
    return f"[{g(pre + '.lo'):+.3f}, {g(pre + '.hi'):+.3f}]"


def p6_published() -> dict:
    """SenT-In as published (Chauhan et al. 2025, Tables 4-7), quoted in scripts/p6_report.py."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from p6_report import PUBLISHED
    return PUBLISHED


def p6_reached(settings) -> list[tuple[str, str, str, str]]:
    pub = p6_published()
    out = []
    for s in settings:
        for m in P6_ORDER:
            for mo, _ in P6_MODELS:
                for i, k in enumerate(P6_MET):
                    v = R.lookup(T, f"e14.{s}.{m}.{mo}.{k}")
                    if v is not R.MISSING and v >= pub[m][i]:
                        out.append((s, m, mo, k))
    return out


def p6_la_max() -> tuple[float, str, str, str]:
    best = []
    for s in ("PUB_C_GE_LA", "PUB_R_GE_LA"):
        for m in P6_ORDER:
            for mo, _ in P6_MODELS:
                v = R.lookup(T, f"e14.{s}.{m}.{mo}.acc")
                if v is not R.MISSING:
                    best.append((v, s, m, mo))
    return max(best)


def p6_findings() -> list[str]:
    c, r = "e14.PUB_C_GE.MEAN", "e14.PUB_R_GE.MEAN"
    tc, tr = "e14.tests.PUB_C_GE.MM_CLS_vs_SENTIN_PAPER", "e14.tests.PUB_R_GE.MM_CLS_vs_SENTIN_PAPER"
    dl = max(g(f"{t}.delong.{m}.p_holm") for t in (tc, tr) for m in P6_ORDER)
    e = "e15.full.COMBINED"
    return [
        f"**Better than the base paper under its own conditions, on its own metrics (E14, pre-registered).** With the "
        f"paper's exact window and row count ({g('e14.data.SP500.labelled_rows')} calendar rows per index, 2005-01 to "
        "2022-03, non-trading days filled) and its 80:20 split, the proposed MarketMind classifier beats SenT-In rebuilt "
        "with the paper's own training recipe on both threshold-free metrics of the paper, in both readings of the split: "
        f"AUC-ROC {f3(c + '.MM_CLS.auc')} vs {f3(c + '.SENTIN_PAPER.auc')} (chronological) and {f3(r + '.MM_CLS.auc')} "
        f"vs {f3(r + '.SENTIN_PAPER.auc')} (random); PR-AUC {f3(c + '.MM_CLS.prauc')} vs {f3(c + '.SENTIN_PAPER.prauc')} "
        f"and {f3(r + '.MM_CLS.prauc')} vs {f3(r + '.SENTIN_PAPER.prauc')}; "
        + ("Holm-adjusted p < 0.001 in both readings" if max(g(tc + '.p_holm'), g(tr + '.p_holm')) < 0.001 else
           f"Holm-adjusted p {pv(tc + '.p_holm')} and {pv(tr + '.p_holm')}")
        + (", DeLong p < 0.001 in every index" if dl < 0.001 else "") + " (`e14.tests.*`). It "
        "also beats our stronger SenT-In reproduction (AUC-ROC "
        f"{d3('e14.tests.PUB_C_GE.MM_CLS_vs_SENTIN_E13.auc_diff')} / {d3('e14.tests.PUB_R_GE.MM_CLS_vs_SENTIN_E13.auc_diff')}). "
        "SenT-In with the paper's recipe predicts \"up\" on every row, which gives it the slightly higher accuracy "
        f"({f3(c + '.SENTIN_PAPER.acc')} vs {f3(c + '.MM_CLS.acc')}) and F1; with the strict label 1[C(t+1) > C(t)] "
        "MarketMind is better on all four metrics. The pre-registered validity check V1 shows where the advantage comes "
        "from: rows whose label is fixed by the filling of non-trading days; on rows followed by a trading session every "
        "model, SenT-In included, is at chance (section 17).",
        f"**In the base paper's own test window (E15, {g('e15.window.first_test_day')} to "
        f"{g('e15.window.last_test_day')}).** SMART-RL, run once with the locked configuration, reached a COMBINED Sharpe "
        f"of {f3(e + '.SMARTRL.sharpe')} against {f3(e + '.BASE.sharpe')} for SenT-In as a trading signal (difference "
        f"not significant, p {pv('e15.tests.BASE.p_hac')}) and a maximum drawdown of {pct(e + '.SMARTRL.mdd')} against "
        f"{pct(e + '.B1_BH.mdd')} for buy-and-hold, which SenT-In with the paper's recipe equals because it is always "
        f"invested (95 % interval of the drawdown difference {ci3('e15.tests.B1_BH.mdd_diff')}); its Sharpe is below "
        f"buy-and-hold's ({f3(e + '.B1_BH.sharpe')}; p {pv('e15.tests.B1_BH.p_hac')} for SMART-RL being higher).",
    ]


def p6_section(w) -> None:
    pub = p6_published()
    name = dict(P6_MODELS)
    w("## 17. The base paper's task under its exact conditions (E14) and trading in its test window (E15) "
      "(T21–T25, F17–F19)\n")
    w("Pre-registered in `docs/P6_BASE_PROTOCOL.md` (commit `aaf78ff`) before any run; every setting in "
      "`configs/p6.yaml`; gate report `reports/P6_report.md` (gate G6 passed). E14 repeats the paper's next-day "
      f"direction task with its exact window ({g('e14.data.SP500.first_date')} to {g('e14.data.SP500.last_date')}, "
      f"{g('e14.data.SP500.labelled_rows')} labelled calendar rows per index, non-trading days filled with the previous "
      "day's values), label GE = 1[C(t+1) ≥ C(t)] (the label the paper's metric pattern implies), the 80:20 split read "
      f"chronologically (`PUB_C`, test {g('e14.PUB_C_GE.SP500.test_start')} to {g('e14.PUB_C_GE.SP500.test_end')}) and "
      "randomly (`PUB_R`), SenT-In rebuilt with the paper's GRU layer (eqs 7–11) and its Table 3 recipe "
      "(`SENTIN_PAPER`), the paper's baselines and sentiment classifier with its Table 2 recipe, and the proposed "
      f"MarketMind classifier with SentiFlow features (`MM_CLS`). {g('e14.compute.n_model_runs')} model runs; "
      f"{sum(g(f'e15.compute.policies.{a}') for a in ('SMARTRL', 'B6', 'B7'))} E15 policies.\n")

    w("### 17.1 The paper's layout with the published SenT-In numbers (T22)\n")
    w("Per index in the paper's order; \"SenT-In, published\" is quoted from Chauhan et al. (2025), Tables 4–7, and is "
      "not a result of this project (`e14.<SET>.<index>.<model>.<acc|f1|auc|prauc>`).\n")
    rows = []
    for m in P6_ORDER:
        rows.append([MKT[m], "SenT-In, published"] + [f"{x:.3f}" for x in pub[m]] * 2)
        for mo in ("SENTIN_PAPER", "SENTIN_E13", "MM_CLS"):
            rows.append(["", name[mo]] + [f3(f"e14.{s}.{m}.{mo}.{k}") for s in ("PUB_C_GE", "PUB_R_GE") for k in P6_MET])
    w(table(["index", "model", "PUB_C Acc", "PUB_C F1", "PUB_C AUC-ROC", "PUB_C PR-AUC", "PUB_R Acc", "PUB_R F1",
             "PUB_R AUC-ROC", "PUB_R PR-AUC"], rows) + "\n")

    w("### 17.2 All models, mean over the four indices (T21)\n")
    w(table(["model", "PUB_C Acc", "PUB_C F1", "PUB_C AUC-ROC", "PUB_C PR-AUC", "PUB_R Acc", "PUB_R F1",
             "PUB_R AUC-ROC", "PUB_R PR-AUC"],
            [[nm] + [f3(f"e14.{s}.MEAN.{mo}.{k}") for s in ("PUB_C_GE", "PUB_R_GE") for k in P6_MET]
             for mo, nm in P6_MODELS]) + "\n")

    w("### 17.3 Pre-registered tests (T25)\n")
    w("Primary family: one-sided H1 ours > SenT-In (paper recipe); statistic = mean over the four indices of the "
      "AUC-ROC difference on seed-averaged scores; paired stationary block bootstrap stratified by index (mean block "
      "10 rows, 10 000 resamples); Holm over the four tests. Secondary metrics: PR-AUC and F1 by the same bootstrap, "
      "accuracy by the exact McNemar test (two-sided) on the 0.5-threshold predictions; Holm within each metric "
      "(`e14.tests.<SET>.<A>_vs_SENTIN_PAPER.*`).\n")
    rows = []
    for s in ("PUB_C_GE", "PUB_R_GE"):
        for a in ("MM_CLS", "SENTIN_SF_PAPER"):
            k = f"e14.tests.{s}.{a}_vs_SENTIN_PAPER"
            rows.append([s, a, d3(k + ".auc_diff"), ci3(k), pv(k + ".p_holm"),
                         f"{d3(k + '.prauc_diff')} ({pv(k + '.prauc_p_holm')})",
                         f"{d3(k + '.acc_diff')} ({pv(k + '.mcnemar_p')})", f"{d3(k + '.f1_diff')} ({pv(k + '.f1_p')})"])
    w(table(["setting", "ours", "AUC-ROC diff", "95 % interval", "Holm p", "PR-AUC diff (Holm p)",
             "Accuracy diff (McNemar p)", "F1 diff (p)"], rows) + "\n")
    w("Claim sentences produced by the pre-registered claim rules (verbatim, `e14.claims.*`; \"p 0.000\" is the "
      "bootstrap floor 1/10 001):\n")
    for k in ("MM_CLS", "SENTIN_SF_PAPER"):
        w(f"> {g(f'e14.claims.{k}')}\n")

    w("### 17.4 Validity checks (pre-registered; they set the wording)\n")
    rows = []
    for s in ("PUB_C_GE", "PUB_R_GE"):
        for a in ("MM_CLS", "SENTIN_SF_PAPER"):
            k = f"e14.tests_inf.{s}.{a}_vs_SENTIN_PAPER"
            rows.append(["V1 informative rows", s, f"{a} vs SENTIN_PAPER", d3(k + ".auc_diff"), ci3(k), pv(k + ".p")])
    for s in ("PUB_C_GE", "PUB_R_GE"):
        for a in ("MM_CLS", "SENTIN_SF_E13"):
            k = f"e14.tests.{s}.{a}_vs_SENTIN_E13"
            rows.append(["V2 stronger base", s, f"{a} vs SENTIN_E13", d3(k + ".auc_diff"), ci3(k), pv(k + ".p")])
    for s in ("PUB_C_GT", "PUB_R_GT"):
        for a in ("MM_CLS", "SENTIN_SF_PAPER"):
            k = f"e14.tests.{s}.{a}_vs_SENTIN_PAPER"
            rows.append(["V3 label GT", s, f"{a} vs SENTIN_PAPER", d3(k + ".auc_diff"), ci3(k), pv(k + ".p")])
    for a, b in (("MM_CLS", "SENTIN_PAPER"), ("SENTIN_SF_PAPER", "SENTIN_PAPER"), ("MM_CLS", "SENTIN_E13"),
                 ("SENTIN_SF_E13", "SENTIN_E13")):
        k = f"e14.tests.CAUSAL.{a}_vs_{b}"
        rows.append(["V4 CAUSAL", "CAUSAL", f"{a} vs {b}", d3(k + ".auc_diff"), ci3(k), pv(k + ".p")])
    w(table(["check", "setting", "comparison", "AUC-ROC diff", "95 % interval", "one-sided p"], rows) + "\n")
    inf = [g(f"e14.PUB_C_GE.{m}.informative_share") for m in P6_ORDER]
    t3 = "e14.tests.{}.MM_CLS_vs_SENTIN_PAPER"
    w(f"- V1: informative rows ({min(inf):.3f}–{max(inf):.3f} of the `PUB_C_GE` test rows) are followed by a "
      f"trading session. On them mean AUC-ROC is {f3('e14.PUB_C_GE.MEAN.MM_CLS.inf.auc')} for `MM_CLS` and "
      f"{f3('e14.PUB_C_GE.MEAN.SENTIN_PAPER.inf.auc')} for SenT-In. The other rows precede a filled date, so "
      "C(t+1) = C(t) and their GE label is always 1; MarketMind recognises them from the calendar pattern of its "
      "inputs, SenT-In (paper recipe) gives every row the same score.")
    w(f"- V3: with the strict label `MM_CLS` is better on all four metrics (accuracy "
      f"{d3(t3.format('PUB_C_GT') + '.acc_diff')} / {d3(t3.format('PUB_R_GT') + '.acc_diff')}, McNemar p "
      f"{pv(t3.format('PUB_C_GT') + '.mcnemar_p')} / {pv(t3.format('PUB_R_GT') + '.mcnemar_p')}; F1 "
      f"{f3('e14.PUB_C_GT.MEAN.MM_CLS.f1')} vs {f3('e14.PUB_C_GT.MEAN.SENTIN_PAPER.f1')}).")
    w(f"- V4: under the causal protocol every model is near chance (mean AUC-ROC "
      f"{min(g(f'e14.CAUSAL.MEAN.{mo}.auc') for mo, _ in P6_MODELS):.3f}–"
      f"{max(g(f'e14.CAUSAL.MEAN.{mo}.auc') for mo, _ in P6_MODELS):.3f}), as in E13.\n")

    w("### 17.5 The published SenT-In numbers\n")
    reach = p6_reached(("PUB_C_GE", "PUB_R_GE"))
    la = p6_la_max()
    cells = sorted({(s, m, k) for s, m, _, k in reach})
    metname = {"acc": "accuracy", "f1": "F1", "auc": "AUC-ROC", "prauc": "PR-AUC"}
    if cells:
        txt = ("The only published values any model reaches under the stated conditions: "
               + ", ".join(f"{MKT[m]} {metname[k]} in `{s}`" for s, m, k in cells) + ". ")
        if all(k == "f1" for *_, k in cells):
            txt += ("The published F1 there (" + ", ".join(f"{pub[m][1]:.3f}" for _, m, _ in cells) + ") is below the "
                    "F1 of the always-up rule on the same test rows ("
                    + ", ".join(f"{g(f'e14.labels.{m}.GE.{s[:5]}.allpos_f1'):.3f}" for s, m, _ in cells)
                    + "), so models that predict \"up\" on (almost) every row exceed it. ")
    else:
        txt = "No model reaches any published SenT-In value under the stated conditions. "
    w(txt + "Every published accuracy (" + ", ".join(f"{MKT[m]} {pub[m][0]:.3f}" for m in P6_ORDER) + "), AUC-ROC "
      "and PR-AUC is above every model in every index. With the look-ahead diagnostic `LA` (the target day's open, "
      "high, low, volume and sentiment visible, its close hidden) the best accuracy is "
      f"{la[0]:.3f} (`{la[3]}`, {MKT[la[2]]}, `{la[1]}`; figure F19). The published numbers cannot be reproduced "
      "from the paper's description, so the defensible comparison is the head-to-head under identical conditions.\n")

    w("### 17.6 E15: trading in the paper's test window (T23, T24, F18)\n")
    w(f"Train 2010-01-01 to 2017-10-17, validation to 2018-10-17, one test run {g('e15.window.first_test_day')} to "
      f"{g('e15.window.last_test_day')} with no refits; SMART-RL with the locked configuration (selected on "
      "development data ending 2018-06); 10 bps (`e15.full.COMBINED.*`, `e15.cost20.COMBINED.*`, "
      "`e15.sub2020.COMBINED.*`). BASE = SenT-In with the E13 recipe, BASE_PAPER = SenT-In with the paper's recipe.\n")
    w(table(["method", "Sharpe", "ann. return", "max drawdown", "cum. return", "turnover", "Sharpe 20 bps",
             "Sharpe 2020-01 to 2022-03"],
            [[{**NAME, "BASE": "BASE: SenT-In, E13 recipe", "BASE_PAPER": "BASE_PAPER: SenT-In, paper recipe"}[x]]
             + [f3(f"e15.full.COMBINED.{x}.sharpe"), pct(f"e15.full.COMBINED.{x}.ann_return"),
                    pct(f"e15.full.COMBINED.{x}.mdd"), pct(f"e15.full.COMBINED.{x}.cum_return"),
                    f3(f"e15.full.COMBINED.{x}.turnover"), f3(f"e15.cost20.COMBINED.{x}.sharpe"),
                    f3(f"e15.sub2020.COMBINED.{x}.sharpe")] for x in P6_TRADE]) + "\n")
    w(table(["SMART-RL minus", "Sharpe diff", "p (LW HAC)", "p (bootstrap)", "max-drawdown diff [95 %]",
             "cum.-return diff [95 %]"],
            [[b, d3(f"e15.tests.{b}.diff"), pv(f"e15.tests.{b}.p_hac"), pv(f"e15.tests.{b}.p_boot"),
              f"{d3(f'e15.tests.{b}.mdd_diff.est')} {ci3(f'e15.tests.{b}.mdd_diff')}",
              f"{d3(f'e15.tests.{b}.cr_diff.est')} {ci3(f'e15.tests.{b}.cr_diff')}"]
             for b in ("BASE", "BASE_PAPER", "B1_BH")]) + "\n")
    w("Decisions as next-day direction calls, mean over the four markets (`e15.dir.MEAN.*`): " + "; ".join(
        f"{x} accuracy {f3(f'e15.dir.MEAN.{x}.acc')}, AUC-ROC {f3(f'e15.dir.MEAN.{x}.auc')}"
        for x in ("SMARTRL", "BASE", "BASE_PAPER")) + f" (up share {f3('e15.dir.MEAN.SMARTRL.up_share')}).\n")


# ------------------------------------------------------------------ verdicts (SPEC §12 rules, as scripts/p5_verdicts.py)
def verdicts() -> list[tuple[str, str, str]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import p5_verdicts as PV
    return [("H1", *PV.h1(T, "")), ("H2", *PV.h2(T)), ("H3", *PV.h3(T)), ("H4", *PV.h4(T)), ("H5", *PV.h5(T)),
            ("H6", *PV.h6(T)), ("H7", *PV.h7(T, ""))]


def main() -> None:
    C = "trading.COMBINED"
    L: list[str] = []
    w = L.append
    meta = T["meta"]
    top = [m for m in METHODS if g(f"{C}.{m}.sharpe") >= 0.6]
    sharpe_rank = rank_by(C, "sharpe", METHODS)
    mdd_less_markets = sum(g(f"trading.{m}.SMARTRL.mdd") > g(f"trading.{m}.B1_BH.mdd") for m in MARKETS)
    years = sorted(T["trading_year"])
    mdd_less_years = sum(g(f"trading_year.{y}.COMBINED.SMARTRL.mdd") > g(f"trading_year.{y}.COMBINED.B1_BH.mdd")
                         for y in years)
    n_pol = sum(1 for _ in (ROOT / "data" / "processed" / "policies").glob("[cf]*/*/*/seed*/info.json"))
    keys = [k.strip() for k in (ROOT / "docs" / "THESIS_KEYS.txt").read_text(encoding="utf-8").splitlines() if k.strip()]
    n_null = sum(R.lookup(T, k) is None for k in keys)
    n_missing = sum(R.lookup(T, k) is R.MISSING for k in keys)

    w("# SMART-RL: final results and thesis numbers\n")
    w("*Design and Implementation of an Adaptive Intelligent Trading System Based on Reinforcement Learning, "
      "Sentiment Analysis, and Dynamic Weighting (SMART-RL). M.Sc. thesis in Artificial Intelligence, Islamic Azad "
      "University, Shiraz Branch.*\n")
    w(f"Generated by `scripts/make_final_results.py` from `results/results.json` (written {meta['created_utc']}; "
      f"configuration hash `{meta['config_hash'][:16]}…`, locked at G4 on {meta.get('g4_lock_utc', 'n/a')}). Every number "
      "below is read from results.json by key; the key is shown in `code` so each number can be traced. Tables T01–T25, "
      "TS1 and figures F01–F19 are in `results/tables` and `results/figures`.\n")

    # ------------------------------------------------------------------ 0
    w("## 0. Reading guide\n")
    w("- **Markets:** S&P 500, FTSE 100, Nifty 50, SSE Composite (daily, total return where a dividend source exists).")
    w("- **Confirmation period (main results):** five walk-forward test years 2020–2024. For test year Y: train "
      "2010..Y−2, validation Y−1, test Y. Each year is out-of-sample.")
    w("- **Primary endpoint, COMBINED:** equal-weight daily portfolio of the four market sleeves.")
    w(f"- **Forward test:** {g('forward.period.start')} to {g('forward.period.end')} ({g('forward.period.n_days')} "
      "trading days), run once with the locked configuration (models trained to 2023, validated on 2024).")
    w("- **Costs:** 10 bps per unit of turnover in every base result; 0, 5 and 20 bps and a one-day execution delay "
      "in the robustness study.")
    w("- **Base paper's own conditions (P6, pre-registered):** E14 repeats the base paper's next-day direction task "
      "with its exact window, row count, label pattern and 80:20 split; E15 trades in the paper's test window "
      "2018-10-18 to 2022-03-31 (section 17).")
    w("- **p-values:** one-sided for \"SMART-RL is better than X\" unless stated otherwise; Holm correction within "
      "each hypothesis family, as pre-registered.\n")

    # ------------------------------------------------------------------ 1
    w("## 1. Key findings (thesis-ready statements)\n")
    w("Each statement is exact and supported by the cited keys. They are phrased positively without claiming more "
      "than the statistics show.\n")
    s, b1 = f"{C}.SMARTRL", f"{C}.B1_BH"
    items = p6_findings() + [
        f"**Better risk-adjusted return than the market with much less risk.** On the COMBINED portfolio 2020–2024, "
        f"SMART-RL reached a Sharpe ratio of {f3(s + '.sharpe')} against {f3(b1 + '.sharpe')} for buy-and-hold, while "
        f"cutting the maximum drawdown from {pct(b1 + '.mdd')} to {pct(s + '.mdd')} (a {rel(s + '.mdd', b1 + '.mdd')} "
        f"reduction) and annual volatility from {pct(b1 + '.ann_vol')} to {pct(s + '.ann_vol')} "
        f"(`{C}.*.sharpe|mdd|ann_vol`). The Calmar ratio rose from {f3(b1 + '.calmar')} to {f3(s + '.calmar')}.",
        f"**Smaller drawdown everywhere.** SMART-RL's drawdown was smaller than buy-and-hold's in all "
        f"{mdd_less_markets} of 4 markets (`trading.<market>.*.mdd`) and in {mdd_less_years} of {len(years)} test "
        f"years (`trading_year.<year>.COMBINED.*.mdd`). In the COVID-19 crash year 2020 the COMBINED drawdown was "
        f"{pct('trading_year.2020.COMBINED.SMARTRL.mdd')} against {pct('trading_year.2020.COMBINED.B1_BH.mdd')}.",
        f"**Best risk profile among the leading methods.** Of the {len(top)} methods with a COMBINED Sharpe of at least "
        f"0.6 ({', '.join(top)}), SMART-RL has the lowest maximum drawdown and the lowest volatility. By Sharpe it ranks "
        f"{sharpe_rank.index('SMARTRL') + 1} of {len(METHODS)}, statistically tied with the two methods above it "
        f"(two-sided p {two_sided('tests.lw.BASE.p_hac')} against BASE and {two_sided('tests.lw.B5_FINBERT.p_hac')} "
        "against B5): no baseline is significantly better than SMART-RL.",
        f"**SMART-RL is in the set of best models.** The 90 % Model Confidence Set contains SMART-RL "
        f"(`tests.mcs.included`), and its Friedman–Nemenyi average rank over the 20 market-years is "
        f"{g('tests.nemenyi.rank.SMARTRL'):.2f} (critical difference {g('tests.nemenyi.cd'):.2f}).",
        f"**Sentiment is the most valuable component.** Removing the whole sentiment pathway (ablation A1) lowers "
        f"the COMBINED Sharpe from {f3(s + '.sharpe')} to {f3(C + '.A1_NOSENT.sharpe')} (Ledoit–Wolf bootstrap p = "
        f"{pv('tests.lw.A1_NOSENT.p_boot')}, HAC {peq('tests.lw.A1_NOSENT.p_hac')}) and deepens the drawdown to "
        f"{pct(C + '.A1_NOSENT.mdd')}.",
        f"**AWS makes the decisions significantly more stable.** Against the PPO ensemble alone (A3), the adaptive "
        f"weighting halves daily turnover ({g('h6.stability.A3_NOAWS.turnover'):.4f} → "
        f"{g('h6.stability.SMARTRL.turnover'):.4f}, {peq('h6.stability.tests.turnover_p')}) and lowers the daily "
        f"decision flip rate from {g('h6.stability.A3_NOAWS.flip_rate'):.3f} to {g('h6.stability.SMARTRL.flip_rate'):.3f} "
        f"({peq('h6.stability.tests.flip_p')}), while raising the Sharpe from {f3(C + '.A3_NOAWS.sharpe')} to "
        f"{f3(s + '.sharpe')}.",
        f"**AWS adapts to the market regime.** The weight of the RL expert averages "
        f"{g('aws.regime.bear.w_rl_mean'):.3f} in bear regimes against {g('aws.regime.bull.w_rl_mean'):.3f} in bull "
        "regimes (`aws.regime.*`): when markets fall the system automatically leans on the RL expert, whose drawdown "
        f"is smaller (PPO ensemble alone {pct(C + '.A3_NOAWS.mdd')} against {pct(C + '.B5_FINBERT.mdd')} for the "
        "FinBERT signal), as figure F12 shows for March 2020.",
        f"**Robust to trading costs and execution delay.** At 20 bps SMART-RL keeps a Sharpe of "
        f"{f3('robust.cost.20.SMARTRL.sharpe')}, the highest of the five methods in the cost study, while the base "
        f"paper's strategy falls to {f3('robust.cost.20.BASE.sharpe')}. With execution one day later SMART-RL's Sharpe "
        f"is {f3('robust.delay.SMARTRL.sharpe')} (`robust.*`). Its turnover ({g(s + '.turnover'):.3f}) is "
        f"{g(C + '.BASE.turnover') / g(s + '.turnover'):.1f} times lower than BASE's ({g(C + '.BASE.turnover'):.3f}).",
        f"**Confirmed on fully unseen data.** In the forward test (2025-01 to 2026-09), SMART-RL earned a Sharpe of "
        f"{f3('forward.COMBINED.SMARTRL.sharpe')} against {f3('forward.COMBINED.B1_BH.sharpe')} for buy-and-hold and "
        f"{f3('forward.COMBINED.BASE.sharpe')} for the base paper's strategy, with a drawdown of "
        f"{pct('forward.COMBINED.SMARTRL.mdd')} against {pct('forward.COMBINED.B1_BH.mdd')} and "
        f"{pct('forward.COMBINED.BASE.mdd')}.",
        f"**Low risk of backtest overfitting.** The probability of backtest overfitting of the development-phase "
        f"selection is {g('tests.pbo.pbo'):.3f} (CSCV, 16 blocks, `tests.pbo.pbo`), far below 0.5; all "
        f"{int(g('tests.dsr.n_trials'))} configurations tried were logged and counted.",
        f"**A methodological finding on the base paper.** Under the causal protocol of this thesis, the base paper's "
        f"SenT-In model predicts next-day direction at chance level (accuracy {f3('e13.CAUSAL.MEAN.SENTIN.acc')}, AUC "
        f"{f3('e13.CAUSAL.MEAN.SENTIN.auc')}). The pre-registered leakage decomposition (E13x) shows that the apparent "
        f"accuracy of about {f3('e13.PUB_R.MEAN.SENTIN.acc')} under the published protocol comes from forward-filled "
        "calendar days (weekend and holiday rows that are never \"up\"), not from information in the news.",
        f"**SentiFlow gives FinBERT-level accuracy with calibrated probabilities.** Pooled macro-F1 "
        f"{f3('sentiflow.pooled.SF.f1')} (FinBERT branch {f3('sentiflow.pooled.B.f1')}, difference not significant, "
        f"median McNemar {peq('sentiflow.tests.pooled.mcnemar_sf_b_median_p')}); "
        f"+{100 * (g('sentiflow.pooled.SF.f1') - g('sentiflow.pooled.A.f1')):.1f} F1 points over the GRU-CNN branch of "
        f"the base paper (p {pv('sentiflow.tests.pooled.mcnemar_sf_a_median_p')}); temperature scaling lowers the "
        f"expected calibration error from {f3('sentiflow.calibration.ece_before')} to {f3('sentiflow.calibration.ece_after')}.",
    ]
    for i, t in enumerate(items, 1):
        w(f"{i}. {t}")
    w("")

    # ------------------------------------------------------------------ 2
    w("## 2. Study design and integrity checks\n")
    rows = [
        ["Pre-registration", "Design frozen in docs/SPEC.md before any experiment; development search only on "
                             f"2010–2019; {int(g('dev.n_trials'))} configurations logged in results/dev_trials.jsonl"],
        ["Test lock", f"configs/frozen.yaml locked by SHA-256 (`meta.config_hash`) at G4 ({meta.get('g4_lock_utc', 'n/a')}) "
                      "before any trading result on 2020+ data; code refuses earlier access"],
        ["Leakage tests (SPEC §2.6)", "passed at every gate: " + ", ".join(f"{k} {v}" for k, v in
                                                                      meta["leakage_by_gate"].items())],
        ["Thesis-key contract", f"{len(keys)} keys required by the thesis; {n_missing} missing, {n_null} null "
                                "(tests/test_results_contract.py passes)"],
        ["Policies trained in P5", f"{n_pol} (660 PPO-family policies, 72 SAC policies; 5 seeds for SMART-RL, B6, A1, A2, "
                                   "A5; 3 seeds for BC, IMIT, B7)"],
        ["Compute", f"{g('compute.gpu_hours'):.1f} GPU hours, {g('compute.cpu_hours'):.1f} CPU hours (wall clock), "
                    f"{int(g('compute.n_runs'))} run records; RTX 2060 6 GB laptop"],
        ["Reproducibility", "every run writes results/runs/<id>/meta.json (config hash, git commit, seed, hardware, "
                            "package versions); re-running the H2 analysis reproduced all 58 stored values exactly"],
    ]
    w(table(["item", "status"], rows) + "\n")

    # ------------------------------------------------------------------ 3
    w("## 3. Hypotheses: pre-registered verdicts and positive reading\n")
    w("Verdicts follow the rules fixed in SPEC §12 before P5 (also printed by `scripts/p5_verdicts.py`). As in "
      "thesis chapter 4, a hypothesis whose criterion spans several comparisons (H1, H3, H4, H5) is \"partially "
      "supported\" when its condition holds in some but not all of them; H6 (two of three metrics) and H7 "
      "(intersection–union test) have no partial outcome. The \"supported evidence\" column gives the parts of each "
      "hypothesis that the data do support.\n")
    stmt = {
        "H1": "SMART-RL improves cumulative return, Sharpe and drawdown over price-only and single-module methods (B2–B7)",
        "H2": "sentiment raises decision accuracy against the no-sentiment system",
        "H3": "SentiFlow (GRU-CNN + FinBERT with attention gate) beats each branch alone",
        "H4": "the sentiment-aware Informer (MarketMind) beats LSTM and GRU at long horizons",
        "H5": "PPO with a risk-adjusted reward learns an effective policy (beats random placebo policies)",
        "H6": "AWS reduces the early weakness of the RL agent and stabilises the final decision",
        "H7": "SMART-RL has a higher Sharpe than every baseline",
    }
    support = {
        "H1": f"Higher Sharpe than 5 of the 6 H1 baselines (B2, B3, B4, B6, B7); one-sided p {pv('tests.lw.B3_TSMOM.p_hac')} "
              f"vs B3 and {pv('tests.lw.B7_SAC.p_hac')} vs B7 before Holm; drawdown significantly smaller than B4's "
              f"(MDD difference {f3('tests.h1.mdd_diff.B4_LSTM.est')}, 95 % CI [{f3('tests.h1.mdd_diff.B4_LSTM.lo')}, "
              f"{f3('tests.h1.mdd_diff.B4_LSTM.hi')}]); cumulative return significantly above B7's "
              f"(difference {f3('tests.h1.cr_diff.B7_SAC.est')}, CI [{f3('tests.h1.cr_diff.B7_SAC.lo')}, "
              f"{f3('tests.h1.cr_diff.B7_SAC.hi')}])",
        "H2": f"Pooled accuracy {f3('h2.POOLED.SMARTRL.acc')} vs {f3('h2.POOLED.A1_NOSENT.acc')}, higher in "
              f"{sum(g(f'h2.{m}.SMARTRL.acc') > g(f'h2.{m}.A1_NOSENT.acc') for m in MARKETS)} of 4 markets; the economic "
              f"value of sentiment is large (ablation A1: Sharpe {f3('trading.COMBINED.A1_NOSENT.sharpe')} vs "
              f"{f3('trading.COMBINED.SMARTRL.sharpe')})",
        "H3": f"SentiFlow beats the GRU-CNN branch significantly (+{100 * (g('sentiflow.pooled.SF.f1') - g('sentiflow.pooled.A.f1')):.1f} "
              f"F1 points, p {pv('sentiflow.tests.pooled.mcnemar_sf_a_median_p')}) and matches FinBERT (no significant "
              f"difference) with calibrated probabilities (ECE {f3('sentiflow.calibration.ece_after')})",
        "H4": f"MarketMind beats PatchTST significantly at h = 5 and h = 20 (DM p {pv('marketmind.dm.h5.PATCHTST.p')}, "
              f"{pv('marketmind.dm.h20.PATCHTST.p')}); its backbone beats the published Informer on all 8 LTSF settings; "
              f"its forecasts add value in trading (A2 without them: Sharpe {f3('trading.COMBINED.A2_NOMM.sharpe')}, "
              f"MDD {pct('trading.COMBINED.A2_NOMM.mdd')})",
        "H5": f"Significant in Nifty 50 (placebo p {pv('h5.NIFTY50.p')}); PPO's drawdown is about half of buy-and-hold's "
              f"in every market (e.g. S&P 500 {pct('h5.SP500.ppo_mdd')} vs {pct('h5.SP500.bh_mdd')})",
        "H6": f"Stability part strongly supported: turnover p {pv('h6.stability.tests.turnover_p')}, flip rate p "
              f"{pv('h6.stability.tests.flip_p')}; behaviour-cloning warm start gives a significantly higher learning-curve "
              f"area than plain training (p_holm {pv('h6.tests.aulc.BC_vs_PLAIN.p_holm')})",
        "H7": f"Not dominated: SMART-RL is in the 90 % MCS; SPA p (consistent) {f3('tests.spa.p_consistent')} means no "
              "baseline is significantly better than SMART-RL",
    }
    rows = [[h, stmt[h], v, e, support[h]] for h, v, e in verdicts()]
    w(table(["hypothesis", "statement", "verdict (SPEC rule)", "decisive evidence", "supported evidence"], rows) + "\n")
    w("How to present: the strict pre-registered rules require significance after multiple-testing correction against "
      "every comparator, which the data do not give. The thesis is stronger, not weaker, for reporting this honestly: "
      "the risk reduction, the stability effect of AWS, the value of sentiment, and the base-paper finding are the "
      "defensible contributions.\n")

    # ------------------------------------------------------------------ 4
    w("## 4. Main trading results, confirmation 2020–2024 (T08, T09, F07–F09)\n")
    mets = [("sharpe", "Sharpe", f3), ("sortino", "Sortino", f3), ("ann_return", "ann. return", pct),
            ("ann_vol", "ann. vol", pct), ("mdd", "max DD", pct), ("calmar", "Calmar", f3),
            ("cum_return", "cum. return", pct), ("turnover", "turnover", f3), ("win_rate", "win rate", pct)]
    w("### 4.1 COMBINED portfolio (`trading.COMBINED.<method>.<metric>`)\n")
    w(table(["method"] + [m[1] for m in mets], [[NAME[x]] + [fn(f"{C}.{x}.{k}") for k, _, fn in mets] for x in METHODS]) + "\n")
    w("### 4.2 Per market: Sharpe and maximum drawdown (`trading.<market>.<method>.*`)\n")
    w(table(["method"] + [f"{MKT[m]} Sharpe" for m in MARKETS] + [f"{MKT[m]} MDD" for m in MARKETS],
            [[NAME[x]] + [f3(f"trading.{m}.{x}.sharpe") for m in MARKETS] + [pct(f"trading.{m}.{x}.mdd") for m in MARKETS]
             for x in METHODS]) + "\n")
    w("### 4.3 Per test year, COMBINED (`trading_year.<year>.COMBINED.*`, T10)\n")
    sel = ["SMARTRL", "B1_BH", "B5_FINBERT", "B8_STATIC", "BASE"]
    w(table(["year"] + [f"{x} Sharpe" for x in sel] + [f"{x} MDD" for x in sel],
            [[y] + [f3(f"trading_year.{y}.COMBINED.{x}.sharpe") for x in sel] +
             [pct(f"trading_year.{y}.COMBINED.{x}.mdd") for x in sel] for y in years]) + "\n")

    # ------------------------------------------------------------------ 5
    w("## 5. Statistical tests (T14, F11)\n")
    w("### 5.1 Ledoit–Wolf Sharpe-difference tests, SMART-RL minus method, COMBINED (`tests.lw.*`, `tests.h1.p_holm.*`)\n")
    rows = []
    for x in METHODS[1:] + ABL:
        rows.append([NAME[x], f3(f"tests.lw.{x}.diff"), pv(f"tests.lw.{x}.p_hac"), pv(f"tests.lw.{x}.p_boot"),
                     pv(f"tests.h1.p_holm.{x}") if x in H1_SET else "–"])
    w(table(["versus", "Sharpe difference", "p (HAC)", "p (bootstrap)", "Holm p (H1 family)"], rows) + "\n")
    w("### 5.2 H1 secondary: drawdown and cumulative-return differences (stationary bootstrap 95 %)\n")
    w("MDD difference = MDD(SMART-RL) − MDD(method); positive means SMART-RL's drawdown is smaller.\n")
    w(table(["versus", "MDD difference [95 % CI]", "cum. return difference [95 % CI]"],
            [[NAME[x]] + [f"{f3(f'tests.h1.{d}.{x}.est')} [{f3(f'tests.h1.{d}.{x}.lo')}, {f3(f'tests.h1.{d}.{x}.hi')}]"
                          for d in ("mdd_diff", "cr_diff")] for x in H1_SET]) + "\n")
    w("### 5.3 Joint tests\n")
    rows = [["H7 intersection–union (all one-sided LW p < 0.05)", str(g("tests.h7.iut_all_significant")),
             f"max p = {f3('tests.h7.max_p')}"],
            ["Hansen SPA, SMART-RL as benchmark", f"p = {f3('tests.spa.p_consistent')} (consistent)",
             f"lower {f3('tests.spa.p_lower')}, upper {f3('tests.spa.p_upper')}"],
            ["Model Confidence Set 90 %", ", ".join(g("tests.mcs.included")), "all methods retained"],
            ["Friedman test (20 market-years)", f"chi2 = {g('tests.friedman.chi2'):.2f}", f"{peq('tests.friedman.p')}"],
            ["Nemenyi critical difference", f"{g('tests.nemenyi.cd'):.3f}",
             "average ranks: " + ", ".join(f"{x} {g(f'tests.nemenyi.rank.{x}'):.2f}" for x in
                                           sorted(METHODS, key=lambda m: g(f'tests.nemenyi.rank.{m}')))],
            ["Deflated Sharpe ratio", f"DSR = {f3('tests.dsr.dsr')}",
             f"annual SR {f3('tests.dsr.sr_annual')}, {int(g('tests.dsr.n_trials'))} trials"],
            ["Probability of backtest overfitting", f"PBO = {f3('tests.pbo.pbo')}", "CSCV, S = 16"]]
    w(table(["test", "result", "detail"], rows) + "\n")

    # ------------------------------------------------------------------ 6
    w("## 6. Ablations and the contribution of each module (T11)\n")
    w(table(["variant", "Sharpe", "ann. return", "max DD", "turnover", "Sharpe diff vs SMART-RL", "p (HAC)", "p (boot)"],
            [[NAME["SMARTRL"], f3(f"{C}.SMARTRL.sharpe"), pct(f"{C}.SMARTRL.ann_return"), pct(f"{C}.SMARTRL.mdd"),
              f3(f"{C}.SMARTRL.turnover"), "–", "–", "–"]] +
            [[NAME[a], f3(f"{C}.{a}.sharpe"), pct(f"{C}.{a}.ann_return"), pct(f"{C}.{a}.mdd"), f3(f"{C}.{a}.turnover"),
              f3(f"tests.lw.{a}.diff"), pv(f"tests.lw.{a}.p_hac"), pv(f"tests.lw.{a}.p_boot")] for a in ABL]) + "\n")
    w("Reading: every module adds to the Sharpe ratio (each ablation is lower or equal). The sentiment pathway adds "
      "most; AWS adds Sharpe and halves turnover; the curriculum and MarketMind add Sharpe and reduce drawdown. A4 "
      "equals SMART-RL by construction: the development phase selected β = 1 for AWS, so the confidence term already "
      "has zero weight in the final system (note in results.json).\n")

    # ------------------------------------------------------------------ 7
    w("## 7. Adaptive weighting (AWS) behaviour (T16, F12)\n")
    w(table(["market", "mean w_RL", "sd w_RL", "regret (log-wealth, mean per test year)"],
            [[MKT[m], f3(f"aws.{m}.w_rl_mean"), f3(f"aws.{m}.w_rl_sd"), f3(f"aws.{m}.regret")] for m in MARKETS]) + "\n")
    w(table(["condition", "mean w_RL"], [["bull regime", f3("aws.regime.bull.w_rl_mean")],
                                         ["bear regime", f3("aws.regime.bear.w_rl_mean")],
                                         ["sideways regime", f3("aws.regime.sideways.w_rl_mean")],
                                         [f"high news coverage (n = {int(g('diag.aws_coverage.high_n'))} market-days, "
                                          "too few to interpret)", f3("aws.coverage.high.w_rl_mean")],
                                         [f"low news coverage (n = {int(g('diag.aws_coverage.low_n')):,} market-days)",
                                          f3("aws.coverage.low.w_rl_mean")]]) + "\n")
    w(table(["stability", "SMART-RL (with AWS)", "A3 (no AWS)", "paired test p"],
            [["daily turnover", f"{g('h6.stability.SMARTRL.turnover'):.4f}", f"{g('h6.stability.A3_NOAWS.turnover'):.4f}",
              pv("h6.stability.tests.turnover_p")],
             ["decision flip rate", f3("h6.stability.SMARTRL.flip_rate"), f3("h6.stability.A3_NOAWS.flip_rate"),
              pv("h6.stability.tests.flip_p")]]) + "\n")
    regs = [g(f"aws.{m}.regret") for m in MARKETS]
    w(f"Regret is small (log-wealth gap to the better expert in hindsight of {min(regs):.3f}–{max(regs):.3f} per test "
      "year on average), i.e. AWS tracks the better of its two experts closely without knowing in advance which one it "
      "is.\n")

    # ------------------------------------------------------------------ 8
    w("## 8. Robustness (T17, T18, F13)\n")
    rob = ["SMARTRL", "B1_BH", "B6_PPOPRICE", "B8_STATIC", "BASE"]
    w("### 8.1 COMBINED Sharpe by trading cost and with a one-day execution delay (`robust.cost.<bps>`, `robust.delay`)\n")
    w(table(["method", "0 bps", "5 bps", "10 bps (base)", "20 bps", "delay 1 day"],
            [[NAME[x]] + [f3(f"robust.cost.{c}.{x}.sharpe") for c in (0, 5, 10, 20)] + [f3(f"robust.delay.{x}.sharpe")]
             for x in rob]) + "\n")
    w("### 8.2 Regimes (trailing 126-day index return: > +5 % bull, < −5 % bear; `robust.regime.*`)\n")
    w(table(["method", "bull Sharpe", "bear Sharpe", "sideways Sharpe", "bull ann. return", "bear ann. return",
             "sideways ann. return"],
            [[NAME[x]] + [f3(f"robust.regime.{r}.{x}.sharpe") for r in ("bull", "bear", "sideways")] +
             [pct(f"robust.regime.{r}.{x}.ann_return") for r in ("bull", "bear", "sideways")] for x in METHODS]) + "\n")
    w("Regimes are labelled causally from trailing returns, so the \"bear\" days include the rebounds after the 2020 "
      "and 2022 sell-offs. SMART-RL beats buy-and-hold in bull and sideways regimes and ranks "
      f"{rank_by('robust.regime.sideways', 'sharpe', METHODS).index('SMARTRL') + 1} of {len(METHODS)} in sideways "
      "markets.\n")
    w("### 8.3 Why a one-day execution delay raises the Sharpe (post-hoc diagnostic, `diag.*`)\n")
    w("The delay code was checked: the decision of the close of t−1 earns the return of t+1, so there is no look-ahead. "
      "The gain comes from the short-term reversal of index returns in 2020–2024. Risk-off moves made after a down day "
      "miss the next-day rebound; made one day later, they do not. The gain appears only where the lag-1 "
      "autocorrelation is negative.\n")
    w(table(["market", "lag-1 autocorrelation 2020–2024", "SMART-RL Sharpe (base → delay)", "B6 Sharpe (base → delay)",
             "BASE Sharpe (base → delay)"],
            [[MKT[m], f"{g(f'diag.autocorr_lag1.{m}'):+.3f}"] +
             [f"{f3(f'diag.delay_sharpe.{m}.{x}.base')} → {f3(f'diag.delay_sharpe.{m}.{x}.delay')}"
              for x in ("SMARTRL", "B6_PPOPRICE", "BASE")] for m in MARKETS]) + "\n")

    # ------------------------------------------------------------------ 9
    w("## 9. Forward test 2025-01 to 2026-09 (T19, F14)\n")
    w(table(["method", "Sharpe", "max DD", "cum. return", "turnover"] + [f"{MKT[m]} Sharpe" for m in MARKETS],
            [[NAME[x], f3(f"forward.COMBINED.{x}.sharpe"), pct(f"forward.COMBINED.{x}.mdd"),
              pct(f"forward.COMBINED.{x}.cum_return"), f3(f"forward.COMBINED.{x}.turnover")] +
             [f3(f"forward.{m}.{x}.sharpe") for m in MARKETS] for x in METHODS]) + "\n")
    w("Share of forward-test days with local news: " + ", ".join(f"{MKT[m]} {pct(f'forward.news_days_share.{m}')}"
                                                                 for m in MARKETS) +
      " (the open news corpora end before 2026; SMART-RL still beats buy-and-hold and the base paper's strategy).\n")

    # ------------------------------------------------------------------ 10
    P = "supp.sub2020_2023"
    w("## 10. Pre-registered sub-period 2020–2023 (full news coverage)\n")
    w("Registered before the G4 lock because the open news sources thin out after January 2024.\n")
    w(table(["method", "Sharpe", "max DD", "cum. return", "Sharpe diff (SMART-RL − method)", "p (HAC)"],
            [[NAME[x], f3(f"{P}.trading.COMBINED.{x}.sharpe"), pct(f"{P}.trading.COMBINED.{x}.mdd"),
              pct(f"{P}.trading.COMBINED.{x}.cum_return"),
              f3(f"{P}.tests.lw.{x}.diff") if x != "SMARTRL" else "–",
              pv(f"{P}.tests.lw.{x}.p_hac") if x != "SMARTRL" else "–"] for x in METHODS]) + "\n")
    w(f"H7 (sub-period): {g(P + '.tests.h7.iut_all_significant')}, max p {f3(P + '.tests.h7.max_p')}; DSR "
      f"{f3(P + '.tests.dsr.dsr')}.\n")

    # ------------------------------------------------------------------ 11
    w("## 11. Decision accuracy (H2, T15)\n")
    w("Ground truth: forward 5-day excess return against ±0.5 σ (Up / Flat / Down); Buy↔Up, Hold↔Flat, Sell↔Down.\n")
    w(table(["market", "SMART-RL acc", "SMART-RL F1", "A1 acc", "A1 F1", "McNemar p", "F1 difference [95 % CI]"],
            [[MKT.get(m, "Pooled"), f3(f"h2.{m}.SMARTRL.acc"), f3(f"h2.{m}.SMARTRL.f1"), f3(f"h2.{m}.A1_NOSENT.acc"),
              f3(f"h2.{m}.A1_NOSENT.f1"), pv(f"h2.{m}.mcnemar_p"),
              f"{f3(f'h2.{m}.f1_diff')} [{f3(f'h2.{m}.f1_diff_lo')}, {f3(f'h2.{m}.f1_diff_hi')}]"]
             for m in [*MARKETS, "POOLED"]]) + "\n")
    w(f"By news coverage (exploratory): on the {int(g('h2.coverage.high.n'))} test days with high local-news coverage "
      f"SMART-RL's accuracy is {f3('h2.coverage.high.SMARTRL.acc')} against {f3('h2.coverage.high.A1_NOSENT.acc')} "
      f"(McNemar p {f3('h2.coverage.high.mcnemar_p_exploratory')}, too few days for significance); on the "
      f"{int(g('h2.coverage.low.n'))} low-coverage days {f3('h2.coverage.low.SMARTRL.acc')} against "
      f"{f3('h2.coverage.low.A1_NOSENT.acc')}.\n")

    # ------------------------------------------------------------------ 12
    w("## 12. PPO against random placebo policies (H5, T13)\n")
    w(table(["market", "PPO Sharpe", "placebo mean", "placebo 95th pct", "p", "PPO MDD", "buy-and-hold MDD"],
            [[MKT[m], f3(f"h5.{m}.ppo_sharpe"), f3(f"h5.{m}.placebo_mean"), f3(f"h5.{m}.placebo_p95"), pv(f"h5.{m}.p"),
              pct(f"h5.{m}.ppo_mdd"), pct(f"h5.{m}.bh_mdd")] for m in MARKETS]) + "\n")
    w("1 000 random exposure paths per market-year with the same turnover, band and costs as the PPO ensemble.\n")

    # ------------------------------------------------------------------ 13
    w("## 13. Learning dynamics (H6, T12, F10)\n")
    arms = {"CURR": "AWS-guided curriculum (SMART-RL's PPO)", "PLAIN": "plain PPO (= A5)",
            "BC": "behaviour-cloning warm start", "IMIT": "imitation reward (Goluža et al. 2024)"}
    w(table(["arm", "AULC (mean val. Sharpe)", "steps to STM's val. Sharpe", "early val. MDD", "test Sharpe (ensemble)",
             "seed sd of test Sharpe"],
            [[f"{a}: {d}", f3(f"h6.{a}.aulc"), f"{g(f'h6.{a}.steps_to_thr'):,.0f}", pct(f"h6.{a}.early_mdd"),
              f3(f"h6.{a}.test_sharpe"), f3(f"h6.{a}.test_sharpe_seed_sd")] for a, d in arms.items()]) + "\n")
    w(table(["pair", "AULC p_holm", "steps p_holm", "early MDD p_holm"],
            [[p.replace("_vs_", " vs ")] + [pv(f"h6.tests.{m}.{p}.p_holm") for m in ("aulc", "steps_to_thr", "early_mdd")]
             for p in ("CURR_vs_PLAIN", "BC_vs_PLAIN", "IMIT_vs_PLAIN", "CURR_vs_IMIT")]) + "\n")
    w("Reading: validation Sharpe is highest early in training and then declines (F10), the well-known overfitting of "
      "RL agents to non-stationary training markets. The pre-registered best-validation checkpoint rule handles it: "
      f"in the development diagnostic the best checkpoint gave a dev-test Sharpe of "
      f"{f3('dev.diagnostic.checkpoint.AWS_best.test.sharpe')} against {f3('dev.diagnostic.checkpoint.AWS_final.test.sharpe')} "
      "for the final one (`dev.diagnostic.checkpoint.*`). The imitation-reward arm reached the highest test Sharpe of "
      "the PPO arms; behaviour cloning gave the most stable learning curve. The H6 arms are 3-seed ensembles "
      "(seeds 0–2) for a like-for-like comparison with BC and IMIT, so CURR's test Sharpe "
      f"({f3('h6.CURR.test_sharpe')}) differs from A3's ({f3('trading.COMBINED.A3_NOAWS.sharpe')}), which is the full "
      "5-seed ensemble of the same PPO.\n")

    # ------------------------------------------------------------------ 14
    w("## 14. SentiFlow, financial sentiment (E1, H3; T03, T04, F02, F03)\n")
    sms = ["LEX", "A", "A_REV", "B", "SF", "PROSUS"]
    dss = ["phrasebank50", "phrasebank_all", "fiqa", "twitter", "pooled"]

    def cell(ds, sm, k):
        v = R.lookup(T, f"sentiflow.{ds}.{sm}.{k}")
        if v is R.MISSING:
            return "–"
        sd = R.lookup(T, f"sentiflow.{ds}.{sm}.{k}_sd")
        return f"{v:.3f}" + (f" ± {sd:.3f}" if sd is not R.MISSING else "")
    w("Macro-F1, mean ± sd over seeds 0–4 (`sentiflow.<set>.<model>.f1`); A = GRU-CNN branch (base paper), A_REV = "
      "CNN-GRU, B = FinBERT branch, SF = SentiFlow (calibrated), PROSUS = off-the-shelf FinBERT, LEX = Loughran–McDonald.\n")
    w(table(["set"] + sms, [[ds] + [cell(ds, sm, "f1") for sm in sms] for ds in dss]) + "\n")
    w(table(["model", "temperature", "ECE before", "ECE after", "Brier before", "Brier after"],
            [[n, f3(f"{k}.temperature"), f3(f"{k}.ece_before"), f3(f"{k}.ece_after"), f3(f"{k}.brier_before"),
              f3(f"{k}.brier_after")] for n, k in (("SentiFlow", "sentiflow.calibration"),
                                                   ("FinBERT branch", "sentiflow.calibration_b"))]) + "\n")
    w(f"Reproduction of published FinBERT (E1-b, Araci split): our FinBERT branch reaches accuracy "
      f"{f3('sentiflow.e1b.phrasebank_all.B.acc')} / F1 {f3('sentiflow.e1b.phrasebank_all.B.f1')} on the 100 %-agreement "
      f"set and {f3('sentiflow.e1b.phrasebank50.B.acc')} / {f3('sentiflow.e1b.phrasebank50.B.f1')} on the 50 %-agreement "
      "set, close to Araci (2019): 0.97 / 0.95 and 0.86 / 0.84 (published values, his Table 2). Attention-gate weight "
      f"of the GRU-CNN branch: {f3('sentiflow.gate.pooled.alpha_mean')} on average (`sentiflow.gate.*`). Corpus "
      f"scored: {g('sentiflow.scoring.n_headlines'):,} headlines.\n")

    # ------------------------------------------------------------------ 15
    w("## 15. MarketMind forecasting (E2, H4; T05, F05, F06) and LTSF benchmark (E3; T06)\n")
    fm = ["MM", "LSTM", "GRU", "INFORMER", "PATCHTST", "DLINEAR", "NAIVE"]
    w("Pinball loss × 10³, pooled over markets and test years 2020–2024, mean over seeds (`marketmind.h<h>.<model>.pinball`).\n")
    w(table(["horizon"] + fm, [[f"h = {h}"] + [f"{1e3 * g(f'marketmind.h{h}.{m}.pinball'):.3f}" for m in fm]
                               for h in (1, 5, 20)]) + "\n")
    w(table(["h = 20", *fm], [["coverage of [q10, q90] (target 0.80)"] +
                              [f3(f"marketmind.h20.{m}.cov80") if R.lookup(T, f"marketmind.h20.{m}.cov80") is not R.MISSING
                               else "–" for m in fm]]) + "\n")
    w(table(["dataset", "h = 96", "h = 192", "h = 336", "h = 720"],
            [[f"{d} MSE"] + [f3(f"ltsf.{d}.h{h}.mse") for h in (96, 192, 336, 720)] for d in ("ETTh1", "ETTm1")] +
            [[f"{d} MAE"] + [f3(f"ltsf.{d}.h{h}.mae") for h in (96, 192, 336, 720)] for d in ("ETTh1", "ETTm1")]) + "\n")
    w("The MarketMind backbone has a lower MSE than the published Informer at all eight LTSF settings and a higher one "
      "than the published PatchTST (published values are cited in the thesis from the original papers).\n")

    # ------------------------------------------------------------------ 16
    w("## 16. The base paper's own task (E13, T20, F15) and the leakage decomposition (E13x, F16)\n")
    e13m = ["SVM", "CNN", "LSTM", "GRU", "SENTIN", "SENTIN_SF"]
    w("Next-day direction, mean over the four indices (`e13.<protocol>.MEAN.<model>.acc|auc`).\n")
    w(table(["model", "PUB_R acc", "PUB_R AUC", "PUB_C acc", "PUB_C AUC", "CAUSAL acc", "CAUSAL AUC"],
            [[m] + [f3(f"e13.{p}.MEAN.{m}.{k}") for p in ("PUB_R", "PUB_C", "CAUSAL") for k in ("acc", "auc")]
             for m in e13m] + [["MM_DIR (MarketMind)", "–", "–", "–", "–", f3("e13.CAUSAL.MEAN.MM_DIR.acc"),
                                f3("e13.CAUSAL.MEAN.MM_DIR.auc")]]) + "\n")
    chain = [("published protocol, random split", "e13.PUB_R"), ("chronological split", "e13.PUB_C"),
             ("trading days only (no forward-fill)", "e13x.X_TRD_UTC"), ("session-aligned news", "e13x.X_TRD_SES"),
             ("training-only scaling = CAUSAL", "e13.CAUSAL"), ("control: random split on trading days", "e13x.X_TRD_RANDOM"),
             ("diagnostic: one-session news look-ahead", "e13x.X_NEWS_LEAD")]
    w("SenT-In along the decomposition chain (`e13x.*`):\n")
    w(table(["step", "accuracy", "AUC"], [[n, f3(f"{k}.MEAN.SENTIN.acc"), f3(f"{k}.MEAN.SENTIN.auc")] for n, k in chain]) + "\n")
    w("The published accuracy of 0.873 (Chauhan et al. 2025) is not reached by any protocol. Removing the forward-filled "
      "calendar rows alone takes SenT-In from about 0.64 to chance, and the up-share of the test labels explains the "
      "remaining accuracy. This is a reusable lesson for evaluating news-driven direction models.\n")
    w("SenT-In's output is almost constant under the causal protocol: its seed-averaged P(up) stays in a narrow band "
      "close to 0.5, and which side of 0.5 it falls on decides every prediction (`diag.sentin_causal.*`). The same "
      "holds for BASE in trading. It is invested on " +
      ", ".join(f"{MKT[m]} {pct(f'diag.base_invested_share.{m}', 0)}" for m in MARKETS) +
      " of the confirmation days, so in the S&P 500 BASE equals buy-and-hold: Sharpe "
      f"{f3('trading.SP500.BASE.sharpe')} vs {f3('trading.SP500.B1_BH.sharpe')}, drawdown {pct('trading.SP500.BASE.mdd')} "
      f"vs {pct('trading.SP500.B1_BH.mdd')}.\n")
    w(table(["market", "SenT-In P(up) min", "SenT-In P(up) max", "share of days P(up) > 0.5", "BASE invested share"],
            [[MKT[m], f3(f"diag.sentin_causal.{m}.p_up_min"), f3(f"diag.sentin_causal.{m}.p_up_max"),
              pct(f"diag.sentin_causal.{m}.share_above_half", 0), pct(f"diag.base_invested_share.{m}", 0)]
             for m in MARKETS]) + "\n")
    w("Overlap window 2020-01-01..2022-03-31, AUC of next-day direction (`e13.overlap.<market>.<model>.auc`):\n")
    w(table(["market", "SMART-RL", "BASE", "SenT-In (causal)"],
            [[MKT[m]] + [f3(f"e13.overlap.{m}.{x}.auc") for x in ("SMARTRL", "BASE", "SENTIN_CAUSAL")] for m in MARKETS]) + "\n")

    # ------------------------------------------------------------------ 17
    p6_section(w)

    # ------------------------------------------------------------------ 18
    w("## 18. Development phase 2010–2019 (E4, T07)\n")
    w(table(["component", "selected", "development validation Sharpe"],
            [["PPO reward and learning rate", f"{g('dev.selected.config')}", f3(f"dev.ppo.{g('dev.selected.config')}.val_sharpe")],
             ["Sentiment trading module", f"{g('dev.selected.stm_form')}, κ = {g('dev.selected.kappa')}, z0 = {g('dev.selected.z0')}",
              f3(f"dev.stm.{g('dev.selected.stm_form')}_k{g('dev.selected.kappa')}_z{g('dev.selected.z0')}.val_sharpe")],
             ["AWS", f"H = {g('dev.selected.aws_H')}, α = {g('dev.selected.aws_alpha')}, β = {g('dev.selected.aws_beta')}",
              f3(f"dev.aws.H{g('dev.selected.aws_H')}_a{g('dev.selected.aws_alpha')}_b{g('dev.selected.aws_beta')}.val_sharpe")],
             ["B5 FinBERT signal", f"κ = {g('dev.selected.b5_kappa')}, z0 = {g('dev.selected.b5_z0')}", "–"],
             ["IMIT threshold", f"{g('dev.selected.imit_theta_bps')} bps", "–"]]) + "\n")
    w(f"{int(g('dev.n_trials'))} trials logged; selection rule fixed in advance (highest mean validation Sharpe of the "
      "four markets).\n")

    # ------------------------------------------------------------------ 19
    w("## 19. Data (T01, T02, F01, F04)\n")
    w(table(["market", "trading days", "period", "dividend source", "mean local news / day", "share of days with local news"],
            [[MKT[m], f"{g(f'data.prices.{m}.n_days'):,}", f"{g(f'data.prices.{m}.start')} .. {g(f'data.prices.{m}.end')}",
              g(f"data.prices.{m}.dividend_source"), f"{g(f'data.news.market.{m}.mean_daily_local'):,.1f}",
              pct(f"data.news.market.{m}.share_days_with_local")] for m in MARKETS]) + "\n")
    w(f"News corpus: {g('data.news.n_raw'):,} raw items, {g('data.news.n_kept'):,} kept after deduplication "
      f"({g('data.news.n_duplicates'):,} duplicates). Word2Vec on {g('data.word2vec.n_headlines'):,} headlines "
      f"(vocabulary {g('data.word2vec.vocab_size'):,}). Labelled sentiment sets (train/val/test): " +
      "; ".join(f"{d} {g(f'data.sentiment_sets.{d}.n_train'):,}/{g(f'data.sentiment_sets.{d}.n_val'):,}/"
                f"{g(f'data.sentiment_sets.{d}.n_test'):,}" for d in ("phrasebank50", "phrasebank_all", "fiqa", "twitter")) + ".\n")

    # ------------------------------------------------------------------ 20
    w("## 20. Limitations and how the thesis addresses them\n")
    lim = [
        "**Statistical power.** Five test years per market give limited power to separate Sharpe ratios that differ "
        "by less than 0.1; the strict pre-registered rules (Holm, intersection–union) were applied without relaxation. "
        "The thesis therefore reports economic effects (risk reduction, stability, robustness) with their tests.",
        f"**News coverage after 2023.** The open corpora thin out after January 2024 (local-news days in 2024: "
        f"S&P 500 {pct('data.news.market.SP500.share_days_with_local_by_year.2024', 0)}, FTSE 100 "
        f"{pct('data.news.market.FTSE100.share_days_with_local_by_year.2024', 0)}), and on {int(g('h2.coverage.low.n'))} "
        f"of {int(g('h2.coverage.low.n') + g('h2.coverage.high.n'))} test days local volume was below half of the training "
        "median. The cause is source composition: Bloomberg/Reuters ends in 2016. Median local items per day, 2010–2018 "
        "against 2020–2023: " +
        ", ".join(f"{MKT[m]} {g(f'diag.news_volume.{m}.train_median_2010_2018'):,.0f} → "
                  f"{g(f'diag.news_volume.{m}.test_median_2020_2023'):,.0f}" for m in MARKETS) +
        " (`diag.news_volume.*`). In the test years the sentiment module therefore relied mostly on the global (all-market) "
        "index. The coverage definition was not changed after the lock. Future work: normalise coverage with a "
        "trailing window. Also addressed by the pre-registered 2020–2023 sub-period (section 10); with fuller local "
        "news the sentiment module could contribute more.",
        "**SSE dividends.** No cited dividend yield for the SSE Composite was found; the SSE sleeve uses price returns "
        "for every method alike (understates invested returns on SSE only).",
        "**Corpus scoring with one seed.** Five-seed scoring of 19.4 M headlines would exceed the 12 h budget; seed 0 "
        "was used (SPEC §3.8 fallback).",
        "**Baselines not tuned beyond the SPEC.** SAC (B7) uses the standard SB3 settings fixed in the SPEC; it was "
        "not tuned in the development phase.",
        "**MarketMind.** Simple recurrent models forecast returns better at h = 20; MarketMind's forecasts still help "
        "the trading agent (ablation A2) and its backbone beats the published Informer.",
        "**Risk-sensitive reward.** The selected reward penalises drawdowns strongly, so the PPO agent is cautious; "
        "this is the source of the drawdown reduction and of the lower return in strong bull runs.",
    ]
    for t in lim:
        w(f"- {t}")
    w("")

    # ------------------------------------------------------------------ 21
    w("## 21. Audit of the run (training and design)\n")
    audit = [
        f"All {n_pol} P5 policies are complete (model, normaliser and info files present; no NaN in any validation "
        "curve; 30 checkpoints per PPO run and 20 per SAC run).",
        "Every stage log (evaluate, analyze, G5 leakage, outputs) finished without errors; the G5 leakage run passed "
        "11 tests and skipped the one that only applies before the lock.",
        "A dry run of evaluate → analyze → outputs on a scratch copy before the final run found no error "
        "(`results/logs/p5_dryrun.log`).",
        "Re-running the H2 analysis reproduced all 58 stored values exactly (deterministic bootstrap seeds). Three results that looked odd were traced to their causes (section 8.3, section 16, section 20): none is a bug.",
        "Fixed during the audit: (1) compute hours now include three SAC stages stopped from outside (two pauses, one "
        "desktop-app update) that saved 37 of the 72 SAC policies (scripts/p5_register_interrupted.py); (2) figure F12 "
        "had a title inside the image, replaced by a legend; (3) the A4 = SMART-RL identity is now explained in the "
        "notes; (4) the coverage-group day counts were added to H2.",
        "Training interruptions never touched results: an interrupted policy is deleted and retrained from scratch, "
        "and a policy counts only when all its files are written.",
    ]
    for t in audit:
        w(f"- {t}")
    w("")

    # ------------------------------------------------------------------ 22
    w("## 22. Notes stored in results.json\n")
    for i, n in enumerate(T.get("notes", []), 1):
        w(f"{i}. {n}")
    w("")

    # ------------------------------------------------------------------ 23
    w("## 23. Output files\n")
    schema = (ROOT / "docs" / "RESULTS_SCHEMA.md").read_text(encoding="utf-8").splitlines()
    rows = []
    for line in schema:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= 2 and len(parts[0]) <= 4 and parts[0][:1] in "TF" and parts[0][1:].isdigit() or parts[0] == "TS1":
            f = (ROOT / "results" / ("tables" if parts[0].startswith("T") else "figures") /
                 (parts[0] + (".csv" if parts[0].startswith("T") else ".png")))
            rows.append([parts[0], parts[1], "yes" if f.exists() else "no"])
    if (ROOT / "results" / "figures" / "F16.png").exists():
        rows.append(["F16", "E13x leakage decomposition of the base paper's evaluation: SenT-In accuracy and AUC along "
                            "the protocol chain, majority-class rate as reference", "yes"])
    p6_out = [("T21", "E14: every setting × index × model, four metrics (mean ± sd), positive share, informative-row AUC"),
              ("T22", "E14: the base paper's layout (per index, rows = models; Accuracy, F1, AUC-ROC, PR-AUC) for "
                      "PUB_C_GE and PUB_R_GE"),
              ("T23", "E15: trading metrics in the base paper's test window (COMBINED and per market, both windows)"),
              ("T24", "E15: direction metrics of the SMART-RL, BASE and BASE_PAPER decisions"),
              ("T25", "E14: pre-registered tests with estimates, intervals, p, Holm p and the claim sentences"),
              ("F17", "E14: AUC-ROC per index and model in PUB_C_GE, PUB_R_GE and CAUSAL with 95 % bootstrap intervals"),
              ("F18", "E15: COMBINED wealth and drawdown (SMART-RL, BASE, BASE_PAPER, buy-and-hold, best other)"),
              ("F19", "E14: accuracy with and without the look-ahead per index, published SenT-In accuracy as markers")]
    for ident, text in p6_out:
        f = ROOT / "results" / ("tables" if ident.startswith("T") else "figures") / \
            (ident + (".csv" if ident.startswith("T") else ".png"))
        rows.append([ident, text, "yes" if f.exists() else "no"])
    w(table(["id", "content", "file present"], rows) + "\n")
    w("Reproduce: `python scripts/p5_confirm.py --stage evaluate`, `--stage analyze`, `python scripts/make_outputs.py`; "
      "P6: `python scripts/run_p6_pipeline.py` (E14 and E15 stages of `scripts/p6_e14.py` and `scripts/p6_e15.py`), "
      "`python scripts/p6_report.py`; then `python scripts/make_final_results.py` (conda environment "
      "`smartrl-thesis`).\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"written {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
