"""Write reports/P6_report.md (gate G6) from results/results.json, the P6 run records, the P6 logs and git.

Every result is read from results.json; runtime comes from the run records and logs, and the integrity check from
git. The only typed numbers are the base paper's published SenT-In values (quoted from the paper, marked as such).

    python scripts/p6_report.py --compute   # once: e14.compute.* and e15.compute.* from the P6 run records
    python scripts/p6_report.py             # write the report
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import ROOT  # noqa: E402
from smartrl.p6.cfg import p6, p6_hash  # noqa: E402

OUT = ROOT / "reports" / "P6_report.md"
RUNS = ROOT / "results" / "runs"
LOGS = ROOT / "results" / "logs" / "p6"
POLICIES = ROOT / "data" / "processed" / "policies" / "e15"
PRE_P6 = "aaf78ff"                       # commit "P6 pre-registration v2": results.json before any P6 run
ORDER = ["SP500", "FTSE100", "SSE", "NIFTY50"]            # the paper's order
MKT = {"SP500": "S&P 500", "FTSE100": "FTSE 100", "SSE": "SSE Composite", "NIFTY50": "Nifty 50"}
NAME = {"SVM_P": "SVM (paper recipe)", "CNN_P": "CNN (paper recipe)", "LSTM_P": "LSTM (paper recipe)",
        "GRU_P": "GRU (paper recipe)", "SENTIN_PAPER": "SenT-In, paper recipe", "SENTIN_E13": "SenT-In, E13 recipe",
        "SENTIN_SF_PAPER": "SenT-In + SentiFlow, paper recipe", "SENTIN_SF_E13": "SenT-In + SentiFlow, E13 recipe",
        "MM_CLS": "MarketMind classifier (proposed)", "MM_DIR": "MarketMind q50 direction"}
MODELS = list(NAME)
TRADE = {"SMARTRL": "SMART-RL (proposed)", "BASE": "BASE: SenT-In, E13 recipe", "BASE_PAPER": "BASE_PAPER: SenT-In, paper recipe",
         "B1_BH": "B1 Buy & hold", "B2_SMA": "B2 SMA 50/200", "B3_TSMOM": "B3 Momentum 12-1", "B4_LSTM": "B4 LSTM signal",
         "B5_FINBERT": "B5 FinBERT signal", "B6_PPOPRICE": "B6 PPO price-only", "B7_SAC": "B7 SAC",
         "B8_STATIC": "B8 Static 50/50"}
# SenT-In as published: Chauhan et al. (2025), Tables 4-5 (printed p. 15) and 6-7 (printed p. 16).
# (Accuracy, F1-Score, AUC-ROC, PR-AUC). Quoted from the paper; not results of this project.
PUBLISHED = {"SP500": (0.8159, 0.8174, 0.9331, 0.9465), "FTSE100": (0.9579, 0.9521, 0.9970, 0.9963),
             "SSE": (0.8770, 0.8851, 0.9483, 0.9567), "NIFTY50": (0.8414, 0.8464, 0.9212, 0.9309)}
MET = ["acc", "f1", "auc", "prauc"]
METNAME = {"acc": "accuracy", "f1": "F1-score", "auc": "AUC-ROC", "prauc": "PR-AUC"}

T: dict = {}


# ---------------------------------------------------------------- helpers
def g(key: str):
    v = R.lookup(T, key)
    if v is R.MISSING:
        raise KeyError(key)
    return v


def f3(key: str) -> str:
    v = g(key)
    return "n/a" if v is None else f"{v:.3f}"


def d3(key: str) -> str:
    return f"{g(key):+.3f}"


def pct(key: str) -> str:
    v = g(key)
    return "n/a" if v is None else f"{100 * v:.1f} %"


def pv(key: str) -> str:
    v = g(key)
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def ci(pre: str) -> str:
    return f"[{g(pre + '.lo'):+.3f}, {g(pre + '.hi'):+.3f}]"


def table(head: list, rows: list) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def flat(d: dict, p: str = ""):
    for k, v in d.items():
        kk = f"{p}.{k}" if p else k
        if isinstance(v, dict) and v:
            yield from flat(v, kk)
        else:
            yield kk, v


def run_records() -> list[dict]:
    out = []
    for m in RUNS.glob("P6_*/meta.json"):
        d = json.loads(m.read_text(encoding="utf-8"))
        if d.get("status") == "ok":
            out.append(d)
    return out


def cuda(d: dict) -> bool:
    return str(d.get("device") or "").startswith("cuda")


# ---------------------------------------------------------------- compute keys
def stage_compute() -> None:
    recs = run_records()
    items = []
    for exp in ("E14", "E15"):
        rr = [d for d in recs if d["name"].startswith(exp)]
        pre = exp.lower()
        items += [(f"{pre}.compute.n_runs", len(rr)),
                  (f"{pre}.compute.gpu_hours", sum(d.get("seconds") or 0 for d in rr if cuda(d)) / 3600),
                  (f"{pre}.compute.cpu_hours", sum(d.get("seconds") or 0 for d in rr if not cuda(d)) / 3600)]
    items.append(("e14.compute.n_model_runs", sum(1 for d in recs if d["name"].startswith(("E14_PUB", "E14_CAUSAL")))))
    for arm in ("SMARTRL", "B6", "B7"):
        items.append((f"e15.compute.policies.{arm}", len(list((POLICIES / arm).glob("*/seed*/info.json")))))
    tree = R.load()                                   # re-running replaces this stage's own note instead of adding one
    tree["notes"] = [n for n in tree.get("notes", []) if not n.startswith("e14.compute / e15.compute:")]
    R.save(tree)
    R.update(items, ["e14.compute / e15.compute: P6 run records with status ok (results/runs/P6_*): n_runs, and "
                     "gpu_hours / cpu_hours = summed run durations by device as in compute.* (parallel jobs counted "
                     "separately); e14.compute.n_model_runs = run records of the E14 run matrix (one per setting, "
                     "index or pooled model, and seed); e15.compute.policies.<ARM> = trained E15 policies on disk."])
    for k, v in items:
        print(k, v)


# ---------------------------------------------------------------- facts from logs and git
def pipeline_minutes() -> dict[str, float]:
    text = (LOGS / "pipeline.log").read_text(encoding="utf-8")
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"DONE (\S+) in ([\d.]+) min", text)}


def pytest_summary(name: str) -> str:
    text = (LOGS / name).read_text(encoding="utf-8")
    m = re.findall(r"(\d+ passed(?:, \d+ skipped)?)", text)
    return m[-1] if m else "no summary line"


def integrity() -> dict:
    old = json.loads(subprocess.run(["git", "show", f"{PRE_P6}:results/results.json"], cwd=ROOT, capture_output=True,
                                    check=True).stdout.decode("utf-8"))
    fo, fn = dict(flat(old)), dict(flat(T))
    changed = sorted(k for k in fo if k in fn and fo[k] != fn[k])
    added = [k for k in fn if k not in fo]
    return {"n_old": len(fo), "missing": [k for k in fo if k not in fn], "changed": changed,
            "notes_appended": fn["notes"][:len(fo["notes"])] == fo["notes"], "n_notes": (len(fo["notes"]), len(fn["notes"])),
            "n_added": len(added), "added_outside": [k for k in added if not k.startswith(("e14.", "e15."))]}


def lock_ok() -> tuple[bool, str]:
    import hashlib
    sha = hashlib.sha256((ROOT / "configs" / "frozen.yaml").read_bytes()).hexdigest()
    return (ROOT / "locks" / "G4.lock").read_text(encoding="utf-8").strip().startswith(sha), sha


# ---------------------------------------------------------------- derived facts (all from results.json keys)
def always_up(setting: str) -> bool:
    """True when the paper-recipe SenT-In has the accuracy of the always-up rule in every index."""
    rd = "PUB_C" if setting.startswith("PUB_C") else "PUB_R"
    return all(abs(g(f"e14.{setting}.{m}.SENTIN_PAPER.acc") - g(f"e14.labels.{m}.GE.{rd}.allpos_acc")) < 1e-9
               for m in ORDER)


def best(setting: str, market: str) -> tuple[str, float]:
    models = [mo for mo in MODELS if R.lookup(T, f"e14.{setting}.{market}.{mo}.acc") is not R.MISSING]
    mo = max(models, key=lambda x: g(f"e14.{setting}.{market}.{x}.acc"))
    return mo, g(f"e14.{setting}.{market}.{mo}.acc")


def la_max() -> tuple[float, str, str, str]:
    out = []
    for s in ("PUB_C_GE_LA", "PUB_R_GE_LA"):
        for m in ORDER:
            mo, v = best(s, m)
            out.append((v, s, m, mo))
    return max(out)


# ---------------------------------------------------------------- report
def persian_summary(pub_acc: dict) -> list[str]:
    c, r = "e14.PUB_C_GE.MEAN", "e14.PUB_R_GE.MEAN"
    pos = [g(f"e14.labels.{m}.GE.{rd}.pos_share") for m in ORDER for rd in ("PUB_C", "PUB_R")]
    la = la_max()
    n_pol = sum(g(f"e15.compute.policies.{a}") for a in ("SMARTRL", "B6", "B7"))
    e = "e15.full.COMBINED"
    return [
        "# گزارش فاز P6 (دروازه‌ی G6)",
        "",
        "خلاصه:",
        f"- E14 (وظیفه‌ی خود مقاله‌ی مبنا با شرایط دقیق آن) و E15 (معامله در پنجره‌ی آزمون مقاله) کامل اجرا شدند: "
        f"{g('e14.compute.n_model_runs')} اجرای مدل در E14 و {n_pol} سیاست معاملاتی در E15. همه‌ی آزمون‌های از پیش ثبت‌شده "
        "انجام شدند. آزمون‌های نشت داده پاس شدند، `frozen.yaml` و `G4.lock` دست نخوردند و هیچ کلید قبلی "
        "`results.json` تغییر نکرد.",
        f"- **نتیجه‌ی اصلی (شرایط دقیق مقاله، معیارهای خود مقاله):** مدل پیش‌بینی ما (MarketMind با ویژگی‌های "
        "SentiFlow) در AUC-ROC و PR-AUC به‌طور معنادار از SenT-In بهتر است، در هر دو خوانش تقسیم 80:20. "
        f"AUC-ROC در خوانش زمانی {f3(c + '.MM_CLS.auc')} در برابر {f3(c + '.SENTIN_PAPER.auc')} و در خوانش تصادفی "
        f"{f3(r + '.MM_CLS.auc')} در برابر {f3(r + '.SENTIN_PAPER.auc')} است (p با تصحیح Holm کمتر از 0.001). "
        f"PR-AUC در خوانش زمانی {f3(c + '.MM_CLS.prauc')} در برابر {f3(c + '.SENTIN_PAPER.prauc')} است. این برتری "
        "در برابر نسخه‌ی قوی‌تر SenT-In (دستور E13) هم برقرار است.",
        f"- در Accuracy و F1 مدل ما کمی پایین‌تر است (دقت در خوانش زمانی {f3(c + '.MM_CLS.acc')} در برابر "
        f"{f3(c + '.SENTIN_PAPER.acc')}). دلیلش این است که SenT-In با دستور آموزش مقاله برای همه‌ی ردیف‌ها «بالا» "
        f"پیش‌بینی می‌کند و در این پروتکل {100 * min(pos):.0f} تا {100 * max(pos):.0f} درصد ردیف‌های آزمون «بالا» هستند. "
        "با برچسب سخت‌گیرانه‌ی GT، مدل ما در هر چهار معیار بهتر است.",
        "- بررسی اعتبار V1 (از پیش ثبت‌شده) نشان می‌دهد این برتری از ردیف‌هایی می‌آید که برچسبشان با پرکردن روزهای "
        "تعطیل ثابت شده است. روی ردیف‌هایی که بعدشان یک جلسه‌ی واقعی معاملاتی است، همه‌ی مدل‌ها، از جمله SenT-In، "
        "در حد شانس‌اند. طبق قاعده‌ی ادعا، این جمله باید در پایان‌نامه بیاید.",
        "- SentiFlow داخل SenT-In (با دستور مقاله) بهبود معناداری نمی‌دهد.",
        f"- دقت منتشرشده‌ی SenT-In در مقاله ({min(pub_acc.values()):.3f} تا {max(pub_acc.values()):.3f}) در شرایط "
        "اعلام‌شده با هیچ مدلی به دست نیامد. حتی وقتی مدل اطلاعات روز هدف را می‌بیند (آزمایش تشخیصی LA)، بیشترین "
        f"دقت {la[0]:.3f} است.",
        f"- **E15** ({g('e15.window.first_test_day')} تا {g('e15.window.last_test_day')}): Sharpe سبد ترکیبی SMART-RL "
        f"{f3(e + '.SMARTRL.sharpe')} است، در برابر {f3(e + '.BASE.sharpe')} برای SenT-In با دستور E13 و "
        f"{f3(e + '.BASE_PAPER.sharpe')} برای SenT-In با دستور مقاله. نسخه‌ی دوم همیشه در بازار است و همان خرید و "
        "نگهداری است. هیچ‌کدام از این دو اختلاف معنادار نیست. بیشینه‌ی افت سرمایه‌ی SMART-RL "
        f"({pct(e + '.SMARTRL.mdd')}) به‌طور معنادار از خرید و نگهداری ({pct(e + '.B1_BH.mdd')}) کمتر است.",
        "- **تصمیم لازم:** برای تأیید G6 بنویسید «ادامه». برای E16 (اختیاری)، کد TDQN را دانلود کنید و بنویسید "
        "«E16 تأیید»، یا بنویسید که E16 اجرا نشود.",
        "",
    ]


def section_headline() -> list[str]:
    L = ["## Details (English)", "",
         "### 1. Headline: the proposed predictive model against SenT-In under the base paper's own conditions", ""]
    L.append("Setting (protocol §2.1–2.4): the paper's window and row count "
             f"({g('e14.data.SP500.first_date')} → {g('e14.data.SP500.last_date')}, "
             f"{g('e14.data.SP500.labelled_rows')} labelled calendar rows per index, non-trading days filled with the "
             "previous day's values), the paper's 80:20 split read chronologically (`PUB_C`: test "
             f"{g('e14.PUB_C_GE.SP500.test_start')} → {g('e14.PUB_C_GE.SP500.test_end')}) and randomly (`PUB_R`, seed "
             f"{p6()['e14']['random_seed']}), {g('e14.PUB_C_GE.SP500.n_train')} training and "
             f"{g('e14.PUB_C_GE.SP500.n_test')} test rows per index, label GE = 1[C(t+1) ≥ C(t)], and the paper's four "
             "metrics. The base model is SenT-In rebuilt from the paper (awareness layer, GRU of eqs 7–11, Informer) "
             "and trained with its Table 3 recipe (`SENTIN_PAPER`). Ours is the MarketMind classifier with the "
             "SentiFlow features (`MM_CLS`). Mean over the four indices (seed mean; `e14.<SET>.MEAN.<MODEL>.*`):")
    L.append("")
    rows = []
    for mo in MODELS:
        rows.append([NAME[mo]] + [f3(f"e14.{s}.MEAN.{mo}.{k}") for s in ("PUB_C_GE", "PUB_R_GE") for k in MET])
    L.append(table(["model", "PUB_C Acc", "PUB_C F1", "PUB_C AUC-ROC", "PUB_C PR-AUC",
                    "PUB_R Acc", "PUB_R F1", "PUB_R AUC-ROC", "PUB_R PR-AUC"], rows))
    L.append("")
    L.append("Primary pre-registered tests (protocol §2.7: one-sided H1 ours > SenT-In, statistic = mean over the "
             "four indices of the AUC-ROC difference on seed-averaged scores, paired stationary block bootstrap "
             "stratified by index, 10 000 resamples, Holm over the four tests; `e14.tests.<SET>.<CMP>.*`):")
    L.append("")
    rows = []
    for s in ("PUB_C_GE", "PUB_R_GE"):
        for a in ("MM_CLS", "SENTIN_SF_PAPER"):
            k = f"e14.tests.{s}.{a}_vs_SENTIN_PAPER"
            dl = [g(f"{k}.delong.{m}.p_holm") for m in ORDER]
            rows.append([s, f"{a} vs SENTIN_PAPER", d3(k + ".auc_diff"), ci(k), pv(k + ".p"), pv(k + ".p_holm"),
                         "< 0.001 in all four" if max(dl) < 0.001 else f"{min(dl):.3f} – {max(dl):.3f}"])
    L.append(table(["setting", "comparison", "AUC-ROC diff", "95 % interval", "p", "Holm p",
                    "DeLong p per index (Holm)"], rows))
    L.append("")
    L.append("Secondary metrics of the same comparisons (difference ours − SenT-In; PR-AUC and F1: bootstrap, "
             "one-sided; accuracy: exact McNemar on the 0.5-threshold predictions pooled over the indices, "
             "two-sided; Holm within each metric):")
    L.append("")
    rows = []
    for s in ("PUB_C_GE", "PUB_R_GE"):
        for a in ("MM_CLS", "SENTIN_SF_PAPER"):
            k = f"e14.tests.{s}.{a}_vs_SENTIN_PAPER"
            rows.append([s, a, f"{d3(k + '.prauc_diff')} (p {pv(k + '.prauc_p')}, Holm {pv(k + '.prauc_p_holm')})",
                         f"{d3(k + '.acc_diff')} (McNemar p {pv(k + '.mcnemar_p')}, Holm {pv(k + '.mcnemar_p_holm')})",
                         f"{d3(k + '.f1_diff')} (p {pv(k + '.f1_p')}, Holm {pv(k + '.f1_p_holm')})"])
    L.append(table(["setting", "ours", "PR-AUC diff", "Accuracy diff", "F1 diff"], rows))
    L.append("")
    up = always_up("PUB_C_GE") and always_up("PUB_R_GE")
    L.append("Reading: `MM_CLS` ranks the days far better than SenT-In on both threshold-free metrics, in both "
             "readings and in every index. On accuracy and F1 it is slightly lower: "
             + ("the paper-recipe SenT-In has exactly the accuracy of the always-up rule in every index (it predicts "
                "\"up\" on every test row), " if up else "the paper-recipe SenT-In is close to the always-up rule, ")
             + "and under GE most test rows are up (positive share "
             f"{min(g(f'e14.labels.{m}.GE.PUB_C.pos_share') for m in ORDER):.3f}–"
             f"{max(g(f'e14.labels.{m}.GE.PUB_C.pos_share') for m in ORDER):.3f} in `PUB_C`), so a constant \"up\" is "
             "hard to beat on threshold metrics. Under the strict label GT (V3 below) `MM_CLS` is better on all four "
             "metrics.")
    L.append("")
    L.append("**Claim sentences produced by the pre-registered claim rules (verbatim, `e14.claims.*`):**")
    L.append("")
    for k in ("MM_CLS", "SENTIN_SF_PAPER"):
        L.append(f"> {g(f'e14.claims.{k}')}")
        L.append("")
    L.append("\"p 0.000\" in the generated sentence is the bootstrap floor 1/10 001 printed with three decimals "
             "(reported as p < 0.001 elsewhere).")
    L.append("")
    return L


def section_published(pub_acc: dict) -> list[str]:
    L = ["### 2. The paper's own layout (T22) with the published SenT-In numbers", ""]
    L.append("Per index in the paper's order (`results/tables/T22.csv` has every model). The row \"SenT-In, published\" "
             "is quoted from Chauhan et al. (2025), Tables 4–7 (printed pp. 15–16); it is not a result of this project.")
    L.append("")
    for s, label in (("PUB_C_GE", "chronological reading `PUB_C_GE`"), ("PUB_R_GE", "random reading `PUB_R_GE`")):
        L.append(f"**{label}**")
        L.append("")
        rows = []
        for m in ORDER:
            rows.append([MKT[m], "SenT-In, published"] + [f"{v:.3f}" for v in PUBLISHED[m]])
            for mo in ("SENTIN_PAPER", "SENTIN_E13", "MM_CLS"):
                rows.append(["", NAME[mo]] + [f3(f"e14.{s}.{m}.{mo}.{k}") for k in MET])
        L.append(table(["index", "model", "Accuracy", "F1-Score", "AUC-ROC", "PR-AUC"], rows))
        L.append("")
    la = la_max()
    best_c = {m: best("PUB_C_GE", m) for m in ORDER}
    reach = reached(("PUB_C_GE", "PUB_R_GE"))
    if not reach:
        first = "No model reaches any published SenT-In value under the stated conditions, on any metric or index. "
    else:
        cells = sorted({(s, m, k) for s, m, _, k in reach})
        who = sorted({mo for *_, mo, _ in reach}, key=MODELS.index)
        what = ", ".join(f"the {MKT[m]} {METNAME[k]} in `{s}`" for s, m, k in cells)
        first = (("The only published SenT-In value that any model reaches under the stated conditions is "
                  if len(cells) == 1 else "The only published SenT-In values that any model reaches under the stated "
                                          "conditions are ")
                 + what + f" (reached by {', '.join(f'`{x}`' for x in who)}). ")
        if all(k == "f1" for *_, k in cells):
            first += ("These models predict \"up\" on (almost) every row, and the always-up rule itself scores "
                      + ", ".join(f"{g(f'e14.labels.{m}.GE.{s[:5]}.allpos_f1'):.3f} ({MKT[m]}, `{s}`)"
                                  for s, m, _ in cells)
                      + ", so a published F1 below that level says nothing about the model. "
                      "Every published accuracy, AUC-ROC and PR-AUC is above every model in every index. ")
    L.append(first + "The best test accuracy in `PUB_C_GE` is "
             + ", ".join(f"{best_c[m][1]:.3f} ({MKT[m]})" for m in ORDER)
             + f", against the published {', '.join(f'{pub_acc[m]:.3f}' for m in ORDER)}. Even the look-ahead "
             "diagnostic `LA` (the target day's open, high, low, volume and sentiment are visible, its close is not; "
             f"§4) reaches at most {la[0]:.3f} (`{la[3]}`, {MKT[la[2]]}, `{la[1]}`). The published numbers can "
             "therefore not be reproduced from the paper's description; the defensible comparison is the head-to-head "
             "under identical conditions of §1, which is what the pre-registered tests measure.")
    L.append("")
    return L


def reached(settings) -> list[tuple[str, str, str, str]]:
    """(setting, index, model, metric) where a model's value is at or above the published SenT-In value."""
    out = []
    for s in settings:
        for m in ORDER:
            for mo in MODELS:
                for i, k in enumerate(MET):
                    v = R.lookup(T, f"e14.{s}.{m}.{mo}.{k}")
                    if v is not R.MISSING and v >= PUBLISHED[m][i]:
                        out.append((s, m, mo, k))
    return out


def section_validity() -> list[str]:
    L = ["### 3. Validity checks (pre-registered; they set the wording of the claim)", ""]
    rows = []
    for s in ("PUB_C_GE", "PUB_R_GE"):
        for a in ("MM_CLS", "SENTIN_SF_PAPER"):
            k = f"e14.tests_inf.{s}.{a}_vs_SENTIN_PAPER"
            rows.append(["V1 informative rows", s, f"{a} vs SENTIN_PAPER", d3(k + ".auc_diff"), ci(k), pv(k + ".p")])
    for s in ("PUB_C_GE", "PUB_R_GE"):
        for a in ("MM_CLS", "SENTIN_SF_E13"):
            k = f"e14.tests.{s}.{a}_vs_SENTIN_E13"
            rows.append(["V2 stronger base", s, f"{a} vs SENTIN_E13", d3(k + ".auc_diff"), ci(k), pv(k + ".p")])
    for s in ("PUB_C_GT", "PUB_R_GT"):
        for a in ("MM_CLS", "SENTIN_SF_PAPER"):
            k = f"e14.tests.{s}.{a}_vs_SENTIN_PAPER"
            rows.append(["V3 label GT", s, f"{a} vs SENTIN_PAPER", d3(k + ".auc_diff"), ci(k), pv(k + ".p")])
    for a, b in p6()["e14"]["v4"]:
        k = f"e14.tests.CAUSAL.{a}_vs_{b}"
        rows.append(["V4 CAUSAL", "CAUSAL", f"{a} vs {b}", d3(k + ".auc_diff"), ci(k), pv(k + ".p")])
    L.append(table(["check", "setting", "comparison", "AUC-ROC diff", "95 % interval", "one-sided p"], rows))
    L.append("")
    inf_c = [g(f"e14.PUB_C_GE.{m}.informative_share") for m in ORDER]
    L.append(f"- **V1.** Informative rows are the test rows whose next calendar date is a session "
             f"({min(inf_c):.3f}–{max(inf_c):.3f} of the `PUB_C_GE` test rows). On them every model is near chance "
             f"(mean AUC-ROC `MM_CLS` {f3('e14.PUB_C_GE.MEAN.MM_CLS.inf.auc')}, `SENTIN_PAPER` "
             f"{f3('e14.PUB_C_GE.MEAN.SENTIN_PAPER.inf.auc')}). The other rows precede a filled date, so C(t+1) = C(t) "
             "and their GE label is always 1. `MM_CLS` recognises these rows from the calendar pattern of the inputs "
             "and gives them its highest scores; SenT-In (paper recipe) gives every row the same score. The AUC "
             "advantage therefore comes from the rows whose label is fixed by the filling, as the claim rule states.")
    k = "e14.tests.{}.MM_CLS_vs_SENTIN_PAPER"
    L.append(f"- **V2.** Against our stronger reproduction of SenT-In (E13 recipe, `SENTIN_E13`) `MM_CLS` is also "
             f"better ({d3('e14.tests.PUB_C_GE.MM_CLS_vs_SENTIN_E13.auc_diff')} and "
             f"{d3('e14.tests.PUB_R_GE.MM_CLS_vs_SENTIN_E13.auc_diff')}, p < 0.001). SentiFlow in the E13 recipe "
             f"(`SENTIN_SF_E13` vs `SENTIN_E13`) changes AUC-ROC by "
             f"{d3('e14.tests.PUB_C_GE.SENTIN_SF_E13_vs_SENTIN_E13.auc_diff')} in `PUB_C_GE` "
             f"(p {pv('e14.tests.PUB_C_GE.SENTIN_SF_E13_vs_SENTIN_E13.p')}) and by "
             f"{d3('e14.tests.PUB_R_GE.SENTIN_SF_E13_vs_SENTIN_E13.auc_diff')} in `PUB_R_GE` "
             f"(p {pv('e14.tests.PUB_R_GE.SENTIN_SF_E13_vs_SENTIN_E13.p')}); no claim rests on this comparison.")
    down = all(g(f"e14.{s}.{m}.SENTIN_PAPER.f1") == 0 and
               abs(g(f"e14.{s}.{m}.SENTIN_PAPER.acc") - (1 - g(f"e14.labels.{m}.GT.{s[:5]}.pos_share"))) < 1e-9
               for s in ("PUB_C_GT", "PUB_R_GT") for m in ORDER)
    L.append(f"- **V3.** With E13's strict label GT (positive share "
             f"{min(g(f'e14.labels.{m}.GT.PUB_C.pos_share') for m in ORDER):.3f}–"
             f"{max(g(f'e14.labels.{m}.GT.PUB_C.pos_share') for m in ORDER):.3f}) `MM_CLS` is better on all four "
             f"metrics: accuracy {d3(k.format('PUB_C_GT') + '.acc_diff')} / {d3(k.format('PUB_R_GT') + '.acc_diff')} "
             f"(McNemar p {pv(k.format('PUB_C_GT') + '.mcnemar_p')} / {pv(k.format('PUB_R_GT') + '.mcnemar_p')}), "
             f"F1 {f3('e14.PUB_C_GT.MEAN.MM_CLS.f1')} vs {f3('e14.PUB_C_GT.MEAN.SENTIN_PAPER.f1')}"
             + (" (the paper-recipe SenT-In predicts \"down\" on every row)" if down else "")
             + f", PR-AUC {d3(k.format('PUB_C_GT') + '.prauc_diff')} / "
             f"{d3(k.format('PUB_R_GT') + '.prauc_diff')}. The forced rows are now always 0, and `MM_CLS` again "
             "separates them.")
    L.append(f"- **V4.** Under the causal protocol of this thesis (trading days only, session-aligned news, scaling "
             f"fitted on the training rows) every model is near chance (mean AUC-ROC "
             f"{min(g(f'e14.CAUSAL.MEAN.{mo}.auc') for mo in MODELS):.3f}–"
             f"{max(g(f'e14.CAUSAL.MEAN.{mo}.auc') for mo in MODELS):.3f}) and no comparison favours ours. This repeats "
             "the E13 and E13x findings: with prices and headlines known at the close, no model calls the next-day "
             "direction of these indices better than chance.")
    L.append("")
    return L


def section_diagnostics() -> list[str]:
    L = ["### 4. Diagnostics (descriptive; never used for a claim)", ""]
    L.append("**Label evidence** (no training; `e14.labels.*`): the always-up rule on the test rows.")
    L.append("")
    rows = []
    for m in ORDER:
        b = f"e14.labels.{m}"
        rows.append([MKT[m], f3(f"{b}.GE.PUB_C.pos_share"), f3(f"{b}.GE.PUB_C.allpos_f1"), f3(f"{b}.GE.PUB_C.allpos_prauc"),
                     f3(f"{b}.GE.PUB_R.pos_share"), f3(f"{b}.GT.PUB_C.pos_share"), f3(f"{b}.GT.PUB_R.pos_share")])
    L.append(table(["index", "GE PUB_C pos share (= always-up Acc)", "always-up F1", "always-up PR-AUC",
                    "GE PUB_R pos share", "GT PUB_C pos share", "GT PUB_R pos share"], rows))
    L.append("")
    L.append("The paper's weakest models have F1 and PR-AUC above their accuracy (protocol §1, item 3), which needs an "
             "up-majority; GE gives one, GT does not. This is why GE is the primary label.")
    L.append("")
    L.append("**Look-ahead `LA`** (accuracy; `e14.PUB_*_GE_LA.*`, figure F19):")
    L.append("")
    rows = []
    for m in ORDER:
        r = [MKT[m], f"{PUBLISHED[m][0]:.3f}"]
        for s in ("PUB_C_GE", "PUB_C_GE_LA", "PUB_R_GE", "PUB_R_GE_LA"):
            mo, v = best(s, m)
            r.append(f"{v:.3f} ({mo})")
        r.append(f"{f3(f'e14.PUB_C_GE.{m}.MM_CLS.acc')} → {f3(f'e14.PUB_C_GE_LA.{m}.MM_CLS.acc')}")
        rows.append(r)
    L.append(table(["index", "SenT-In published", "best PUB_C_GE", "best PUB_C_GE_LA", "best PUB_R_GE",
                    "best PUB_R_GE_LA", "MM_CLS PUB_C: no LA → LA"], rows))
    L.append("")
    L.append(f"**`ROBUST`** (median/IQR scaling instead of q01/q99, `PUB_C_GE`): `MM_CLS` AUC-ROC "
             f"{f3('e14.PUB_C_GE_ROBUST.MEAN.MM_CLS.auc')} (q01/q99: {f3('e14.PUB_C_GE.MEAN.MM_CLS.auc')}), "
             f"`SENTIN_PAPER` {f3('e14.PUB_C_GE_ROBUST.MEAN.SENTIN_PAPER.auc')} "
             f"({f3('e14.PUB_C_GE.MEAN.SENTIN_PAPER.auc')}). The scaling choice does not change the picture.")
    L.append("")
    L.append(f"**`MM_DIR`** (frozen MarketMind q50 at h = 1): mean AUC-ROC {f3('e14.PUB_C_GE.MEAN.MM_DIR.auc')} / "
             f"{f3('e14.PUB_R_GE.MEAN.MM_DIR.auc')} under GE and {f3('e14.PUB_C_GT.MEAN.MM_DIR.auc')} / "
             f"{f3('e14.PUB_R_GT.MEAN.MM_DIR.auc')} under GT. Its median forecast is lower on average on the rows "
             "before a filled date (their next-row return is exactly 0) than on the other rows; those rows are "
             "positive under GE and negative under GT, which moves its AUC below and above 0.5.")
    L.append("")
    return L


def section_data() -> list[str]:
    L = ["### 5. E14 data", ""]
    rows = []
    for m in ORDER:
        b = f"e14.data.{m}"
        rows.append([MKT[m], g(f"{b}.calendar_rows"), g(f"{b}.labelled_rows"), g(f"{b}.trading_rows"),
                     g(f"e14.causal.zero_change.{m}"), f"{g(f'e14.news.{m}.2005.local'):,}", f"{g(f'e14.news.{m}.2006.local'):,}",
                     f"{sum(v['local'] for v in g(f'e14.news.{m}').values()):,}"])
    L.append(table(["index", "calendar rows", "labelled rows", "trading rows", "zero-change sessions (GE ≠ GT)",
                    "local headlines 2005", "local headlines 2006", "local headlines 2005–2022"], rows))
    L.append("")
    news_new = g("e14.news.SP500.2005.global") + g("e14.news.SP500.2006.global")
    L.append(f"- Nifty 50 before 2007-09-17 comes from NSE Indices (niftyindices.com) and from Yahoo afterwards; the "
             f"largest relative close difference on the first 20 common days is "
             f"{g('e14.data.NIFTY50.nifty_splice_maxdiff'):.1e} (limit 0.1 %). Nifty volume is masked.")
    L.append(f"- The 2005–2006 news ({news_new:,} headlines kept by the P1 rules) was extracted with the P1 code and "
             "scored with the frozen models. Local coverage in 2005–2006 is thin outside the S&P 500, so many early "
             "training rows of the other indices have sentiment index 0 (SPEC §9.1). The paper's own news "
             "(investing.com, crawled every 3 minutes) is not public and cannot be matched (protocol §1, item 11).")
    rows = [[d, f3(f"e14.a_paper.{d}.acc"), f3(f"e14.a_paper.{d}.macro_f1")]
            for d in ("phrasebank50", "phrasebank_all", "fiqa", "twitter", "pooled")]
    L.append("- `A_PAPER`, the paper's GRU→CNN sentiment classifier with its Table 2 recipe (400 epochs, no early "
             "stopping), mean over seeds 0–2 on the SPEC §2.4 test splits (`e14.a_paper.*`):")
    L.append("")
    L.append(table(["test split", "accuracy", "macro-F1"], rows))
    L.append("")
    L.append(f"  On the same pooled split, E13's Branch A reaches {f3('sentiflow.pooled.A.acc')} / "
             f"{f3('sentiflow.pooled.A.f1')} and SentiFlow {f3('sentiflow.pooled.SF.acc')} / "
             f"{f3('sentiflow.pooled.SF.f1')} (E1, accuracy / macro-F1): the paper recipe (400 epochs, no early "
             "stopping) scores below both.")
    L.append("")
    return L


def section_e15() -> list[str]:
    L = ["### 6. E15: trading in the base paper's test window", ""]
    L.append(f"One split per market (protocol §3.1): train 2010-01-01 → 2017-10-17, validation 2017-10-18 → "
             f"2018-10-17, one test run {g('e15.window.first_test_day')} → {g('e15.window.last_test_day')} "
             f"({g('e15.window.n_days')} distinct trading dates over the four markets), no refits inside it. SMART-RL runs with "
             "`configs/frozen.yaml` unchanged; that configuration was selected on development validation data ending "
             "2018-06 (SPEC §1), before this window starts. Costs 10 bps. COMBINED portfolio "
             "(`e15.<full|sub2020|cost20>.COMBINED.*`):")
    L.append("")
    rows = []
    for meth in TRADE:
        f, s = f"e15.full.COMBINED.{meth}", f"e15.sub2020.COMBINED.{meth}"
        rows.append([TRADE[meth], f3(f + ".sharpe"), pct(f + ".ann_return"), pct(f + ".mdd"), pct(f + ".cum_return"),
                     f3(f + ".turnover"), f3(f"e15.cost20.COMBINED.{meth}.sharpe"), f3(s + ".sharpe"), pct(s + ".mdd")])
    L.append(table(["method", "Sharpe", "ann. return", "max drawdown", "cum. return", "turnover", "Sharpe at 20 bps",
                    "Sharpe 2020-01 → 2022-03", "max DD 2020-01 → 2022-03"], rows))
    L.append("")
    rows = []
    for b in ("BASE", "BASE_PAPER", "B1_BH"):
        k = f"e15.tests.{b}"
        rows.append([f"SMART-RL − {b}", d3(k + ".diff"), pv(k + ".p_hac"), pv(k + ".p_boot"),
                     f"{d3(k + '.mdd_diff.est')} [{g(k + '.mdd_diff.lo'):+.3f}, {g(k + '.mdd_diff.hi'):+.3f}]",
                     f"{d3(k + '.cr_diff.est')} [{g(k + '.cr_diff.lo'):+.3f}, {g(k + '.cr_diff.hi'):+.3f}]"])
    L.append(table(["comparison (COMBINED)", "Sharpe diff", "p (LW HAC, one-sided)", "p (bootstrap)",
                    "max-drawdown diff [95 %]", "cum.-return diff [95 %]"], rows))
    L.append("")
    same = all(g(f"e15.full.COMBINED.BASE_PAPER.{k}") == g(f"e15.full.COMBINED.B1_BH.{k}") for k in ("sharpe", "mdd", "cum_return"))
    L.append("- " + ("`BASE_PAPER` (SenT-In, paper recipe) is invested on every day, so its returns equal buy-and-hold; "
                     "in figure F18 its curve lies under the buy-and-hold curve. " if same else "")
             + "A positive drawdown difference means a smaller drawdown for SMART-RL; against buy-and-hold its 95 % "
             "interval excludes 0.")
    dd = [(m, g(f"e15.full.{m}.SMARTRL.mdd"), g(f"e15.full.{m}.B1_BH.mdd")) for m in ORDER]
    smaller = [m for m, a, b in dd if a > b]
    L.append(f"- Per market (T23), SMART-RL's drawdown against buy-and-hold: "
             + ", ".join(f"{MKT[m]} {100 * a:.1f} % vs {100 * b:.1f} %" for m, a, b in dd)
             + (" (smaller in every index)." if len(smaller) == len(ORDER) else
                f" (smaller in {len(smaller)} of {len(ORDER)} indices)."))
    L.append(f"- `BASE` holds no position on the S&P 500 from 2020-01-01 on, so its Sharpe ratio there is undefined "
             f"(`e15.sub2020.SP500.BASE.sharpe` = null; max drawdown {pct('e15.sub2020.SP500.BASE.mdd')}).")
    top = max(TRADE, key=lambda x: g(f"e15.full.COMBINED.{x}.sharpe"))
    L.append(f"- The highest COMBINED Sharpe ratio in this window is {f3(f'e15.full.COMBINED.{top}.sharpe')} "
             f"({TRADE[top]}).")
    L.append("")
    L.append("Decisions as next-day direction calls (prediction 1[e(t) > 0.5], score e(t), label 1[C(t+1) > C(t)] on "
             "trading days; mean over the four markets, `e15.dir.MEAN.*`):")
    L.append("")
    rows = [[TRADE[x]] + [f3(f"e15.dir.MEAN.{x}.{k}") for k in ("acc", "f1", "auc", "prauc", "up_share")]
            for x in ("SMARTRL", "BASE", "BASE_PAPER")]
    L.append(table(["method", "Accuracy", "F1", "AUC-ROC", "PR-AUC", "up share"], rows))
    L.append("")
    o = "e15.p5overlap.COMBINED"
    L.append(f"- Overlap with P5 without retraining (`e15.p5overlap`, 2020-01-01 → 2022-03-31 from the P5 annual-refit "
             f"daily returns): Sharpe SMART-RL {f3(o + '.SMARTRL.sharpe')}, BASE {f3(o + '.BASE.sharpe')}, buy-and-hold "
             f"{f3(o + '.B1_BH.sharpe')}; max drawdown {pct(o + '.SMARTRL.mdd')}, {pct(o + '.BASE.mdd')}, "
             f"{pct(o + '.B1_BH.mdd')}. The single-split E15 run and the annually refitted P5 run differ for BASE "
             "(one SenT-In fit before 2018-10-18 against yearly refits).")
    L.append("")
    return L


def section_ran(mins: dict) -> list[str]:
    recs = run_records()
    data_names = ("E14_prices", "E14_news_2005_2006", "E14_news_score_2005_2006", "E14_A_PAPER", "E14_A_PAPER_scoring",
                  "E14_daily_indices")
    data_h = sum(d["seconds"] for d in recs if d["name"] in data_names) / 3600
    smoke = sum(1 for d in recs if d["name"] == "E14_A_PAPER" and (d.get("params") or {}).get("epochs") != 400)
    n_model = sum(1 for d in recs if d["name"].startswith(("E14_PUB", "E14_CAUSAL")))
    ppo = re.search(r"(\d+) trained in ([\d.]+) h", (LOGS / "e15_train_ppo.log").read_text(encoding="utf-8"))
    pol = {a: g(f"e15.compute.policies.{a}") for a in ("SMARTRL", "B6", "B7")}
    L = ["### 7. What ran", ""]
    rows = [
        ["E14 data: Nifty splice, 2005–2006 news extraction and scoring, `A_PAPER` (3 × 400 epochs), scoring of "
         "2005–2022 headlines, daily indices", f"{data_h:.2f} h (run records)"],
        ["E15 forecasts: one MarketMind and one LSTM_PRICE fit before 2018-10-18; BASE and BASE_PAPER fits",
         f"{mins.get('e15_fc', float('nan')):.1f} min"],
        [f"E14 run matrix: {n_model} model runs (GPU, 3 workers)", f"{mins.get('e14_run', float('nan')):.1f} min"],
        ["E14 evaluation and tests (tables and figures re-run by hand after the F17 fix, §9)",
         f"{mins.get('e14_eval', 0) + mins.get('e14_tests', 0):.1f} min"],
        [f"E15 PPO-family policies: SMART-RL {pol['SMARTRL']}, B6 {pol['B6']} (CPU, 6 workers)",
         f"{ppo.group(2)} h" if ppo else "see log"],
        [f"E15 SAC policies: B7 {pol['B7']} (GPU, 6 workers)", f"{mins.get('e15_sac', float('nan')):.1f} min"],
        ["E15 evaluation, tables and figure", f"{mins.get('e15_eval', 0) + mins.get('e15_outputs', 0):.1f} min"],
    ]
    L.append(table(["step", "wall clock"], rows))
    L.append("")
    L.append(f"Run records (`results/runs/P6_*/meta.json`): E14 {g('e14.compute.n_runs')} ({g('e14.compute.gpu_hours'):.1f} GPU h, "
             f"{g('e14.compute.cpu_hours'):.1f} CPU h summed over parallel jobs), E15 {g('e15.compute.n_runs')} "
             f"({g('e15.compute.gpu_hours'):.1f} GPU h, {g('e15.compute.cpu_hours'):.1f} CPU h). {smoke} of the E14 "
             "records are 2-epoch smoke runs of `A_PAPER`, not used for any result. Unattended pipeline: "
             "`scripts/run_p6_pipeline.py` (log `results/logs/p6/pipeline.log`).")
    L.append("")
    L.append("Code: `configs/p6.yaml`; `smartrl/p6/` (`cfg.py`, `prices.py`, `news.py`, `a_paper.py`, `models.py`, "
             "`e14data.py`, `e14run.py`, `e14eval.py`, `stats.py`, `e15.py`, `e15eval.py`); `scripts/p6_fetch_nifty.py`, "
             "`scripts/p6_e14.py`, `scripts/p6_e15.py`, `scripts/run_p6_pipeline.py`, `scripts/p6_report.py`; "
             "`smartrl/data/news.py` gained optional arguments (defaults unchanged). Outputs: `results/tables/T21.csv` "
             "… `T25.csv`, `results/figures/F17` … `F19` (.png, .pdf).")
    L.append("")
    return L


def section_interpretations() -> list[str]:
    L = ["### 8. Interpretations fixed before the runs (`configs/p6.yaml`, `interpretations`)", ""]
    for k, v in p6()["interpretations"].items():
        L.append(f"- `{k}`: {' '.join(str(v).split())}")
    L.append("")
    return L


def section_deviations() -> list[str]:
    L = ["### 9. Deviations, the bug fix and problems", ""]
    L.append("- **Method:** no deviation from the protocol. The choices the protocol leaves open are listed in §8; the "
             "unmatched items of protocol §1 (the paper's news source, the training data of its sentiment classifier, "
             "look-back, Informer sizes, SVM kernel, threshold) are handled as the protocol prescribes.")
    ex = [(s, m) for s in ("PUB_C_GE",) for m in ORDER]
    pairs = "; ".join(f"{MKT[m]} {f3(f'e14.{s}.{m}.MM_CLS.auc')} → {f3(f'e14.ci.{s}.{m}.MM_CLS.auc')}" for s, m in ex)
    L.append("- **Bug fix (figure F17 only, protocol §0 rule 4).** The first `e14_outputs` step stopped: F17 drew the "
             "mean of the per-seed AUCs (`e14.<SET>.<MKT>.<MODEL>.auc`) with a bootstrap interval of the AUC of the "
             "seed-averaged scores (`e14.ci.*.auc_lo/auc_hi`), two different estimators, so some points fell outside "
             "their intervals and matplotlib refused the negative error bars. F17 now draws the AUC of the "
             "seed-averaged scores (new keys `e14.ci.<SET>.<MKT>.<MODEL>.auc`), the estimator the interval and the "
             "tests use. No model was re-run and no stored number changed; old and new points are both kept in "
             f"results.json (example, `MM_CLS` in `PUB_C_GE`, old → new: {pairs}). The pipeline was changed so that a "
             "failing table/figure step no longer stops the training steps.")
    L.append("- **Undefined values:** `e15.sub2020.SP500.BASE.<sharpe|sortino|calmar|win_rate>` are null because BASE "
             "has no position on the S&P 500 in that sub-window (zero volatility).")
    L.append("- **Limitations:** (1) the paper's news database is not public, so all models use the thesis corpus "
             "extended to 2005; (2) the published SenT-In numbers could not be reproduced by any model (§2), so "
             "the comparison with the paper rests on the faithful re-implementation; (3) the AUC advantage under the "
             "paper's protocol comes from the calendar filling (V1); under the causal protocol no model predicts the "
             "next-day direction (V4).")
    L.append("")
    return L


def section_integrity() -> list[str]:
    it = integrity()
    ok, sha = lock_ok()
    recs = run_records()
    hashes = {(d.get("params") or {}).get("p6_hash") for d in recs}
    L = ["### 10. Integrity checks", ""]
    L.append(f"- `configs/frozen.yaml` SHA-256 {sha[:12]}… {'equals' if ok else 'DIFFERS FROM'} `locks/G4.lock`.")
    L.append(f"- `configs/p6.yaml` hash {p6_hash()[:12]}…: "
             + (f"the same hash is stored in all {len(recs)} P6 run records (the settings never changed during P6)."
                if hashes == {p6_hash()} else f"run records carry {len(hashes)} different hashes: CHECK."))
    changed = [k for k in it["changed"] if k not in ("meta.created_utc", "notes")]
    L.append(f"- `results/results.json` against the pre-registration commit {PRE_P6}: {it['n_old']} existing keys, "
             f"{len(it['missing'])} missing, {len(changed)} changed (besides `meta.created_utc`; `notes` only gained "
             f"entries: {it['n_notes'][0]} → {it['n_notes'][1]}, old entries "
             f"{'unchanged' if it['notes_appended'] else 'CHANGED'}); {it['n_added']} new keys, "
             f"{len(it['added_outside'])} outside `e14.*` / `e15.*`.")
    L.append(f"- Leakage tests at G6 (`tests/test_leakage.py`): {pytest_summary('g6_leakage.log')} (the skipped test "
             "only applies before the G4 lock). Contract test (`tests/test_results_contract.py`): "
             f"{pytest_summary('g6_contract.log')}. Logs: `results/logs/p6/g6_leakage.log`, `g6_contract.log`.")
    L.append("")
    return L


def section_wording() -> list[str]:
    c, r = "e14.tests.PUB_C_GE.MM_CLS_vs_SENTIN_PAPER", "e14.tests.PUB_R_GE.MM_CLS_vs_SENTIN_PAPER"
    la = la_max()
    L = ["### 11. Wording for the thesis (follows the claim rules; numbers from the keys above)", ""]
    L.append(f"> Under the base paper's own conditions (2005-01-01 → 2022-03-31, {g('e14.data.SP500.labelled_rows')} "
             "calendar rows per index with filled non-trading days, 80:20 split) and judged by the paper's own "
             "metrics, the proposed predictive model (MarketMind with SentiFlow features) is significantly better than "
             f"SenT-In on AUC-ROC ({d3(c + '.auc_diff')} in the chronological and {d3(r + '.auc_diff')} in the random "
             f"reading; Holm-adjusted p < 0.001) and PR-AUC ({d3(c + '.prauc_diff')}, {d3(r + '.prauc_diff')}), in "
             "every index, and also against a stronger reproduction of SenT-In. On accuracy and F1 the paper-recipe "
             "SenT-In is marginally higher because it predicts \"up\" on every row; with the strict label "
             "1[C(t+1) > C(t)] the proposed model is better on all four metrics. A pre-registered check shows that "
             "the advantage comes from rows whose label is fixed by the filling of non-trading days: on rows followed "
             "by a trading session all models, SenT-In included, are at chance level. The published SenT-In "
             "accuracies could not be reached by any model under the stated conditions, even with the target day's "
             f"open, high, low and volume visible (best {la[0]:.3f}).")
    L.append("")
    e = "e15.full.COMBINED"
    L.append(f"> In the paper's test window ({g('e15.window.first_test_day')} → {g('e15.window.last_test_day')}), "
             f"SMART-RL reached a Sharpe ratio of {f3(e + '.SMARTRL.sharpe')} against {f3(e + '.BASE.sharpe')} for "
             f"SenT-In (E13 recipe) as a trading signal (difference not significant, p {pv('e15.tests.BASE.p_hac')}) and a maximum "
             f"drawdown of {pct(e + '.SMARTRL.mdd')} against {pct(e + '.B1_BH.mdd')} for buy-and-hold, which the "
             "paper-recipe SenT-In equals because it is always invested (the 95 % interval of the drawdown difference "
             f"excludes 0); its Sharpe ratio is below buy-and-hold's ({f3(e + '.B1_BH.sharpe')}; not significant).")
    L.append("")
    return L


def section_questions() -> list[str]:
    return ["### 12. Decisions for Roham", "",
            "1. Gate G6: reply \"ادامه\" to accept E14 and E15 as reported.",
            "2. E16 (optional, protocol §4): TDQN of Théate and Ernst (2021) under the SMART-RL trading conditions. "
            "It needs the authors' official code: download the repository as a ZIP from "
            "https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading "
            "(Code → Download ZIP) into `thesis/data/raw/tdqn/`, then write \"E16 تأیید\". Or say that E16 is skipped.",
            "3. After G6: add the P6 section to `results/FINAL_RESULTS.md` and the P6 tables and figures to the thesis "
            "text (the published SenT-In numbers are added there as a quoted reference row, as the protocol asks).",
            ""]


def stage_report() -> None:
    pub_acc = p6()["e14"]["paper_sentin_acc"]
    assert all(abs(pub_acc[m] - PUBLISHED[m][0]) < 1e-12 for m in ORDER), "published accuracies differ from p6.yaml"
    assert always_up("PUB_C_GE") and always_up("PUB_R_GE"), "the summary says SENTIN_PAPER always predicts up"
    mins = pipeline_minutes()
    L = persian_summary(pub_acc)
    for part in (section_headline(), section_published(pub_acc), section_validity(), section_diagnostics(),
                 section_data(), section_e15(), section_ran(mins), section_interpretations(), section_deviations(),
                 section_integrity(), section_wording(), section_questions()):
        L += part
    OUT.write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines)")


def main() -> None:
    global T
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true")
    a = ap.parse_args()
    if a.compute:
        stage_compute()
        return
    T = R.load()
    stage_report()


if __name__ == "__main__":
    main()
