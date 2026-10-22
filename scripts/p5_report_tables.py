"""Markdown tables for reports/P5_report.md, printed from results/results.json only (no hand-typed numbers).

    python scripts/p5_report_tables.py > results/logs/p5_report_tables.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS  # noqa: E402

METHODS = ["SMARTRL", "B1_BH", "B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC", "B8_STATIC", "BASE"]
ABL = ["SMARTRL", "A1_NOSENT", "A2_NOMM", "A3_NOAWS", "A4_NOCONF", "A5_NOCURR"]
PCT = {"ann_return", "ann_vol", "cum_return", "mdd", "win_rate"}


def get(t, key):
    v = R.lookup(t, key)
    return None if v is R.MISSING else v


def f(v, metric: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{100 * v:.1f} %" if metric in PCT else f"{v:.3f}"
    return str(v)


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def trading(t, pre: str, block: str, methods: list[str], metrics: list[str]) -> str:
    rows = [[m] + [f(get(t, f"{pre}{block}.{m}.{k}"), k) for k in metrics] for m in methods]
    return table(["method"] + metrics, rows)


def main() -> None:
    t = R.load()
    mets = ["sharpe", "sortino", "ann_return", "ann_vol", "mdd", "calmar", "cum_return", "turnover"]
    print("### Confirmation 2020-2024, COMBINED (equal-weight four-market portfolio)\n")
    print(trading(t, "", "trading.COMBINED", METHODS, mets))
    print("\n### Sharpe per market (confirmation 2020-2024)\n")
    rows = [[m] + [f(get(t, f"trading.{mk}.{m}.sharpe")) for mk in MARKETS] for m in METHODS]
    print(table(["method", *MARKETS], rows))
    print("\n### Maximum drawdown per market (confirmation 2020-2024)\n")
    rows = [[m] + [f(get(t, f"trading.{mk}.{m}.mdd"), "mdd") for mk in MARKETS] for m in METHODS]
    print(table(["method", *MARKETS], rows))
    print("\n### Ledoit-Wolf Sharpe-difference tests, SMART-RL minus method (COMBINED)\n")
    rows = [[m, f(get(t, f"tests.lw.{m}.diff")), f(get(t, f"tests.lw.{m}.p_hac")), f(get(t, f"tests.lw.{m}.p_boot")),
             f(get(t, f"tests.h1.p_holm.{m}"))] for m in METHODS[1:] + ABL[1:]]
    print(table(["versus", "Sharpe diff", "p (HAC)", "p (boot)", "p Holm (H1)"], rows))
    print("\n### H1 secondary: MDD and cumulative-return differences (stationary bootstrap 95 %)\n")
    rows = []
    for m in ["B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC"]:
        rows.append([m] + [f"{f(get(t, f'tests.h1.{d}.{m}.est'))} [{f(get(t, f'tests.h1.{d}.{m}.lo'))}, "
                           f"{f(get(t, f'tests.h1.{d}.{m}.hi'))}]" for d in ("mdd_diff", "cr_diff")])
    print(table(["versus", "MDD diff", "cum-return diff"], rows))
    print("\n### Ablations, COMBINED\n")
    print(trading(t, "", "trading.COMBINED", ABL, ["sharpe", "ann_return", "mdd", "turnover"]))
    print("\n### Joint tests\n")
    rows = [[k, f(get(t, k))] for k in ("tests.h7.iut_all_significant", "tests.h7.max_p", "tests.spa.p_consistent",
                                         "tests.spa.p_lower", "tests.spa.p_upper", "tests.mcs.included",
                                         "tests.friedman.chi2", "tests.friedman.p", "tests.nemenyi.cd",
                                         "tests.nemenyi.rank.SMARTRL", "tests.dsr.sr_annual", "tests.dsr.dsr",
                                         "tests.dsr.n_trials", "tests.pbo.pbo")]
    print(table(["key", "value"], rows))
    print("\n### Decision accuracy (H2)\n")
    rows = []
    for mk in [*MARKETS, "POOLED"]:
        rows.append([mk] + [f(get(t, f"h2.{mk}.{a}.{k}")) for a in ("SMARTRL", "A1_NOSENT") for k in ("acc", "f1")]
                    + [f(get(t, f"h2.{mk}.mcnemar_p")),
                       f"{f(get(t, f'h2.{mk}.f1_diff'))} [{f(get(t, f'h2.{mk}.f1_diff_lo'))}, {f(get(t, f'h2.{mk}.f1_diff_hi'))}]"])
    print(table(["market", "SMART-RL acc", "SMART-RL F1", "A1 acc", "A1 F1", "McNemar p", "F1 diff [95 %]"], rows))
    print("\n### Placebo (H5)\n")
    rows = [[mk] + [f(get(t, f"h5.{mk}.{k}"), "mdd" if "mdd" in k else "") for k in
                    ("ppo_sharpe", "placebo_mean", "placebo_p95", "p", "ppo_mdd", "bh_mdd")] for mk in MARKETS]
    print(table(["market", "PPO Sharpe", "placebo mean", "placebo p95", "p", "PPO MDD", "B&H MDD"], rows))
    print("\n### Learning curves (H6)\n")
    rows = [[a] + [f(get(t, f"h6.{a}.{k}")) for k in ("aulc", "steps_to_thr", "early_mdd", "test_sharpe")]
            for a in ("CURR", "PLAIN", "BC", "IMIT")]
    print(table(["arm", "AULC", "steps to STM Sharpe", "early MDD", "test Sharpe"], rows))
    rows = [[f"{m} {p}", f(get(t, f"h6.tests.{m}.{p}.p_holm"))] for p in ("CURR_vs_PLAIN", "BC_vs_PLAIN", "IMIT_vs_PLAIN",
                                                                        "CURR_vs_IMIT")
            for m in ("aulc", "steps_to_thr", "early_mdd")]
    print("\n" + table(["metric, pair", "p Holm"], rows))
    print("\n### AWS weights\n")
    rows = [[mk] + [f(get(t, f"aws.{mk}.{k}")) for k in ("w_rl_mean", "w_rl_sd", "regret")] for mk in MARKETS]
    print(table(["market", "w_RL mean", "w_RL sd", "regret"], rows))
    print("\n### Robustness: COMBINED Sharpe by cost (bps) and with one-day delay\n")
    rows = [[m] + [f(get(t, f"robust.cost.{c}.{m}.sharpe")) for c in (0, 5, 10, 20)] + [f(get(t, f"robust.delay.{m}.sharpe"))]
            for m in ("SMARTRL", "B1_BH", "B6_PPOPRICE", "B8_STATIC", "BASE")]
    print(table(["method", "0", "5", "10", "20", "delay 1 day"], rows))
    print("\n### Forward test (2025-01 to 2026-09), COMBINED\n")
    print(trading(t, "", "forward.COMBINED", METHODS, ["sharpe", "mdd", "cum_return", "turnover"]))
    print("\nnews-day share: " + ", ".join(f"{mk} {f(get(t, f'forward.news_days_share.{mk}'))}" for mk in MARKETS))
    print("\n### Supplementary sub-period 2020-2023 (pre-registered), COMBINED\n")
    print(trading(t, "supp.sub2020_2023.", "trading.COMBINED", METHODS, ["sharpe", "mdd", "cum_return"]))
    rows = [[m, f(get(t, f"supp.sub2020_2023.tests.lw.{m}.diff")), f(get(t, f"supp.sub2020_2023.tests.lw.{m}.p_hac")),
             f(get(t, f"supp.sub2020_2023.tests.h1.p_holm.{m}"))] for m in METHODS[1:]]
    print("\n" + table(["versus", "Sharpe diff", "p (HAC)", "p Holm"], rows))
    print("\n### Base-paper task overlap (AUC of next-day direction)\n")
    rows = [[mk] + [f(get(t, f"e13.overlap.{mk}.{m}.auc")) for m in ("SMARTRL", "BASE", "SENTIN_CAUSAL")] for mk in MARKETS]
    print(table(["market", "SMART-RL", "BASE", "SENTIN (causal)"], rows))
    print("\n### Compute\n")
    print(table(["key", "value"], [[k, f(get(t, f"compute.{k}"))] for k in ("cpu_hours", "gpu_hours", "n_runs")]))


if __name__ == "__main__":
    main()
