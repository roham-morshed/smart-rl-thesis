"""Apply the verdict rules of SPEC §12 (fixed before P5) to results/results.json and print a Markdown table.

Reads only results.json; writes nothing. The thesis applies the same rules to the same keys.

    python scripts/p5_verdicts.py            # main confirmation period 2020-2024
    python scripts/p5_verdicts.py --sub      # pre-registered supplementary sub-period 2020-2023 (H1, H7)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS  # noqa: E402

H1_BASELINES = ["B2_SMA", "B3_TSMOM", "B4_LSTM", "B5_FINBERT", "B6_PPOPRICE", "B7_SAC"]
ALPHA = 0.05


def get(tree: dict, key: str):
    v = R.lookup(tree, key)
    return None if v is R.MISSING else v


def fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def h1(t: dict, pre: str) -> tuple[str, str]:
    ps = {b: get(t, f"{pre}tests.h1.p_holm.{b}") for b in H1_BASELINES}
    if any(p is None for p in ps.values()):
        return "n/a", "missing p-values"
    sig = [b for b, p in ps.items() if p < ALPHA]
    verdict = "supported" if len(sig) == len(ps) else ("partially supported" if sig else "not supported")
    return verdict, "Holm p: " + ", ".join(f"{b} {fmt(p)}" for b, p in ps.items())


def h2(t: dict) -> tuple[str, str]:
    a, b = get(t, "h2.POOLED.SMARTRL.acc"), get(t, "h2.POOLED.A1_NOSENT.acc")
    p, lo = get(t, "h2.POOLED.mcnemar_p"), get(t, "h2.POOLED.f1_diff_lo")
    if None in (a, b, p, lo):
        return "n/a", "missing keys"
    ok = (a > b and p < ALPHA) or lo > 0
    return ("supported" if ok else "not supported"), f"acc {fmt(a)} vs {fmt(b)}, McNemar p {fmt(p)}, F1-diff lo {fmt(lo)}"


def h3(t: dict) -> tuple[str, str]:
    sf, a, b = (get(t, f"sentiflow.pooled.{m}.f1") for m in ("SF", "A", "B"))
    pa, pb = get(t, "sentiflow.tests.pooled.mcnemar_sf_a_median_p"), get(t, "sentiflow.tests.pooled.mcnemar_sf_b_median_p")
    if None in (sf, a, b, pa, pb):
        return "n/a", "missing keys"
    ok = sf > a and sf > b and pa < ALPHA and pb < ALPHA
    return ("supported" if ok else "not supported"), (f"F1 SF {fmt(sf)}, A {fmt(a)}, B {fmt(b)}; median McNemar p "
                                                      f"SF-A {fmt(pa)}, SF-B {fmt(pb)}")


def h4(t: dict) -> tuple[str, str]:
    mm = get(t, "marketmind.h20.MM.pinball")
    rows, ok = [], mm is not None
    for b in ("LSTM", "GRU"):
        pb, p = get(t, f"marketmind.h20.{b}.pinball"), get(t, f"marketmind.dm.h20.{b}.p")
        if None in (pb, p) or mm is None:
            return "n/a", "missing keys"
        ok = ok and mm < pb and p < ALPHA
        rows.append(f"{b} {fmt(pb)} (DM p {fmt(p)})")
    return ("supported" if ok else "not supported"), f"pinball h20 MM {fmt(mm)} vs " + ", ".join(rows)


def h5(t: dict) -> tuple[str, str]:
    ps = {m: get(t, f"h5.{m}.p") for m in MARKETS}
    if any(p is None for p in ps.values()):
        return "n/a", "missing p-values"
    ok = all(p < ALPHA for p in ps.values())
    return ("supported" if ok else "not supported"), "placebo p: " + ", ".join(f"{m} {fmt(p)}" for m, p in ps.items())


def h6(t: dict) -> tuple[str, str]:
    # expected direction for CURR vs PLAIN: higher AULC, fewer steps, smaller early drawdown (MDD < 0, so higher)
    wins, rows = 0, []
    for met, better in (("aulc", 1), ("steps_to_thr", -1), ("early_mdd", 1)):
        c, p_, ph = get(t, f"h6.CURR.{met}"), get(t, f"h6.PLAIN.{met}"), get(t, f"h6.tests.{met}.CURR_vs_PLAIN.p_holm")
        if None in (c, p_, ph):
            return "n/a", "missing keys"
        hit = (c - p_) * better > 0 and ph < ALPHA
        wins += hit
        rows.append(f"{met} {fmt(c)} vs {fmt(p_)} (p_holm {fmt(ph)})")
    return ("supported" if wins >= 2 else "not supported"), f"{wins}/3 metrics: " + "; ".join(rows)


def h7(t: dict, pre: str) -> tuple[str, str]:
    v, mp = get(t, f"{pre}tests.h7.iut_all_significant"), get(t, f"{pre}tests.h7.max_p")
    if v is None:
        return "n/a", "missing key"
    return ("supported" if v else "not supported"), f"max one-sided LW p {fmt(mp)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", action="store_true", help="supplementary sub-period 2020-2023 (H1 and H7 only)")
    args = ap.parse_args()
    t = R.load()
    pre = "supp.sub2020_2023." if args.sub else ""
    rows = [("H1", *h1(t, pre)), ("H7", *h7(t, pre))]
    if not args.sub:
        rows = [("H1", *h1(t, pre)), ("H2", *h2(t)), ("H3", *h3(t)), ("H4", *h4(t)), ("H5", *h5(t)), ("H6", *h6(t)),
                ("H7", *h7(t, pre))]
    print("| hypothesis | verdict | evidence |\n| --- | --- | --- |")
    for h, v, e in rows:
        print(f"| {h} | {v} | {e} |")


if __name__ == "__main__":
    main()
