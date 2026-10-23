"""Post-hoc diagnostics that explain notable P5 results (written after P5; not pre-registered; no verdict uses them).

Writes diag.* keys to results/results.json:
- diag.autocorr_lag1.<M>: lag-1 autocorrelation of daily index total returns over the test years 2020-2024
  (explains why a one-day execution delay raises the Sharpe of risk-off strategies);
- diag.delay_sharpe.<M>.<method>.base|delay: Sharpe per market with and without the one-day delay;
- diag.news_volume.<M>.train_median_2010_2018 | test_median_2020_2023: median local news items per trading day
  (explains why cov < 0.5 on almost every test day);
- diag.aws_coverage.high_n | low_n: AWS days with cov >= 0.5 / < 0.5 (sample size behind aws.coverage.*);
- diag.sentin_causal.<M>.p_up_min | p_up_max | share_above_half: SenT-In's seed-averaged P(up) on its E13 CAUSAL
  test set (shows the near-constant output);
- diag.base_invested_share.<M>: share of confirmation days on which BASE is invested (P(up) above the threshold).

    python scripts/p5_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import LOGS, MARKETS, PROCESSED, frozen  # noqa: E402
from smartrl.data import folds, panel  # noqa: E402
from smartrl.evaluation.analysis import fold_cov  # noqa: E402
from smartrl.evaluation.metrics import sharpe  # noqa: E402
from smartrl.sentiflow import daily as D  # noqa: E402

DELAY_METHODS = ["SMARTRL", "B6_PPOPRICE", "B8_STATIC", "BASE"]
NOTE = ("diag.*: post-hoc descriptive diagnostics computed after P5 by scripts/p5_diagnostics.py to explain notable "
        "results (delay effect, news coverage, SenT-In output, BASE exposure); not pre-registered and not used by any "
        "verdict.")


def main() -> None:
    items = []
    base = pd.read_parquet(LOGS / "returns_base.parquet")
    dl = pd.read_parquet(LOGS / "returns_delay.parquet")
    conf_b, conf_d = base[base["fold"] != "forward"], dl[dl["fold"] != "forward"]
    for m in MARKETS:
        r = pd.read_parquet(panel.processed_path(m)).set_index("date")["r_tr"]
        r.index = pd.to_datetime(r.index)
        items.append((f"diag.autocorr_lag1.{m}", float(r["2020":"2024"].autocorr(1))))
        for meth in DELAY_METHODS:
            for tag, df in (("base", conf_b), ("delay", conf_d)):
                x = df[(df["market"] == m) & (df["method"] == meth)].sort_values("date")["r"].to_numpy()
                items.append((f"diag.delay_sharpe.{m}.{meth}.{tag}", sharpe(x)))
        n = pd.read_parquet(D.daily_path(m)).set_index("date")["n_loc"].fillna(0)
        n.index = pd.to_datetime(n.index)
        items += [(f"diag.news_volume.{m}.train_median_2010_2018", float(n["2010":"2018"].median())),
                  (f"diag.news_volume.{m}.test_median_2020_2023", float(n["2020":"2023"].median()))]
    # AWS days by coverage group (same grouping as analysis.aws_analysis)
    hi = lo = 0
    for m in MARKETS:
        w = pd.concat([pd.read_parquet(LOGS / f"aws_daily_{m}_{y}.parquet") for y in range(2020, 2025)])
        cov = fold_cov(m).reindex(pd.to_datetime(w["date"])).to_numpy()
        hi += int(np.sum(cov >= 0.5))
        lo += int(np.sum(cov < 0.5))
    items += [("diag.aws_coverage.high_n", hi), ("diag.aws_coverage.low_n", lo)]
    # SenT-In output under the causal protocol (E13) and BASE's exposure in trading
    e = pd.read_parquet(PROCESSED / "e13_predictions.parquet")
    e = e[(e["protocol"] == "CAUSAL") & (e["model"] == "SENTIN")]
    thr = float(frozen()["base"]["trading_rule"]["p_up"])
    sn_all = folds.forecast_frame("SENTIN")
    for m in MARKETS:
        p = e[e["market"] == m].groupby("date")["score"].mean()
        items += [(f"diag.sentin_causal.{m}.p_up_min", float(p.min())), (f"diag.sentin_causal.{m}.p_up_max", float(p.max())),
                  (f"diag.sentin_causal.{m}.share_above_half", float((p > 0.5).mean()))]
        inv = []
        for name in folds.fold_names()[1:-1]:
            f = folds.make_fold(name, m)
            sn = sn_all[sn_all["market"] == m].drop_duplicates("date", keep="last").set_index("date").reindex(f.dates)
            pu = sn["p_up"].to_numpy()[f.test[0]:f.test[1] + 1]
            inv.append(pu > thr)
        items.append((f"diag.base_invested_share.{m}", float(np.mean(np.concatenate(inv)))))
    R.update(items, [NOTE])
    for k, v in items:
        print(k, "=", round(v, 4) if isinstance(v, float) else v)


if __name__ == "__main__":
    main()
