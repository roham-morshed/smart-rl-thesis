"""P0 compute plan and news-subset proposal, computed from the probe and benchmark files.

Writes `p0.compute_plan.*`, `p0.news_proposal.*` and `p0.etf_summary.*` to results/results.json and prints the
markdown tables used in reports/P0_report.md and docs/EXPERIMENTS.md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import RESULTS, frozen  # noqa: E402

# Proposed classification of the 24 subsets (SPEC §2.3: keep finance and markets subsets; exclude Reddit,
# general-news and recommendation-dataset subsets).
NEWS_CLASS = {
    "fnspid_news": ("keep", "FNSPID financial news (expected in SPEC)"),
    "bloomberg_reuters": ("keep", "Bloomberg / Reuters (expected in SPEC)"),
    "benzinga_6000stocks": ("keep", "Benzinga (expected in SPEC)"),
    "yahoo_finance_felixdrinkall": ("keep", "Yahoo Finance (expected in SPEC)"),
    "yahoo_finance_articles": ("keep", "Yahoo Finance (expected in SPEC); only subset covering 2025"),
    "cnbc_headlines": ("keep", "CNBC (expected in SPEC)"),
    "sp500_daily_headlines": ("keep", "S&P 500 headlines (expected in SPEC)"),
    "nyt_headlines_1990_2020": ("keep_business", "NYT; keep only business/financial desk rows (expected in SPEC)"),
    "nyt_headlines_2010_2021": ("keep_business", "NYT; keep only business/financial desk rows (expected in SPEC)"),
    "nyt_articles_2000_present": ("keep_business", "NYT; keep only business/financial desk rows (expected in SPEC)"),
    "us_financial_news_2018_jeet": ("keep", "U.S. financial news, multi-publisher (finance subset)"),
    "finsen_us_2007_2023": ("keep", "FinSen U.S. financial news (finance subset)"),
    "sentarl_combined": ("keep", "SentARL financial news for 20 assets (finance subset)"),
    "gold_news_kaggle": ("keep", "gold/commodity market headlines (markets subset)"),
    "reddit_worldnews_2008_2016": ("exclude", "Reddit"),
    "reddit_finance_sp500": ("exclude", "Reddit"),
    "djia_stock_headlines": ("exclude", "top-25 world headlines per day (Reddit r/worldnews style), general news"),
    "frontpage_news": ("exclude", "general news front pages"),
    "headlines_10sites_2007_2022": ("exclude", "general news, 10 major sites"),
    "all_the_news_2": ("exclude", "general news"),
    "american_news_jonasbecker": ("exclude", "general news crawl"),
    "huffpost_news": ("exclude", "general news"),
    "wikinews_articles": ("exclude", "general news"),
    "mind_news_2019": ("exclude", "recommendation dataset (MIND)"),
}


def news_proposal(probe: dict) -> tuple[dict, str]:
    subs = probe["hf"]["subsets"]
    rows, out = [], {"subsets": {}}
    for name, s in sorted(subs.items(), key=lambda kv: (NEWS_CLASS.get(kv[0], ("?",))[0] == "exclude", -kv[1]["rows"])):
        dec, why = NEWS_CLASS.get(name, ("ask", "unclassified"))
        out["subsets"][name] = {"decision": dec, "reason": why, "rows": s["rows"], "gb": s["gb"],
                                "date_min": str(s["date_min"])[:10], "date_max": str(s["date_max"])[:10]}
        rows.append(f"| {name} | {s['rows']:,} | {s['gb']:.2f} | {str(s['date_min'])[:10]} | {str(s['date_max'])[:10]} | "
                    f"**{dec}** | {why} |")
    keep = [n for n, v in out["subsets"].items() if v["decision"] != "exclude"]
    out["keep"] = keep
    out["keep_rows"] = sum(out["subsets"][n]["rows"] for n in keep)
    out["keep_gb"] = sum(out["subsets"][n]["gb"] for n in keep)
    out["all_rows"] = probe["hf"]["total_rows"]
    out["all_gb"] = probe["hf"]["total_gb"]
    out["latest_kept_date"] = max(out["subsets"][n]["date_max"] for n in keep)
    out["latest_kept_date_excl_yahoo2025_nyt"] = max(out["subsets"][n]["date_max"] for n in keep
                                                    if n not in ("yahoo_finance_articles", "nyt_articles_2000_present"))
    table = ("| subset | rows | GB | first | last | proposal | reason |\n| --- | ---: | ---: | --- | --- | --- | --- |\n"
             + "\n".join(rows))
    return out, table


def compute_plan(bench: dict, sac: dict) -> tuple[dict, str]:
    cfg = frozen()
    ppo_steps, sac_steps = cfg["ppo"]["total_timesteps"], cfg["sac"]["total_timesteps"]
    rl = bench["rl"]
    best_jobs = max((int(k[8:]) for k in rl if k.startswith("ppo_jobs")),
                    key=lambda n: rl[f"ppo_jobs{n}"]["aggregate_steps_per_s"])
    ppo_agg = rl[f"ppo_jobs{best_jobs}"]["aggregate_steps_per_s"]
    sac_agg = sac["cuda_x3"]["aggregate_steps_per_s"]
    n_mkt, n_folds, n5, n3 = 4, 5, 5, 3
    runs = {
        "dev_ppo_grid": 12 * n_mkt * n3,
        "dev_imit": 3 * n_mkt * n3,
        "conf_smartrl_ppo": n_folds * n_mkt * n5,
        "conf_b6_ppo_price": n_folds * n_mkt * n5,
        "conf_a1_nosent": n_folds * n_mkt * n5,
        "conf_a2_nomm": n_folds * n_mkt * n5,
        "conf_a5_nocurr": n_folds * n_mkt * n5,
        "conf_bc": n_folds * n_mkt * n3,
        "conf_imit": n_folds * n_mkt * n3,
        "fwd_smartrl_ppo": n_mkt * n5,
        "fwd_b6_ppo_price": n_mkt * n5,
    }
    sac_runs = {"conf_b7_sac": n_folds * n_mkt * n3, "fwd_b7_sac": n_mkt * n3}
    ppo_total = sum(runs.values())
    sac_total = sum(sac_runs.values())
    ppo_hours = ppo_total * ppo_steps / ppo_agg / 3600
    sac_hours = sac_total * sac_steps / sac_agg / 3600
    fb = bench["finbert"]
    infer = max(fb["infer_bs256_headlines_per_s"], fb["infer_bs512_headlines_per_s"])
    mm_epoch = bench["marketmind"]["epoch_s"]
    ltsf = bench["ltsf"]["epoch_s_h720"]
    # SentiFlow branch B fine-tuning: ≈ 13.8k training sentences × 4 epochs per seed (exact size known in P1)
    sf_train_h = 5 * 13800 * 4 / fb["train_bs32_samples_per_s"] / 3600
    mm_trainings = 5 * 16 + 80
    mm_hours_upper = mm_trainings * cfg["marketmind"]["train"]["max_epochs"] * mm_epoch / 3600
    ltsf_hours_upper = 3 * 4 * (ltsf["ETTh1"] + ltsf["ETTm1"]) * cfg["ltsf_benchmark"]["train"]["max_epochs"] / 3600
    plan = {
        "ppo_best_parallel_jobs": best_jobs, "ppo_aggregate_steps_per_s": ppo_agg,
        "ppo_per_job_steps_per_s": rl[f"ppo_jobs{best_jobs}"]["per_job_steps_per_s"],
        "ppo_run_minutes_at_best_jobs": ppo_steps / rl[f"ppo_jobs{best_jobs}"]["per_job_steps_per_s"] / 60,
        "ppo_runs": runs, "ppo_runs_total": ppo_total, "ppo_wall_hours": ppo_hours,
        "sac_runs": sac_runs, "sac_runs_total": sac_total, "sac_aggregate_steps_per_s_cuda_x3": sac_agg,
        "sac_wall_hours_cuda_x3": sac_hours,
        "sac_wall_hours_cpu_x4": sac_total * sac_steps / (4 * rl["sac_jobs4"]["train_steps_per_s"]) / 3600,
        "finbert_infer_headlines_per_s": infer,
        "finbert_scoring_hours_per_seed_per_10M": 1e7 / infer / 3600,
        "finbert_max_headlines_for_5seed_ensemble_in_12h": 12 * 3600 * infer / 5,
        "sentiflow_branch_b_training_hours_5_seeds": sf_train_h,
        "marketmind_epoch_s_largest_window": mm_epoch, "marketmind_trainings": mm_trainings,
        "marketmind_hours_upper_bound": mm_hours_upper,
        "ltsf_hours_upper_bound": ltsf_hours_upper,
    }
    t = [
        "| block | runs | measured speed | estimate |", "| --- | ---: | --- | --- |",
        f"| SentiFlow branch B fine-tuning (5 seeds) + A, A_rev, fusion | 20 | {fb['train_bs32_samples_per_s']:.0f} samples/s (batch 32, fp16) | ≈ {sf_train_h:.1f} GPU h (+ E1-b < 0.5 h) |",
        f"| Corpus scoring with FinBERT | per seed | {infer:.0f} headlines/s (fp16, max_len 64) | {plan['finbert_scoring_hours_per_seed_per_10M']:.1f} GPU h per seed per 10 M headlines; 5-seed ensemble fits the 12 h rule up to {plan['finbert_max_headlines_for_5seed_ensemble_in_12h']/1e6:.1f} M headlines |",
        f"| MarketMind and comparison models (E2, incl. annual refits) | ≈ {mm_trainings} | {mm_epoch:.1f} s per epoch on the largest window (16.8 k samples) | ≤ {mm_hours_upper:.1f} GPU h (60 epochs each; early stopping makes it shorter) |",
        f"| LTSF benchmark (E3) | 24 | ETTh1 {ltsf['ETTh1']:.1f} s / ETTm1 {ltsf['ETTm1']:.1f} s per epoch (h = 720) | ≤ {ltsf_hours_upper:.1f} GPU h (100 epochs each; patience 10 makes it shorter) |",
        f"| PPO runs (P4 grid + IMIT, P5 E5–E7, forward) | {ppo_total} | {ppo_agg:.0f} steps/s with {best_jobs} parallel jobs ({plan['ppo_run_minutes_at_best_jobs']:.1f} min per 300 k-step run) | ≈ {ppo_hours:.1f} h |",
        f"| SAC runs (B7, P5 + forward) | {sac_total} | {sac_agg:.0f} steps/s with 3 jobs on the GPU (CPU: {4*rl['sac_jobs4']['train_steps_per_s']:.0f} steps/s with 4 jobs) | ≈ {sac_hours:.1f} h on the GPU, parallel to PPO on the CPU |",
    ]
    return plan, "\n".join(t)


def main() -> None:
    probe = json.loads((RESULTS / "p0_probe.json").read_text(encoding="utf-8"))
    bench = json.loads((RESULTS / "p0_benchmark.json").read_text(encoding="utf-8"))
    sac = json.loads((RESULTS / "p0_sac_bench.json").read_text(encoding="utf-8"))
    news, news_table = news_proposal(probe)
    plan, plan_table = compute_plan(bench, sac)
    etf = {m: {k: v for k, v in e.items() if k != "ttm_yield_mean_by_year"} for m, e in probe["yf"]["etf_check"].items()}
    R.update([("p0.news_proposal", news), ("p0.compute_plan", plan), ("p0.etf_summary", etf),
              ("p0.benchmark.sac_device", sac)])
    out = RESULTS / "p0_tables.md"
    out.write_text("## News subsets\n\n" + news_table + "\n\n## Compute plan\n\n" + plan_table + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(json.dumps(R._clean({k: v for k, v in news.items() if k != "subsets"}), indent=1))
    print(json.dumps(R._clean(plan), indent=1))


if __name__ == "__main__":
    main()
