# SMART-RL — Results contract

`results/results.json` is one nested JSON object. A key written below as `a.b.c` means
`{"a": {"b": {"c": value}}}`. The thesis text is bound to these exact keys, so do not rename anything.
Floats are stored unrounded (fractions, not percentages: 0.123 means 12.3 %). Missing values are `null` and
get an entry in the top-level `notes` list explaining why. Some identifiers contain dots (the PPO configuration
ids such as `R2_l10_d0.5_lr3e-4`); such an id is one JSON key — never split it into nested objects.

## Identifiers

- Markets `MKT`: `SP500`, `FTSE100`, `NIFTY50`, `SSE`, and `COMBINED` (primary endpoint, SPEC §11).
- Methods `METHOD`: `SMARTRL`, `B1_BH`, `B2_SMA`, `B3_TSMOM`, `B4_LSTM`, `B5_FINBERT`, `B6_PPOPRICE`, `B7_SAC`,
  `B8_STATIC`, `BASE`, `A1_NOSENT`, `A2_NOMM`, `A3_NOAWS`, `A4_NOCONF`, `A5_NOCURR`.
- Trading metrics `M`: `cum_return`, `ann_return`, `ann_vol`, `sharpe`, `sortino`, `mdd` (negative fraction),
  `calmar`, `turnover` (mean |Δe| per day), `win_rate`.
- Sentiment datasets `DS`: `phrasebank50`, `phrasebank_all`, `fiqa`, `twitter`, `pooled`.
- Sentiment models `SM`: `LEX`, `A`, `A_REV`, `B`, `SF`, `PROSUS`.
- Base-paper task (E13): protocols `PROTO` = `PUB_R`, `PUB_C`, `CAUSAL`; direction models `DM` = `SVM`, `CNN`,
  `LSTM`, `GRU`, `SENTIN`, `SENTIN_SF`, `MM_DIR` (CAUSAL only).
- Forecast models `FM`: `MM`, `LSTM`, `GRU`, `INFORMER`, `PATCHTST`, `DLINEAR`, `NAIVE`; volatility also `EWMA`, `HAR`.

## Keys

### meta, compute, notes
- `meta.config_hash`, `meta.git_commit`, `meta.created_utc`, `meta.packages` (dict)
- `meta.leakage_tests_passed` (true when every test of SPEC §2.6 passed at every gate from G1 to G5)
- `compute.gpu_hours`, `compute.cpu_hours`, `compute.n_runs`
- `notes` (list of strings)

### data (E0)
- `data.prices.<MKT>.start`, `.end`, `.n_days`, `.dividend_source` (string), `.n_flagged_returns`
- `data.news.n_raw`, `data.news.n_kept`, `data.news.n_duplicates`
- `data.news.source.<SOURCE>.n_raw`, `.n_kept`, `.first_date`, `.last_date`
- `data.news.tag.<GLOBAL|US|UK|IN|CN>.n`
- `data.news.market.<SP500|FTSE100|NIFTY50|SSE>.mean_daily_local`, `.share_days_with_local`, `.mean_daily_global`
- `data.news.market.<…>.share_days_with_local_by_year.<YYYY>`
- `data.sentiment_sets.<DS without pooled>.n_train`, `.n_val`, `.n_test`
- `data.word2vec.n_headlines`, `data.word2vec.vocab_size`

### sentiflow (E1)
- `sentiflow.<DS>.<SM>.acc`, `.acc_sd`, `.f1`, `.f1_sd`, `.ece`, `.ece_sd`, `.brier`, `.brier_sd`
  (`_sd` = standard deviation over seeds; absent for deterministic models)
- `sentiflow.tests.<DS>.mcnemar_sf_b_median_p`, `.mcnemar_sf_b_n_sig` (seeds with p < 0.05),
  `.mcnemar_sf_a_median_p`, `.f1_diff_sf_b`, `.f1_diff_sf_b_lo`, `.f1_diff_sf_b_hi`,
  `.f1_diff_sf_a`, `.f1_diff_sf_a_lo`, `.f1_diff_sf_a_hi`, `.mcnemar_a_arev_median_p`, `.f1_diff_a_arev`
- `sentiflow.calibration.temperature`, `.ece_before`, `.ece_after`, `.brier_before`, `.brier_after`
  (SF, pooled test); same block for B: `sentiflow.calibration_b.*`
- `sentiflow.gate.<DS>.alpha_mean`, `.alpha_sd`
- `sentiflow.e1b.<phrasebank_all|phrasebank50>.<B|SF>.acc`, `.acc_sd`, `.f1`, `.f1_sd`
- `sentiflow.scoring.n_headlines`, `.ensemble` (true/false), `.hours`

### marketmind (E2) and ltsf (E3)
- `marketmind.h<1|5|20>.<FM>.pinball`, `.pinball_sd`, `.mae`, `.rmse`, `.diracc`, `.pt_p`, `.cov80`, `.width`
- `marketmind.vol.h<5|20>.<FM or EWMA or HAR>.qlike`, `.mse`
- `marketmind.dm.h<1|5|20>.<FM other than MM>.stat`, `.p`
- `marketmind.by_market.<SP500|…>.h20.<FM>.pinball` (per-market pinball at h = 20)
- `ltsf.<ETTh1|ETTm1>.h<96|192|336|720>.mse`, `.mse_sd`, `.mae`, `.mae_sd`

### dev (E4)
- `dev.ppo.<CFG>.val_sharpe`, `.val_mdd`, `.devtest_sharpe` with CFG = `R1_lr3e-4`, `R1_lr1e-4`,
  `R2_l10_d0.5_lr3e-4`, … , `R3_lr1e-4` (mean over markets and seeds)
- `dev.stm.<CFG>.val_sharpe`, `dev.aws.<CFG>.val_sharpe`, `dev.b5.<CFG>.val_sharpe`
- `dev.imit.<20|50|100>.val_sharpe`, `.val_mdd`, `.devtest_sharpe` (IMIT arm, mean over markets and seeds)
- `dev.selected.config` (the CFG id of the chosen PPO configuration, exactly as in `dev.ppo.<CFG>`, e.g.
  `R2_l10_d0.5_lr3e-4`), `dev.selected.reward` (`"R1"`, `"R2"` or `"R3"`), `.lr` (float),
  `.stm_form` (`"STM-1"` or `"STM-2"`), `.kappa`, `.z0` (null when STM-2 is chosen), `.aws_H`, `.aws_alpha`,
  `.aws_beta`, `.b5_kappa`, `.b5_z0`, `.imit_theta_bps`
- `dev.n_trials`

### e13 (base paper's task)
- `e13.sentin.choice` (string: "sgd_mse" or "adam_bce"), `e13.sentin.val_auc.<sgd_mse|adam_bce>`
- `e13.<PROTO>.<MKT>.n_train`, `.n_test`, `.test_start`, `.test_end`, `.up_share`
- `e13.<PROTO>.<MKT>.<DM>.acc`, `.acc_sd`, `.f1`, `.f1_sd`, `.auc`, `.auc_sd`, `.prauc`, `.prauc_sd`
  (MKT without COMBINED; `_sd` over seeds, absent for SVM)
- `e13.<PROTO>.MEAN.<DM>.acc`, `.f1`, `.auc`, `.prauc` (mean over the four markets)
- `e13.overlap.<MKT>.<SMARTRL|BASE|SENTIN_CAUSAL>.acc`, `.f1`, `.auc`, `.prauc`, `.up_share` (2020-01-01 → 2022-03-31)

### trading (E5, E6) — every method, including ablations and BASE
- `trading.<MKT>.<METHOD>.<M>` for test years 2020–2024 concatenated
- `trading_year.<YYYY>.<MKT>.<METHOD>.sharpe`, `.ann_return`, `.mdd`

### tests (E9)
- `tests.lw.<METHOD>.diff` (Sharpe of SMARTRL minus METHOD on COMBINED), `.p_hac`, `.p_boot`
- `tests.h1.p_holm.<B2_SMA|B3_TSMOM|B4_LSTM|B5_FINBERT|B6_PPOPRICE|B7_SAC>`
- `tests.h1.mdd_diff.<METHOD>.est`, `.lo`, `.hi`; `tests.h1.cr_diff.<METHOD>.est`, `.lo`, `.hi`
  (SMARTRL minus METHOD on COMBINED; MDD is a negative fraction, so a positive MDD difference means a smaller
  drawdown for SMART-RL)
- `tests.h7.iut_all_significant` (bool), `tests.h7.max_p`
- `tests.spa.p_lower`, `.p_consistent`, `.p_upper`
- `tests.mcs.included` (list of METHOD ids in the 90 % set)
- `tests.friedman.chi2`, `.p`; `tests.nemenyi.cd`; `tests.nemenyi.rank.<METHOD>`
- `tests.dsr.sr_annual`, `.dsr`, `.n_trials`, `.sr_var`
- `tests.pbo.pbo`

### h2 (E10)
- `h2.<MKT or POOLED>.<SMARTRL|A1_NOSENT>.acc`, `.f1`, `.hit`
- `h2.<MKT or POOLED>.mcnemar_p`, `.f1_diff`, `.f1_diff_lo`, `.f1_diff_hi`
- `h2.coverage.<high|low>.<SMARTRL|A1_NOSENT>.acc`, `.f1`

### h5 (E8)
- `h5.<MKT>.ppo_sharpe`, `.placebo_mean`, `.placebo_p95`, `.p`
- `h5.<MKT>.ppo_mdd`, `.bh_mdd`

### h6 (E7, E11)
- `h6.<CURR|PLAIN|BC|IMIT>.aulc`, `.aulc_sd`, `.steps_to_thr`, `.steps_to_thr_sd`, `.early_mdd`, `.early_mdd_sd`,
  `.test_sharpe_seed_sd`, `.test_sharpe` (COMBINED, seed-ensemble PPO alone)
- `h6.tests.<aulc|steps_to_thr|early_mdd>.<CURR_vs_PLAIN|BC_vs_PLAIN|IMIT_vs_PLAIN|CURR_vs_IMIT>.stat`, `.p`,
  `.p_holm` (two-sided Wilcoxon p; `p_holm` = Holm adjustment over the three metrics within the same pair)
- `h6.stability.<SMARTRL|A3_NOAWS>.turnover`, `.flip_rate`; `h6.stability.tests.turnover_p`, `.flip_p`

### aws (E11)
- `aws.<SP500|…>.w_rl_mean`, `.w_rl_sd`, `.regret`
- `aws.regime.<bull|bear|sideways>.w_rl_mean`; `aws.coverage.<high|low>.w_rl_mean`

### robust and forward (E12)
- `robust.cost.<0|5|10|20>.<METHOD>.sharpe` (COMBINED)
- `robust.delay.<METHOD>.sharpe`, `.ann_return` (COMBINED)
- `robust.regime.<bull|bear|sideways>.<METHOD>.sharpe`, `.ann_return`, `.days` (pooled over markets)
- `forward.period.start`, `.end`, `.n_days`; `forward.news_days_share.<SP500|…>`
- `forward.<MKT>.<METHOD>.<M>` for SMARTRL, B1–B8, BASE

## Tables (`results/tables/<ID>.csv`)

| ID | content | keys |
| --- | --- | --- |
| T01 | data sources, periods, trading days, dividend source per market | data.prices |
| T02 | news items per source and tag; daily coverage per market | data.news |
| T03 | SentiFlow versus LEX, A, A_rev, B, PROSUS per dataset (acc, F1, ECE, Brier, mean ± sd) | sentiflow.<DS> |
| T04 | E1-b PhraseBank results with Araci's split (our columns) and calibration before/after | sentiflow.e1b, sentiflow.calibration |
| T05 | MarketMind versus comparison models per horizon (pinball, MAE, DirAcc, coverage, DM p) | marketmind |
| T06 | LTSF benchmark (our MSE/MAE) | ltsf |
| T07 | development-phase reward comparison (RQ4) and selected settings | dev |
| T08 | COMBINED results, all methods (all metrics) | trading.COMBINED |
| T09 | per-market Sharpe, MDD and annual return, all methods | trading.<MKT> |
| T10 | per-year results (appendix) | trading_year |
| T11 | ablations versus SMART-RL | trading.*.A* |
| T12 | H6 learning-curve metrics and tests | h6 |
| T13 | H5 placebo results | h5 |
| T14 | statistical tests (LW, Holm, IUT, SPA, MCS, Friedman, DSR, PBO) | tests |
| T15 | H2 decision accuracy and F1 | h2 |
| T16 | AWS weights, regret, stability | aws, h6.stability |
| T17 | cost sensitivity and execution delay | robust.cost, robust.delay |
| T18 | regime breakdown | robust.regime |
| T19 | forward test 2025–2026 | forward |
| T20 | base paper's task: direction metrics per protocol, market and model; overlap rows | e13 |

## Figures (`results/figures/<ID>.png` and `.pdf`)

Produce every figure. The thesis prints the five marked "thesis"; the others are for review in the gate
reports. Size: 16 cm wide at 300 dpi; height 10 cm for F02, F09 and F10, 12 cm for F12 and 11 cm for F14
(the thesis layout assumes these proportions); review figures 16 × 10 cm unless their panels need more.

| ID | content | use |
| --- | --- | --- |
| F01 | share of trading days with local news, per market and year | review |
| F02 | reliability diagrams of SentiFlow before and after temperature scaling | thesis |
| F03 | histogram of the fusion weight α per dataset | review |
| F04 | daily smoothed sentiment (local and global) per market, 2010–2025 | review |
| F05 | pinball loss by horizon and model (the thesis shows these numbers as a table) | review |
| F06 | interval coverage of [q10, q90] by model | review |
| F07 | cumulative wealth 2020–2024 per market (SMART-RL, B1, BASE, best other baseline) | review |
| F08 | drawdown curves per market (same methods) | review |
| F09 | COMBINED cumulative wealth 2020–2024 | thesis |
| F10 | validation Sharpe versus training steps for the four H6 arms (CURR, PLAIN, BC, IMIT) with seed bands | thesis |
| F11 | Nemenyi critical-difference diagram | review |
| F12 | AWS weight w_RL over time per market with regime shading | thesis |
| F13 | COMBINED Sharpe versus trading cost | review |
| F14 | forward-test cumulative wealth (COMBINED and per market) | thesis |
| F15 | SENTIN accuracy and AUC-ROC per protocol (PUB_R, PUB_C, CAUSAL) and market, up-share as reference | review |

Every label is Persian with the Latin name in parentheses (PROJECT_RULES.md rule 9). Use the Persian method names:
مدل پیشنهادی (SMART-RL)، خرید و نگهداری (Buy & Hold)، میانگین متحرک ۵۰/۲۰۰ (SMA)، مومنتوم سری‌زمانی (TSMOM)،
سیگنال LSTM (LSTM)، سیگنال FinBERT (FinBERT)، PPO فقط قیمت (PPO price-only)، عامل SAC (SAC)،
ترکیب ایستا (Static hybrid)، مدل مقاله‌ی مبنا (SenT-In)؛ بازوهای فرضیه‌ی ششم: برنامه‌ی آموزشی (CURR)، آموزش ساده (PLAIN)،
شروع گرم (BC)، پاداش تقلیدی (IMIT)؛ و برای بازارها: S&P 500، FTSE 100، Nifty 50،
شاخص کل بورس شانگهای (SSE Composite)، سبد ترکیبی (Combined).

## Contract with the thesis

`docs/THESIS_KEYS.txt` lists every key that the thesis text, tables and abstracts read from `results.json`
(one key per line). `tests/test_results_contract.py` ships with this kit (do not edit it): it loads
`results/results.json` and checks that every key in that file exists, matching dotted identifiers as one key. A key may hold `null` only when the top-level `notes` list explains why; the test prints those
keys. Strings must use the exact identifiers of this schema (for example `"STM-1"`, `"adam_bce"`, method ids such
as `"B3_TSMOM"` inside `tests.mcs.included`). Dates are ISO strings (`YYYY-MM-DD`). Run the test at G5; G5 passes
only when it reports no missing key.
