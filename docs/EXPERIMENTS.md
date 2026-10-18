# SMART-RL — Experiment matrix

Every experiment below is implemented as one entry-point script under `scripts/` and writes only the
results keys listed for it (RESULTS_SCHEMA.md). Hypothesis numbers follow the approved proposal
(H1–H7, research questions RQ1–RQ4). Runtimes are first estimates; replace them with the P0 benchmark.

| id | phase | serves | short name |
| --- | --- | --- | --- |
| E0 | P1 | data description | data inventory and coverage |
| E1 | P2 | H3, RQ3 | SentiFlow versus single branches and the lexicon |
| E2 | P3 | H4 | MarketMind versus LSTM, GRU, Informer, PatchTST, DLinear; annual refits |
| E3 | P3 | H4, literature comparison | MarketMind backbone on ETTh1 and ETTm1 |
| E4 | P4 | RQ4, DSR, PBO | development phase: reward, STM, AWS selection |
| E5 | P5 | H1, H7, base-paper comparison | main confirmation 2020–2024 |
| E6 | P5 | H1, H2 | ablations A1–A5 |
| E7 | P5 | H6 | learning curves: curriculum, plain, behaviour cloning |
| E8 | P5 | H5 | PPO versus random placebo policies |
| E9 | P5 | H1, H7 | statistical tests, DSR, PBO, SPA, MCS, Friedman–Nemenyi |
| E10 | P5 | H2, RQ2 | decision accuracy and F1 |
| E11 | P5 | H6, RQ1 | AWS weights, regret, stability |
| E12 | P5 | robustness, unseen data | costs, execution delay, regimes, coverage, forward test 2025–2026 |
| E13 | P3 (+P5 overlap rows) | base-paper comparison | the base paper's own task: next-day direction, three protocols |

## E0 — Data inventory (P1)
- Prices: first and last date, trading days per market, flagged extreme returns, dividend proxy used.
- News: rows read per source, kept after cleaning, duplicates removed, items per tag (GLOBAL, US, UK, IN, CN),
  mean daily items per market, share of trading days with at least one local item, per year.
- Labelled sets: train/val/test sizes per dataset.
- Outputs: `data.*`; tables T01, T02; figure F01 (daily local item counts per market, yearly means).

## E1 — SentiFlow (P2, H3)
- Models: LEX, A (= the base paper's GRU→CNN sentiment model), A_rev, B, SF, PROSUS (SPEC §3.7).
  Seeds 0–4 (LEX and PROSUS deterministic).
- Evaluation sets: test splits of phrasebank50, phrasebank_all, fiqa, twitter, and the pooled test set.
- E1-b: SF and B on phrasebank_all and phrasebank50 with Araci's 64/16/20 split (SPEC §2.4, seed 2026), seeds 0–4.
- Calibration: T, ECE and Brier before and after for SF and B.
- Gate statistics: distribution of α on each test set.
- Corpus scoring (after the H3 numbers are saved): SF ensemble, B, and A (uncalibrated, for SENTIN) for every headline.
- Outputs: `sentiflow.*`; tables T03, T04; figures F02 (reliability diagrams before/after), F03 (α histogram),
  F04 (daily Se_loc and Se_glob per market, 2010–2025).
- Estimate: 6–10 GPU hours including corpus scoring.

## E2 — MarketMind (P3, H4)
- Models: MM, LSTM, GRU, INFORMER, PATCHTST, DLINEAR, NAIVE; volatility references EWMA and HAR.
- Test years 2020–2024 from annual refits; seeds 0–2 (NAIVE, EWMA, HAR deterministic).
- Feature refits (seed 0, years 2010–2024 + forward): MM, MM_NOSENT, INFORMER, LSTM_PRICE, and SENTIN
  (P(up) for the BASE method; its optimiser/loss choice comes from E13, so run E13's CAUSAL pilot first).
- Outputs: `marketmind.*`; table T05; figures F05 (pinball loss by horizon and model), F06 (interval coverage).
- Estimate: 10–15 GPU hours.

## E3 — LTSF benchmark (P3, comparison with published work)
- MarketMind backbone on ETTh1 and ETTm1, horizons 96/192/336/720, seeds 0–2 (SPEC §15).
- Outputs: `ltsf.*`; table T06 (our columns only; published columns are added in the thesis).
- Estimate: 2–4 GPU hours.

## E4 — Development phase (P4)
- Split 2010–2019 per market 70/15/15 (SPEC §1).
- Order inside E4: (1) STM grid and STM selection; (2) B5 grid; (3) PPO grid with the selected STM inside the
  curriculum; (4) AWS grid with the selected PPO and STM.
- STM grid: STM-1 (κ ∈ {0.5, 1, 2}, z0 ∈ {0, 0.5}) and STM-2 = 7 configurations (no training).
- B5 grid: STM-1 parameters on Branch B sentiment (6 configurations).
- PPO grid (full SMART-RL state, curriculum on): reward ∈ {R1, R2(λ1, λ2) for λ1 ∈ {10, 50}, λ2 ∈ {0.5, 1.0},
  R3} × learning rate ∈ {3e−4, 1e−4} = 12 configurations × 4 markets × seeds 0–2.
- AWS grid with the selected PPO and STM: H ∈ {20, 60} × α ∈ {1, 2, 5} × β ∈ {0.5, 0.7, 1.0} = 18 configurations.
- IMIT comparison arm (not a candidate for SMART-RL): imitation reward with ϑ ∈ {20, 50, 100} bps, selected
  learning rate, no curriculum, 4 markets × seeds 0–2 = 36 runs; ϑ chosen by the same rule.
- Selection rule: highest mean development-validation Sharpe across the four markets; ties → lower turnover.
  The development test split is used only to report and for PBO; it never changes a choice.
- Log every configuration to `results/dev_trials.jsonl`.
- Write the chosen values into `configs/frozen.yaml` and `dev.selected.*` (including `dev.selected.config`),
  recompute the hash, report, STOP at G4.
- Outputs: `dev.*`; table T07 (reward comparison, RQ4).
- Estimate: 144 grid runs + 36 IMIT runs, about 4 hours with 4 parallel jobs (update after P0).

## E5 — Main confirmation 2020–2024 (P5)
- Folds Y = 2020 … 2024 (SPEC §1). For each fold, market and seed: train, select checkpoint on validation,
  evaluate on test.
- Methods: SMART-RL (PPO + curriculum, STM, AWS), B1–B8, BASE.
- PPO-type training runs: SMART-RL PPO 100; B6 100; B7 (SAC) 60. BASE needs no RL training (SENTIN refits, E2).
- Outputs: `trading.*`; tables T08 (COMBINED, all methods), T09 (per market: Sharpe, MDD, annual return),
  T10 (per year, appendix); figures F07 (cumulative wealth per market), F08 (drawdowns per market),
  F09 (COMBINED wealth: SMART-RL, B1, best baseline, BASE).

## E6 — Ablations (P5)
- A1 (100 runs), A2 (100 runs), A5 (100 runs); A3 and A4 reuse existing policies.
- Outputs: `trading.<MKT>.<A1_NOSENT|A2_NOMM|A3_NOAWS|A4_NOCONF|A5_NOCURR>.*` and `tests.lw.<A…>.*`; table T11.

## E7 — Learning curves (P5, H6)
- Arms: curriculum (= SMART-RL PPO runs), plain (= A5 runs), BC (+ plain PPO after BC, seeds 0–2 → 60 runs),
  IMIT (imitation reward with the selected ϑ, no curriculum, seeds 0–2 → 60 runs; Goluža et al. 2024).
- Uses the validation curves logged every 10 000 steps.
- Outputs: `h6.*`; table T12; figure F10 (mean validation Sharpe versus steps with seed bands, per arm).

## E8 — Placebo (P5, H5)
- 1 000 random exposure paths per market-year for the SMART-RL PPO ensemble (SPEC §12).
- Outputs: `h5.*`; table T13.

## E9 — Statistics (P5)
- LW tests (HAC and bootstrap), Holm, IUT for H7, SPA, MCS, Friedman–Nemenyi, DSR, PBO (SPEC §12).
- Outputs: `tests.*`; table T14; figure F11 (critical-difference diagram, Persian labels).

## E10 — Decision accuracy (P5, H2)
- Decisions of SMART-RL and A1 versus the ground truth of SPEC §13; per market, pooled, and by coverage.
- Outputs: `h2.*`; table T15.

## E11 — AWS analysis (P5)
- Mean and dispersion of w_RL per market, by regime and by coverage; empirical regret; stability versus A3.
- Outputs: `aws.*`; table T16; figure F12 (w_RL over time per market with regime shading).

## E12 — Robustness and forward test (P5)
- Costs 0/5/10/20 bps; execution at close t+1; regimes; coverage split (SPEC §14).
- Forward test 2025-01-01 → last day ≤ 2026-09-30 for SMART-RL, B1–B8, BASE; report the share of days with news.
- Outputs: `robust.*`, `forward.*`; tables T17 (costs and delay), T18 (regimes), T19 (forward);
  figures F13 (Sharpe versus cost), F14 (forward cumulative wealth, COMBINED and per market).

## E13 — The base paper's own task (P3; overlap rows in P5)
- Next-day direction of each index under PUB_R, PUB_C and CAUSAL (SPEC §15), window from the first common date
  to 2022-03-31.
- First a pilot on the CAUSAL validation split: SENTIN with {SGD + MSE (as published), Adam + BCE}, seed 0;
  keep the higher validation AUC (SPEC §9.1) and record it in `e13.sentin.choice`.
- Models per protocol and market: SVM, CNN, LSTM, GRU, SENTIN, SENTIN_SF (seeds 0–2 for neural models);
  MM_DIR under CAUSAL only.
- P5 adds the overlap rows (2020-01-01 → 2022-03-31): SMART-RL, BASE and SENTIN-CAUSAL with the same metrics.
- Outputs: `e13.*`; table T20 (our columns; the paper's published Tables 4–7 are added in the thesis);
  figure F15 (accuracy and AUC-ROC of SENTIN per protocol and market, with the up-share as a reference line).
- Estimate: 2–3 GPU hours.

## Run budget summary (first estimate, RTX 2060 6 GB, 8 cores)
| block | runs | estimate |
| --- | --- | --- |
| SentiFlow training and scoring | 5 seeds × 4 models + scoring | 6–10 GPU h |
| MarketMind and comparison models | ≈ 150 small trainings | 10–15 GPU h |
| LTSF benchmark | 24 trainings | 2–4 GPU h |
| Base-paper task (E13) and SENTIN refits | ≈ 250 small trainings | 3–5 GPU h |
| Development PPO (incl. IMIT ϑ) | 180 | ≈ 4 h with 4 jobs |
| Confirmation RL runs (E5–E7, incl. IMIT arm) | ≈ 780 | 15–25 h with 4 jobs |
| Statistics, placebo, figures | — | 1–2 h |

## Run budget measured in P0 (scripts/p0_benchmark.py, results.json `p0.benchmark` and `p0.compute_plan`)

RTX 2060 6 GB, AMD Ryzen 8 cores / 16 threads, 32 GB RAM. PPO on the CPU (DummyVecEnv, 8 envs, one torch thread per job); SAC on the GPU (3 parallel jobs) at the same time as the PPO jobs.

| block | runs | measured speed | estimate |
| --- | ---: | --- | --- |
| SentiFlow branch B fine-tuning (5 seeds) + A, A_rev, fusion | 20 | 241 samples/s (batch 32, fp16) | ≈ 0.3 GPU h (+ E1-b < 0.5 h) |
| Corpus scoring with FinBERT | per seed | 1192 headlines/s (fp16, max_len 64) | 2.3 GPU h per seed per 10 M headlines; 5-seed ensemble fits the 12 h rule up to 10.3 M headlines |
| MarketMind and comparison models (E2, incl. annual refits) | ≈ 160 | 6.0 s per epoch on the largest window (16.8 k samples) | ≤ 15.9 GPU h (60 epochs each; early stopping makes it shorter) |
| LTSF benchmark (E3) | 24 | ETTh1 4.2 s / ETTm1 18.1 s per epoch (h = 720) | ≤ 7.4 GPU h (100 epochs each; patience 10 makes it shorter) |
| PPO runs (P4 grid + IMIT, P5 E5–E7, forward) | 840 | 5604 steps/s with 6 parallel jobs (3.3 min per 300 k-step run) | ≈ 12.5 h |
| SAC runs (B7, P5 + forward) | 72 | 120 steps/s with 3 jobs on the GPU (CPU: 83 steps/s with 4 jobs) | ≈ 33.2 h on the GPU, parallel to PPO on the CPU |

Recommendation: `parallel.rl_jobs: 6` for PPO (aggregate throughput +35 % over 4 jobs) and SAC with `device: cuda`, 3 parallel jobs.
