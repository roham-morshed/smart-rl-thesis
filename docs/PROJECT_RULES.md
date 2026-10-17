# SMART-RL thesis — implementation rules

## What this project is

This repository implements and runs every experiment of the M.Sc. thesis
"Design and Implementation of an Adaptive Intelligent Trading System Based on
Reinforcement Learning, Sentiment Analysis, and Dynamic Weighting" (SMART-RL),
Islamic Azad University, Shiraz Branch.

The design is frozen:

- `docs/SPEC.md` — data, modules, environment, rewards, baselines, statistics. The single source of truth.
- `docs/EXPERIMENTS.md` — every experiment (E0–E13), its arms, seeds, outputs and the hypothesis it serves.
- `docs/RESULTS_SCHEMA.md` — the exact keys of `results/results.json`, the tables and the figures to produce.
- `docs/THESIS_KEYS.txt` — every key the thesis reads; `tests/test_results_contract.py` (fixed before any
  experiment and never edited) checks them at G5.
- `configs/frozen.yaml` — every parameter value. Development-phase search spaces are in its `dev_grid` block.

The implementation follows exactly this design, runs the experiments phase by phase, stops at every gate,
and finally delivers `results/results.json`, `results/tables/*.csv` and `results/figures/*`.
The design is not changed on the way. If something in the SPEC turns out to be impossible (data missing,
library broken, out of memory), the gate report explains it and proposes the smallest change that keeps the
scientific meaning, and the change is applied only after it is approved.

## Hard rules

1. **No hand-typed results.** Every number that appears in a report, table or figure is computed by code
   and written to `results/results.json` first. Reports quote values from that file.
2. **Test lock.** Any data dated on or after 2020-01-01 is read only through `smartrl/data/access.py`,
   which raises `TestLockError` unless `locks/G4.lock` exists and stores the SHA-256 of `configs/frozen.yaml`.
   Exceptions allowed by the SPEC before G4: (a) SentiFlow training and evaluation on the labelled sentiment
   datasets, which carry no market dates; (b) MarketMind H4 evaluation and the annual refits of every forecast
   variant, SENTIN included (E2), which use only past data for each predicted year; (c) LTSF benchmark (E3) and
   the base-paper direction task (E13, without its SMART-RL/BASE overlap rows, which wait for P5);
   (d) building the price and news panels
   (P1), scoring the news corpus and computing daily sentiment features (P2) for all dates, and the data
   description statistics of E0 — deterministic transforms whose parameters come only from the labelled sets
   or from data before 2020. No trading result (returns, Sharpe, drawdown, decisions) on a date ≥ 2020-01-01 may
   be computed before the lock. Policies, rewards, the Sentiment Trading Module and AWS must never see 2020+
   data before the lock. `locks/G4.lock` is created only after the G4 review is approved.
3. **Causality.** A feature for day t uses only information available at the close of day t.
   `tests/test_leakage.py` (see SPEC §2.6) must pass before gate G1 and on every later phase.
4. **Seeds and run records.** Every run writes `results/runs/<run_id>/meta.json` with config hash,
   git commit, seed, start and end time, hardware, and package versions. Seeds are exactly those in the SPEC.
5. **Trials log.** Every configuration evaluated in the development phase (P4) is appended to
   `results/dev_trials.jsonl` with its daily returns on the development validation and development test
   splits. Never delete or overwrite a trial. DSR and PBO are computed from this file.
6. **Stay in this repository.** Nothing is written outside it. Raw files may be read from the paths listed
   in `configs/paths.yaml`.
7. **Own environment.** Conda environment `smartrl-thesis`, Python 3.11, versions pinned in `environment.yml`.
8. **Windows laptop (RTX 2060 6 GB, 32 GB RAM).** Use `pathlib`. Stable-Baselines3 with `DummyVecEnv`.
   Inside parallel workers call `torch.set_num_threads(1)`. Run up to 4 RL jobs in parallel with joblib
   unless the P0 benchmark recommends another number. FinBERT and MarketMind use fp16 autocast on the GPU;
   if a batch does not fit, halve it and log the change in the run record.
9. **Persian figures.** Every figure label is Persian with the Latin name in parentheses, e.g.
   `نسبت شارپ (Sharpe)`, `مدل پیشنهادی (SMART-RL)`. Tick labels use Persian digits. Font "B Nazanin";
   fallback Vazirmatn. Shape Persian text with `arabic_reshaper` and `python-bidi`. No caption inside the
   image: captions go into the thesis. Figure sizes are listed under Figures in `docs/RESULTS_SCHEMA.md`.
10. **No silent choices.** If a choice could change a result and the SPEC does not settle it, it is raised in
    the gate report with two or three concrete options and decided explicitly. Never guess silently.

## Phases and gates

Work in this order. At every gate: write `reports/P<k>_report.md` (a short Persian summary first,
then English details), update `results/results.json`, commit, and stop until the gate is approved.

| Phase | Work | Gate | Gate passes when |
| --- | --- | --- | --- |
| P0 | Environment, data-access probe, speed benchmark | G0 | every data source is reachable or has an approved replacement; FinBERT throughput and PPO steps/s measured; compute plan updated |
| P1 | Data Processing Module, news alignment, labelled sets, leakage tests | G1 | all leakage tests pass; news coverage per market reported; price and news panels saved |
| P2 | SentiFlow (E1), corpus scoring, daily sentiment features | G2 | E1 results in results.json; temperature fitted; daily features saved |
| P3 | MarketMind (E2), the LTSF benchmark (E3), the base-paper task (E13); annual refits for PPO features and BASE | G3 | E2, E3 and E13 results in results.json; out-of-sample forecast files saved for every variant SPEC §4.7 lists |
| P4 | Development phase 2010–2019 (E4): reward, STM and AWS selection; config freeze | G4 | `configs/frozen.yaml` final; dev_trials.jsonl complete; selection report; G4 review approved |
| P5 | Confirmation 2020–2024 and forward test (E5–E12), statistics, tables, figures | G5 | every key of RESULTS_SCHEMA filled or explicitly marked unavailable with a reason; `tests/test_results_contract.py` reports no missing key of `docs/THESIS_KEYS.txt`; tables and figures saved |

## Repository layout

```
smart-rl/
  environment.yml
  configs/  frozen.yaml  paths.yaml
  docs/     PROJECT_RULES.md  SPEC.md  EXPERIMENTS.md  RESULTS_SCHEMA.md  THESIS_KEYS.txt
  smartrl/  data/  sentiflow/  marketmind/  env/  agents/  stm/  aws/  baselines/  evaluation/  stats/  figures/
  scripts/  p0_probe.py  p1_build_data.py  ...  (one entry point per experiment)
  tests/    test_leakage.py  test_results_contract.py  ...
  data/     raw/  processed/         (git-ignored)
  results/  results.json  dev_trials.jsonl  runs/  tables/  figures/  logs/
  reports/  P0_report.md ... P6_report.md
  locks/    G4.lock (created only after approval)
```

## Gate report template (reports/P<k>_report.md)

```
# گزارش فاز P<k>
خلاصه (۵ تا ۱۰ خط فارسی): چه انجام شد، چه نتیجه‌ای گرفتیم، چه تصمیمی لازم است.
## Details (English)
- What was implemented (files), what ran (run ids), runtime.
- Key numbers (quoted from results.json keys).
- Deviations from SPEC (if any) and why.
- Problems and proposed fixes.
- Open questions for the gate review.
```
