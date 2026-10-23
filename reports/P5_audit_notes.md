# P5 audit notes and open steps (paused 2026-10-23, user going offline)

## Audit of "illogical-looking" numbers: done so far, no bug found

1. **One-day execution delay.**
   - **Observation:** the delay *raises* the COMBINED Sharpe of the active strategies (SMART-RL 0.712 → 0.760).
   - **Code check:** `smartrl/env/evaluate.py` (`path_from_targets`, `path_ensemble`) and `smartrl/aws/aws.py`
     (`run_aws`). The decision of the close of t−1 earns the return of t+1, and AWS weights use past returns only.
     There is no look-ahead.
   - **Cause: short-term reversal of index returns in 2020–2024.** Lag-1 autocorrelation of total returns:
     S&P 500 −0.182, FTSE 100 −0.042, Nifty 50 −0.071, SSE +0.035. Risk-off moves made after a down day lose the
     next-day rebound; one day later they do not.
   - **Confirmation, SMART-RL Sharpe per market (base → delay):**

     | market | base | delay |
     | --- | --- | --- |
     | S&P 500 | 0.648 | 0.700 |
     | FTSE 100 | 0.274 | 0.302 |
     | Nifty 50 | 0.940 | 0.984 |
     | SSE | 0.114 | 0.114 |

     SSE is the only market with positive autocorrelation and the only one with no gain. B6 in SSE gets worse with
     the delay (0.372 → 0.199).
   - **Not yet in results.json:** these numbers were computed ad hoc. To do: a `scripts/p5_diagnostics.py` writing
     `diag.*` keys, plus a note marking them post-hoc and explanatory only.
2. **News coverage counts as "low" on 4,932 of 4,968 test days.**
   - **Code check:** `smartrl/data/folds.py` `fold_arrays` and `analysis.fold_cov` use each fold's *training*
     median only. There is no leakage.
   - **Cause: source composition.** The median local items per day fell sharply after 2016, when Bloomberg/Reuters
     ends:

     | market | training median (2010–2018) | 2020–2023 |
     | --- | --- | --- |
     | S&P 500 | 3,861 | 476–889 |
     | FTSE 100 | 98 | 4–6 |
     | Nifty 50 | 63.5 | 2–3 |
     | SSE | 160 | 7–11 |

   - **Effect:** the STM therefore leans on global sentiment in the test years.
   - **Decision:** this is a limitation for the thesis. Changing the frozen definition now would be a post-hoc
     change. Future work: normalise coverage with a trailing window.
3. **E13 overlap: SenT-In (causal) predicts "down" on every S&P 500 day, while BASE (trading) is always invested.**
   - **Cause:** SenT-In has collapsed to a near-constant probability close to 0.5.
     - E13 CAUSAL, S&P 500: P(up) between 0.454 and 0.499.
     - Nifty 50: between 0.526 and 0.551.
   - **Consequences:** which side of 0.5 it falls on is arbitrary. The trading refits give P(up) > 0.5 on 100 % of
     S&P 500 days in 2020–2024, so BASE equals buy-and-hold there: same Sharpe 0.651 and same drawdown −33.8 %. This
     is consistent with AUC 0.50 and is a useful point for the thesis.
   - **To do:** add to the `diag.*` keys.
4. **Consistency check: passed.**
   - 575 metric blocks: Calmar = annual return / |MDD|, drawdown and turnover ranges, signs.
   - Over 790,000 daily returns: no NaN, every exposure within [0, 1], no duplicate rows.
   - Buy-and-hold equals the index total return exactly.
   - Each fold-year has 242–254 days.
5. **Still to check:**
   - H6 test Sharpe of CURR (0.653, seeds 0–2) against A3 (0.599, 5 seeds). Expected cause: different seed sets.
     Confirm, then state it in FINAL_RESULTS.
   - `aws.coverage.high` is based on only about 36 days. Add n to FINAL_RESULTS.
   - Remaining keys of FINAL_RESULTS.

## Word thesis build: paused

- **Approved by Roham (2026-10-23):** install pandoc, poppler and LibreOffice, then build
  `thesis_writing/builder/build.py --results ../thesis/results/results.json --figures ../thesis/results/figures`.
- **conda env `thesis-build`:** creation was started and then stopped while still resolving, so nothing is
  installed. Command to re-run:

      conda create -y -n thesis-build -c conda-forge --override-channels python=3.11 pandoc poppler lxml fonttools

- **LibreOffice 26.2.6 MSI:** partial download (about 19.8 MB of about 350 MB) in
  `work Space/tools/downloads/libreoffice/`. Resume with `curl -C -` from download.documentfoundation.org and
  check the `.sha256` file.
- **Install plan:** install without admin rights via an administrative extract
  (`msiexec /a <msi> /qn TARGETDIR=<work Space>\tools\LibreOffice`), then put `soffice` on PATH for the build only.
- **Fonts:** B Nazanin is present in `C:\Windows\Fonts`. B Titr was not found. Check this before building, because
  the builder needs both.
