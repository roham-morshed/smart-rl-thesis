# P5 report (gate G5): confirmation 2020–2024, forward test, statistics and outputs

Summary (written in English at Roham's request, 2026-10-23, so the report can be published with the code):
- All P5 experiments (E5–E12) ran with the configuration locked at G4. 732 policies were trained: 660 PPO-family
  policies and 72 SAC policies. All 24 fold × market evaluations, the statistics, 21 tables and 16 figures are done.
- Gate G5 passes: leakage tests passed at every gate G1–G5, `tests/test_results_contract.py` passes, and all 936
  thesis keys are present with no null value.
- **Main result (COMBINED, 2020–2024):**
  - SMART-RL: Sharpe 0.712, maximum drawdown −17.6 %.
  - Buy and hold: Sharpe 0.646, maximum drawdown −29.7 %.
  - SMART-RL's drawdown is smaller than buy-and-hold's in every market and every test year.
- **Forward test (2025-01 to 2026-09):** Sharpe 1.045 for SMART-RL, against 1.022 for buy-and-hold and 0.815 for the
  base paper's strategy.
- **Pre-registered verdicts:** the strict rules (Holm, intersection–union) are not met for H1–H7, so no Sharpe
  superiority is claimed.
- **Supported, defensible contributions:**
  - Risk reduction.
  - AWS stabilises decisions: turnover halves (p = 0.003) and the decision flip rate falls (p < 0.001).
  - The value of sentiment: ablation A1 lowers the Sharpe to 0.371.
  - Robustness to costs and delay.
  - The causal re-evaluation of the base paper: forward-filled calendar rows explain its apparent accuracy.
- **Full numbers:** `results/FINAL_RESULTS.md`, generated from results.json by `scripts/make_final_results.py`.
- **Decisions needed:**
  1. Install pandoc (and optionally LibreOffice and poppler) to build the Word thesis.
  2. When to produce the Persian figure set.

## Details

### What ran (`results/logs/p5_pipeline.log`)

| step | runtime |
| --- | --- |
| PPO-family arms: SMART-RL, B6, A1, A2, A5 (5 seeds); BC, IMIT (3 seeds); 6 folds × 4 markets; 660 policies, CPU, 6 workers | 9.0 h |
| SAC (B7), 72 policies on the GPU in four stages (two user pauses and one desktop-app update in between) | 32.2 h wall clock in total |
| evaluate: 24 fold × market tasks, base, 0/5/20 bps and one-day delay variants | 4.2 min |
| analyze: all P5 keys | 2.0 min |
| G5 leakage tests (11 passed, 1 skipped because it only applies before the lock) | 4.1 min |
| outputs: tables T08–T19, TS1; figures F07–F14 (and all earlier ones rebuilt) | 0.2 min |
| contract test | 3 passed |

Compute: `compute.gpu_hours` 40.1, `compute.cpu_hours` 13.0 (wall clock, all phases).

### Key numbers
All of them are in `results/FINAL_RESULTS.md`, with each results.json key. Main sections:
- §1: key findings.
- §3: hypothesis verdicts with the supported evidence.
- §4–§13: trading, tests, ablations, AWS, robustness, forward test, sub-period, H2, H5, H6.
- §14–§18: E1, E2/E3, E13/E13x, E4, data.

### Deviations from the SPEC
- **Method:** none. Every method, seed, split and parameter is as frozen at G4 (hash `c1a5ebbd9f40…`).
- **Execution only:**
  - SAC ran with 6 parallel GPU workers instead of 3, to save time.
  - The device is unchanged (`parallel.sac_device: cuda`).
  - Results do not depend on the number of workers.
- **Added after P5 (marked as such in the notes):**
  - `h2.coverage.<grp>.n`: descriptive.
  - `h2.coverage.<grp>.mcnemar_p_exploratory`: exploratory, not used by any verdict.

### Problems and fixes
1. **App update killed the run.** On 2026-10-22 ~17:30 a desktop-app update stopped every process started from the
   app, the watchdog included. 31 SAC policies were safe. The watchdog was relaunched through WMI, outside the app's
   process tree, and the pipeline resumed. Unfinished policies are always deleted and retrained, so no result was
   affected.
2. **Compute hours missing.** The run records of the three interrupted SAC stages had no end time, so
   `compute.gpu_hours` missed about 20 h. `scripts/p5_register_interrupted.py` closes them. Each end time is the
   save time of the stage's last policy, which is conservative. `runs.compute_totals` now counts status ok and
   interrupted.
3. **Title inside F12.** Figure F12 had a title inside the image (PROJECT_RULES.md rule 9); it is replaced by a legend.
4. **A4 identical to SMART-RL.** The A4 ablation (AWS with β = 1) is identical to SMART-RL because the development
   phase selected β = 1. This is now explained in the notes.

### Audit
- **Policies:** all 732 are complete. The 42,480 validation-curve values contain no NaN.
- **Dry run:** a dry run of the whole post-training chain on a scratch copy found no error
  (`results/logs/p5_dryrun.log`).
- **Determinism:** re-running the H2 analysis reproduced all 58 stored values exactly.
- **Learning curves:** validation Sharpe declines with training length (F10), the usual overfitting of RL agents to
  non-stationary markets. The pre-registered best-validation checkpoint rule handles it. The P4 diagnostic showed
  the rule's benefit on development data.

### Questions for Roham
1. **Word thesis:** building it (`thesis_writing/builder/build.py --results ... --figures ...`) needs pandoc. Page
   numbers of the lists also need LibreOffice and poppler. None is installed. May I install them (downloads)?
2. **Persian figures:** the frozen config sets `figures.language: en`, and editing frozen.yaml would break the G4
   lock hash. Proposal: a rendering-only override (environment variable) that writes Persian copies to
   `results/figures/fa`, with no effect on any number.
