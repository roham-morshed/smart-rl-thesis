# گزارش فاز P4 (بسته‌ی تصمیم دروازه‌ی G4)

خلاصه:
- فاز توسعه فقط روی داده‌ی 2010 تا 2019 اجرا شد. انتخاب‌ها با قاعده‌ی SPEC انجام شد: بیشترین میانگین Sharpe اعتبارسنجی چهار بازار.
  - **پاداش:** R2 با λ1 = 10 و λ2 = 1.0
  - **نرخ یادگیری:** 1e-4
  - **STM:** STM-1 با κ = 0.5 و z0 = 0.5
  - **AWS:** H = 60، α = 5، β = 1.0
  - **B5:** κ = 0.5 و z0 = 0.5
  - **IMIT:** ϑ = 50bp
- در کل 50 پیکربندی آزمایش و ثبت شد. همه در DSR و PBO شمرده می‌شوند.
- **نتیجه‌ی کلیدی روی داده‌ی آزمونِ توسعه (2018-07 تا 2019-12، سبد ترکیبی):**
  - SMART-RL کامل: Sharpe برابر 0.59، بیشترین افت −7.4٪.
  - خرید و نگهداری: Sharpe برابر 0.72، بیشترین افت −11.8٪.
  - پس SMART-RL **ریسک را حدود 40٪ کمتر می‌کند ولی Sharpe پایین‌تری دارد.**
  - عامل PPO به‌تنهایی بیش از حد محتاط است: Sharpe برابر 0.12 و فقط 39٪ از زمان در بازار.
- آزمایش تشخیصی با قاعده‌ی از پیش ثبت‌شده نشان داد قاعده‌ی انتخاب checkpoint علت این ضعف نیست. استفاده از مدل نهایی نتیجه را بدتر کرد (Sharpe برابر 0.41)، پس قاعده‌ی قفل‌شده حفظ می‌شود.
- **پیشنهاد:** پیکربندی را همین‌طور قفل کنیم («G4 تأیید») و فاز P5 اجرا شود.
- هر تغییر دیگری اکنون یعنی انتخاب بر اساس داده‌ی آزمونِ توسعه، و اعتبار نتایج را کم می‌کند.

## Details (English)

### What ran (offline pipeline, `results/logs/pipeline.log`)
| step | runtime |
| --- | --- |
| STM grid (7) | < 1 min |
| B5 grid (6) | < 1 min |
| PPO grid (12 configurations × 4 markets × 3 seeds = 144 runs) | 101.9 min |
| IMIT arm (3 × 4 × 3 = 36 runs) | 20.0 min |
| AWS grid (18) | < 1 min |
| selection and config freeze | — |
| pre-lock leakage tests (12 passed) | — |
| checkpoint diagnostic (12 runs, `scripts/p4_ckpt_diagnostic.py`) | — |

### Selection (`dev.selected.*`, written to configs/frozen.yaml; hash c1a5ebbd9f40…)

| component | selected | dev-val Sharpe |
| --- | --- | --- |
| PPO (`dev.selected.config`) | R2_l10_d1.0_lr1e-4 | 1.245 |
| STM | STM-1, κ = 0.5, z0 = 0.5 | 0.738 |
| AWS | H = 60, α = 5, β = 1.0 | 1.225 |
| B5 | κ = 0.5, z0 = 0.5 | 0.734 |
| IMIT | ϑ = 50 bp | 1.294 |

- **SENTIN optimiser** (E13 pilot): adam_bce.
- **`dev.n_trials`:** 50 (stm 7, b5 6, ppo 12, imit 3, aws 18, diagnostic 4).

### PPO grid, mean over markets and seeds 0-2 (`dev.ppo.<CFG>`)

| configuration | val Sharpe | val MDD | dev-test Sharpe |
| --- | --- | --- | --- |
| R2_l10_d1.0_lr1e-4 | 1.245 | -0.047 | 0.097 |
| R2_l50_d0.5_lr1e-4 | 1.206 | -0.047 | 0.127 |
| R3_lr1e-4 | 1.190 | -0.044 | 0.201 |
| R2_l10_d0.5_lr1e-4 | 1.169 | -0.045 | 0.177 |
| R2_l50_d1.0_lr1e-4 | 1.143 | -0.041 | 0.089 |
| R2_l10_d1.0_lr3e-4 | 1.118 | -0.039 | -0.083 |
| R3_lr3e-4 | 1.112 | -0.044 | 0.078 |
| R2_l50_d1.0_lr3e-4 | 1.067 | -0.048 | -0.056 |
| R1_lr1e-4 | 1.065 | -0.045 | -0.014 |
| R2_l10_d0.5_lr3e-4 | 0.940 | -0.048 | 0.034 |
| R2_l50_d0.5_lr3e-4 | 0.928 | -0.041 | 0.052 |
| R1_lr3e-4 | 0.833 | -0.056 | 0.175 |

The learning rate 1e-4 is better on validation for every reward. The rank correlation between validation and
dev-test Sharpe across configurations is weak. PBO will quantify this in P5.

### Development-test check of the full system (COMBINED; computed for this report, development data only)

| strategy | dev-val Sharpe | dev-test Sharpe | dev-test MDD | dev-test ann. return | mean exposure |
| --- | --- | --- | --- | --- | --- |
| Buy and hold | 1.032 | 0.721 | -0.118 | 0.088 | 1.00 |
| STM (selected) | 1.018 | 0.674 | -0.118 | 0.083 | 1.00 |
| PPO seed ensemble alone | 1.979 | 0.122 | -0.049 | 0.024 | 0.39 |
| SMART-RL (PPO + STM + AWS) | 1.852 | 0.588 | -0.074 | 0.059 | 0.71 |

Reading:
- The PPO agent trained with the risk-averse reward R2 holds cash much of the time. It wins in the volatile
  validation window but loses return in 2018-19.
- AWS moves weight back to the STM, which is almost always invested, and recovers most of the Sharpe while
  keeping a drawdown about 37 % smaller than buy and hold.
- Calmar on dev-test: SMART-RL 0.80 against 0.75 for buy and hold.

### Checkpoint-selection diagnostic (`dev.diagnostic.checkpoint`, rule fixed in advance in the script)

| variant | dev-val Sharpe | dev-test Sharpe | dev-test MDD | mean exposure |
| --- | --- | --- | --- | --- |
| PPO ensemble, best-validation checkpoint (frozen) | 1.979 | 0.122 | -0.049 | 0.39 |
| PPO ensemble, final policy | 1.195 | -0.102 | -0.044 | 0.31 |
| SMART-RL, best checkpoint (frozen) | 1.852 | 0.588 | -0.074 | 0.71 |
| SMART-RL, final policy | 1.431 | 0.405 | -0.064 | 0.72 |

- The rerun of the frozen rule reproduced the original numbers exactly (deterministic training).
- The final-policy rule lowers the dev-test Sharpe, so by the pre-stated rule the frozen checkpoint rule stays.
- The defensiveness comes from the reward, not from checkpoint selection.

### What the G4 lock means
- `locks/G4.lock` will store the SHA-256 of configs/frozen.yaml.
- From then on, no configuration value changes, and every policy is trained and evaluated once on the
  confirmation folds 2020-2024 and on the forward test.

**Registered additions:**
- E13x, done.
- S1, the 2020-2023 sub-period, ready in `smartrl/evaluation/analysis.py`.

### Expectation for P5 (stated before the lock, from development data only)
- SMART-RL behaves as a risk-reducing allocation: lower drawdown and volatility than buy and hold.
- Its Sharpe may be lower than buy and hold in strong bull years.
- H1 and H7 test Sharpe superiority. They may well not be supported against B1; the thesis text covers both
  outcomes.
- The comparison with the base paper as a trading signal (BASE) and with the other baselines is open.

### P5 compute plan
- **PPO runs:** about 660 runs (SMART-RL, B6, A1, A2 and A5 with 5 seeds × 5 folds; BC and IMIT with 3 seeds;
  forward fold) take about 11 h with 6 jobs on the CPU.
- **SAC (B7):** 72 runs take about 33 h on the GPU, in parallel.
- **Evaluation, statistics and outputs:** 2-3 h.
- **Wall time:** about 1.5 days.

### Decision needed
Write **"G4 تأیید"** to lock the configuration and start P5. No 2020+ trading result exists before that.
