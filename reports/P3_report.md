# گزارش فاز P3

خلاصه:
- **MarketMind (E2):** در هر سه افق، زیان پین‌بالش از LSTM، GRU، DLinear، Informer و حتی پیش‌بینی ساده بیشتر بود؛ آزمون DM با p کمتر از 0.001. فقط از PatchTST بهتر بود. بازه‌هایش بیش از حد مطمئن‌اند: پوشش 80٪ در افق 20 فقط 0.61 بود. برای نوسان، مدل ساده‌ی HAR بهترین بود. **فرضیه‌ی H4 پشتیبانی نشد.** این با یافته‌ی شناخته‌شده‌ی Zeng و همکاران (2023) سازگار است: ترنسفورمرها روی داده‌ی روزانه‌ی پرنویز مالی بیش‌برازش می‌کنند.
- **معیار ETT (E3):** ستون فقرات MarketMind در هر 8 حالت از Informer منتشرشده بهتر و از PatchTST ضعیف‌تر بود.
- **وظیفه‌ی مقاله‌ی مبنا (E13):** با پروتکل علّی همه‌ی مدل‌ها در حد شانس‌اند. میانگین دقت SenT-In برابر 0.50 و AUC آن 0.50 است، در برابر دقت 0.87 که مقاله گزارش کرده است.
- **تجزیه‌ی نشت (E13x، افزوده‌ی ثبت‌شده پیش از G4):** تقریباً همه‌ی پیش‌بینی‌پذیری ظاهری از پرکردن آخر هفته‌ها با قیمت جمعه می‌آید. تقسیم تصادفی، زمان‌بندی خبر و مقیاس‌دهی اثر ناچیزی دارند. حتی دادن خبر روز بعد به مدل دقت را بالا نبرد. در همه‌ی پروتکل‌ها دقت SenT-In دقیقاً با دقت «پیش‌بینی همیشه‌ی دسته‌ی غالب» هم‌خوان است.
- شاخص احساسات ما معتبر است: با بازده همان روز همبستگی 0.04 تا 0.13 دارد، ولی همبستگی‌اش با بازده روز بعد تقریباً صفر است.
- همه‌ی 12 آزمون نشت داده پاس شد.

## Details (English)

### What ran
Through the offline pipeline (`results/logs/pipeline.log`):

| step | runtime |
| --- | --- |
| MarketMind refits: MM and INFORMER (LSTM_PRICE and MM_NOSENT ran earlier) | 14.5 min |
| E13 pilot | 1.1 min |
| E13 main | 32.1 min |
| SENTIN refits | 7.2 min |
| H4 comparison (80 trainings) | — |
| E2 evaluation | — |

E13x ran separately with `scripts/p3_e13x.py`. E3 ran on 2026-10-18.

### E2 — MarketMind versus the comparison models (test years 2020-2024, seeds 0-2; `marketmind.*`)

Pinball loss × 1e-3, mean over seeds:

| h | MM | LSTM | GRU | INFORMER | PATCHTST | DLINEAR | NAIVE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.836 | 2.571 | 2.566 | 2.706 | 2.839 | 2.569 | 2.716 |
| 5 | 6.489 | 5.751 | 5.744 | 6.120 | 6.691 | 5.816 | 6.040 |
| 20 | 14.553 | 11.197 | 11.171 | 12.287 | 16.639 | 11.666 | 11.991 |

- **DM test, MM minus model, HLN-corrected:**
  - Every model except PatchTST has a significantly lower loss than MM at every horizon (positive stat, p < 0.001).
  - PatchTST is significantly worse than MM at h = 5 (p = 0.005) and h = 20 (p = 0.026).
- **Coverage of [q10, q90] at h = 20:** MM 0.61, PatchTST 0.59, LSTM/GRU 0.79, DLinear 0.76, NAIVE 0.85.
- **Seed dispersion of the h = 20 pinball (× 1e-3):** MM 1.71 and PatchTST 4.46, against 0.07 for LSTM and GRU.
  The patch models are unstable from seed to seed.
- **Volatility, QLIKE at h = 20:** HAR 0.419 < EWMA 0.545 < GRU 0.603 < LSTM 0.676 < DLinear 0.716 < Informer
  0.753 < MM 0.889 < PatchTST 0.973.
- **H4 (SPEC §12):** not supported. MM is not better than LSTM and GRU at h = 20; it is significantly worse.
- **Caveat for the thesis:** PT p-values at h = 20 are computed on overlapping 20-day returns. They are reported
  as specified but overstate significance.

### E3 — LTSF benchmark (`ltsf.*`)

Test MSE, mean over seeds:

| dataset | h = 96 | h = 192 | h = 336 | h = 720 |
| --- | --- | --- | --- | --- |
| ETTh1 | 0.447 | 0.533 | 0.594 | 0.610 |
| ETTm1 | 0.337 | 0.379 | 0.406 | 0.453 |

All 8 below published Informer and all 8 above published PatchTST. No RevIN (SPEC §15).

### E13 — the base paper's task (`e13.*`)
- **SENTIN optimiser** (`e13.sentin.choice`): adam_bce, by validation AUC 0.5485 against 0.5475.
- **Mean AUC over the four indices:**

  | model | PUB_R | PUB_C | CAUSAL |
  | --- | --- | --- | --- |
  | SVM | 0.588 | 0.516 | 0.495 |
  | CNN | 0.548 | 0.530 | 0.491 |
  | LSTM | 0.588 | 0.569 | 0.513 |
  | GRU | 0.642 | 0.597 | 0.515 |
  | SENTIN | 0.547 | 0.609 | 0.503 |
  | SENTIN_SF | 0.561 | 0.580 | 0.500 |
  | MM_DIR | — | — | 0.521 |

- **Up share of the test labels:** 0.36-0.39 under PUB, 0.53-0.57 under CAUSAL. Forward-filled non-trading days
  are never "up".

### E13x — leakage decomposition (`e13x.*`, figure F16)

SENTIN, mean over the four indices:

| step of the chain | protocol | accuracy | AUC |
| --- | --- | --- | --- |
| published, random split | PUB_R | 0.641 | 0.547 |
| chronological split | PUB_C | 0.628 | 0.609 |
| trading days, no forward-fill | X_TRD_UTC | 0.530 | 0.504 |
| session-aligned news | X_TRD_SES | 0.521 | 0.508 |
| training-row scaling | CAUSAL | 0.501 | 0.503 |

- X_TRD_RANDOM: accuracy 0.511, AUC 0.509.
- X_NEWS_LEAD: accuracy 0.520, AUC 0.490.
- In every protocol SENTIN's accuracy equals the majority-class rate (F16).
- The published 0.873 cannot be reached by any of the protocols. The calendar forward-fill alone explains the
  rise from chance to about 0.64.
- **Sentiment index check, 2010-2019:**
  - corr(S_loc_t, r_t): SP500 0.13, FTSE100 0.04, NIFTY50 0.11, SSE 0.10.
  - corr(S_loc_t, r_{t+1}): between −0.03 and 0.04.
  - So the index describes the day's move but carries no next-day information.

### Annual refits
- `forecast_MM`, `forecast_MM_NOSENT`, `forecast_INFORMER`, `forecast_LSTM_PRICE` and `forecast_SENTIN`.
- Each covers 2010-01-04 .. 2026-09-30 from 16 refits.
- Leakage test 5 checks that each predicted year uses a model trained on data up to the end of the previous year.

**Leakage tests:** `meta.leakage_by_gate.G3` = true (12 passed).

### Outputs
- Tables T05, T06, T20.
- Figures F05, F06, F15 and F16 (new).
