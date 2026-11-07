# گزارش فاز P6 (دروازه‌ی G6)

خلاصه:
- E14 (وظیفه‌ی خود مقاله‌ی مبنا با شرایط دقیق آن) و E15 (معامله در پنجره‌ی آزمون مقاله) کامل اجرا شدند: 461 اجرای مدل در E14 و 52 سیاست معاملاتی در E15. همه‌ی آزمون‌های از پیش ثبت‌شده انجام شدند. آزمون‌های نشت داده پاس شدند، `frozen.yaml` و `G4.lock` دست نخوردند و هیچ کلید قبلی `results.json` تغییر نکرد.
- **نتیجه‌ی اصلی (شرایط دقیق مقاله، معیارهای خود مقاله):** مدل پیش‌بینی ما (MarketMind با ویژگی‌های SentiFlow) در AUC-ROC و PR-AUC به‌طور معنادار از SenT-In بهتر است، در هر دو خوانش تقسیم 80:20. AUC-ROC در خوانش زمانی 0.712 در برابر 0.511 و در خوانش تصادفی 0.718 در برابر 0.506 است (p با تصحیح Holm کمتر از 0.001). PR-AUC در خوانش زمانی 0.875 در برابر 0.698 است. این برتری در برابر نسخه‌ی قوی‌تر SenT-In (دستور E13) هم برقرار است.
- در Accuracy و F1 مدل ما کمی پایین‌تر است (دقت در خوانش زمانی 0.667 در برابر 0.688). دلیلش این است که SenT-In با دستور آموزش مقاله برای همه‌ی ردیف‌ها «بالا» پیش‌بینی می‌کند و در این پروتکل 67 تا 70 درصد ردیف‌های آزمون «بالا» هستند. با برچسب سخت‌گیرانه‌ی GT، مدل ما در هر چهار معیار بهتر است.
- بررسی اعتبار V1 (از پیش ثبت‌شده) نشان می‌دهد این برتری از ردیف‌هایی می‌آید که برچسبشان با پرکردن روزهای تعطیل ثابت شده است. روی ردیف‌هایی که بعدشان یک جلسه‌ی واقعی معاملاتی است، همه‌ی مدل‌ها، از جمله SenT-In، در حد شانس‌اند. طبق قاعده‌ی ادعا، این جمله باید در پایان‌نامه بیاید.
- SentiFlow داخل SenT-In (با دستور مقاله) بهبود معناداری نمی‌دهد.
- دقت منتشرشده‌ی SenT-In در مقاله (0.816 تا 0.958) در شرایط اعلام‌شده با هیچ مدلی به دست نیامد. حتی وقتی مدل اطلاعات روز هدف را می‌بیند (آزمایش تشخیصی LA)، بیشترین دقت 0.790 است.
- **E15** (2018-10-18 تا 2022-03-31): Sharpe سبد ترکیبی SMART-RL 0.638 است، در برابر 0.467 برای SenT-In با دستور E13 و 0.750 برای SenT-In با دستور مقاله. نسخه‌ی دوم همیشه در بازار است و همان خرید و نگهداری است. هیچ‌کدام از این دو اختلاف معنادار نیست. بیشینه‌ی افت سرمایه‌ی SMART-RL (-19.3 %) به‌طور معنادار از خرید و نگهداری (-29.7 %) کمتر است.
- **تصمیم لازم:** برای تأیید G6 بنویسید «ادامه». برای E16 (اختیاری)، کد TDQN را دانلود کنید و بنویسید «E16 تأیید»، یا بنویسید که E16 اجرا نشود.

## Details (English)

### 1. Headline: the proposed predictive model against SenT-In under the base paper's own conditions

Setting (protocol §2.1–2.4): the paper's window and row count (2005-01-01 → 2022-03-31, 6298 labelled calendar rows per index, non-trading days filled with the previous day's values), the paper's 80:20 split read chronologically (`PUB_C`: test 2018-10-18 → 2022-03-30) and randomly (`PUB_R`, seed 2026), 5038 training and 1260 test rows per index, label GE = 1[C(t+1) ≥ C(t)], and the paper's four metrics. The base model is SenT-In rebuilt from the paper (awareness layer, GRU of eqs 7–11, Informer) and trained with its Table 3 recipe (`SENTIN_PAPER`). Ours is the MarketMind classifier with the SentiFlow features (`MM_CLS`). Mean over the four indices (seed mean; `e14.<SET>.MEAN.<MODEL>.*`):

| model | PUB_C Acc | PUB_C F1 | PUB_C AUC-ROC | PUB_C PR-AUC | PUB_R Acc | PUB_R F1 | PUB_R AUC-ROC | PUB_R PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM (paper recipe) | 0.688 | 0.815 | 0.529 | 0.713 | 0.685 | 0.813 | 0.569 | 0.741 |
| CNN (paper recipe) | 0.656 | 0.780 | 0.583 | 0.761 | 0.665 | 0.788 | 0.601 | 0.763 |
| LSTM (paper recipe) | 0.688 | 0.815 | 0.524 | 0.710 | 0.683 | 0.812 | 0.526 | 0.706 |
| GRU (paper recipe) | 0.680 | 0.806 | 0.530 | 0.711 | 0.680 | 0.808 | 0.553 | 0.731 |
| SenT-In, paper recipe | 0.688 | 0.815 | 0.511 | 0.698 | 0.685 | 0.813 | 0.506 | 0.692 |
| SenT-In, E13 recipe | 0.687 | 0.813 | 0.563 | 0.740 | 0.683 | 0.810 | 0.564 | 0.741 |
| SenT-In + SentiFlow, paper recipe | 0.688 | 0.815 | 0.513 | 0.698 | 0.685 | 0.813 | 0.506 | 0.693 |
| SenT-In + SentiFlow, E13 recipe | 0.681 | 0.803 | 0.575 | 0.749 | 0.682 | 0.808 | 0.553 | 0.734 |
| MarketMind classifier (proposed) | 0.667 | 0.769 | 0.712 | 0.875 | 0.670 | 0.777 | 0.718 | 0.876 |
| MarketMind q50 direction | 0.476 | 0.438 | 0.428 | 0.653 | 0.521 | 0.618 | 0.477 | 0.673 |

Primary pre-registered tests (protocol §2.7: one-sided H1 ours > SenT-In, statistic = mean over the four indices of the AUC-ROC difference on seed-averaged scores, paired stationary block bootstrap stratified by index, 10 000 resamples, Holm over the four tests; `e14.tests.<SET>.<CMP>.*`):

| setting | comparison | AUC-ROC diff | 95 % interval | p | Holm p | DeLong p per index (Holm) |
| --- | --- | --- | --- | --- | --- | --- |
| PUB_C_GE | MM_CLS vs SENTIN_PAPER | +0.194 | [+0.173, +0.216] | < 0.001 | < 0.001 | < 0.001 in all four |
| PUB_C_GE | SENTIN_SF_PAPER vs SENTIN_PAPER | -0.001 | [-0.005, +0.002] | 0.776 | 1.000 | 0.646 – 0.927 |
| PUB_R_GE | MM_CLS vs SENTIN_PAPER | +0.203 | [+0.183, +0.224] | < 0.001 | < 0.001 | < 0.001 in all four |
| PUB_R_GE | SENTIN_SF_PAPER vs SENTIN_PAPER | -0.001 | [-0.003, +0.002] | 0.769 | 1.000 | 0.686 – 0.973 |

Secondary metrics of the same comparisons (difference ours − SenT-In; PR-AUC and F1: bootstrap, one-sided; accuracy: exact McNemar on the 0.5-threshold predictions pooled over the indices, two-sided; Holm within each metric):

| setting | ours | PR-AUC diff | Accuracy diff | F1 diff |
| --- | --- | --- | --- | --- |
| PUB_C_GE | MM_CLS | +0.171 (p < 0.001, Holm < 0.001) | -0.016 (McNemar p 0.014, Holm 0.042) | -0.039 (p 1.000, Holm 1.000) |
| PUB_C_GE | SENTIN_SF_PAPER | -0.003 (p 0.973, Holm 0.973) | +0.000 (McNemar p 1.000, Holm 1.000) | +0.000 (p 1.000, Holm 1.000) |
| PUB_R_GE | MM_CLS | +0.178 (p < 0.001, Holm < 0.001) | -0.017 (McNemar p 0.009, Holm 0.036) | -0.037 (p 1.000, Holm 1.000) |
| PUB_R_GE | SENTIN_SF_PAPER | +0.000 (p 0.426, Holm 0.853) | +0.000 (McNemar p 1.000, Holm 1.000) | +0.000 (p 1.000, Holm 1.000) |

Reading: `MM_CLS` ranks the days far better than SenT-In on both threshold-free metrics, in both readings and in every index. On accuracy and F1 it is slightly lower: the paper-recipe SenT-In has exactly the accuracy of the always-up rule in every index (it predicts "up" on every test row), and under GE most test rows are up (positive share 0.673–0.699 in `PUB_C`), so a constant "up" is hard to beat on threshold metrics. Under the strict label GT (V3 below) `MM_CLS` is better on all four metrics.

**Claim sentences produced by the pre-registered claim rules (verbatim, `e14.claims.*`):**

> MM_CLS is significantly better than SenT-In under the paper's conditions; in PUB_C_GE the advantage comes from rows whose label is fixed by the filling (V1 p 0.810); in PUB_C_GE it holds also against the stronger reproduction of SenT-In (V2 p 0.000); in PUB_R_GE the advantage comes from rows whose label is fixed by the filling (V1 p 0.748); in PUB_R_GE it holds also against the stronger reproduction of SenT-In (V2 p 0.000).

> SENTIN_SF_PAPER is not significantly better than SenT-In under the paper's conditions (Holm-adjusted one-sided p 1.000 in PUB_C_GE and 1.000 in PUB_R_GE).

"p 0.000" in the generated sentence is the bootstrap floor 1/10 001 printed with three decimals (reported as p < 0.001 elsewhere).

### 2. The paper's own layout (T22) with the published SenT-In numbers

Per index in the paper's order (`results/tables/T22.csv` has every model). The row "SenT-In, published" is quoted from Chauhan et al. (2025), Tables 4–7 (printed pp. 15–16); it is not a result of this project.

**chronological reading `PUB_C_GE`**

| index | model | Accuracy | F1-Score | AUC-ROC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| S&P 500 | SenT-In, published | 0.816 | 0.817 | 0.933 | 0.947 |
|  | SenT-In, paper recipe | 0.697 | 0.821 | 0.509 | 0.705 |
|  | SenT-In, E13 recipe | 0.697 | 0.821 | 0.638 | 0.821 |
|  | MarketMind classifier (proposed) | 0.661 | 0.770 | 0.689 | 0.868 |
| FTSE 100 | SenT-In, published | 0.958 | 0.952 | 0.997 | 0.996 |
|  | SenT-In, paper recipe | 0.673 | 0.805 | 0.517 | 0.687 |
|  | SenT-In, E13 recipe | 0.673 | 0.805 | 0.494 | 0.670 |
|  | MarketMind classifier (proposed) | 0.657 | 0.748 | 0.729 | 0.877 |
| SSE Composite | SenT-In, published | 0.877 | 0.885 | 0.948 | 0.957 |
|  | SenT-In, paper recipe | 0.684 | 0.812 | 0.512 | 0.696 |
|  | SenT-In, E13 recipe | 0.679 | 0.804 | 0.592 | 0.752 |
|  | MarketMind classifier (proposed) | 0.681 | 0.777 | 0.733 | 0.882 |
| Nifty 50 | SenT-In, published | 0.841 | 0.846 | 0.921 | 0.931 |
|  | SenT-In, paper recipe | 0.699 | 0.823 | 0.508 | 0.704 |
|  | SenT-In, E13 recipe | 0.699 | 0.823 | 0.530 | 0.717 |
|  | MarketMind classifier (proposed) | 0.670 | 0.781 | 0.698 | 0.873 |

**random reading `PUB_R_GE`**

| index | model | Accuracy | F1-Score | AUC-ROC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| S&P 500 | SenT-In, published | 0.816 | 0.817 | 0.933 | 0.947 |
|  | SenT-In, paper recipe | 0.689 | 0.816 | 0.508 | 0.692 |
|  | SenT-In, E13 recipe | 0.681 | 0.805 | 0.708 | 0.871 |
|  | MarketMind classifier (proposed) | 0.669 | 0.779 | 0.711 | 0.876 |
| FTSE 100 | SenT-In, published | 0.958 | 0.952 | 0.997 | 0.996 |
|  | SenT-In, paper recipe | 0.665 | 0.799 | 0.506 | 0.675 |
|  | SenT-In, E13 recipe | 0.665 | 0.799 | 0.480 | 0.651 |
|  | MarketMind classifier (proposed) | 0.643 | 0.748 | 0.709 | 0.865 |
| SSE Composite | SenT-In, published | 0.877 | 0.885 | 0.948 | 0.957 |
|  | SenT-In, paper recipe | 0.700 | 0.824 | 0.503 | 0.708 |
|  | SenT-In, E13 recipe | 0.701 | 0.822 | 0.578 | 0.766 |
|  | MarketMind classifier (proposed) | 0.685 | 0.788 | 0.731 | 0.887 |
| Nifty 50 | SenT-In, published | 0.841 | 0.846 | 0.921 | 0.931 |
|  | SenT-In, paper recipe | 0.684 | 0.812 | 0.507 | 0.695 |
|  | SenT-In, E13 recipe | 0.684 | 0.812 | 0.489 | 0.676 |
|  | MarketMind classifier (proposed) | 0.684 | 0.792 | 0.721 | 0.876 |

The only published SenT-In value that any model reaches under the stated conditions is the S&P 500 F1-score in `PUB_C_GE` (reached by `SVM_P`, `LSTM_P`, `SENTIN_PAPER`, `SENTIN_E13`, `SENTIN_SF_PAPER`). These models predict "up" on (almost) every row, and the always-up rule itself scores 0.821 (S&P 500, `PUB_C_GE`), so a published F1 below that level says nothing about the model. Every published accuracy, AUC-ROC and PR-AUC is above every model in every index. The best test accuracy in `PUB_C_GE` is 0.697 (S&P 500), 0.673 (FTSE 100), 0.685 (SSE Composite), 0.699 (Nifty 50), against the published 0.816, 0.958, 0.877, 0.841. Even the look-ahead diagnostic `LA` (the target day's open, high, low, volume and sentiment are visible, its close is not; §4) reaches at most 0.790 (`MM_CLS`, S&P 500, `PUB_C_GE_LA`). The published numbers can therefore not be reproduced from the paper's description; the defensible comparison is the head-to-head under identical conditions of §1, which is what the pre-registered tests measure.

### 3. Validity checks (pre-registered; they set the wording of the claim)

| check | setting | comparison | AUC-ROC diff | 95 % interval | one-sided p |
| --- | --- | --- | --- | --- | --- |
| V1 informative rows | PUB_C_GE | MM_CLS vs SENTIN_PAPER | -0.014 | [-0.042, +0.016] | 0.810 |
| V1 informative rows | PUB_C_GE | SENTIN_SF_PAPER vs SENTIN_PAPER | +0.000 | [-0.003, +0.005] | 0.424 |
| V1 informative rows | PUB_R_GE | MM_CLS vs SENTIN_PAPER | -0.010 | [-0.038, +0.019] | 0.748 |
| V1 informative rows | PUB_R_GE | SENTIN_SF_PAPER vs SENTIN_PAPER | -0.001 | [-0.004, +0.002] | 0.728 |
| V2 stronger base | PUB_C_GE | MM_CLS vs SENTIN_E13 | +0.144 | [+0.125, +0.164] | < 0.001 |
| V2 stronger base | PUB_C_GE | SENTIN_SF_E13 vs SENTIN_E13 | +0.013 | [-0.002, +0.028] | 0.043 |
| V2 stronger base | PUB_R_GE | MM_CLS vs SENTIN_E13 | +0.136 | [+0.117, +0.154] | < 0.001 |
| V2 stronger base | PUB_R_GE | SENTIN_SF_E13 vs SENTIN_E13 | -0.031 | [-0.048, -0.014] | 1.000 |
| V3 label GT | PUB_C_GT | MM_CLS vs SENTIN_PAPER | +0.242 | [+0.222, +0.260] | < 0.001 |
| V3 label GT | PUB_C_GT | SENTIN_SF_PAPER vs SENTIN_PAPER | +0.006 | [-0.001, +0.014] | 0.055 |
| V3 label GT | PUB_R_GT | MM_CLS vs SENTIN_PAPER | +0.236 | [+0.214, +0.258] | < 0.001 |
| V3 label GT | PUB_R_GT | SENTIN_SF_PAPER vs SENTIN_PAPER | +0.005 | [+0.001, +0.010] | 0.021 |
| V4 CAUSAL | CAUSAL | MM_CLS vs SENTIN_PAPER | -0.026 | [-0.053, +0.003] | 0.967 |
| V4 CAUSAL | CAUSAL | SENTIN_SF_PAPER vs SENTIN_PAPER | -0.001 | [-0.009, +0.006] | 0.664 |
| V4 CAUSAL | CAUSAL | MM_CLS vs SENTIN_E13 | -0.018 | [-0.045, +0.009] | 0.913 |
| V4 CAUSAL | CAUSAL | SENTIN_SF_E13 vs SENTIN_E13 | -0.023 | [-0.044, -0.003] | 0.986 |

- **V1.** Informative rows are the test rows whose next calendar date is a session (0.664–0.692 of the `PUB_C_GE` test rows). On them every model is near chance (mean AUC-ROC `MM_CLS` 0.505, `SENTIN_PAPER` 0.514). The other rows precede a filled date, so C(t+1) = C(t) and their GE label is always 1. `MM_CLS` recognises these rows from the calendar pattern of the inputs and gives them its highest scores; SenT-In (paper recipe) gives every row the same score. The AUC advantage therefore comes from the rows whose label is fixed by the filling, as the claim rule states.
- **V2.** Against our stronger reproduction of SenT-In (E13 recipe, `SENTIN_E13`) `MM_CLS` is also better (+0.144 and +0.136, p < 0.001). SentiFlow in the E13 recipe (`SENTIN_SF_E13` vs `SENTIN_E13`) changes AUC-ROC by +0.013 in `PUB_C_GE` (p 0.043) and by -0.031 in `PUB_R_GE` (p 1.000); no claim rests on this comparison.
- **V3.** With E13's strict label GT (positive share 0.348–0.386) `MM_CLS` is better on all four metrics: accuracy +0.019 / +0.023 (McNemar p 0.020 / 0.007), F1 0.518 vs 0.000 (the paper-recipe SenT-In predicts "down" on every row), PR-AUC +0.167 / +0.156. The forced rows are now always 0, and `MM_CLS` again separates them.
- **V4.** Under the causal protocol of this thesis (trading days only, session-aligned news, scaling fitted on the training rows) every model is near chance (mean AUC-ROC 0.484–0.507) and no comparison favours ours. This repeats the E13 and E13x findings: with prices and headlines known at the close, no model calls the next-day direction of these indices better than chance.

### 4. Diagnostics (descriptive; never used for a claim)

**Label evidence** (no training; `e14.labels.*`): the always-up rule on the test rows.

| index | GE PUB_C pos share (= always-up Acc) | always-up F1 | always-up PR-AUC | GE PUB_R pos share | GT PUB_C pos share | GT PUB_R pos share |
| --- | --- | --- | --- | --- | --- | --- |
| S&P 500 | 0.697 | 0.821 | 0.697 | 0.689 | 0.386 | 0.379 |
| FTSE 100 | 0.673 | 0.805 | 0.673 | 0.665 | 0.365 | 0.354 |
| SSE Composite | 0.684 | 0.812 | 0.684 | 0.700 | 0.348 | 0.360 |
| Nifty 50 | 0.699 | 0.823 | 0.699 | 0.684 | 0.373 | 0.355 |

The paper's weakest models have F1 and PR-AUC above their accuracy (protocol §1, item 3), which needs an up-majority; GE gives one, GT does not. This is why GE is the primary label.

**Look-ahead `LA`** (accuracy; `e14.PUB_*_GE_LA.*`, figure F19):

| index | SenT-In published | best PUB_C_GE | best PUB_C_GE_LA | best PUB_R_GE | best PUB_R_GE_LA | MM_CLS PUB_C: no LA → LA |
| --- | --- | --- | --- | --- | --- | --- |
| S&P 500 | 0.816 | 0.697 (SENTIN_PAPER) | 0.790 (MM_CLS) | 0.689 (SVM_P) | 0.789 (MM_CLS) | 0.661 → 0.790 |
| FTSE 100 | 0.958 | 0.673 (SVM_P) | 0.771 (GRU_P) | 0.665 (SVM_P) | 0.761 (GRU_P) | 0.657 → 0.738 |
| SSE Composite | 0.877 | 0.685 (SENTIN_SF_E13) | 0.695 (MM_CLS) | 0.701 (SENTIN_E13) | 0.720 (MM_CLS) | 0.681 → 0.695 |
| Nifty 50 | 0.841 | 0.699 (SVM_P) | 0.761 (MM_CLS) | 0.684 (SVM_P) | 0.752 (MM_CLS) | 0.670 → 0.761 |

**`ROBUST`** (median/IQR scaling instead of q01/q99, `PUB_C_GE`): `MM_CLS` AUC-ROC 0.709 (q01/q99: 0.712), `SENTIN_PAPER` 0.506 (0.511). The scaling choice does not change the picture.

**`MM_DIR`** (frozen MarketMind q50 at h = 1): mean AUC-ROC 0.428 / 0.477 under GE and 0.588 / 0.576 under GT. Its median forecast is lower on average on the rows before a filled date (their next-row return is exactly 0) than on the other rows; those rows are positive under GE and negative under GT, which moves its AUC below and above 0.5.

### 5. E14 data

| index | calendar rows | labelled rows | trading rows | zero-change sessions (GE ≠ GT) | local headlines 2005 | local headlines 2006 | local headlines 2005–2022 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S&P 500 | 6299 | 6298 | 4342 | 2 | 9,519 | 9,987 | 10,809,459 |
| FTSE 100 | 6299 | 6298 | 4358 | 6 | 281 | 455 | 283,819 |
| SSE Composite | 6299 | 6298 | 4186 | 0 | 699 | 673 | 424,648 |
| Nifty 50 | 6299 | 6298 | 4237 | 4 | 140 | 264 | 174,316 |

- Nifty 50 before 2007-09-17 comes from NSE Indices (niftyindices.com) and from Yahoo afterwards; the largest relative close difference on the first 20 common days is 4.3e-08 (limit 0.1 %). Nifty volume is masked.
- The 2005–2006 news (77,878 headlines kept by the P1 rules) was extracted with the P1 code and scored with the frozen models. Local coverage in 2005–2006 is thin outside the S&P 500, so many early training rows of the other indices have sentiment index 0 (SPEC §9.1). The paper's own news (investing.com, crawled every 3 minutes) is not public and cannot be matched (protocol §1, item 11).
- `A_PAPER`, the paper's GRU→CNN sentiment classifier with its Table 2 recipe (400 epochs, no early stopping), mean over seeds 0–2 on the SPEC §2.4 test splits (`e14.a_paper.*`):

| test split | accuracy | macro-F1 |
| --- | --- | --- |
| phrasebank50 | 0.717 | 0.644 |
| phrasebank_all | 0.829 | 0.782 |
| fiqa | 0.529 | 0.491 |
| twitter | 0.783 | 0.694 |
| pooled | 0.740 | 0.671 |

  On the same pooled split, E13's Branch A reaches 0.784 / 0.752 and SentiFlow 0.841 / 0.814 (E1, accuracy / macro-F1): the paper recipe (400 epochs, no early stopping) scores below both.

### 6. E15: trading in the base paper's test window

One split per market (protocol §3.1): train 2010-01-01 → 2017-10-17, validation 2017-10-18 → 2018-10-17, one test run 2018-10-18 → 2022-03-31 (900 distinct trading dates over the four markets), no refits inside it. SMART-RL runs with `configs/frozen.yaml` unchanged; that configuration was selected on development validation data ending 2018-06 (SPEC §1), before this window starts. Costs 10 bps. COMBINED portfolio (`e15.<full|sub2020|cost20>.COMBINED.*`):

| method | Sharpe | ann. return | max drawdown | cum. return | turnover | Sharpe at 20 bps | Sharpe 2020-01 → 2022-03 | max DD 2020-01 → 2022-03 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SMART-RL (proposed) | 0.638 | 7.9 % | -19.3 % | 31.2 % | 0.025 | 0.606 | 0.558 | -19.3 % |
| BASE: SenT-In, E13 recipe | 0.467 | 6.3 % | -21.4 % | 24.3 % | 0.036 | 0.379 | 0.243 | -21.4 % |
| BASE_PAPER: SenT-In, paper recipe | 0.750 | 12.4 % | -29.7 % | 52.0 % | 0.000 | 0.750 | 0.628 | -29.7 % |
| B1 Buy & hold | 0.750 | 12.4 % | -29.7 % | 52.0 % | 0.000 | 0.750 | 0.628 | -29.7 % |
| B2 SMA 50/200 | -0.053 | 0.9 % | -25.1 % | 3.2 % | 0.007 | -0.069 | 0.001 | -25.1 % |
| B3 Momentum 12-1 | 0.087 | 2.3 % | -27.7 % | 8.6 % | 0.035 | 0.014 | 0.027 | -27.7 % |
| B4 LSTM signal | 0.767 | 12.5 % | -29.7 % | 52.5 % | 0.014 | 0.743 | 0.612 | -29.7 % |
| B5 FinBERT signal | 0.840 | 11.8 % | -20.2 % | 48.8 % | 0.007 | 0.826 | 0.717 | -20.2 % |
| B6 PPO price-only | 0.204 | 3.1 % | -12.6 % | 11.3 % | 0.077 | -0.120 | 0.128 | -12.6 % |
| B7 SAC | 0.123 | 2.6 % | -16.8 % | 9.7 % | 0.109 | -0.236 | 0.057 | -16.8 % |
| B8 Static 50/50 | 0.723 | 8.6 % | -18.1 % | 34.1 % | 0.026 | 0.655 | 0.630 | -18.1 % |

| comparison (COMBINED) | Sharpe diff | p (LW HAC, one-sided) | p (bootstrap) | max-drawdown diff [95 %] | cum.-return diff [95 %] |
| --- | --- | --- | --- | --- | --- |
| SMART-RL − BASE | +0.170 | 0.197 | 0.201 | +0.021 [-0.022, +0.064] | +0.068 [-0.078, +0.215] |
| SMART-RL − BASE_PAPER | -0.113 | 0.786 | 0.689 | +0.104 [+0.014, +0.129] | -0.209 [-0.712, +0.052] |
| SMART-RL − B1_BH | -0.113 | 0.786 | 0.689 | +0.104 [+0.014, +0.129] | -0.209 [-0.712, +0.052] |

- `BASE_PAPER` (SenT-In, paper recipe) is invested on every day, so its returns equal buy-and-hold; in figure F18 its curve lies under the buy-and-hold curve. A positive drawdown difference means a smaller drawdown for SMART-RL; against buy-and-hold its 95 % interval excludes 0.
- Per market (T23), SMART-RL's drawdown against buy-and-hold: S&P 500 -17.7 % vs -33.8 %, FTSE 100 -27.3 % vs -34.3 %, SSE Composite -17.9 % vs -18.7 %, Nifty 50 -25.9 % vs -38.3 % (smaller in every index).
- `BASE` holds no position on the S&P 500 from 2020-01-01 on, so its Sharpe ratio there is undefined (`e15.sub2020.SP500.BASE.sharpe` = null; max drawdown 0.0 %).
- The highest COMBINED Sharpe ratio in this window is 0.840 (B5 FinBERT signal).

Decisions as next-day direction calls (prediction 1[e(t) > 0.5], score e(t), label 1[C(t+1) > C(t)] on trading days; mean over the four markets, `e15.dir.MEAN.*`):

| method | Accuracy | F1 | AUC-ROC | PR-AUC | up share |
| --- | --- | --- | --- | --- | --- |
| SMART-RL (proposed) | 0.533 | 0.682 | 0.516 | 0.557 | 0.542 |
| BASE: SenT-In, E13 recipe | 0.509 | 0.533 | 0.498 | 0.540 | 0.542 |
| BASE_PAPER: SenT-In, paper recipe | 0.542 | 0.702 | 0.500 | 0.542 | 0.542 |

- Overlap with P5 without retraining (`e15.p5overlap`, 2020-01-01 → 2022-03-31 from the P5 annual-refit daily returns): Sharpe SMART-RL 0.705, BASE 0.868, buy-and-hold 0.628; max drawdown -17.6 %, -19.0 %, -29.7 %. The single-split E15 run and the annually refitted P5 run differ for BASE (one SenT-In fit before 2018-10-18 against yearly refits).

### 7. What ran

| step | wall clock |
| --- | --- |
| E14 data: Nifty splice, 2005–2006 news extraction and scoring, `A_PAPER` (3 × 400 epochs), scoring of 2005–2022 headlines, daily indices | 0.74 h (run records) |
| E15 forecasts: one MarketMind and one LSTM_PRICE fit before 2018-10-18; BASE and BASE_PAPER fits | 2.8 min |
| E14 run matrix: 461 model runs (GPU, 3 workers) | 776.8 min |
| E14 evaluation and tests (tables and figures re-run by hand after the F17 fix, §9) | 3.2 min |
| E15 PPO-family policies: SMART-RL 20, B6 20 (CPU, 6 workers) | 0.51 h |
| E15 SAC policies: B7 12 (GPU, 6 workers) | 251.5 min |
| E15 evaluation, tables and figure | 0.8 min |

Run records (`results/runs/P6_*/meta.json`): E14 474 (39.3 GPU h, 0.1 CPU h summed over parallel jobs), E15 14 (4.3 GPU h, 0.5 CPU h). 3 of the E14 records are 2-epoch smoke runs of `A_PAPER`, not used for any result. Unattended pipeline: `scripts/run_p6_pipeline.py` (log `results/logs/p6/pipeline.log`).

Code: `configs/p6.yaml`; `smartrl/p6/` (`cfg.py`, `prices.py`, `news.py`, `a_paper.py`, `models.py`, `e14data.py`, `e14run.py`, `e14eval.py`, `stats.py`, `e15.py`, `e15eval.py`); `scripts/p6_fetch_nifty.py`, `scripts/p6_e14.py`, `scripts/p6_e15.py`, `scripts/run_p6_pipeline.py`, `scripts/p6_report.py`; `smartrl/data/news.py` gained optional arguments (defaults unchanged). Outputs: `results/tables/T21.csv` … `T25.csv`, `results/figures/F17` … `F19` (.png, .pdf).

### 8. Interpretations fixed before the runs (`configs/p6.yaml`, `interpretations`)

- `mm_cls_frozen`: "MarketMind encoder, tokens and sentiment gate frozen (SPEC §4.1-4.3)" is read as the frozen SPEC design: the model is trained from scratch on each protocol's training rows. Weights of earlier MarketMind fits are not reused, because they were trained on dates inside the E14 test periods.
- `mm_on_calendar_rows`: Under PUB_C and PUB_R the MarketMind price features (SPEC §2.2) and SentiFlow features (§3.9) are computed on the forward-filled calendar rows (the paper's row grid); under CAUSAL on each market's trading days.
- `mm_dir_targets`: MM_DIR trains its quantile heads on training rows whose 20-row target is known; under PUB_C the target window must end inside the training rows (as in the annual refits).
- `sentin_paper_gru`: The GRU (eqs 7-11) runs over the weighted encoder inputs; its hidden states replace the Informer value embedding of the encoder tokens. The decoder tokens are the last 48 GRU states plus one GRU step on the zero placeholder.
- `a_paper_padding`: Headlines are padded to 48 tokens; GRU outputs at padded positions are zeroed before the convolution (as in Branch A). The convolution has no padding (48 -> 46), MaxPool1d(2) -> 23, flatten 23 x 64.
- `rnn_dropout`: dropout 0.1 between the two recurrent layers and on the last hidden state.
- `robust_scaling`: (x - median) / IQR fitted on the same rows as the q01/q99 scaling, no clipping.
- `la_ghost_row`: LA windows end at row t+1 with close = C_t and the open, high, low, volume and sentiment of t+1. For MM_CLS the price features of that row are recomputed from the modified close (r_k, rv, range, rsi, macd, gaps, dd252, vchg).

### 9. Deviations, the bug fix and problems

- **Method:** no deviation from the protocol. The choices the protocol leaves open are listed in §8; the unmatched items of protocol §1 (the paper's news source, the training data of its sentiment classifier, look-back, Informer sizes, SVM kernel, threshold) are handled as the protocol prescribes.
- **Bug fix (figure F17 only, protocol §0 rule 4).** The first `e14_outputs` step stopped: F17 drew the mean of the per-seed AUCs (`e14.<SET>.<MKT>.<MODEL>.auc`) with a bootstrap interval of the AUC of the seed-averaged scores (`e14.ci.*.auc_lo/auc_hi`), two different estimators, so some points fell outside their intervals and matplotlib refused the negative error bars. F17 now draws the AUC of the seed-averaged scores (new keys `e14.ci.<SET>.<MKT>.<MODEL>.auc`), the estimator the interval and the tests use. No model was re-run and no stored number changed; old and new points are both kept in results.json (example, `MM_CLS` in `PUB_C_GE`, old → new: S&P 500 0.689 → 0.686; FTSE 100 0.729 → 0.735; SSE Composite 0.733 → 0.738; Nifty 50 0.698 → 0.698). The pipeline was changed so that a failing table/figure step no longer stops the training steps.
- **Undefined values:** `e15.sub2020.SP500.BASE.<sharpe|sortino|calmar|win_rate>` are null because BASE has no position on the S&P 500 in that sub-window (zero volatility).
- **Limitations:** (1) the paper's news database is not public, so all models use the thesis corpus extended to 2005; (2) the published SenT-In numbers could not be reproduced by any model (§2), so the comparison with the paper rests on the faithful re-implementation; (3) the AUC advantage under the paper's protocol comes from the calendar filling (V1); under the causal protocol no model predicts the next-day direction (V4).

### 10. Integrity checks

- `configs/frozen.yaml` SHA-256 c1a5ebbd9f40… equals `locks/G4.lock`.
- `configs/p6.yaml` hash 703366973a1d…: the same hash is stored in all 488 P6 run records (the settings never changed during P6).
- `results/results.json` against the pre-registration commit aaf78ff: 7482 existing keys, 0 missing, 0 changed (besides `meta.created_utc`; `notes` only gained entries: 19 → 32, old entries unchanged); 5649 new keys, 0 outside `e14.*` / `e15.*`.
- Leakage tests at G6 (`tests/test_leakage.py`): 11 passed, 1 skipped (the skipped test only applies before the G4 lock). Contract test (`tests/test_results_contract.py`): 3 passed. Logs: `results/logs/p6/g6_leakage.log`, `g6_contract.log`.

### 11. Wording for the thesis (follows the claim rules; numbers from the keys above)

> Under the base paper's own conditions (2005-01-01 → 2022-03-31, 6298 calendar rows per index with filled non-trading days, 80:20 split) and judged by the paper's own metrics, the proposed predictive model (MarketMind with SentiFlow features) is significantly better than SenT-In on AUC-ROC (+0.194 in the chronological and +0.203 in the random reading; Holm-adjusted p < 0.001) and PR-AUC (+0.171, +0.178), in every index, and also against a stronger reproduction of SenT-In. On accuracy and F1 the paper-recipe SenT-In is marginally higher because it predicts "up" on every row; with the strict label 1[C(t+1) > C(t)] the proposed model is better on all four metrics. A pre-registered check shows that the advantage comes from rows whose label is fixed by the filling of non-trading days: on rows followed by a trading session all models, SenT-In included, are at chance level. The published SenT-In accuracies could not be reached by any model under the stated conditions, even with the target day's open, high, low and volume visible (best 0.790).

> In the paper's test window (2018-10-18 → 2022-03-31), SMART-RL reached a Sharpe ratio of 0.638 against 0.467 for SenT-In (E13 recipe) as a trading signal (difference not significant, p 0.197) and a maximum drawdown of -19.3 % against -29.7 % for buy-and-hold, which the paper-recipe SenT-In equals because it is always invested (the 95 % interval of the drawdown difference excludes 0); its Sharpe ratio is below buy-and-hold's (0.750; not significant).

### 12. Decisions for Roham

1. Gate G6: reply "ادامه" to accept E14 and E15 as reported.
2. E16 (optional, protocol §4): TDQN of Théate and Ernst (2021) under the SMART-RL trading conditions. It needs the authors' official code: download the repository as a ZIP from https://github.com/ThibautTheate/An-Application-of-Deep-Reinforcement-Learning-to-Algorithmic-Trading (Code → Download ZIP) into `thesis/data/raw/tdqn/`, then write "E16 تأیید". Or say that E16 is skipped.
3. After G6: add the P6 section to `results/FINAL_RESULTS.md` and the P6 tables and figures to the thesis text (the published SenT-In numbers are added there as a quoted reference row, as the protocol asks).
