# گزارش فاز P2

خلاصه:
- Word2Vec حوزه‌ی مالی روی 18.3 میلیون سرخطِ تا پایان 2019 آموزش دید. واژگانش 268,827 کلمه است.
- **SentiFlow (SF)** روی مجموعه‌ی تجمیعی به macro-F1 برابر 0.814 رسید.
  - **فرضیه‌ی H3 پشتیبانی نشد.** SF از مدل احساسات مقاله‌ی مبنا (شاخه‌ی A، با F1 برابر 0.752) معنادار بهتر است: اختلاف F1 برابر +0.062، بازه‌ی 95٪ [0.049, 0.075]، مک‌نمار p کمتر از 0.001. ولی از FinBERT (شاخه‌ی B، با F1 برابر 0.820) کمی ضعیف‌تر است: اختلاف −0.006، مک‌نمار p = 0.34.
  - وزن دروازه‌ی ترکیب به‌طور میانگین 0.15 است، یعنی SF بیشتر به FinBERT تکیه می‌کند.
- ترتیب GRU→CNN (مدل مقاله‌ی مبنا) با ترتیب معکوس تفاوت معنادار ندارد.
- بازتولید FinBERT ما با اعداد منتشرشده‌ی Araci (2019) سازگار است (جدول در بخش انگلیسی).
- کالیبراسیون دمایی خطای کالیبراسیون (ECE) را از 0.094 به 0.026 رساند.
- همه‌ی 19.4 میلیون سرخط امتیاز گرفتند. فقط از بذر 0 استفاده شد، چون پنج بذر حدود 12.7 ساعت طول می‌کشید و طبق قانون SPEC (سقف 12 ساعت) باید بذر 0 به‌تنهایی به کار می‌رفت.
- ویژگی‌های روزانه‌ی احساسات چهار بازار ساخته شد.
- همه‌ی 13 آزمون نشت داده پاس شد.

## Details (English)

### What ran (offline pipeline, `results/logs/pipeline.log`)
| step | runtime |
| --- | --- |
| Branch B pre-training (15 FinBERT fine-tunes) | — |
| `p2_word2vec.py` | 68.8 min (workers = 1, deterministic) |
| `p2_sentiflow.py` (E1 + E1-b) | 18.4 min |
| `p2_score_and_daily.py` | 153.6 min |

### Key numbers (results.json)

**`data.word2vec`:** n_headlines 18,302,487; vocab_size 268,827.

**Macro-F1, mean ± sd over seeds 0-4 (`sentiflow.<DS>.<SM>.f1`):**

| set | LEX | A | A_REV | B | SF | PROSUS |
| --- | --- | --- | --- | --- | --- | --- |
| phrasebank50 | 0.443 | 0.726 ± 0.017 | 0.718 ± 0.009 | 0.817 ± 0.007 | 0.810 ± 0.024 | — |
| phrasebank_all | 0.441 | 0.872 ± 0.016 | 0.866 ± 0.025 | 0.961 ± 0.007 | 0.954 ± 0.019 | — |
| fiqa | 0.320 | 0.564 ± 0.025 | 0.571 ± 0.027 | 0.633 ± 0.013 | 0.625 ± 0.036 | 0.495 |
| twitter | 0.432 | 0.771 ± 0.006 | 0.760 ± 0.021 | 0.823 ± 0.004 | 0.819 ± 0.020 | 0.656 |
| pooled | 0.444 | 0.752 ± 0.007 | 0.744 ± 0.010 | 0.820 ± 0.002 | 0.814 ± 0.023 | — |

Pooled accuracy: LEX 0.542, A 0.784, A_REV 0.776, B 0.845, SF 0.841.

**Tests on pooled (`sentiflow.tests.pooled`):**
- SF − B: F1 diff −0.0062 [−0.0111, −0.0013]; median McNemar p 0.337; significant in 1 of 5 seeds.
- SF − A: +0.0622 [0.049, 0.0754]; median McNemar p < 0.001.
- A − A_REV: +0.0074; median McNemar p 0.440.

H3 is decided by the rule of SPEC §12: SF must beat both A and B. It beats A significantly but not B, so H3 is
**not supported**.

**Gate (`sentiflow.gate.<DS>.alpha_mean`):** 0.15 pooled (sd 0.29); 0.11-0.25 on the single sets. The weight
of Branch A is small; fusion leans on FinBERT.

**Calibration (pooled test, mean of five per-seed temperatures):**

| model | T | ECE before → after | Brier before → after |
| --- | --- | --- | --- |
| SF | 1.853 | 0.0945 → 0.0260 | 0.2553 → 0.2326 |
| B | 1.970 | 0.0973 → 0.0254 | 0.2512 → 0.2264 |

**E1-b (Araci split, `sentiflow.e1b`):**

| set | model | accuracy | F1 | Araci (2019) acc / F1 |
| --- | --- | --- | --- | --- |
| phrasebank_all | B | 0.958 | 0.941 | 0.97 / 0.95 |
| phrasebank_all | SF | 0.937 | 0.913 | |
| phrasebank50 | B | 0.851 | 0.829 | 0.86 / 0.84 |

Our Branch B reproduces published FinBERT closely, which supports the soundness of the training pipeline.

**Scoring (`sentiflow.scoring`):** n_headlines 19,382,825; ensemble = false; 2.54 h.
- Seed 0 alone took 2.54 h, so five seeds would need about 12.7 h, more than the 12 h of SPEC §3.8; seed 0 was
  used and the note is in results.json.
- The A-branch index for SENTIN and the calibrated B index for B5 were computed in the same pass.

**Leakage tests** (`meta.leakage_by_gate.G2` = true): 13 passed.

### Deviations / reading of the SPEC
1. **Loss of Branch B.** Plain cross-entropy (SPEC §3.4 lists no class weights). A and A_REV use class-weighted
   cross-entropy (§3.2).
2. **Early stopping.** Keeps the best epoch, patience 3, for every branch.
3. **Embedding weight decay.** The Branch-A embedding table is excluded from AdamW weight decay, so rows of
   words that never occur in the labelled data keep their Word2Vec values for corpus scoring.
4. **Pooled set.** The pooled test set is phrasebank50 + fiqa + twitter (phrasebank_all ⊂ phrasebank50 is not
   counted twice).

### Outputs
- Tables T03, T04.
- Figures F02 (reliability before/after, thesis figure), F03 (α histograms) and F04 (daily Se_loc / Se_glob per
  market; the March 2020 drop is visible in all four markets).
