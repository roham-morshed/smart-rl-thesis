# گزارش فاز P0

خلاصه:
- محیط `smartrl-thesis` (پایتون ۳٫۱۱، torch 2.6.0+cu124 مطابق درایور 551.76) ساخته شد. GPU (RTX 2060، ۶ گیگ)، فونت B Nazanin و ۱۵۴ گیگابایت فضای خالی تأیید شد.
- قیمت هر ۸ نماد از Yahoo دریافت شد. اولین تاریخ مشترک چهار شاخص ۲۰۰۷-۰۹-۱۷ است.
- آزمون ETF سود نقدی برای FTSE (ISF.L) قبول شد، ولی برای Nifty (NIFTYBEES) و SSE (510210) رد شد. برای Nifty داده‌ی رسمی روزانه‌ی بازده سود نقدی NSE پیدا و کامل دانلود شد. برای SSE به تصمیم شما نیاز است.
- پیکره‌ی خبری در دسترس است: ۲۴ زیرمجموعه، ۵۷٫۱ میلیون ردیف، ۱۹٫۹ گیگ. پیشنهاد: ۱۴ زیرمجموعه‌ی مالی نگه داشته شود (۴۴٫۹ میلیون ردیف، ۱۳٫۷ گیگ) و ۱۰ زیرمجموعه کنار برود.
- محدودیت مهم: پوشش خبری بعد از ۲۰۲۴ کم است و بعد از ۲۰۲۵-۰۹-۰۶ هیچ خبری نیست. بخشی از آزمون پیش‌رو بدون خبر خواهد بود.
- مجموعه‌های برچسب‌دار (PhraseBank، Twitter، FiQA)، هر دو FinBERT، واژه‌نامه‌ی LM و داده‌های ETT همه در دسترس‌اند.
- سنجش سرعت: FinBERT حدود ۱۱۹۰ تیتر بر ثانیه. PPO با ۶ کار موازی ۵۶۰۰ گام بر ثانیه (هر اجرای ۳۰۰ هزار گامی حدود ۳٫۳ دقیقه). SAC روی GPU با ۳ کار موازی ۱۲۰ گام بر ثانیه.
- برآورد کل زمان ماشین: حدود ۲ تا ۳ روز.
- برای عبور از G0 به تأیید شما درباره‌ی ۷ پرسش انتهای گزارش نیاز است.

## Details (English)

### What was implemented
- Environment: conda env `smartrl-thesis` (Python 3.11.17, conda-forge) created with the existing Anaconda
  install. torch 2.6.0+cu124 was installed first. All other packages of `environment.yml` were then installed
  with torch pinned, so pip could not replace it with a CPU build. Exact versions are in `meta.packages`, and the
  full `pip freeze` is in `environment.lock.txt`.
- Package skeleton:
  - `smartrl/config.py`: paths, frozen config, SHA-256.
  - `smartrl/results.py`: dotted-key store; dotted ids kept as one key; atomic writes.
  - `smartrl/runs.py`: `results/runs/<run_id>/meta.json` plus the SQLite registry `results/registry.sqlite`.
  - `smartrl/data/access.py`: `TestLockError`, the G4 lock check, and the SPEC's pre-lock purposes.
  - `smartrl/data/{prices,dividends,hf}.py`.
  - `smartrl/env/{portfolio,trading_env}.py`: SPEC §5 environment, rewards R1/R2/R3/IMIT, the oracle DP and the
    curriculum hook.
  - `smartrl/marketmind/{informer,model}.py`: ProbSparse attention, distilling, sentiment gate, quantile heads.
  - `smartrl/sentiflow/finbert.py`.
- Scripts: `scripts/p0_probe.py`, `scripts/p0_benchmark.py`, `scripts/p0_compute_plan.py`.
- Git initialised. `tests/test_results_contract.py` is unchanged.
- `configs/paths.yaml`: `project_root` and `hf_cache` now point to the real folder,
  `<project_root>`. This does not affect the config hash.
- Run records: `P0_probe_*` and `P0_bench_*` under `results/runs/`.

### Key numbers (results.json)

**Environment** (`p0.probe.env`): GPU RTX 2060 with 6.0 GB, CUDA 12.4, 154.5 GB free on D:. Font
`bnazanin.ttf` was found; the Vazirmatn fallback is also available.

**Prices** (`p0.probe.yf`): all 8 tickers downloaded up to 2026-09-30. `first_common_index_date` = 2007-09-17
(set by `^NSEI`).
- Volume: `^NSEI` has zero volume on every day of 2007–2012 and real volume from 2013 on. The other three
  indices have volume throughout.

**Dividend proxy check** (`p0.etf_summary`, SPEC §2.1: correlation ≥ 0.98 and a non-empty dividend history):

| market | series | corr. 2015–2024 | dividends on Yahoo | result |
| --- | --- | ---: | --- | --- |
| SP500 | ^SP500TR (total return, no proxy needed) | 0.9998 | — | ok |
| FTSE100 | ISF.L | 0.9878 | 55 events, 2013-02-20 → 2026-09-17 | **pass**, but no dividends before 2013-02 |
| NIFTY50 | NIFTYBEES.NS | 0.0477 | none | **fail** |
| SSE | 510210.SS | 0.0354 | 4 events, all in 2023 | **fail** |

- NIFTYBEES has a bad Yahoo record around a unit split: −90 % on 2019-12-19 and +897 % on 2019-12-23.
  Excluding that glitch, the correlation is still 0.970.
- 510210.SS has several erroneous jumps (e.g. +403 % on 2016-05-25). Excluding them, the correlation is 0.71.
- Other Shanghai ETFs also fail: 510180 (0.942), 510300 (0.945) and 510050 (0.891), all with gaps in their
  dividend history.
- Replacement found for NIFTY50: the official daily dividend yield of the Nifty 50 from NSE Indices Ltd
  (niftyindices.com, "P/E, P/B & Div. Yield"). It has been downloaded to `data/raw/dividends/` and covers every
  ^NSEI trading day from 2007-09-17 to 2026-09-30 (0 missing days).

**News corpus** (`p0.probe.hf`, `p0.news_proposal`): access works with the token. 131 parquet files, 24 subsets,
57,114,479 rows, 19.94 GB. Schema: `date`, `text`, `extra_fields`.
- The proposed keep set is 14 subsets: 44,892,240 rows and 13.73 GB to download.
- The last kept date is 2025-09-06. Without the Yahoo-2025 and NYT subsets, the last date is 2024-07-30.

| subset | rows | GB | first | last | proposal | reason |
| --- | ---: | ---: | --- | --- | --- | --- |
| fnspid_news | 28,606,801 | 10.32 | 1999-08-31 | 2024-01-09 | **keep** | FNSPID financial news (expected in SPEC) |
| bloomberg_reuters | 8,203,282 | 1.21 | 2006-10-20 | 2016-08-16 | **keep** | Bloomberg / Reuters (expected in SPEC) |
| nyt_headlines_1990_2020 | 2,802,355 | 0.25 | 1990-01-01 | 2020-12-31 | **keep, business rows only** | NYT business (expected in SPEC) |
| nyt_articles_2000_present | 2,175,632 | 0.64 | 2000-01-01 | 2025-05-01 | **keep, business rows only** | NYT business (expected in SPEC) |
| benzinga_6000stocks | 1,568,363 | 0.16 | 2009-02-14 | 2020-06-11 | **keep** | Benzinga (expected in SPEC) |
| nyt_headlines_2010_2021 | 597,708 | 0.17 | 2007-07-07 | 2021-12-31 | **keep, business rows only** | NYT business (expected in SPEC) |
| sentarl_combined | 318,307 | 0.06 | 1997-01-03 | 2021-03-09 | **keep** | financial news for 20 assets |
| us_financial_news_2018_jeet | 305,947 | 0.43 | 2017-12-07 | 2018-05-31 | **keep** | U.S. financial news, multi-publisher |
| yahoo_finance_articles | 198,926 | 0.42 | 2025-03-11 | 2025-09-06 | **keep** | Yahoo Finance; only subset covering 2025 |
| yahoo_finance_felixdrinkall | 70,286 | 0.07 | 2017-01-03 | 2023-12-30 | **keep** | Yahoo Finance (expected in SPEC) |
| sp500_daily_headlines | 18,153 | 0.00 | 2008-01-02 | 2024-03-04 | **keep** | S&P 500 headlines (expected in SPEC) |
| finsen_us_2007_2023 | 15,523 | 0.00 | 2007-06-04 | 2023-07-16 | **keep** | FinSen U.S. financial news |
| gold_news_kaggle | 10,404 | 0.00 | 2000-02-15 | 2019-02-01 | **keep** | gold/commodity market headlines |
| cnbc_headlines | 553 | 0.00 | 2006-11-30 | 2024-07-30 | **keep** | CNBC (expected in SPEC) |
| frontpage_news | 4,994,950 | 0.95 | 2015-07-16 | 2020-11-08 | exclude | general news front pages |
| headlines_10sites_2007_2022 | 4,405,397 | 0.42 | 2007-01-01 | 2023-01-01 | exclude | general news |
| all_the_news_2 | 1,972,976 | 3.65 | 2016-01-01 | 2020-04-01 | exclude | general news |
| american_news_jonasbecker | 265,552 | 0.98 | 2016-02-05 | 2021-06-25 | exclude | general news crawl |
| huffpost_news | 209,415 | 0.04 | 2012-01-28 | 2022-09-23 | exclude | general news |
| mind_news_2019 | 128,621 | 0.03 | 2019-11-09 | 2019-11-22 | exclude | recommendation dataset |
| djia_stock_headlines | 102,324 | 0.01 | 2000-01-03 | 2016-07-01 | exclude | top-25 world headlines per day, general news |
| reddit_worldnews_2008_2016 | 73,578 | 0.01 | 2008-06-08 | 2016-07-01 | exclude | Reddit |
| reddit_finance_sp500 | 48,029 | 0.10 | 2008-11-11 | 2025-07-03 | exclude | Reddit |
| wikinews_articles | 21,397 | 0.03 | 2002-04-01 | 2022-11-19 | exclude | general news |

**Labelled sentiment sets** (`p0.probe.sets`):
- PhraseBank, from the official zip in `takala/financial_phrasebank`: 50agree = 4,846 sentences
  (2,879 neu / 1,363 pos / 604 neg); allagree = 2,264.
- Twitter (`zeroshot/twitter-financial-news-sentiment`): train 9,543, validation 2,388.
- FiQA-2018 Task 1, mirror `pauri32/fiqa-2018`: `sentence` + `sentiment_score`; splits 961 / 102 / 150, which
  will be re-split per SPEC.
- The TheFinAI mirrors are gated (they need a separate access request), and the ChanceFocus mirror no longer
  exists.

**Models** (`p0.probe.models`):
- `yiyanghkust/finbert-pretrain` (FinVocab of 30,873 tokens, lower case) and `ProsusAI/finbert` both load and
  run.
- Note: transformers 5.x cannot auto-build the finbert-pretrain tokenizer (its config has no `model_type` and the
  repo ships only `vocab.txt`). It is loaded explicitly with `BertTokenizer(vocab=...)`; the tokenisation was
  checked against the reference WordPiece.
- The `pysentiment2` LM dictionary works.

**ETT** (`p0.probe.ett`): ETTh1 has 17,420 rows and ETTm1 69,680, both covering 2016-07-01 → 2018-06-26.

**Speed** (`p0.benchmark`, `p0.compute_plan`):
- FinBERT: 1,192 headlines/s for inference (fp16, max_len 64, 1.4 GB GPU) and 241 samples/s for training.
- PPO, steps/s per job and aggregate:

  | parallel jobs | per job | aggregate |
  | ---: | ---: | ---: |
  | 1 | 1,882 | 1,683 |
  | 2 | 1,858 | 2,296 |
  | 4 | 1,738 | 4,154 |
  | 6 | 1,510 | 5,604 |

  A validation pass takes 0.15–0.25 s.
- SAC: 20.6 steps/s per CPU job versus 52.8 on the GPU alone and 120 aggregate with 3 GPU jobs.
- MarketMind: one epoch over 16.8 k samples takes 6.0 s.
- E3 backbone: 4.2 s per ETTh1 epoch and 18.1 s per ETTm1 epoch.
- Plan: about 840 PPO runs take ≈ 12.5 h, and 72 SAC runs take ≈ 33 h on the GPU in parallel with PPO.
  The GPU blocks take ≤ 24 h as an upper bound, plus corpus scoring.

### Deviations from SPEC (none applied yet; all need approval)
1. Dividend fallbacks: see questions 2 and 3.
2. `parallel.rl_jobs` 4 → 6, and SAC on `cuda` with 3 parallel jobs: question 6. Changing frozen.yaml changes
   the config hash; this will be done after your approval.

### Problems and proposed fixes
- News coverage ends 2025-09-06, and 2024 is thin: only NYT business, Yahoo and the first months of FNSPID and
  the S&P 500 headlines cover it. The forward test (2025-01 → 2026-09) will have news only until 2025-09-06
  (`p0.news_proposal.latest_kept_date`). `forward.news_days_share` will report the exact share.
- `^NSEI` has no volume before 2013.

### Questions for Roham
1. **News subsets:** approve the 14 / 10 split above? NYT subsets would keep only rows whose desk or section is
   business, financial or economy; the exact filter will be reported in P1.
2. **NIFTY50 dividends:** use the official NSE daily dividend yield (dy_t straight from niftyindices.com) instead of
   NIFTYBEES?
3. **SSE dividends** (no usable ETF):
   - (a) Annual dividend yields of the SSE Composite from a cited source (e.g. SSE or CSIndex statistics), entered
     in `dividend_fallback.SSE`. You or the design lead would provide the values and the citation.
   - (b) Price return only (dy = 0), recorded as a limitation. This understates TR by about 2 %/yr and slightly
     favours cash.
   - FTSE100 has the same gap before 2013-02. Fill 2009–2013 with cited annual FTSE 100 yields, or accept and
     record it?
4. **NIFTY50 volume** (zero volume 2007–2012):
   - (a) Disable `vchg` for NIFTY50 on all dates (recommended: same meaning in every fold).
   - (b) Keep it and mask only the zero-volume days.
5. **MarketMind channels:** SPEC §4.1 says C_p = 11 or 12, but §2.2 lists 13 price features (12 without `vchg`).
   I propose C_p = 13 for the pooled model, with `vchg` set to 0 wherever volume is disabled.
6. **Parallelism:** `parallel.rl_jobs: 6` (PPO on CPU) and SAC on the GPU with 3 jobs?
7. **FiQA duplicates:** the mirror has repeated sentences with different targets (102 → 87 unique in its
   validation file). I propose one row per normalised sentence, with the mean score, before thresholding and
   splitting.
