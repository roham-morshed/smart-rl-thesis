# گزارش فاز P1

خلاصه:
- پنل قیمت چهار بازار از 2007-09-17 تا 2026-09-30 ساخته شد و 13 ویژگی علّی دارد. هیچ بازده روزانه‌ای بالای 20٪ پیدا نشد.
- بازده کل (با سود نقدی) هر بازار از یک منبع آمد:
  - S&P 500: شاخص بازده کل رسمی.
  - FTSE: سابقه‌ی توزیع سود ISF از iShares.
  - Nifty: بازده سود نقدی رسمی روزانه‌ی NSE.
  - SSE: فقط بازده قیمتی، چون منبع رسمی پیدا نشد. به‌عنوان محدودیت ثبت شده است.
- پیکره‌ی خبری: از 42.0 میلیون سرخط در بازه‌ی 2007 تا 2025، پس از پاک‌سازی و حذف تکراری‌ها 19.4 میلیون سرخط نگه داشته شد.
- پوشش خبر محلی در 2008 تا 2023 برای هر چهار بازار نزدیک 100٪ روزهاست. **در 2024 به 2 تا 23٪ و در 2025 به حدود 40 تا 50٪ می‌رسد و در 2026 صفر است.** این مهم‌ترین محدودیت داده‌ای پروژه است و روی سال آزمون 2024 و آزمون پیش‌رو اثر دارد.
- همه‌ی 14 آزمون نشت داده پاس شد، از جمله بررسی واقعی زمان هر خبر نسبت به ساعت بسته‌شدن بازار.
- پیشنهاد: G1 پذیرفته شود. محدودیت پوشش خبری 2024 به بعد هم در پایان‌نامه گزارش شود.

## Details (English)

### What ran
- `scripts/p1_build_prices.py`, `scripts/p1_sentiment_sets.py`, `scripts/p1_build_news.py` and
  `scripts/p1_outputs.py`, the last two through the unattended offline pipeline (`scripts/run_offline_pipeline.py`,
  log `results/logs/pipeline.log`).
- The news download (101 files, 13.73 GB) was verified against `manifest.json` (exact sizes) and against the P0
  inventory: 44,892,240 rows, identical.

### Key numbers (results.json)

**Prices (`data.prices.*`):**
- Trading days per market: SP500 4,790; FTSE100 4,809; NIFTY50 4,671; SSE 4,623.
- `n_flagged_returns` = 0 in every market.
- Volume is disabled for NIFTY50 (G0).

**News (`data.news.*`):**
- `n_raw` 42,046,439 (rows of the 14 approved subsets dated 2007-01-01 .. 2025-12-31); `n_kept` 19,382,825;
  `n_duplicates` 19,470,200.
- The largest contributors after deduplication: FNSPID 9.44 M, Bloomberg/Reuters 7.47 M, Benzinga 1.50 M.
- NYT subsets after the business filter: 83 k, 67 k and 17 k rows.
- Tags: US 11,182,800; UK 285,594; IN 175,599; CN 428,888.
- Mean local items per trading day, 2010-2025: SP500 2,285; FTSE100 56.7; NIFTY50 35.5; SSE 90.4.
- Share of days with local news, 2010-2025: 0.920 / 0.893 / 0.855 / 0.910.
- By year (`share_days_with_local_by_year`): close to 1.0 in 2008-2023 for every market.

  | year | SP500 | FTSE100 | NIFTY50 | SSE |
  | --- | ---: | ---: | ---: | ---: |
  | 2024 | 0.23 | 0.04 | 0.02 | 0.08 |
  | 2025 | 0.50 | 0.43 | 0.39 | 0.50 |
  | 2026 | 0.00 | 0.00 | 0.00 | 0.00 |

  The cause is source coverage: FNSPID ends 2024-01-10, Bloomberg/Reuters ends 2016, and only NYT business
  (to 2025-05) and Yahoo Finance (2025-03 .. 2025-09) remain afterwards.

**Labelled sets (`data.sentiment_sets.*`, train / val / test):**

| set | train | val | test |
| --- | ---: | ---: | ---: |
| phrasebank50 | 3,384 | 725 | 726 |
| phrasebank_all | 1,601 | 343 | 315 |
| fiqa | 777 | 167 | 167 |
| twitter | 9,543 | 1,186 | 1,187 |

**Leakage tests** (`meta.leakage_by_gate.G1` = true): 14 passed, none skipped. This includes test 2 on the real
aligned news (every item before its session's close) and test 5 on the forecast files.

### Deviations from the SPEC (all conservative or required by the data)
1. **Day-only timestamps.** A timestamp at exactly midnight New York time is treated as day-only, like midnight
   UTC (SPEC §2.3), and moved to 23:59:59 UTC. This affects about 9 % of NYT rows, which carry print dates. It can
   only delay news, never bring it forward.
2. **NYT business rows.** A row is kept when its news desk or section is business, financial, economy, DealBook,
   Your Money or markets, or when its nytimes.com URL path is one of those sections. The desks kept are in
   `results/p1_news.json`.
3. **NYT 2010-2021 headlines.** This subset stores abstract + headline + lead, so the headline field is taken
   instead of the first line (`headline_from_text`).
4. **phrasebank_all splits are nested in phrasebank50's** (decided with Roham). An independent split would leave
   95 / 95 validation / test sentences after the overlap rule.
5. **Figures.** Standard English labels and Latin digits (Roham, 2026-10-18); the Persian variant is available
   with `figures.language: fa`. matplotlib 3.11 shapes Persian natively, so arabic_reshaper/python-bidi are not
   applied to it (they would reverse the text).

### Problems and proposed handling
- **News coverage after 2023 (main limitation).** The policies of the 2024 fold and of the forward test see sparse
  or carried-over sentiment. Proposal:
  - keep the design unchanged and report the coverage table above;
  - in P5, add as supplementary evidence (outside the frozen keys) the confirmation statistics for 2020-2023,
    where coverage is complete;
  - the decision is Roham's at G4.

### Outputs
- Tables T01, T02.
- Figure F01: share of days with local news per market and year.
