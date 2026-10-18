## News subsets

| subset | rows | GB | first | last | proposal | reason |
| --- | ---: | ---: | --- | --- | --- | --- |
| fnspid_news | 28,606,801 | 10.32 | 1999-08-31 | 2024-01-09 | **keep** | FNSPID financial news (expected in SPEC) |
| bloomberg_reuters | 8,203,282 | 1.21 | 2006-10-20 | 2016-08-16 | **keep** | Bloomberg / Reuters (expected in SPEC) |
| nyt_headlines_1990_2020 | 2,802,355 | 0.25 | 1990-01-01 | 2020-12-31 | **keep_business** | NYT; keep only business/financial desk rows (expected in SPEC) |
| nyt_articles_2000_present | 2,175,632 | 0.64 | 2000-01-01 | 2025-05-01 | **keep_business** | NYT; keep only business/financial desk rows (expected in SPEC) |
| benzinga_6000stocks | 1,568,363 | 0.16 | 2009-02-14 | 2020-06-11 | **keep** | Benzinga (expected in SPEC) |
| nyt_headlines_2010_2021 | 597,708 | 0.17 | 2007-07-07 | 2021-12-31 | **keep_business** | NYT; keep only business/financial desk rows (expected in SPEC) |
| sentarl_combined | 318,307 | 0.06 | 1997-01-03 | 2021-03-09 | **keep** | SentARL financial news for 20 assets (finance subset) |
| us_financial_news_2018_jeet | 305,947 | 0.43 | 2017-12-07 | 2018-05-31 | **keep** | U.S. financial news, multi-publisher (finance subset) |
| yahoo_finance_articles | 198,926 | 0.42 | 2025-03-11 | 2025-09-06 | **keep** | Yahoo Finance (expected in SPEC); only subset covering 2025 |
| yahoo_finance_felixdrinkall | 70,286 | 0.07 | 2017-01-03 | 2023-12-30 | **keep** | Yahoo Finance (expected in SPEC) |
| sp500_daily_headlines | 18,153 | 0.00 | 2008-01-02 | 2024-03-04 | **keep** | S&P 500 headlines (expected in SPEC) |
| finsen_us_2007_2023 | 15,523 | 0.00 | 2007-06-04 | 2023-07-16 | **keep** | FinSen U.S. financial news (finance subset) |
| gold_news_kaggle | 10,404 | 0.00 | 2000-02-15 | 2019-02-01 | **keep** | gold/commodity market headlines (markets subset) |
| cnbc_headlines | 553 | 0.00 | 2006-11-30 | 2024-07-30 | **keep** | CNBC (expected in SPEC) |
| frontpage_news | 4,994,950 | 0.95 | 2015-07-16 | 2020-11-08 | **exclude** | general news front pages |
| headlines_10sites_2007_2022 | 4,405,397 | 0.42 | 2007-01-01 | 2023-01-01 | **exclude** | general news, 10 major sites |
| all_the_news_2 | 1,972,976 | 3.65 | 2016-01-01 | 2020-04-01 | **exclude** | general news |
| american_news_jonasbecker | 265,552 | 0.98 | 2016-02-05 | 2021-06-25 | **exclude** | general news crawl |
| huffpost_news | 209,415 | 0.04 | 2012-01-28 | 2022-09-23 | **exclude** | general news |
| mind_news_2019 | 128,621 | 0.03 | 2019-11-09 | 2019-11-22 | **exclude** | recommendation dataset (MIND) |
| djia_stock_headlines | 102,324 | 0.01 | 2000-01-03 | 2016-07-01 | **exclude** | top-25 world headlines per day (Reddit r/worldnews style), general news |
| reddit_worldnews_2008_2016 | 73,578 | 0.01 | 2008-06-08 | 2016-07-01 | **exclude** | Reddit |
| reddit_finance_sp500 | 48,029 | 0.10 | 2008-11-11 | 2025-07-03 | **exclude** | Reddit |
| wikinews_articles | 21,397 | 0.03 | 2002-04-01 | 2022-11-19 | **exclude** | general news |

## Compute plan

| block | runs | measured speed | estimate |
| --- | ---: | --- | --- |
| SentiFlow branch B fine-tuning (5 seeds) + A, A_rev, fusion | 20 | 241 samples/s (batch 32, fp16) | ≈ 0.3 GPU h (+ E1-b < 0.5 h) |
| Corpus scoring with FinBERT | per seed | 1192 headlines/s (fp16, max_len 64) | 2.3 GPU h per seed per 10 M headlines; 5-seed ensemble fits the 12 h rule up to 10.3 M headlines |
| MarketMind and comparison models (E2, incl. annual refits) | ≈ 160 | 6.0 s per epoch on the largest window (16.8 k samples) | ≤ 15.9 GPU h (60 epochs each; early stopping makes it shorter) |
| LTSF benchmark (E3) | 24 | ETTh1 4.2 s / ETTm1 18.1 s per epoch (h = 720) | ≤ 7.4 GPU h (100 epochs each; patience 10 makes it shorter) |
| PPO runs (P4 grid + IMIT, P5 E5–E7, forward) | 840 | 5604 steps/s with 6 parallel jobs (3.3 min per 300 k-step run) | ≈ 12.5 h |
| SAC runs (B7, P5 + forward) | 72 | 120 steps/s with 3 jobs on the GPU (CPU: 83 steps/s with 4 jobs) | ≈ 33.2 h on the GPU, parallel to PPO on the CPU |
