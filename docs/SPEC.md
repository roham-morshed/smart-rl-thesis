# SMART-RL — Technical Specification (frozen design v1.0)

This document fixes every design decision of the thesis implementation. Values in `configs/frozen.yaml`
mirror it; if the two disagree, stop and ask. Section numbers are referenced by EXPERIMENTS.md and
RESULTS_SCHEMA.md.

## 0. Names and scope

- System: **SMART-RL**. Modules (use these names in code, logs, figures):
  Data Processing Module (**DPM**), **SentiFlow**, **MarketMind**, **PPO Agent**,
  Sentiment Trading Module (**STM**), Adaptive Weighting System (**AWS**).
- Task: daily long/flat exposure control of four equity indices. Market keys:
  `SP500` (S&P 500), `FTSE100` (FTSE 100), `NIFTY50` (Nifty 50), `SSE` (SSE Composite).
- Exposure e ∈ [0, 1]: 1 = fully invested in the index (total-return series), 0 = cash.
  No leverage, no short selling.
- Base paper: Chauhan, Ahmed & Sinha (2025), "A novel deep learning model for stock market prediction using a
  sentiment analysis system from authoritative financial website's data", Connection Science 37(1): 2455070
  (model name SenT-In). It builds a daily sentiment index with a GRU→CNN classifier on Word2Vec (CBOW)
  vectors, weights the inputs with a sentiment-awareness layer and predicts the next-day direction of the same
  four indices with an Informer (2005-01 → 2022-03, 80/20 split; accuracy, F1, AUC-ROC, PR-AUC). It has no
  trading policy, no reinforcement learning, no transaction costs and no dynamic weighting. Its reproduction is
  `SENTIN` (§9.1), used by the BASE method (E5) and the base-paper task comparison (E13, §15).

## 1. Time protocol

| Block | Dates | Use |
| --- | --- | --- |
| Raw history | first date all four index series exist (expected 2007-09-17) → 2026-09-30 | feature warm-up (252 days), MarketMind training |
| MarketMind annual refits | predicted years 2010 … 2024, plus forward 2025–2026 | out-of-sample forecast features (§4.7) |
| Policy data start | 2010-01-01 | PPO, SAC, STM, AWS never see earlier days |
| Development (P4) | 2010-01-01 → 2019-12-31, split by trading days of each market into 70% train / 15% validation / 15% dev-test (≈ 2010-01→2016-12 / 2017-01→2018-06 / 2018-07→2019-12) | choose reward, STM form, AWS parameters; trials logged |
| Confirmation (P5) | five folds, test year Y ∈ {2020, 2021, 2022, 2023, 2024}: train 2010-01-01 → (Y−2)-12-31, validation (Y−1), test Y | main results of Chapter 4 |
| Forward (P5) | train 2010-01-01 → 2023-12-31, validation 2024, test 2025-01-01 → last trading day ≤ 2026-09-30 | validation on unseen data |

The test lock (PROJECT_RULES.md rule 2) guards every read of dates ≥ 2020-01-01 by policy, STM or AWS code.

## 2. Data Processing Module (DPM)

### 2.1 Prices

- Source: Yahoo Finance via `yfinance` (daily, `auto_adjust=False`).
- Index series (features, regimes): `^GSPC`, `^FTSE`, `^NSEI`, `000001.SS`.
- Total-return series for profit and loss:
  - SP500: `^SP500TR` close.
  - FTSE100, NIFTY50, SSE: index price return plus a daily dividend accrual `dy_t / 252`, where `dy_t` is the
    trailing-12-month dividend yield of a tracking ETF: FTSE100 → `ISF.L`, NIFTY50 → `NIFTYBEES.NS`,
    SSE → `510210.SS` (SSE Composite Index ETF). Verify each ETF in P0: correlation of daily price returns
    with its index ≥ 0.98 over 2015–2024 and a non-empty dividend history. If an ETF fails, use the fallback
    in `configs/frozen.yaml: dividend_fallback` and record it as a limitation.
- Cleaning: keep each market's own trading calendar (the dates of its index series). Drop rows without a
  close. Flag |daily log return| > 20 % for review in the P1 report (do not delete silently). If volume is
  missing or zero for most days of a market, disable volume features for that market (mask, do not impute).
- Cash return: `r_f = 1.02^(1/252) − 1` per trading day (2 % per year, as in the proposal).

### 2.2 Price features (causal, per market)

| name | definition |
| --- | --- |
| `r1` | log(C_t / C_{t−1}) |
| `r5`, `r20`, `r60` | log(C_t / C_{t−k}) |
| `rv5`, `rv20` | sqrt(mean of r1² over last k days) · sqrt(252) |
| `range` | (H_t − L_t) / C_t |
| `rsi14` | Wilder RSI(14) / 100 |
| `macd` | MACD histogram (12, 26, 9) / C_t |
| `gap50`, `gap200` | C_t / MA_k(C) − 1 |
| `dd252` | C_t / max(C over last 252 days) − 1 |
| `vchg` | log(V_t / MA20(V)) (only if volume is available) |

Normalisation (the proposal's min-max): for each fold, per market and feature, `x' = (x − q01)/(q99 − q01)`
with q01, q99 the 1st and 99th percentiles on that fold's training dates only; clip to [−0.5, 1.5].

### 2.3 News corpus

- Primary source: Hugging Face `Brianferrell787/financial-news-multisource` (gated: Roham accepts the terms
  on the website; token in env var `HF_TOKEN`). Use rows dated 2007-01-01 → 2025-12-31 from finance and
  markets subsets. In P0 list all subsets with row counts and date ranges, propose the subsets to keep
  (expected: Bloomberg/Reuters, Benzinga, FNSPID, NYT business, Yahoo Finance, CNBC, S&P 500 headlines;
  exclude Reddit, general-news and recommendation-dataset subsets) and wait for approval at G0.
- Optional secondary source: local FNSPID files under the path in `configs/paths.yaml` (read only).
- Headline extraction: multisource → first line of `text` (split at the first newline), FNSPID →
  `Article_title`. Strip HTML and URLs. Drop headlines with fewer than 4 tokens. Truncate to 64 tokens.
- Deduplication: key = lower-cased headline with non-alphanumerics removed and whitespace collapsed,
  plus UTC date. Keep the earliest timestamp. Report removed duplicates per source.
- Timestamps: UTC. Day-only items (midnight UTC or flagged day-only in `extra_fields`) are treated as
  published at 23:59:59 UTC of that date (conservative: after all four closes).
- Market tagging: `GLOBAL` = every kept item. `US` = items from US-market subsets or matching the US
  keyword list; `UK`, `IN`, `CN` = keyword matches (case-insensitive, word boundaries) with the lists in
  `configs/frozen.yaml: news_keywords`. An item may carry several tags.
- Session assignment for market m: the first trading date d of m with `ts_utc < close_utc(m, d)`.
  Official closes (convert with `zoneinfo`, DST-aware): SP500 16:00 America/New_York; FTSE100 16:30
  Europe/London; NIFTY50 15:30 Asia/Kolkata; SSE 15:00 Asia/Shanghai. Weekend and holiday news roll to the
  next trading date; nothing is dropped. `GLOBAL` items are assigned to each market's calendar with the same rule.
- Store `data/processed/news_items.parquet` (id, source, ts_utc, headline, tags) and
  `data/processed/news_aligned_<MKT>.parquet` (id, trading_date, tag).

### 2.4 Labelled sentiment datasets (for SentiFlow only)

| key | source | labels | split |
| --- | --- | --- | --- |
| `phrasebank50` | HF `takala/financial_phrasebank`, `sentences_50agree` | neg / neu / pos | stratified 70/15/15, seed 2026 |
| `phrasebank_all` | same, `sentences_allagree` | neg / neu / pos | stratified 70/15/15, seed 2026 (also the split of Araci 2019 for the literature comparison E1-b: 20 % test, then 20 % of the rest for validation, i.e. 64/16/20, stratified, seed 2026) |
| `fiqa` | FiQA-2018 Task 1 sentiment (locate on HF, e.g. the TheFinAI or ChanceFocus mirror; verify it holds sentence + score) | score > 0.1 pos, < −0.1 neg, else neu | stratified 70/15/15, seed 2026 |
| `twitter` | HF `zeroshot/twitter-financial-news-sentiment` | 0 bearish→neg, 1 bullish→pos, 2 neutral→neu | official train; official validation split 50/50 stratified (seed 2026) into val/test |

Remove from validation and test any sentence that also occurs (after normalisation) in any training split.
Training set for SentiFlow = union of the four training splits (`phrasebank_all` is a subset of
`phrasebank50`; use `phrasebank50` for training and evaluate on both). Record sizes.

### 2.5 Storage

Parquet for panels, SQLite for the run registry, CSV only for exported tables (as in the proposal).

### 2.6 Leakage tests (`tests/test_leakage.py`, must pass from G1 on)

1. Perturbing prices after day t leaves every feature at day ≤ t unchanged.
2. Every news item assigned to (m, d) has `ts_utc < close_utc(m, d)`.
3. Scalers and thresholds are fitted only on training indices of the current fold.
4. Reading a date ≥ 2020-01-01 through policy code raises `TestLockError` when no valid lock exists.
5. MarketMind predictions for year y come from a model whose training data end on or before (y−1)-12-31.
6. No sentence of a SentiFlow validation or test split appears in any training split.
7. The STM, AWS and every policy use only features dated ≤ t for the decision at t.

## 3. SentiFlow

### 3.1 Domain Word2Vec
gensim 4; corpus = deduplicated headlines with `ts_utc ≤ 2019-12-31`; tokens = lower-case regex
`[a-z0-9$%']+`; vector_size 300, window 5, min_count 5, sg 0 (CBOW, the base paper's choice), negative 10,
epochs 5, seed 2026.

### 3.2 Branch A — GRU-RNN branch (the proposal's left branch; the same GRU→CNN order as the base paper's sentiment model, Chauhan et al. 2025 §4.5)
Embedding initialised from Word2Vec (trainable; OOV ~ N(0, 0.1)), max length 48 →
GRU(hidden 128, returns the sequence) → Conv1d(128→64, kernel 3, padding 1) + ReLU →
global max-pool → dropout 0.3 → Linear(64→3). Representation h_A = the pooled 64-d vector.
AdamW lr 1e-3, batch 64, up to 20 epochs, early stop on validation macro-F1 (patience 3),
class-weighted cross-entropy.

### 3.3 Branch A_rev — CNN→GRU (reverse order, an H3 reference for the GRU-first design)
Same embedding → Conv1d(300→64, kernel 3) + ReLU → GRU(128) → last hidden state → Linear(128→3).
Same training recipe.

### 3.4 Branch B — FinBERT
`yiyanghkust/finbert-pretrain` (pre-trained on financial text, not fine-tuned on PhraseBank) +
Linear(768→3) on [CLS]; max length 64; AdamW lr 2e-5, weight decay 0.01, batch 32, up to 4 epochs,
linear warm-up 10 %, fp16; early stop on validation macro-F1. h_B = [CLS] (768-d).
If the checkpoint cannot be downloaded, stop at G0 and ask (fallback `ProsusAI/finbert` would make the
PhraseBank comparison contaminated).

### 3.5 Fusion (attention-based adaptive fusion, the proposal's Figure 1)
α(x) = σ(W2 · ReLU(W1 · [h_A; h_B] + b1) + b2), W1: 832→32, W2: 32→1.
p_SF = α · p_A + (1 − α) · p_B. Train α with both branches frozen (NLL loss, AdamW lr 1e-3,
up to 10 epochs, early stop on validation NLL). Seed s fuses A_s with B_s.

### 3.6 Calibration
Temperature T fitted on the pooled validation split by minimising the NLL of softmax(log p_SF / T)
(L-BFGS). T is applied in every downstream use (corpus scoring included). Report ECE (15 equal-width
bins) and Brier before and after.

### 3.7 Models compared (H3)
`LEX` Loughran–McDonald dictionary (`pysentiment2`): polarity (pos − neg)/(pos + neg + 1e−6), classes by
two thresholds tuned on validation macro-F1; `A` (= the base paper's sentiment model); `A_rev`; `B`;
`SF` (fusion, calibrated); `PROSUS` (`ProsusAI/finbert` zero-shot, reported on `fiqa` and `twitter` only,
because it was fine-tuned on PhraseBank). Seeds 0–4 for A, A_rev, B, SF. Key pairs: SF versus A (SentiFlow
versus the base paper's sentiment model), SF versus B, and A versus A_rev (GRU-first versus CNN-first order).

### 3.8 Corpus scoring
Final SentiFlow = average of the five seeds' calibrated probabilities (if runtime exceeds 12 h, use
seed 0 and record it). Per headline: p_pos, p_neg, p_neu, polarity s = p_pos − p_neg, confidence
c = max(p). Also score every headline with calibrated Branch B alone (needed by baseline B5) and with
Branch A alone, uncalibrated and seed-averaged (the base paper's sentiment index, needed by `SENTIN`, §9.1).

### 3.9 Daily sentiment features (per market m, on m's calendar; tag = local tag of m, plus GLOBAL)
- n_loc = count of local items; n_glob = count of GLOBAL items.
- S_loc = Σ c·s / Σ c over local items (0 if none); S_glob likewise over GLOBAL items.
- disp_loc = standard deviation of s over local items (0 if fewer than 2).
- cov = min(1, n_loc / median daily n_loc over the current fold's training dates).
- Se_loc, Se_glob = exponentially weighted S with half-life 3 trading days, carried over days without news.
- Feature vector: [Se_loc, Se_glob, log1p(n_loc), log1p(n_glob), disp_loc, cov].
- Mean calibrated confidence of the day's local items (else GLOBAL items) = c̄ (used by AWS, §8).

## 4. MarketMind (Patch-Informer with sentiment-aware gate)

### 4.1 Inputs
Per market and day: price channels (§2.2, C_p = 11 or 12) and sentiment channels (§3.9, C_s = 6).
Look-back L = 96 trading days; patch length P = 8, stride S = 4, end padding by repeating the last value
S times → N = 24 patches per channel.

### 4.2 Tokens and the sentiment-aware gate (the proposal's Figure 3)
- Price stream, channel-independent: each price channel's patch (length P) → shared Linear(P→d), d = 64.
- Sentiment stream: each patch of all sentiment channels (P × C_s flattened) → Linear → d.
- Gate per price channel c and patch n: w = σ(W_e · [e_{c,n}; s_n] + b_e) ∈ R^d;
  token_{c,n} = e_{c,n} + w ⊙ s_n.
- Add a learned positional embedding (24 × d) and a learned market embedding (4 × d).

### 4.3 Encoder (Informer)
Two encoder layers with ProbSparse self-attention (factor c = 5), 4 heads, d_ff 128, dropout 0.1, GELU;
a distilling layer between them (Conv1d kernel 3 + BatchNorm + ELU + MaxPool kernel 3 stride 2).
Shared across channels (sequences of tokens per channel).

### 4.4 Heads
Flatten all channel outputs → Linear → heads:
- Returns: for horizons h ∈ {1, 5, 20} the cumulative log return r_{t→t+h}: q50_h plus two softplus
  deltas, q10 = q50 − Δlo, q90 = q50 + Δhi (no crossing).
- Volatility: log RV_h for h ∈ {5, 20}, RV_h = mean of r1² over the next h days.
Targets are scaled by the training-window standard deviation of r_h of each market; invert for evaluation.

### 4.5 Loss and training
Mean pinball loss over the nine (h, τ) pairs + 0.5 · mean MSE of the two log-RV heads.
AdamW lr 1e-3, weight decay 1e-4, batch 128, up to 60 epochs, early stopping (patience 8) on the last 10 %
of the training window, gradient clipping 1.0. One model pooled over the four markets.

### 4.6 H4 comparison models (same inputs, heads and losses)
- `LSTM`, `GRU`: two layers, hidden 64, one step per day (all channels), last hidden → heads.
- `INFORMER`: vanilla Informer, one token per day (Linear(C→64)), same encoder, no patching, no gate
  (sentiment channels concatenated as inputs). Also used by the BASE configuration.
- `PATCHTST`: channel-independent patching (P 8, S 4), three full-attention Transformer layers, d 64,
  4 heads; sentiment channels as extra channels.
- `DLINEAR`: moving-average decomposition (kernel 25) + one linear map per channel over L, channels mixed by
  a final linear layer → heads.
- `NAIVE`: q50 = 0 with empirical 10/90 % quantiles of r_h over the training window; volatility references
  `EWMA` (λ = 0.94) and `HAR` (Corsi 2009 HAR-RV fitted on the training window).
- `LSTM_PRICE`: the LSTM with price channels only (used by baseline B4).
- `MM_NOSENT`: MarketMind without the sentiment stream and gate (used by ablation A1).
- `SENTIN`: the base-paper reproduction (§9.1); it predicts P(next-day close > today's close), not quantiles,
  so it is evaluated in E13 and used by BASE, not in the H4 table.

### 4.7 Annual refits (out-of-sample features for policies)
For each variant in {`MM`, `MM_NOSENT`, `INFORMER`, `LSTM_PRICE`, `SENTIN`} and each predicted year
y = 2010 … 2024: train on [first usable date, (y−1)-12-31] (early stopping on the last 10 % of that window),
predict every day of year y using inputs dated ≤ t. Forward: train to 2024-12-31, predict 2025-01-01 onward.
Seed 0 for the feature files; `MM` and the comparison models also run seeds 1 and 2 for test years 2020–2024
(H4). Save `data/processed/forecast_<VARIANT>.parquet` (market, date, q10/q50/q90 for h1/h5/h20, logRV5,
logRV20; for `SENTIN`: market, date, p_up).

### 4.8 Metrics (test years 2020–2024, per market and pooled)
Pinball loss (mean over τ) per h; MAE and RMSE of q50 against the realised r_h; directional accuracy of
sign(q50) against sign(r_h) (exclude r_h = 0) with the Pesaran–Timmermann test; empirical coverage and mean
width of [q10, q90]; volatility QLIKE and MSE of log RV. Diebold–Mariano tests with the Harvey–Leybourne–
Newbold correction and Newey–West lag h−1 on the daily cross-market average pinball-loss differential:
MM versus each comparison model, per h.

## 5. Trading environment (shared)

- Decision at the close of day t: observe o_t, choose target exposure e_t^target ∈ [0, 1].
- No-trade band δ = 0.05: if |e_t^target − e_{t−1}| ≤ δ then e_t = e_{t−1}, else e_t = e_t^target.
- Execution at the close of t (main); robustness variant executes at the close of t+1 (§14).
- Net return over (t, t+1]: R^p_{t+1} = e_t · R^tr_{t+1} + (1 − e_t) · r_f − c · |e_t − e_{t−1}|,
  c = 10 basis points per unit of turnover (base case).
- Observation o_t = flattened last W = 20 days of [price features; sentiment features if the variant uses
  them; MarketMind features if used (q10, q50, q90 for h = 1, 5, 20 and logRV5, logRV20, min-max scaled
  on the fold's training dates)] + [e_{t−1}, 20-day strategy return, 20-day strategy volatility,
  current strategy drawdown].
- Training episodes: random start inside the training window, length 252 days (or to the window end),
  e_{−1} = 1. Evaluation: deterministic policy over a whole split, starting from e = 1 on its first day.

### 5.1 Reward forms (one is selected in P4; the proposal's "risk-adjusted reward")
- **R1 — differential Sharpe ratio** (Moody & Saffell 2001): with excess return x_t = R^p_t − r_f,
  A_t = A_{t−1} + η(x_t − A_{t−1}), B_t = B_{t−1} + η(x_t² − B_{t−1}), η = 0.01, A_0 = 0, B_0 = 1e−4,
  D_t = (B_{t−1}·ΔA_t − ½·A_{t−1}·ΔB_t) / (B_{t−1} − A_{t−1}²)^{3/2}, clipped to [−3, 3].
- **R2 — four-component risk-adjusted reward**: 100 · [ln(1 + R^p_t) − λ1 · min(x_t, 0)² − λ2 · max(0, DD_t − DD_{t−1})],
  where R^p already carries the trading cost (the fourth component); λ1 ∈ {10, 50}, λ2 ∈ {0.5, 1.0}.
- **R3 — relative to buy-and-hold**: 100 · [ln(1 + R^p_t) − ln(1 + R^tr_t)].
- **IMIT — imitation reward (comparison arm only, Goluža et al. 2024; not in the selection set):** oracle
  labels y_t ∈ {0, 1} on the training window from the optimal trend-labelling dynamic programme (Kovačević
  et al. 2023): maximise Σ_t [y_t · ln(1 + R^tr_{t+1}) + (1 − y_t) · ln(1 + r_f)] − ϑ · Σ_t |y_t − y_{t−1}|
  over the two states, backtracked from the end. Oracle return R^o_{t+1} = y_t · R^tr_{t+1} + (1 − y_t) · r_f.
  Reward 100 · [ln(1 + R^p_{t+1}) − ln(1 + R^o_{t+1})] (the agent's profit minus the expert's, their eq. 10).
  ϑ ∈ {20, 50, 100} basis points is chosen in P4 (E4) with the selected learning rate.

## 6. PPO Agent

- Stable-Baselines3 PPO, MlpPolicy, net_arch [128, 128], tanh; action space Box(−1, 1), e = (a + 1)/2
  clipped to [0, 1]; 8 environments (DummyVecEnv); n_steps 512; batch 256; 10 epochs; γ 0.99; GAE λ 0.95;
  clip 0.2; entropy coefficient 0.0; learning rate chosen in P4 from {3e−4, 1e−4} (`dev_grid.lr`), with
  linear decay to 0; 300 000 timesteps; VecNormalize on observations (frozen at evaluation).
- Every 10 000 steps evaluate the deterministic policy on the validation split; keep the checkpoint with the
  best validation Sharpe (ties → later). Log the whole validation curve (needed for H6).
- Seeds 0–4. Ensemble action e_RL = mean of the five deterministic actions; observations use the
  ensemble's previous exposure. Disagreement u_t = standard deviation of the five actions;
  RL confidence C_RL = 1 − min(1, u_t / 0.5).

### 6.1 AWS-guided curriculum (default for SMART-RL; the proposal's cold-start mechanism)
A training wrapper executes e^exec = ω · e_SA + (1 − ω) · e_RL, with e_SA from the selected STM (so in P4 the
STM is selected before the PPO grid runs, see EXPERIMENTS.md E4). After each rollout (4 096 steps),
ω = w_SA · max(0, 1 − steps / K0), K0 = 0.5 × total timesteps, and
w_SA = softmax(α_c · [tanh(Sh_RL / 2), tanh(Sh_SA / 2)])_SA with the fixed α_c = 2 (the curriculum does not use
the AWS parameters chosen in P4), where Sh_k is the annualised Sharpe ratio of expert k's virtual daily
returns (environment rules of §5) over the last 20 completed training episodes; before 20 episodes exist,
w_SA = 0.5. Reward is computed on the executed portfolio; the observation carries e^exec_{t−1}.
From K0 on, the agent acts alone (ω = 0), so the final policy matches the deployment setting.

### 6.2 Behaviour-cloning warm start (H6 comparison arm only)
Before PPO, regress the actor mean onto the STM exposure (a = 2e − 1) over the training window:
MSE, Adam lr 1e−3, 3 000 steps; then train PPO normally.

## 7. Sentiment Trading Module (STM)

- **STM-1 (risk-off rule):** z_t = (S̃_t − μ)/σ with S̃_t = cov_t · Se_loc + (1 − cov_t) · Se_glob and μ, σ from
  the training dates; e_SA = 1 − clip(κ · max(0, −z_t − z0), 0, 1), κ ∈ {0.5, 1, 2}, z0 ∈ {0, 0.5}.
- **STM-2 (logistic):** logistic regression (L2, C = 1) on [Se_loc, Se_glob, disp_loc, cov, log1p(n_loc)]
  for P(r_{t→t+5} > 0); e_SA = clip((p − p20)/(p80 − p20), 0, 1), p20 and p80 = in-sample percentiles.
- The form and parameters are chosen in P4 by mean validation Sharpe across markets, then frozen.
- STM confidence C_SA = c̄_t · cov_t (0 when the market has no news that day).

## 8. Adaptive Weighting System (AWS)

- Experts k ∈ {RL, SA}. Each keeps a virtual book with the environment rules of §5.
- μ_k = EWMA_λ(r_k − r_f), v_k = EWMA_λ((r_k − r_f)²), σ_k = sqrt(max(v_k − μ_k², 1e−10)),
  Sh_k = sqrt(252) · μ_k / σ_k, λ = 2^(−1/H).
- Score P_k = β · tanh(Sh_k / 2) + (1 − β) · C_k (the proposal's performance-plus-confidence score).
- Weights w_k = softmax(α · P)_k; a_final = w_RL · e_RL + w_SA · e_SA; band δ; executed exposure.
- Parameters chosen in P4: H ∈ {20, 60}, α ∈ {1, 2, 5}, β ∈ {0.5, 0.7, 1.0}.
- Warm start at deployment: run AWS through the validation year and carry its EWMA state into the test start.
- Decision label each day: Buy if a_final − e_{t−1} > δ, Sell if < −δ, otherwise Hold.
- Log daily: w_RL, P_RL, P_SA, C_RL, C_SA, e_RL, e_SA, a_final, executed e, decision.
- Empirical regret per market-year: max over experts of final log-wealth minus AWS final log-wealth.

## 9. Baselines and the base-paper configuration

| id | name | rule |
| --- | --- | --- |
| B1 | Buy and hold | e = 1 |
| B2 | SMA 50/200 | e = 1 if MA50 > MA200 of the index, else 0 |
| B3 | Time-series momentum 12-1 | e = 1 if log(C_{t−21}/C_{t−252}) > 0, else 0 |
| B4 | LSTM signal | `LSTM_PRICE` forecast: e = 1 if q50 at h = 5 > 0, else 0 |
| B5 | FinBERT signal | Branch B calibrated sentiment through STM-1 with its own κ, z0 chosen in P4 |
| B6 | PPO price-only | PPO with price features only (no sentiment, no MarketMind), selected reward, no curriculum, 5 seeds, no AWS |
| B7 | SAC | SB3 SAC on the SMART-RL state and reward: lr 3e−4, buffer 200 000, batch 256, τ 0.005, γ 0.99, automatic entropy, learning_starts 5 000, 200 000 steps, seeds 0–2, ensemble mean, no AWS |
| B8 | Static hybrid | 0.5 · e_RL + 0.5 · e_SA with the band (fixed weights instead of AWS) |
| BASE | Base paper (SenT-In) as a trading signal | e = 1 if `SENTIN`'s out-of-sample P(up) > 0.5, else 0 (§9.1) |

All baselines use the same environment rules (§5): costs, band, execution timing.

### 9.1 SENTIN — reproduction of the base paper, and the BASE method
The paper leaves several details open; the choices below fill them and are reported as such.
- Sentiment index (paper §4.1, §4.3, §4.5): Branch A probabilities (GRU→CNN on CBOW vectors, §3.1–3.2;
  uncalibrated, seed-averaged); per headline s = p_pos − p_neg; daily index S_t = mean of s over the market's
  local headlines of the session (0 when there is none).
- Inputs per day (paper §4.2): open, high, low, close, volume (masked when missing) and S_t, scaled with the
  q01/q99 min-max of §2.2 ("normalise ... by removing quantiles"). Look-back 96 rows.
- Sentiment-awareness layer (paper §4.4, §4.6): embeddings e_i = Linear_i(x_i) ∈ R^16 for each input i at each
  step and e_s for S_t; activation unit w_i = σ(MLP_16([e_i; e_s; e_i ⊙ e_s])) (one hidden layer of 16, ReLU),
  w_s from (e_s, e_s); the weighted inputs w_i · x_i go to the Informer.
- Informer (paper §3.2, §4.7): encoder–decoder; 2 encoder layers with ProbSparse attention (factor 5) and a
  distilling layer, 1 decoder layer with cross-attention; d_model 64, 4 heads, d_ff 128, dropout 0.1;
  decoder input = the last 48 steps plus one zero placeholder; Linear(64→1) on the placeholder → logit of
  P(C_{t+1} > C_t).
- Training (paper Table 3: batch 256, lr 1e−3, SGD, MSE, 100 epochs): batch 256, up to 100 epochs, early stopping
  on validation AUC (patience 10). Optimiser and loss are chosen once, in E13 on the CAUSAL validation split,
  between {SGD lr 1e−3 + MSE on the sigmoid output (as published), Adam lr 1e−3 + binary cross-entropy}, by
  validation AUC, so the base model gets its stronger setting; the choice is frozen for all its uses.
- `SENTIN_SF`: the same model with SentiFlow's calibrated daily index S_loc (§3.9) in place of S_t
  (E13 only; it shows SentiFlow's effect inside the base paper's own model).
- BASE method (E5, E12): SENTIN annual refits (§4.7) give an out-of-sample P(up) for every day; e_t = 1 if
  P(up) > 0.5, else 0, traded with the environment rules of §5 (costs, band, timing).

## 10. Ablations

| id | removes | how |
| --- | --- | --- |
| A1 | sentiment | PPO state without sentiment features and with `MM_NOSENT` forecasts; no STM, therefore no AWS |
| A2 | MarketMind | PPO state without forecast features; STM and AWS kept |
| A3 | AWS | SMART-RL's PPO ensemble alone (no STM blending) |
| A4 | confidence term | AWS with β = 1 |
| A5 | curriculum | PPO trained without the curriculum (plain), STM and AWS kept |

## 11. Evaluation metrics

- Daily net return series per method and market; cumulative return Π(1 + R) − 1; annualised return
  (Π(1 + R))^(252/T) − 1; annualised volatility; Sharpe = mean(R − r_f)/std(R − r_f) · sqrt(252);
  Sortino with downside deviation of min(R − r_f, 0); maximum drawdown; Calmar = annualised return / |MDD|;
  turnover = mean |Δe| per day; win rate = share of invested days (e_{t−1} > 0) with R > r_f.
- **Primary endpoint — COMBINED:** equal-weight portfolio of the four market sleeves, daily on the union
  calendar (a sleeve's return is 0 on its market's holidays), rebalanced daily to 1/4, no extra cost.
- Breakdowns: per market, per test year, per market-year (20 blocks), per regime.
- Regimes (index level, causal): trailing 126-day index return > +5 % bull, < −5 % bear, otherwise sideways.

## 12. Statistical tests

- **Sharpe difference:** Ledoit & Wolf (2008) HAC test (Parzen kernel, Andrews automatic bandwidth),
  one-sided for superiority; robustness: studentised circular block bootstrap, block 10, 5 000 resamples.
- **H1:** SMART-RL versus B2–B7 on COMBINED; Holm-adjusted one-sided LW p-values for Sharpe; differences in
  MDD and cumulative return with stationary-bootstrap 95 % intervals (Politis & Romano 1994, mean block 10,
  5 000 resamples).
- **H7:** intersection–union test: SMART-RL's Sharpe significantly higher than each of B1–B8 and BASE
  (one-sided LW p < 0.05 for every one). Supporting evidence: Hansen (2005) SPA with SMART-RL as the
  benchmark (`arch.bootstrap.SPA`, loss = −daily excess return), Model Confidence Set at 90 %
  (`arch.bootstrap.MCS`) over {SMART-RL, B1–B8, BASE}, Friedman test and Nemenyi critical difference over the
  20 market-years (Sharpe ranks, α = 0.05).
- **DSR:** Bailey & López de Prado (2014) deflated Sharpe ratio of SMART-RL on COMBINED, with N = number of
  configurations in `dev_trials.jsonl` and the variance of their development-validation Sharpe ratios.
- **PBO:** combinatorially symmetric cross-validation (Bailey et al. 2017), S = 16 blocks, on the P4 trials
  (development validation + development test days).
- **H2:** McNemar exact test on daily decision correctness (full SMART-RL versus A1) and a paired bootstrap
  interval for the macro-F1 difference; per market and pooled.
- **H5:** placebo: for each market-year, 1 000 random exposure paths (a random walk on [0, 1] whose mean
  absolute step equals the PPO ensemble's mean |Δe|, same band and costs); p = share of placebo Sharpe ≥ PPO
  Sharpe. Also PPO versus B1 on MDD.
- **H6:** arms CURR (selected reward + curriculum, = SMART-RL's PPO), PLAIN (selected reward, no curriculum,
  = A5), BC (behaviour-cloning warm start + selected reward), IMIT (imitation reward of §5.1 with the selected ϑ,
  no curriculum; Goluža et al. 2024). Per (market, fold, seed) unit: area under the validation-Sharpe learning
  curve; steps until the validation Sharpe first reaches the STM's validation Sharpe (capped at the budget);
  mean validation MDD in the first 20 % of training. Paired Wilcoxon signed-rank tests: CURR vs PLAIN,
  BC vs PLAIN, IMIT vs PLAIN, CURR vs IMIT (units matched on market, fold and seed 0–2).
  Stability: test turnover and daily decision flip rate of SMART-RL versus A3, paired over market-years.
- Multiple testing: Holm within each hypothesis family; for H6, Holm over the three metrics within each pair
  (`h6.tests.<metric>.<pair>.p_holm`).
- Verdict rules (fixed before P5; the thesis applies them to `results.json`): H1 supported when the Holm-adjusted
  one-sided LW p < 0.05 against all of B2–B7 (partially supported when against some); H2 when, on POOLED, SMART-RL's
  accuracy is higher with McNemar p < 0.05 or the lower bound of the F1-difference interval is above 0; H3 when,
  on the pooled test set, SF has a higher macro-F1 than both A and B with median McNemar p < 0.05 for each;
  H4 when MM's h = 20 pinball loss is lower than LSTM's and GRU's with DM p < 0.05 for each; H5 when the placebo
  p < 0.05 in all four markets; H6 when CURR beats PLAIN with `p_holm` < 0.05 in the expected direction in at
  least two of the three metrics (higher AULC, fewer steps, smaller early MDD); H7 when `tests.h7.iut_all_significant`.

## 13. Decision accuracy (H2)

Forward 5-day excess return x_t = Σ_{i=1..5} (r^tr_{t+i} − r_f); threshold τ_t = 0.5 · standard deviation of
5-day excess returns over the previous 252 days (past data only). Ground truth: Up if x_t > τ_t, Down if
x_t < −τ_t, otherwise Flat. Correct pairs: (Buy, Up), (Sell, Down), (Hold, Flat). Report accuracy,
macro-F1 over the three classes, and directional hit rate on Buy and Sell days.

## 14. Robustness

- Costs 0, 5, 10 (base) and 20 basis points (re-evaluate frozen policies; no retraining).
- Execution at the close of t+1 instead of t.
- Regime breakdown (§11).
- High versus low news coverage days (cov ≥ 0.5 versus < 0.5) for H2 metrics.

## 15. Comparison with published work

- **E1-b (PhraseBank):** SentiFlow and Branch B on `phrasebank_all` and `phrasebank50` with Araci's split
  (64/16/20, §2.4), for comparison with the numbers published by Araci (2019): all data 0.86 accuracy / 0.84 F1,
  100 % agreement 0.97 / 0.95 (his Table 2; added in the thesis, not by code).
- **E3 (LTSF benchmark):** the MarketMind backbone (channel-independent patches + ProbSparse encoder with
  distilling, ProbSparse factor 5; no sentiment stream, no market embedding) on ETTh1 and ETTm1 (from the public ETDataset
  repository), standard 12/4/4-month splits (ETTm ×4 rows), look-back 336, patch 16, stride 8, horizons
  96/192/336/720, standardised data, MSE and MAE, seeds 0–2; d 128, 16 heads, d_ff 256, dropout 0.2,
  3 encoder layers with distilling, lr 1e−4, batch 128, up to 100 epochs (patience 10), PatchTST-style flatten
  head. Published Informer, PatchTST and DLinear numbers are added to the thesis from the papers, not by code.
- **BASE in trading (E5, E12):** the base paper's model as a trading signal on exactly our protocol (§9.1).
- **E13 — the base paper's own task:** next-day direction y_t = 1[C_{t+1} > C_t] of each index, on the window
  from the first common date of the four index series to 2022-03-31 (the paper used 2005-01-01 → 2022-03-31;
  `^NSEI` starts in 2007 on Yahoo Finance). Three protocols:
  - `PUB_R` (as published, random split): calendar-day rows with non-trading days forward-filled (the paper's
    6 298 rows for 2005-01-01 → 2022-03-31 are calendar days, and it fills holidays with the previous values);
    news by UTC calendar date; scaling fitted on all rows; 80/20 split of the samples in random order
    (seed 2026); 10 % of the training samples, at random, for early stopping.
  - `PUB_C` (as published, chronological split): the same rows, news and scaling; first 80 % of the samples
    for training (its last 10 % for early stopping), last 20 % for test.
  - `CAUSAL` (this thesis): trading days only; news assigned by session (§2.3); scaling fitted on training
    rows; first 80 % of trading days for training (last 10 % of them for validation), last 20 % for test.
  Models: `SVM` (RBF on the flattened last 10 rows, C ∈ {0.1, 1, 10} by validation AUC), `CNN` (two Conv1d
  layers, 32 filters, kernel 3, global max-pool, linear), `LSTM` and `GRU` (two layers, hidden 64), `SENTIN`,
  `SENTIN_SF` (§9.1) — all on the SENTIN inputs — and `MM_DIR` (MarketMind trained on the CAUSAL split;
  score = q50 at h = 1; CAUSAL only). Seeds 0–2 for neural models. Metrics: accuracy, F1 (positive class = up),
  AUC-ROC, PR-AUC (average precision), plus the share of up days in the test labels. The paper's published
  numbers (its Tables 4–7) are added in the thesis, not by code.
  Overlap with the trading protocol: on 2020-01-01 → 2022-03-31, the same four direction metrics for SMART-RL
  and BASE (prediction = 1[e_t > 0.5], score = e_t, from the E5 test folds) and for SENTIN under CAUSAL
  (its E13 test predictions restricted to that window).

## 16. Outputs

- `results/results.json` — keys in RESULTS_SCHEMA.md; floats unrounded; missing values as null with a
  `notes` entry explaining why.
- `results/tables/<ID>.csv` — one per table ID in RESULTS_SCHEMA.md.
- `results/figures/<ID>.png` (300 dpi) and `.pdf` — one per figure ID, Persian labels (PROJECT_RULES.md rule 9),
  at the sizes listed under Figures in RESULTS_SCHEMA.md.
- `results/logs/aws_daily_<MKT>_<YEAR>.parquet`, `results/logs/decisions_<method>_<MKT>.parquet`,
  `results/logs/learning_curves.parquet`.
