### Confirmation 2020-2024, COMBINED (equal-weight four-market portfolio)

| method | sharpe | sortino | ann_return | ann_vol | mdd | calmar | cum_return | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SMARTRL | 0.712 | 0.955 | 7.8 % | 8.3 % | -17.6 % | 0.447 | 47.8 % | 0.018 |
| B1_BH | 0.646 | 0.860 | 9.9 % | 12.8 % | -29.7 % | 0.332 | 62.9 % | 0.000 |
| B2_SMA | 0.279 | 0.355 | 4.2 % | 9.1 % | -25.1 % | 0.166 | 23.6 % | 0.007 |
| B3_TSMOM | 0.157 | 0.199 | 3.1 % | 10.1 % | -27.7 % | 0.112 | 17.1 % | 0.032 |
| B4_LSTM | 0.682 | 0.922 | 10.1 % | 12.3 % | -28.6 % | 0.352 | 64.4 % | 0.018 |
| B5_FINBERT | 0.725 | 0.977 | 9.6 % | 10.7 % | -20.4 % | 0.468 | 60.5 % | 0.009 |
| B6_PPOPRICE | 0.492 | 0.646 | 4.8 % | 5.8 % | -14.4 % | 0.331 | 27.3 % | 0.031 |
| B7_SAC | 0.320 | 0.412 | 4.1 % | 7.2 % | -17.5 % | 0.234 | 23.1 % | 0.101 |
| B8_STATIC | 0.703 | 0.944 | 7.9 % | 8.5 % | -18.2 % | 0.435 | 48.2 % | 0.016 |
| BASE | 0.735 | 1.004 | 8.8 % | 9.4 % | -19.0 % | 0.463 | 54.6 % | 0.063 |

### Sharpe per market (confirmation 2020-2024)

| method | SP500 | FTSE100 | NIFTY50 | SSE |
| --- | --- | --- | --- | --- |
| SMARTRL | 0.648 | 0.274 | 0.940 | 0.114 |
| B1_BH | 0.651 | 0.285 | 0.769 | 0.085 |
| B2_SMA | 0.538 | 0.046 | 0.385 | -0.386 |
| B3_TSMOM | 0.447 | -0.172 | 0.093 | -0.045 |
| B4_LSTM | 0.680 | 0.409 | 0.570 | 0.201 |
| B5_FINBERT | 0.688 | 0.213 | 0.986 | 0.138 |
| B6_PPOPRICE | 0.368 | 0.191 | 0.548 | 0.372 |
| B7_SAC | 0.503 | -0.168 | 0.573 | -0.123 |
| B8_STATIC | 0.703 | 0.232 | 0.885 | 0.121 |
| BASE | 0.651 | 0.246 | 0.848 | -0.127 |

### Maximum drawdown per market (confirmation 2020-2024)

| method | SP500 | FTSE100 | NIFTY50 | SSE |
| --- | --- | --- | --- | --- |
| SMARTRL | -19.0 % | -22.5 % | -24.7 % | -20.9 % |
| B1_BH | -33.8 % | -34.3 % | -38.3 % | -27.3 % |
| B2_SMA | -33.8 % | -21.9 % | -35.8 % | -19.8 % |
| B3_TSMOM | -33.8 % | -34.3 % | -38.3 % | -13.2 % |
| B4_LSTM | -33.8 % | -33.6 % | -38.3 % | -22.2 % |
| B5_FINBERT | -24.9 % | -32.7 % | -25.5 % | -28.0 % |
| B6_PPOPRICE | -14.2 % | -17.6 % | -19.5 % | -7.4 % |
| B7_SAC | -16.0 % | -27.1 % | -20.2 % | -12.9 % |
| B8_STATIC | -19.5 % | -25.7 % | -24.5 % | -20.7 % |
| BASE | -33.8 % | -11.5 % | -38.3 % | -32.0 % |

### Ledoit-Wolf Sharpe-difference tests, SMART-RL minus method (COMBINED)

| versus | Sharpe diff | p (HAC) | p (boot) | p Holm (H1) |
| --- | --- | --- | --- | --- |
| B1_BH | 0.067 | 0.248 | 0.207 | n/a |
| B2_SMA | 0.434 | 0.060 | 0.051 | 0.240 |
| B3_TSMOM | 0.556 | 0.011 | 0.005 | 0.064 |
| B4_LSTM | 0.030 | 0.398 | 0.354 | 0.796 |
| B5_FINBERT | -0.012 | 0.542 | 0.572 | 0.796 |
| B6_PPOPRICE | 0.220 | 0.119 | 0.093 | 0.358 |
| B7_SAC | 0.393 | 0.023 | 0.016 | 0.117 |
| B8_STATIC | 0.009 | 0.421 | 0.420 | n/a |
| BASE | -0.023 | 0.553 | 0.541 | n/a |
| A1_NOSENT | 0.342 | 0.054 | 0.041 | n/a |
| A2_NOMM | 0.059 | 0.141 | 0.127 | n/a |
| A3_NOAWS | 0.113 | 0.150 | 0.110 | n/a |
| A4_NOCONF | 0.000 | 0.500 | 1.000 | n/a |
| A5_NOCURR | 0.049 | 0.163 | 0.145 | n/a |

### H1 secondary: MDD and cumulative-return differences (stationary bootstrap 95 %)

| versus | MDD diff | cum-return diff |
| --- | --- | --- |
| B2_SMA | 0.076 [-0.029, 0.155] | 0.242 [-0.095, 0.616] |
| B3_TSMOM | 0.102 [-0.011, 0.189] | 0.307 [-0.004, 0.637] |
| B4_LSTM | 0.111 [0.015, 0.142] | -0.166 [-0.712, 0.101] |
| B5_FINBERT | 0.029 [-0.003, 0.094] | -0.127 [-0.536, 0.081] |
| B6_PPOPRICE | -0.032 [-0.099, -0.003] | 0.205 [-0.048, 0.601] |
| B7_SAC | -0.000 [-0.053, 0.042] | 0.247 [0.020, 0.595] |

### Ablations, COMBINED

| method | sharpe | ann_return | mdd | turnover |
| --- | --- | --- | --- | --- |
| SMARTRL | 0.712 | 7.8 % | -17.6 % | 0.018 |
| A1_NOSENT | 0.371 | 4.3 % | -19.1 % | 0.048 |
| A2_NOMM | 0.653 | 7.5 % | -19.8 % | 0.019 |
| A3_NOAWS | 0.599 | 5.9 % | -15.2 % | 0.036 |
| A4_NOCONF | 0.712 | 7.8 % | -17.6 % | 0.018 |
| A5_NOCURR | 0.663 | 7.6 % | -20.4 % | 0.025 |

### Joint tests

| key | value |
| --- | --- |
| tests.h7.iut_all_significant | False |
| tests.h7.max_p | 0.553 |
| tests.spa.p_consistent | 0.430 |
| tests.spa.p_lower | 0.201 |
| tests.spa.p_upper | 0.444 |
| tests.mcs.included | ['SMARTRL', 'B1_BH', 'B2_SMA', 'B3_TSMOM', 'B4_LSTM', 'B5_FINBERT', 'B6_PPOPRICE', 'B7_SAC', 'B8_STATIC', 'BASE'] |
| tests.friedman.chi2 | 30.746 |
| tests.friedman.p | 0.000 |
| tests.nemenyi.cd | 3.029 |
| tests.nemenyi.rank.SMARTRL | 4.350 |
| tests.dsr.sr_annual | 0.712 |
| tests.dsr.dsr | 0.565 |
| tests.dsr.n_trials | 50.000 |
| tests.pbo.pbo | 0.152 |

### Decision accuracy (H2)

| market | SMART-RL acc | SMART-RL F1 | A1 acc | A1 F1 | McNemar p | F1 diff [95 %] |
| --- | --- | --- | --- | --- | --- | --- |
| SP500 | 0.395 | 0.295 | 0.382 | 0.307 | 0.415 | -0.012 [-0.045, 0.020] |
| FTSE100 | 0.423 | 0.294 | 0.392 | 0.337 | 0.061 | -0.042 [-0.077, -0.008] |
| NIFTY50 | 0.407 | 0.268 | 0.386 | 0.328 | 0.189 | -0.060 [-0.094, -0.027] |
| SSE | 0.404 | 0.288 | 0.432 | 0.348 | 0.077 | -0.061 [-0.095, -0.026] |
| POOLED | 0.407 | 0.287 | 0.398 | 0.330 | 0.215 | -0.044 [-0.060, -0.027] |

### Placebo (H5)

| market | PPO Sharpe | placebo mean | placebo p95 | p | PPO MDD | B&H MDD |
| --- | --- | --- | --- | --- | --- | --- |
| SP500 | 0.636 | 0.527 | 0.766 | 0.216 | -15.3 % | -33.8 % |
| FTSE100 | 0.236 | 0.260 | 0.549 | 0.528 | -18.9 % | -34.3 % |
| NIFTY50 | 0.714 | 0.549 | 0.710 | 0.043 | -21.9 % | -38.3 % |
| SSE | 0.026 | -0.088 | 0.100 | 0.192 | -12.3 % | -27.3 % |

### Learning curves (H6)

| arm | AULC | steps to STM Sharpe | early MDD | test Sharpe |
| --- | --- | --- | --- | --- |
| CURR | 0.290 | 72000.000 | -0.076 | 0.653 |
| PLAIN | 0.291 | 74166.667 | -0.080 | 0.607 |
| BC | 0.568 | 142333.333 | -0.126 | 0.520 |
| IMIT | 0.423 | 68000.000 | -0.082 | 0.812 |

| metric, pair | p Holm |
| --- | --- |
| aulc CURR_vs_PLAIN | 1.000 |
| steps_to_thr CURR_vs_PLAIN | 1.000 |
| early_mdd CURR_vs_PLAIN | 0.159 |
| aulc BC_vs_PLAIN | 0.000 |
| steps_to_thr BC_vs_PLAIN | 0.001 |
| early_mdd BC_vs_PLAIN | 0.000 |
| aulc IMIT_vs_PLAIN | 0.153 |
| steps_to_thr IMIT_vs_PLAIN | 1.000 |
| early_mdd IMIT_vs_PLAIN | 1.000 |
| aulc CURR_vs_IMIT | 0.011 |
| steps_to_thr CURR_vs_IMIT | 0.639 |
| early_mdd CURR_vs_IMIT | 0.230 |

### AWS weights

| market | w_RL mean | w_RL sd | regret |
| --- | --- | --- | --- |
| SP500 | 0.527 | 0.114 | 0.048 |
| FTSE100 | 0.527 | 0.110 | 0.023 |
| NIFTY50 | 0.502 | 0.110 | 0.028 |
| SSE | 0.506 | 0.132 | 0.027 |

### Robustness: COMBINED Sharpe by cost (bps) and with one-day delay

| method | 0 | 5 | 10 | 20 | delay 1 day |
| --- | --- | --- | --- | --- | --- |
| SMARTRL | 0.768 | 0.744 | 0.712 | 0.663 | 0.760 |
| B1_BH | 0.646 | 0.646 | 0.646 | 0.646 | 0.646 |
| B6_PPOPRICE | 0.615 | 0.555 | 0.492 | 0.367 | 0.499 |
| B8_STATIC | 0.746 | 0.724 | 0.703 | 0.660 | 0.711 |
| BASE | 0.894 | 0.814 | 0.735 | 0.576 | 0.742 |

### Forward test (2025-01 to 2026-09), COMBINED

| method | sharpe | mdd | cum_return | turnover |
| --- | --- | --- | --- | --- |
| SMARTRL | 1.045 | -6.6 % | 16.4 % | 0.013 |
| B1_BH | 1.022 | -9.9 % | 20.8 % | 0.000 |
| B2_SMA | 0.824 | -9.1 % | 14.8 % | 0.005 |
| B3_TSMOM | 1.073 | -8.9 % | 20.5 % | 0.012 |
| B4_LSTM | 0.905 | -9.9 % | 18.2 % | 0.011 |
| B5_FINBERT | 1.072 | -9.3 % | 21.0 % | 0.004 |
| B6_PPOPRICE | 1.150 | -3.7 % | 10.2 % | 0.012 |
| B7_SAC | 0.271 | -7.6 % | 6.3 % | 0.077 |
| B8_STATIC | 0.940 | -6.5 % | 14.1 % | 0.010 |
| BASE | 0.815 | -9.0 % | 15.5 % | 0.104 |

news-day share: SP500 0.284, FTSE100 0.249, NIFTY50 0.224, SSE 0.285

### Supplementary sub-period 2020-2023 (pre-registered), COMBINED

| method | sharpe | mdd | cum_return |
| --- | --- | --- | --- |
| SMARTRL | 0.588 | -17.6 % | 32.5 % |
| B1_BH | 0.536 | -29.7 % | 41.6 % |
| B2_SMA | 0.128 | -25.1 % | 12.0 % |
| B3_TSMOM | -0.016 | -27.7 % | 5.1 % |
| B4_LSTM | 0.572 | -28.6 % | 42.9 % |
| B5_FINBERT | 0.571 | -20.4 % | 38.2 % |
| B6_PPOPRICE | 0.391 | -14.4 % | 19.2 % |
| B7_SAC | 0.156 | -17.5 % | 12.7 % |
| B8_STATIC | 0.574 | -18.2 % | 32.5 % |
| BASE | 0.649 | -19.0 % | 38.5 % |

| versus | Sharpe diff | p (HAC) | p Holm |
| --- | --- | --- | --- |
| B1_BH | 0.051 | 0.318 | n/a |
| B2_SMA | 0.460 | 0.066 | 0.265 |
| B3_TSMOM | 0.604 | 0.011 | 0.063 |
| B4_LSTM | 0.016 | 0.452 | 0.901 |
| B5_FINBERT | 0.016 | 0.450 | 0.901 |
| B6_PPOPRICE | 0.197 | 0.158 | 0.473 |
| B7_SAC | 0.431 | 0.024 | 0.119 |
| B8_STATIC | 0.013 | 0.400 | n/a |
| BASE | -0.061 | 0.628 | n/a |

### Base-paper task overlap (AUC of next-day direction)

| market | SMART-RL | BASE | SENTIN (causal) |
| --- | --- | --- | --- |
| SP500 | 0.484 | 0.500 | 0.513 |
| FTSE100 | 0.501 | 0.505 | 0.515 |
| NIFTY50 | 0.570 | 0.528 | 0.487 |
| SSE | 0.511 | 0.497 | 0.511 |

### Compute

| key | value |
| --- | --- |
| cpu_hours | 12.989 |
| gpu_hours | 20.076 |
| n_runs | 683.000 |
