"""Rule-based target exposures of the baselines (SPEC §9); all are decided at the close of day t from data <= t
and traded with the shared environment rules by smartrl.env.evaluate.path_from_targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def b1_buy_and_hold(n: int) -> np.ndarray:
    return np.ones(n)


def b2_sma(close: np.ndarray, fast: int = 50, slow: int = 200) -> np.ndarray:
    """e = 1 if MA50 > MA200 of the index, else 0 (index history before the fold is allowed for the averages)."""
    c = pd.Series(close)
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float).to_numpy()


def b3_tsmom(close: np.ndarray, lookback: int = 252, skip: int = 21) -> np.ndarray:
    """e = 1 if log(C_{t−21} / C_{t−252}) > 0, else 0."""
    c = pd.Series(close)
    return (np.log(c.shift(skip) / c.shift(lookback)) > 0).astype(float).to_numpy()


def b4_lstm(q50_h5: np.ndarray) -> np.ndarray:
    """LSTM_PRICE forecast: e = 1 if q50 at h = 5 > 0, else 0."""
    return (np.nan_to_num(q50_h5) > 0).astype(float)


def base_sentin(p_up: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """BASE (SenT-In as a trading signal): e = 1 if out-of-sample P(up) > 0.5, else 0."""
    return (np.nan_to_num(p_up) > threshold).astype(float)


def b8_static(e_rl: np.ndarray, e_sa: np.ndarray, w_rl: float = 0.5) -> np.ndarray:
    return w_rl * e_rl + (1 - w_rl) * e_sa
