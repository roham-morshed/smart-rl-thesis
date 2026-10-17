"""Portfolio book and rewards shared by the trading environment, the evaluators and AWS (SPEC §5, §5.1).

Conventions (one market, daily):
- `ret_tr[t]` is the total-return of the index from the close of t-1 to the close of t.
- A decision at the close of t sets exposure e_t; it earns R^p_{t+1} = e_t·R^tr_{t+1} + (1-e_t)·r_f − c·|e_t − e_{t−1}|.
- The no-trade band keeps e_{t-1} when |target − e_{t−1}| <= δ.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

RF_DAILY = 1.02 ** (1 / 252) - 1


def apply_band(target: float, prev: float, band: float) -> float:
    return prev if abs(target - prev) <= band else target


@dataclass
class RewardSpec:
    name: str                      # R1 | R2 | R3 | IMIT
    lambda1: float = 0.0           # R2
    lambda2: float = 0.0           # R2
    scale: float = 100.0           # R2, R3, IMIT
    eta: float = 0.01              # R1
    a0: float = 0.0                # R1
    b0: float = 1e-4               # R1
    clip: float = 3.0              # R1

    @property
    def cfg_id(self) -> str:
        if self.name == "R2":
            return f"R2_l{self.lambda1:g}_d{self.lambda2:g}"
        return self.name


@dataclass
class Book:
    """Running state of one strategy's virtual book: exposure, recent returns, wealth, drawdown, DSR moments."""
    rf: float = RF_DAILY
    cost: float = 0.001
    band: float = 0.05
    window: int = 20
    e_prev: float = 1.0
    wealth: float = 1.0
    peak: float = 1.0
    dd_mag: float = 0.0            # 1 - W/peak >= 0
    A: float = 0.0
    B: float = 1e-4
    hist: deque = field(default_factory=deque)

    def reset(self, e_init: float, prehistory: np.ndarray, reward: RewardSpec | None = None) -> None:
        self.e_prev = float(e_init)
        self.wealth, self.peak, self.dd_mag = 1.0, 1.0, 0.0
        self.hist = deque((float(x) for x in prehistory[-self.window:]), maxlen=self.window)
        if reward is not None:
            self.A, self.B = reward.a0, reward.b0

    def step(self, target: float, r_tr_next: float) -> tuple[float, float]:
        """Apply the band, book the next-day net return. Returns (executed exposure, net return)."""
        e = apply_band(float(np.clip(target, 0.0, 1.0)), self.e_prev, self.band)
        r = e * r_tr_next + (1.0 - e) * self.rf - self.cost * abs(e - self.e_prev)
        self.e_prev = e
        self.hist.append(r)
        self.wealth *= 1.0 + r
        self.peak = max(self.peak, self.wealth)
        self.dd_mag = 1.0 - self.wealth / self.peak
        return e, r

    def extras(self) -> np.ndarray:
        """[e_{t-1}, 20-day strategy log return, 20-day strategy volatility (annualised), current drawdown]."""
        h = np.fromiter(self.hist, dtype=np.float64) if self.hist else np.zeros(1)
        ret20 = float(np.sum(np.log1p(h)))
        vol20 = float(np.std(h) * math.sqrt(252)) if len(h) > 1 else 0.0
        return np.array([self.e_prev, ret20, vol20, -self.dd_mag], dtype=np.float32)


def reward_value(spec: RewardSpec, book: Book, r_p: float, r_tr_next: float, dd_prev: float,
                 r_oracle: float | None = None) -> float:
    """Reward for the step whose net portfolio return is r_p (already booked into `book`)."""
    x = r_p - book.rf
    if spec.name == "R1":
        A, B = book.A, book.B
        dA, dB = x - A, x * x - B
        denom = (B - A * A)
        d = (B * dA - 0.5 * A * dB) / denom ** 1.5 if denom > 1e-12 else 0.0
        book.A = A + spec.eta * dA
        book.B = B + spec.eta * dB
        return float(np.clip(d, -spec.clip, spec.clip))
    if spec.name == "R2":
        dd_inc = max(0.0, book.dd_mag - dd_prev)
        return spec.scale * (math.log1p(r_p) - spec.lambda1 * min(x, 0.0) ** 2 - spec.lambda2 * dd_inc)
    if spec.name == "R3":
        return spec.scale * (math.log1p(r_p) - math.log1p(r_tr_next))
    if spec.name == "IMIT":
        assert r_oracle is not None
        return spec.scale * (math.log1p(r_p) - math.log1p(r_oracle))
    raise ValueError(spec.name)


def oracle_labels(ret_tr_next: np.ndarray, rf: float, theta: float) -> np.ndarray:
    """Optimal two-state trend labels (Kovačević et al. 2023) for the IMIT reward (SPEC §5.1).

    `ret_tr_next[t]` = R^tr_{t+1} for the decision days of the training window. Maximises
    Σ y_t·ln(1+R^tr_{t+1}) + (1−y_t)·ln(1+r_f) − θ·Σ|y_t − y_{t−1}| by dynamic programming, backtracked from the end.
    """
    n = len(ret_tr_next)
    g_in = np.log1p(ret_tr_next)
    g_out = math.log1p(rf)
    v = np.zeros((n, 2))
    back = np.zeros((n, 2), dtype=np.int8)
    v[0, 0], v[0, 1] = g_out, g_in[0]
    for t in range(1, n):
        for s, gain in ((0, g_out), (1, g_in[t])):
            stay, switch = v[t - 1, s], v[t - 1, 1 - s] - theta
            if stay >= switch:
                v[t, s], back[t, s] = stay + gain, s
            else:
                v[t, s], back[t, s] = switch + gain, 1 - s
    y = np.zeros(n, dtype=np.int8)
    y[-1] = int(v[-1, 1] > v[-1, 0])
    for t in range(n - 1, 0, -1):
        y[t - 1] = back[t, y[t]]
    return y
