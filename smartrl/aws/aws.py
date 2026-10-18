"""Adaptive Weighting System (SPEC §8).

Experts k ∈ {RL, SA} keep virtual books with the environment rules (band, costs). On each day t, after the returns
of day t are booked: μ_k = EWMA_λ(r_k − r_f), v_k = EWMA_λ((r_k − r_f)²), σ_k = sqrt(max(v_k − μ_k², 1e−10)),
Sh_k = sqrt(252)·μ_k/σ_k, λ = 2^(−1/H); P_k = β·tanh(Sh_k/2) + (1 − β)·C_k; w = softmax(α·P);
a_final = w_RL·e_RL + w_SA·e_SA, then the band. The state can be carried from the validation year into the test
(warm start).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..env.portfolio import RF_DAILY, Book


@dataclass
class AWSState:
    mu: np.ndarray = field(default_factory=lambda: np.zeros(2))
    v: np.ndarray = field(default_factory=lambda: np.full(2, 1e-4))


class AWS:
    def __init__(self, H: float, alpha: float, beta: float, rf: float = RF_DAILY, state: AWSState | None = None):
        self.lam = 2.0 ** (-1.0 / H)
        self.alpha, self.beta, self.rf = alpha, beta, rf
        self.state = state or AWSState()

    def update(self, r_rl: float, r_sa: float) -> None:
        x = np.array([r_rl, r_sa]) - self.rf
        s = self.state
        s.mu = self.lam * s.mu + (1 - self.lam) * x
        s.v = self.lam * s.v + (1 - self.lam) * x ** 2

    def sharpe(self) -> np.ndarray:
        s = self.state
        sd = np.sqrt(np.maximum(s.v - s.mu ** 2, 1e-10))
        return math.sqrt(252) * s.mu / sd

    def weights(self, c_rl: float, c_sa: float) -> tuple[np.ndarray, np.ndarray]:
        P = self.beta * np.tanh(self.sharpe() / 2) + (1 - self.beta) * np.array([c_rl, c_sa])
        z = self.alpha * P
        w = np.exp(z - z.max())
        return w / w.sum(), P


def run_aws(e_rl: np.ndarray, e_sa: np.ndarray, c_rl: np.ndarray, c_sa: np.ndarray, ret_tr: np.ndarray, a: int, b: int,
            H: float, alpha: float, beta: float, cost: float, band: float, state: AWSState | None = None,
            e_init: float = 1.0, rf: float = RF_DAILY) -> dict:
    """Run AWS over days a..b.

    e_rl, e_sa, c_rl, c_sa: expert target exposures and confidences at each day's close. Day a is held at e_init;
    at the close of day t the returns of day t update the EWMA state, the weights give a_final, the band gives the
    executed exposure, which earns the return of day t+1. Returns the daily log, the final state and the regret
    (max over experts of final log-wealth minus AWS final log-wealth)."""
    aws = AWS(H, alpha, beta, rf, state)
    books = {k: Book(rf=rf, cost=cost, band=band) for k in ("RL", "SA", "AWS")}
    for bk in books.values():
        bk.reset(e_init, np.zeros(1))
    n = b - a + 1
    L = {k: np.zeros(n) for k in ("r", "r_rl", "r_sa", "held", "w_rl", "P_rl", "P_sa", "C_rl", "C_sa", "e_rl",
                                  "e_sa", "a_final", "e", "decision")}
    r0 = e_init * ret_tr[a] + (1 - e_init) * rf
    L["r"][0] = L["r_rl"][0] = L["r_sa"][0] = r0
    L["held"][0] = e_init
    for i, t in enumerate(range(a, b + 1)):
        aws.update(L["r_rl"][i], L["r_sa"][i])
        w, P = aws.weights(c_rl[t], c_sa[t])
        a_final = w[0] * e_rl[t] + w[1] * e_sa[t]
        prev = books["AWS"].e_prev
        L["decision"][i] = 1 if a_final - prev > band else (-1 if a_final - prev < -band else 0)
        for k, v in (("w_rl", w[0]), ("P_rl", P[0]), ("P_sa", P[1]), ("C_rl", c_rl[t]), ("C_sa", c_sa[t]),
                     ("e_rl", e_rl[t]), ("e_sa", e_sa[t]), ("a_final", a_final)):
            L[k][i] = v
        if t < b:
            r_next = ret_tr[t + 1]
            L["r_rl"][i + 1] = books["RL"].step(e_rl[t], r_next)[1]
            L["r_sa"][i + 1] = books["SA"].step(e_sa[t], r_next)[1]
            e, L["r"][i + 1] = books["AWS"].step(a_final, r_next)
            L["e"][i] = L["held"][i + 1] = e
        else:
            L["e"][i] = prev if abs(a_final - prev) <= band else a_final
    L["final_state"] = aws.state
    L["regret"] = float(max(np.log1p(L["r_rl"]).sum(), np.log1p(L["r_sa"]).sum()) - np.log1p(L["r"]).sum())
    return L
