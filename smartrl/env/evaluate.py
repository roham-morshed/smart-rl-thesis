"""Deterministic evaluation over a split (SPEC §5): day a is held at e_init; the exposure decided at the close of t
(after the band) earns the return of t+1; costs c·|Δe|. `delay=1` executes each decision one day later
(robustness, SPEC §14): the exposure decided at t−1 is executed at the close of t."""
from __future__ import annotations

import numpy as np

from .portfolio import RF_DAILY, Book


def path_from_targets(targets: np.ndarray, ret_tr: np.ndarray, a: int, b: int, cost: float = 0.001,
                      band: float = 0.05, e_init: float = 1.0, delay: int = 0, rf: float = RF_DAILY) -> dict:
    """targets: target exposure decided at each day's close (full-length array). Returns r, held, e over a..b."""
    book = Book(rf=rf, cost=cost, band=band)
    book.reset(e_init, np.zeros(1))
    n = b - a + 1
    r, held, e = np.zeros(n), np.zeros(n), np.zeros(n)
    r[0], held[0] = e_init * ret_tr[a] + (1 - e_init) * rf, e_init
    for i, t in enumerate(range(a, b)):
        tgt = targets[t - delay] if t - delay >= a else e_init
        e[i], r[i + 1] = book.step(tgt, ret_tr[t + 1])
        held[i + 1] = e[i]
    e[-1] = book.e_prev
    return {"r": r, "held": held, "e": e}


class PolicyEnsemble:
    """Deterministic SB3 policies with their frozen VecNormalize statistics."""

    def __init__(self, models: list, normalizers: list):
        self.models, self.norms = models, normalizers

    def act(self, obs: np.ndarray) -> tuple[float, float, np.ndarray]:
        acts = np.array([float(m.predict(n.normalize_obs(obs[None]), deterministic=True)[0].reshape(-1)[0])
                         for m, n in zip(self.models, self.norms)])
        acts = np.clip(acts, -1.0, 1.0)
        e = np.clip((acts + 1.0) / 2.0, 0.0, 1.0)
        return float(e.mean()), float(acts.std()), e


def path_ensemble(ens: PolicyEnsemble, win: np.ndarray, window: int, ret_tr: np.ndarray, a: int, b: int,
                  cost: float = 0.001, band: float = 0.05, e_init: float = 1.0, delay: int = 0,
                  rf: float = RF_DAILY, conf_divisor: float = 0.5) -> dict:
    """The ensemble acts on its own book: observation = flattened feature window + [e_{t−1}, 20-day return,
    20-day volatility, drawdown] of that book. e_rl = mean of the deterministic exposures; disagreement u = std of
    the actions; C_RL = 1 − min(1, u / 0.5). `win[t − window + 1]` is the flattened window ending at t."""
    book = Book(rf=rf, cost=cost, band=band, window=window)
    pre = np.nan_to_num(ret_tr[max(0, a - window + 1): a + 1])
    book.reset(e_init, pre)
    n = b - a + 1
    out = {k: np.zeros(n) for k in ("r", "held", "e", "e_rl", "u", "c_rl")}
    out["r"][0], out["held"][0] = e_init * ret_tr[a] + (1 - e_init) * rf, e_init
    pending = e_init
    for i, t in enumerate(range(a, b + 1)):
        obs = np.concatenate([win[t - window + 1], book.extras()]).astype(np.float32)
        e_rl, u, _ = ens.act(obs)
        out["e_rl"][i], out["u"][i], out["c_rl"][i] = e_rl, u, 1.0 - min(1.0, u / conf_divisor)
        if t < b:
            tgt = e_rl if delay == 0 else pending
            pending = e_rl
            out["e"][i], out["r"][i + 1] = book.step(tgt, ret_tr[t + 1])
            out["held"][i + 1] = out["e"][i]
        else:
            out["e"][i] = book.e_prev
    return out
