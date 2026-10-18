"""Statistical tests of SPEC §12 for daily return series."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER = 0.5772156649015329


# ---------------------------------------------------------------- Ledoit & Wolf (2008)
def _parzen(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= 0.5, 1 - 6 * a ** 2 + 6 * a ** 3, np.where(a <= 1, 2 * (1 - a) ** 3, 0.0))


def andrews_parzen_bandwidth(V: np.ndarray) -> float:
    """Andrews (1991) automatic bandwidth for the Parzen kernel from AR(1) fits of each column of V."""
    T = len(V)
    num = den = 0.0
    for j in range(V.shape[1]):
        x = V[:, j]
        rho = np.clip(np.corrcoef(x[1:], x[:-1])[0, 1] if x.std() > 0 else 0.0, -0.97, 0.97)
        s2 = np.var(x[1:] - rho * x[:-1])
        num += 4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8
        den += s2 ** 2 / (1 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    return float(2.6614 * (alpha2 * T) ** 0.2)


def hac_cov(V: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Long-run covariance of the rows of V (already centred) with Parzen weights."""
    T = len(V)
    S = bandwidth if bandwidth is not None else andrews_parzen_bandwidth(V)
    Psi = V.T @ V / T
    for j in range(1, T):
        w = float(_parzen(np.array([j / S]))[0]) if S > 0 else 0.0
        if w == 0.0:
            break
        G = V[j:].T @ V[:-j] / T
        Psi += w * (G + G.T)
    return Psi * T / (T - 4)              # small-sample adjustment of Ledoit & Wolf


def _sharpe_diff_parts(r1: np.ndarray, r2: np.ndarray):
    m1, m2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    d = m1 / np.sqrt(g1 - m1 ** 2) - m2 / np.sqrt(g2 - m2 ** 2)
    grad = np.array([g1 / (g1 - m1 ** 2) ** 1.5, -g2 / (g2 - m2 ** 2) ** 1.5,
                     -0.5 * m1 / (g1 - m1 ** 2) ** 1.5, 0.5 * m2 / (g2 - m2 ** 2) ** 1.5])
    V = np.column_stack([r1 - m1, r2 - m2, r1 ** 2 - g1, r2 ** 2 - g2])
    return d, grad, V


def lw_hac(x1: np.ndarray, x2: np.ndarray) -> dict:
    """Ledoit–Wolf (2008) HAC test of SR1 − SR2 on excess returns; one-sided p for SR1 > SR2.
    `diff` is annualised (× sqrt(252))."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    d, grad, V = _sharpe_diff_parts(x1, x2)
    se = math.sqrt(max(grad @ hac_cov(V) @ grad / len(x1), 1e-300))
    z = d / se
    return {"diff": d * math.sqrt(252), "se": se, "stat": z, "p": float(1 - stats.norm.cdf(z))}


def lw_bootstrap(x1: np.ndarray, x2: np.ndarray, block: int = 10, reps: int = 5000, seed: int = 2026) -> dict:
    """Studentised circular block bootstrap of Ledoit & Wolf (2008, §3.2); one-sided p for SR1 > SR2.
    The bootstrap standard error uses the block-sum estimator of their eq. (8)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    T = len(x1)
    d, grad, V = _sharpe_diff_parts(x1, x2)
    t_hat = d / math.sqrt(max(grad @ hac_cov(V) @ grad / T, 1e-300))
    n_blocks = int(math.ceil(T / block))
    rng = np.random.default_rng(seed)
    idx_base = np.arange(block)
    exceed = 0
    for _ in range(reps):
        starts = rng.integers(0, T, n_blocks)
        idx = ((starts[:, None] + idx_base[None, :]) % T).ravel()[:T]
        b1, b2 = x1[idx], x2[idx]
        db, gb, Vb = _sharpe_diff_parts(b1, b2)
        n_full = T // block
        sums = Vb[: n_full * block].reshape(n_full, block, 4).sum(axis=1) / math.sqrt(block)
        Psi = sums.T @ sums / n_full
        se_b = math.sqrt(max(gb @ Psi @ gb / T, 1e-300))
        if (db - d) / se_b >= t_hat:
            exceed += 1
    return {"stat": t_hat, "p": (exceed + 1) / (reps + 1)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


# ---------------------------------------------------------------- bootstrap intervals of differences
def max_dd(r: np.ndarray) -> float:
    w = np.cumprod(1 + r)
    return float((w / np.maximum.accumulate(np.concatenate([[1.0], w]))[1:] - 1).min())


def stationary_bootstrap_diff(r1: np.ndarray, r2: np.ndarray, mean_block: float = 10, reps: int = 5000,
                              seed: int = 2026) -> dict:
    """Paired stationary bootstrap (Politis & Romano 1994): 95 % intervals of MDD(1) − MDD(2) and CR(1) − CR(2)."""
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(mean_block, np.asarray(r1, float), np.asarray(r2, float), seed=seed)
    f = lambda a, b: np.array([max_dd(a) - max_dd(b), np.prod(1 + a) - np.prod(1 + b)])
    draws = np.array([f(*pos) for pos, _ in bs.bootstrap(reps)])
    est = f(np.asarray(r1, float), np.asarray(r2, float))
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return {"mdd": {"est": float(est[0]), "lo": float(lo[0]), "hi": float(hi[0])},
            "cr": {"est": float(est[1]), "lo": float(lo[1]), "hi": float(hi[1])}}


# ---------------------------------------------------------------- SPA, MCS
def spa(bench_excess: np.ndarray, others_excess: np.ndarray, reps: int = 5000, block: int = 10, seed: int = 2026) -> dict:
    """Hansen (2005) SPA with SMART-RL as benchmark; loss = −daily excess return."""
    from arch.bootstrap import SPA
    s = SPA(-np.asarray(bench_excess), -np.asarray(others_excess), reps=reps, block_size=block,
            bootstrap="stationary", seed=seed)
    s.compute()
    p = s.pvalues
    return {"p_lower": float(p["lower"]), "p_consistent": float(p["consistent"]), "p_upper": float(p["upper"])}


def mcs(losses: np.ndarray, names: list[str], size: float = 0.10, reps: int = 5000, block: int = 10,
        seed: int = 2026) -> list[str]:
    from arch.bootstrap import MCS
    m = MCS(losses, size=size, reps=reps, block_size=block, bootstrap="stationary", seed=seed)
    m.compute()
    return [names[i] for i in sorted(m.included)]


# ---------------------------------------------------------------- Friedman / Nemenyi
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
               11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def friedman_nemenyi(scores: np.ndarray, names: list[str]) -> dict:
    """scores [N blocks, k methods] (higher = better). Ranks 1 = best; CD at α = 0.05 (Demšar 2006)."""
    N, k = scores.shape
    ranks = np.vstack([stats.rankdata(-row) for row in scores])
    chi2, p = stats.friedmanchisquare(*[scores[:, j] for j in range(k)])
    cd = NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6 * N))
    return {"chi2": float(chi2), "p": float(p), "cd": cd, "rank": dict(zip(names, ranks.mean(0).tolist()))}


# ---------------------------------------------------------------- DSR, PBO
def deflated_sharpe(r: np.ndarray, n_trials: int, sr_var_annual: float, rf: float) -> dict:
    """Bailey & López de Prado (2014). SR̂ and the trials' variance in daily units; returns the probability that
    the true Sharpe exceeds the expected maximum of `n_trials` unskilled trials."""
    x = np.asarray(r, float) - rf
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    v = sr_var_annual / 252.0
    sr0 = math.sqrt(v) * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials) + EULER * stats.norm.ppf(1 - 1 / (n_trials * math.e)))
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"sr_annual": float(sr * math.sqrt(252)), "sr0_annual": float(sr0 * math.sqrt(252)), "dsr": float(stats.norm.cdf(z)),
            "n_trials": int(n_trials), "sr_var": float(sr_var_annual)}


def pbo_cscv(M: np.ndarray, S: int = 16) -> dict:
    """Probability of backtest overfitting (Bailey et al. 2017), CSCV with S blocks. M [T, N] returns."""
    T, N = M.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]
    sr = lambda X: X.mean(0) / np.where(X.std(0, ddof=1) > 0, X.std(0, ddof=1), np.inf)
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        best = int(np.argmax(sr(M[is_idx])))
        oos = sr(M[oos_idx])
        omega = (stats.rankdata(oos)[best]) / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": len(logits), "logit_median": float(np.median(logits))}
