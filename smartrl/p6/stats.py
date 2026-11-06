"""Tests of protocol §2.7: paired stationary block bootstrap stratified by index, DeLong, exact McNemar, Holm."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from ..stats.classification import mcnemar_exact
from ..stats.trading import holm


def auc_fast(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = sps.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap_fast(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (sklearn's definition: step interpolation over distinct thresholds)."""
    n1 = y.sum()
    if n1 == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys)
    last = np.r_[ss[1:] != ss[:-1], True]                     # last index of each distinct threshold
    tp_t, k_t = tp[last], np.flatnonzero(last) + 1
    prec, rec = tp_t / k_t, tp_t / n1
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def f1_fast(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap (circular), geometric block lengths with the given mean."""
    p = 1.0 / mean_block
    starts = rng.random(n) < p
    starts[0] = True
    jumps = rng.integers(n, size=n)
    pos = np.arange(n)
    block = np.cumsum(starts) - 1                            # block number of every position
    first = pos[starts]                                      # position where each block begins
    return (jumps[starts][block] + pos - first[block]) % n


def metric_set(y, s_a, s_b, pa, pb) -> np.ndarray:
    """[auc, acc, f1, prauc] differences (a - b)."""
    return np.array([auc_fast(y, s_a) - auc_fast(y, s_b), (pa == y).mean() - (pb == y).mean(),
                     f1_fast(y, pa) - f1_fast(y, pb), ap_fast(y, s_a) - ap_fast(y, s_b)])


def paired_bootstrap(groups: list[dict], reps: int, mean_block: float, seed: int) -> dict:
    """groups: one dict per index with y, s_a, s_b, p_a, p_b (test rows sorted by date). Statistic = mean over the
    groups of the metric difference a - b. One-sided p for H1: a > b (centred bootstrap), two-sided 95 % interval."""
    rng = np.random.default_rng(seed)
    est = np.mean([metric_set(g["y"], g["s_a"], g["s_b"], g["p_a"], g["p_b"]) for g in groups], axis=0)
    boot = np.empty((reps, 4))
    for r in range(reps):
        vals = []
        for g in groups:
            i = stationary_indices(len(g["y"]), mean_block, rng)
            vals.append(metric_set(g["y"][i], g["s_a"][i], g["s_b"][i], g["p_a"][i], g["p_b"][i]))
        boot[r] = np.nanmean(vals, axis=0)
    centred = boot - est
    p = (1 + (centred >= est[None, :]).sum(0)) / (reps + 1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    names = ("auc", "acc", "f1", "prauc")
    return {n: {"diff": float(est[k]), "lo": float(lo[k]), "hi": float(hi[k]), "p": float(p[k])}
            for k, n in enumerate(names)}


def _midrank(x: np.ndarray) -> np.ndarray:
    return sps.rankdata(x)


def delong(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> dict:
    """Two-sided DeLong test for two correlated AUCs (Sun and Xu 2014 fast algorithm)."""
    pos, neg = s_a[y == 1], s_a[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return {"z": float("nan"), "p": float("nan")}
    preds = np.vstack([s_a, s_b])
    X, Y = preds[:, y == 1], preds[:, y == 0]
    tx = np.array([_midrank(r) for r in X])
    ty = np.array([_midrank(r) for r in Y])
    tz = np.array([_midrank(r) for r in np.hstack([X, Y])])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"z": 0.0, "p": 1.0}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"z": float(z), "p": float(2 * sps.norm.sf(abs(z)))}


__all__ = ["auc_fast", "ap_fast", "f1_fast", "paired_bootstrap", "delong", "mcnemar_exact", "holm"]
