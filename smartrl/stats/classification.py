"""Paired tests for classifiers: exact McNemar and paired bootstrap of the macro-F1 difference (SPEC §12)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom
from sklearn.metrics import f1_score


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Two-sided exact McNemar test on paired correctness (binomial test on the discordant pairs)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2.0 * binom.cdf(min(n01, n10), n, 0.5))
    return {"p": float(p), "a_only": n01, "b_only": n10}


def macro_f1(y, pred, labels=(0, 1, 2)) -> float:
    """Macro-F1 over `labels` from confusion counts (same value as sklearn with zero_division=0)."""
    k = len(labels)
    if tuple(labels) != tuple(range(k)):
        return float(f1_score(y, pred, average="macro", labels=list(labels), zero_division=0))
    cm = np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)
    return float(f1.mean())


def paired_bootstrap_f1_diff(y: np.ndarray, preds_a: list[np.ndarray], preds_b: list[np.ndarray], reps: int = 5000,
                             seed: int = 2026, alpha: float = 0.05, labels=(0, 1, 2)) -> dict:
    """Difference of seed-averaged macro-F1 (A minus B) with a percentile interval from resampling the test items.

    `preds_a`, `preds_b`: one prediction vector per seed (seed s of A is paired with seed s of B)."""
    y = np.asarray(y)
    PA, PB = np.stack(preds_a), np.stack(preds_b)
    est = float(np.mean([macro_f1(y, p, labels) for p in PA]) - np.mean([macro_f1(y, p, labels) for p in PB]))
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        diffs[r] = np.mean([macro_f1(y[i], p[i], labels) for p in PA]) - np.mean([macro_f1(y[i], p[i], labels) for p in PB])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"est": est, "lo": float(lo), "hi": float(hi)}
