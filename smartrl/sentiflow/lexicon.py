"""LEX baseline (SPEC §3.7): Loughran–McDonald polarity with two thresholds tuned on validation macro-F1."""
from __future__ import annotations

import numpy as np

from .train import macro_f1

_LM = None


def polarity(texts) -> np.ndarray:
    global _LM
    if _LM is None:
        import pysentiment2 as ps
        _LM = ps.LM()
    out = np.empty(len(texts))
    for i, t in enumerate(texts):
        s = _LM.get_score(_LM.tokenize(str(t)))
        out[i] = (s["Positive"] - s["Negative"]) / (s["Positive"] + s["Negative"] + 1e-6)
    return out


def classify(pol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(pol > hi, 2, np.where(pol < lo, 0, 1))


def tune(pol: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Grid over midpoints between the distinct polarity values (plus the ends); returns (lo, hi, val macro-F1).
    Ties are broken toward the widest neutral band."""
    v = np.unique(pol)
    cand = np.concatenate([[v[0] - 1e-9], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-9]])
    best = (-1.0, 0.0, 0.0, -1.0)
    for i, lo in enumerate(cand):
        for hi in cand[i:]:
            f = macro_f1(y, classify(pol, lo, hi))
            key = (f, hi - lo)
            if key > (best[0], best[3]):
                best = (f, lo, hi, hi - lo)
    return float(best[1]), float(best[2]), float(best[0])
