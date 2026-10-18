"""Forecast evaluation (SPEC §4.8): pinball, MAE/RMSE, directional accuracy with the Pesaran–Timmermann test,
interval coverage and width, QLIKE and MSE of log RV, Diebold–Mariano with the Harvey–Leybourne–Newbold correction."""
from __future__ import annotations

import numpy as np
from scipy import stats

TAUS = np.array([0.1, 0.5, 0.9])


def pinball(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mean pinball loss over the three quantiles for each row; y [N], q [N, 3]."""
    d = y[:, None] - q
    return np.maximum(TAUS * d, (TAUS - 1) * d).mean(axis=1)


def pesaran_timmermann(actual: np.ndarray, pred: np.ndarray) -> dict:
    """PT (1992) test of predictive independence for signs; one-sided p (H1: better than chance)."""
    a, p = (np.asarray(actual) > 0).astype(float), (np.asarray(pred) > 0).astype(float)
    n = len(a)
    if n < 10:
        return {"stat": np.nan, "p": np.nan}
    pa, pp = a.mean(), p.mean()
    ph = (a == p).mean()
    pstar = pa * pp + (1 - pa) * (1 - pp)
    v_ph = pstar * (1 - pstar) / n
    v_ps = ((2 * pa - 1) ** 2 * pp * (1 - pp) / n + (2 * pp - 1) ** 2 * pa * (1 - pa) / n
            + 4 * pa * pp * (1 - pa) * (1 - pp) / n ** 2)
    den = v_ph - v_ps
    if den <= 0:
        return {"stat": np.nan, "p": np.nan}
    s = (ph - pstar) / np.sqrt(den)
    return {"stat": float(s), "p": float(1 - stats.norm.cdf(s))}


def diebold_mariano(d: np.ndarray, h: int) -> dict:
    """DM test on a loss differential d_t (model A minus model B) with Newey–West variance (lag h−1, Bartlett
    weights) and the HLN small-sample correction; two-sided p from Student t with T−1 df. stat < 0 means A has
    the lower loss."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = len(d)
    m = d.mean()
    x = d - m
    lag = max(h - 1, 0)
    gamma0 = x @ x / T
    lrv = gamma0
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / T
    if lrv <= 0:
        lrv = gamma0
    dm = m / np.sqrt(lrv / T)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * corr
    return {"stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), df=T - 1)), "mean_diff": float(m), "n": T}


def qlike(rv: np.ndarray, var_hat: np.ndarray) -> np.ndarray:
    """QLIKE = RV/σ̂² − log(RV/σ̂²) − 1 (Patton 2011), per row."""
    ratio = np.maximum(rv, 1e-14) / np.maximum(var_hat, 1e-14)
    return ratio - np.log(ratio) - 1


def point_metrics(y: np.ndarray, q: np.ndarray) -> dict:
    """y [N] realised r_h; q [N, 3] quantile forecasts (q10, q50, q90)."""
    ok = np.isfinite(y) & np.isfinite(q).all(1)
    y, q = y[ok], q[ok]
    e = q[:, 1] - y
    nz = y != 0
    pt = pesaran_timmermann(y[nz], q[nz, 1])
    return {"pinball": float(pinball(y, q).mean()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "diracc": float((np.sign(q[nz, 1]) == np.sign(y[nz])).mean()), "pt_p": pt["p"],
            "cov80": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean()), "width": float((q[:, 2] - q[:, 0]).mean()),
            "n": int(ok.sum())}
