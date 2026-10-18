"""Daily sentiment features (SPEC §3.9) on a hand-checkable example."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.sentiflow import daily  # noqa: E402


def test_daily_features():
    cal = pd.DatetimeIndex(pd.bdate_range("2019-01-01", periods=6))
    aligned = pd.DataFrame({"id": [0, 1, 2, 0, 1, 2, 3], "trading_date": [cal[1]] * 3 + [cal[1]] * 3 + [cal[3]],
                            "tag": ["GLOBAL"] * 3 + ["US"] * 3 + ["GLOBAL"]})
    scores = pd.DataFrame({"s": [0.5, -0.2, 0.9, 0.1], "c": [0.8, 0.5, 0.9, 0.6], "b_s": [0.4, -0.1, 0.8, 0.0],
                           "b_c": [0.7, 0.6, 0.9, 0.5], "a_s": [0.6, -0.3, 0.7, 0.2]}, index=pd.Index([0, 1, 2, 3], name="id"))
    out = daily.build("SP500", aligned, scores, cal).set_index("date")
    s_loc = (0.8 * 0.5 + 0.5 * -0.2 + 0.9 * 0.9) / (0.8 + 0.5 + 0.9)
    assert out.loc[cal[1], "n_loc"] == 3 and out.loc[cal[3], "n_loc"] == 0 and out.loc[cal[3], "n_glob"] == 1
    assert out.loc[cal[1], "S_loc"] == pytest.approx(s_loc)
    assert out.loc[cal[0], "S_loc"] == 0.0 and out.loc[cal[2], "S_loc"] == 0.0
    lam = 2 ** (-1 / 3)
    assert out.loc[cal[1], "Se_loc"] == pytest.approx((1 - lam) * s_loc)
    assert out.loc[cal[2], "Se_loc"] == pytest.approx(out.loc[cal[1], "Se_loc"])          # carried, no news
    assert out.loc[cal[1], "disp_loc"] == pytest.approx(np.std([0.5, -0.2, 0.9], ddof=1))
    assert out.loc[cal[1], "sentin_S"] == pytest.approx(np.mean([0.6, -0.3, 0.7]))
    assert out.loc[cal[3], "cbar"] == pytest.approx(0.6)                                   # global when no local
    assert out.loc[cal[0], "cbar"] == 0.0
    cov = daily.coverage(out["n_loc"], pd.Series(True, index=out.index))
    assert set(cov.unique()) <= {0.0, 1.0}                                                 # median 0 -> indicator
