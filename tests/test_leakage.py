"""Leakage tests of SPEC §2.6. They must pass from gate G1 on.

Tests that need files of a later phase (forecasts, STM, AWS) check those files when they exist and otherwise test
the code path on synthetic data, so the suite can run at every gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartrl.config import MARKETS, PROCESSED, frozen  # noqa: E402
from smartrl.data import access, news, panel  # noqa: E402
from smartrl.data import sentiment_sets as S  # noqa: E402
from smartrl.env.portfolio import RewardSpec  # noqa: E402
from smartrl.env.trading_env import MarketData, TradingEnv  # noqa: E402


def _bars(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h, l = c * (1 + rng.uniform(0, 0.01, n)), c * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({"open": c, "high": h, "low": l, "close": c, "volume": rng.uniform(1e6, 2e6, n)},
                        index=pd.bdate_range("2008-01-01", periods=n))


# 1 ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t", [300, 500, 800])
def test_1_price_features_ignore_future(t):
    bars = _bars()
    f0 = panel.price_features(bars, use_volume=True)
    pert = bars.copy()
    pert.iloc[t + 1:, :] *= np.random.default_rng(1).uniform(0.5, 1.5, size=pert.iloc[t + 1:, :].shape)
    f1 = panel.price_features(pert, use_volume=True)
    pd.testing.assert_frame_equal(f0.iloc[: t + 1], f1.iloc[: t + 1])


def test_1_processed_panels_match_recomputation():
    """Features stored in the processed panels equal a recomputation on bars truncated at t (real data)."""
    for mkt in MARKETS:
        p = panel.processed_path(mkt)
        if not p.exists():
            pytest.skip("price panels not built")
        df = pd.read_parquet(p).set_index("date")
        t = df.index[len(df) // 2]
        trunc = df.loc[:t, ["open", "high", "low", "close", "volume"]]
        use_vol = mkt not in frozen()["features"].get("volume_disabled", [])
        rec = panel.price_features(trunc, use_vol)
        pd.testing.assert_frame_equal(rec.loc[t:t], df.loc[t:t, panel.PRICE_FEATURES], check_names=False)


# 2 ------------------------------------------------------------------------------------------------
def test_2_session_assignment_synthetic():
    cal = pd.DatetimeIndex(pd.bdate_range("2019-03-01", "2019-03-29"))
    ts = pd.Series(pd.to_datetime(["2019-03-08 20:59:00", "2019-03-08 21:00:00", "2019-03-09 12:00:00",
                                   "2019-03-11 19:59:00", "2019-03-11 20:00:01"], utc=True))
    d = news.assign_sessions(ts, "SP500", cal)
    # 2019-03-08 (EST): close 21:00 UTC; 2019-03-11 (EDT, after the DST change): close 20:00 UTC
    assert list(d.dt.strftime("%Y-%m-%d")) == ["2019-03-08", "2019-03-11", "2019-03-11", "2019-03-11", "2019-03-12"]


def test_2_every_aligned_item_before_close():
    if not news.ITEMS.exists():
        pytest.skip("news corpus not built yet (download pending)")
    items = pd.read_parquet(news.ITEMS, columns=["id", "ts_utc"]).set_index("id")
    for mkt in MARKETS:
        al = pd.read_parquet(news.aligned_path(mkt))
        ts = news.utc_seconds(items.loc[al["id"], "ts_utc"].to_numpy())
        closes = dict(zip(pd.DatetimeIndex(al["trading_date"].unique()),
                          news.close_utc(mkt, pd.DatetimeIndex(al["trading_date"].unique()))))
        c = al["trading_date"].map(closes).to_numpy()
        assert np.all(ts < c), f"{mkt}: news after the close of its trading date"


# 3 ------------------------------------------------------------------------------------------------
def test_3_scaler_uses_training_rows_only():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=1000)})
    train = df.iloc[:700]
    p0 = panel.fit_minmax(train, ["x"])
    df2 = df.copy()
    df2.iloc[700:, 0] = 1e6
    p1 = panel.fit_minmax(df2.iloc[:700], ["x"])
    assert p0 == p1
    lo, hi = frozen()["features"]["minmax_quantiles"]
    assert p0["x"] == pytest.approx((np.quantile(train["x"], lo), np.quantile(train["x"], hi)))
    out = panel.apply_minmax(df2, p1)
    assert out["x"].max() <= frozen()["features"]["clip"][1]


# 4 ------------------------------------------------------------------------------------------------
def test_4_test_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(access, "LOCK_FILE", tmp_path / "G4.lock")
    with pytest.raises(access.TestLockError):
        access.check(pd.to_datetime(["2019-12-31", "2020-01-02"]), "policy")
    access.check(pd.to_datetime(["2019-12-31"]), "policy")                   # pre-2020 is fine
    access.check(pd.to_datetime(["2021-06-01"]), "panel_build")              # allowed pre-lock purpose
    (tmp_path / "G4.lock").write_text("not-the-hash")
    with pytest.raises(access.TestLockError):
        access.request_end("2020-01-01", "policy")
    (tmp_path / "G4.lock").write_text(access.config_hash())
    access.request_end("2024-12-31", "policy")


def test_4_real_lock_absent_before_g4():
    """Until Roham's G4 approval there must be no lock: policy reads of 2020+ fail."""
    if access.LOCK_FILE.exists():
        pytest.skip("G4 lock present")
    with pytest.raises(access.TestLockError):
        access.request_end("2020-01-02", "policy")


# 5 ------------------------------------------------------------------------------------------------
def test_5_forecasts_from_past_models():
    files = sorted(PROCESSED.glob("forecast_*.parquet"))
    if not files:
        pytest.skip("forecast files are produced in P3")
    for f in files:
        df = pd.read_parquet(f, columns=["date", "train_end"])
        year = pd.to_datetime(df["date"]).dt.year
        assert (pd.to_datetime(df["train_end"]) <= pd.to_datetime((year - 1).astype(str) + "-12-31")).all(), f.name


# 6 ------------------------------------------------------------------------------------------------
def test_6_no_sentiment_eval_sentence_in_training():
    p = PROCESSED / "sentiment_sets.parquet"
    if not p.exists():
        pytest.skip("sentiment sets not built")
    df = pd.read_parquet(p)
    k = df["sentence"].map(S.norm)
    train = set(k[df["split"] == "train"])
    assert not k[df["split"] != "train"].isin(train).any()
    ar = pd.read_parquet(PROCESSED / "sentiment_sets_araci.parquet")
    for ds in ar["dataset"].unique():
        a = ar[ar["dataset"] == ds]
        ka = a["sentence"].map(S.norm)
        assert not ka[a["split"] != "train"].isin(set(ka[a["split"] == "train"])).any()


# 7 ------------------------------------------------------------------------------------------------
def test_7_env_observation_uses_past_only():
    rng = np.random.default_rng(0)
    T, F = 400, 7
    feats = rng.normal(size=(T, F)).astype(np.float32)
    ret = rng.normal(0, 0.01, T)
    t_check = 200
    env = TradingEnv(MarketData(feats, ret), 19, 390, RewardSpec("R3"), random_start=False)
    obs, _ = env.reset()
    for _ in range(t_check - 19):
        obs, *_ = env.step(np.array([0.3]))
    feats2 = feats.copy()
    feats2[t_check + 1:] = 99.0
    ret2 = ret.copy()
    ret2[t_check + 1:] = 0.5
    env2 = TradingEnv(MarketData(feats2, ret2), 19, 390, RewardSpec("R3"), random_start=False)
    obs2, _ = env2.reset()
    for _ in range(t_check - 19):
        obs2, *_ = env2.step(np.array([0.3]))
    assert env.t == env2.t == t_check
    np.testing.assert_array_equal(obs, obs2)
