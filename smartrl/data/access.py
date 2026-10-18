"""The only gate to data dated on or after 2020-01-01 (PROJECT_RULES.md rule 2, SPEC §1 and §2.6 test 4).

Every reader of processed panels passes a `purpose`. Purposes in `PRE_LOCK_PURPOSES` are the exceptions the
SPEC allows before gate G4 (deterministic transforms and forecast refits that only use past data for each
predicted year). Every other purpose, in particular `"policy"` (PPO, SAC, STM, AWS, rewards, trading
evaluation), may read rows dated >= 2020-01-01 only when locks/G4.lock exists and stores the SHA-256 of
configs/frozen.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import LOCKS, config_hash

LOCK_FROM = pd.Timestamp("2020-01-01")
LOCK_FILE = LOCKS / "G4.lock"

PRE_LOCK_PURPOSES = frozenset({
    "panel_build",        # (d) price and news panels, P1
    "news_scoring",       # (d) corpus scoring and daily sentiment features, P2
    "data_description",   # (d) E0 statistics, P0 ETF check
    "sentiflow_labelled", # (a) labelled sentiment sets (no market dates)
    "marketmind_refit",   # (b) annual refits and H4 evaluation, past data only per predicted year
    "ltsf",               # (c) ETTh1 / ETTm1
    "e13",                # (c) base-paper direction task (without SMART-RL/BASE overlap rows)
})


class TestLockError(RuntimeError):
    """Raised when code reads test-period data before the G4 lock exists."""


def lock_valid() -> bool:
    if not LOCK_FILE.exists():
        return False
    stored = LOCK_FILE.read_text(encoding="utf-8").strip().split()
    return bool(stored) and stored[0] == config_hash()


def check(dates: Iterable | pd.Series | pd.Index, purpose: str) -> None:
    """Raise TestLockError if `dates` reach into the locked period and `purpose` is not allowed yet."""
    if purpose in PRE_LOCK_PURPOSES:
        return
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(list(dates)) if not isinstance(dates, (pd.Series, pd.Index))
                                          else dates))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    if len(idx) and idx.max() >= LOCK_FROM and not lock_valid():
        raise TestLockError(
            f"purpose={purpose!r} requested data up to {idx.max().date()} (>= {LOCK_FROM.date()}) "
            "but locks/G4.lock is missing or does not match configs/frozen.yaml")


def request_end(end: str | pd.Timestamp, purpose: str) -> pd.Timestamp:
    """Validate the last date a reader wants before loading anything."""
    end = pd.Timestamp(end)
    check([end], purpose)
    return end


def read_parquet(path: Path, purpose: str, date_col: str = "date",
                 start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read a processed parquet file, cut to [start, end], and enforce the lock on what is returned."""
    if end is not None:
        request_end(end, purpose)
    df = pd.read_parquet(path)
    d = pd.to_datetime(df[date_col])
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= d >= pd.Timestamp(start)
    if end is not None:
        mask &= d <= pd.Timestamp(end)
    out = df.loc[mask]
    check(pd.to_datetime(out[date_col]), purpose)
    return out
