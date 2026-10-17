"""Project paths, the frozen configuration and its hash."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FROZEN = CONFIGS / "frozen.yaml"
PATHS = CONFIGS / "paths.yaml"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"
REPORTS = ROOT / "reports"
LOCKS = ROOT / "locks"

MARKETS = ("SP500", "FTSE100", "NIFTY50", "SSE")


def config_hash() -> str:
    """SHA-256 of configs/frozen.yaml as stored on disk (the G4 lock stores this value)."""
    return hashlib.sha256(FROZEN.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _frozen_cached(digest: str) -> dict:
    return yaml.safe_load(FROZEN.read_text(encoding="utf-8"))


def frozen() -> dict:
    """The frozen configuration; reloaded whenever the file changes."""
    return _frozen_cached(config_hash())


def paths() -> dict:
    return yaml.safe_load(PATHS.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for d in (RAW, PROCESSED, RUNS, TABLES, FIGURES, LOGS, REPORTS, LOCKS):
        d.mkdir(parents=True, exist_ok=True)
