"""P6 configuration (configs/p6.yaml), its hash, paths and the run record wrapper."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from functools import lru_cache

import yaml

from ..config import CONFIGS, PROCESSED
from ..runs import run_record

P6_FILE = CONFIGS / "p6.yaml"
P6_DIR = PROCESSED / "p6"


def p6_hash() -> str:
    return hashlib.sha256(P6_FILE.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _load(digest: str) -> dict:
    return yaml.safe_load(P6_FILE.read_text(encoding="utf-8"))


def p6() -> dict:
    return _load(p6_hash())


@contextmanager
def p6_run(name: str, seed: int | None = None, device: str = "cpu", params: dict | None = None):
    """run_record with experiment 'P6' and the p6.yaml hash in the parameters (protocol §0 rule 2)."""
    with run_record("P6", name, seed, device, {**(params or {}), "p6_hash": p6_hash()}) as r:
        yield r
