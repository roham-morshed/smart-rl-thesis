"""Run records: results/runs/<run_id>/meta.json plus a SQLite registry (PROJECT_RULES.md rule 4, SPEC §2.5)."""
from __future__ import annotations

import json
import platform
import sqlite3
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

from .config import RESULTS, ROOT, RUNS, config_hash

REGISTRY = RESULTS / "registry.sqlite"

KEY_PACKAGES = (
    "torch", "numpy", "pandas", "pyarrow", "duckdb", "scipy", "statsmodels", "scikit-learn", "matplotlib",
    "arabic-reshaper", "python-bidi", "yfinance", "datasets", "huggingface-hub", "transformers", "gensim",
    "pysentiment2", "gymnasium", "stable-baselines3", "arch", "joblib", "tqdm", "PyYAML", "pytest",
)


def package_versions(names: tuple[str, ...] = KEY_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return r.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(),
                            "processor": platform.processor()}
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except Exception:  # pragma: no cover
        info["cuda_available"] = False
    return info


def _registry() -> sqlite3.Connection:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(REGISTRY, timeout=60)
    con.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, experiment TEXT, name TEXT, seed INTEGER, config_hash TEXT,
        git_commit TEXT, start_utc TEXT, end_utc TEXT, seconds REAL, device TEXT, status TEXT, extra TEXT)""")
    return con


class Run:
    def __init__(self, experiment: str, name: str, seed: int | None, device: str, params: dict | None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.run_id = f"{experiment}_{name}_s{seed if seed is not None else 'na'}_{stamp}_{uuid.uuid4().hex[:6]}"
        self.dir = RUNS / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = {
            "run_id": self.run_id, "experiment": experiment, "name": name, "seed": seed, "device": device,
            "config_hash": config_hash(), "git_commit": git_commit(),
            "start_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hardware": hardware(), "packages": package_versions(), "params": params or {}, "events": [],
        }
        self._t0 = time.perf_counter()
        self._write()

    def log(self, message: str, **fields: Any) -> None:
        """Record an event (e.g. a halved batch size, PROJECT_RULES.md rule 8)."""
        self.meta["events"].append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "message": message, **fields})
        self._write()

    def set(self, **fields: Any) -> None:
        self.meta.update(fields)
        self._write()

    def _write(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str), encoding="utf-8")

    def close(self, status: str) -> None:
        self.meta["end_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.meta["seconds"] = time.perf_counter() - self._t0
        self.meta["status"] = status
        self._write()
        con = _registry()
        with con:
            con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                self.run_id, self.meta["experiment"], self.meta["name"], self.meta["seed"],
                self.meta["config_hash"], self.meta["git_commit"], self.meta["start_utc"], self.meta["end_utc"],
                self.meta["seconds"], self.meta["device"], status, json.dumps(self.meta["params"], default=str)))
        con.close()


@contextmanager
def run_record(experiment: str, name: str, seed: int | None = None, device: str = "cpu",
               params: dict | None = None) -> Iterator[Run]:
    run = Run(experiment, name, seed, device, params)
    try:
        yield run
    except BaseException:
        run.close("failed")
        raise
    run.close("ok")


def compute_totals() -> dict[str, float]:
    """gpu_hours, cpu_hours and n_runs over every successful run in the registry."""
    if not REGISTRY.exists():
        return {"gpu_hours": 0.0, "cpu_hours": 0.0, "n_runs": 0}
    con = _registry()
    rows = con.execute("SELECT device, SUM(seconds), COUNT(*) FROM runs WHERE status='ok' GROUP BY device").fetchall()
    con.close()
    gpu = sum(s for d, s, _ in rows if d and d.startswith("cuda")) / 3600
    cpu = sum(s for d, s, _ in rows if not (d and d.startswith("cuda"))) / 3600
    return {"gpu_hours": gpu, "cpu_hours": cpu, "n_runs": int(sum(n for *_, n in rows))}
