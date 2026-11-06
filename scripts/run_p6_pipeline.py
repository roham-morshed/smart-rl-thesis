"""Unattended, offline, resumable P6 run (E14 and E15; E16 is not part of it).

Steps (each skipped when done; state in results/logs/p6/pipeline_state.json, log in results/logs/p6/pipeline.log):
  e14_data   wait until the E14 data chain (news, A_PAPER, scoring, daily indices) has finished
  e15_fc     E15 forecasts (BASE_PAPER needs the daily indices)
  e14_run    the E14 run matrix (GPU, 3 workers)
  e14_eval, e14_tests, e14_outputs
  e15_ppo    SMART-RL and B6 policies (CPU, 6 workers)
  e15_sac    B7 policies (GPU, 6 workers)
  e15_eval, e15_outputs
Commits results after each step.

    python scripts/run_p6_pipeline.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = ROOT / "results" / "logs" / "p6"
LOG = LOGS / "pipeline.log"
STATE = LOGS / "pipeline_state.json"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

STEPS = [
    ("e15_fc", ["scripts/p6_e15.py", "--stage", "forecasts"]),
    ("e14_run", ["scripts/p6_e14.py", "--stage", "run", "--jobs", "3"]),
    ("e14_eval", ["scripts/p6_e14.py", "--stage", "eval"]),
    ("e14_tests", ["scripts/p6_e14.py", "--stage", "tests", "--jobs", "4"]),
    ("e14_outputs", ["scripts/p6_e14.py", "--stage", "outputs"]),
    ("e15_ppo", ["scripts/p6_e15.py", "--stage", "train", "--jobs", "6"]),
    ("e15_sac", ["scripts/p6_e15.py", "--stage", "train_sac", "--jobs", "6"]),
    ("e15_eval", ["scripts/p6_e15.py", "--stage", "evaluate", "--jobs", "4"]),
    ("e15_outputs", ["scripts/p6_e15.py", "--stage", "outputs"]),
]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def data_stage_running() -> bool:
    out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"], capture_output=True,
                         text=True, timeout=120).stdout
    return any("p6_e14.py" in l and any(f"--stage {s}" in l for s in ("news", "a_paper", "score", "daily"))
               for l in out.splitlines())


def data_stage_done(stage: str) -> bool:
    p6dir = ROOT / "data" / "processed" / "p6"
    if stage == "a_paper":
        return all((p6dir / "a_paper" / f"seed{s}.pt").exists() for s in (0, 1, 2))
    if stage == "score":
        return (p6dir / "a_paper_scores.parquet").exists()
    return all((p6dir / f"daily_{g}_{m}.parquet").exists() for g in ("cal", "ses")
               for m in ("SP500", "FTSE100", "NIFTY50", "SSE"))


def commit(msg: str) -> None:
    subprocess.run(["git", "add", "-A", "results", "reports", "configs/p6.yaml"], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, capture_output=True)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"done": []}
    chain = LOGS / "e14_data_chain.log"
    text = lambda: chain.read_text(encoding="utf-8") if chain.exists() else ""
    while "CHAIN COMPLETE" not in text() and "e14_data" not in state["done"]:
        if "FAILED" in text():
            log("E14 data chain FAILED: see results/logs/p6/e14_data_chain.log")
            sys.exit(1)
        if not data_stage_running():                        # the chain was cut off: finish its stages here
            log("E14 data chain not running and not complete: running the missing data stages")
            for st in ("a_paper", "score", "daily"):
                if data_stage_done(st):
                    continue
                r = subprocess.run([PY, "-W", "ignore", "-u", "scripts/p6_e14.py", "--stage", st], cwd=ROOT, env=ENV,
                                   stdout=(LOGS / f"e14_{st}.log").open("a", encoding="utf-8"), stderr=subprocess.STDOUT)
                if r.returncode != 0:
                    log(f"FAILED data stage {st}")
                    sys.exit(1)
            break
        time.sleep(60)
    if "e14_data" not in state["done"]:
        state["done"].append("e14_data")
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        log("E14 data ready")
    for name, args in STEPS:
        if name in state["done"]:
            continue
        log(f"START {name}")
        t0 = time.time()
        with (LOGS / f"pipeline_{name}.log").open("a", encoding="utf-8") as f:
            r = subprocess.run([PY, "-W", "ignore", "-u", *args], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log(f"FAILED {name} (exit {r.returncode}) after {(time.time() - t0) / 60:.1f} min")
            commit(f"P6 pipeline stopped at {name}")
            sys.exit(1)
        log(f"DONE {name} in {(time.time() - t0) / 60:.1f} min")
        state["done"].append(name)
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        commit(f"P6 pipeline: {name}")
    log("P6 PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
