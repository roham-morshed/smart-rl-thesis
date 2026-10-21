"""Watchdog for unattended, offline progress of P5 (no network, no interactive session needed).

Every `--interval` seconds:
- if results/logs/p5_pipeline.log says the pipeline is complete → stop;
- if neither scripts/run_p5_pipeline.py nor any scripts/p5_confirm.py worker is running → restart the pipeline
  (it resumes: finished policies and stages are skipped). After 3 consecutive failed restarts with no progress
  in between, it stops and writes NEEDS ATTENTION to its log instead of looping;
- while the pipeline runs, ask Windows not to sleep (SetThreadExecutionState, released when the watchdog exits;
  this does not change any power setting and does not stop a sleep forced by the user or by closing the lid).

    python scripts/watchdog.py              # runs until P5 is complete
    python scripts/watchdog.py --once       # one check (test)
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
LOG = LOGS / "watchdog.log"
PIPE_LOG = LOGS / "p5_pipeline.log"
PY = sys.executable
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def processes() -> dict[str, int]:
    out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
                         capture_output=True, text=True, timeout=120).stdout
    return {"pipeline": sum("run_p5_pipeline.py" in l for l in out.splitlines()),
            "workers": sum("p5_confirm.py" in l for l in out.splitlines())}


def pipeline_complete() -> bool:
    return PIPE_LOG.exists() and "P5 PIPELINE COMPLETE" in PIPE_LOG.read_text(encoding="utf-8", errors="ignore")


def n_policies() -> int:
    return sum(1 for _ in (ROOT / "data" / "processed" / "policies").glob("*/*/*/seed*/info.json"))


def keep_awake(on: bool) -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def start_pipeline() -> None:
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0          # DETACHED_PROCESS | NEW_PROCESS_GROUP
    with (LOGS / "p5_console.log").open("a", encoding="utf-8") as out:
        subprocess.Popen([PY, "-W", "ignore", "-u", "scripts/run_p5_pipeline.py"], cwd=ROOT, stdout=out,
                         stderr=subprocess.STDOUT, creationflags=flags,
                         env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})


def check(state: dict) -> bool:
    """One watchdog cycle; returns False when the watchdog should stop."""
    if pipeline_complete():
        log("P5 complete; watchdog stops")
        return False
    p = processes()
    done = n_policies()
    if p["pipeline"] or p["workers"]:
        keep_awake(True)
        if done > state["last_done"]:
            state["fails"] = 0
        state["last_done"] = max(state["last_done"], done)
        return True
    if state["fails"] >= 3:
        log("NEEDS ATTENTION: the pipeline stopped 3 times without progress; see results/logs/p5_pipeline.log")
        return False
    if state["restarted_at"] is not None and done <= state["last_done"]:
        state["fails"] += 1
    state["last_done"] = done
    log(f"pipeline not running ({done} policies done); restarting (attempt {state['fails'] + 1})")
    start_pipeline()
    state["restarted_at"] = time.time()
    keep_awake(True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    state = {"fails": 0, "last_done": n_policies(), "restarted_at": None}
    log(f"watchdog started (pid {os.getpid()}); {state['last_done']} policies done; processes {processes()}")
    try:
        while check(state) and not args.once:
            time.sleep(args.interval)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
