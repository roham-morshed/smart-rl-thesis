"""Ask Windows not to sleep while the P6 pipeline runs (SetThreadExecutionState; no power setting is changed).
Exits when results/logs/p6/pipeline.log says COMPLETE or FAILED, or when no P6 pipeline process is left.

    python scripts/keep_awake_p6.py
"""
from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "results" / "logs" / "p6" / "pipeline.log"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def pipeline_alive() -> bool:
    out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"], capture_output=True,
                         text=True, timeout=120).stdout
    return any("run_p6_pipeline.py" in l for l in out.splitlines())


def main() -> None:
    misses = 0
    try:
        while True:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            text = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
            if "P6 PIPELINE COMPLETE" in text or "FAILED" in text:
                break
            misses = 0 if pipeline_alive() else misses + 1
            if misses >= 3:
                break
            time.sleep(300)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
