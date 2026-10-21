"""Unattended offline run of P1 (rest) → P2 → P3 → P4 in the order of docs/PROJECT_RULES.md, without network access.

Every step is a separate process with HF_HUB_OFFLINE=1. A step that already left its output is skipped, so the
script can be restarted after an interruption. The run stops at the first failure (see results/logs/pipeline.log)
and never creates locks/G4.lock: it ends after the development-phase selection (gate G4 needs Roham's approval).

    python scripts/run_offline_pipeline.py
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
LOG = ROOT / "results" / "logs" / "pipeline.log"
STATE = ROOT / "results" / "logs" / "pipeline_state.json"
P = ROOT / "data" / "processed"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}

STEPS = [
    # (name, command, output that marks the step as done or None)
    ("P1 news panel", ["scripts/p1_build_news.py"], P / "news_aligned_SSE.parquet"),
    ("P1 outputs", ["scripts/p1_outputs.py"], None),
    ("G1 leakage tests", ["-m", "pytest", "-q", "tests/test_leakage.py", "tests/test_news_pipeline.py",
                          "tests/test_daily_features.py"], None),
    ("P2 Branch B pre-training", ["scripts/p2_sentiflow.py", "--pretrain_b"], None),
    ("P2 Word2Vec", ["scripts/p2_word2vec.py"], P / "word2vec_cbow300.kv"),
    ("P2 E1 SentiFlow", ["scripts/p2_sentiflow.py"], ROOT / "results" / "p2_e1_detail.json"),
    ("P2 scoring and daily features", ["scripts/p2_score_and_daily.py"], P / "sentiment_daily_SSE.parquet"),
    ("G2 leakage tests", ["-m", "pytest", "-q", "tests/test_leakage.py", "tests/test_daily_features.py"], None),
    ("P3 MarketMind refits", ["scripts/p3_marketmind.py", "--stage", "refits"], P / "forecast_INFORMER.parquet"),
    ("P3 E13 pilot", ["scripts/p3_e13.py", "--stage", "pilot"], None),
    ("P3 E13 main", ["scripts/p3_e13.py", "--stage", "main"], P / "e13_predictions.parquet"),
    ("P3 SENTIN refits", ["scripts/p3_e13.py", "--stage", "refits"], P / "forecast_SENTIN.parquet"),
    ("P3 H4 comparison", ["scripts/p3_marketmind.py", "--stage", "h4"], None),
    ("P3 E2 evaluation", ["scripts/p3_marketmind.py", "--stage", "eval"], ROOT / "results" / "p3_marketmind_detail.json"),
    ("G3 leakage tests", ["-m", "pytest", "-q", "tests/test_leakage.py"], None),
    ("P4 STM grid", ["scripts/p4_dev.py", "--stage", "stm"], None),
    ("P4 B5 grid", ["scripts/p4_dev.py", "--stage", "b5"], None),
    ("P4 PPO grid", ["scripts/p4_dev.py", "--stage", "ppo"], None),
    ("P4 IMIT arm", ["scripts/p4_dev.py", "--stage", "imit"], None),
    ("P4 AWS grid", ["scripts/p4_dev.py", "--stage", "aws"], None),
    ("P4 selection and config freeze", ["scripts/p4_dev.py", "--stage", "select"], None),
    ("G4 leakage tests (pre-lock)", ["-m", "pytest", "-q", "tests/test_leakage.py"], None),
    ("Outputs", ["scripts/make_outputs.py"], None),
]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def others_running() -> list[str]:
    """Earlier background jobs of this project that must finish first (news build, Branch B pre-training)."""
    try:
        out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    me = str(os.getpid())
    return [l.strip() for l in out.splitlines()
            if ("p1_build_news.py" in l or "--pretrain_b" in l) and not l.strip().endswith(me)]


def git_commit(msg: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, capture_output=True)


def main() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"done": []}
    while others_running():
        log("waiting for the running news build / Branch B pre-training to finish")
        time.sleep(300)
    for name, cmd, marker in STEPS:
        if name in state["done"] or (marker is not None and marker.exists() and name != "P1 news panel"):
            if name not in state["done"]:
                state["done"].append(name)
            continue
        if name == "P1 news panel" and marker.exists():
            state["done"].append(name)
            continue
        log(f"START {name}: {' '.join(cmd)}")
        t0 = time.time()
        step_log = ROOT / "results" / "logs" / f"pipeline_{name.lower().replace(' ', '_').replace('/', '')}.log"
        with step_log.open("a", encoding="utf-8") as f:
            r = subprocess.run([PY, "-W", "ignore", "-u", *cmd], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if r.returncode != 0:
            log(f"FAILED {name} after {mins:.1f} min (exit {r.returncode}); see {step_log.name}")
            git_commit(f"Offline pipeline: stopped at {name} (failed)")
            sys.exit(1)
        log(f"DONE {name} in {mins:.1f} min")
        state["done"].append(name)
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        if name.startswith("G") and "leakage" in name:
            gate = name.split()[0]
            subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'.'); from smartrl import results as R; "
                                      f"R.update([('meta.leakage_by_gate.{gate}', True)])"], cwd=ROOT, env=ENV)
        git_commit(f"Offline pipeline: {name}")
    log("PIPELINE COMPLETE up to gate G4 (no lock created; waiting for Roham's 'G4 تأیید')")


if __name__ == "__main__":
    main()
