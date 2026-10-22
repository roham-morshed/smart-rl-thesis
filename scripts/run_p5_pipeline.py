"""Unattended P5 run (after the G4 lock): PPO arms on the CPU and SAC (B7) on the GPU in parallel, then evaluation,
analysis, the G5 leakage and contract tests, and the outputs. Resumable; each training job skips finished policies.

    python scripts/run_p5_pipeline.py
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
LOGS = ROOT / "results" / "logs"
LOG = LOGS / "p5_pipeline.log"
STATE = LOGS / "p5_pipeline_state.json"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def commit(msg: str) -> None:
    subprocess.run(["git", "add", "-A", "results", "reports"], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, capture_output=True)


def run(name: str, args: list[str], state: dict) -> None:
    if name in state["done"]:
        return
    log(f"START {name}")
    t0 = time.time()
    with (LOGS / f"p5_{name}.log").open("a", encoding="utf-8") as f:
        r = subprocess.run([PY, "-W", "ignore", "-u", *args], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        log(f"FAILED {name} (exit {r.returncode}) after {(time.time() - t0) / 60:.1f} min")
        commit(f"P5 pipeline stopped at {name}")
        sys.exit(1)
    log(f"DONE {name} in {(time.time() - t0) / 60:.1f} min")
    state["done"].append(name)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    commit(f"P5 pipeline: {name}")


def clean_partial_policies() -> None:
    """A policy counts as finished only when model.zip, obs_rms.pkl and a readable info.json exist; anything else
    left by a shutdown is removed so that it is trained again."""
    import shutil
    removed = []
    for d in (ROOT / "data" / "processed" / "policies").glob("*/*/*/seed*"):
        if d.parent.parent.parent.name == "dev":
            continue
        ok = all((d / f).exists() for f in ("model.zip", "obs_rms.pkl", "info.json"))
        if ok:
            try:
                json.loads((d / "info.json").read_text(encoding="utf-8"))
            except Exception:
                ok = False
        if not ok:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d.relative_to(ROOT)))
    if removed:
        log(f"removed {len(removed)} incomplete policy folders: {removed[:6]}")


def main() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"done": []}
    clean_partial_policies()
    if "train" not in state["done"]:
        log("START train (PPO arms, CPU) and train_sac (B7, GPU) in parallel")
        t0 = time.time()
        fp = (LOGS / "p5_train.log").open("a", encoding="utf-8")
        fs = (LOGS / "p5_train_sac.log").open("a", encoding="utf-8")
        p_ppo = subprocess.Popen([PY, "-W", "ignore", "-u", "scripts/p5_confirm.py", "--stage", "train"], cwd=ROOT, env=ENV,
                                 stdout=fp, stderr=subprocess.STDOUT)
        sac_jobs = os.environ.get("SMARTRL_SAC_JOBS", "6")
        p_sac = subprocess.Popen([PY, "-W", "ignore", "-u", "scripts/p5_confirm.py", "--stage", "train_sac", "--jobs",
                                  sac_jobs], cwd=ROOT,
                                 env=ENV, stdout=fs, stderr=subprocess.STDOUT)
        rc_ppo = p_ppo.wait()
        log(f"PPO arms finished (exit {rc_ppo}) after {(time.time() - t0) / 3600:.2f} h")
        commit("P5 pipeline: PPO arms trained")
        rc_sac = p_sac.wait()
        log(f"SAC finished (exit {rc_sac}) after {(time.time() - t0) / 3600:.2f} h")
        if rc_ppo != 0 or rc_sac != 0:
            commit("P5 pipeline stopped in training")
            sys.exit(1)
        state["done"].append("train")
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        commit("P5 pipeline: all policies trained")
    run("evaluate", ["scripts/p5_confirm.py", "--stage", "evaluate"], state)
    run("analyze", ["scripts/p5_confirm.py", "--stage", "analyze"], state)
    run("g5_leakage", ["-m", "pytest", "-q", "tests/test_leakage.py"], state)
    subprocess.run([PY, "-c", "import sys; sys.path.insert(0,'.'); from smartrl import results as R; "
                              "t=R.load(); g=t['meta'].get('leakage_by_gate',{}); g['G5']=True; "
                              "R.update([('meta.leakage_by_gate', g), ('meta.leakage_tests_passed', all(g.values()))])"],
                   cwd=ROOT, env=ENV)
    run("outputs", ["scripts/make_outputs.py"], state)
    r = subprocess.run([PY, "-m", "pytest", "-q", "tests/test_results_contract.py"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    (LOGS / "p5_contract_test.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    log(f"contract test exit {r.returncode}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
    commit("P5 pipeline: analysis, outputs and contract test")
    log("P5 PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
