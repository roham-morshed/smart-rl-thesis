"""P5: confirmation 2020-2024 and forward test (E5-E12). Needs locks/G4.lock (written only after "G4 تأیید").

    python scripts/p5_confirm.py --stage train      # every PPO arm, parallel.rl_jobs workers (resumable)
    python scripts/p5_confirm.py --stage train_sac  # B7 on the GPU, parallel.sac_jobs workers (resumable)
    python scripts/p5_confirm.py --stage evaluate   # daily paths of every method: base, costs 0/5/20 bps, delay
    python scripts/p5_confirm.py --stage analyze    # all P5 keys of results.json (smartrl/evaluation/analysis.py)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.config import LOGS, MARKETS, frozen  # noqa: E402
from smartrl.data import access, folds  # noqa: E402
from smartrl.evaluation import confirm as C  # noqa: E402
from smartrl.runs import run_record  # noqa: E402

VARIANTS = {"base": (None, 0), "cost0": (0, 0), "cost5": (5, 0), "cost20": (20, 0), "delay": (None, 1)}


def returns_path(variant: str) -> Path:
    return LOGS / f"returns_{variant}.parquet"


def stage_train(sac: bool) -> None:
    cfg = frozen()
    arms = ["B7"] if sac else [a for a in C.ARMS if a != "B7"]
    jobs = C.jobs(arms)
    n_jobs = cfg["parallel"]["sac_jobs"] if sac else cfg["parallel"]["rl_jobs"]
    t0 = time.time()
    with run_record("P5", "train_sac" if sac else "train_ppo", None, "cuda" if sac else "cpu",
                    {"n_runs": len(jobs), "n_jobs": n_jobs}):
        res = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(delayed(C.train_job)(*j) for j in jobs)
    done = sum(1 for r in res if "dir" in r)
    print(f"{done} trained, {len(res) - done} already present, {(time.time() - t0) / 3600:.2f} h", flush=True)


def _rows(variant: str, fold: str, market: str, out: dict) -> list[pd.DataFrame]:
    dates = out["dates"]
    rows = []
    for method, v in out.items():
        if method.startswith("_") or method == "dates":
            continue
        df = pd.DataFrame({"variant": variant, "fold": fold, "market": market, "method": method, "date": dates,
                           "r": v["r"], "held": v["held"]})
        if "decision" in v:
            df["decision"] = v["decision"]
        rows.append(df)
    return rows


def evaluate_one(fold: str, market: str) -> dict[str, list[pd.DataFrame]]:
    import torch
    torch.set_num_threads(1)
    inp = C.fold_inputs(fold, market)
    out = {}
    for variant, (cost, delay) in VARIANTS.items():
        if fold == "forward" and variant != "base":
            continue
        res = C.evaluate_fold(fold, market, cost, delay, inp)
        out[variant] = _rows(variant, fold, market, res)
        if variant == "base":
            aws = res["SMARTRL"]["aws"]
            year = res["dates"][0].year
            log = pd.DataFrame({k: aws[k] for k in ("w_rl", "P_rl", "P_sa", "C_rl", "C_sa", "e_rl", "e_sa", "a_final", "e",
                                                    "decision", "r", "r_rl", "r_sa", "held")})
            log.insert(0, "date", res["dates"])
            log["regret"] = aws["regret"]
            tag = "forward" if fold == "forward" else str(year)
            log.to_parquet(LOGS / f"aws_daily_{market}_{tag}.parquet", index=False)
    return out


def stage_evaluate() -> None:
    access.request_end(frozen()["dates"]["raw_end"], "policy")
    tasks = [(f, m) for f in folds.fold_names()[1:] for m in MARKETS]
    with run_record("P5", "evaluate", None, "cpu", {"n": len(tasks)}):
        res = Parallel(n_jobs=frozen()["parallel"]["rl_jobs"], backend="loky", verbose=10)(
            delayed(evaluate_one)(f, m) for f, m in tasks)
    for variant in VARIANTS:
        parts = [df for r in res for df in r.get(variant, [])]
        pd.concat(parts, ignore_index=True).to_parquet(returns_path(variant), index=False)
    print("evaluation written", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "train_sac", "evaluate", "analyze"])
    args = ap.parse_args()
    access.request_end("2020-01-02", "policy")          # refuses to start before the G4 lock
    if args.stage in ("train", "train_sac"):
        stage_train(args.stage == "train_sac")
    elif args.stage == "evaluate":
        stage_evaluate()
    else:
        from smartrl.evaluation import analysis
        analysis.run_all()


if __name__ == "__main__":
    main()
