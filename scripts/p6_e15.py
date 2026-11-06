"""P6 / E15: trading in the base paper's test window, 2018-10-18 .. 2022-03-31 (docs/P6_BASE_PROTOCOL.md §3).

    python scripts/p6_e15.py --stage forecasts        # MM, LSTM_PRICE, SENTIN (BASE), SENTIN_PAPER (BASE_PAPER) fits
    python scripts/p6_e15.py --stage train [--jobs 6] # SMART-RL and B6 PPO policies (CPU), resumable
    python scripts/p6_e15.py --stage train_sac [--jobs 6]   # B7 SAC policies (GPU), resumable
    python scripts/p6_e15.py --stage evaluate         # paths, metrics, direction, tests, P5 overlap -> e15.*
    python scripts/p6_e15.py --stage outputs          # tables T23, T24 and figure F18
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.p6 import e15 as E  # noqa: E402
from smartrl.p6.cfg import p6_run  # noqa: E402


def stage_train(sac: bool, n_jobs: int) -> None:
    from joblib import Parallel, delayed
    from smartrl.evaluation import confirm as C
    E.patch()
    todo = [j for j in E.jobs(sac) if not (C.policy_dir("e15", *j) / "info.json").exists()]
    print(f"{len(todo)} policies to train", flush=True)
    t0 = time.time()
    with p6_run("E15_train_sac" if sac else "E15_train_ppo", None, "cuda" if sac else "cpu",
                {"n_runs": len(todo), "n_jobs": n_jobs}):
        res = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(delayed(E.train_job)(*j) for j in todo)
    print(f"{sum('dir' in r for r in res)} trained in {(time.time() - t0) / 3600:.2f} h", flush=True)


def stage_evaluate(n_jobs: int) -> None:
    from smartrl.p6 import e15eval as EV
    with p6_run("E15_evaluate"):
        df = EV.evaluate_all(n_jobs)
        items = EV.summarise(df)
    R.update(items, [
        "e15: trading in the base paper's test window (protocol §3). One split per market: train 2010-01-01 .. "
        "2017-10-17, validation 2017-10-18 .. 2018-10-17, one test run 2018-10-18 .. 2022-03-31 (no refits inside it). "
        "SMART-RL with the frozen configuration (five PPO seeds, AWS warm-started through the validation year); MarketMind "
        "and LSTM_PRICE features from the annual refits before 2018-10-18 and from one fit on data before 2018-10-18 for "
        "the test window; BASE = SENTIN (E13 recipe) and BASE_PAPER = SENTIN_PAPER (Table 3 recipe, S_PAPER) fitted once "
        "on trading days before 2018-10-18. The frozen configuration was selected on development validation data that end "
        "in 2018-06, before this window starts.",
        "e15.full / e15.sub2020: metrics on 2018-10-18 .. 2022-03-31 and on its sub-window 2020-01-01 .. 2022-03-31 (same "
        "run); e15.cost20: the same policies at 20 bps; e15.dir: decisions as next-day direction calls (prediction "
        "1[e_t > 0.5], score e_t, label 1[C_{t+1} > C_t]); e15.tests: SMART-RL minus method on COMBINED (Ledoit-Wolf "
        "HAC and bootstrap p, one-sided; stationary-bootstrap 95 % intervals of the MDD and cumulative-return "
        "differences); descriptive, no new hypothesis. e15.p5overlap: P5 daily returns on 2020-01-01 .. 2022-03-31."])
    print(f"{len(items)} keys", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["forecasts", "train", "train_sac", "evaluate", "outputs"])
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    if args.stage == "forecasts":
        E.fit_forecasts()
    elif args.stage in ("train", "train_sac"):
        stage_train(args.stage == "train_sac", args.jobs)
    elif args.stage == "evaluate":
        stage_evaluate(args.jobs)
    else:
        from smartrl.p6 import e15eval as EV
        EV.outputs()


if __name__ == "__main__":
    main()
