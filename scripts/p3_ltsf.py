"""P3 / E3: MarketMind backbone on ETTh1 and ETTm1, horizons 96/192/336/720, seeds 0-2 (SPEC §15).

    python scripts/p3_ltsf.py            # full benchmark, resumable (results/p3_ltsf_runs.jsonl)
    python scripts/p3_ltsf.py --smoke    # ETTh1, H 96, 2 epochs; nothing written to results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import RESULTS, frozen  # noqa: E402
from smartrl.marketmind import ltsf  # noqa: E402
from smartrl.runs import run_record  # noqa: E402

RUNS = RESULTS / "p3_ltsf_runs.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = frozen()["ltsf_benchmark"]
    if args.smoke:
        print(ltsf.run("ETTh1", 96, 0, cfg, max_epochs=2))
        return
    done = {}
    if RUNS.exists():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["dataset"], r["horizon"], r["seed"])] = r
    for ds in cfg["datasets"]:
        for H in cfg["horizons"]:
            for seed in cfg["seeds"]:
                if (ds, H, seed) in done:
                    continue
                with run_record("P3", f"E3_{ds}_H{H}", seed, str(ltsf.DEV), {"dataset": ds, "horizon": H}):
                    r = ltsf.run(ds, H, seed, cfg)
                done[(ds, H, seed)] = r
                with RUNS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
    items = []
    for ds in cfg["datasets"]:
        for H in cfg["horizons"]:
            rows = [done[(ds, H, s)] for s in cfg["seeds"]]
            for k in ("mse", "mae"):
                v = np.array([r[k] for r in rows])
                items += [(f"ltsf.{ds}.h{H}.{k}", float(v.mean())), (f"ltsf.{ds}.h{H}.{k}_sd", float(v.std(ddof=1)))]
    R.update(items, ["ltsf.*: mean (and standard deviation) over seeds 0-2 of the test MSE and MAE on standardised data, "
                     "all channels, every test window; no reversible instance normalisation (not part of SPEC §15)."])


if __name__ == "__main__":
    main()
