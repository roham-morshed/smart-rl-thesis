"""P2 steps 3-4: score every headline (SPEC §3.8) and build the daily sentiment features per market (§3.9).
Runs after scripts/p2_sentiflow.py has saved the five seeds' models."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, frozen  # noqa: E402
from smartrl.data import news, panel  # noqa: E402
from smartrl.runs import run_record  # noqa: E402
from smartrl.sentiflow import daily, scoring  # noqa: E402


def main() -> None:
    cfg = frozen()["sentiflow"]
    items = pd.read_parquet(news.ITEMS, columns=["id", "headline"])
    if not scoring.SCORES.exists():
        with run_record("P2", "score_corpus", None, "cuda", {"n": len(items)}) as run:
            res = scoring.score_corpus(items["headline"], cfg["seeds"], cfg["scoring"]["max_hours_for_ensemble"],
                                       log=lambda m: print(m, flush=True))
            out = pd.DataFrame({"id": items["id"].to_numpy(), **res["columns"]})
            out.to_parquet(scoring.SCORES, index=False)
            run.set(seeds_used=res["seeds_used"], ensemble=res["ensemble"], hours=res["hours"])
        items_kv = [("sentiflow.scoring.n_headlines", res["n_headlines"]), ("sentiflow.scoring.ensemble", res["ensemble"]),
                    ("sentiflow.scoring.hours", res["hours"]), ("p2.scoring.seeds_used", res["seeds_used"])]
        notes = [] if res["ensemble"] else [
            "sentiflow.scoring.ensemble: five-seed scoring would exceed 12 h, so seed 0 alone scored the corpus (SPEC §3.8)."]
        R.update(items_kv, notes)
    scores = pd.read_parquet(scoring.SCORES).set_index("id")
    del items
    with run_record("P2", "daily_features", None, "cpu", {}):
        for m in MARKETS:
            cal = pd.DatetimeIndex(pd.read_parquet(panel.processed_path(m), columns=["date"])["date"])
            al = pd.read_parquet(news.aligned_path(m))
            daily.build(m, al, scores, cal).to_parquet(daily.daily_path(m), index=False)
            print(m, "daily features written", flush=True)


if __name__ == "__main__":
    main()
