"""P2 step 1: domain Word2Vec (SPEC §3.1) on the deduplicated headlines dated <= 2019-12-31."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import frozen  # noqa: E402
from smartrl.data import news  # noqa: E402
from smartrl.runs import run_record  # noqa: E402
from smartrl.sentiflow import word2vec  # noqa: E402


def main() -> None:
    end = pd.Timestamp(frozen()["sentiflow"]["word2vec"]["corpus_end"] + " 23:59:59", tz="UTC")
    items = pd.read_parquet(news.ITEMS, columns=["ts_utc", "headline"])
    heads = items.loc[pd.to_datetime(items["ts_utc"], utc=True) <= end, "headline"].tolist()
    del items
    with run_record("P2", "word2vec", 2026, "cpu", {"n_headlines": len(heads)}) as run:
        wv = word2vec.train(heads)
        run.set(vocab_size=len(wv.index_to_key))
    R.update([("data.word2vec.n_headlines", len(heads)), ("data.word2vec.vocab_size", len(wv.index_to_key))])
    print(len(heads), len(wv.index_to_key))


if __name__ == "__main__":
    main()
