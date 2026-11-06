"""P6 / E14: the base paper's task under its exact conditions (docs/P6_BASE_PROTOCOL.md §2).

    python scripts/p6_e14.py --stage prices    # rows, flags, Nifty splice check  -> e14.data.*, e14.causal.zero_change.*
    python scripts/p6_e14.py --stage news      # 2005-2006 items, SentiFlow seed-0 scores, counts -> e14.news.*
    python scripts/p6_e14.py --stage a_paper   # A_PAPER training (seeds 0-2) and test metrics -> e14.a_paper.*
    python scripts/p6_e14.py --stage score     # A_PAPER scores of the market-tagged headlines 2005-01 .. 2022-03
    python scripts/p6_e14.py --stage daily     # daily indices and SPEC §3.9 features per market and grid
    python scripts/p6_e14.py --stage run [--jobs 3]   # the run matrix (resumable)
    python scripts/p6_e14.py --stage eval      # metrics -> e14.<SET>.*, label evidence -> e14.labels.*
    python scripts/p6_e14.py --stage tests     # pre-registered tests -> e14.tests.*, e14.tests_inf.*, e14.claims.*
    python scripts/p6_e14.py --stage outputs   # tables T21, T22, T25 and figures F17, F19
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS  # noqa: E402
from smartrl.p6 import prices as PR  # noqa: E402
from smartrl.p6.cfg import P6_DIR, p6, p6_run  # noqa: E402

INFO = P6_DIR / "e14_data_info.json"


def _info_update(d: dict) -> None:
    P6_DIR.mkdir(parents=True, exist_ok=True)
    cur = json.loads(INFO.read_text(encoding="utf-8")) if INFO.exists() else {}
    cur.update(d)
    INFO.write_text(json.dumps(cur, indent=1, default=str), encoding="utf-8")


def stage_prices() -> None:
    e = p6()["e14"]
    a, b = pd.Timestamp(e["window"][0]), pd.Timestamp(e["window"][1])
    items, info = [], {}
    with p6_run("E14_prices"):
        for m in MARKETS:
            bars, extra = PR.trading_bars(m)
            cal = PR.calendar_rows(bars)
            w = cal[(cal.index >= a) & (cal.index <= b)]
            n_cal, n_lab = len(w), len(w) - 1
            trd = bars[(bars.index >= a) & (bars.index <= b)]
            c = trd["close"].to_numpy()
            zero = int((c[1:] == c[:-1]).sum())
            d = {"calendar_rows": n_cal, "labelled_rows": n_lab, "trading_rows": int(len(trd)),
                 "first_date": str(w.index.min().date()), "last_date": str(w.index.max().date()),
                 "history_first_trading_date": str(bars.index.min().date()),
                 "informative_share": float(w["informative_row"].iloc[:-1].mean())}
            if "nifty_splice" in extra:
                d["nifty_splice_maxdiff"] = extra["nifty_splice"]["max_rel_diff"]
                d["nifty_splice"] = extra["nifty_splice"]
                if not extra["nifty_splice"]["ok"]:
                    sys.exit(f"Nifty splice check failed: {extra['nifty_splice']}")
            info[m] = {**d, "zero_change_sessions": zero}
            items += [(f"e14.data.{m}.{k}", v) for k, v in d.items() if k in
                      ("calendar_rows", "labelled_rows", "trading_rows", "first_date", "last_date", "nifty_splice_maxdiff")]
            items.append((f"e14.causal.zero_change.{m}", zero))
            if n_lab != e["expected"]["labelled_rows"]:
                sys.exit(f"{m}: {n_lab} labelled rows, expected {e['expected']['labelled_rows']}")
            print(m, d, "zero-change sessions:", zero, flush=True)
    _info_update({"prices": info})
    R.update(items, ["e14.data.*: calendar rows 2005-01-01..2022-03-31 (the last date only provides the next close); "
                     "NIFTY 50 before 2007-09-17 from NSE Indices (niftyindices.com), Yahoo from 2007-09-17; "
                     "nifty_splice_maxdiff = max relative close difference on the first 20 common days.",
                     "e14.causal.zero_change: trading-day samples of 2005-01-01..2022-03-31 with C_{t+1} = C_t "
                     "(the only rows where labels GE and GT differ under CAUSAL)."])


def stage_news() -> None:
    from smartrl.p6 import news as NW
    with p6_run("E14_news_2005_2006"):
        ex = NW.extract(print)
    with p6_run("E14_news_score_2005_2006", 0, "cuda"):
        sc = NW.score_new(print)
    counts = NW.counts_by_year()
    _info_update({"news_extract": {k: v for k, v in ex.items() if k != "_dedup"}, "news_dedup": ex["_dedup"],
                  "news_scoring_2005_2006": sc})
    items = []
    for (m, y), v in counts.items():
        items += [(f"e14.news.{m}.{y}.local", v["local"]), (f"e14.news.{m}.{y}.global", v["global"])]
    R.update(items, ["e14.news: headline counts per UTC year of the multisource corpus (P1 rules; 2005-2006 extracted "
                     "in P6 with the same code), local = the market's tag, global = every kept headline."])
    print(ex["_dedup"], sc, flush=True)


def stage_a_paper(epochs: int | None = None) -> None:
    from smartrl.data import sentiment_sets as S
    from smartrl.p6 import a_paper as A
    from smartrl.sentiflow import word2vec
    df, _ = S.load()
    wv = word2vec.load()
    per = {}
    for s in p6()["e14"]["seeds"]:
        with p6_run("E14_A_PAPER", s, "cuda", {"epochs": epochs or p6()["e14"]["a_paper"]["epochs"]}) as run:
            model, vocab, info = A.train(df, wv, s, epochs)
            ev = A.evaluate(model, vocab, df)
            run.set(metrics=ev, final_train_loss=info["train_loss"][-1], seconds=info["seconds"])
        if epochs is None:
            A.save(model, vocab, s, info)
        per[s] = ev
        print("seed", s, ev, flush=True)
    if epochs is not None:
        return
    items = []
    for dset in per[0]:
        for k in ("acc", "macro_f1"):
            v = np.array([per[s][dset][k] for s in per])
            items += [(f"e14.a_paper.{dset}.{k}", float(v.mean())), (f"e14.a_paper.{dset}.{k}_sd", float(v.std(ddof=1)))]
    R.update(items, ["e14.a_paper: the base paper's GRU-CNN sentiment classifier trained with its Table 2 recipe "
                     "(400 epochs, lr 0.01, Adam, batch 64, dropout 0.1, pool 2, no early stopping), mean and sd over "
                     "seeds 0-2 on the SPEC §2.4 test splits."])


def stage_score() -> None:
    from smartrl.p6 import news as NW
    from smartrl.sentiflow import word2vec
    with p6_run("E14_A_PAPER_scoring", None, "cuda") as run:
        res = NW.score_a_paper(word2vec.load(), print)
        run.set(**res)
    _info_update({"a_paper_scoring": res})
    print(res, flush=True)


def stage_daily() -> None:
    from smartrl.p6 import news as NW
    with p6_run("E14_daily_indices"):
        NW.daily_tables(print)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["prices", "news", "a_paper", "a_paper_smoke", "score", "daily", "run", "eval", "tests", "outputs"])
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    if args.stage == "prices":
        stage_prices()
    elif args.stage == "news":
        stage_news()
    elif args.stage == "a_paper":
        stage_a_paper()
    elif args.stage == "a_paper_smoke":
        stage_a_paper(epochs=2)
    elif args.stage == "score":
        stage_score()
    elif args.stage == "daily":
        stage_daily()
    elif args.stage == "run":
        from smartrl.p6 import e14run
        e14run.stage_run(args)
    else:
        from smartrl.p6 import e14eval
        getattr(e14eval, f"stage_{args.stage}")(args)


if __name__ == "__main__":
    main()
