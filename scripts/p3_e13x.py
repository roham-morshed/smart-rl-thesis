"""E13x — leakage decomposition of the base paper's evaluation (pre-registered 2026-10-21, before the G4 lock).

E13 compares the published protocols (PUB_R, PUB_C) with the causal one (CAUSAL). They differ in four ways at once:
the row calendar (calendar days with forward-filled prices versus trading days), the split (random versus
chronological), the news timing (UTC calendar date versus the market session of SPEC §2.3) and the scaling (all
rows versus training rows). E13x removes these one at a time, so the contribution of each to the published
accuracy becomes visible:

    PUB_R ──split──▶ PUB_C ──calendar──▶ X_TRD_UTC ──news timing──▶ X_TRD_SES ──scaling──▶ CAUSAL
    X_TRD_RANDOM : CAUSAL with a random split (overlapping windows shared by train and test)
    X_NEWS_LEAD  : CAUSAL with the next session's news index (diagnostic one-session look-ahead)

Same models, inputs, seeds and training recipe as E13 (SENTIN's optimiser choice of the E13 pilot); the window
ends 2022-03-31 like E13, so no trading result after 2020-01-01 is computed (pre-lock exception (c)).

    python scripts/p3_e13x.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, PROCESSED, RESULTS, frozen  # noqa: E402
from smartrl.data import panel  # noqa: E402
from smartrl.e13 import data as ED  # noqa: E402
from smartrl.e13 import models as EM  # noqa: E402
from smartrl.e13 import train as ET  # noqa: E402
from smartrl.runs import run_record  # noqa: E402
from smartrl.sentiflow import daily  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_e13 import NEURAL, svm_run  # noqa: E402

PROTOCOLS = {
    "X_TRD_UTC": {"calendar": "trading", "news": "utc", "split": "chrono", "scale": "all"},
    "X_TRD_SES": {"calendar": "trading", "news": "session", "split": "chrono", "scale": "all"},
    "X_TRD_RANDOM": {"calendar": "trading", "news": "session", "split": "random", "scale": "train"},
    "X_NEWS_LEAD": {"calendar": "trading", "news": "session_lead", "split": "chrono", "scale": "train"},
}
PRED_FILE = PROCESSED / "e13x_predictions.parquet"


def rows(market: str, spec: dict, kind: str, start: str, end: str) -> pd.DataFrame:
    """Trading-day rows (the E13x protocols all use the market calendar) with the chosen news index."""
    px = pd.read_parquet(panel.processed_path(market)).set_index("date")
    px = px.loc[start:end, ED.INPUTS].copy()
    if market in frozen()["features"].get("volume_disabled", []):
        px["volume"] = np.nan
    px.loc[px["volume"] <= 0, "volume"] = np.nan
    if spec["news"] == "utc":
        s = ED._utc_day_index(market, kind)
        px["S"] = s.reindex(px.index).fillna(0.0).to_numpy()            # every headline of the UTC calendar date
    else:
        d = pd.read_parquet(daily.daily_path(market)).set_index("date")
        col = "sentin_S" if kind == "A" else "S_loc"
        s = d[col]
        if spec["news"] == "session_lead":
            s = s.shift(-1)                                              # news of the next session (look-ahead)
        px["S"] = s.reindex(px.index).fillna(0.0).to_numpy()
    c = px["close"].to_numpy()
    y = np.full(len(c), np.nan)
    y[:-1] = (c[1:] > c[:-1]).astype(float)
    px["y"] = y
    px.index.name = "date"
    return px


def dataset(market: str, spec: dict, kind: str, cfg: dict):
    from smartrl.data.panel import first_common_date
    start, end = first_common_date(cfg["dates"]["raw_end"]), cfg["e13"]["window_end"]
    L = cfg["sentin"]["lookback"]
    r = rows(market, spec, kind, start, end)
    sp = ED.split(len(r), L, "PUB_R" if spec["split"] == "random" else "CAUSAL", cfg)
    fit = np.arange(len(r)) if spec["scale"] == "all" else np.arange(0, max(sp["train"].max(), sp["val"].max()) + 1)
    if spec["split"] == "random" and spec["scale"] == "train":
        fit = np.sort(np.concatenate([sp["train"], sp["val"]]))
    X = ED.scaled_matrix(r, fit)
    y = r["y"].to_numpy()
    return {k: (ED.windows(X, v, L), y[v].astype(int), r.index[v]) for k, v in sp.items()}


def main() -> None:
    cfg = frozen()
    choice = cfg["sentin"]["optimiser_choice"]
    preds, meta = [], {}
    for proto, spec in PROTOCOLS.items():
        for m in MARKETS:
            dA, dS = dataset(m, spec, "A", cfg), dataset(m, spec, "SF", cfg)
            yt, dt = dA["test"][1], dA["test"][2]
            meta[(proto, m)] = {"n_test": int(len(yt)), "up_share": float(yt.mean())}
            score, _ = svm_run(dA)
            preds.append(pd.DataFrame({"protocol": proto, "market": m, "model": "SVM", "seed": -1, "date": dt,
                                       "score": score, "threshold": 0.0, "y": yt}))
            for name in NEURAL:
                d = dS if name == "SENTIN_SF" else dA
                for s in cfg["e13"]["seeds"]:
                    with run_record("P3", f"E13x_{proto}_{m}_{name}", s, str(ET.DEV), {"spec": spec}):
                        model, _ = ET.fit(EM.build(name, cfg), d["train"][0], d["train"][1], d["val"][0], d["val"][1],
                                          choice, s, cfg)
                    preds.append(pd.DataFrame({"protocol": proto, "market": m, "model": name, "seed": s, "date": dt,
                                               "score": ET.predict(model, d["test"][0]), "threshold": 0.5, "y": yt}))
            print(proto, m, "done", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(PRED_FILE, index=False)
    items, tree = [], {}
    for (proto, m, model), g in P.groupby(["protocol", "market", "model"]):
        per_seed = [ET.metrics(h["y"].to_numpy().astype(int), h["score"].to_numpy(), h["threshold"].iloc[0])
                    for _, h in g.groupby("seed")]
        for k in ("acc", "f1", "auc", "prauc"):
            v = np.array([r[k] for r in per_seed])
            items.append((f"e13x.{proto}.{m}.{model}.{k}", float(v.mean())))
            R.put(tree, f"{proto}.{m}.{model}.{k}", float(v.mean()))
            if len(v) > 1:
                items.append((f"e13x.{proto}.{m}.{model}.{k}_sd", float(v.std(ddof=1))))
    for (proto, m), md in meta.items():
        items += [(f"e13x.{proto}.{m}.{k}", v) for k, v in md.items()]
    for proto in PROTOCOLS:
        for model in sorted(P["model"].unique()):
            for k in ("acc", "f1", "auc", "prauc"):
                items.append((f"e13x.{proto}.MEAN.{model}.{k}", float(np.mean([tree[proto][m][model][k] for m in MARKETS]))))
    R.update(items, ["e13x: leakage decomposition pre-registered on 2026-10-21 before the G4 lock (scripts/p3_e13x.py); "
                     "X_NEWS_LEAD is a diagnostic with a deliberate one-session news look-ahead, not a method."])
    print(json.dumps({p: {mo: round(tree[p]["SP500"][mo]["auc"], 3) for mo in tree[p]["SP500"]} for p in tree}, indent=1))


if __name__ == "__main__":
    main()
