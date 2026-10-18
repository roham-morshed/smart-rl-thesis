"""P3 / E13: the base paper's own task (SPEC §15) and the SENTIN refits for BASE (§9.1, §4.7).

    python scripts/p3_e13.py --stage pilot    # SENTIN, CAUSAL validation, seed 0: sgd_mse versus adam_bce
    python scripts/p3_e13.py --stage main     # PUB_R, PUB_C, CAUSAL x 4 markets x SVM, CNN, LSTM, GRU, SENTIN,
                                              #   SENTIN_SF (seeds 0-2), MM_DIR (CAUSAL only)
    python scripts/p3_e13.py --stage refits   # SENTIN annual refits 2010..2024 + forward -> forecast_SENTIN.parquet
    python scripts/p3_e13.py --smoke          # code check on one market, 2 epochs, nothing saved
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import FROZEN, MARKETS, PROCESSED, RESULTS, frozen  # noqa: E402
from smartrl.e13 import data as ED  # noqa: E402
from smartrl.e13 import models as EM  # noqa: E402
from smartrl.e13 import train as ET  # noqa: E402
from smartrl.marketmind import baselines as MB  # noqa: E402
from smartrl.marketmind import data as MD  # noqa: E402
from smartrl.marketmind import train as MT  # noqa: E402
from smartrl.runs import run_record  # noqa: E402

PRED_FILE = PROCESSED / "e13_predictions.parquet"
NEURAL = ["CNN", "LSTM", "GRU", "SENTIN", "SENTIN_SF"]


def window(cfg: dict) -> tuple[str, str]:
    from smartrl.data import panel
    return panel.first_common_date(cfg["dates"]["raw_end"]), cfg["e13"]["window_end"]


def dataset(market: str, protocol: str, kind: str, cfg: dict):
    """Scaled windows for one market and protocol; kind 'A' (SENTIN index) or 'SF' (SentiFlow index)."""
    start, end = window(cfg)
    L = cfg["sentin"]["lookback"]
    r = ED.rows(market, protocol, kind, start, end)
    sp = ED.split(len(r), L, protocol, cfg)
    fit_rows = np.arange(len(r)) if protocol != "CAUSAL" else np.arange(0, sp["val"].max() + 1)
    X = ED.scaled_matrix(r, fit_rows)
    y = r["y"].to_numpy()
    return {k: (ED.windows(X, v, L), y[v].astype(int), r.index[v]) for k, v in sp.items()}


def svm_run(d) -> tuple[np.ndarray, dict]:
    flat = lambda X: X[:, -frozen()["e13"]["svm"]["window"]:, :].reshape(len(X), -1)
    best, best_auc = None, -np.inf
    for C in frozen()["e13"]["svm"]["C"]:
        m = SVC(kernel="rbf", C=C, gamma=frozen()["e13"]["svm"]["gamma"]).fit(flat(d["train"][0]), d["train"][1])
        a = ET.auc(d["val"][1], m.decision_function(flat(d["val"][0])))
        if a > best_auc:
            best, best_auc, best_c = m, a, C
    return best.decision_function(flat(d["test"][0])), {"C": best_c, "val_auc": best_auc}


def stage_pilot(cfg: dict, smoke: bool) -> str:
    aucs = {}
    for choice in ("sgd_mse", "adam_bce"):
        vals = []
        for m in MARKETS[:1] if smoke else MARKETS:
            d = dataset(m, "CAUSAL", "A", cfg)
            with run_record("P3", f"E13_pilot_{choice}_{m}" + ("_smoke" if smoke else ""), 0, str(ET.DEV), {}):
                _, info = ET.fit(EM.build("SENTIN", cfg), d["train"][0], d["train"][1], d["val"][0], d["val"][1],
                                 choice, 0, cfg)
            vals.append(info["best_val_auc"])
            print(choice, m, info, flush=True)
        aucs[choice] = float(np.mean(vals))
    choice = max(aucs, key=aucs.get)
    if not smoke:
        R.update([("e13.sentin.choice", choice)] + [(f"e13.sentin.val_auc.{k}", v) for k, v in aucs.items()])
        t = FROZEN.read_text(encoding="utf-8")
        t = re.sub(r"optimiser_choice: SET_IN_E13", f"optimiser_choice: {choice}", t)
        FROZEN.write_text(t, encoding="utf-8")
    print("SENTIN choice:", choice, aucs, flush=True)
    return choice


def mm_dir(cfg: dict, seeds: list[int], smoke: bool) -> pd.DataFrame:
    """MarketMind trained on the CAUSAL split (each market's first 80 % of trading days); score = q50 at h = 1."""
    panels = MD.load_panels()
    te, pr = {}, {}
    for m in MARKETS:
        d = dataset(m, "CAUSAL", "A", cfg)
        te[m] = str(d["val"][2].max().date())
        pr[m] = (str(d["test"][2].min().date()), str(d["test"][2].max().date()))
    mm = cfg["marketmind"] if not smoke else {**cfg["marketmind"], "train": {**cfg["marketmind"]["train"], "max_epochs": 2}}
    tr, pw, scales = MD.build_refit(panels, te, pr, mm["lookback"], mm["train"]["early_stop_tail"])
    out = []
    for s in seeds:
        with run_record("P3", "E13_MM_DIR" + ("_smoke" if smoke else ""), s, str(MT.DEV), {}):
            model, _ = MT.fit(MB.build("MM", tr.xp.shape[2], tr.xs.shape[2], mm), tr, mm, s)
        q, v = MT.predict(model, pw)
        q_raw, _ = MD.to_raw(pw.market, q, v, scales)
        out.append(pd.DataFrame({"protocol": "CAUSAL", "market": pw.market, "model": "MM_DIR", "seed": s,
                                 "date": pd.to_datetime(pw.date), "score": q_raw[:, 0, 1], "threshold": 0.0}))
    return pd.concat(out, ignore_index=True)


def stage_main(cfg: dict, choice: str, smoke: bool) -> None:
    seeds = [0] if smoke else cfg["e13"]["seeds"]
    preds, meta = [], {}
    if smoke:
        cfg = {**cfg, "sentin": {**cfg["sentin"], "train": {**cfg["sentin"]["train"], "max_epochs": 2}}}
    for proto in cfg["e13"]["protocols"]:
        for m in MARKETS[:1] if smoke else MARKETS:
            dA, dS = dataset(m, proto, "A", cfg), dataset(m, proto, "SF", cfg)
            yt, dt = dA["test"][1], dA["test"][2]
            meta[(proto, m)] = {"n_train": int(len(dA["train"][1]) + len(dA["val"][1])), "n_test": int(len(yt)),
                                "test_start": str(dt.min().date()), "test_end": str(dt.max().date()),
                                "up_share": float(yt.mean())}
            score, info = svm_run(dA)
            preds.append(pd.DataFrame({"protocol": proto, "market": m, "model": "SVM", "seed": -1, "date": dt,
                                       "score": score, "threshold": 0.0, "y": yt}))
            for name in NEURAL:
                d = dS if name == "SENTIN_SF" else dA
                for s in seeds:
                    with run_record("P3", f"E13_{proto}_{m}_{name}" + ("_smoke" if smoke else ""), s, str(ET.DEV), {}):
                        model, info = ET.fit(EM.build(name, cfg), d["train"][0], d["train"][1], d["val"][0],
                                             d["val"][1], choice, s, cfg)
                    preds.append(pd.DataFrame({"protocol": proto, "market": m, "model": name, "seed": s, "date": dt,
                                               "score": ET.predict(model, d["test"][0]), "threshold": 0.5, "y": yt}))
            print(proto, m, "done", flush=True)
    mmd = mm_dir(cfg, seeds, smoke)
    ylab = pd.concat([p[p["model"] == "SVM"][["protocol", "market", "date", "y"]] for p in preds])
    preds.append(mmd.merge(ylab[ylab["protocol"] == "CAUSAL"], on=["protocol", "market", "date"], how="left"))
    P = pd.concat(preds, ignore_index=True)
    P = P[P["y"].notna()].reset_index(drop=True)
    items = []
    for (proto, m, model), g in P.groupby(["protocol", "market", "model"]):
        per_seed = [ET.metrics(h["y"].to_numpy().astype(int), h["score"].to_numpy(), h["threshold"].iloc[0])
                    for _, h in g.groupby("seed")]
        for k in ("acc", "f1", "auc", "prauc"):
            v = np.array([r[k] for r in per_seed])
            items.append((f"e13.{proto}.{m}.{model}.{k}", float(v.mean())))
            if len(v) > 1:
                items.append((f"e13.{proto}.{m}.{model}.{k}_sd", float(v.std(ddof=1))))
    for (proto, m), md in meta.items():
        items += [(f"e13.{proto}.{m}.{k}", v) for k, v in md.items()]
    if smoke:
        print(json.dumps({k: v for k, v in items if ".SENTIN." in k or "up_share" in k}, indent=1, default=str))
        return
    tree = {}
    for k, v in items:
        R.put(tree, k, v)
    for proto in cfg["e13"]["protocols"]:
        models = {mo for (p_, _, mo) in P.groupby(["protocol", "market", "model"]).groups if p_ == proto}
        for model in models:
            for k in ("acc", "f1", "auc", "prauc"):
                items.append((f"e13.{proto}.MEAN.{model}.{k}",
                              float(np.mean([tree["e13"][proto][m][model][k] for m in MARKETS]))))
    P.to_parquet(PRED_FILE, index=False)
    R.update(items, ["e13.*: means (and sd) over seeds 0-2 for neural models; SVM deterministic; MM_DIR CAUSAL only. "
                     "n_train counts the training and early-stopping samples (80 % of the samples)."])


def sentin_refits(cfg: dict, choice: str, smoke: bool) -> None:
    """SENTIN refits on the CAUSAL setup for every predicted year (seed 0): P(up) for the BASE method."""
    from smartrl.data import panel
    lo, hi = cfg["marketmind"]["refit_years"]
    wins = [(f"{y - 1}-12-31", f"{y}-01-01", f"{y}-12-31", str(y)) for y in range(lo, hi + 1)]
    wins.append(("2024-12-31", "2025-01-01", cfg["dates"]["raw_end"], "forward"))
    if smoke:
        wins = wins[:1]
        cfg = {**cfg, "sentin": {**cfg["sentin"], "train": {**cfg["sentin"]["train"], "max_epochs": 2}}}
    L = cfg["sentin"]["lookback"]
    start = panel.first_common_date(cfg["dates"]["raw_end"])
    out = []
    for m in MARKETS[:1] if smoke else MARKETS:
        r = ED.rows(m, "CAUSAL", "A", start, cfg["dates"]["raw_end"])
        dates = r.index
        for te, a, b, label in wins:
            last_tr = int(np.where(dates <= pd.Timestamp(te))[0].max())
            idx = np.arange(L - 1, last_tr)                       # label of t uses C_{t+1} <= train end
            n_va = int(round(cfg["e13"]["split"]["early_stop_share_of_train"] * len(idx)))
            tr_i, va_i = idx[:-n_va], idx[-n_va:]
            X = ED.scaled_matrix(r, np.arange(0, last_tr + 1))
            y = r["y"].to_numpy()
            pi = np.where((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))[0]
            with run_record("P3", f"SENTIN_refit_{m}_{label}" + ("_smoke" if smoke else ""), 0, str(ET.DEV), {}):
                model, info = ET.fit(EM.build("SENTIN", cfg), ED.windows(X, tr_i, L), y[tr_i].astype(int),
                                     ED.windows(X, va_i, L), y[va_i].astype(int), choice, 0, cfg)
            out.append(pd.DataFrame({"market": m, "date": dates[pi], "p_up": ET.predict(model, ED.windows(X, pi, L)),
                                     "train_end": pd.Timestamp(te), "refit": label, "seed": 0}))
            print(m, label, info, flush=True)
    df = pd.concat(out, ignore_index=True)
    if smoke:
        print(df.head())
        return
    df.to_parquet(PROCESSED / "forecast_SENTIN.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot", choices=["pilot", "main", "refits"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = frozen()
    choice = cfg["sentin"]["optimiser_choice"]
    if args.stage == "pilot":
        stage_pilot(cfg if not args.smoke else {**cfg, "sentin": {**cfg["sentin"], "train": {**cfg["sentin"]["train"],
                                                                                         "max_epochs": 2}}}, args.smoke)
        return
    if choice == "SET_IN_E13":
        if not args.smoke:
            sys.exit("run --stage pilot first")
        choice = "adam_bce"
    if args.stage == "main":
        stage_main(cfg, choice, args.smoke)
    else:
        sentin_refits(cfg, choice, args.smoke)


if __name__ == "__main__":
    main()
