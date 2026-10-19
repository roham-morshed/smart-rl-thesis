"""P3 / E2: MarketMind annual refits (out-of-sample forecast features) and the H4 comparison 2020-2024.

    python scripts/p3_marketmind.py --stage refits   # seed 0, MM / MM_NOSENT / INFORMER / LSTM_PRICE, 2010..2024 + forward
    python scripts/p3_marketmind.py --stage h4       # LSTM, GRU, PATCHTST, DLINEAR seeds 0-2; MM, INFORMER seeds 1-2; 2020-2024
    python scripts/p3_marketmind.py --stage eval     # metrics, DM tests, marketmind.* keys
    python scripts/p3_marketmind.py --smoke          # code check: one refit year, 2 epochs, nothing saved

Every model for predicted year y is trained on data dated <= (y-1)-12-31 (SPEC §4.7); the files carry train_end
so tests/test_leakage.py (test 5) can verify it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, PROCESSED, RESULTS, frozen  # noqa: E402
from smartrl.data import access  # noqa: E402
from smartrl.marketmind import baselines, references  # noqa: E402
from smartrl.marketmind import data as MD  # noqa: E402
from smartrl.marketmind import train as MT  # noqa: E402
from smartrl.marketmind.model import HORIZONS, VOL_HORIZONS  # noqa: E402
from smartrl.runs import run_record  # noqa: E402
from smartrl.stats import forecast as FS  # noqa: E402

FEATURE_VARIANTS = ["MM", "MM_NOSENT", "INFORMER", "LSTM_PRICE"]
H4_NEURAL = ["MM", "LSTM", "GRU", "INFORMER", "PATCHTST", "DLINEAR"]
TEST_YEARS = [2020, 2021, 2022, 2023, 2024]
H4_FILE = PROCESSED / "h4_predictions.parquet"
QCOLS = [f"q{t}_h{h}" for h in HORIZONS for t in (10, 50, 90)]
VCOLS = [f"logRV{h}" for h in VOL_HORIZONS]


def forecast_path(variant: str) -> Path:
    return PROCESSED / f"forecast_{variant}.parquet"


def refit_windows(cfg: dict) -> list[tuple[str, str, str, str]]:
    """(label, train_end, predict_start, predict_end) for the annual refits and the forward model."""
    lo, hi = cfg["marketmind"]["refit_years"]
    out = [(str(y), f"{y - 1}-12-31", f"{y}-01-01", f"{y}-12-31") for y in range(lo, hi + 1)]
    out.append(("forward", "2024-12-31", "2025-01-01", cfg["dates"]["raw_end"]))
    return out


def frame(ws: MD.WindowSet, q_raw: np.ndarray, v_raw: np.ndarray, **const) -> pd.DataFrame:
    df = pd.DataFrame(q_raw.reshape(len(q_raw), -1), columns=QCOLS)
    df[VCOLS] = v_raw
    df.insert(0, "date", pd.to_datetime(ws.date))
    df.insert(0, "market", ws.market)
    for k, v in const.items():
        df[k] = v
    return df


def run_one(variant: str, seed: int, panels, train_end: str, pred: tuple[str, str], cfg: dict, cache: dict,
            log=print) -> pd.DataFrame:
    key = (train_end, pred)
    if key not in cache:
        cache.clear()
        cache[key] = MD.build_refit(panels, train_end, pred, cfg["lookback"], cfg["train"]["early_stop_tail"])
    tr, pr, scales = cache[key]
    model = baselines.build(variant, tr.xp.shape[2], tr.xs.shape[2], cfg)
    log(f"{variant} seed {seed} train_end {train_end}: {int((~tr.is_val).sum())} train / {int(tr.is_val.sum())} val rows")
    model, info = MT.fit(model, tr, cfg, seed, log)
    q, v = MT.predict(model, pr)
    q_raw, v_raw = MD.to_raw(pr.market, q, v, scales)
    return frame(pr, q_raw, v_raw, variant=variant, seed=seed, train_end=pd.Timestamp(train_end),
                 epochs=info["epochs"], best_epoch=info["best_epoch"])


PRICE_ONLY = {"MM_NOSENT", "LSTM_PRICE"}          # do not read the sentiment channels


def stage_refits(cfg: dict, panels, smoke: bool, variants: list[str] | None = None) -> None:
    mm = cfg["marketmind"]
    windows = refit_windows(cfg)[:1] if smoke else refit_windows(cfg)
    cache: dict = {}
    for variant in FEATURE_VARIANTS[:1] if smoke else (variants or FEATURE_VARIANTS):
        if forecast_path(variant).exists() and not smoke:
            print(f"{variant}: exists, skipped", flush=True)
            continue
        parts = []
        with run_record("P3", f"refits_{variant}" + ("_smoke" if smoke else ""), 0, str(MT.DEV), {"windows": len(windows)}):
            for label, te, a, b in windows:
                parts.append(run_one(variant, 0, panels, te, (a, b), mm, cache).assign(refit=label))
        df = pd.concat(parts, ignore_index=True)
        if smoke:
            print(df.head(3).T, flush=True)
            return
        df.to_parquet(forecast_path(variant), index=False)


def stage_h4(cfg: dict, panels) -> None:
    mm = cfg["marketmind"]
    done = pd.read_parquet(H4_FILE) if H4_FILE.exists() else pd.DataFrame(columns=["variant", "seed", "train_end"])
    have = set(zip(done["variant"], done["seed"], pd.to_datetime(done["train_end"])))
    cache: dict = {}
    for y in TEST_YEARS:
        te, pred = f"{y - 1}-12-31", (f"{y}-01-01", f"{y}-12-31")
        for variant in H4_NEURAL:
            for seed in mm["h4_seeds"]:
                if (variant, seed, pd.Timestamp(te)) in have:
                    continue
                if seed == 0 and variant in FEATURE_VARIANTS and forecast_path(variant).exists():
                    f = pd.read_parquet(forecast_path(variant))
                    part = f[f["refit"] == str(y)].drop(columns=["refit"])
                else:
                    with run_record("P3", f"h4_{variant}_{y}", seed, str(MT.DEV), {"train_end": te}):
                        part = run_one(variant, seed, panels, te, pred, mm, cache)
                done = pd.concat([done, part], ignore_index=True)
                done.to_parquet(H4_FILE, index=False)          # checkpoint after every model


def _realised(panels) -> pd.DataFrame:
    rows = []
    for m, mp in panels.items():
        y_ret, y_rv = MD.targets(mp)
        d = pd.DataFrame(y_ret, columns=[f"r_h{h}" for h in HORIZONS])
        d[[f"rv{h}" for h in VOL_HORIZONS]] = y_rv
        d["date"], d["market"] = mp.dates, m
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _references(panels) -> pd.DataFrame:
    """NAIVE quantiles, EWMA and HAR variance forecasts for the test years (models fitted per year)."""
    lam = frozen()["marketmind"]["vol_refs"]["ewma_lambda"]
    out = []
    for m, mp in panels.items():
        ew = references.ewma_var(mp.r1, lam)
        for y in TEST_YEARS:
            te = pd.Timestamp(f"{y - 1}-12-31")
            idx = np.where(mp.dates.year == y)[0]
            q = references.naive_quantiles(mp, te, idx)
            har = references.har_forecast(mp, te, idx)
            d = pd.DataFrame(q.reshape(len(idx), -1), columns=QCOLS)
            d["date"], d["market"] = mp.dates[idx], m
            d["var_EWMA_5"] = d["var_EWMA_20"] = ew[idx]
            d["var_HAR_5"], d["var_HAR_20"] = har[:, 0], har[:, 1]
            out.append(d)
    return pd.concat(out, ignore_index=True)


def stage_eval(cfg: dict, panels) -> None:
    preds = pd.read_parquet(H4_FILE)
    preds["date"] = pd.to_datetime(preds["date"])
    real = _realised(panels)
    refs = _references(panels)
    items, detail = [], {}
    # per model, seed and horizon: pooled metrics over markets and test days
    losses = {}            # (model, h) -> DataFrame(date, market, loss) seed-averaged
    for (variant, seed), g in preds.groupby(["variant", "seed"]):
        j = g.merge(real, on=["market", "date"], how="left")
        for k, h in enumerate(HORIZONS):
            q = j[[f"q10_h{h}", f"q50_h{h}", f"q90_h{h}"]].to_numpy()
            y = j[f"r_h{h}"].to_numpy()
            detail.setdefault(variant, {}).setdefault(f"h{h}", []).append(FS.point_metrics(y, q))
            ok = np.isfinite(y)
            lt = pd.DataFrame({"date": j["date"][ok], "market": j["market"][ok], "loss": FS.pinball(y[ok], q[ok])})
            losses.setdefault((variant, h), []).append(lt)
            for m in MARKETS:
                mm_ = (j["market"] == m).to_numpy() & ok
                detail[variant].setdefault(f"by_market_h{h}", {}).setdefault(m, []).append(
                    float(FS.pinball(y[mm_], q[mm_]).mean()))
        for jv, h in enumerate(VOL_HORIZONS):
            rv = j[f"rv{h}"].to_numpy()
            lv = j[f"logRV{h}"].to_numpy()
            ok = np.isfinite(rv) & (rv > 0)
            detail[variant].setdefault(f"vol_h{h}", []).append(
                {"qlike": float(FS.qlike(rv[ok], np.exp(lv[ok])).mean()),
                 "mse": float(((np.log(rv[ok]) - lv[ok]) ** 2).mean())})
    jr = refs.merge(real, on=["market", "date"], how="left")
    for k, h in enumerate(HORIZONS):
        q = jr[[f"q10_h{h}", f"q50_h{h}", f"q90_h{h}"]].to_numpy()
        y = jr[f"r_h{h}"].to_numpy()
        detail.setdefault("NAIVE", {}).setdefault(f"h{h}", []).append(FS.point_metrics(y, q))
        ok = np.isfinite(y)
        losses[("NAIVE", h)] = [pd.DataFrame({"date": jr["date"][ok], "market": jr["market"][ok],
                                              "loss": FS.pinball(y[ok], q[ok])})]
        for m in MARKETS:
            mm_ = (jr["market"] == m).to_numpy() & ok
            detail["NAIVE"].setdefault(f"by_market_h{h}", {}).setdefault(m, []).append(float(FS.pinball(y[mm_], q[mm_]).mean()))
    for ref in ("EWMA", "HAR"):
        for h in VOL_HORIZONS:
            rv = jr[f"rv{h}"].to_numpy()
            vh = jr[f"var_{ref}_{h}"].to_numpy()
            ok = np.isfinite(rv) & (rv > 0)
            detail.setdefault(ref, {})[f"vol_h{h}"] = [{"qlike": float(FS.qlike(rv[ok], vh[ok]).mean()),
                                                         "mse": float(((np.log(rv[ok]) - np.log(vh[ok])) ** 2).mean())}]
    for fm, d in detail.items():
        for h in HORIZONS:
            rows = d.get(f"h{h}")
            if rows:
                for key in ("pinball", "mae", "rmse", "diracc", "cov80", "width"):
                    v = np.array([r[key] for r in rows])
                    items.append((f"marketmind.h{h}.{fm}.{key}", float(v.mean())))
                    if key == "pinball" and len(v) > 1:
                        items.append((f"marketmind.h{h}.{fm}.pinball_sd", float(v.std(ddof=1))))
                items.append((f"marketmind.h{h}.{fm}.pt_p", float(np.nanmedian([r["pt_p"] for r in rows]))))
            if f"by_market_h{h}" in d and h == 20:
                for m, v in d[f"by_market_h{h}"].items():
                    items.append((f"marketmind.by_market.{m}.h20.{fm}.pinball", float(np.mean(v))))
        for h in VOL_HORIZONS:
            if f"vol_h{h}" in d:
                for key in ("qlike", "mse"):
                    items.append((f"marketmind.vol.h{h}.{fm}.{key}", float(np.mean([r[key] for r in d[f"vol_h{h}"]]))))
    # DM tests: MM minus each comparison model, daily cross-market average of the seed-averaged pinball loss
    dm_detail = {}
    for h in HORIZONS:
        base = pd.concat(losses[("MM", h)]).groupby(["date", "market"])["loss"].mean()
        for fm in ("LSTM", "GRU", "INFORMER", "PATCHTST", "DLINEAR", "NAIVE"):
            other = pd.concat(losses[(fm, h)]).groupby(["date", "market"])["loss"].mean()
            diff = (base - other).dropna().groupby(level="date").mean().sort_index()
            r = FS.diebold_mariano(diff.to_numpy(), h)
            items += [(f"marketmind.dm.h{h}.{fm}.stat", r["stat"]), (f"marketmind.dm.h{h}.{fm}.p", r["p"])]
            dm_detail[f"h{h}_{fm}"] = r
    items.append(("p3.marketmind_eval", {"dm": dm_detail, "n_pred_rows": int(len(preds))}))
    notes = ["marketmind.h*: metrics pooled over the four markets and the test years 2020-2024, averaged over seeds "
             "0-2 (pinball_sd = standard deviation over seeds); pt_p is the median over seeds; NAIVE, EWMA and HAR are "
             "deterministic. marketmind.dm: MM minus model on the daily cross-market average of the seed-averaged "
             "pinball loss (HLN-corrected DM, Newey-West lag h-1); a negative stat favours MarketMind."]
    R.update(items, notes)
    (RESULTS / "p3_marketmind_detail.json").write_text(json.dumps(R._clean(detail), indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="refits", choices=["refits", "h4", "eval"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variants", default="", help="comma list for --stage refits (default: all feature variants)")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    cfg = frozen()
    if args.smoke:
        cfg = {**cfg, "marketmind": {**cfg["marketmind"], "train": {**cfg["marketmind"]["train"], "max_epochs": 2}}}
    panels = MD.load_panels()
    for mp in panels.values():
        access.check(mp.dates, "marketmind_refit")
    price_only = args.stage == "refits" and variants and set(variants) <= PRICE_ONLY
    if not all(mp.has_sentiment for mp in panels.values()) and not args.smoke and not price_only:
        sys.exit("sentiment_daily_<MKT>.parquet missing: run P2 first")
    t0 = time.time()
    {"refits": lambda: stage_refits(cfg, panels, args.smoke, variants), "h4": lambda: stage_h4(cfg, panels),
     "eval": lambda: stage_eval(cfg, panels)}[args.stage]()
    print(f"stage {args.stage} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
