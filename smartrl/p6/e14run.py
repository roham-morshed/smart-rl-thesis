"""E14 run matrix (protocol §2.5), metrics (§2.6) and pre-registered tests (§2.7).

Predictions: data/processed/p6/e14_preds/<SET>/<MODEL>/<MKT|ALL>_s<seed>.parquet with market, date, score, y,
informative. Score = P(up) (threshold 0.5), the SVM decision value (threshold 0) or MM_DIR's q50 at h = 1
(prediction 1[q50 >= 0] for GE, 1[q50 > 0] for GT). Jobs skip existing files (resumable).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .. import results as R
from ..config import MARKETS, frozen
from ..sentiflow.train import set_seed
from . import e14data as ED
from . import models as PM
from .cfg import P6_DIR, p6, p6_run

PRED = P6_DIR / "e14_preds"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PER_MARKET = ["SVM_P", "CNN_P", "LSTM_P", "GRU_P", "SENTIN_PAPER", "SENTIN_E13", "SENTIN_SF_PAPER", "SENTIN_SF_E13"]
POOLED = ["MM_CLS", "MM_DIR"]


def pred_path(setting: str, model: str, mkt: str, seed: int):
    return PRED / setting / model / f"{mkt}_s{seed}.parquet"


# ---------------------------------------------------------------- recipes
def _loop(model, X, y, epochs: int, batch: int, opt, loss_fn, seed: int):
    rng = np.random.default_rng(seed)
    Xt, yt = torch.from_numpy(X).to(DEV), torch.from_numpy(y.astype(np.float32)).to(DEV)
    hist = []
    for _ in range(epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(yt))).to(DEV)
        tot = 0.0
        for i in range(0, len(yt), batch):
            sel = order[i:i + batch]
            loss = loss_fn(model(Xt[sel]), yt[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(sel)
        hist.append(tot / len(yt))
    return model.eval(), {"epochs": epochs, "final_train_loss": hist[-1]}


@torch.no_grad()
def predict_prob(model, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(DEV))).float().cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out)


def fit_paper_nn(model, X, y, seed):
    c = p6()["e14"]["paper_nn"]
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    return _loop(model, X, y, c["epochs"], c["batch"], opt, F.binary_cross_entropy_with_logits, seed)


def fit_sentin_paper(model, X, y, seed):
    c = p6()["e14"]["sentin_paper"]
    model = model.to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=c["lr"])
    return _loop(model, X, y, c["epochs"], c["batch"], opt, lambda z, t: F.mse_loss(torch.sigmoid(z), t), seed)


def fit_e13(name, X_tr, y_tr, X_va, y_va, seed):
    from ..e13 import models as EM
    from ..e13 import train as ET
    cfg = frozen()
    model, info = ET.fit(EM.build("SENTIN", cfg), X_tr, y_tr, X_va, y_va, cfg["sentin"]["optimiser_choice"], seed, cfg)
    return model, info


def fit_mm_cls(xp, xs, mk, y, is_val, seed):
    from sklearn.metrics import roc_auc_score as auc
    c = p6()["e14"]["mm_cls"]
    set_seed(seed)
    model = PM.build("MM_CLS", frozen(), p6(), xp.shape[2], xs.shape[2]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    rng = np.random.default_rng(seed)
    T = lambda a, idx, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a[idx])).to(DEV, dt)
    best, best_auc, bad, hist = None, -np.inf, 0, []
    for ep in range(c["max_epochs"]):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, int(np.ceil(len(tr) / c["batch"])))):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                z = model(T(xp, b), T(xs, b), T(mk, b, torch.long))
            loss = F.binary_cross_entropy_with_logits(z.float(), T(y, b))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            scaler.step(opt)
            scaler.update()
        pv = mm_cls_predict(model, xp[va], xs[va], mk[va])
        a = auc(y[va], pv) if len(np.unique(y[va])) == 2 else np.nan
        hist.append(a)
        if np.isfinite(a) and a > best_auc:
            best_auc, best, bad = a, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    model.load_state_dict(best)
    return model.eval(), {"epochs": len(hist), "best_val_auc": float(best_auc), "best_epoch": int(np.nanargmax(hist))}


@torch.no_grad()
def mm_cls_predict(model, xp, xs, mk, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(xp), batch):
        t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a[i:i + batch])).to(DEV, dt)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            z = model(t(xp), t(xs), t(mk, torch.long))
        out.append(torch.sigmoid(z.float()).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- jobs
def _save(setting, model, mkt, seed, frame: pd.DataFrame) -> None:
    p = pred_path(setting, model, mkt, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p, index=False)


def _frame(mr: ED.MarketRows, idx: np.ndarray, score: np.ndarray, seed: int) -> pd.DataFrame:
    return pd.DataFrame({"market": mr.market, "date": mr.frame.index[mr.pos[idx]], "score": score.astype(float),
                         "y": mr.y[idx], "informative": mr.informative[idx], "seed": seed})


def job_market(setting: str, model: str, market: str, seed: int) -> str:
    if pred_path(setting, model, market, seed).exists():
        return "skip"
    torch.set_num_threads(1)
    t0 = time.time()
    mr = ED.market_rows(market, setting)
    X, _ = ED.ohlcv_windows(mr, ED.INDEX[model], setting)
    tr_all, te = mr.train, mr.test
    with p6_run(f"E14_{setting}_{model}_{market}", seed if model != "SVM_P" else None,
                "cpu" if model == "SVM_P" else str(DEV), {"setting": setting, "model": model, "market": market}) as run:
        if model == "SVM_P":
            from sklearn.svm import SVC
            c = p6()["e14"]["svm"]
            flat = lambda A: A[:, -c["window"]:, :].reshape(len(A), -1)
            m = SVC(kernel="rbf", C=c["C"], gamma=c["gamma"]).fit(flat(X[tr_all]), mr.y[tr_all])
            score, info = m.decision_function(flat(X[te])), {}
        elif model in ("CNN_P", "LSTM_P", "GRU_P"):
            set_seed(seed)
            net, info = fit_paper_nn(PM.build(model, frozen(), p6()), X[tr_all], mr.y[tr_all], seed)
            score = predict_prob(net, X[te])
        elif model in ("SENTIN_PAPER", "SENTIN_SF_PAPER"):
            set_seed(seed)
            net, info = fit_sentin_paper(PM.build(model, frozen(), p6()), X[tr_all], mr.y[tr_all], seed)
            score = predict_prob(net, X[te])
        else:                                                    # SENTIN_E13, SENTIN_SF_E13: frozen E13 recipe
            tr = np.setdiff1d(tr_all, mr.val)
            net, info = fit_e13(model, X[tr], mr.y[tr], X[mr.val], mr.y[mr.val], seed)
            score = predict_prob(net, X[te])
        run.set(info={k: v for k, v in info.items() if not isinstance(v, list)}, seconds=time.time() - t0)
    _save(setting, model, market, seed, _frame(mr, te, score, seed))
    return f"{setting} {model} {market} s{seed}: {time.time() - t0:.0f}s"


def job_pooled(setting: str, model: str, seed: int) -> str:
    if pred_path(setting, model, "ALL", seed).exists():
        return "skip"
    torch.set_num_threads(1)
    t0 = time.time()
    rows = ED.all_markets(setting)
    parts = {}
    for m, mr in rows.items():
        xp, xs = ED.mm_arrays(mr, setting)
        parts[m] = (mr, xp, xs)
    s = ED.parse(setting)
    with p6_run(f"E14_{setting}_{model}", seed, str(DEV), {"setting": setting, "model": model}) as run:
        if model == "MM_CLS":
            XP, XS, MK, Y, IV = [], [], [], [], []
            for mr, xp, xs in parts.values():
                sel = np.r_[np.setdiff1d(mr.train, mr.val), mr.val]          # training rows, validation last
                XP.append(xp[sel]); XS.append(xs[sel]); MK.append(np.full(len(sel), ED.market_id(mr.market)))
                Y.append(mr.y[sel]); IV.append(np.isin(sel, mr.val))
            net, info = fit_mm_cls(np.concatenate(XP), np.concatenate(XS), np.concatenate(MK),
                                   np.concatenate(Y).astype(np.float32), np.concatenate(IV), seed)
            frames = []
            for mr, xp, xs in parts.values():
                sc = mm_cls_predict(net, xp[mr.test], xs[mr.test], np.full(len(mr.test), ED.market_id(mr.market)))
                frames.append(_frame(mr, mr.test, sc, seed))
        else:                                                    # MM_DIR: frozen MarketMind recipe, score = q50(h=1)
            from ..marketmind import baselines as MB
            from ..marketmind import train as MT
            from ..marketmind.data import WindowSet
            mm = frozen()["marketmind"]
            P, Sx, MKs, YR, YV, IV, MKT, DT = [], [], [], [], [], [], [], []
            sd = {}
            for m, (mr, xp, xs) in parts.items():
                y_ret, y_rv = ED.mm_targets(mr)
                ok = np.isfinite(y_ret).all(1) & np.isfinite(y_rv).all(1)
                if s["base"] != "PUB_R":                       # chronological: targets end inside the training rows
                    last = mr.pos[mr.train[-1]]
                    ok &= (mr.pos + 20) <= last
                use = np.where(ok)[0]
                use = use[np.isin(use, mr.train)]
                sd_ret = np.nanstd(y_ret[use], axis=0, ddof=1)
                sd_r1 = float(np.nanstd(mr.frame["r1"].to_numpy(float)[mr.pos[use]], ddof=1))
                sd[m] = (sd_ret, sd_r1)
                yv = np.log(np.maximum(y_rv, 1e-12)) - 2 * np.log(sd_r1)
                P.append(xp[use]); Sx.append(xs[use]); MKs.append(np.full(len(use), ED.market_id(m)))
                YR.append((y_ret / sd_ret)[use]); YV.append(yv[use]); IV.append(np.isin(use, mr.val))
                MKT.append(np.full(len(use), m)); DT.append(mr.frame.index[mr.pos[use]].values)
            ws = WindowSet(np.concatenate(P), np.concatenate(Sx), np.concatenate(MKs),
                           np.concatenate(YR).astype(np.float32), np.concatenate(YV).astype(np.float32),
                           np.concatenate(MKT), np.concatenate(DT), np.concatenate(IV))
            net, info = MT.fit(MB.build("MM", ws.xp.shape[2], ws.xs.shape[2], mm), ws, mm, seed)
            info = {k: v for k, v in info.items() if k != "history"}
            frames = []
            for m, (mr, xp, xs) in parts.items():
                n = len(mr.test)
                wt = WindowSet(xp[mr.test], xs[mr.test], np.full(n, ED.market_id(m)), np.zeros((n, 3), np.float32),
                               np.zeros((n, 2), np.float32), np.full(n, m), mr.frame.index[mr.pos[mr.test]].values,
                               np.zeros(n, bool))
                q, _ = MT.predict(net, wt)
                frames.append(_frame(mr, mr.test, q[:, 0, 1] * sd[m][0][0], seed))
        run.set(info=info, seconds=time.time() - t0)
    _save(setting, model, "ALL", seed, pd.concat(frames, ignore_index=True))
    return f"{setting} {model} s{seed}: {time.time() - t0:.0f}s"


def job_list() -> list[tuple]:
    e = p6()["e14"]
    jobs = []
    groups = [(e["settings"]["full"], e["models"]["full"]), (e["settings"]["gt"], e["models"]["gt"]),
              (e["settings"]["la"], e["models"]["la"]), (e["settings"]["robust"], e["models"]["robust"])]
    for settings, models in groups:
        for st in settings:
            for mo in models:
                if mo == "MM_DIR" and st.endswith("_GT"):
                    continue                                   # scored only: the GE fit with the GT rule
                if mo in POOLED:
                    jobs += [("pooled", st, mo, None, s) for s in e["seeds"]]
                elif mo == "SVM_P":
                    jobs += [("market", st, mo, m, -1) for m in MARKETS]
                else:
                    jobs += [("market", st, mo, m, s) for m in MARKETS for s in e["seeds"]]
    jobs.sort(key=lambda j: (j[0] != "pooled", j[1]))
    return jobs


def _run_one(j):
    try:
        return job_pooled(j[1], j[2], j[4]) if j[0] == "pooled" else job_market(j[1], j[2], j[3], j[4])
    except Exception as ex:                                     # a failed job is reported and can be re-run
        import traceback
        return f"FAILED {j}: {type(ex).__name__}: {ex}\n{traceback.format_exc()[-1500:]}"


def stage_run(args) -> None:
    from joblib import Parallel, delayed
    jobs = job_list()
    todo = [j for j in jobs if not pred_path(j[1], j[2], j[3] if j[0] == "market" else "ALL", j[4]).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    res = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(delayed(_run_one)(j) for j in todo)
    fails = [r for r in res if r.startswith("FAILED")]
    for r in res:
        print(r, flush=True)
    if fails:
        raise SystemExit(f"{len(fails)} jobs failed")
