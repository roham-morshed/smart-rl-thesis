"""P4 diagnostic (2026-10-21, before the G4 lock): checkpoint-selection bias of the PPO agent on development data.

Observation that triggered it (dev_trials.jsonl): every PPO configuration has a development-validation Sharpe of
0.8-1.25 but a development-test Sharpe of -0.08 to 0.20, against 0.72 for buy and hold (COMBINED). The frozen rule
keeps the best of 30 validation checkpoints on one short validation window, which favours policies that were lucky
on that window.

What this script does (development data 2010-2019 only): retrain the selected configuration
(R2_l10_d1.0_lr1e-4, curriculum with the selected STM, seeds 0-2, four markets), keep both the best-validation
checkpoint (frozen rule) and the final policy (no selection; the learning rate decays linearly to 0), and evaluate
both — single seeds, the seed ensemble alone, and inside AWS with the selected AWS parameters — on dev-val and
dev-test. All variants are appended to results/dev_trials.jsonl (they count in DSR and PBO).

Decision rule fixed before the run: the final-checkpoint rule is proposed to Roham at G4 only if it raises the
COMBINED dev-test Sharpe of the full system (AWS) over the frozen rule; otherwise the frozen rule stays.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p4_dev as D  # noqa: E402
from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, RESULTS, frozen  # noqa: E402
from smartrl.runs import run_record  # noqa: E402

CFG = "R2_l10_d1.0_lr1e-4"
REWARD = {"name": "R2", "lambda1": 10, "lambda2": 1.0}
OUT_DIR = D.POLICIES / "ckpt_diag" / CFG


def job(m: str, seed: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from smartrl.agents.ppo import TrainSpec, save_policy, train_ppo
    from smartrl.env.portfolio import RewardSpec
    from smartrl.env.trading_env import MarketData
    cfg = frozen()
    inp = D.market_inputs(m)
    f = inp["fold"]
    e_sa = np.nan_to_num(D.stm_targets({"form": "STM-1", "kappa": 0.5, "z0": 0.5}, inp), nan=1.0)
    data = MarketData(inp["X"], np.nan_to_num(inp["ret_tr"]), e_sa=e_sa)
    spec = TrainSpec(data, f.train, f.val, RewardSpec(**REWARD), 1e-4, seed, curriculum=True,
                     total_timesteps=cfg["ppo"]["total_timesteps"], eval_every=cfg["ppo"]["eval_every"], env_cfg=D.env_cfg())
    t0 = time.time()
    out = train_ppo(spec, cfg)
    save_policy(out, OUT_DIR / m / f"seed{seed}")
    return {"market": m, "seed": seed, "seconds": time.time() - t0, "best_val_sharpe": out["info"]["best_val_sharpe"]}


def evaluate() -> dict:
    from smartrl.agents.ppo import load_policy
    from smartrl.aws.aws import run_aws
    from smartrl.env.evaluate import PolicyEnsemble, path_ensemble, path_from_targets
    from smartrl.env.trading_env import window_matrix
    from smartrl.evaluation.metrics import combined, max_drawdown, sharpe, trading_metrics
    from smartrl.stm import stm as STM
    c = D.env_cfg()
    series = {}                     # (variant, split, market) -> (r Series, held)
    single = {}
    for m in MARKETS:
        inp = D.market_inputs(m)
        f, ret = inp["fold"], np.nan_to_num(inp["ret_tr"])
        win = window_matrix(inp["X"], c.window)
        e_sa = np.nan_to_num(D.stm_targets({"form": "STM-1", "kappa": 0.5, "z0": 0.5}, inp), nan=1.0)
        c_sa = STM.confidence(inp["sent"])
        idx = {s: f.dates[getattr(f, s)[0]:getattr(f, s)[1] + 1] for s in ("val", "test")}
        for which in ("best", "final"):
            pols = [load_policy(OUT_DIR / m / f"seed{s}", which=which) for s in range(3)]
            for s, p in enumerate(pols):
                for split in ("val", "test"):
                    pe = path_ensemble(PolicyEnsemble([p[0]], [p[1]]), win, c.window, ret, *getattr(f, split), c.cost, c.band)
                    single[(which, split, m, s)] = sharpe(pe["r"])
            ens = PolicyEnsemble([p[0] for p in pols], [p[1] for p in pols])
            a0 = max(c.window - 1, f.train[1] - 251)
            P = path_ensemble(ens, win, c.window, ret, a0, f.test[1], c.cost, c.band, conf_divisor=0.5)
            e_rl, c_rl = np.zeros(len(f.dates)), np.zeros(len(f.dates))
            e_rl[a0:], c_rl[a0:] = P["e_rl"], P["c_rl"]
            for split in ("val", "test"):
                pe = path_ensemble(ens, win, c.window, ret, *getattr(f, split), c.cost, c.band)
                series[(f"PPO_{which}", split, m)] = (pd.Series(pe["r"], idx[split]), pe["held"])
            kw = dict(H=60, alpha=5, beta=1.0, cost=c.cost, band=c.band)
            warm = run_aws(e_rl, e_sa, c_rl, c_sa, ret, a0, f.val[0] - 1, **kw)
            v = run_aws(e_rl, e_sa, c_rl, c_sa, ret, *f.val, state=warm["final_state"], **kw)
            t = run_aws(e_rl, e_sa, c_rl, c_sa, ret, *f.test, state=v["final_state"], **kw)
            series[(f"AWS_{which}", "val", m)] = (pd.Series(v["r"], idx["val"]), v["held"])
            series[(f"AWS_{which}", "test", m)] = (pd.Series(t["r"], idx["test"]), t["held"])
        for split in ("val", "test"):
            bh = path_from_targets(np.ones(len(ret)), ret, *getattr(f, split), c.cost, c.band)
            series[("B1", split, m)] = (pd.Series(bh["r"], idx[split]), bh["held"])
    summary = {}
    for var in ("B1", "PPO_best", "PPO_final", "AWS_best", "AWS_final"):
        for split in ("val", "test"):
            cm = combined({m: series[(var, split, m)][0] for m in MARKETS}).to_numpy()
            summary.setdefault(var, {})[split] = {
                "sharpe": sharpe(cm), "mdd": max_drawdown(cm), "ann_return": float(np.prod(1 + cm) ** (252 / len(cm)) - 1),
                "exposure": float(np.mean([np.mean(series[(var, split, m)][1]) for m in MARKETS])),
                "per_market_sharpe": {m: sharpe(series[(var, split, m)][0].to_numpy()) for m in MARKETS}}
    for which in ("best", "final"):
        for split in ("val", "test"):
            summary[f"PPO_{which}"][split]["single_seed_mean_sharpe"] = float(np.mean(
                [single[(which, split, m, s)] for m in MARKETS for s in range(3)]))
    # trials log (every evaluated configuration counts)
    for var, fam in (("PPO_final", "diag_ppo_final"), ("AWS_final", "diag_aws_final"), ("PPO_best", "diag_ppo_best_rerun"),
                     ("AWS_best", "diag_aws_best_rerun")):
        pm = {m: {"val_sharpe": summary[var]["val"]["per_market_sharpe"][m], "val_mdd": 0.0, "val_turnover": 0.0,
                  "devtest_sharpe": summary[var]["test"]["per_market_sharpe"][m]} for m in MARKETS}
        vs = {m: {"r": series[(var, "val", m)][0].tolist(), "dates": [str(x.date()) for x in series[(var, "val", m)][0].index]}
              for m in MARKETS}
        ds = {m: {"r": series[(var, "test", m)][0].tolist(), "dates": [str(x.date()) for x in series[(var, "test", m)][0].index]}
              for m in MARKETS}
        D.log_trial(fam, CFG, {"checkpoint": var}, pm, vs, ds)
    return summary


def main() -> None:
    jobs = [(m, s) for m in MARKETS for s in range(3) if not (OUT_DIR / m / f"seed{s}" / "final_model.zip").exists()]
    with run_record("P4", "ckpt_diagnostic", None, "cpu", {"n_runs": len(jobs)}):
        Parallel(n_jobs=frozen()["parallel"]["rl_jobs"], backend="loky")(delayed(job)(m, s) for m, s in jobs)
        summary = evaluate()
    (RESULTS / "p4_ckpt_diagnostic.json").write_text(json.dumps(R._clean(summary), indent=1), encoding="utf-8")
    R.update([("dev.diagnostic.checkpoint", summary)],
             ["dev.diagnostic.checkpoint: pre-lock diagnostic of checkpoint-selection bias on development data "
              "(scripts/p4_ckpt_diagnostic.py); its variants are logged in dev_trials.jsonl and counted in DSR/PBO."])
    for var, v in summary.items():
        print(f"{var:10s} val Sharpe {v['val']['sharpe']:.3f} | devtest Sharpe {v['test']['sharpe']:.3f} MDD "
              f"{v['test']['mdd']:.3f} ann {v['test']['ann_return']:.3f} exposure {v['test']['exposure']:.2f}")


if __name__ == "__main__":
    main()
