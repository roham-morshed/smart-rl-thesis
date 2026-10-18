"""P4 / E4: development phase 2010-2019 (SPEC §1, EXPERIMENTS E4). Order: STM grid → B5 grid → PPO grid (curriculum with
the selected STM) → AWS grid (selected PPO and STM) → IMIT arm. Selection: highest mean development-validation
Sharpe across the four markets (and seeds); ties → lower turnover. The development test split is only reported and
used for PBO. Every configuration is appended to results/dev_trials.jsonl (never overwritten).

    python scripts/p4_dev.py --stage stm|b5|ppo|aws|imit|select
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import FROZEN, MARKETS, PROCESSED, RESULTS, config_hash, frozen  # noqa: E402
from smartrl.data import folds  # noqa: E402
from smartrl.env.portfolio import RF_DAILY, RewardSpec, oracle_labels  # noqa: E402
from smartrl.env.trading_env import EnvConfig, MarketData, window_matrix  # noqa: E402
from smartrl.env.evaluate import PolicyEnsemble, path_ensemble, path_from_targets  # noqa: E402
from smartrl.evaluation.metrics import combined, max_drawdown, sharpe, trading_metrics  # noqa: E402
from smartrl.runs import run_record  # noqa: E402
from smartrl.stm import stm as STM  # noqa: E402
from smartrl.aws.aws import run_aws  # noqa: E402

TRIALS = RESULTS / "dev_trials.jsonl"
POLICIES = PROCESSED / "policies" / "dev"
STATE = RESULTS / "p4_state.json"


def ppo_cfg_id(reward: dict, lr: float) -> str:
    lr_s = {3e-4: "lr3e-4", 1e-4: "lr1e-4"}[lr]
    if reward["name"] == "R2":
        return f"R2_l{int(reward['lambda1'])}_d{float(reward['lambda2']):.1f}_{lr_s}"
    return f"{reward['name']}_{lr_s}"


def stm_cfg_id(c: dict) -> str:
    return "STM-2" if c["form"] == "STM-2" else f"STM-1_k{c['kappa']}_z{c['z0']}"


def env_cfg() -> EnvConfig:
    c = frozen()
    return EnvConfig(window=c["env"]["window"], episode_len=c["env"]["episode_len"],
                     cost=c["costs"]["bps_per_unit_turnover"] / 1e4, band=c["no_trade_band"],
                     initial_exposure=c["env"]["initial_exposure"])


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1), encoding="utf-8")


def log_trial(family: str, cfg_id: str, params: dict, per_market: dict, val_series: dict, dt_series: dict) -> dict:
    """Append one configuration: per-market results and the COMBINED daily returns on dev-val and dev-test."""
    vs = combined({m: pd.Series(v["r"], index=pd.DatetimeIndex(v["dates"])) for m, v in val_series.items()})
    ds = combined({m: pd.Series(v["r"], index=pd.DatetimeIndex(v["dates"])) for m, v in dt_series.items()})
    rec = {"family": family, "config": cfg_id, "params": params, "utc": pd.Timestamp.utcnow().isoformat(),
           "config_hash": config_hash(),
           "val_sharpe": float(np.mean([v["val_sharpe"] for v in per_market.values()])),
           "val_turnover": float(np.mean([v["val_turnover"] for v in per_market.values()])),
           "val_mdd": float(np.mean([v["val_mdd"] for v in per_market.values()])),
           "devtest_sharpe": float(np.mean([v["devtest_sharpe"] for v in per_market.values()])),
           "per_market": per_market,
           "combined_val": {"dates": [str(d.date()) for d in vs.index], "r": vs.tolist()},
           "combined_devtest": {"dates": [str(d.date()) for d in ds.index], "r": ds.tolist()}}
    with TRIALS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(R._clean(rec)) + "\n")
    return rec


def select(recs: list[dict]) -> dict:
    return sorted(recs, key=lambda r: (-r["val_sharpe"], r["val_turnover"]))[0]


# ---------------------------------------------------------------- data per market
def market_inputs(m: str) -> dict:
    f = folds.make_fold("dev", m)
    full = folds.fold_arrays(f, sentiment=True, mm="MM")
    sent = full["sent"].copy()
    scaled = pd.DataFrame(full["X"], columns=full["cols"], index=f.dates)
    return {"fold": f, "X": full["X"], "ret_tr": full["ret_tr"], "close": full["close"], "sent": sent, "scaled": scaled}


def rule_eval(targets: np.ndarray, inp: dict, delay: int = 0) -> tuple[dict, dict, dict]:
    f, c = inp["fold"], env_cfg()
    out = {}
    for split in ("val", "test"):
        a, b = getattr(f, split)
        out[split] = path_from_targets(targets, inp["ret_tr"], a, b, c.cost, c.band, c.initial_exposure, delay)
    pm = {"val_sharpe": sharpe(out["val"]["r"]), "val_mdd": max_drawdown(out["val"]["r"]),
          "val_turnover": trading_metrics(out["val"]["r"], out["val"]["held"])["turnover"],
          "devtest_sharpe": sharpe(out["test"]["r"])}
    d = lambda s: {"r": out[s]["r"].tolist(), "dates": [str(x.date()) for x in f.dates[getattr(f, s)[0]:getattr(f, s)[1] + 1]]}
    return pm, d("val"), d("test")


def stm_targets(c: dict, inp: dict, source: str = "SF") -> np.ndarray:
    tr = slice(inp["fold"].train[0], inp["fold"].train[1] + 1)
    if c["form"] == "STM-2":
        return STM.stm2(inp["sent"], inp["scaled"], inp["ret_tr"], tr, frozen()["stm"]["logistic"]["C"],
                        frozen()["stm"]["logistic"]["horizon"], tuple(frozen()["stm"]["logistic"]["percentiles"]))
    return STM.stm1(inp["sent"], tr, c["kappa"], c["z0"], source)


def stage_rules(family: str, inputs: dict) -> dict:
    grid = frozen()["dev_grid"]["stm"]
    if family == "b5":
        grid = [g for g in grid if g["form"] == "STM-1"]
    recs = []
    for c in grid:
        pm, vs, ds = {}, {}, {}
        for m in MARKETS:
            pm[m], vs[m], ds[m] = rule_eval(stm_targets(c, inputs[m], "SF" if family == "stm" else "B"), inputs[m])
        cid = stm_cfg_id(c) if family == "stm" else f"k{c['kappa']}_z{c['z0']}"
        recs.append(log_trial(family, cid, c, pm, vs, ds))
        print(family, cid, round(recs[-1]["val_sharpe"], 4), flush=True)
    best = select(recs)
    return {"config": best["config"], "params": best["params"], "val_sharpe": best["val_sharpe"]}


# ---------------------------------------------------------------- PPO runs (parallel workers)
def ppo_job(m: str, reward: dict, lr: float, seed: int, curriculum: bool, e_sa: np.ndarray | None, theta: float | None,
            out_dir: str) -> dict:
    import torch
    torch.set_num_threads(1)
    from smartrl.agents.ppo import TrainSpec, save_policy, train_ppo
    cfg = frozen()
    inp = market_inputs(m)
    f, c = inp["fold"], env_cfg()
    r_or = None
    if reward["name"] == "IMIT":
        a, b = f.train
        ret_next = np.nan_to_num(inp["ret_tr"][a + 1:b + 1])
        y = oracle_labels(ret_next, RF_DAILY, theta / 1e4)
        r_or = np.zeros(len(inp["ret_tr"]))
        r_or[a + 1:b + 1] = y * ret_next + (1 - y) * RF_DAILY
    data = MarketData(inp["X"], np.nan_to_num(inp["ret_tr"]), e_sa=e_sa, r_oracle=r_or)
    spec = TrainSpec(data, f.train, f.val, RewardSpec(**{k: v for k, v in reward.items() if k != "theta_bps"}), lr, seed,
                     curriculum=curriculum, total_timesteps=cfg["ppo"]["total_timesteps"], eval_every=cfg["ppo"]["eval_every"],
                     env_cfg=c)
    t0 = time.time()
    out = train_ppo(spec, cfg)
    save_policy(out, Path(out_dir))
    from smartrl.agents.ppo import load_policy
    model, norm = load_policy(Path(out_dir))
    win = window_matrix(inp["X"], c.window)
    res = {}
    for split in ("val", "test"):
        a, b = getattr(f, split)
        p = path_ensemble(PolicyEnsemble([model], [norm]), win, c.window, data.ret_tr, a, b, c.cost, c.band,
                          c.initial_exposure)
        res[split] = {"r": p["r"].tolist(), "held": p["held"].tolist(),
                      "dates": [str(x.date()) for x in f.dates[a:b + 1]]}
    return {"market": m, "seed": seed, "seconds": time.time() - t0, "val": res["val"], "test": res["test"],
            "curve": out["info"]["curve"]}


def run_ppo_grid(family: str, configs: list[tuple[str, dict, float, float | None]], seeds: list[int], curriculum: bool,
                 e_sa: dict | None, n_jobs: int) -> list[dict]:
    jobs = []
    for cid, reward, lr, theta in configs:
        for m in MARKETS:
            for s in seeds:
                d = POLICIES / family / cid / m / f"seed{s}"
                jobs.append((cid, reward, lr, theta, m, s, str(d)))
    t0 = time.time()
    with run_record("P4", f"{family}_grid", None, "cpu", {"n_runs": len(jobs), "n_jobs": n_jobs}):
        outs = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(ppo_job)(m, reward, lr, s, curriculum, (e_sa or {}).get(m), theta, d)
            for cid, reward, lr, theta, m, s, d in jobs)
    print(f"{family}: {len(jobs)} runs in {(time.time() - t0) / 60:.1f} min", flush=True)
    recs = []
    for cid, reward, lr, theta in configs:
        pm, vs, ds = {}, {}, {}
        for m in MARKETS:
            runs = [o for (c_, *_), o in zip(jobs, outs) if c_ == cid and o["market"] == m]
            val_r = [np.array(o["val"]["r"]) for o in runs]
            pm[m] = {"val_sharpe": float(np.mean([sharpe(r) for r in val_r])),
                     "val_mdd": float(np.mean([max_drawdown(r) for r in val_r])),
                     "val_turnover": float(np.mean([trading_metrics(np.array(o["val"]["r"]), np.array(o["val"]["held"]))["turnover"]
                                                    for o in runs])),
                     "devtest_sharpe": float(np.mean([sharpe(np.array(o["test"]["r"])) for o in runs])),
                     "seeds": {o["seed"]: {"val_sharpe": sharpe(np.array(o["val"]["r"])), "curve": o["curve"]} for o in runs}}
            vs[m] = {"r": np.mean(val_r, axis=0).tolist(), "dates": runs[0]["val"]["dates"]}
            ds[m] = {"r": np.mean([np.array(o["test"]["r"]) for o in runs], axis=0).tolist(), "dates": runs[0]["test"]["dates"]}
        params = {"reward": reward, "lr": lr, "theta_bps": theta}
        recs.append(log_trial(family, cid, params, pm, vs, ds))
        print(family, cid, round(recs[-1]["val_sharpe"], 4), round(recs[-1]["devtest_sharpe"], 4), flush=True)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["stm", "b5", "ppo", "aws", "imit", "select"])
    args = ap.parse_args()
    cfg = frozen()
    st = load_state()
    n_jobs = cfg["parallel"]["rl_jobs"]
    if args.stage in ("stm", "b5"):
        inputs = {m: market_inputs(m) for m in MARKETS}
        st[args.stage] = stage_rules(args.stage, inputs)
    elif args.stage == "ppo":
        inputs = {m: market_inputs(m) for m in MARKETS}
        sel = st["stm"]["params"]
        e_sa = {m: np.nan_to_num(stm_targets(sel, inputs[m]), nan=1.0) for m in MARKETS}
        configs = [(ppo_cfg_id(r, lr), r, lr, None) for r in cfg["dev_grid"]["reward"] for lr in cfg["dev_grid"]["lr"]]
        recs = run_ppo_grid("ppo", configs, cfg["dev_grid"]["seeds"], True, e_sa, n_jobs)
        best = select(recs)
        st["ppo"] = {"config": best["config"], "params": best["params"], "val_sharpe": best["val_sharpe"]}
    elif args.stage == "imit":
        lr = st["ppo"]["params"]["lr"]
        configs = [(str(th), {"name": "IMIT", "scale": 100.0}, lr, float(th)) for th in cfg["dev_grid"]["imit"]["theta_bps"]]
        recs = run_ppo_grid("imit", configs, cfg["dev_grid"]["seeds"], False, None, n_jobs)
        best = select(recs)
        st["imit"] = {"config": best["config"], "params": best["params"], "val_sharpe": best["val_sharpe"]}
    elif args.stage == "aws":
        st["aws"] = stage_aws(st, cfg)
    save_state(st)
    if args.stage == "select":
        finalize(st, cfg)


def stage_aws(st: dict, cfg: dict) -> dict:
    """AWS grid with the selected PPO configuration (seed ensemble 0-2) and the selected STM. Warm start: the EWMA
    state runs through the 252 training days before dev-val, and through dev-val before dev-test."""
    from smartrl.agents.ppo import load_policy
    c = env_cfg()
    sel_ppo, sel_stm = st["ppo"]["config"], st["stm"]["params"]
    prep = {}
    for m in MARKETS:
        inp = market_inputs(m)
        f = inp["fold"]
        pols = [load_policy(POLICIES / "ppo" / sel_ppo / m / f"seed{s}") for s in cfg["dev_grid"]["seeds"]]
        ens = PolicyEnsemble([p[0] for p in pols], [p[1] for p in pols])
        win = window_matrix(inp["X"], c.window)
        a0 = max(c.window - 1, f.train[1] - 251)
        p = path_ensemble(ens, win, c.window, np.nan_to_num(inp["ret_tr"]), a0, f.test[1], c.cost, c.band, c.initial_exposure,
                          conf_divisor=cfg["ppo"]["confidence_divisor"])
        e_rl, c_rl = np.zeros(len(f.dates)), np.zeros(len(f.dates))
        e_rl[a0:], c_rl[a0:] = p["e_rl"], p["c_rl"]
        e_sa = np.nan_to_num(stm_targets(sel_stm, inp), nan=1.0)
        prep[m] = (inp, e_rl, c_rl, e_sa, STM.confidence(inp["sent"]), a0)
    recs = []
    for H in cfg["dev_grid"]["aws"]["H"]:
        for al in cfg["dev_grid"]["aws"]["alpha"]:
            for be in cfg["dev_grid"]["aws"]["beta"]:
                pm, vs, ds = {}, {}, {}
                for m, (inp, e_rl, c_rl, e_sa, c_sa, a0) in prep.items():
                    f, ret = inp["fold"], np.nan_to_num(inp["ret_tr"])
                    args = (e_rl, e_sa, c_rl, c_sa, ret)
                    kw = dict(H=H, alpha=al, beta=be, cost=c.cost, band=c.band)
                    warm = run_aws(*args, a0, f.val[0] - 1, **kw)
                    v = run_aws(*args, *f.val, state=warm["final_state"], **kw)
                    t = run_aws(*args, *f.test, state=v["final_state"], **kw)
                    pm[m] = {"val_sharpe": sharpe(v["r"]), "val_mdd": max_drawdown(v["r"]),
                             "val_turnover": trading_metrics(v["r"], v["held"])["turnover"], "devtest_sharpe": sharpe(t["r"])}
                    vs[m] = {"r": v["r"].tolist(), "dates": [str(x.date()) for x in f.dates[f.val[0]:f.val[1] + 1]]}
                    ds[m] = {"r": t["r"].tolist(), "dates": [str(x.date()) for x in f.dates[f.test[0]:f.test[1] + 1]]}
                cid = f"H{H}_a{al}_b{be}"
                recs.append(log_trial("aws", cid, {"H": H, "alpha": al, "beta": be}, pm, vs, ds))
                print("aws", cid, round(recs[-1]["val_sharpe"], 4), flush=True)
    best = select(recs)
    return {"config": best["config"], "params": best["params"], "val_sharpe": best["val_sharpe"]}


def finalize(st: dict, cfg: dict) -> None:
    """dev.* keys, the selected values written into configs/frozen.yaml (SET_IN_P4), new config hash."""
    recs = [json.loads(l) for l in TRIALS.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [("dev.n_trials", len(recs))]
    for r in recs:
        fam = r["family"]
        if fam == "ppo":
            for k in ("val_sharpe", "val_mdd", "devtest_sharpe"):
                items.append((["dev", "ppo", r["config"], k], r[k]))
        elif fam == "imit":
            for k in ("val_sharpe", "val_mdd", "devtest_sharpe"):
                items.append((["dev", "imit", r["config"], k], r[k]))
        else:
            items.append((["dev", fam, r["config"], "val_sharpe"], r["val_sharpe"]))
    ppo = st["ppo"]["params"]
    stm = st["stm"]["params"]
    aws = st["aws"]["params"]
    b5 = st["b5"]["params"]
    sel = {"config": st["ppo"]["config"], "reward": ppo["reward"]["name"], "lr": ppo["lr"], "stm_form": stm["form"],
           "kappa": stm.get("kappa"), "z0": stm.get("z0"), "aws_H": aws["H"], "aws_alpha": aws["alpha"],
           "aws_beta": aws["beta"], "b5_kappa": b5["kappa"], "b5_z0": b5["z0"],
           "imit_theta_bps": st["imit"]["params"]["theta_bps"]}
    items += [(["dev", "selected", k], v) for k, v in sel.items()]
    t = FROZEN.read_text(encoding="utf-8")
    rw = ppo["reward"]
    rep = {
        "  learning_rate: SET_IN_P4": f"  learning_rate: {ppo['lr']}",
        "  reward: SET_IN_P4": f"  reward: {rw['name']}",
        "    R2: {lambda1: SET_IN_P4, lambda2: SET_IN_P4, scale: 100}":
            "    R2: {lambda1: %s, lambda2: %s, scale: 100}" % (rw.get("lambda1", "null"), rw.get("lambda2", "null")),
        "    IMIT: {theta_bps: SET_IN_P4, scale: 100}": "    IMIT: {theta_bps: %s, scale: 100}" % sel["imit_theta_bps"],
        "  form: SET_IN_P4": f"  form: {stm['form']}",
        "  kappa: SET_IN_P4": f"  kappa: {stm.get('kappa', 'null')}",
        "  z0: SET_IN_P4": f"  z0: {stm.get('z0', 'null')}",
        "  H: SET_IN_P4": f"  H: {aws['H']}",
        "  alpha: SET_IN_P4": f"  alpha: {aws['alpha']}",
        "  beta: SET_IN_P4": f"  beta: {aws['beta']}",
        "  B5_FINBERT: {kappa: SET_IN_P4, z0: SET_IN_P4}": "  B5_FINBERT: {kappa: %s, z0: %s}" % (b5["kappa"], b5["z0"]),
    }
    for a, b in rep.items():
        if a in t:
            t = t.replace(a, b, 1)
    FROZEN.write_text(t, encoding="utf-8")
    items.append(("meta.config_hash", config_hash()))
    R.update(items)
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
