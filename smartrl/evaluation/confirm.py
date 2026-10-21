"""Confirmation phase (P5): policy training arms, fold inputs and method paths (SPEC §5–§10, EXPERIMENTS E5–E7, E12).

Arms (all with the reward and learning rate frozen at G4):
  SMARTRL  full state, AWS-guided curriculum, seeds 0-4      (also CURR of H6)
  B6       price features only, no curriculum, seeds 0-4
  A1       price + MM_NOSENT forecasts, no sentiment, no STM → no curriculum, seeds 0-4
  A2       price + sentiment, no forecasts, curriculum, seeds 0-4
  A5       full state, no curriculum, seeds 0-4               (also PLAIN of H6)
  BC       full state, behaviour-cloning warm start, seeds 0-2
  IMIT     full state, imitation reward (selected ϑ), no curriculum, seeds 0-2
  B7       SAC on the full state and the SMART-RL reward, seeds 0-2
The forward fold trains SMARTRL, B6 and B7 only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROCESSED, frozen
from ..data import folds
from ..env.evaluate import PolicyEnsemble, path_ensemble, path_from_targets
from ..env.portfolio import RF_DAILY, RewardSpec, oracle_labels
from ..env.trading_env import EnvConfig, MarketData, window_matrix
from ..stm import stm as STM

POLICIES = PROCESSED / "policies"
STATES = {"full": (True, "MM"), "price": (False, None), "nosent": (False, "MM_NOSENT"), "nomm": (True, None)}
ARMS = {
    "SMARTRL": {"state": "full", "curriculum": True, "seeds": 5},
    "B6": {"state": "price", "curriculum": False, "seeds": 5},
    "A1": {"state": "nosent", "curriculum": False, "seeds": 5},
    "A2": {"state": "nomm", "curriculum": True, "seeds": 5},
    "A5": {"state": "full", "curriculum": False, "seeds": 5},
    "BC": {"state": "full", "curriculum": False, "bc": True, "seeds": 3},
    "IMIT": {"state": "full", "curriculum": False, "imit": True, "seeds": 3},
    "B7": {"state": "full", "curriculum": False, "algo": "SAC", "seeds": 3},
}
FORWARD_ARMS = ("SMARTRL", "B6", "B7")


def env_cfg(cost_bps: float | None = None) -> EnvConfig:
    c = frozen()
    bps = c["costs"]["bps_per_unit_turnover"] if cost_bps is None else cost_bps
    return EnvConfig(window=c["env"]["window"], episode_len=c["env"]["episode_len"], cost=bps / 1e4,
                     band=c["no_trade_band"], initial_exposure=c["env"]["initial_exposure"])


def selected_reward() -> dict:
    p = frozen()["ppo"]
    name = p["reward"]
    if name == "R2":
        return {"name": "R2", "lambda1": float(p["reward_params"]["R2"]["lambda1"]),
                "lambda2": float(p["reward_params"]["R2"]["lambda2"])}
    return {"name": name}


def stm_exposure(fold: folds.Fold, sent: pd.DataFrame, scaled: pd.DataFrame, ret_tr: np.ndarray,
                 source: str = "SF") -> np.ndarray:
    s = frozen()["stm"] if source == "SF" else {"form": "STM-1", **frozen()["baselines"]["B5_FINBERT"]}
    tr = slice(fold.train[0], fold.train[1] + 1)
    if s["form"] == "STM-2":
        lg = frozen()["stm"]["logistic"]
        e = STM.stm2(sent, scaled, ret_tr, tr, lg["C"], lg["horizon"], tuple(lg["percentiles"]))
    else:
        e = STM.stm1(sent, tr, float(s["kappa"]), float(s["z0"]), source)
    return np.nan_to_num(e, nan=1.0)


def fold_inputs(fold_name: str, market: str) -> dict:
    f = folds.make_fold(fold_name, market)
    out = {"fold": f}
    for key, (sent, mm) in STATES.items():
        A = folds.fold_arrays(f, sentiment=sent, mm=mm)
        out[key] = A["X"]
        if key == "full":
            out.update(ret_tr=np.nan_to_num(A["ret_tr"]), close=A["close"], sent=A["sent"],
                       scaled=pd.DataFrame(A["X"], columns=A["cols"], index=f.dates))
    out["e_sa"] = stm_exposure(f, out["sent"], out["scaled"], out["ret_tr"])
    out["e_b5"] = stm_exposure(f, out["sent"], out["scaled"], out["ret_tr"], source="B")
    out["c_sa"] = STM.confidence(out["sent"])
    return out


def policy_dir(fold: str, arm: str, market: str, seed: int) -> Path:
    return POLICIES / fold / arm / market / f"seed{seed}"


def train_job(fold_name: str, arm: str, market: str, seed: int) -> dict:
    """One policy (runs in a joblib worker with one torch thread)."""
    import torch
    torch.set_num_threads(1)
    from ..agents.ppo import TrainSpec, save_policy, train_ppo, train_sac
    cfg = frozen()
    a = ARMS[arm]
    d = policy_dir(fold_name, arm, market, seed)
    if (d / "info.json").exists():              # info.json is written last: a policy cut by a crash is retrained
        return {"skipped": str(d)}
    inp = fold_inputs(fold_name, market)
    f, c = inp["fold"], env_cfg()
    reward = selected_reward()
    r_or = None
    if a.get("imit"):
        theta = float(cfg["ppo"]["reward_params"]["IMIT"]["theta_bps"]) / 1e4
        lo, hi = f.train
        nxt = inp["ret_tr"][lo + 1:hi + 1]
        y = oracle_labels(nxt, RF_DAILY, theta)
        r_or = np.zeros(len(inp["ret_tr"]))
        r_or[lo + 1:hi + 1] = y * nxt + (1 - y) * RF_DAILY
        reward = {"name": "IMIT"}
    data = MarketData(inp[a["state"]], inp["ret_tr"], e_sa=inp["e_sa"], r_oracle=r_or)
    spec = TrainSpec(data, f.train, f.val, RewardSpec(**reward), float(cfg["ppo"]["learning_rate"]), seed,
                     curriculum=a["curriculum"], bc_warm_start=a.get("bc", False),
                     total_timesteps=cfg["ppo"]["total_timesteps"], eval_every=cfg["ppo"]["eval_every"], env_cfg=c)
    t0 = time.time()
    out = train_sac(spec, cfg, cfg["parallel"].get("sac_device", "cuda")) if a.get("algo") == "SAC" else train_ppo(spec, cfg)
    out["info"]["seconds"] = time.time() - t0
    save_policy(out, d)
    return {"dir": str(d), "seconds": out["info"]["seconds"], "best_val_sharpe": out["info"]["best_val_sharpe"]}


def jobs(arms: list[str], fold_list: list[str] | None = None) -> list[tuple[str, str, str, int]]:
    out = []
    for fold in fold_list or folds.fold_names()[1:]:
        for arm in arms:
            if fold == "forward" and arm not in FORWARD_ARMS:
                continue
            for m in frozen()["markets"]:
                for s in range(ARMS[arm]["seeds"]):
                    out.append((fold, arm, m, s))
    return out


# ---------------------------------------------------------------- evaluation of every method on one fold
def ensemble(fold_name: str, arm: str, market: str) -> PolicyEnsemble:
    from ..agents.ppo import load_policy
    algo = "SAC" if ARMS[arm].get("algo") == "SAC" else "PPO"
    ps = [load_policy(policy_dir(fold_name, arm, market, s), algo) for s in range(ARMS[arm]["seeds"])]
    return PolicyEnsemble([p[0] for p in ps], [p[1] for p in ps])


def ensemble_paths(inp: dict, state: str, ens: PolicyEnsemble, c: EnvConfig, delay: int = 0) -> dict:
    """Own-book paths of a policy ensemble on the validation and the test split (fresh book at each split start)."""
    f = inp["fold"]
    win = window_matrix(inp[state], c.window)
    cd = frozen()["ppo"]["confidence_divisor"]
    return {s: path_ensemble(ens, win, c.window, inp["ret_tr"], *getattr(f, s), c.cost, c.band, c.initial_exposure,
                             delay, conf_divisor=cd) for s in ("val", "test")}


def full_series(n: int, a: int, values: np.ndarray) -> np.ndarray:
    out = np.zeros(n)
    out[a:a + len(values)] = values
    return out


def _aws(inp: dict, e_rl: np.ndarray, c_rl: np.ndarray, beta: float | None, c: EnvConfig, delay: int) -> tuple[dict, dict]:
    """AWS warm-started through the validation split, then run over the test split (SPEC §8)."""
    from ..aws.aws import run_aws
    f, a = inp["fold"], frozen()["aws"]
    kw = dict(H=float(a["H"]), alpha=float(a["alpha"]), beta=float(a["beta"] if beta is None else beta), cost=c.cost,
              band=c.band, e_init=c.initial_exposure, delay=delay)
    args = (e_rl, inp["e_sa"], c_rl, inp["c_sa"], inp["ret_tr"])
    v = run_aws(*args, *f.val, **kw)
    return run_aws(*args, *f.test, state=v["final_state"], **kw), v


def _decisions(e_target: np.ndarray, held: np.ndarray, band: float) -> np.ndarray:
    d = e_target - held
    return np.where(d > band, 1, np.where(d < -band, -1, 0))


def evaluate_fold(fold_name: str, market: str, cost_bps: float | None = None, delay: int = 0, inp: dict | None = None) -> dict:
    """Test-split paths of every method (and the validation paths needed by AWS) for one fold and market."""
    from ..baselines import rules
    inp = inp or fold_inputs(fold_name, market)
    f, c = inp["fold"], env_cfg(cost_bps)
    n = len(f.dates)
    a, b = f.test
    out: dict = {"dates": f.dates[a:b + 1]}
    path = lambda tgt: path_from_targets(tgt, inp["ret_tr"], a, b, c.cost, c.band, c.initial_exposure, delay)
    arms = [x for x in ARMS if not (fold_name == "forward" and x not in FORWARD_ARMS) and x not in ("BC", "IMIT")]
    ens = {}
    for arm in arms:
        state = ARMS[arm]["state"]
        P = ensemble_paths(inp, state, ensemble(fold_name, arm, market), c, delay)
        ens[arm] = P
    # SMART-RL: PPO ensemble (own book) and STM blended by AWS
    def expert_arrays(P):
        e_rl = full_series(n, f.val[0], P["val"]["e_rl"]) + full_series(n, a, P["test"]["e_rl"])
        c_rl = full_series(n, f.val[0], P["val"]["c_rl"]) + full_series(n, a, P["test"]["c_rl"])
        return e_rl, c_rl
    e_rl, c_rl = expert_arrays(ens["SMARTRL"])
    sm, sm_val = _aws(inp, e_rl, c_rl, None, c, delay)
    out["SMARTRL"] = {"r": sm["r"], "held": sm["held"], "decision": sm["decision"], "aws": sm, "aws_val": sm_val}
    out["A3_NOAWS"] = {k: ens["SMARTRL"]["test"][k] for k in ("r", "held")}
    out["A3_NOAWS"]["decision"] = _decisions(ens["SMARTRL"]["test"]["e_rl"], ens["SMARTRL"]["test"]["held"], c.band)
    out["A3_NOAWS"]["e_rl"] = ens["SMARTRL"]["test"]["e_rl"]
    nc, _ = _aws(inp, e_rl, c_rl, 1.0, c, delay)
    out["A4_NOCONF"] = {"r": nc["r"], "held": nc["held"]}
    out["B8_STATIC"] = path(rules.b8_static(e_rl, inp["e_sa"], frozen()["baselines"]["B8_STATIC"]["w_rl"]))
    if fold_name != "forward":
        p1 = ens["A1"]["test"]
        out["A1_NOSENT"] = {"r": p1["r"], "held": p1["held"], "decision": _decisions(p1["e_rl"], p1["held"], c.band)}
        for arm, key in (("A2", "A2_NOMM"), ("A5", "A5_NOCURR")):
            er, cr = expert_arrays(ens[arm])
            x, _ = _aws(inp, er, cr, None, c, delay)
            out[key] = {"r": x["r"], "held": x["held"]}
    out["B6_PPOPRICE"] = {k: ens["B6"]["test"][k] for k in ("r", "held")}
    out["B7_SAC"] = {k: ens["B7"]["test"][k] for k in ("r", "held")}
    close = inp["close"]
    out["B1_BH"] = path(rules.b1_buy_and_hold(n))
    out["B2_SMA"] = path(np.nan_to_num(rules.b2_sma(close), nan=1.0))
    out["B3_TSMOM"] = path(np.nan_to_num(rules.b3_tsmom(close), nan=1.0))
    lp = folds.forecast_frame("LSTM_PRICE")
    lp = lp[lp["market"] == market].drop_duplicates("date", keep="last").set_index("date").reindex(f.dates)
    out["B4_LSTM"] = path(rules.b4_lstm(lp["q50_h5"].to_numpy()))
    out["B5_FINBERT"] = path(inp["e_b5"])
    sn = folds.forecast_frame("SENTIN")
    sn = sn[sn["market"] == market].drop_duplicates("date", keep="last").set_index("date").reindex(f.dates)
    out["BASE"] = path(rules.base_sentin(sn["p_up"].to_numpy(), frozen()["base"]["trading_rule"]["p_up"]))
    if fold_name != "forward":
        # H6: each training arm alone (seed ensemble 0-2 and every single seed), PPO without STM/AWS
        from ..agents.ppo import load_policy
        for arm, key in (("SMARTRL", "CURR"), ("A5", "PLAIN"), ("BC", "BC"), ("IMIT", "IMIT")):
            pols = [load_policy(policy_dir(fold_name, arm, market, s)) for s in range(3)]
            win = window_matrix(inp["full"], c.window)
            for name, ps in [(f"H6_{key}", pols)] + [(f"H6_{key}_s{s}", [pols[s]]) for s in range(3)]:
                p = path_ensemble(PolicyEnsemble([q[0] for q in ps], [q[1] for q in ps]), win, c.window, inp["ret_tr"], a, b,
                                  c.cost, c.band, c.initial_exposure, delay)
                out[name] = {"r": p["r"], "held": p["held"]}
    out["_inp"] = {"ret_tr": inp["ret_tr"][a:b + 1], "close": inp["close"], "sent": inp["sent"].iloc[a:b + 1],
                   "fold": f}
    return out
