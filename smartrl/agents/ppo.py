"""PPO agent (SPEC §6), AWS-guided curriculum (§6.1), behaviour-cloning warm start (§6.2) and SAC (B7).

Training: 8 random-start environments (DummyVecEnv) on the fold's training rows, VecNormalize on observations.
Every `eval_every` steps the deterministic policy runs over the validation split on its own book (e = 1 before
the first decision); the checkpoint with the best validation Sharpe is kept (ties → later) and the whole curve is
logged (H6). VecNormalize statistics are frozen at evaluation.
"""
from __future__ import annotations

import copy
import io
import json
import math
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..env.evaluate import PolicyEnsemble, path_ensemble
from ..env.portfolio import RF_DAILY, RewardSpec
from ..env.trading_env import EnvConfig, MarketData, TradingEnv, window_matrix
from ..evaluation.metrics import max_drawdown, sharpe


@dataclass
class TrainSpec:
    data: MarketData
    train: tuple[int, int]          # rows of the training split (decisions need t+1 inside it)
    val: tuple[int, int]
    reward: RewardSpec
    lr: float
    seed: int
    curriculum: bool = False
    bc_warm_start: bool = False
    total_timesteps: int = 300_000
    eval_every: int = 10_000
    env_cfg: EnvConfig = field(default_factory=EnvConfig)


def _ann_sharpe(r: np.ndarray) -> float:
    return sharpe(r) if len(r) > 2 else 0.0


class CurriculumCallback(BaseCallback):
    """ω = w_SA · max(0, 1 − steps/K0); w_SA = softmax(α_c·[tanh(Sh_RL/2), tanh(Sh_SA/2)])_SA over the virtual returns
    of the last 20 completed training episodes (0.5 before 20 episodes); updated after every rollout."""

    def __init__(self, k0: float, window: int, alpha: float, w_before: float):
        super().__init__()
        self.k0, self.alpha, self.w_before = k0, alpha, w_before
        self.rl, self.sa = deque(maxlen=window), deque(maxlen=window)
        self.trace: list[dict] = []

    def _set(self) -> None:
        if len(self.rl) < self.rl.maxlen:
            w_sa = self.w_before
            sh = (float("nan"), float("nan"))
        else:
            sh = (_ann_sharpe(np.concatenate(self.rl)), _ann_sharpe(np.concatenate(self.sa)))
            z = self.alpha * np.tanh(np.array(sh) / 2)
            w = np.exp(z - z.max())
            w_sa = float(w[1] / w.sum())
        omega = w_sa * max(0.0, 1.0 - self.num_timesteps / self.k0)
        self.training_env.env_method("set_omega", omega)
        self.trace.append({"steps": int(self.num_timesteps), "omega": omega, "w_sa": w_sa, "sh_rl": sh[0], "sh_sa": sh[1]})

    def _on_training_start(self) -> None:
        self._set()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "virtual_rl" in info:
                self.rl.append(info["virtual_rl"])
                self.sa.append(info["virtual_sa"])
        return True

    def _on_rollout_end(self) -> None:
        self._set()


class ValidationCallback(BaseCallback):
    def __init__(self, spec: TrainSpec, venv: VecNormalize):
        super().__init__()
        self.spec, self.venv = spec, venv
        self.win = window_matrix(spec.data.feats, spec.env_cfg.window)
        self.curve: list[dict] = []
        self.best = (-math.inf, None, None)
        self.next_eval = spec.eval_every

    def evaluate(self) -> dict:
        ens = PolicyEnsemble([self.model], [FrozenNorm(copy.deepcopy(self.venv.obs_rms))])
        a, b = self.spec.val
        c = self.spec.env_cfg
        p = path_ensemble(ens, self.win, c.window, self.spec.data.ret_tr, a, b, c.cost, c.band, c.initial_exposure)
        return {"steps": int(self.num_timesteps), "val_sharpe": sharpe(p["r"]), "val_mdd": max_drawdown(p["r"]),
                "val_return": float(np.prod(1 + p["r"]) - 1)}

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.spec.eval_every
            res = self.evaluate()
            self.curve.append(res)
            s = res["val_sharpe"] if np.isfinite(res["val_sharpe"]) else -math.inf
            if s >= self.best[0]:
                buf = io.BytesIO()
                self.model.save(buf)
                self.best = (s, buf.getvalue(), pickle.dumps(self.venv.obs_rms))
        return True


def _venv(spec: TrainSpec) -> VecNormalize:
    a, b = spec.train
    fns = [lambda i=i: TradingEnv(spec.data, max(a, spec.env_cfg.window - 1), b, spec.reward, spec.env_cfg,
                                  random_start=True, curriculum=spec.curriculum, seed=spec.seed * 1000 + i)
           for i in range(8)]
    return VecNormalize(DummyVecEnv(fns), norm_obs=True, norm_reward=False)


def _bc_warm_start(model: PPO, venv: VecNormalize, spec: TrainSpec, steps: int, lr: float) -> dict:
    """Regress the actor mean onto the STM exposure (a = 2e − 1) over the training rows, on observations of the
    STM's own path; VecNormalize statistics are initialised from those observations."""
    a, b = spec.train
    c = spec.env_cfg
    env = TradingEnv(spec.data, max(a, c.window - 1), b, spec.reward, c, random_start=False)
    obs, _ = env.reset()
    O, Y = [], []
    done = False
    while not done:
        e_sa = float(spec.data.e_sa[env.t])
        O.append(obs)
        Y.append(2 * e_sa - 1)
        obs, _, done, _, _ = env.step(np.array([2 * e_sa - 1]))
    O, Y = np.asarray(O, np.float32), np.asarray(Y, np.float32)
    venv.obs_rms.update(O)
    On = torch.tensor(venv.normalize_obs(O), dtype=torch.float32)
    Yt = torch.tensor(Y)[:, None]
    pol = model.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(spec.seed)
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(On), (256,), generator=g)
        lat = pol.mlp_extractor.forward_actor(pol.extract_features(On[idx], pol.pi_features_extractor))
        loss = torch.nn.functional.mse_loss(pol.action_net(lat), Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return {"bc_rows": int(len(O)), "bc_final_mse": float(np.mean(losses[-100:]))}


def train_ppo(spec: TrainSpec, cfg: dict) -> dict:
    """Returns the best model (bytes), its VecNormalize statistics, the validation curve and diagnostics."""
    torch.set_num_threads(1)
    pc = cfg["ppo"]
    venv = _venv(spec)
    model = PPO("MlpPolicy", venv, n_steps=pc["n_steps"], batch_size=pc["batch_size"], n_epochs=pc["n_epochs"],
                gamma=pc["gamma"], gae_lambda=pc["gae_lambda"], clip_range=pc["clip_range"], ent_coef=pc["ent_coef"],
                learning_rate=lambda progress: spec.lr * progress, seed=spec.seed, device="cpu", verbose=0,
                policy_kwargs={"net_arch": pc["net_arch"], "activation_fn": torch.nn.Tanh})
    info: dict = {}
    if spec.bc_warm_start:
        info.update(_bc_warm_start(model, venv, spec, pc["bc_warm_start"]["steps"], pc["bc_warm_start"]["lr"]))
    val_cb = ValidationCallback(spec, venv)
    cbs = [val_cb]
    cur = None
    if spec.curriculum:
        cc = pc["curriculum"]
        cur = CurriculumCallback(cc["k0_fraction"] * spec.total_timesteps, cc["window_episodes"], cc["alpha"],
                                 cc["w_sa_before_window"])
        cbs.insert(0, cur)
    model.learn(total_timesteps=spec.total_timesteps, callback=cbs)
    if cur is not None:
        venv.env_method("set_omega", 0.0)
    info.update({"curve": val_cb.curve, "best_val_sharpe": val_cb.best[0],
                 "curriculum": cur.trace if cur is not None else None})
    return {"model_bytes": val_cb.best[1], "obs_rms": val_cb.best[2], "info": info}


def train_sac(spec: TrainSpec, cfg: dict, device: str = "cuda") -> dict:
    """B7: SB3 SAC on the SMART-RL state and reward, one environment, same validation checkpointing as PPO."""
    torch.set_num_threads(1)
    sc = cfg["sac"]
    a, b = spec.train
    venv = VecNormalize(DummyVecEnv([lambda: TradingEnv(spec.data, max(a, spec.env_cfg.window - 1), b, spec.reward,
                                                        spec.env_cfg, random_start=True, seed=spec.seed * 1000)]),
                        norm_obs=True, norm_reward=False)
    model = SAC("MlpPolicy", venv, learning_rate=sc["learning_rate"], buffer_size=sc["buffer_size"],
                batch_size=sc["batch_size"], tau=sc["tau"], gamma=sc["gamma"], ent_coef=sc["ent_coef"],
                learning_starts=sc["learning_starts"], seed=spec.seed, device=device, verbose=0)
    val_cb = ValidationCallback(spec, venv)
    model.learn(total_timesteps=sc["total_timesteps"], callback=[val_cb])
    return {"model_bytes": val_cb.best[1], "obs_rms": val_cb.best[2],
            "info": {"curve": val_cb.curve, "best_val_sharpe": val_cb.best[0]}}


def save_policy(out: dict, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.zip").write_bytes(out["model_bytes"])
    (path / "obs_rms.pkl").write_bytes(out["obs_rms"])
    (path / "info.json").write_text(json.dumps(out["info"], default=float), encoding="utf-8")


class FrozenNorm:
    """Observation normalisation with frozen running statistics (VecNormalize's formula, clip 10)."""

    def __init__(self, rms, clip: float = 10.0, eps: float = 1e-8):
        self.rms, self.clip, self.eps = rms, clip, eps

    def normalize_obs(self, obs):
        return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + self.eps), -self.clip, self.clip).astype(np.float32)


def load_policy(path: Path, algo: str = "PPO"):
    """(model, frozen normalizer) for PolicyEnsemble."""
    cls = PPO if algo == "PPO" else SAC
    model = cls.load(io.BytesIO((path / "model.zip").read_bytes()), device="cpu")
    return model, FrozenNorm(pickle.loads((path / "obs_rms.pkl").read_bytes()))
