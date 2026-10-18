"""P0 speed benchmark (P0 plan, step 4): FinBERT inference and fine-tuning throughput, PPO and SAC environment
steps per second for 1/2/4/6 parallel jobs on synthetic data, one MarketMind epoch.

Writes `p0.benchmark.*` to results/results.json and results/p0_benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import RAW, RESULTS, frozen  # noqa: E402
from smartrl.runs import run_record  # noqa: E402

BENCH_JSON = RESULTS / "p0_benchmark.json"
N_FEATS = 13 + 6 + 11          # price + sentiment + MarketMind features (full SMART-RL state)


def _texts(n: int) -> list[str]:
    import pandas as pd
    s = RAW / "sentiment"
    parts = [pd.read_parquet(s / "phrasebank50.parquet")["sentence"],
             pd.read_parquet(s / "twitter_train.parquet")["text"]]
    t = pd.concat(parts).astype(str).tolist()
    return (t * (n // len(t) + 1))[:n]


# ------------------------------------------------------------------ FinBERT
def bench_finbert(n: int = 20000) -> dict:
    import torch
    from smartrl.sentiflow.finbert import load_pretrain
    tok, enc = load_pretrain()
    dev = torch.device("cuda")
    head = torch.nn.Linear(768, 3)
    enc, head = enc.to(dev).eval(), head.to(dev)
    texts = _texts(n)
    out: dict = {"n_headlines": n, "max_len": 64}
    t0 = time.perf_counter()
    batches = [tok(texts[i:i + 512], padding=True, truncation=True, max_length=64, return_tensors="pt")
               for i in range(0, n, 512)]
    out["tokenize_per_s"] = n / (time.perf_counter() - t0)
    for bs in (256, 512):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, n, bs):
                b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
                h = enc(**b).last_hidden_state[:, 0]
                head(h).softmax(-1).float().cpu()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out[f"infer_bs{bs}_headlines_per_s"] = n / dt
        out[f"infer_bs{bs}_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    # fine-tuning step throughput (batch 32, fp16, AdamW), SPEC §3.4
    enc.train()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    y = torch.randint(0, 3, (32,), device=dev)
    torch.cuda.reset_peak_memory_stats()
    steps = 60
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(steps):
        b = tok(texts[k * 32:(k + 1) * 32], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(head(enc(**b).last_hidden_state[:, 0]), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    out["train_bs32_samples_per_s"] = steps * 32 / (time.perf_counter() - t0)
    out["train_bs32_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return out


# ------------------------------------------------------------------ PPO / SAC
def _synthetic_market(T: int = 2600, seed: int = 0):
    from smartrl.env.trading_env import MarketData
    rng = np.random.default_rng(seed)
    feats = rng.uniform(-0.5, 1.5, (T, N_FEATS)).astype(np.float32)
    ret = rng.normal(0.0003, 0.011, T)
    return MarketData(feats, ret, e_sa=rng.uniform(0, 1, T))


def _ppo_job(job: int, timesteps: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from smartrl.env.portfolio import RewardSpec
    from smartrl.env.trading_env import TradingEnv
    cfg = frozen()["ppo"]
    data = _synthetic_market(seed=job)
    spec = RewardSpec("R2", lambda1=10, lambda2=0.5)
    venv = DummyVecEnv([lambda i=i: TradingEnv(data, 19, 2200, spec, curriculum=True, seed=100 * job + i)
                        for i in range(cfg["n_envs"])])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", venv, n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
                gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
                ent_coef=cfg["ent_coef"], learning_rate=3e-4, seed=job, device="cpu", verbose=0,
                policy_kwargs={"net_arch": cfg["net_arch"], "activation_fn": torch.nn.Tanh})
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    t_learn = time.perf_counter() - t0
    # one deterministic validation pass (380 days), as done every 10 000 steps
    ev = TradingEnv(data, 2200, 2580, spec, random_start=False)
    obs, _ = ev.reset()
    t0 = time.perf_counter()
    done = False
    while not done:
        a, _ = model.predict(venv.normalize_obs(obs[None]), deterministic=True)
        obs, _, done, _, _ = ev.step(a[0])
    t_eval = time.perf_counter() - t0
    return {"steps_per_s": timesteps / t_learn, "eval_pass_s": t_eval}


def _sac_job(job: int, timesteps: int, learning_starts: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from smartrl.env.portfolio import RewardSpec
    from smartrl.env.trading_env import TradingEnv
    data = _synthetic_market(seed=job)
    spec = RewardSpec("R2", lambda1=10, lambda2=0.5)
    venv = VecNormalize(DummyVecEnv([lambda: TradingEnv(data, 19, 2200, spec, seed=job)]), norm_reward=False)
    model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=200000, batch_size=256, tau=0.005, gamma=0.99,
                ent_coef="auto", learning_starts=learning_starts, seed=job, device="cpu", verbose=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=learning_starts)
    t1 = time.perf_counter()
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    t2 = time.perf_counter()
    return {"warmup_steps_per_s": learning_starts / (t1 - t0), "train_steps_per_s": timesteps / (t2 - t1)}


def bench_rl(timesteps: int, jobs_list=(1, 2, 4, 6)) -> dict:
    from joblib import Parallel, delayed
    out: dict = {"timesteps_per_job": timesteps, "obs_dim": 20 * N_FEATS + 4}
    for n in jobs_list:
        t0 = time.perf_counter()
        res = Parallel(n_jobs=n, backend="loky")(delayed(_ppo_job)(j, timesteps) for j in range(n))
        wall = time.perf_counter() - t0
        per_job = float(np.mean([r["steps_per_s"] for r in res]))
        out[f"ppo_jobs{n}"] = {"per_job_steps_per_s": per_job, "aggregate_steps_per_s": n * timesteps / wall,
                               "wall_s": wall, "eval_pass_s": float(np.mean([r["eval_pass_s"] for r in res]))}
        print(n, out[f"ppo_jobs{n}"], flush=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=4, backend="loky")(delayed(_sac_job)(j, 3000, 1000) for j in range(4))
    out["sac_jobs4"] = {"train_steps_per_s": float(np.mean([r["train_steps_per_s"] for r in res])),
                        "warmup_steps_per_s": float(np.mean([r["warmup_steps_per_s"] for r in res])),
                        "wall_s": time.perf_counter() - t0}
    print("sac", out["sac_jobs4"], flush=True)
    return out


# ------------------------------------------------------------------ MarketMind
def bench_marketmind(days: int = 4300) -> dict:
    import torch
    from smartrl.marketmind.model import MarketMind, quantile_vol_loss
    cfg = frozen()["marketmind"]
    dev = torch.device("cuda")
    L, n_mkt = cfg["lookback"], 4
    n = n_mkt * (days - L)
    rng = np.random.default_rng(0)
    xp = torch.tensor(rng.normal(size=(n_mkt, days, 13)), dtype=torch.float32)
    xs = torch.tensor(rng.normal(size=(n_mkt, days, 6)), dtype=torch.float32)
    idx = [(m, t) for m in range(n_mkt) for t in range(L, days)]
    model = MarketMind(13, 6, L, cfg["patch"], cfg["stride"], cfg["d_model"], cfg["heads"], cfg["d_ff"],
                       cfg["layers"], cfg["probsparse_factor"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    bs = cfg["train"]["batch"]
    order = rng.permutation(len(idx))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(0, len(order), bs):
        sel = [idx[i] for i in order[k:k + bs]]
        bp = torch.stack([xp[m, t - L:t] for m, t in sel]).to(dev, non_blocking=True)
        bsn = torch.stack([xs[m, t - L:t] for m, t in sel]).to(dev, non_blocking=True)
        mk = torch.tensor([m for m, _ in sel], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(bp, bsn, mk)
            loss = quantile_vol_loss(pred, torch.randn(len(sel), 3, device=dev), torch.randn(len(sel), 2, device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"samples": len(idx), "epoch_s": dt, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def bench_ltsf(steps: int = 40) -> dict:
    """Training-step time of the E3 backbone (SPEC §15) at the ETT sizes, extrapolated to one epoch."""
    import torch
    from smartrl.marketmind.informer import encoder_out_len, informer_encoder
    from smartrl.marketmind.model import make_patches, n_patches
    cfg = frozen()["ltsf_benchmark"]
    dev = torch.device("cuda")
    L, P, S, d, C, H = cfg["lookback"], cfg["patch"], cfg["stride"], cfg["d_model"], 7, 720
    N = n_patches(L, P, S)
    emb = torch.nn.Linear(P, d).to(dev)
    enc = informer_encoder(d, cfg["heads"], cfg["d_ff"], cfg["layers"], 5, cfg["dropout"], True, True).to(dev)
    head = torch.nn.Linear(encoder_out_len(N, cfg["layers"], True) * d, H).to(dev)
    params = list(emb.parameters()) + list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    scaler = torch.amp.GradScaler("cuda")
    bs = cfg["train"]["batch"]
    x = torch.randn(bs, L, C, device=dev)
    y = torch.randn(bs * C, H, device=dev)
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            z = enc(emb(make_patches(x, P, S)).reshape(bs * C, N, d))
            loss = torch.nn.functional.mse_loss(head(z.reshape(bs * C, -1)).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / steps
    # standard splits: ETTh1 train = 12 months of hourly rows, ETTm1 = 12 months of 15-minute rows
    n_train = {"ETTh1": 12 * 30 * 24 - L - H + 1, "ETTm1": 12 * 30 * 96 - L - H + 1}
    return {"batch": bs, "step_s": per_step, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "epoch_s_h720": {k: per_step * np.ceil(v / bs) for k, v in n_train.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="finbert,rl,marketmind")
    ap.add_argument("--ppo_steps", type=int, default=20480)
    args = ap.parse_args()
    bench = json.loads(BENCH_JSON.read_text(encoding="utf-8")) if BENCH_JSON.exists() else {}
    for name in args.only.split(","):
        dev = "cpu" if name == "rl" else "cuda"
        with run_record("P0", f"bench_{name}", None, dev, {"section": name}):
            bench[name] = {"finbert": bench_finbert, "marketmind": bench_marketmind, "ltsf": bench_ltsf,
                           "rl": lambda: bench_rl(args.ppo_steps)}[name]()
        print(name, json.dumps(R._clean(bench[name]), indent=1), flush=True)
        BENCH_JSON.write_text(json.dumps(R._clean(bench), indent=1), encoding="utf-8")
    R.update([("p0.benchmark", bench)])


if __name__ == "__main__":
    main()
