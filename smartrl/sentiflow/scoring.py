"""Corpus scoring (SPEC §3.8): SentiFlow ensemble, calibrated Branch B (for B5) and uncalibrated Branch A (for SENTIN).

Each seed is a full pass over the headlines (Branch A, Branch B, gate). Seed 0 runs first; if 5 × its runtime would
exceed `max_hours_for_ensemble`, only seed 0 is used and `ensemble` is recorded as false.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import PROCESSED, frozen
from . import calibration as C
from . import finbert, word2vec
from .models import BranchA, BranchB, FusionGate
from .train import DEV, predict_bert, predict_rnn

MODELS = PROCESSED / "sentiflow"
SCORES = PROCESSED / "news_scores.parquet"


class SeedScorer:
    def __init__(self, seed: int):
        d = MODELS / f"seed{seed}"
        a = torch.load(d / "branch_a.pt", map_location="cpu", weights_only=False)
        self.itos = a["itos"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_len = a["cfg"]["max_len"]
        emb = np.zeros((len(self.itos), 300), dtype=np.float32)
        self.a = BranchA(emb, a["cfg"]["gru_hidden"], a["cfg"]["conv_filters"], a["cfg"]["kernel"]).to(DEV)
        self.a.load_state_dict(a["state"])
        self.a.eval()
        self.tok, enc = finbert.load_pretrain()
        self.b = BranchB(enc).to(DEV)
        self.b.load_state_dict(torch.load(d / "branch_b.pt", map_location="cpu", weights_only=False)["state"])
        self.b.eval()
        g = torch.load(d / "gate.pt", map_location="cpu", weights_only=False)
        self.gate = FusionGate(self.a.h_dim, self.b.h_dim, frozen()["sentiflow"]["fusion"]["gate_hidden"]).to(DEV)
        self.gate.load_state_dict(g["state"])
        self.gate.eval()
        self.T_sf, self.T_b = g["T_sf"], g["T_b"]

    def encode_a(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        ids = np.zeros((len(texts), self.max_len), dtype=np.int64)
        lens = np.zeros(len(texts), dtype=np.int64)
        for i, s in enumerate(texts):
            t = [self.stoi.get(w, 1) for w in word2vec.tokenize(s)][:self.max_len] or [1]
            ids[i, :len(t)] = t
            lens[i] = len(t)
        return ids, lens

    @torch.no_grad()
    def score(self, texts: list[str]) -> dict[str, np.ndarray]:
        p_a, h_a = predict_rnn(self.a, *self.encode_a(texts))
        p_b, h_b = predict_bert(self.b, self.tok, texts, 64, batch=512)
        p_sf = []
        for i in range(0, len(texts), 65536):
            t = lambda x: torch.tensor(x[i:i + 65536]).to(DEV)
            p, _ = self.gate(t(h_a), t(h_b), t(p_a), t(p_b))
            p_sf.append(p.cpu().numpy())
        return {"sf": C.apply_temperature(np.concatenate(p_sf), self.T_sf), "b": C.apply_temperature(p_b, self.T_b),
                "a": p_a}


def score_corpus(headlines: pd.Series, seeds: list[int], max_hours: float, chunk: int = 200_000,
                 log=print) -> dict:
    """Average per-seed probabilities over the headlines; returns summary and writes SCORES."""
    texts = headlines.astype(str).tolist()
    n = len(texts)
    acc = {k: np.zeros((n, 3), dtype=np.float64) for k in ("sf", "b", "a")}
    used, t_start = [], time.time()
    for k, s in enumerate(seeds):
        t0 = time.time()
        sc = SeedScorer(s)
        for i in range(0, n, chunk):
            out = sc.score(texts[i:i + chunk])
            for key in acc:
                acc[key][i:i + chunk] += out[key]
            if i // chunk % 10 == 0:
                log(f"seed {s}: {min(i + chunk, n):,}/{n:,} headlines, {(time.time() - t0) / 60:.1f} min")
        used.append(s)
        del sc
        torch.cuda.empty_cache()
        if k == 0 and len(seeds) > 1 and (time.time() - t0) * len(seeds) / 3600 > max_hours:
            log(f"5-seed scoring would take {(time.time() - t0) * len(seeds) / 3600:.1f} h > {max_hours} h: seed 0 only")
            break
    for key in acc:
        acc[key] /= len(used)
    cols = {}
    for key, name in (("sf", ""), ("b", "b_"), ("a", "a_")):
        p = acc[key].astype(np.float32)
        cols[f"{name}p_neg"], cols[f"{name}p_neu"], cols[f"{name}p_pos"] = p[:, 0], p[:, 1], p[:, 2]
        cols[f"{name}s"] = p[:, 2] - p[:, 0]
        cols[f"{name}c"] = p.max(axis=1)
    return {"columns": cols, "seeds_used": used, "ensemble": len(used) == len(seeds),
            "hours": (time.time() - t_start) / 3600, "n_headlines": n}
