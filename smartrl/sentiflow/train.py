"""Training and prediction for the SentiFlow branches and the fusion gate (SPEC §3.2–3.5).

Reading of the SPEC where it is silent:
- Branch A and A_rev use class-weighted cross-entropy (§3.2, "same training recipe" for A_rev); Branch B lists its
  own recipe without class weights, so it uses plain cross-entropy.
- Weight decay: Branch B 0.01 as specified (not applied to biases and LayerNorm, the usual BERT practice);
  Branch A/A_rev use AdamW's default 0.01 except on the embedding table, whose rows for words absent from the
  labelled data would otherwise decay away from their Word2Vec values.
- "Early stop on validation macro-F1" keeps the weights of the best epoch; patience 3 epochs (Branch A's value)
  for every branch.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .models import BranchA, BranchARev, BranchB, FusionGate

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(y: np.ndarray, k: int = 3) -> torch.Tensor:
    n = np.bincount(y, minlength=k).astype(float)
    return torch.tensor(len(y) / (k * np.maximum(n, 1.0)), dtype=torch.float32)


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))


@dataclass
class History:
    epochs: list = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = -math.inf


# ---------------------------------------------------------------- Branch A / A_rev
def train_rnn(kind: str, emb: np.ndarray, tr: tuple, va: tuple, seed: int, cfg: dict) -> tuple[nn.Module, History]:
    """tr, va = (ids, lens, y) numpy arrays. kind: 'A' (GRU→CNN) or 'A_REV' (CNN→GRU)."""
    set_seed(seed)
    model = (BranchA if kind == "A" else BranchARev)(emb, cfg["gru_hidden"], cfg["conv_filters"], cfg["kernel"],
                                                     cfg.get("dropout", 0.3)).to(DEV)
    emb_p = [p for n, p in model.named_parameters() if n.startswith("emb.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
    opt = torch.optim.AdamW([{"params": emb_p, "weight_decay": 0.0}, {"params": rest}], lr=cfg["lr"])
    w = class_weights(tr[2]).to(DEV)
    g = torch.Generator().manual_seed(seed)
    ids, lens, y = (torch.tensor(a) for a in tr)
    hist, best, bad = History(), None, 0
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(y), generator=g)
        tot = 0.0
        for i in range(0, len(y), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            logits, _ = model(ids[b].to(DEV), lens[b].to(DEV))
            loss = F.cross_entropy(logits, y[b].to(DEV), weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        p, _ = predict_rnn(model, va[0], va[1])
        f1 = macro_f1(va[2], p.argmax(1))
        hist.epochs.append({"epoch": ep, "train_loss": tot / len(y), "val_f1": f1})
        if f1 > hist.best_score:
            hist.best_score, hist.best_epoch, best, bad = f1, ep, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best)
    return model.eval(), hist


@torch.no_grad()
def predict_rnn(model, ids: np.ndarray, lens: np.ndarray, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, hs = [], []
    for i in range(0, len(ids), batch):
        logits, h = model(torch.tensor(ids[i:i + batch]).to(DEV), torch.tensor(lens[i:i + batch]).to(DEV))
        ps.append(torch.softmax(logits.float(), -1).cpu().numpy())
        hs.append(h.float().cpu().numpy())
    return np.concatenate(ps), np.concatenate(hs)


# ---------------------------------------------------------------- Branch B
def _bert_batches(tok, texts: list[str], max_len: int, batch: int, order: np.ndarray):
    for i in range(0, len(order), batch):
        sel = order[i:i + batch]
        enc = tok([texts[j] for j in sel], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield sel, {k: v.to(DEV, non_blocking=True) for k, v in enc.items()}


def train_bert(tok, encoder, tr: tuple, va: tuple, seed: int, cfg: dict, patience: int = 3) -> tuple[nn.Module, History]:
    """tr, va = (texts list, y array). fp16 autocast, linear warm-up over the first 10 % of the steps."""
    set_seed(seed)
    model = BranchB(encoder).to(DEV)
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
               "weight_decay": cfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    steps = cfg["max_epochs"] * math.ceil(len(tr[1]) / cfg["batch"])
    warm = max(1, int(cfg["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("fp16", True))
    rng = np.random.default_rng(seed)
    y = torch.tensor(tr[1])
    hist, best, bad = History(), None, 0
    for ep in range(cfg["max_epochs"]):
        model.train()
        tot = 0.0
        for sel, enc in _bert_batches(tok, tr[0], cfg["max_len"], cfg["batch"], rng.permutation(len(tr[1]))):
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.get("fp16", True)):
                logits, _ = model(**enc)
            loss = F.cross_entropy(logits.float(), y[sel].to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(sel)
        p, _ = predict_bert(model, tok, va[0], cfg["max_len"])
        f1 = macro_f1(va[1], p.argmax(1))
        hist.epochs.append({"epoch": ep, "train_loss": tot / len(y), "val_f1": f1})
        if f1 > hist.best_score:
            hist.best_score, hist.best_epoch, bad = f1, ep, 0
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return model.eval(), hist


@torch.no_grad()
def predict_bert(model, tok, texts: list[str], max_len: int = 64, batch: int = 512,
                 want_h: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Batches are formed from texts sorted by length (less padding); outputs are returned in input order."""
    model.eval()
    order = np.argsort([len(t) for t in texts], kind="stable")
    P = np.zeros((len(texts), 3), dtype=np.float32)
    H = np.zeros((len(texts), model.h_dim), dtype=np.float32) if want_h else None
    for sel, enc in _bert_batches(tok, texts, max_len, batch, order):
        with torch.autocast("cuda", dtype=torch.float16):
            logits, h = model(**enc)
        P[sel] = torch.softmax(logits.float(), -1).cpu().numpy()
        if want_h:
            H[sel] = h.float().cpu().numpy()
    return P, H


# ---------------------------------------------------------------- fusion gate
def train_fusion(tr: dict, va: dict, seed: int, cfg: dict, batch: int = 64, patience: int = 3):
    """tr, va: dicts with h_a, h_b, p_a, p_b, y. Both branches are frozen; only the gate learns (NLL of p_SF)."""
    set_seed(seed)
    gate = FusionGate(tr["h_a"].shape[1], tr["h_b"].shape[1], cfg["gate_hidden"]).to(DEV)
    opt = torch.optim.AdamW(gate.parameters(), lr=cfg["lr"])
    T = {k: torch.tensor(v).to(DEV) for k, v in tr.items()}
    g = torch.Generator().manual_seed(seed)
    best, best_nll, bad, hist = None, math.inf, 0, []
    for ep in range(cfg["max_epochs"]):
        gate.train()
        perm = torch.randperm(len(tr["y"]), generator=g).to(DEV)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            p, _ = gate(T["h_a"][b], T["h_b"][b], T["p_a"][b], T["p_b"][b])
            loss = F.nll_loss(torch.log(p.clamp_min(1e-12)), T["y"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        p, _ = predict_fusion(gate, va)
        nll = float(-np.log(np.clip(p[np.arange(len(va["y"])), va["y"]], 1e-12, 1)).mean())
        hist.append({"epoch": ep, "val_nll": nll})
        if nll < best_nll:
            best_nll, best, bad = nll, copy.deepcopy(gate.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    gate.load_state_dict(best)
    return gate.eval(), hist


@torch.no_grad()
def predict_fusion(gate, d: dict, batch: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    ps, al = [], []
    n = len(d["p_a"])
    for i in range(0, n, batch):
        t = {k: torch.tensor(d[k][i:i + batch]).to(DEV) for k in ("h_a", "h_b", "p_a", "p_b")}
        p, a = gate(t["h_a"], t["h_b"], t["p_a"], t["p_b"])
        ps.append(p.cpu().numpy())
        al.append(a.cpu().numpy())
    return np.concatenate(ps), np.concatenate(al)
