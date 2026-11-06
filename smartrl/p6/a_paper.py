"""A_PAPER: the base paper's GRU→CNN sentiment classifier trained with its own recipe (protocol §2.2, paper §4.5,
Table 2).

Embedding from the frozen CBOW Word2Vec (SPEC §3.1) → GRU(128, full sequence) → Conv1d along time (kernel 3,
64 filters, spanning all GRU units) + ReLU → MaxPool1d(2) → flatten → dropout 0.1 → Linear → softmax (neg/neu/pos).
Cross-entropy (paper eq. 6), 400 epochs, Adam lr 0.01 (the optimiser is not stated), batch 64, no early stopping,
training data SPEC §2.4. Headlines are padded to 48 tokens; GRU outputs at padded positions are zeroed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sentiflow import word2vec
from ..sentiflow.train import DEV, macro_f1, set_seed
from .cfg import P6_DIR, p6

MODEL_DIR = P6_DIR / "a_paper"
DATASETS = ("phrasebank50", "phrasebank_all", "fiqa", "twitter")
POOLED_FROM = ("phrasebank50", "fiqa", "twitter")


class APaper(nn.Module):
    def __init__(self, emb: np.ndarray, gru_hidden: int = 128, filters: int = 64, kernel: int = 3, pool: int = 2,
                 dropout: float = 0.1, max_len: int = 48):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=True, padding_idx=0)
        self.gru = nn.GRU(emb.shape[1], gru_hidden, batch_first=True)
        self.conv = nn.Conv1d(gru_hidden, filters, kernel)
        self.pool = nn.MaxPool1d(pool)
        n_out = (max_len - kernel + 1) // pool
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * n_out, 3)

    def forward(self, ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        m = (torch.arange(ids.shape[1], device=ids.device)[None, :] < lens[:, None]).float()
        x, _ = self.gru(self.emb(ids))
        x = x * m[..., None]
        z = self.pool(torch.relu(self.conv(x.transpose(1, 2))))
        return self.out(self.drop(z.flatten(1)))


def split_frame(df: pd.DataFrame, split: str, datasets=POOLED_FROM) -> pd.DataFrame:
    return df[(df["split"] == split) & df["dataset"].isin(datasets)].reset_index(drop=True)


def eval_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {d: df[(df["split"] == "test") & (df["dataset"] == d)].reset_index(drop=True) for d in DATASETS}
    out["pooled"] = split_frame(df, "test")
    return out


def train(df: pd.DataFrame, wv, seed: int, epochs: int | None = None, log=print) -> tuple[APaper, word2vec.Vocab, dict]:
    c = p6()["e14"]["a_paper"]
    set_seed(seed)
    tr = split_frame(df, "train")
    vocab = word2vec.Vocab(wv, tr["sentence"])
    emb = vocab.embedding_matrix(wv, seed)
    model = APaper(emb, c["gru_hidden"], c["conv_filters"], c["kernel"], c["pool"], c["dropout"], c["max_len"]).to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=c["lr"])
    ids, lens = vocab.encode(tr["sentence"], c["max_len"])
    ids, lens, y = torch.tensor(ids), torch.tensor(lens), torch.tensor(tr["label"].to_numpy())
    g = torch.Generator().manual_seed(seed)
    n_ep = epochs or c["epochs"]
    hist, t0 = [], time.time()
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(len(y), generator=g)
        tot = 0.0
        for i in range(0, len(y), c["batch"]):
            b = perm[i:i + c["batch"]]
            loss = F.cross_entropy(model(ids[b].to(DEV), lens[b].to(DEV)), y[b].to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        hist.append(tot / len(y))
        if ep % 50 == 0 or ep == n_ep - 1:
            log(f"A_PAPER seed {seed} epoch {ep}: train loss {hist[-1]:.4f} ({(time.time() - t0) / 60:.1f} min)")
    return model.eval(), vocab, {"train_loss": hist, "seconds": time.time() - t0, "n_train": int(len(y))}


@torch.no_grad()
def predict(model: APaper, vocab: word2vec.Vocab, texts, batch: int = 4096) -> np.ndarray:
    ids, lens = vocab.encode(texts, model.max_len)
    out = []
    for i in range(0, len(ids), batch):
        logits = model(torch.tensor(ids[i:i + batch]).to(DEV), torch.tensor(lens[i:i + batch]).to(DEV))
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def evaluate(model: APaper, vocab: word2vec.Vocab, df: pd.DataFrame) -> dict[str, dict]:
    res = {}
    for name, d in eval_sets(df).items():
        p = predict(model, vocab, d["sentence"]).argmax(1)
        y = d["label"].to_numpy()
        res[name] = {"acc": float((p == y).mean()), "macro_f1": float(macro_f1(y, p))}
    return res


def save(model: APaper, vocab: word2vec.Vocab, seed: int, info: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "itos": vocab.itos, "n_w2v": vocab.n_w2v, "info": info},
               MODEL_DIR / f"seed{seed}.pt")


def load(seed: int, wv) -> tuple[APaper, word2vec.Vocab]:
    c = p6()["e14"]["a_paper"]
    d = torch.load(MODEL_DIR / f"seed{seed}.pt", map_location="cpu", weights_only=False)
    vocab = word2vec.Vocab.__new__(word2vec.Vocab)
    vocab.itos, vocab.n_w2v = d["itos"], d["n_w2v"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    emb = np.zeros((len(vocab.itos), wv.vector_size), dtype=np.float32)
    model = APaper(emb, c["gru_hidden"], c["conv_filters"], c["kernel"], c["pool"], c["dropout"], c["max_len"])
    model.load_state_dict(d["state"])
    return model.to(DEV).eval(), vocab
