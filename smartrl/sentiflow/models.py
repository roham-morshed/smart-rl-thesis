"""SentiFlow networks (SPEC §3.2–3.5): Branch A (GRU→CNN), A_rev (CNN→GRU), Branch B (FinBERT), fusion gate."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def _mask(lens: torch.Tensor, L: int) -> torch.Tensor:
    return torch.arange(L, device=lens.device)[None, :] < lens[:, None]


class BranchA(nn.Module):
    """Embedding → GRU(128, sequence) → Conv1d(128→64, k3, p1) + ReLU → global max-pool → dropout → Linear(64→3).
    h_A is the pooled 64-d vector. Padded positions are zeroed before the convolution and excluded from the pool."""

    def __init__(self, emb: np.ndarray, gru_hidden: int = 128, filters: int = 64, kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=False, padding_idx=0)
        self.gru = nn.GRU(emb.shape[1], gru_hidden, batch_first=True)
        self.conv = nn.Conv1d(gru_hidden, filters, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters, 3)
        self.h_dim = filters

    def forward(self, ids, lens):
        m = _mask(lens, ids.shape[1])
        x, _ = self.gru(self.emb(ids))
        x = x * m[..., None]
        z = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        h = z.masked_fill(~m[..., None], float("-inf")).max(dim=1).values
        return self.out(self.drop(h)), h


class BranchARev(nn.Module):
    """Embedding → Conv1d(300→64, k3, p1) + ReLU → GRU(128) → hidden state at the last real token → Linear(128→3)."""

    def __init__(self, emb: np.ndarray, gru_hidden: int = 128, filters: int = 64, kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=False, padding_idx=0)
        self.conv = nn.Conv1d(emb.shape[1], filters, kernel, padding=kernel // 2)
        self.gru = nn.GRU(filters, gru_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(gru_hidden, 3)
        self.h_dim = gru_hidden

    def forward(self, ids, lens):
        m = _mask(lens, ids.shape[1])
        z = torch.relu(self.conv(self.emb(ids).transpose(1, 2))).transpose(1, 2) * m[..., None]
        packed = pack_padded_sequence(z, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        return self.out(self.drop(h)), h


class BranchB(nn.Module):
    """finbert-pretrain + Linear(768→3) on [CLS]; h_B = [CLS]."""

    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.out = nn.Linear(encoder.config.hidden_size, 3)
        self.h_dim = encoder.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids).last_hidden_state[:, 0]
        return self.out(h), h


class FusionGate(nn.Module):
    """α(x) = σ(W2·ReLU(W1·[h_A; h_B] + b1) + b2); p_SF = α·p_A + (1 − α)·p_B (the proposal's Figure 1)."""

    def __init__(self, d_a: int = 64, d_b: int = 768, hidden: int = 32):
        super().__init__()
        self.w1 = nn.Linear(d_a + d_b, hidden)
        self.w2 = nn.Linear(hidden, 1)

    def alpha(self, h_a, h_b):
        return torch.sigmoid(self.w2(torch.relu(self.w1(torch.cat([h_a, h_b], dim=-1))))).squeeze(-1)

    def forward(self, h_a, h_b, p_a, p_b):
        a = self.alpha(h_a, h_b)
        return a[:, None] * p_a + (1 - a[:, None]) * p_b, a
