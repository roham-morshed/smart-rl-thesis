"""Informer building blocks (Zhou et al. 2021): ProbSparse attention, distilling layer, encoder and decoder layers.

Follows the reference implementation (zhouhaoyi/Informer2020) so that the factor c, the sampling of keys
(c·⌈ln L_K⌉) and the number of active queries (c·⌈ln L_Q⌉) mean the same as in the paper.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, mask_flag: bool = False, dropout: float = 0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):  # [B, L, H, E]
        B, L, H, E = q.shape
        scores = torch.einsum("blhe,bshe->bhls", q, k) / math.sqrt(E)
        if self.mask_flag:
            S = k.shape[1]
            mask = torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        a = self.dropout(torch.softmax(scores.float(), dim=-1).to(v.dtype))
        return torch.einsum("bhls,bshd->blhd", a, v).contiguous()


class ProbAttention(nn.Module):
    def __init__(self, mask_flag: bool = False, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.mask_flag, self.factor = mask_flag, factor
        self.dropout = nn.Dropout(dropout)

    def _prob_qk(self, Q, K, sample_k, n_top):  # Q, K: [B, H, L, E]
        B, H, L_K, E = K.shape
        L_Q = Q.shape[2]
        idx = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)[:, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), idx, :]
        qk_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = qk_sample.max(-1)[0] - qk_sample.sum(-1) / L_K
        top = M.topk(n_top, sorted=False)[1]
        Q_red = Q[torch.arange(B, device=Q.device)[:, None, None], torch.arange(H, device=Q.device)[None, :, None], top, :]
        return torch.matmul(Q_red, K.transpose(-2, -1)), top

    def forward(self, q, k, v):  # [B, L, H, E]
        B, L_Q, H, E = q.shape
        L_K = k.shape[1]
        Q, K, V = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        U_part = min(self.factor * int(np.ceil(np.log(L_K))), L_K)
        u = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)
        scores, top = self._prob_qk(Q, K, U_part, u)
        scores = scores / math.sqrt(E)
        if self.mask_flag:
            ctx = V.cumsum(dim=-2)
            mask = torch.ones(L_Q, L_K, dtype=torch.bool, device=q.device).triu(1)
            mask = mask[None, None].expand(B, H, L_Q, L_K)[
                torch.arange(B, device=q.device)[:, None, None], torch.arange(H, device=q.device)[None, :, None], top, :]
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            ctx = V.mean(dim=-2, keepdim=True).expand(B, H, L_Q, V.shape[-1]).clone()
        attn = self.dropout(torch.softmax(scores.float(), dim=-1).to(V.dtype))
        ctx = ctx.to(V.dtype)
        ctx[torch.arange(B, device=q.device)[:, None, None], torch.arange(H, device=q.device)[None, :, None], top, :] = \
            torch.matmul(attn, V).to(ctx.dtype)
        return ctx.transpose(1, 2).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int):
        super().__init__()
        d_k = d_model // n_heads
        self.inner, self.H = attention, n_heads
        self.q = nn.Linear(d_model, d_k * n_heads)
        self.k = nn.Linear(d_model, d_k * n_heads)
        self.v = nn.Linear(d_model, d_k * n_heads)
        self.o = nn.Linear(d_k * n_heads, d_model)

    def forward(self, xq, xk, xv):
        B, L, _ = xq.shape
        S, H = xk.shape[1], self.H
        out = self.inner(self.q(xq).view(B, L, H, -1), self.k(xk).view(B, S, H, -1), self.v(xv).view(B, S, H, -1))
        return self.o(out.view(B, L, -1))


class DistilLayer(nn.Module):
    """Conv1d(k=3) + BatchNorm + ELU + MaxPool(k=3, s=2): halves the sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode="circular")
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):  # [B, L, D]
        x = self.pool(self.act(self.norm(self.conv(x.transpose(1, 2)))))
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.attn = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x, x, x)))
        y = self.drop(self.act(self.conv1(x.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    """attn_layers[0], distil, attn_layers[1], distil, ... , attn_layers[-1], LayerNorm."""

    def __init__(self, layers: list[EncoderLayer], distils: list[DistilLayer] | None, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.distils = nn.ModuleList(distils) if distils else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.distils is not None:
            for layer, distil in zip(self.layers[:-1], self.distils):
                x = distil(layer(x))
            x = self.layers[-1](x)
        else:
            for layer in self.layers:
                x = layer(x)
        return self.norm(x)


def informer_encoder(d_model: int, n_heads: int, d_ff: int, n_layers: int, factor: int, dropout: float,
                     distil: bool = True, prob: bool = True) -> Encoder:
    att = (lambda: ProbAttention(False, factor, dropout)) if prob else (lambda: FullAttention(False, dropout))
    layers = [EncoderLayer(AttentionLayer(att(), d_model, n_heads), d_model, d_ff, dropout) for _ in range(n_layers)]
    distils = [DistilLayer(d_model) for _ in range(n_layers - 1)] if distil else None
    return Encoder(layers, distils, d_model)


def encoder_out_len(L: int, n_layers: int, distil: bool) -> int:
    if not distil:
        return L
    for _ in range(n_layers - 1):
        L = (L + 2 - 3) // 2 + 1
    return L


class DecoderLayer(nn.Module):
    def __init__(self, self_attn: AttentionLayer, cross_attn: AttentionLayer, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn, self.cross_attn = self_attn, cross_attn
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x)))
        x = self.norm2(x + self.drop(self.cross_attn(x, cross, cross)))
        y = self.drop(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)
