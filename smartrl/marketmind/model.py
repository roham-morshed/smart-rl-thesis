"""MarketMind: Patch-Informer with a sentiment-aware gate and quantile/volatility heads (SPEC §4.1–4.5)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .informer import encoder_out_len, informer_encoder

HORIZONS = (1, 5, 20)
QUANTILES = (0.1, 0.5, 0.9)
VOL_HORIZONS = (5, 20)
N_RET_OUT = 3 * len(HORIZONS)   # q50, Δlo, Δhi per horizon
N_VOL_OUT = len(VOL_HORIZONS)


def make_patches(x: torch.Tensor, patch: int, stride: int) -> torch.Tensor:
    """x [B, L, C] -> [B, C, N, P], end-padded by repeating the last value `stride` times (SPEC §4.1)."""
    x = x.transpose(1, 2)                                   # [B, C, L]
    x = torch.cat([x, x[..., -1:].expand(-1, -1, stride)], dim=-1)
    return x.unfold(-1, patch, stride)                      # [B, C, N, P]


def n_patches(L: int, patch: int, stride: int) -> int:
    return (L + stride - patch) // stride + 1


def split_heads(out: torch.Tensor) -> dict[str, torch.Tensor]:
    """Raw head outputs [B, 11] -> quantiles (no crossing) and log-RV, all in scaled units."""
    r = out[:, :N_RET_OUT].view(-1, len(HORIZONS), 3)
    q50 = r[..., 0]
    q10 = q50 - F.softplus(r[..., 1])
    q90 = q50 + F.softplus(r[..., 2])
    return {"q": torch.stack([q10, q50, q90], dim=-1), "logrv": out[:, N_RET_OUT:]}  # q: [B, 3h, 3τ]


def quantile_vol_loss(pred: dict[str, torch.Tensor], y_ret: torch.Tensor, y_vol: torch.Tensor,
                      vol_weight: float = 0.5) -> torch.Tensor:
    """Mean pinball over the nine (h, τ) pairs + vol_weight · mean MSE of the log-RV heads (SPEC §4.5)."""
    tau = torch.tensor(QUANTILES, device=y_ret.device, dtype=pred["q"].dtype)
    diff = y_ret.unsqueeze(-1) - pred["q"]                     # [B, 3, 3]
    pinball = torch.maximum(tau * diff, (tau - 1) * diff).mean()
    mse = F.mse_loss(pred["logrv"], y_vol)
    return pinball + vol_weight * mse


class MarketMind(nn.Module):
    def __init__(self, c_price: int, c_sent: int, lookback: int = 96, patch: int = 8, stride: int = 4,
                 d_model: int = 64, n_heads: int = 4, d_ff: int = 128, n_layers: int = 2, factor: int = 5,
                 dropout: float = 0.1, n_markets: int = 4, use_sentiment: bool = True):
        super().__init__()
        self.c_price, self.c_sent, self.patch, self.stride = c_price, c_sent, patch, stride
        self.use_sentiment = use_sentiment and c_sent > 0
        self.N = n_patches(lookback, patch, stride)
        self.price_embed = nn.Linear(patch, d_model)
        if self.use_sentiment:
            self.sent_embed = nn.Linear(patch * c_sent, d_model)
            self.gate = nn.Linear(2 * d_model, d_model)
        self.pos = nn.Parameter(torch.zeros(self.N, d_model))
        self.market = nn.Embedding(n_markets, d_model)
        nn.init.normal_(self.pos, std=0.02)
        self.encoder = informer_encoder(d_model, n_heads, d_ff, n_layers, factor, dropout, distil=True, prob=True)
        n_out = encoder_out_len(self.N, n_layers, True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c_price * n_out * d_model, N_RET_OUT + N_VOL_OUT)

    def forward(self, x_price: torch.Tensor, x_sent: torch.Tensor | None, market: torch.Tensor):
        B = x_price.shape[0]
        e = self.price_embed(make_patches(x_price, self.patch, self.stride))      # [B, Cp, N, d]
        if self.use_sentiment:
            ps = make_patches(x_sent, self.patch, self.stride)                      # [B, Cs, N, P]
            s = self.sent_embed(ps.permute(0, 2, 1, 3).reshape(B, self.N, -1))      # [B, N, d]
            s = s.unsqueeze(1).expand_as(e)
            w = torch.sigmoid(self.gate(torch.cat([e, s], dim=-1)))
            e = e + w * s
        e = e + self.pos + self.market(market)[:, None, None, :]
        z = self.encoder(e.reshape(B * self.c_price, self.N, -1))                   # [B·Cp, N', d]
        out = self.head(self.drop(z.reshape(B, -1)))
        return split_heads(out)

    @torch.no_grad()
    def gate_values(self, x_price, x_sent):
        """Mean gate activation per sample (diagnostics)."""
        B = x_price.shape[0]
        e = self.price_embed(make_patches(x_price, self.patch, self.stride))
        ps = make_patches(x_sent, self.patch, self.stride)
        s = self.sent_embed(ps.permute(0, 2, 1, 3).reshape(B, self.N, -1)).unsqueeze(1).expand_as(e)
        return torch.sigmoid(self.gate(torch.cat([e, s], dim=-1))).mean(dim=(1, 2, 3))
