"""H4 comparison models (SPEC §4.6) with MarketMind's inputs, heads and losses.

Every model receives the market identity, as MarketMind does through its market embedding (one pooled model over
four markets): a learned embedding added to the tokens (INFORMER, PATCHTST) or a one-hot vector concatenated to
the inputs (LSTM, GRU, DLINEAR). `price_only=True` drops the sentiment channels (LSTM_PRICE, baseline B4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .informer import AttentionLayer, Encoder, EncoderLayer, FullAttention, encoder_out_len, informer_encoder
from .model import N_RET_OUT, N_VOL_OUT, MarketMind, make_patches, n_patches, split_heads

N_OUT = N_RET_OUT + N_VOL_OUT


def _inputs(x_price, x_sent, price_only: bool):
    return x_price if price_only or x_sent is None else torch.cat([x_price, x_sent], dim=-1)


class RNNModel(nn.Module):
    """LSTM or GRU: two layers, hidden 64, one step per day, last hidden state → heads."""

    def __init__(self, cell: str, c_in: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1,
                 n_markets: int = 4, price_only: bool = False):
        super().__init__()
        self.price_only, self.n_markets = price_only, n_markets
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(c_in + n_markets, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, N_OUT)

    def forward(self, x_price, x_sent, market):
        x = _inputs(x_price, x_sent, self.price_only)
        oh = F.one_hot(market, self.n_markets).to(x.dtype)[:, None, :].expand(-1, x.shape[1], -1)
        out, _ = self.rnn(torch.cat([x, oh], dim=-1))
        return split_heads(self.head(out[:, -1]))


class InformerModel(nn.Module):
    """Vanilla Informer: one token per day (Linear(C→64)), the MarketMind encoder, no patching, no gate."""

    def __init__(self, c_in: int, lookback: int = 96, d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
                 n_layers: int = 2, factor: int = 5, dropout: float = 0.1, n_markets: int = 4):
        super().__init__()
        self.embed = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(lookback, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.market = nn.Embedding(n_markets, d_model)
        self.encoder = informer_encoder(d_model, n_heads, d_ff, n_layers, factor, dropout, distil=True, prob=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder_out_len(lookback, n_layers, True) * d_model, N_OUT)

    def forward(self, x_price, x_sent, market):
        x = _inputs(x_price, x_sent, False)
        z = self.embed(x) + self.pos + self.market(market)[:, None, :]
        z = self.encoder(z)
        return split_heads(self.head(self.drop(z.reshape(z.shape[0], -1))))


class PatchTSTModel(nn.Module):
    """Channel-independent patches (P 8, S 4) of every channel (sentiment as extra channels), three full-attention
    Transformer layers (d 64, 4 heads), flatten → heads."""

    def __init__(self, c_in: int, lookback: int = 96, patch: int = 8, stride: int = 4, d_model: int = 64,
                 n_heads: int = 4, d_ff: int = 128, n_layers: int = 3, dropout: float = 0.1, n_markets: int = 4):
        super().__init__()
        self.patch, self.stride, self.c_in = patch, stride, c_in
        self.N = n_patches(lookback, patch, stride)
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(self.N, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.market = nn.Embedding(n_markets, d_model)
        layers = [EncoderLayer(AttentionLayer(FullAttention(False, dropout), d_model, n_heads), d_model, d_ff, dropout)
                  for _ in range(n_layers)]
        self.encoder = Encoder(layers, None, d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c_in * self.N * d_model, N_OUT)

    def forward(self, x_price, x_sent, market):
        x = _inputs(x_price, x_sent, False)
        B = x.shape[0]
        e = self.embed(make_patches(x, self.patch, self.stride)) + self.pos + self.market(market)[:, None, None, :]
        z = self.encoder(e.reshape(B * self.c_in, self.N, -1))
        return split_heads(self.head(self.drop(z.reshape(B, -1))))


class DLinearModel(nn.Module):
    """Moving-average decomposition (kernel 25); one linear map per channel over L for the trend and the remainder;
    a final linear layer mixes the channels (and the market one-hot) into the heads."""

    def __init__(self, c_in: int, lookback: int = 96, kernel: int = 25, n_markets: int = 4):
        super().__init__()
        self.kernel, self.n_markets = kernel, n_markets
        self.trend = nn.Parameter(torch.zeros(c_in, lookback, N_OUT))
        self.season = nn.Parameter(torch.zeros(c_in, lookback, N_OUT))
        nn.init.xavier_uniform_(self.trend)
        nn.init.xavier_uniform_(self.season)
        self.mix = nn.Linear(c_in * N_OUT + n_markets, N_OUT)

    def forward(self, x_price, x_sent, market):
        x = _inputs(x_price, x_sent, False)                                  # [B, L, C]
        pad = (self.kernel - 1) // 2
        xt = x.transpose(1, 2)
        xp = torch.cat([xt[..., :1].expand(-1, -1, pad), xt, xt[..., -1:].expand(-1, -1, self.kernel - 1 - pad)], -1)
        trend = F.avg_pool1d(xp, self.kernel, stride=1)                       # [B, C, L]
        season = xt - trend
        z = torch.einsum("bcl,clo->bco", trend, self.trend) + torch.einsum("bcl,clo->bco", season, self.season)
        z = torch.cat([z.reshape(z.shape[0], -1), F.one_hot(market, self.n_markets).to(z.dtype)], -1)
        return split_heads(self.mix(z))


def build(variant: str, c_price: int, c_sent: int, cfg: dict) -> nn.Module:
    """Neural forecast variants of SPEC §4.6–4.7 by name."""
    mm = dict(lookback=cfg["lookback"], d_model=cfg["d_model"], n_heads=cfg["heads"], d_ff=cfg["d_ff"],
              n_layers=cfg["layers"], factor=cfg["probsparse_factor"], dropout=cfg["dropout"])
    c_all = c_price + c_sent
    if variant == "MM":
        return MarketMind(c_price, c_sent, patch=cfg["patch"], stride=cfg["stride"], **mm)
    if variant == "MM_NOSENT":
        return MarketMind(c_price, c_sent, patch=cfg["patch"], stride=cfg["stride"], use_sentiment=False, **mm)
    if variant in ("LSTM", "GRU"):
        return RNNModel(variant, c_all, dropout=cfg["dropout"])
    if variant == "LSTM_PRICE":
        return RNNModel("LSTM", c_price, dropout=cfg["dropout"], price_only=True)
    if variant == "INFORMER":
        return InformerModel(c_all, **mm)
    if variant == "PATCHTST":
        return PatchTSTModel(c_all, cfg["lookback"], cfg["patch"], cfg["stride"], cfg["d_model"], cfg["heads"],
                             cfg["d_ff"], 3, cfg["dropout"])
    if variant == "DLINEAR":
        return DLinearModel(c_all, cfg["lookback"])
    raise KeyError(variant)
