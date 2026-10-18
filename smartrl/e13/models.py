"""Direction models of E13 and the SENTIN reproduction of the base paper (SPEC §9.1, §15).

SENTIN (Chauhan, Ahmed & Sinha 2025, SenT-In):
- sentiment-awareness layer: for each input i (open, high, low, close, volume) e_i = Linear_i(x_i) ∈ R^16 and
  e_s = Linear_s(S_t); weight w_i = σ(MLP([e_i; e_s; e_i ⊙ e_s])) with one hidden layer of 16 (ReLU); w_s from
  (e_s, e_s); the weighted inputs w_i · x_i (and w_s · S_t) are the Informer inputs;
- Informer encoder–decoder: 2 ProbSparse encoder layers (factor 5) with a distilling layer, 1 decoder layer
  (masked ProbSparse self-attention + full cross-attention), d_model 64, 4 heads, d_ff 128, dropout 0.1;
  decoder input = the last 48 steps plus one zero placeholder; Linear(64→1) on the placeholder = logit of P(up).
Token embedding as in the Informer reference code: Conv1d(k=3, circular) plus a sinusoidal position encoding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..marketmind.informer import AttentionLayer, DecoderLayer, FullAttention, ProbAttention, informer_encoder


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[: x.shape[1]][None]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class AwarenessLayer(nn.Module):
    """Sentiment-awareness layer; input [B, L, n_inputs + 1] with the sentiment index in the last column."""

    def __init__(self, n_inputs: int, embed: int = 16, hidden: int = 16):
        super().__init__()
        self.n = n_inputs
        self.emb = nn.ModuleList([nn.Linear(1, embed) for _ in range(n_inputs)])
        self.emb_s = nn.Linear(1, embed)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(3 * embed, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                                  for _ in range(n_inputs + 1)])

    def forward(self, x):
        s = x[..., -1:]
        e_s = self.emb_s(s)
        ws = []
        for i in range(self.n):
            e_i = self.emb[i](x[..., i:i + 1])
            ws.append(torch.sigmoid(self.mlp[i](torch.cat([e_i, e_s, e_i * e_s], -1))))
        ws.append(torch.sigmoid(self.mlp[self.n](torch.cat([e_s, e_s, e_s * e_s], -1))))
        w = torch.cat(ws, -1)
        return w * x, w


class SentIn(nn.Module):
    def __init__(self, n_inputs: int = 5, lookback: int = 96, label_len: int = 48, d_model: int = 64, heads: int = 4,
                 d_ff: int = 128, e_layers: int = 2, factor: int = 5, dropout: float = 0.1, embed: int = 16,
                 hidden: int = 16):
        super().__init__()
        self.label_len = label_len
        c = n_inputs + 1
        self.aware = AwarenessLayer(n_inputs, embed, hidden)
        self.enc_emb, self.dec_emb = TokenEmbedding(c, d_model), TokenEmbedding(c, d_model)
        self.pe = SinusoidalPE(d_model, lookback + 8)
        self.drop = nn.Dropout(dropout)
        self.encoder = informer_encoder(d_model, heads, d_ff, e_layers, factor, dropout, distil=True, prob=True)
        self.decoder = DecoderLayer(AttentionLayer(ProbAttention(True, factor, dropout), d_model, heads),
                                    AttentionLayer(FullAttention(False, dropout), d_model, heads), d_model, d_ff, dropout)
        self.dec_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):                                   # x [B, L, n_inputs + 1] -> logit [B]
        xw, _ = self.aware(x)
        enc = self.encoder(self.drop(self.enc_emb(xw) + self.pe(xw)))
        dec_in = torch.cat([xw[:, -self.label_len:], torch.zeros_like(xw[:, :1])], 1)
        dec = self.decoder(self.drop(self.dec_emb(dec_in) + self.pe(dec_in)), enc)
        return self.out(self.dec_norm(dec[:, -1])).squeeze(-1)


class CNNDir(nn.Module):
    """Two Conv1d layers (32 filters, kernel 3, ReLU), global max-pool, linear."""

    def __init__(self, c_in: int = 6, filters: int = 32, kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c_in, filters, kernel, padding=1), nn.ReLU(),
                                 nn.Conv1d(filters, filters, kernel, padding=1), nn.ReLU())
        self.out = nn.Linear(filters, 1)

    def forward(self, x):
        return self.out(self.net(x.transpose(1, 2)).max(dim=-1).values).squeeze(-1)


class RNNDir(nn.Module):
    """LSTM or GRU, two layers, hidden 64, last hidden state → linear."""

    def __init__(self, cell: str, c_in: int = 6, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "LSTM" else nn.GRU)(c_in, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(o[:, -1]).squeeze(-1)


def build(name: str, cfg: dict) -> nn.Module:
    s = cfg["sentin"]
    if name in ("SENTIN", "SENTIN_SF"):
        i = s["informer"]
        return SentIn(5, s["lookback"], i["label_len"], i["d_model"], i["heads"], i["d_ff"], i["e_layers"], i["factor"],
                      i["dropout"], s["awareness"]["embed_dim"], s["awareness"]["hidden"])
    if name == "CNN":
        return CNNDir(6, cfg["e13"]["cnn"]["filters"], cfg["e13"]["cnn"]["kernel"])
    if name in ("LSTM", "GRU"):
        return RNNDir(name, 6, cfg["e13"]["rnn"]["hidden"], cfg["e13"]["rnn"]["layers"])
    raise KeyError(name)
