"""E14 models (protocol §2.5).

- `CNN_P`: the E13 CNN (two Conv1d, 32 filters, kernel 3), each followed by ReLU and MaxPool1d(2), dropout 0.1,
  flatten, Linear.
- `LSTM_P`, `GRU_P`: the E13 RNNs (two layers, hidden 64), dropout 0.1 between the layers and on the last hidden
  state, Linear.
- `SENTIN_PAPER`: SPEC §9.1 sentiment-awareness layer → GRU (hidden 64, one layer; paper eqs 7-11) whose hidden
  states replace the Informer's value embedding (sinusoidal position encoding kept) → SPEC §9.1 Informer
  encoder-decoder → Linear(64→1). The decoder tokens are the last 48 GRU states plus one GRU step on the zero
  placeholder.
- `MM_CLS`: MarketMind (SPEC §4.1-4.3) with the §4.4 heads replaced by one binary head (Linear → logit).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..e13.models import AwarenessLayer, SinusoidalPE
from ..marketmind.informer import AttentionLayer, DecoderLayer, FullAttention, ProbAttention, informer_encoder
from ..marketmind.model import MarketMind, make_patches


class CNNP(nn.Module):
    def __init__(self, c_in: int = 6, lookback: int = 96, filters: int = 32, kernel: int = 3, pool: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c_in, filters, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool1d(pool),
                                 nn.Conv1d(filters, filters, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool1d(pool))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * (lookback // pool // pool), 1)

    def forward(self, x):
        return self.out(self.drop(self.net(x.transpose(1, 2)).flatten(1))).squeeze(-1)


class RNNP(nn.Module):
    def __init__(self, cell: str, c_in: int = 6, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "LSTM" else nn.GRU)(c_in, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(self.drop(o[:, -1])).squeeze(-1)


class SentinPaper(nn.Module):
    def __init__(self, n_inputs: int = 5, lookback: int = 96, label_len: int = 48, d_model: int = 64, heads: int = 4,
                 d_ff: int = 128, e_layers: int = 2, factor: int = 5, dropout: float = 0.1, embed: int = 16,
                 hidden: int = 16, gru_hidden: int = 64):
        super().__init__()
        assert gru_hidden == d_model, "the GRU states are the Informer tokens"
        self.label_len = label_len
        self.aware = AwarenessLayer(n_inputs, embed, hidden)
        self.gru = nn.GRU(n_inputs + 1, gru_hidden, batch_first=True)
        self.pe = SinusoidalPE(d_model, lookback + 8)
        self.drop = nn.Dropout(dropout)
        self.encoder = informer_encoder(d_model, heads, d_ff, e_layers, factor, dropout, distil=True, prob=True)
        self.decoder = DecoderLayer(AttentionLayer(ProbAttention(True, factor, dropout), d_model, heads),
                                    AttentionLayer(FullAttention(False, dropout), d_model, heads), d_model, d_ff, dropout)
        self.dec_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):                                   # x [B, L, n_inputs + 1] -> logit [B]
        xw, _ = self.aware(x)
        H, h_T = self.gru(xw)                               # eqs 7-11: H = GRU(X)
        enc = self.encoder(self.drop(H + self.pe(H)))
        h_ph, _ = self.gru(torch.zeros_like(xw[:, :1]), h_T)   # one step on the zero placeholder
        dec_tok = torch.cat([H[:, -self.label_len:], h_ph], 1)
        dec = self.decoder(self.drop(dec_tok + self.pe(dec_tok)), enc)
        return self.out(self.dec_norm(dec[:, -1])).squeeze(-1)


class MMCls(MarketMind):
    """MarketMind tokens, gate and encoder with one binary head."""

    def __init__(self, c_price: int, c_sent: int, **kw):
        super().__init__(c_price, c_sent, **kw)
        self.head = nn.Linear(self.head.in_features, 1)

    def forward(self, x_price, x_sent, market):
        B = x_price.shape[0]
        e = self.price_embed(make_patches(x_price, self.patch, self.stride))
        if self.use_sentiment:
            ps = make_patches(x_sent, self.patch, self.stride)
            s = self.sent_embed(ps.permute(0, 2, 1, 3).reshape(B, self.N, -1)).unsqueeze(1).expand_as(e)
            w = torch.sigmoid(self.gate(torch.cat([e, s], dim=-1)))
            e = e + w * s
        e = e + self.pos + self.market(market)[:, None, None, :]
        z = self.encoder(e.reshape(B * self.c_price, self.N, -1))
        return self.head(self.drop(z.reshape(B, -1))).squeeze(-1)


def build(name: str, frozen_cfg: dict, p6_cfg: dict, c_price: int = 13, c_sent: int = 6) -> nn.Module:
    e = p6_cfg["e14"]
    L = e["lookback"]
    if name == "CNN_P":
        return CNNP(6, L, e["cnn"]["filters"], e["cnn"]["kernel"], e["paper_nn"]["pool"], e["paper_nn"]["dropout"])
    if name in ("LSTM_P", "GRU_P"):
        return RNNP(name[:-2], 6, e["rnn"]["hidden"], e["rnn"]["layers"], e["paper_nn"]["dropout"])
    if name in ("SENTIN_PAPER", "SENTIN_SF_PAPER"):
        s = frozen_cfg["sentin"]
        i = s["informer"]
        return SentinPaper(5, L, i["label_len"], i["d_model"], i["heads"], i["d_ff"], i["e_layers"], i["factor"],
                           e["sentin_paper"]["dropout"], s["awareness"]["embed_dim"], s["awareness"]["hidden"],
                           e["sentin_paper"]["gru_hidden"])
    if name == "MM_CLS":
        mm = frozen_cfg["marketmind"]
        return MMCls(c_price, c_sent, lookback=mm["lookback"], patch=mm["patch"], stride=mm["stride"],
                     d_model=mm["d_model"], n_heads=mm["heads"], d_ff=mm["d_ff"], n_layers=mm["layers"],
                     factor=mm["probsparse_factor"], dropout=mm["dropout"])
    raise KeyError(name)
