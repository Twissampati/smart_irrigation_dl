"""Baseline DL models and the proposed SoilCastNet.

Every model maps x:(B,L,F) -> (B,H,3): the 5/50/95 % quantiles of the normalised target at
H forecast horizons. Channel 0 of x is always the (normalised) soil-moisture target.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.05, 0.5, 0.95)


def _head(d, H, Q=3): return nn.Linear(d, H * Q)


class Base(nn.Module):
    def __init__(self, H): super().__init__(); self.H = H
    def finish(self, o):                       # (B,H*3)->(B,H,3) sorted (no quantile crossing)
        return torch.sort(o.view(-1, self.H, 3), dim=-1).values
    aux = None


# ------------------------------------------------------------------ baselines
class MLP(Base):
    def __init__(self, L, Fin, H, d=256, p=0.1):
        super().__init__(H)
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(L * Fin, d), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), _head(d, H))
    def forward(self, x): return self.finish(self.net(x))


class LSTM(Base):
    def __init__(self, L, Fin, H, d=64, layers=2, p=0.1, cell=nn.LSTM):
        super().__init__(H)
        self.rnn = cell(Fin, d, layers, batch_first=True, dropout=p); self.out = _head(d, H)
    def forward(self, x): return self.finish(self.out(self.rnn(x)[0][:, -1]))


class GRU(LSTM):
    def __init__(self, *a, **k): super().__init__(*a, cell=nn.GRU, **k)


class CNN1D(Base):
    def __init__(self, L, Fin, H, d=64, p=0.1):
        super().__init__(H)
        self.conv = nn.Sequential(nn.Conv1d(Fin, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, d, 3, padding=1), nn.ReLU(),
                                  nn.MaxPool1d(2), nn.Conv1d(d, d, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.out = nn.Sequential(nn.Dropout(p), _head(d, H))
    def forward(self, x): return self.finish(self.out(self.conv(x.transpose(1, 2)).squeeze(-1)))


class CNNLSTM(Base):
    """The CNN-LSTM used in the authors' earlier simulation paper."""
    def __init__(self, L, Fin, H, d=64, p=0.1):
        super().__init__(H)
        self.conv = nn.Sequential(nn.Conv1d(Fin, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, d, 3, padding=1), nn.ReLU())
        self.rnn = nn.LSTM(d, d, batch_first=True); self.out = nn.Sequential(nn.Dropout(p), _head(d, H))
    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.finish(self.out(self.rnn(h)[0][:, -1]))


class _Chomp(nn.Module):
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, x): return x[..., :-self.n] if self.n else x


class TCN(Base):
    def __init__(self, L, Fin, H, d=64, k=3, dil=(1, 2, 4, 8), p=0.1):
        super().__init__(H)
        blocks, c = [], Fin
        for dl in dil:
            pad = (k - 1) * dl
            blocks.append(nn.Sequential(nn.Conv1d(c, d, k, padding=pad, dilation=dl), _Chomp(pad), nn.ReLU(), nn.Dropout(p),
                                        nn.Conv1d(d, d, k, padding=pad, dilation=dl), _Chomp(pad), nn.ReLU()))
            c = d
        self.blocks = nn.ModuleList(blocks); self.skip = nn.ModuleList([nn.Conv1d(Fin if i == 0 else d, d, 1) for i in range(len(dil))])
        self.out = _head(d, H)
    def forward(self, x):
        h = x.transpose(1, 2)
        for b, s in zip(self.blocks, self.skip): h = F.relu(b(h) + s(h))
        return self.finish(self.out(h[:, :, -1]))


class TransformerEnc(Base):
    def __init__(self, L, Fin, H, d=64, heads=4, layers=2, p=0.1):
        super().__init__(H)
        self.inp = nn.Linear(Fin, d)
        pe = torch.zeros(L, d); pos = torch.arange(L).unsqueeze(1); div = torch.exp(torch.arange(0, d, 2) * (-math.log(1e4) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div); self.register_buffer("pe", pe)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, p, batch_first=True, norm_first=True), layers)
        self.out = _head(d, H)
    def forward(self, x):
        h = self.enc(self.inp(x) + self.pe)
        return self.finish(self.out(h[:, -1] + h.mean(1)))


class DLinear(Base):
    """Zeng et al. 2023: trend/seasonal decomposition + linear maps."""
    def __init__(self, L, Fin, H, ma=25):
        super().__init__(H); self.ma = ma
        self.lt, self.ls = nn.Linear(L * Fin, H * 3), nn.Linear(L * Fin, H * 3)
    def forward(self, x):
        pad = self.ma // 2
        xp = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)], 1)
        trend = F.avg_pool1d(xp.transpose(1, 2), self.ma, 1).transpose(1, 2)
        return self.finish(self.lt(trend.flatten(1)) + self.ls((x - trend).flatten(1)))


# ------------------------------------------------------------------ proposed model
class SoilCastNet(nn.Module):
    """Physics-guided, multi-scale, event-gated quantile forecaster.

    1. Multi-scale encoder: parallel dilated causal convolutions (hours -> days) capture
       diurnal ET cycles and multi-day drainage at once; a GRU + temporal attention summarises them.
    2. Physics-guided decomposition of the forecast (replaces a free 'black-box' output):
           median_h = a - lambda_h * relu(a - floor) + g_h * r_h
       a: last observed soil moisture;  lambda_h in (0,1): fraction of the plant-available
       water lost by drainage + evapotranspiration by horizon h;  floor: learnable residual
       (wilting-like) water content;  r_h >= 0: recharge magnitude;  g_h in (0,1): probability that a
       wetting event (rain/irrigation) occurs before h  (supervised with the observed events).
       The drying and wetting terms are signed by construction, so the model cannot hallucinate
       a rise without the gate opening -> interpretable and better behaved under shift.
    3. Non-crossing 5/50/95 % quantiles (learned positive widths), calibrated afterwards with
       conformalised quantile regression.
    4. Physics-aware loss: pinball + range penalty + 'no rise without gate' penalty + gate BCE.
    ablation flags: multiscale / physics / gate
    """
    def __init__(self, L, Fin, H, d=64, dil=(1, 2, 4, 8), k=3, p=0.1, multiscale=True, physics=True, gate=True):
        super().__init__()
        self.H, self.physics, self.use_gate = H, physics, gate
        dil = dil if multiscale else (1,)
        self.branches = nn.ModuleList([nn.Sequential(nn.ConstantPad1d(((k - 1) * dl, 0), 0.0), nn.Conv1d(Fin, d, k, dilation=dl), nn.GELU()) for dl in dil])
        self.mix = nn.Sequential(nn.Linear(d * len(dil), d), nn.LayerNorm(d))
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p))
        self.lam, self.rech, self.gate = nn.Linear(d, H), nn.Linear(d, H), nn.Linear(d, H)
        self.wlo, self.whi = nn.Linear(d, H), nn.Linear(d, H)
        self.direct = nn.Linear(d, H)                     # used when physics=False (plain residual head)
        self.floor = nn.Parameter(torch.tensor(-1.5))     # learnable residual-water level (z-units)
        self.aux = None

    def forward(self, x):
        a = x[:, -1, 0:1]                                          # last observed target (B,1)
        h = x.transpose(1, 2)
        h = torch.cat([b(h) for b in self.branches], 1).transpose(1, 2)
        h = self.mix(h)
        o, _ = self.rnn(h)
        w = torch.softmax(self.att(o), 1)
        z = self.fuse(torch.cat([(w * o).sum(1), o[:, -1]], -1))
        g = torch.sigmoid(self.gate(z)) if self.use_gate else torch.ones(x.size(0), self.H, device=x.device)
        if self.physics:
            lam = torch.sigmoid(self.lam(z) - 2.0)                 # start with slow drying
            rec = F.softplus(self.rech(z))
            med = a - lam * F.relu(a - self.floor) + g * rec
        else:
            lam, rec = None, None
            med = a + self.direct(z)
        lo = med - F.softplus(self.wlo(z)) - 1e-3
        hi = med + F.softplus(self.whi(z)) + 1e-3
        self.aux = dict(gate=g, lam=lam, rech=rec, anchor=a)
        return torch.stack([lo, med, hi], -1)


# ------------------------------------------------------------------ losses
def pinball(pred, y, qs=QUANTILES):
    q = torch.tensor(qs, device=pred.device).view(1, 1, -1)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def soilcast_loss(model, pred, y, bounds, w_phys=0.1, w_gate=0.1, eps=0.05):
    """pinball + physics-guided regularisers (only used for SoilCastNet)."""
    loss = pinball(pred, y)
    aux = model.aux
    lo_b, hi_b = bounds
    rng = (F.relu(lo_b - pred) + F.relu(pred - hi_b)).mean()               # stay in plausible range
    loss = loss + w_phys * rng
    if model.use_gate and aux is not None:
        a = aux["anchor"]; g = aux["gate"]
        event = ((y - a) > eps).float()                                    # observed wetting
        loss = loss + w_gate * F.binary_cross_entropy(g, event)
        rise = F.relu(pred[..., 1] - a)                                    # rise only if gate open
        loss = loss + w_phys * (rise * (1 - g)).mean()
    return loss


def make_model(name, L, Fin, H, **kw):
    reg = {"MLP": MLP, "LSTM": LSTM, "GRU": GRU, "CNN1D": CNN1D, "CNN-LSTM": CNNLSTM, "TCN": TCN,
           "Transformer": TransformerEnc, "DLinear": DLinear}
    if name in reg: return reg[name](L, Fin, H)
    flags = {"SoilCastNet": {}, "SoilCastNet w/o multiscale": dict(multiscale=False),
             "SoilCastNet w/o physics": dict(physics=False), "SoilCastNet w/o gate": dict(gate=False)}
    return SoilCastNet(L, Fin, H, **flags[name])
