"""
Temperature network architectures for the Asymmetric Duo.

All three networks share the same architecture skeleton and initialise to the
known-good Naive TS baseline, so training starts at a calibrated state.

Architecture (per-sample path):
    feat_l → LayerNorm(dim_large) → [Linear(dim_large, proj_dim) → ReLU]?
    feat_s → LayerNorm(dim_small) → [Linear(dim_small, proj_dim) → ReLU]?
    → concat → MLP trunk → output head

When proj_dim is set (e.g. 512), both model features are projected to the
same dimension before concatenation.  This equalises the contribution of the
two models (ResNet-50's 2048-dim features would otherwise dominate the 768-dim
ViT features numerically) and reduces the input dimensionality of the trunk.

Classes
-------
ResidualJointTempNet  — T = softplus(δ + inv_softplus(T_naive)), zero-init δ
DirectJointTempNet    — T = softplus(u), bias-init to inv_softplus(T_naive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.duo.dann import ReverseLayerF

def inv_softplus(y):
    """The inverse of softplus."""
    y = torch.as_tensor(y)
    return torch.where(y > 87.5, y, torch.log(torch.expm1(y)))


def _make_proj(in_dim: int, proj_dim: int | None) -> nn.Module | None:
    """Return a Linear+ReLU projection, or None if proj_dim is unset."""
    if proj_dim is None:
        return None
    return nn.Sequential(nn.Linear(in_dim, proj_dim), nn.ReLU())

class DirectJointTempNet(nn.Module):
    """
    Predicts per-sample temperatures for the Asymmetric Duo directly.

    Parametrization:
        T_l = softplus(u_l)   where u_l = out(trunk(feat))

    Initialization:
        out.weight = 0,  out.bias = [inv_softplus(T_l_naive), inv_softplus(T_s_naive)]
        → at init: u = bias → T = softplus(inv_softplus(T_naive)) = T_naive exactly
    """

    def __init__(self, dim_large: int, dim_small: int,
                 T_l_naive: float, T_s_naive: float,
                 hidden_dim: int = 256, n_layers: int = 2,
                 dropout_p: float = 0.0,
                 proj_dim: int | None = None):
        super().__init__()
        self.register_buffer("T_l_naive", torch.tensor(T_l_naive, dtype=torch.float32))
        self.register_buffer("T_s_naive", torch.tensor(T_s_naive, dtype=torch.float32))

        self.norm_l = nn.LayerNorm(dim_large)
        self.norm_s = nn.LayerNorm(dim_small)

        self.proj_l = _make_proj(dim_large, proj_dim)
        self.proj_s = _make_proj(dim_small, proj_dim)

        in_dim  = (proj_dim or dim_large) + (proj_dim or dim_small)
        layers  = []
        cur_dim = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(cur_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
            cur_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Zero-init weights; bias initialised so that softplus(bias) = T_naive exactly.
        self.out = nn.Linear(cur_dim, 2)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.stack([
                inv_softplus(torch.tensor(T_l_naive, dtype=torch.float32)),
                inv_softplus(torch.tensor(T_s_naive, dtype=torch.float32)),
            ]))

    @property
    def trunk_out_dim(self) -> int:
        return self.out.in_features

    def forward(self, feat_l: torch.Tensor, feat_s: torch.Tensor,
                return_features: bool = False):
        fl = self.norm_l(feat_l)
        fs = self.norm_s(feat_s)
        if self.proj_l is not None:
            fl = self.proj_l(fl)
            fs = self.proj_s(fs)
        x = torch.cat([fl, fs], dim=1)
        h = self.trunk(x)
        u = self.out(h)                                                      # (B, 2)

        T_l = F.softplus(u[:, 0]).unsqueeze(1)                              # (B, 1)
        T_s = F.softplus(u[:, 1]).unsqueeze(1)                              # (B, 1)

        reg_signal = torch.stack([
            T_l.squeeze(1) / self.T_l_naive - 1.0,
            T_s.squeeze(1) / self.T_s_naive - 1.0,
        ], dim=1)                                                             # (B, 2)

        if return_features:
            return T_l, T_s, reg_signal, h
        return T_l, T_s, reg_signal

