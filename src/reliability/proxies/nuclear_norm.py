"""
nuclear_norm.py
===============
Confidence + dispersity via the nuclear norm of the softmax matrix.
https://github.com/cuishuhao/BNM/blob/2d23c61f864af489d84fe5f8b66bc0a5ca51cda9/UODR/train_loader.py#L197
https://arxiv.org/pdf/2302.01094

Stateless: no source-data fitting needed.
"""

from __future__ import annotations

import torch

from src.reliability.proxies.base import Proxy, register


@torch.no_grad()
def nuclear_norm_score(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=1)
    n, c = p.shape
    nuc = torch.linalg.matrix_norm(p, ord="nuc")
    return float(nuc / (n * min(n, c)) ** 0.5)


@register
class NuclearNormProxy(Proxy):
    name = "nuclear_norm"

    def score(self, logits: torch.Tensor, features: torch.Tensor | None, labels: torch.Tensor | None = None) -> float:
        return nuclear_norm_score(logits)
