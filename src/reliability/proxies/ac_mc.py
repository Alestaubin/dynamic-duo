"""
ac_mc.py
========
Average Confidence with Max Confidence (AC-MC): the baseline unsupervised
accuracy-estimation signal — mean of the top-1 softmax probability over the
batch. https://arxiv.org/abs/1610.02136

Stateless: no source-data fitting needed.
"""

from __future__ import annotations

import torch

from src.reliability.proxies.base import Proxy, register


@torch.no_grad()
def ac_mc_score(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=1)
    return float(p.max(dim=1).values.mean())


@register
class AcMcProxy(Proxy):
    name = "ac_mc"

    def score(self, logits: torch.Tensor, features: torch.Tensor | None) -> float:
        return ac_mc_score(logits)


if __name__ == "__main__":
    torch.manual_seed(0)

    # A confident, peaked batch scores higher than a near-uniform one.
    confident = torch.eye(5).repeat(4, 1) * 10.0  # near one-hot logits
    uniform = torch.zeros(20, 5)
    assert ac_mc_score(confident) > ac_mc_score(uniform)

    # Score always lies in (1/K, 1].
    logits = torch.randn(16, 10)
    s = ac_mc_score(logits)
    assert 1.0 / 10 < s <= 1.0

    proxy = AcMcProxy()
    assert proxy.is_fitted  # stateless
    assert abs(proxy.score(logits, None) - ac_mc_score(logits)) < 1e-9
    print("ac_mc self-test passed")
